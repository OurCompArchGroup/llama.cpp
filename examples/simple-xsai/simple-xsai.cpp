// simple-xsai.cpp — llama.cpp inference benchmark for XiangShan RTL simulation
//
// Based on examples/simple/simple.cpp; adds:
//   - RISC-V rdcycle / rdinstret CSR reads for cycle-accurate measurement
//   - Separate profiling of prefill and per-token decode phases
//   - Structured performance report (TTFT, TPOT, IPC, tok/s) at a configurable
//     assumed CPU frequency (default 2 GHz)
//   - NEMU/XiangShan profiler protocol signals around the prefill phase so that
//     RTL checkpointing and SimPoint workflows can work as before

#include "llama.h"
#include "simple-xsai-params.h"

#ifdef GGML_XSAI_ALLOC
#    include "xsai_alloc.h"
#endif

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// RISC-V hardware performance counters
// On non-RISC-V hosts both read as 0 so the binary remains buildable on x86.
// ---------------------------------------------------------------------------

static inline uint64_t read_cycle(void) {
#if defined(__riscv)
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
#else
    return 0;
#endif
}

static inline uint64_t read_instret(void) {
#if defined(__riscv)
    uint64_t c;
    asm volatile("rdinstret %0" : "=r"(c));
    return c;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// NEMU / XiangShan profiler protocol
//   .insn r 0x6B, 0, 0, x0, x0, x0   (with a0 = code)
// ---------------------------------------------------------------------------

#define NEMU_DISABLE_TIME_INTR   0x100
#define NEMU_NOTIFY_PROFILER     0x101
#define NEMU_NOTIFY_PROFILE_EXIT 0x102

static void nemu_signal(int code) {
#if 1
    asm volatile(
        "mv a0, %0\n\t"
        ".insn r 0x6B, 0, 0, x0, x0, x0\n\t"
        :
        : "r"(code)
        : "a0");
#else
    (void)code;
#endif
}

// ---------------------------------------------------------------------------
// Cycle → time helpers (assumed CPU frequency)
// ---------------------------------------------------------------------------

static constexpr double CPU_FREQ_GHZ  = 2.0;          // 2 GHz
static constexpr double CYCLES_PER_MS = CPU_FREQ_GHZ * 1e6;

static inline double cy_to_ms(uint64_t cy) {
    return (double)cy / CYCLES_PER_MS;
}

static inline double cy_to_tps(uint64_t cy, int n) {
    if (cy == 0 || n <= 0) return 0.0;
    return (double)n / (cy_to_ms(cy) / 1000.0);
}

static inline double ipc(uint64_t cy, uint64_t ir) {
    return cy > 0 ? (double)ir / (double)cy : 0.0;
}

// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nUsage: %s -m model.gguf [-n n_predict] [prompt]\n\n",
        prog);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt    = "Hello my name is";
    int         n_predict = 32;

    // ---- argument parsing ----
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) { model_path = argv[++i]; }
                else              { print_usage(argv[0]); return 1; }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try { n_predict = std::stoi(argv[++i]); }
                    catch (...) { print_usage(argv[0]); return 1; }
                } else { print_usage(argv[0]); return 1; }
            } else {
                break; // remainder is the prompt
            }
        }
        if (model_path.empty()) { print_usage(argv[0]); return 1; }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) { prompt += " "; prompt += argv[i]; }
        }
    }

    ggml_backend_load_all();

    // ---- load model (timed) ----
    llama_model_xsai_parms model_xsai_parms = llama_model_xsai_default_parms();

    llama_model_params model_params = to_llama_model_params(model_xsai_parms);

    const uint64_t cy_load_start = read_cycle();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    const uint64_t cy_load_end   = read_cycle();

    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ---- tokenize ----
    const int n_prompt = -llama_tokenize(
        vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // ---- context ----
    llama_context_xsai_parms ctx_xsai_parms = llama_context_xsai_default_parms();
    ctx_xsai_parms.n_ctx   = n_prompt + n_predict - 1;
    ctx_xsai_parms.n_batch = n_prompt;
    ctx_xsai_parms.no_perf = false;

    llama_context_params ctx_params = to_llama_context_params(ctx_xsai_parms);

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create llama_context\n", __func__);
        return 1;
    }

    // ---- greedy sampler ----
    auto sparams    = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // ---- echo prompt tokens ----
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        printf("%.*s", n, buf);
    }

    // ---- initial batch ----
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s: failed to encode\n", __func__);
            return 1;
        }
        llama_token dec_start = llama_model_decoder_start_token(model);
        if (dec_start == LLAMA_TOKEN_NULL) dec_start = llama_vocab_bos(vocab);
        batch = llama_batch_get_one(&dec_start, 1);
    }

    // ---- per-phase cycle counters ----
    uint64_t cy_prefill = 0, ir_prefill = 0;
    uint64_t cy_decode  = 0, ir_decode  = 0;
    std::vector<uint64_t> cy_per_token;   // per-decode-step cycles for TPOT variance
    cy_per_token.reserve(n_predict);

    int  n_pos    = 0;
    int  n_decode = 0;
    bool is_prefill = true;
    llama_token new_token_id;

    // ==============================================================
    // Main inference loop
    // ==============================================================
    for (; n_pos + batch.n_tokens < n_prompt + n_predict; ) {

        if (is_prefill) {
            // -------- PREFILL phase --------
            // Signal profiler to start sampling (used by NEMU SimPoint / RTL ckpt)
            nemu_signal(NEMU_DISABLE_TIME_INTR);
            nemu_signal(NEMU_NOTIFY_PROFILER);

            printf("\n--- prefill: processing %d prompt tokens ---\n", batch.n_tokens);
            const uint64_t cy0 = read_cycle();
            const uint64_t ir0 = read_instret();
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s: failed to eval (prefill)\n", __func__);
                return 1;
            }
            cy_prefill = read_cycle()   - cy0;
            ir_prefill = read_instret() - ir0;

            // nemu_signal(NEMU_NOTIFY_PROFILE_EXIT);
            is_prefill = false;

        } else {
            // -------- DECODE phase (one token per call) --------
            const uint64_t cy0 = read_cycle();
            const uint64_t ir0 = read_instret();
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s: failed to eval (decode)\n", __func__);
                return 1;
            }
            const uint64_t dcy = read_cycle()   - cy0;
            const uint64_t dir = read_instret() - ir0;
            cy_decode += dcy;
            ir_decode += dir;
            cy_per_token.push_back(dcy);
        }

        n_pos += batch.n_tokens;

        // ---- sample next token ----
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        printf("%.*s", n, buf);
        fflush(stdout);

        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode++;
    }

    printf("\n");

    // ==============================================================
    // Performance report
    // ==============================================================
    const uint64_t cy_load  = cy_load_end - cy_load_start;
    const uint64_t cy_total = cy_prefill + cy_decode;

    fprintf(stderr, "\n");
    fprintf(stderr, "======= XSAI Inference Report  (CPU @ %.0f GHz assumed) =======\n",
            CPU_FREQ_GHZ);

    fprintf(stderr, "  model load        : %13" PRIu64 " cycles  %10.3f ms\n",
            cy_load, cy_to_ms(cy_load));
    fprintf(stderr, "\n");

    // -- prefill --
    fprintf(stderr, "  prompt tokens     : %d\n", n_prompt);
    fprintf(stderr, "  prefill cycles    : %13" PRIu64 "         %10.3f ms\n",
            cy_prefill, cy_to_ms(cy_prefill));
    fprintf(stderr, "  prefill IPC       : %.3f\n", ipc(cy_prefill, ir_prefill));
    fprintf(stderr, "  TTFT              :                   %10.3f ms\n",
            cy_to_ms(cy_prefill));
    fprintf(stderr, "  prompt throughput : %10.2f tok/s\n",
            cy_to_tps(cy_prefill, n_prompt));
    fprintf(stderr, "\n");

    // -- decode --
    fprintf(stderr, "  decode tokens     : %d\n", n_decode);
    if (n_decode > 0) {
        const double tpot_avg_ms = cy_to_ms(cy_decode) / n_decode;

        fprintf(stderr, "  decode cycles     : %13" PRIu64 "         %10.3f ms\n",
                cy_decode, cy_to_ms(cy_decode));
        fprintf(stderr, "  decode IPC        : %.3f\n", ipc(cy_decode, ir_decode));
        fprintf(stderr, "  decode throughput : %10.2f tok/s\n",
                cy_to_tps(cy_decode, n_decode));
        fprintf(stderr, "  TPOT avg          : %13" PRIu64 " cycles  %10.3f ms/tok\n",
                cy_decode / (uint64_t)n_decode, tpot_avg_ms);

        if (!cy_per_token.empty()) {
            uint64_t cy_min = cy_per_token[0], cy_max = cy_per_token[0];
            for (auto c : cy_per_token) {
                if (c < cy_min) cy_min = c;
                if (c > cy_max) cy_max = c;
            }
            fprintf(stderr, "  TPOT min          : %13" PRIu64 " cycles  %10.3f ms/tok\n",
                    cy_min, cy_to_ms(cy_min));
            fprintf(stderr, "  TPOT max          : %13" PRIu64 " cycles  %10.3f ms/tok\n",
                    cy_max, cy_to_ms(cy_max));
        }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "  total inference   : %13" PRIu64 " cycles  %10.3f ms\n",
            cy_total, cy_to_ms(cy_total));
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "\n");
#ifdef GGML_XSAI_ALLOC
    xsai_alloc_print_stats();
#endif
    // llama_perf_sampler_print(smpl);
    // llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
