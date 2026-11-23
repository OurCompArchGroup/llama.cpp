// Benchmark quantization specific functions on synthetic data

#include "ggml.h"
#include "ggml-cpu.h"
#if defined(GGML_TEST_RMS_INTERNAL)
#include "ggml-cpu/ops.h"
#include "ggml-cpu/binary-ops.h"
#include "ggml-cpu/arch/riscv/ame/ame.h"
#endif
#if defined(GGML_TEST_RMS_INTERNAL) || defined(GGML_MXFP4_PHASE_PROFILE)
#include "ggml-cpu/quants.h"
#endif
#include <inttypes.h>

#undef NDEBUG
#include <algorithm>
#include <array>
#include <assert.h>
#include <cfenv>
#include <cstring>
#include <functional>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_ALIGNMENT 64
#define QK 32
#define WARMUP 5
#define ITERATIONS 10
#define MAX_ITERATIONS 100000000

#define L1_SIZE      32*128
#define L2_SIZE     32*2048
#define L3_SIZE    32*20480
#define MEM_SIZE 32*2048000

struct quantize_perf_params {
    std::vector<std::string> include_types;
    std::vector<size_t> test_sizes;
    size_t alignment_offset = 0;
    bool op_quantize_row_q_reference = false;
    bool op_quantize_row_q = false;
    bool op_dequantize_row_q = false;
    bool op_quantize_row_q_dot = false;
    bool op_vec_dot_q = false;
    bool rmsnorm_mxfp4_512 = false;
    bool rmsnorm_mxfp4_512_check = false;
    bool rmsnorm_mxfp4_512_checkpoint = false;
    int64_t iterations = ITERATIONS;
};

#if defined(__x86_64__) || defined(__i386__)

#include <x86intrin.h>
inline int64_t cpu_cycles() {
// Rough way to detect new-ish CPUs
#ifdef __POPCNT__
    unsigned int dummy;
    return __rdtscp(&dummy);
#else
    return __rdtsc();
#endif
}

#elif defined(__riscv) && __riscv_xlen == 64

inline int64_t cpu_cycles() {
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles) : : "memory");
    return (int64_t) cycles;
}

#else

#define cpu_cycles() 0

#endif


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

static float gigabytes_per_second(size_t bytes, int64_t usecs) {
    return bytes / (float) usecs * 1000000 / (1024*1024*1024);
}

static void * align_with_offset(void * ptr, int offset) {
    size_t dummy_size = MAX_ALIGNMENT * 4;
    return (char *) std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, ptr, dummy_size) + offset;
}

struct benchmark_hooks {
    std::function<void(void)> prepare;
    std::function<void(void)> verify;
    const char * stage = nullptr;
    uint64_t quant_calls = 0;
    uint64_t rms_rows = 0;
};

static void benchmark_function(size_t size, size_t q_size, int64_t iterations,
        const std::function<float(void)> & func, bool profile_mxfp4 = false,
        const benchmark_hooks * hooks = nullptr) {
    int64_t min_time_us = INT64_MAX;
    int64_t total_time_us = 0;
    int64_t min_time_cycles = INT64_MAX;
    int64_t total_time_cycles = 0;
    std::vector<int64_t> samples;
#if defined(GGML_MXFP4_PHASE_PROFILE)
    std::vector<ggml_mxfp4_phase_profile> phase_samples;
    if (hooks && profile_mxfp4) phase_samples.reserve(size_t(iterations));
#endif

    for (int i = 0; i < WARMUP; i++) {
        if (hooks) hooks->prepare();
        func();
        if (hooks) hooks->verify();
    }

#if defined(GGML_MXFP4_PHASE_PROFILE)
    if (profile_mxfp4) {
        ggml_mxfp4_phase_profile_reset();
    }
#else
    (void) profile_mxfp4;
#endif

    for (int i = 0; i < iterations; i++) {
        if (hooks) hooks->prepare();
        const int64_t start_time = ggml_time_us();
        const int64_t start_cycles = cpu_cycles();

        func();

        const int64_t end_cycles = cpu_cycles();
        const int64_t end_time = ggml_time_us();
        if (hooks) hooks->verify();
#if defined(GGML_MXFP4_PHASE_PROFILE)
        // Capture cumulative counters AFTER the timed call and its FP checks.
        // Snapshot/delta accounting must never enter the uninstrumented build.
        if (hooks && profile_mxfp4) {
            ggml_mxfp4_phase_profile snapshot;
            ggml_mxfp4_phase_profile_get(&snapshot);
            phase_samples.push_back(snapshot);
        }
#endif

        total_time_cycles += end_cycles - start_cycles;
        min_time_cycles = std::min(min_time_cycles, end_cycles - start_cycles);
        total_time_us += end_time - start_time;
        min_time_us = std::min(min_time_us, end_time - start_time);
        if (hooks) samples.push_back(end_cycles - start_cycles);
    }

    if (hooks) {
        for (size_t i = 0; i < samples.size(); ++i) {
            printf("RMS_MXFP4_512_SAMPLE stage=%s sample=%zu cycles=%" PRId64 "\n",
                hooks->stage, i + 1, samples[i]);
        }
        std::sort(samples.begin(), samples.end());
        const size_t n = samples.size();
        assert(n > 0);
        const double median = n % 2 ? double(samples[n / 2]) :
            double(samples[n / 2 - 1]) / 2 + double(samples[n / 2]) / 2;
        printf("RMS_MXFP4_512_RESULT stage=%s elements=%zu samples=%zu"
               " median_cycles=%.1f min_cycles=%" PRId64 " max_cycles=%" PRId64
               " scope=one_row_warm_repeated_setup_and_checks_excluded\n",
               hooks->stage, size, n, median, samples.front(), samples.back());
    } else {
        printf("      min cycles/%d vals   : %9.2f\n",  QK, QK * min_time_cycles / (float) size);
        printf("      avg cycles/%d vals   : %9.2f\n",  QK, QK * total_time_cycles / (float) (size * iterations));
        printf("      float32 throughput   : %9.2f GB/s\n",  gigabytes_per_second(4 * size * iterations, total_time_us));
        printf("      quantized throughput : %9.2f GB/s\n",  gigabytes_per_second(q_size * iterations, total_time_us));
    }
#if defined(GGML_MXFP4_PHASE_PROFILE)
    if (profile_mxfp4 && iterations > 0) {
        struct ggml_mxfp4_phase_profile phases;
        ggml_mxfp4_phase_profile_get(&phases);
        const uint64_t quant_calls = hooks ? hooks->quant_calls : 1;
        assert(phases.calls == quant_calls * (uint64_t) iterations);
        assert(phases.blocks == (quant_calls ? size / QK * (uint64_t) iterations : 0));
        assert(phases.rms_rows == (hooks ? hooks->rms_rows : 0) * (uint64_t) iterations);
        assert(phases.threshold_blocks + phases.paired_blocks + phases.fallback_blocks == phases.blocks);
        if (hooks) {
            assert(phase_samples.size() == size_t(iterations));
            ggml_mxfp4_phase_profile previous = {};
            for (size_t i = 0; i < phase_samples.size(); ++i) {
                const auto & sample = phase_samples[i];
                assert(sample.calls - previous.calls == hooks->quant_calls);
                assert(sample.blocks - previous.blocks == (hooks->quant_calls ? size / QK : 0));
                assert(sample.rms_rows - previous.rms_rows == hooks->rms_rows);
                printf("RMS_MXFP4_512_PHASE_SAMPLE stage=%s sample=%zu"
                       " rms_stats=%" PRIu64 " rms_copy=%" PRIu64 " rms_apply=%" PRIu64
                       " quant_amax=%" PRIu64 " quant_scale=%" PRIu64
                       " quant_encode=%" PRIu64 " quant_nibble_pack=%" PRIu64
                       " quant_fallback=%" PRIu64 " quant_calls=%" PRIu64
                       " blocks=%" PRIu64 " rms_rows=%" PRIu64 "\n",
                       hooks->stage, i + 1,
                       sample.rms_stats_cycles - previous.rms_stats_cycles,
                       sample.rms_copy_cycles - previous.rms_copy_cycles,
                       sample.rms_apply_cycles - previous.rms_apply_cycles,
                       sample.amax_cycles - previous.amax_cycles,
                       sample.scale_cycles - previous.scale_cycles,
                       sample.encode_cycles - previous.encode_cycles,
                       sample.pack_cycles - previous.pack_cycles,
                       sample.fallback_cycles - previous.fallback_cycles,
                       sample.calls - previous.calls, sample.blocks - previous.blocks,
                       sample.rms_rows - previous.rms_rows);
                previous = sample;
            }
            auto report_phase = [&](const char * name, uint64_t ggml_mxfp4_phase_profile::* field) {
                std::vector<uint64_t> durations;
                uint64_t before = 0;
                for (const auto & sample : phase_samples) {
                    assert(sample.*field >= before);
                    durations.push_back(sample.*field - before);
                    before = sample.*field;
                }
                std::sort(durations.begin(), durations.end());
                const size_t count = durations.size();
                const double median = count % 2 ? double(durations[count / 2]) :
                    double(durations[count / 2 - 1]) / 2 + double(durations[count / 2]) / 2;
                printf("RMS_MXFP4_512_PHASE_RESULT stage=%s phase=%s samples=%zu"
                       " median_cycles=%.1f min_cycles=%" PRIu64 " max_cycles=%" PRIu64
                       " scope=one_row_phase_accumulation_diagnostic_perturbed\n",
                       hooks->stage, name, count, median, durations.front(), durations.back());
            };
            if (phases.rms_rows) {
                report_phase("rms_stats_reciprocal", &ggml_mxfp4_phase_profile::rms_stats_cycles);
                report_phase("rms_copy", &ggml_mxfp4_phase_profile::rms_copy_cycles);
                report_phase("rms_apply", &ggml_mxfp4_phase_profile::rms_apply_cycles);
            }
            if (phases.blocks) {
                report_phase("quant_amax", &ggml_mxfp4_phase_profile::amax_cycles);
                report_phase("quant_scale_eligibility", &ggml_mxfp4_phase_profile::scale_cycles);
                report_phase("quant_encode", &ggml_mxfp4_phase_profile::encode_cycles);
                report_phase("quant_nibble_pack", &ggml_mxfp4_phase_profile::pack_cycles);
                report_phase("quant_fallback", &ggml_mxfp4_phase_profile::fallback_cycles);
            }
        }
        printf("      MXFP4_PHASE_PROFILE diagnostic=perturbed calls=%" PRIu64 " blocks=%" PRIu64
               " threshold=%" PRIu64 " paired=%" PRIu64 " fallback=%" PRIu64
               " adjacent_rdcycle_min=%" PRIu64 "\n",
               phases.calls, phases.blocks, phases.threshold_blocks, phases.paired_blocks,
               phases.fallback_blocks, phases.adjacent_tick_min);
        if (phases.blocks) printf("      phase cycles/block   : amax=%.2f scale_eligibility=%.2f encode=%.2f"
               " nibble_pack=%.2f fallback_after_amax=%.2f\n",
               phases.amax_cycles / (double) phases.blocks, phases.scale_cycles / (double) phases.blocks,
               phases.encode_cycles / (double) phases.blocks, phases.pack_cycles / (double) phases.blocks,
               phases.fallback_cycles / (double) phases.blocks);
        if (phases.rms_rows) {
            printf("      phase cycles/row     : rms_stats_reciprocal=%.2f rms_copy=%.2f rms_apply=%.2f rows=%" PRIu64 "\n",
                phases.rms_stats_cycles / (double) phases.rms_rows,
                phases.rms_copy_cycles / (double) phases.rms_rows,
                phases.rms_apply_cycles / (double) phases.rms_rows, phases.rms_rows);
        }
        printf("      phase scope          : 5 timestamps/normal block, 3/fallback;"
               " includes probe effects, not an unperturbed partition or speedup\n");
    }
#endif
}

// One partial activation row is a software microbenchmark, NOT an admitted AME
// panel. Call the actual library op owners (including their RVV code), never
// a test-local copy of the RMS/quantization arithmetic.
#if defined(GGML_TEST_RMS_INTERNAL)
static void benchmark_rmsnorm_mxfp4_512(int64_t iterations, bool check_only, bool checkpoint_roi) {
    constexpr int K = 512, NB = K / QK, MN = 128, TK = 64;
    constexpr size_t PANEL_BLOCKS = MN * NB;
    constexpr float guard = -0x1.37p+9f;
    std::fenv_t saved;
    assert(std::fegetenv(&saved) == 0);
    const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->from_float;
    assert(quantize != nullptr);
    ggml_quantize_init(GGML_TYPE_MXFP4);
    alignas(64) std::array<float, K + 32> raw, gamma, norm, weighted;
    alignas(64) std::array<block_mxfp4, NB + 2> encoded;
    alignas(64) std::array<uint8_t, PANEL_BLOCKS * 16 + 128> payload;
    alignas(64) std::array<uint8_t, PANEL_BLOCKS + 128> sf;
    raw.fill(guard);
    gamma.fill(guard);
    norm.fill(guard);
    weighted.fill(guard);
    ggml_init_params init = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    assert(ctx != nullptr);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    x->data = raw.data() + 16;
    g->data = gamma.data() + 16;
    ggml_tensor * n = ggml_rms_norm(ctx, x, 0x1p-10f);
    ggml_tensor * w = ggml_mul(ctx, n, g);
    n->data = norm.data() + 16;
    w->data = weighted.data() + 16;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, w);
    const ggml_compute_params compute = { 0, 1, 0, nullptr, nullptr };
    std::array<float, K + 32> expected_norm, expected_weighted, raw_before, gamma_before;
    std::array<block_mxfp4, NB + 2> expected_encoded;
    decltype(payload) expected_payload;
    decltype(sf) expected_sf;
    std::fenv_t initial_env, norm_env, gamma_env, quant_env;
    int norm_flags = 0, gamma_flags = 0, quant_flags = 0, round_mode = FE_TONEAREST;
    auto seed = [](int round, int flags) {
        assert(std::fesetround(round) == 0);
        assert(std::feclearexcept(FE_ALL_EXCEPT) == 0);
        assert(std::feraiseexcept(flags) == 0);
        assert(std::feclearexcept(FE_ALL_EXCEPT & ~flags) == 0);
        assert(std::fetestexcept(FE_ALL_EXCEPT) == flags);
    };
    auto pack = [&]() {
        for (int kb = 0; kb < NB; kb += TK / QK) {
            const size_t index = ggml_ame_mx_packed_block_index(0, kb, NB, TK, MN);
            assert(ggml_ame_pack_mxfp4_blocks(payload.data() + 64 + index * 16,
                sf.data() + 64 + index, encoded.data() + 1 + kb, TK / QK));
        }
    };
    auto quant = [&]() {
        for (int kb = 0; kb < NB; kb += 8) {
            quantize(weighted.data() + 16 + kb * QK, encoded.data() + 1 + kb, 8 * QK);
        }
    };
    auto rms = [&]() { ggml_compute_forward_rms_norm(&compute, n); };
    auto mul = [&]() { ggml_compute_forward_mul(&compute, w); };
    auto poison = [&]() {
        norm.fill(guard);
        weighted.fill(guard);
        std::memset(encoded.data(), 0x96, sizeof(encoded));
        payload.fill(0xc3);
        sf.fill(0xa7);
    };
    auto reference = [&](int pattern, int round, int flags) {
        seed(FE_TONEAREST, 0);
        for (int k = 0; k < K; ++k) {
            raw[16 + k] = pattern ? 1.0f : float((k * 13) % 127 - 63) * 0x1p-5f;
            gamma[16 + k] = pattern ? 1.0f : float(32 + (k * 7) % 65) * 0x1p-6f;
        }
        const float eps = pattern == 0 ? 0x1p-10f :
            pattern == 2 ? 3.0f - 3.0f / K : 3.0f;
        if (pattern) gamma[16 + 7 * QK + 3] = 0x1p-149f;
        if (pattern == 2) raw[16 + 7 * QK + 3] = 2.0f;
        std::memcpy(n->op_params, &eps, sizeof(eps));
        raw_before = raw;
        gamma_before = gamma;
        poison();
        round_mode = round;
        seed(round, flags);
        assert(std::fegetenv(&initial_env) == 0);
        // Independent execution via the ordinary two-node GGML graph.
        assert(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS);
        expected_norm = norm;
        expected_weighted = weighted;
        if (pattern == 2) assert(std::fetestexcept(FE_ALL_EXCEPT) == flags);
        if (pattern == 1) assert((std::fetestexcept(FE_ALL_EXCEPT) &
            (FE_UNDERFLOW | FE_INEXACT)) == (FE_UNDERFLOW | FE_INEXACT));
        // One full-row quant call is an oracle for the production batch8 calls.
        quantize(weighted.data() + 16, encoded.data() + 1, K);
        quant_flags = std::fetestexcept(FE_ALL_EXCEPT);
        expected_encoded = encoded;
        expected_payload.fill(0xc3);
        expected_sf.fill(0xa7);
        for (int kb = 0; kb < NB; ++kb) {
            const size_t index = size_t(kb / 2) * MN * 2 + kb % 2;
            assert(ggml_ame_pack_mxfp4_blocks(expected_payload.data() + 64 + index * 16,
                expected_sf.data() + 64 + index, expected_encoded.data() + 1 + kb, 1));
        }
        assert(std::fegetround() == round);
        // Capture genuine prefix FP states for independent stage measurements;
        // never artificially seed NX to select the quantizer's threshold path.
        assert(std::fesetenv(&initial_env) == 0);
        rms();
        norm_flags = std::fetestexcept(FE_ALL_EXCEPT);
        assert(std::fegetenv(&norm_env) == 0);
        mul();
        gamma_flags = std::fetestexcept(FE_ALL_EXCEPT);
        assert(std::fegetenv(&gamma_env) == 0);
        quant();
        assert(std::fetestexcept(FE_ALL_EXCEPT) == quant_flags);
        assert(std::fegetenv(&quant_env) == 0);
    };
    auto verify = [&](int stage, int flags) {
        assert(std::fetestexcept(FE_ALL_EXCEPT) == flags);
        assert(std::fegetround() == round_mode);
        assert(raw == raw_before && gamma == gamma_before);
        if (stage >= 1) assert(std::memcmp(norm.data(), expected_norm.data(), sizeof(norm)) == 0);
        if (stage >= 2) assert(std::memcmp(weighted.data(), expected_weighted.data(), sizeof(weighted)) == 0);
        if (stage >= 3) assert(std::memcmp(encoded.data(), expected_encoded.data(), sizeof(encoded)) == 0);
        if (stage >= 4) assert(payload == expected_payload && sf == expected_sf);
        // Check all output guards even for stages that do not consume them.
        for (int i = 0; i < 16; ++i) {
            assert(norm[i] == guard && norm[16 + K + i] == guard);
            assert(weighted[i] == guard && weighted[16 + K + i] == guard);
        }
        const uint8_t * encoded_bytes = reinterpret_cast<const uint8_t *>(encoded.data());
        for (size_t i = 0; i < sizeof(block_mxfp4); ++i) {
            assert(encoded_bytes[i] == 0x96);
            assert(encoded_bytes[(NB + 1) * sizeof(block_mxfp4) + i] == 0x96);
        }
        for (size_t i = 0; i < 64; ++i) {
            assert(payload[i] == 0xc3 && payload[64 + PANEL_BLOCKS * 16 + i] == 0xc3);
            assert(sf[i] == 0xa7 && sf[64 + PANEL_BLOCKS + i] == 0xa7);
        }
    };
    const int exceptions[] = { FE_INEXACT, FE_UNDERFLOW, FE_OVERFLOW, FE_DIVBYZERO, FE_INVALID };
    unsigned cases = 0;
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (int round : { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO }) {
            for (unsigned mask = 0; mask < 32; ++mask) {
                int flags = 0;
                for (unsigned bit = 0; bit < 5; ++bit) if (mask & (1u << bit)) flags |= exceptions[bit];
                reference(pattern, round, flags);
                poison();
                assert(std::fesetenv(&initial_env) == 0);
                rms();
                verify(1, norm_flags);
                mul();
                verify(2, gamma_flags);
                quant();
                verify(3, quant_flags);
                pack();
                verify(4, quant_flags);
                ++cases;
            }
        }
    }
    printf("RMS_MXFP4_512_CHECK PASS cases=%u blocks=%u frm=4 initial_flags=32"
           " oracle=real_GGML_graph_and_fullrow_quant payload=exact scales=exact"
           " five_flags=exact canaries=pass input=unchanged native_issue=none\n", cases, cases * NB);
    if (!check_only) {
        reference(0, FE_TONEAREST, 0);
#if defined(__riscv) && defined(GGML_TEST_RV_AME)
        if (checkpoint_roi) {
            // Setup and all384 oracle cases are complete. Reuse the canonical
            // system ROI markers, not a new runner or checkpoint protocol.
            std::fflush(stdout);
            ggml_ame_nemu_signal(0x100);
            ggml_ame_nemu_signal(0x101);
        }
#endif
        printf("RMS_MXFP4_512_BEGIN elements=512 rows=1 K32_blocks=16 warmup=%d repeats=%" PRId64
               " logical_payload=256 logical_scales=16 padded_payload=%zu padded_scales=%zu"
               " quant_calls_per_row=2 batch_blocks=8 tile_mn=128 tile_k=64 native_issue=none"
               " cache_state=post_setup_not_cold_DRAM"
#if defined(GGML_MXFP4_PHASE_PROFILE)
               " internal_probes=on_diagnostic_perturbed"
#else
               " internal_probes=off"
#endif
               " checkpoint_roi=%d\n", WARMUP, iterations, PANEL_BLOCKS * 16, PANEL_BLOCKS,
               int(checkpoint_roi));
        const char * names[] = { "timer_control", "rmsnorm_total", "gamma", "mxfp4_quant", "ame_reorder_pack", "full_chain" };
        for (int stage = 0; stage < 6; ++stage) {
            const int completed = stage == 5 ? 4 : stage;
            const int flags = stage == 0 ? 0 : stage == 1 ? norm_flags :
                stage == 2 ? gamma_flags : quant_flags;
            benchmark_hooks hooks;
            hooks.stage = names[stage];
            hooks.quant_calls = stage == 3 || stage == 5 ? 2 : 0;
            hooks.rms_rows = stage == 1 || stage == 5 ? 1 : 0;
            hooks.prepare = [&]() {
                poison();
                if (stage >= 2 && stage != 5) norm = expected_norm;
                if (stage >= 3 && stage != 5) weighted = expected_weighted;
                if (stage == 4) encoded = expected_encoded;
                const std::fenv_t & prefix = stage == 2 ? norm_env :
                    stage == 3 ? gamma_env : stage == 4 ? quant_env : initial_env;
                assert(std::fesetenv(&prefix) == 0);
            };
            hooks.verify = [&]() { verify(completed, flags); };
            auto func = [&]() -> float {
                if (stage == 1 || stage == 5) rms();
                if (stage == 2 || stage == 5) mul();
                if (stage == 3 || stage == 5) quant();
                if (stage == 4 || stage == 5) pack();
                return 0.0f;
            };
            benchmark_function(K, NB * sizeof(block_mxfp4), iterations, func, true, &hooks);
        }
        printf("RMS_MXFP4_512_END stages=6 correctness=pass"
               " timing=independent_not_additive full_chain_excludes_graph_executor\n");
#if defined(__riscv) && defined(GGML_TEST_RV_AME)
        if (checkpoint_roi) {
            std::fflush(stdout);
            ggml_ame_nemu_signal(0x102);
        }
#endif
    }
    ggml_free(ctx);
    assert(std::fesetenv(&saved) == 0);
#if defined(__riscv) && defined(GGML_TEST_RV_AME)
    if (checkpoint_roi) {
        std::fflush(stdout);
        ggml_ame_nemu_signal(0);
    }
#endif
}
#endif

static void usage(char * argv[]) {
    printf("Benchmark quantization specific functions on synthetic data\n");
    printf("\n");
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options: (default)\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --size SIZE           set test size, divisible by 32 (L1_SIZE:%d)\n", L1_SIZE);
    printf("  -3                    use size as L1, L2, L3 sizes (L1:%d L2:%d L3:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE);
    printf("  -4                    use size as L1, L2, L3, MEM sizes (L1:%d L2:%d L3:%d MEM:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE, MEM_SIZE);
    printf("  --op OP               set test operation as quantize_row_q_reference, quantize_row_q, dequantize_row_q,\n");
    printf("                        quantize_row_q_dot, vec_dot_q (all)\n");
    printf("  --type TYPE           set test type as");
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (ggml_type_name(type) != NULL) {
            if (qfns_cpu->from_float && qfns->to_float) {
                printf(" %s", ggml_type_name(type));
            }
        }
    }
    printf(" (all)\n");
    printf("  --alignment-offset OFFSET\n");
    printf("                        set alignment offset as OFFSET (0)\n");
    printf("  -i NUM, --iterations NUM\n");
    printf("                        set test iteration number (%d)\n", ITERATIONS);
    printf("  --rmsnorm-mxfp4-512    one-row512 RMS/gamma/quant/AME-pack stages, no GEMM\n");
    printf("  --rmsnorm-mxfp4-512-check  correctness only; no performance claim\n");
    printf("  --rmsnorm-mxfp4-512-checkpoint  same measurement with explicit system-mode ROI markers\n");
}

int main(int argc, char * argv[]) {
    quantize_perf_params params {};

    // read command line

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "--rmsnorm-mxfp4-512") {
            params.rmsnorm_mxfp4_512 = true;
        } else if (arg == "--rmsnorm-mxfp4-512-check") {
            params.rmsnorm_mxfp4_512_check = true;
        } else if (arg == "--rmsnorm-mxfp4-512-checkpoint") {
            params.rmsnorm_mxfp4_512 = params.rmsnorm_mxfp4_512_checkpoint = true;
        } else if (arg == "--size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            size_t size = std::stoi(argv[i]);
            if (size % 32 != 0) {
                fprintf(stderr, "error: size %zu not divisible by 32\n", size);
                invalid_param = true;
                break;
            }
            params.test_sizes.push_back(size);
        } else if (arg == "-3") {
            // quick select sizes that probably fit in CPU caches
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
        } else if (arg == "-4") {
            // quick select cache sizes + memory
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
            params.test_sizes.push_back(MEM_SIZE);
        } else if (arg == "--op") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string op {argv[i]};
            if (op == "quantize_row_q_reference") {
                params.op_quantize_row_q_reference = true;
            } else if (op == "quantize_row_q") {
                params.op_quantize_row_q = true;
            } else if (op == "dequantize_row_q") {
                params.op_dequantize_row_q = true;
            } else if (op == "quantize_row_q_dot") {
                params.op_quantize_row_q_dot = true;
            } else if (op == "vec_dot_q") {
                params.op_vec_dot_q = true;
            } else {
                invalid_param = true;
                break;
            }
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.include_types.push_back(argv[i]);
        } else if (arg == "--alignment-offset") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int alignment = std::stoi(argv[i]);
            if (alignment < 0 || alignment > MAX_ALIGNMENT) {
            fprintf(stderr, "error: alignment-offset must be less than %d\n", MAX_ALIGNMENT);
                invalid_param = true;
                break;
            }
            params.alignment_offset = alignment;
        } else if ((arg == "-i") || (arg == "--iterations")) {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int number = std::stoi(argv[i]);
            if (number < 0 || number > MAX_ITERATIONS) {
            fprintf(stderr, "error: iterations must be less than %d\n", MAX_ITERATIONS);
                invalid_param = true;
                break;
            }
            params.iterations = number;
        } else if ((arg == "-h") || (arg == "--help")) {
            usage(argv);
            return 1;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return 1;
    }

    if (params.rmsnorm_mxfp4_512 || params.rmsnorm_mxfp4_512_check) {
        if (params.rmsnorm_mxfp4_512_checkpoint && params.rmsnorm_mxfp4_512_check) {
            fprintf(stderr, "error: checkpoint measurement and correctness-only options are mutually exclusive\n");
            return 1;
        }
#if !defined(__riscv) || !defined(GGML_TEST_RV_AME)
        if (params.rmsnorm_mxfp4_512_checkpoint) {
            fprintf(stderr, "checkpoint ROI requires the RV64 system-mode AME build\n");
            return 77;
        }
#endif
        if (!params.rmsnorm_mxfp4_512_check && (params.iterations < 3 || params.iterations > 1000)) {
            fprintf(stderr, "error: one-row512 measurement requires 3..1000 iterations\n");
            return 1;
        }
        ggml_cpu_init();
#if defined(GGML_TEST_RMS_INTERNAL)
        benchmark_rmsnorm_mxfp4_512(params.iterations, params.rmsnorm_mxfp4_512_check,
            params.rmsnorm_mxfp4_512_checkpoint);
        return 0;
#else
        fprintf(stderr, "one-row512 diagnostic requires BUILD_SHARED_LIBS=OFF (private CPU op owners)\n");
        return 77;
#endif
    }

    if (params.test_sizes.empty()) {
        params.test_sizes.push_back(L1_SIZE);
    }
    if (!(params.op_quantize_row_q_reference || params.op_quantize_row_q || params.op_dequantize_row_q || params.op_quantize_row_q_dot || params.op_vec_dot_q)) {
        params.op_quantize_row_q_reference = params.op_quantize_row_q = params.op_dequantize_row_q = params.op_quantize_row_q_dot = params.op_vec_dot_q = true;
    }

    std::sort(params.test_sizes.begin(), params.test_sizes.end());
    size_t largest = params.test_sizes.back();

    std::vector<uint8_t> test_data1_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_data2_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q1_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q2_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_out_v  (largest*4 + MAX_ALIGNMENT*2);

    float * test_data1 = (float *) align_with_offset(test_data1_v.data(), params.alignment_offset);
    float * test_data2 = (float *) align_with_offset(test_data2_v.data(), params.alignment_offset);
    float * test_q1    = (float *) align_with_offset(test_q1_v.data(),    params.alignment_offset);
    float * test_q2    = (float *) align_with_offset(test_q2_v.data(),    params.alignment_offset);
    float * test_out   = (float *) align_with_offset(test_out_v.data(),   params.alignment_offset);

    generate_data(0, largest, test_data1);
    generate_data(1, largest, test_data2);

    int64_t iterations = params.iterations;

    ggml_cpu_init();

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (!params.include_types.empty() && ggml_type_name(type) && std::find(params.include_types.begin(), params.include_types.end(), ggml_type_name(type)) == params.include_types.end()) {
            continue;
        }

        if (qfns_cpu->from_float && qfns->to_float) {
            printf("%s\n", ggml_type_name(type));

            ggml_quantize_init(type);

            if (params.op_quantize_row_q_reference) {
                printf("  quantize_row_q_reference\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns->from_float_ref(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q) {
                printf("  quantize_row_q\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns_cpu->from_float(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn, type == GGML_TYPE_MXFP4);
                }
                printf("\n");
            }

            if (params.op_dequantize_row_q) {
                printf("  dequantize_row_q\n");
                qfns_cpu->from_float(test_data1, test_q1, largest);
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns->to_float(test_q1, test_out, size);
                        return test_out[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q_dot) {
                printf("  quantize_row_q_dot\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
                        vdot->from_float(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_vec_dot_q) {
                printf("  vec_dot_q\n");
                qfns_cpu->from_float(test_data1, test_q1, largest);
                qfns_cpu->from_float(test_data2, test_q2, largest);
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        float result;
                        qfns_cpu->vec_dot(size, &result, 0, test_q1, 0, test_q2, 0, 1);
                        return result;
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }
        }
    }

    return 0;
}
