#include "ame.h"
#include "common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "vec.h"

#if defined(GGML_XSAI_ALLOC)
#include "xsai_alloc.h"
#endif

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

struct ame_profile_counters {
    uint64_t baseline_calls;
    uint64_t packed_calls;
    uint64_t logical_macs;
    uint64_t tile_calls;
    uint64_t whole_k_kloop_calls;
    uint64_t whole_k_pipeline_calls;
    uint64_t edge_tile_calls;
    uint64_t output_tile_calls;
    uint64_t cycles_baseline_total;
    uint64_t cycles_packed_total;
    uint64_t cycles_prepare_cache;
    uint64_t cycles_quant_scan;
    uint64_t cycles_quant_convert;
    uint64_t cycles_pack_a;
    uint64_t cycles_pack_b;
    uint64_t cycles_zero_c;
    uint64_t cycles_ame_tile_call;
    uint64_t cycles_accumulator_initialization;
    uint64_t cycles_scale;
    uint64_t cycles_store;
};

static struct ame_profile_counters g_ame_profile;
ggml_ame_sync_state ggml_ame_sync_state_global = {0};
static uint64_t g_ame_sparse_op_sequence;
static uint64_t g_ame_ckpt_roi_start_cycle;
static int g_ame_ckpt_roi_active;

static inline uint64_t ame_read_cycle(void) {
#if defined(__riscv)
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#else
    return 0;
#endif
}

static int ame_profile_enabled(void);

static int ame_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0 &&
        strcmp(value, "false") != 0 && strcmp(value, "off") != 0 && strcmp(value, "no") != 0;
}

static int64_t ame_env_i64(const char * name, int64_t fallback) {
    const char * value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    char * end = NULL;
    const long long parsed = strtoll(value, &end, 0);
    return end != value ? (int64_t) parsed : fallback;
}

static inline void ame_nemu_signal(int a) {
#if defined(__riscv)
    asm volatile ("mv a0, %0\n\t"
                  ".insn r 0x6B, 0, 0, x0, x0, x0\n\t"
                  :
                  : "r"(a)
                  : "a0");
#else
    GGML_UNUSED(a);
#endif
}

static int ame_skip_tile_b_zero(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_SKIP_TILE_B_ZERO") ? 1 : 0;
    }
    return cached;
}

static int ame_skip_tile_c_zero(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_SKIP_TILE_C_ZERO") ? 1 : 0;
    }
    return cached;
}

static int ame_use_packed_b_panel(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_USE_PACKED_B_PANEL") ? 1 : 0;
    }
    return cached;
}

static int ame_disable_packed_a_tiles(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_DISABLE_PACKED_A_TILES") ? 1 : 0;
    }
    return cached;
}

static int ame_whole_k_q8_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_WHOLE_K_Q8") ? 1 : 0;
    }
    return cached;
}

static int ame_whole_k_fused_epilogue_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_WHOLE_K_FUSED_EPILOGUE") ? 1 : 0;
    }
    return cached;
}

static int ame_whole_k_output_pipeline_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_WHOLE_K_OUTPUT_PIPELINE") ? 1 : 0;
    }
    return cached;
}

static int ame_whole_k_acc_pipeline_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_WHOLE_K_ACC_PIPELINE") ? 1 : 0;
    }
    return cached;
}

static int ame_transposed_c_epilogue_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_TRANSPOSED_C_EPILOGUE") ? 1 : 0;
    }
    return cached;
}

static int ame_whole_k_bf16_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_WHOLE_K_BF16") ? 1 : 0;
    }
    return cached;
}

static int ame_bf16_register_pairs(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_i64("GGML_AME_BF16_REG_PAIRS", 2) >= 2 ? 2 : 1;
    }
    return cached;
}

// The CUTE macro-instruction FIFO is four entries deep.  Keep this knob
// opt-in: the original one-submit/one-acquire path remains the default and
// is also used when the immutable packed-A cache is unavailable.
static int ame_async_batch_size(void) {
    static int cached = -1;
    if (cached == -1) {
        int64_t requested = ame_env_i64("GGML_AME_ASYNC_BATCH", 0);
        if (requested < 2) {
            cached = 0;
        } else if (requested > 4) {
            cached = 4;
        } else {
            cached = (int) requested;
        }
    }
    return cached;
}

static int ame_async_register_pairs(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_i64("GGML_AME_ASYNC_REG_PAIRS", 1) >= 2 ? 2 : 1;
    }
    return cached;
}

static int ame_panel_log_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_PANEL_LOG") ? 1 : 0;
    }
    return cached;
}

static int ame_panel_progress_log_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_PANEL_PROGRESS_LOG") ? 1 : 0;
    }
    return cached;
}

static int64_t ame_sparse_progress_interval(void) {
    static int64_t cached = -1;
    if (cached == -1) {
        cached = ame_env_i64("GGML_AME_SPARSE_PROGRESS_INTERVAL", 0);
        if (cached < 0) {
            cached = 0;
        }
    }
    return cached;
}

static uint64_t ame_sparse_progress_tile_begin(void) {
    const int64_t value = ame_env_i64("GGML_AME_SPARSE_PROGRESS_TILE_BEGIN", 0);
    return value > 0 ? (uint64_t) value : 0;
}

static uint64_t ame_sparse_progress_tile_end(uint64_t total) {
    const int64_t value = ame_env_i64("GGML_AME_SPARSE_PROGRESS_TILE_END", -1);
    if (value < 0 || (uint64_t) value >= total) {
        return total > 0 ? total - 1 : 0;
    }
    return (uint64_t) value;
}

static int ame_sparse_progress_shape_matches(int64_t M, int64_t N, int64_t K) {
    const int64_t target_m = ame_env_i64("GGML_AME_SPARSE_PROGRESS_M", 0);
    const int64_t target_n = ame_env_i64("GGML_AME_SPARSE_PROGRESS_N", 0);
    const int64_t target_k = ame_env_i64("GGML_AME_SPARSE_PROGRESS_K", 0);

    return (target_m <= 0 || M == target_m) &&
        (target_n <= 0 || N == target_n) &&
        (target_k <= 0 || K == target_k);
}

static int ame_sparse_progress_should_log(uint64_t tile, uint64_t total, int64_t interval) {
    if (interval <= 0 || total == 0) {
        return 0;
    }

    const uint64_t begin = ame_sparse_progress_tile_begin();
    const uint64_t end = ame_sparse_progress_tile_end(total);
    if (begin > end || tile < begin || tile > end) {
        return 0;
    }

    return tile == begin || tile == end || tile % (uint64_t) interval == 0;
}

static void ame_sparse_progress_log(
    uint64_t op,
    const char * phase,
    uint64_t tile,
    uint64_t total,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t i0,
    int64_t j0,
    int64_t kb
) {
    fprintf(stderr,
            "[AME_SPARSE] op=%llu phase=%s tile=%llu/%llu target=%lu "
            "M=%lld N=%lld K=%lld i0=%lld j0=%lld kb=%lld\n",
            (unsigned long long) op,
            phase,
            (unsigned long long) tile,
            (unsigned long long) total,
            ggml_ame_sync_state_global.sync0_release_target,
            (long long) M,
            (long long) N,
            (long long) K,
            (long long) i0,
            (long long) j0,
            (long long) kb);
    fflush(stderr);
}

static int ame_ckpt_on_shape_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_CKPT_ON_SHAPE") ? 1 : 0;
    }
    return cached;
}

static int ame_ckpt_shape_matches(int64_t M, int64_t N, int64_t K) {
    if (!ame_ckpt_on_shape_enabled()) {
        return 0;
    }

    const int64_t target_m = ame_env_i64("GGML_AME_CKPT_M", 2048);
    const int64_t target_n = ame_env_i64("GGML_AME_CKPT_N", 128);
    const int64_t target_k = ame_env_i64("GGML_AME_CKPT_K", 8192);
    return M == target_m && N == target_n && K == target_k;
}

static int ame_ckpt_tile_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_CKPT_TILE") ? 1 : 0;
    }
    return cached;
}

static uint64_t ame_ckpt_tile_ordinal(void) {
    const int64_t value = ame_env_i64("GGML_AME_CKPT_TILE_ORDINAL", 0);
    return value > 0 ? (uint64_t) value : 0;
}

static int ame_ckpt_rebase_every_tile_enabled(void) {
    static int enabled = -1;

    if (enabled == -1) {
        enabled = ame_env_enabled("GGML_AME_CKPT_REBASE_EVERY_TILE") ? 1 : 0;
    }
    return enabled;
}

static void ame_ckpt_rebase_after_tile(void) {
    if (!ame_ckpt_rebase_every_tile_enabled()) {
        return;
    }

    // gemm() returns only after mrelease/macquire/mfence, so no AME work is
    // outstanding when the checkpoint-only token epoch is rebased.
    ggml_ame_sync_begin_op();
}

static void ame_ckpt_notify_start(int64_t M, int64_t N, int64_t K, int64_t j0) {
    static int emitted = 0;
    static int64_t occurrence = 0;
    if (ame_ckpt_tile_enabled() || emitted || !ame_ckpt_shape_matches(M, N, K)) {
        return;
    }

    const int64_t target_occurrence = ame_env_i64("GGML_AME_CKPT_START_OCCURRENCE", 1);
    ++occurrence;
    if (occurrence != (target_occurrence > 0 ? target_occurrence : 1)) {
        return;
    }

    emitted = 1;
    fprintf(stderr,
            "[AME_CKPT] start ROI before AME accumulation: M=%lld N=%lld K=%lld j0=%lld occurrence=%lld\n",
            (long long) M, (long long) N, (long long) K, (long long) j0,
            (long long) occurrence);
    fflush(stderr);
    ame_nemu_signal(0x100);
    ame_nemu_signal(0x101);
    // The single-slice flow snapshots 100 retired instructions after 0x101.
    // Keep the cycle sample beyond that point so checkpoint restore executes
    // it in the RTL cycle domain. These handshake instructions precede the ROI.
    for (int i = 0; i < 256; ++i) {
#if defined(__riscv)
        asm volatile("nop");
#endif
    }
    g_ame_ckpt_roi_start_cycle = ame_read_cycle();
    g_ame_ckpt_roi_active = 1;
}

static void ame_ckpt_notify_stop(int64_t M, int64_t N, int64_t K) {
    static int emitted = 0;
    static int64_t occurrence = 0;
    if (ame_ckpt_tile_enabled() || emitted || !ame_ckpt_shape_matches(M, N, K)) {
        return;
    }

    const int64_t target_occurrence = ame_env_i64("GGML_AME_CKPT_STOP_OCCURRENCE", 1);
    ++occurrence;
    if (occurrence != (target_occurrence > 0 ? target_occurrence : 1)) {
        return;
    }

    const uint64_t stop_cycle = ame_read_cycle();
    const uint64_t roi_cycles = g_ame_ckpt_roi_active ?
        stop_cycle - g_ame_ckpt_roi_start_cycle : 0;
    emitted = 1;
    fprintf(stderr,
            "[AME_CKPT] stop ROI after AME accumulation: M=%lld N=%lld K=%lld "
            "occurrence=%lld pipeline=%d acc_pipeline=%d transposed_c=%d roi_cycles=%llu\n",
            (long long) M, (long long) N, (long long) K, (long long) occurrence,
            ame_whole_k_output_pipeline_enabled(), ame_whole_k_acc_pipeline_enabled(),
            ame_transposed_c_epilogue_enabled(),
            (unsigned long long) roi_cycles);
    fflush(stderr);
    ame_nemu_signal(0x102);
}

static uint64_t ame_ckpt_notify_tile_start(
    int64_t M,
    int64_t N,
    int64_t K,
    uint64_t tile,
    int64_t i0,
    int64_t j0,
    int64_t kb
) {
    static uint64_t occurrence = 0;
    if (!ame_ckpt_tile_enabled() || !ame_ckpt_shape_matches(M, N, K) ||
            tile != ame_ckpt_tile_ordinal()) {
        return 0;
    }

    const int64_t target_occurrence = ame_env_i64("GGML_AME_CKPT_START_OCCURRENCE", 1);
    const uint64_t current_occurrence = __atomic_add_fetch(&occurrence, 1, __ATOMIC_RELAXED);
    if (current_occurrence != (uint64_t) (target_occurrence > 0 ? target_occurrence : 1)) {
        return 0;
    }

#if defined(__riscv)
    // Scalar tile-buffer stores are not part of the architectural checkpoint.
    asm volatile("fence rw, rw" ::: "memory");
#endif
    fprintf(stderr,
            "[AME_CKPT] start tile ROI before AME sequence: M=%lld N=%lld K=%lld "
            "tile=%llu i0=%lld j0=%lld kb=%lld occurrence=%lld\n",
            (long long) M, (long long) N, (long long) K,
            (unsigned long long) tile, (long long) i0, (long long) j0, (long long) kb,
            (long long) current_occurrence);
    fflush(stderr);
    ame_nemu_signal(0x100);
    ame_nemu_signal(0x101);
    return current_occurrence;
}

static void ame_ckpt_notify_tile_stop(
    uint64_t occurrence,
    int64_t M,
    int64_t N,
    int64_t K,
    uint64_t tile,
    int64_t i0,
    int64_t j0,
    int64_t kb
) {
    if (occurrence == 0) {
        return;
    }
    fprintf(stderr,
            "[AME_CKPT] stop tile ROI after AME sequence: M=%lld N=%lld K=%lld "
            "tile=%llu i0=%lld j0=%lld kb=%lld occurrence=%lld\n",
            (long long) M, (long long) N, (long long) K,
            (unsigned long long) tile, (long long) i0, (long long) j0, (long long) kb,
            (long long) occurrence);
    fflush(stderr);
    ame_nemu_signal(0x102);
}

#define AME_PANEL_LOG(...) do {                         \
    if (ame_panel_log_enabled()) {                      \
        fprintf(stderr, "[AME_PANEL] " __VA_ARGS__);    \
        fprintf(stderr, "\n");                         \
        fflush(stderr);                                 \
    }                                                   \
} while (0)

#define AME_PANEL_PROGRESS_LOG(...) do {                         \
    if (ame_panel_progress_log_enabled()) {                      \
        fprintf(stderr, "[AME_PANEL_PROGRESS] " __VA_ARGS__);    \
        fprintf(stderr, "\n");                                  \
        fflush(stderr);                                         \
    }                                                           \
} while (0)

static void ame_profile_dump(void) {
    if (!ame_profile_enabled()) {
        return;
    }

    const uint64_t calls = g_ame_profile.baseline_calls + g_ame_profile.packed_calls;
    if (calls == 0) {
        return;
    }

    const uint64_t ops = g_ame_profile.logical_macs * 2;
    const uint64_t cycles_total = g_ame_profile.cycles_baseline_total + g_ame_profile.cycles_packed_total;
    const uint64_t tile_a_bytes = AME_TILE_M * AME_TILE_K * (uint64_t) sizeof(int8_t);
    const uint64_t tile_b_bytes = AME_TILE_N * AME_TILE_K * (uint64_t) sizeof(int8_t);
    const uint64_t tile_c_i32_bytes = AME_TILE_M * AME_TILE_N * (uint64_t) sizeof(int32_t);
    const uint64_t output_tile_f32_bytes = AME_TILE_M * AME_TILE_N * (uint64_t) sizeof(float);
    const uint64_t ame_tile_bytes_model = g_ame_profile.tile_calls *
        (tile_a_bytes + tile_b_bytes + tile_c_i32_bytes);
    const uint64_t tile_c_cpu_read_bytes_model = g_ame_profile.tile_calls * tile_c_i32_bytes;
    const uint64_t dst_f32_write_bytes_model = g_ame_profile.output_tile_calls * output_tile_f32_bytes;
    const uint64_t sw_bytes_model = ame_tile_bytes_model + tile_c_cpu_read_bytes_model + dst_f32_write_bytes_model;
    const double ops_per_cycle = cycles_total ? (double) ops / (double) cycles_total : 0.0;
    const double macs_per_cycle = cycles_total ? (double) g_ame_profile.logical_macs / (double) cycles_total : 0.0;
    const double ame_tile_bytes_per_cycle = cycles_total ? (double) ame_tile_bytes_model / (double) cycles_total : 0.0;
    const double sw_bytes_per_cycle = cycles_total ? (double) sw_bytes_model / (double) cycles_total : 0.0;
    const uint64_t quant_profiled = g_ame_profile.cycles_quant_scan + g_ame_profile.cycles_quant_convert;
    const uint64_t prepare_other = g_ame_profile.cycles_prepare_cache > quant_profiled ?
        g_ame_profile.cycles_prepare_cache - quant_profiled : 0;

    fprintf(stderr,
        "[AME_PROFILE] calls baseline=%llu packed=%llu tiles=%llu whole_k_kloop=%llu whole_k_pipeline=%llu edge_tiles=%llu output_tiles=%llu logical_macs=%llu logical_ops=%llu\n",
        (unsigned long long) g_ame_profile.baseline_calls,
        (unsigned long long) g_ame_profile.packed_calls,
        (unsigned long long) g_ame_profile.tile_calls,
        (unsigned long long) g_ame_profile.whole_k_kloop_calls,
        (unsigned long long) g_ame_profile.whole_k_pipeline_calls,
        (unsigned long long) g_ame_profile.edge_tile_calls,
        (unsigned long long) g_ame_profile.output_tile_calls,
        (unsigned long long) g_ame_profile.logical_macs,
        (unsigned long long) ops);
    fprintf(stderr,
        "[AME_PROFILE] cycles total=%llu baseline_total=%llu packed_total=%llu prepare_cache=%llu quant_scan=%llu quant_convert=%llu prepare_other=%llu pack_a=%llu pack_b=%llu zero_c=%llu ame_tile_call=%llu accumulator_initialization=%llu scale=%llu store=%llu\n",
        (unsigned long long) cycles_total,
        (unsigned long long) g_ame_profile.cycles_baseline_total,
        (unsigned long long) g_ame_profile.cycles_packed_total,
        (unsigned long long) g_ame_profile.cycles_prepare_cache,
        (unsigned long long) g_ame_profile.cycles_quant_scan,
        (unsigned long long) g_ame_profile.cycles_quant_convert,
        (unsigned long long) prepare_other,
        (unsigned long long) g_ame_profile.cycles_pack_a,
        (unsigned long long) g_ame_profile.cycles_pack_b,
        (unsigned long long) g_ame_profile.cycles_zero_c,
        (unsigned long long) g_ame_profile.cycles_ame_tile_call,
        (unsigned long long) g_ame_profile.cycles_accumulator_initialization,
        (unsigned long long) g_ame_profile.cycles_scale,
        (unsigned long long) g_ame_profile.cycles_store);
    fprintf(stderr,
        "[AME_PROFILE] derived ops_per_cycle=%.2f macs_per_cycle=%.2f ame_tile_bytes_model=%llu tile_c_cpu_read_bytes_model=%llu dst_f32_write_bytes_model=%llu sw_bytes_model=%llu ame_tile_B_per_cycle=%.4f sw_B_per_cycle=%.4f\n",
        ops_per_cycle,
        macs_per_cycle,
        (unsigned long long) ame_tile_bytes_model,
        (unsigned long long) tile_c_cpu_read_bytes_model,
        (unsigned long long) dst_f32_write_bytes_model,
        (unsigned long long) sw_bytes_model,
        ame_tile_bytes_per_cycle,
        sw_bytes_per_cycle);
    fflush(stderr);
}

void ggml_ame_profile_reset(void) {
    memset(&g_ame_profile, 0, sizeof(g_ame_profile));
}

void ggml_ame_profile_dump_now(void) {
    ame_profile_dump();
}

static int ame_profile_enabled(void) {
    static int cached = -1;
    static int registered = 0;
    if (cached == -1) {
        cached = ame_env_enabled("GGML_AME_PROFILE") ? 1 : 0;
    }
    if (cached && !registered) {
        atexit(ame_profile_dump);
        registered = 1;
    }
    return cached;
}

static void ame_profile_accumulate(const struct ame_profile_counters * local) {
    g_ame_profile.baseline_calls       += local->baseline_calls;
    g_ame_profile.packed_calls         += local->packed_calls;
    g_ame_profile.logical_macs         += local->logical_macs;
    g_ame_profile.tile_calls           += local->tile_calls;
    g_ame_profile.whole_k_kloop_calls  += local->whole_k_kloop_calls;
    g_ame_profile.whole_k_pipeline_calls += local->whole_k_pipeline_calls;
    g_ame_profile.edge_tile_calls      += local->edge_tile_calls;
    g_ame_profile.output_tile_calls    += local->output_tile_calls;
    g_ame_profile.cycles_baseline_total += local->cycles_baseline_total;
    g_ame_profile.cycles_packed_total  += local->cycles_packed_total;
    g_ame_profile.cycles_prepare_cache += local->cycles_prepare_cache;
    g_ame_profile.cycles_quant_scan    += local->cycles_quant_scan;
    g_ame_profile.cycles_quant_convert += local->cycles_quant_convert;
    g_ame_profile.cycles_pack_a        += local->cycles_pack_a;
    g_ame_profile.cycles_pack_b        += local->cycles_pack_b;
    g_ame_profile.cycles_zero_c        += local->cycles_zero_c;
    g_ame_profile.cycles_ame_tile_call += local->cycles_ame_tile_call;
    g_ame_profile.cycles_accumulator_initialization += local->cycles_accumulator_initialization;
    g_ame_profile.cycles_scale         += local->cycles_scale;
    g_ame_profile.cycles_store         += local->cycles_store;
}

/* Method A safety gate: CUTE AMU computes per-row addresses as
 *   PA_row = PA_base + row * stride
 * without issuing a TLB re-query for each virtual page boundary.
 * This is only correct when the tile buffers are physically contiguous.
 * tile_a = AME_TILE_M * AME_TILE_K = 128*64 = 8192 bytes spans 3 virtual
 * pages, so non-contiguous allocation silently corrupts rows >= 63 and >=127.
 */
static void ame_assert_phys_contiguous(void) {
#if defined(GGML_USE_RV_AME) && defined(GGML_XSAI_ALLOC)
    static int checked = 0;
    if (checked) return;
    checked = 1;
    if (!xsai_pool_phys_contiguous() && !xsai_alloc_host_test_mode()) {
        fprintf(stderr,
            "[AME] FATAL: xsai memory pool is not physically contiguous.\n"
            "             CUTE AMU row-address formula: PA_row = PA_base + row*stride\n"
            "             (no per-row TLB re-query, LocalMMU TODO not yet implemented).\n"
            "             tile_a (8192 B) spans 3 virtual pages; rows >=63 and >=127\n"
            "             will read wrong physical addresses => silent data corruption.\n"
            "             Fix: boot kernel with hugepages=512, or define\n"
            "             RESERVED_PHYS_BASE_ADDR for /dev/mem-backed pool.\n");
        abort();
    }
    if (!xsai_pool_phys_contiguous()) {
        fprintf(stderr,
            "[AME] WARNING: allowing non-physical xsai allocator in explicit host-test mode.\n"
            "              This is intended for qemu-user correctness tests only.\n");
    }
#endif
}

// Forward declaration of atomic GEMM function
extern void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,      // Input matrix A: MxK
    const int8_t * B,      // Input matrix B (transposed): NxK
    int32_t * C            // Output matrix C: MxN
);

extern void ggml_ame_gemm_tile_bf16_fp32_bT(
    const ggml_bf16_t * A,
    const ggml_bf16_t * B,
    float * C
);

#if defined(__riscv_v)
#include <riscv_vector.h>

// Dot product using RVV for Q8_0 blocks
// Compatible with block_q8_0 layout
static void ame_vec_dot_q8_0_rvv(int n, float * s, const void * vx, const void * vy) {
    const int qk = 32;
    const int nb = n / qk;
    const block_q8_0 * restrict x = (const block_q8_0 *)vx;
    const block_q8_0 * restrict y = (const block_q8_0 *)vy;
    
    float sumf = 0;
    
    for (int i = 0; i < nb; ++i) {
        // Each Q8_0 block has 32 int8 elements
        // Use vsetvli to handle any VLEN (128, 256, 512, 1024, etc.)
        size_t vl = __riscv_vsetvl_e8m2(qk);
        
        int sumi = 0;
        size_t offset = 0;
        
        // Process block in chunks that fit in vector registers
        while (offset < qk) {
            vl = __riscv_vsetvl_e8m2(qk - offset);
            
            // Load elements
            vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[i].qs + offset, vl);
            vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[i].qs + offset, vl);

            // Widen multiply: int8 * int8 -> int16
            vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);

            // Reduce sum: int16 -> int32
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);

            sumi += __riscv_vmv_x_s_i32m1_i32(v_sum);
            offset += vl;
        }

        sumf += sumi * (GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));
    }
    *s = sumf;
}

// Dot product for Q4_0 (standard format) using RVV
static void ame_vec_dot_q4_0_rvv(int n, float * s, const void * vx, const void * vy) {
    const int qk = 32;
    const int nb = n / qk;
    const block_q4_0 * restrict x = (const block_q4_0 *)vx;
    const block_q8_0 * restrict y = (const block_q8_0 *)vy; // y is always quantized to Q8_0 in our gemm
    
    float sumf = 0;
    
    for (int i = 0; i < nb; ++i) {
        // Unpack Q4_0 block (4-bit to 8-bit)
        int8_t x_unpacked[32];
        for (int j = 0; j < 16; j++) {
            uint8_t v = x[i].qs[j];
            x_unpacked[j] = (int8_t)(v & 0x0F) - 8;
            x_unpacked[j+16] = (int8_t)((v >> 4) & 0x0F) - 8;
        }

        // Now compute dot product with proper vsetvl
        int sumi = 0;
        size_t offset = 0;
        
        while (offset < qk) {
            size_t vl = __riscv_vsetvl_e8m2(qk - offset);
            
            vint8m2_t bx = __riscv_vle8_v_i8m2(x_unpacked + offset, vl);
            vint8m2_t by = __riscv_vle8_v_i8m2(y[i].qs + offset, vl);
            
            vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx, by, vl);
            
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);
            
            sumi += __riscv_vmv_x_s_i32m1_i32(v_sum);
            offset += vl;
        }
        
        sumf += sumi * (GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));
    }
    *s = sumf;
}
#endif

static float ame_vec_dot_bf16_scalar(int n, const ggml_bf16_t * x, const ggml_bf16_t * y) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_BF16_TO_FP32(x[i]) * GGML_BF16_TO_FP32(y[i]);
    }
    return sum;
}

static float ame_vec_dot_bf16_fallback(int n, const ggml_bf16_t * x, const ggml_bf16_t * y) {
#if defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfwma)
    float sum = 0.0f;
    ggml_vec_dot_bf16(n, &sum, 0, (ggml_bf16_t *) x, 0, (ggml_bf16_t *) y, 0, 1);
    return sum;
#else
    return ame_vec_dot_bf16_scalar(n, x, y);
#endif
}

// ggml_ame_quantize_row_f32_to_q8_0 is now in ame-helper.c

static size_t ame_align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static inline void ggml_ame_quantize_block_f32_to_q8_64(const float * x, int valid, block_q8_ame64 * y) {
#if defined(__riscv_v)
    if (valid <= 0) {
        y->d = GGML_FP32_TO_FP16(0.0f);
        memset(y->qs, 0, AME_Q8_PACK_K);
        return;
    }

    float amax = 0.0f;
    int offset = 0;
    while (offset < valid) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (valid - offset));
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + offset, vl);
        vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vx, vl);
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vabs, vzero, vl);
        const float chunk_amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
        if (chunk_amax > amax) {
            amax = chunk_amax;
        }
        offset += (int) vl;
    }

    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    y->d = GGML_FP32_TO_FP16(d);

    offset = 0;
    while (offset < valid) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (valid - offset));
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + offset, vl);
        vx = __riscv_vfmul_vf_f32m8(vx, id, vl);
        vint16m4_t vi16 = __riscv_vfncvt_x_f_w_i16m4(vx, vl);
        vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vl);
        __riscv_vse8_v_i8m2(y->qs + offset, vi8, vl);
        offset += (int) vl;
    }

    if (valid < AME_Q8_PACK_K) {
        memset(y->qs + valid, 0, (size_t) (AME_Q8_PACK_K - valid));
    }
#else
    float tmp[AME_Q8_PACK_K];
    memset(tmp, 0, sizeof(tmp));
    if (valid > 0) {
        memcpy(tmp, x, (size_t) valid * sizeof(float));
    }

    float amax = 0.0f;
    for (int j = 0; j < AME_Q8_PACK_K; ++j) {
        const float av = fabsf(tmp[j]);
        if (av > amax) {
            amax = av;
        }
    }

    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;

    y->d = GGML_FP32_TO_FP16(d);
    for (int j = 0; j < AME_Q8_PACK_K; ++j) {
        y->qs[j] = roundf(tmp[j] * id);
    }
#endif
}

static void ame_quantize_row_f32_to_q8_whole_k(
    const float * x,
    int64_t k,
    block_q8_ame64 * blocks,
    float * scales,
    struct ame_profile_counters * prof
) {
    const int64_t nb64 = (k + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    float amax = 0.0f;
    uint64_t prof_t0 = prof != NULL ? ame_read_cycle() : 0;

#if defined(__riscv_v)
    int64_t offset = 0;
    while (offset < k) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (k - offset));
        const vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + offset, vl);
        const vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vx, vl);
        const vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        const vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vabs, vzero, vl);
        const float chunk_amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
        if (chunk_amax > amax) {
            amax = chunk_amax;
        }
        offset += (int64_t) vl;
    }
#else
    for (int64_t i = 0; i < k; ++i) {
        const float av = fabsf(x[i]);
        if (av > amax) {
            amax = av;
        }
    }
#endif
    if (prof != NULL) {
        prof->cycles_quant_scan += ame_read_cycle() - prof_t0;
        prof_t0 = ame_read_cycle();
    }

    const float delta = amax / 127.0f;
    const float inv_delta = delta ? 1.0f / delta : 0.0f;
    const ggml_fp16_t delta_fp16 = GGML_FP32_TO_FP16(delta);
    const float stored_delta = GGML_FP16_TO_FP32(delta_fp16);

    for (int64_t kb = 0; kb < nb64; ++kb) {
        block_q8_ame64 * block = &blocks[kb];
        block->d = delta_fp16;
        scales[kb] = stored_delta;
        const int64_t base = kb * AME_Q8_PACK_K;
        const int valid = base + AME_Q8_PACK_K <= k ?
            AME_Q8_PACK_K : (base < k ? (int) (k - base) : 0);

#if defined(__riscv_v)
        int block_offset = 0;
        while (block_offset < valid) {
            const size_t vl = __riscv_vsetvl_e32m8((size_t) (valid - block_offset));
            vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + base + block_offset, vl);
            vx = __riscv_vfmul_vf_f32m8(vx, inv_delta, vl);
            const vint16m4_t vi16 = __riscv_vfncvt_x_f_w_i16m4(vx, vl);
            const vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vl);
            __riscv_vse8_v_i8m2(block->qs + block_offset, vi8, vl);
            block_offset += (int) vl;
        }
        if (valid < AME_Q8_PACK_K) {
            memset(block->qs + valid, 0, (size_t) (AME_Q8_PACK_K - valid));
        }
#else
        for (int j = 0; j < AME_Q8_PACK_K; ++j) {
            const int64_t index = base + j;
            if (index >= k) {
                block->qs[j] = 0;
                continue;
            }

            float quantized = roundf(x[index] * inv_delta);
            if (quantized > 127.0f) {
                quantized = 127.0f;
            } else if (quantized < -127.0f) {
                quantized = -127.0f;
            }
            block->qs[j] = (int8_t) quantized;
        }
#endif
    }
    if (prof != NULL) {
        prof->cycles_quant_convert += ame_read_cycle() - prof_t0;
    }
}

static inline void ame_accumulate_scaled_row(
    float * restrict acc,
    const int32_t * restrict c,
    const float d_x,
    const float * restrict y_scales,
    const int jmax
) {
#if defined(__riscv_v)
    int j = 0;
    while (j < jmax) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (jmax - j));
        vint32m8_t vc_i32 = __riscv_vle32_v_i32m8(c + j, vl);
        vfloat32m8_t vc_f32 = __riscv_vfcvt_f_x_v_f32m8(vc_i32, vl);
        vfloat32m8_t vs = __riscv_vle32_v_f32m8(y_scales + j, vl);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(acc + j, vl);

        vs = __riscv_vfmul_vf_f32m8(vs, d_x, vl);
        vacc = __riscv_vfmacc_vv_f32m8(vacc, vc_f32, vs, vl);
        __riscv_vse32_v_f32m8(acc + j, vacc, vl);
        j += (int) vl;
    }
#else
    for (int j = 0; j < jmax; ++j) {
        acc[j] += c[j] * (d_x * y_scales[j]);
    }
#endif
}

static inline void ame_accumulate_scaled_tile(
    float * restrict acc,
    const int32_t * restrict c,
    const float * restrict x_scales,
    const float * restrict y_scales,
    const int tile_n,
    const int imax,
    const int jmax
) {
#if defined(__riscv_v)
    int j = 0;
    while (j < jmax) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (jmax - j));
        const vfloat32m8_t vy = __riscv_vle32_v_f32m8(y_scales + j, vl);

        for (int i = 0; i < imax; ++i) {
            float * restrict acc_row = acc + i * tile_n + j;
            const int32_t * restrict c_row = c + i * tile_n + j;

            const vint32m8_t vc_i32 = __riscv_vle32_v_i32m8(c_row, vl);
            const vfloat32m8_t vc_f32 = __riscv_vfcvt_f_x_v_f32m8(vc_i32, vl);
            const vfloat32m8_t vscale = __riscv_vfmul_vf_f32m8(vy, x_scales[i], vl);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(acc_row, vl);

            vacc = __riscv_vfmacc_vv_f32m8(vacc, vc_f32, vscale, vl);
            __riscv_vse32_v_f32m8(acc_row, vacc, vl);
        }

        j += (int) vl;
    }
#else
    for (int i = 0; i < imax; ++i) {
        const float d_x = x_scales[i];
        for (int j = 0; j < jmax; ++j) {
            acc[i * tile_n + j] += c[i * tile_n + j] * (d_x * y_scales[j]);
        }
    }
#endif
}

static inline void ame_store_acc_tile(
    float * restrict out,
    const float * restrict acc,
    const int64_t M,
    const int tile_n,
    const int imax,
    const int jmax
) {
#if defined(__riscv_v)
    const ptrdiff_t acc_stride = (ptrdiff_t) tile_n * (ptrdiff_t) sizeof(float);
    for (int j = 0; j < jmax; ++j) {
        int i = 0;
        while (i < imax) {
            const size_t vl = __riscv_vsetvl_e32m8((size_t) (imax - i));
            const vfloat32m8_t v = __riscv_vlse32_v_f32m8(acc + i * tile_n + j, acc_stride, vl);
            __riscv_vse32_v_f32m8(out + (int64_t) j * M + i, v, vl);
            i += (int) vl;
        }
    }
#else
    for (int j = 0; j < jmax; ++j) {
        for (int i = 0; i < imax; ++i) {
            out[(int64_t) j * M + i] = acc[i * tile_n + j];
        }
    }
#endif
}

static inline void ame_store_scaled_tile(
    float * restrict out,
    const int32_t * restrict c,
    const float * restrict x_scales,
    const float * restrict y_scales,
    const int64_t M,
    const int tile_n,
    const int imax,
    const int jmax
) {
#if defined(__riscv_v)
    const ptrdiff_t c_stride = (ptrdiff_t) tile_n * (ptrdiff_t) sizeof(int32_t);
    for (int j = 0; j < jmax; ++j) {
        int i = 0;
        while (i < imax) {
            const size_t vl = __riscv_vsetvl_e32m8((size_t) (imax - i));
            const vint32m8_t vc_i32 = __riscv_vlse32_v_i32m8(c + i * tile_n + j, c_stride, vl);
            vfloat32m8_t vc_f32 = __riscv_vfcvt_f_x_v_f32m8(vc_i32, vl);
            vfloat32m8_t vscale = __riscv_vle32_v_f32m8(x_scales + i, vl);

            vscale = __riscv_vfmul_vf_f32m8(vscale, y_scales[j], vl);
            vc_f32 = __riscv_vfmul_vv_f32m8(vc_f32, vscale, vl);
            __riscv_vse32_v_f32m8(out + (int64_t) j * M + i, vc_f32, vl);
            i += (int) vl;
        }
    }
#else
    for (int j = 0; j < jmax; ++j) {
        const float d_y = y_scales[j];
        for (int i = 0; i < imax; ++i) {
            out[(int64_t) j * M + i] = c[i * tile_n + j] * (x_scales[i] * d_y);
        }
    }
#endif
}

static inline void ame_store_scaled_tile_transposed(
    float * restrict out,
    const int32_t * restrict c,
    const float * restrict x_scales,
    const float * restrict y_scales,
    const int64_t M,
    const int tile_m,
    const int imax,
    const int jmax
) {
#if defined(__riscv_v)
    for (int j = 0; j < jmax; ++j) {
        int i = 0;
        while (i < imax) {
            const size_t vl = __riscv_vsetvl_e32m8((size_t) (imax - i));
            const vint32m8_t vc_i32 = __riscv_vle32_v_i32m8(c + j * tile_m + i, vl);
            vfloat32m8_t vc_f32 = __riscv_vfcvt_f_x_v_f32m8(vc_i32, vl);
            vfloat32m8_t vscale = __riscv_vle32_v_f32m8(x_scales + i, vl);

            vscale = __riscv_vfmul_vf_f32m8(vscale, y_scales[j], vl);
            vc_f32 = __riscv_vfmul_vv_f32m8(vc_f32, vscale, vl);
            __riscv_vse32_v_f32m8(out + (int64_t) j * M + i, vc_f32, vl);
            i += (int) vl;
        }
    }
#else
    for (int j = 0; j < jmax; ++j) {
        const float d_y = y_scales[j];
        for (int i = 0; i < imax; ++i) {
            out[(int64_t) j * M + i] = c[j * tile_m + i] * (x_scales[i] * d_y);
        }
    }
#endif
}

static inline void ame_finish_whole_k_pipeline_tile(
    float * restrict out,
    const int32_t * restrict c,
    const float * restrict src0_tile_scales,
    const block_q8_ame64 * restrict xq,
    const float * restrict xq_scales,
    const int64_t mt,
    const int64_t j0,
    const int64_t nb64,
    const int64_t M,
    const int tile_m,
    const int tile_n,
    const int imax,
    const int jmax,
    const int transposed_c,
    struct ame_profile_counters * prof
) {
    uint64_t prof_t0 = prof != NULL ? ame_read_cycle() : 0;
    float x_scales[AME_TILE_M_MAX];
    float y_scales[AME_TILE_N_MAX];
    const float * restrict cached_scales =
        src0_tile_scales + (size_t) mt * (size_t) nb64 * (size_t) tile_m;
    memcpy(x_scales, cached_scales, (size_t) imax * sizeof(float));
    for (int j = 0; j < jmax; ++j) {
        const block_q8_ame64 * bx = &xq[(j0 + j) * nb64];
        y_scales[j] = xq_scales != NULL ?
            xq_scales[(j0 + j) * nb64] : GGML_FP16_TO_FP32(bx->d);
    }
    if (transposed_c) {
        ame_store_scaled_tile_transposed(out, c, x_scales, y_scales, M, tile_m, imax, jmax);
    } else {
        ame_store_scaled_tile(out, c, x_scales, y_scales, M, tile_n, imax, jmax);
    }
    if (prof != NULL) {
        prof->cycles_scale += ame_read_cycle() - prof_t0;
    }
}

struct ame_whole_k_overlap_context {
    float * out;
    const int32_t * c;
    const float * src0_tile_scales;
    const block_q8_ame64 * xq;
    const float * xq_scales;
    int64_t mt;
    int64_t j0;
    int64_t nb64;
    int64_t M;
    int tile_m;
    int tile_n;
    int imax;
    int jmax;
    int transposed_c;
    struct ame_profile_counters * prof;
};

static void ame_finish_whole_k_pipeline_tile_callback(void * opaque) {
    struct ame_whole_k_overlap_context * ctx =
        (struct ame_whole_k_overlap_context *) opaque;
    ame_finish_whole_k_pipeline_tile(
        ctx->out,
        ctx->c,
        ctx->src0_tile_scales,
        ctx->xq,
        ctx->xq_scales,
        ctx->mt,
        ctx->j0,
        ctx->nb64,
        ctx->M,
        ctx->tile_m,
        ctx->tile_n,
        ctx->imax,
        ctx->jmax,
        ctx->transposed_c,
        ctx->prof);
}

struct ame_whole_k_acc_pipeline_context {
    float * out;
    const int32_t * c_slots;
    const float * src0_tile_scales;
    const float * y_scales;
    int64_t nb64;
    int64_t M;
    size_t tile_c_elements;
    int tile_m;
    int tile_n;
    int jmax;
    int transposed_c;
    struct ame_profile_counters * prof;
};

static void ame_finish_whole_k_acc_pipeline_tile(void * opaque, int tile_index) {
    struct ame_whole_k_acc_pipeline_context * ctx =
        (struct ame_whole_k_acc_pipeline_context *) opaque;
    const int64_t i0 = (int64_t) tile_index * ctx->tile_m;
    const int imax = i0 + ctx->tile_m <= ctx->M ? ctx->tile_m : (int) (ctx->M - i0);
    const int slot = tile_index & 1;
    const float * restrict x_scales =
        ctx->src0_tile_scales + (size_t) tile_index * (size_t) ctx->nb64 * (size_t) ctx->tile_m;
    const int32_t * restrict c = ctx->c_slots + (size_t) slot * ctx->tile_c_elements;
    const uint64_t prof_t0 = ctx->prof != NULL ? ame_read_cycle() : 0;

    if (ctx->transposed_c) {
        ame_store_scaled_tile_transposed(
            ctx->out + i0, c, x_scales, ctx->y_scales,
            ctx->M, ctx->tile_m, imax, ctx->jmax);
    } else {
        ame_store_scaled_tile(
            ctx->out + i0, c, x_scales, ctx->y_scales,
            ctx->M, ctx->tile_n, imax, ctx->jmax);
    }
    if (ctx->prof != NULL) {
        ctx->prof->cycles_scale += ame_read_cycle() - prof_t0;
        ctx->prof->output_tile_calls++;
    }
}

struct ame_x_q64_cache {
    const void * key;
    const void * graph_key;
    int graph_id;
    uint64_t generation;
    size_t stride;
    int64_t K;
    int64_t N;
    block_q8_ame64 * blocks;
    float * scales;
    size_t blocks_count;
};

static struct ame_x_q64_cache g_ame_x_q64_cache = {0};
static uint64_t g_ame_x_q64_oneshot_generation = 1;

static const block_q8_ame64 * ame_prepare_x_q64_cache(
    const void * src1_key,
    const void * src1,
    int64_t K,
    int64_t N,
    size_t src1_stride,
    int graph_id,
    const void * graph_key,
    uint64_t src1_generation,
    const float ** scales_out,
    struct ame_profile_counters * prof
) {
    if (scales_out != NULL) {
        *scales_out = NULL;
    }
    if (src1_generation == 0 && (graph_id <= 0 || graph_key == NULL)) {
        return NULL;
    }

    const int64_t nb64 = (K + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    const size_t blocks_count = (size_t) N * (size_t) nb64;

    if (g_ame_x_q64_cache.key == src1_key &&
        g_ame_x_q64_cache.graph_key == graph_key &&
        g_ame_x_q64_cache.graph_id == graph_id &&
        g_ame_x_q64_cache.generation == src1_generation &&
        g_ame_x_q64_cache.stride == src1_stride &&
        g_ame_x_q64_cache.K == K &&
        g_ame_x_q64_cache.N == N &&
        g_ame_x_q64_cache.blocks != NULL) {
        if (scales_out != NULL) {
            *scales_out = g_ame_x_q64_cache.scales;
        }
        return g_ame_x_q64_cache.blocks;
    }

    const size_t alloc_size = blocks_count * sizeof(block_q8_ame64);
    const size_t scales_size = blocks_count * sizeof(float);
    if (g_ame_x_q64_cache.blocks == NULL || g_ame_x_q64_cache.blocks_count != blocks_count) {
        if (g_ame_x_q64_cache.blocks != NULL) {
            ggml_aligned_free(g_ame_x_q64_cache.blocks, g_ame_x_q64_cache.blocks_count * sizeof(block_q8_ame64));
        }
        if (g_ame_x_q64_cache.scales != NULL) {
            ggml_aligned_free(g_ame_x_q64_cache.scales, g_ame_x_q64_cache.blocks_count * sizeof(float));
        }
        g_ame_x_q64_cache.blocks = (block_q8_ame64 *) ggml_aligned_malloc(alloc_size);
        g_ame_x_q64_cache.scales = (float *) ggml_aligned_malloc(scales_size);
        if (g_ame_x_q64_cache.blocks == NULL || g_ame_x_q64_cache.scales == NULL) {
            if (g_ame_x_q64_cache.blocks != NULL) {
                ggml_aligned_free(g_ame_x_q64_cache.blocks, alloc_size);
            }
            if (g_ame_x_q64_cache.scales != NULL) {
                ggml_aligned_free(g_ame_x_q64_cache.scales, scales_size);
            }
            memset(&g_ame_x_q64_cache, 0, sizeof(g_ame_x_q64_cache));
            return NULL;
        }
        g_ame_x_q64_cache.blocks_count = blocks_count;
    }

    for (int64_t j = 0; j < N; ++j) {
        const float * src1_col = (const float *) ((const char *) src1 + j * src1_stride);
        if (ame_whole_k_q8_enabled()) {
            ame_quantize_row_f32_to_q8_whole_k(
                src1_col,
                K,
                &g_ame_x_q64_cache.blocks[j * nb64],
                &g_ame_x_q64_cache.scales[j * nb64],
                prof);
            continue;
        }

        for (int64_t kb = 0; kb < nb64; ++kb) {
            const int64_t base = kb * AME_Q8_PACK_K;
            const int valid = (base + AME_Q8_PACK_K <= K) ? AME_Q8_PACK_K : (K > base ? (int) (K - base) : 0);
            block_q8_ame64 * dst = &g_ame_x_q64_cache.blocks[j * nb64 + kb];
            ggml_ame_quantize_block_f32_to_q8_64(src1_col + base, valid, dst);
            g_ame_x_q64_cache.scales[j * nb64 + kb] = GGML_FP16_TO_FP32(dst->d);
        }
    }

    g_ame_x_q64_cache.key = src1_key;
    g_ame_x_q64_cache.graph_key = graph_key;
    g_ame_x_q64_cache.graph_id = graph_id;
    g_ame_x_q64_cache.generation = src1_generation;
    g_ame_x_q64_cache.stride = src1_stride;
    g_ame_x_q64_cache.K = K;
    g_ame_x_q64_cache.N = N;
    if (scales_out != NULL) {
        *scales_out = g_ame_x_q64_cache.scales;
    }
    return g_ame_x_q64_cache.blocks;
}

// Wrapper for AME-accelerated Q8_0 GEMM
void ggml_ame_mul_mat_q8_0(
    const void * src0,  // Weight matrix (Q8_0)
    const void * src1,  // Input matrix (F32)
    void * dst,         // Output (F32)
    int64_t ne00,       // K
    int64_t ne01,       // M
    int64_t ne10,       // K (unused)
    int64_t ne11,       // N
    size_t src1_stride,
    void * work_data,
    size_t work_size
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    
    const int qk = 32;
    const int64_t nb_x = K / qk;
    const int prof = ame_profile_enabled();
    struct ame_profile_counters prof_local = {0};
    uint64_t prof_total_start = 0;
    if (prof) {
        prof_total_start = ame_read_cycle();
        prof_local.baseline_calls = 1;
        prof_local.logical_macs = (uint64_t) M * (uint64_t) N * (uint64_t) K;
    }

    const block_q8_0 * restrict x = (const block_q8_0 *)src0;
    float * restrict out = (float *)dst;

    const ggml_ame_i8_kernel * kernel = ggml_ame_select_i8_kernel(M, N, K);
    if (kernel == NULL) {
        return;
    }
    const int tile_m = kernel->tile_m;
    const int tile_k = kernel->tile_k;
    const int tile_n = kernel->tile_n;
    const size_t required_wsize = ggml_ame_i8_kernel_workspace_size(kernel, N, K, 0);
    uint8_t * workspace = (uint8_t *)work_data;
    int allocated_workspace = 0;

    if (workspace == NULL || work_size < required_wsize) {
        workspace = (uint8_t *)ggml_aligned_malloc(required_wsize);
        if (!workspace) return;
        work_size = required_wsize;
        allocated_workspace = 1;
    }

    uintptr_t ws_ptr = (uintptr_t)workspace;
    uintptr_t ws_end = ws_ptr + work_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_a = (int8_t *)ws_ptr;
    ws_ptr += (size_t) tile_m * tile_k * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_b = (int8_t *)ws_ptr;
    ws_ptr += (size_t) tile_n * tile_k * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int32_t * tile_c = (int32_t *)ws_ptr;
    ws_ptr += (size_t) tile_m * tile_n * sizeof(int32_t);

    if (ws_ptr > ws_end) {
        if (allocated_workspace) {
            ggml_aligned_free(workspace, work_size);
        }
        return;
    }

    // Method A: verify physical contiguity contract before first AME instruction
    ame_assert_phys_contiguous();
    ggml_ame_sync_begin_op();
    memset(tile_a, 0, (size_t) tile_m * tile_k * sizeof(int8_t));
    memset(tile_b, 0, (size_t) tile_n * tile_k * sizeof(int8_t));

    for (int64_t i0 = 0; i0 < M; i0 += tile_m) {
        const int imax = (i0 + tile_m <= M) ? tile_m : (M - i0);

        for (int64_t j0 = 0; j0 < N; j0 += tile_n) {
            const int jmax = (j0 + tile_n <= N) ? tile_n : (N - j0);
            
            // Full AME Path for full tiles
            float acc_f32[AME_TILE_M_MAX * AME_TILE_N_MAX];
            uint64_t prof_t0 = prof ? ame_read_cycle() : 0;
            memset(acc_f32, 0, sizeof(acc_f32));
            if (prof) {
                prof_local.cycles_accumulator_initialization += ame_read_cycle() - prof_t0;
            }

            for (int64_t kb = 0; kb < nb_x; kb++) {
                float y_scales[AME_TILE_N_MAX];
                float x_scales[AME_TILE_M_MAX];
                memset(y_scales, 0, sizeof(y_scales));
                memset(x_scales, 0, sizeof(x_scales));

                // Prepare Tile A (16 x 32)
                prof_t0 = prof ? ame_read_cycle() : 0;
                for (int i = 0; i < tile_m; i++) {
                     if (i < imax) {
                         const block_q8_0 * b = &x[(i0 + i) * nb_x + kb];
                         memcpy(&tile_a[i * tile_k], b->qs, qk);
                     } else {
                         memset(&tile_a[i * tile_k], 0, qk);
                     }
                }
                if (prof) prof_local.cycles_pack_a += ame_read_cycle() - prof_t0;

                // Prepare Tile B (16 x 32)
                 prof_t0 = prof ? ame_read_cycle() : 0;
                 if (!ame_skip_tile_b_zero()) {
                     memset(tile_b, 0, (size_t) tile_n * tile_k * sizeof(int8_t));
                 }
                 for (int j = 0; j < jmax; j++) {
                     const float * src1_col = (const float *)((const char *)src1 + (j0 + j) * src1_stride);
                     block_q8_0 tmp_block;
                     ggml_ame_quantize_row_f32_to_q8_0(src1_col + kb * qk, &tmp_block, qk);
                     y_scales[j] = GGML_FP16_TO_FP32(tmp_block.d);
                     memcpy(&tile_b[j * tile_k], tmp_block.qs, qk);
                 }
                if (prof) prof_local.cycles_pack_b += ame_read_cycle() - prof_t0;

                prof_t0 = prof ? ame_read_cycle() : 0;
                if (!ame_skip_tile_c_zero()) {
                    memset(tile_c, 0, (size_t) tile_m * tile_n * sizeof(int32_t));
                    if (prof) prof_local.cycles_zero_c += ame_read_cycle() - prof_t0;
                }

                prof_t0 = prof ? ame_read_cycle() : 0;
                kernel->gemm(tile_a, tile_b, tile_c);
                ame_ckpt_rebase_after_tile();
                if (prof) {
                    prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                    prof_local.tile_calls++;
                    if (imax < tile_m || jmax < tile_n) {
                        prof_local.edge_tile_calls++;
                    }
                }

                // Accumulate scaling factors
                prof_t0 = prof ? ame_read_cycle() : 0;
                for (int i = 0; i < imax; i++) {
                     const block_q8_0 * bx = &x[(i0 + i) * nb_x + kb];
                     x_scales[i] = GGML_FP16_TO_FP32(bx->d);
                }
                ame_accumulate_scaled_tile(
                    acc_f32,
                    tile_c,
                    x_scales,
                    y_scales,
                    tile_n,
                    imax,
                    jmax);
                if (prof) prof_local.cycles_scale += ame_read_cycle() - prof_t0;
            }

            // Copy back
            prof_t0 = prof ? ame_read_cycle() : 0;
            ame_store_acc_tile(
                out + j0 * M + i0,
                acc_f32,
                M,
                tile_n,
                imax,
                jmax);
            if (prof) {
                prof_local.cycles_store += ame_read_cycle() - prof_t0;
                prof_local.output_tile_calls++;
            }
        }
    }

    if (prof) {
        prof_local.cycles_baseline_total = ame_read_cycle() - prof_total_start;
        ame_profile_accumulate(&prof_local);
    }

    if (allocated_workspace) {
        ggml_aligned_free(workspace, work_size);
    }
}

void ggml_ame_mul_mat_q8_0_ame64(
    const void * src0,
    const void * src1_key,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride,
    const int8_t * src0_tile_a,
    const float * src0_tile_scales,
    int graph_id,
    const void * graph_key,
    uint64_t src1_generation,
    void * work_data,
    size_t work_size
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    const int64_t nb64 = (K + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    const ggml_ame_i8_kernel * kernel = ggml_ame_select_i8_kernel(M, N, K);
    if (kernel == NULL) {
        return;
    }
    const int tile_m = kernel->tile_m;
    const int tile_k = kernel->tile_k;
    const int tile_n = kernel->tile_n;
    const int use_packed_b_panel = ame_use_packed_b_panel();
    const int use_packed_a_tiles =
        kernel->kind == GGML_AME_I8_KERNEL_64_64_64 &&
        !ame_disable_packed_a_tiles() &&
        src0_tile_a != NULL &&
        src0_tile_scales != NULL;
    const int async_batch_requested = ame_async_batch_size();
    const int async_reg_pairs_requested = ame_async_register_pairs();
    const int async_reg_pairs =
        async_reg_pairs_requested >= 2 && kernel->submit_alt != NULL ? 2 : 1;
    const int async_batch_limit = async_reg_pairs >= 2 ? 2 : 4;
    const int async_batch =
        use_packed_b_panel && use_packed_a_tiles && kernel->submit != NULL ?
        (async_batch_requested < async_batch_limit ? async_batch_requested : async_batch_limit) : 0;
    const int64_t sparse_interval = ame_sparse_progress_interval();
    const int sparse_progress = sparse_interval > 0 && ame_sparse_progress_shape_matches(M, N, K);
    const int use_whole_k_kloop =
        use_packed_b_panel &&
        use_packed_a_tiles &&
        ame_whole_k_q8_enabled() &&
        kernel->kind == GGML_AME_I8_KERNEL_64_64_64 &&
        async_batch == 0 &&
        !sparse_progress &&
        !ame_ckpt_tile_enabled() &&
        nb64 <= INT_MAX &&
        K <= INT32_MAX / (127 * 127);
    const int use_whole_k_fused_epilogue =
        use_whole_k_kloop && ame_whole_k_fused_epilogue_enabled();
    const int use_whole_k_output_pipeline =
        use_whole_k_fused_epilogue && ame_whole_k_output_pipeline_enabled();
    const int use_whole_k_acc_pipeline =
        use_whole_k_output_pipeline &&
        ame_whole_k_acc_pipeline_enabled() &&
        !ame_ckpt_rebase_every_tile_enabled();
    const int use_transposed_c_epilogue =
        use_whole_k_output_pipeline && ame_transposed_c_epilogue_enabled();
    const uint64_t sparse_total_tiles =
        (uint64_t) ((M + tile_m - 1) / tile_m) *
        (uint64_t) ((N + tile_n - 1) / tile_n) *
        (uint64_t) nb64;
    const uint64_t sparse_op = sparse_progress ? ++g_ame_sparse_op_sequence : 0;
    uint64_t sparse_tile = 0;
    const size_t tile_b_size = (size_t) tile_n * tile_k * sizeof(int8_t);
    const size_t packed_b_panel_size = (size_t) nb64 * tile_b_size;
    const int prof = ame_profile_enabled();
    struct ame_profile_counters prof_local = {0};
    uint64_t prof_total_start = 0;
    if (prof) {
        prof_total_start = ame_read_cycle();
        prof_local.packed_calls = 1;
        prof_local.logical_macs = (uint64_t) M * (uint64_t) N * (uint64_t) K;
    }

    const block_q8_ame64 * restrict x = (const block_q8_ame64 *) src0;
    float * restrict out = (float *) dst;

    const size_t tile_c_bytes = (size_t) tile_m * tile_n * sizeof(int32_t);
    const int async_c_slots = use_whole_k_output_pipeline ? 2 : (async_batch > 1 ? async_batch : 1);
    const size_t required_wsize =
        ggml_ame_i8_kernel_workspace_size(kernel, N, K, use_packed_b_panel) +
        (size_t) (async_c_slots - 1) * tile_c_bytes;
    uint8_t * workspace = (uint8_t *) work_data;
    int allocated_workspace = 0;
    AME_PANEL_PROGRESS_LOG(
        "ame64 start: kernel=%s M=%lld N=%lld K=%lld nb64=%lld use_packed_b_panel=%d whole_k_kloop=%d fused_epilogue=%d output_pipeline=%d acc_pipeline=%d transposed_c=%d async_batch=%d requested=%d reg_pairs=%d work_data=%p work_size=%zu",
        kernel->name, (long long) M, (long long) N, (long long) K, (long long) nb64,
        use_packed_b_panel, use_whole_k_kloop, use_whole_k_fused_epilogue, use_whole_k_output_pipeline,
        use_whole_k_acc_pipeline, use_transposed_c_epilogue, async_batch, async_batch_requested,
        async_reg_pairs, work_data, work_size);

    if (workspace == NULL || work_size < required_wsize) {
        AME_PANEL_PROGRESS_LOG(
            "ame64 workspace_alloc_begin: required=%zu current=%zu",
            required_wsize, work_size);
        workspace = (uint8_t *) ggml_aligned_malloc(required_wsize);
        if (!workspace) {
            AME_PANEL_PROGRESS_LOG("ame64 workspace_alloc_failed: required=%zu", required_wsize);
            return;
        }
        work_size = required_wsize;
        allocated_workspace = 1;
        AME_PANEL_PROGRESS_LOG(
            "ame64 workspace_alloc_done: workspace=%p size=%zu",
            (void *) workspace, work_size);
    }

    uintptr_t ws_ptr = (uintptr_t) workspace;
    uintptr_t ws_end = ws_ptr + work_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_a = (int8_t *) ws_ptr;
    ws_ptr += (size_t) tile_m * tile_k * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_b = (int8_t *) ws_ptr;
    ws_ptr += use_packed_b_panel ? packed_b_panel_size : tile_b_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int32_t * tile_c = (int32_t *) ws_ptr;
    ws_ptr += (size_t) async_c_slots * tile_c_bytes;

    if (ws_ptr > ws_end) {
        AME_PANEL_PROGRESS_LOG(
            "ame64 workspace_layout_overflow: ws_ptr=%p ws_end=%p",
            (void *) ws_ptr, (void *) ws_end);
        if (allocated_workspace) {
            ggml_aligned_free(workspace, work_size);
        }
        return;
    }

    AME_PANEL_PROGRESS_LOG(
        "ame64 workspace_ready: tile_a=%p tile_b=%p tile_c=%p tile_m=%d tile_n=%d tile_k=%d",
        (void *) tile_a, (void *) tile_b, (void *) tile_c, tile_m, tile_n, tile_k);
    AME_PANEL_PROGRESS_LOG("ame64 assert_phys_begin");
    ame_assert_phys_contiguous();
    AME_PANEL_PROGRESS_LOG("ame64 assert_phys_done");
    AME_PANEL_PROGRESS_LOG("ame64 sync_begin_op_begin");
    ggml_ame_sync_begin_op();
    AME_PANEL_PROGRESS_LOG("ame64 sync_begin_op_done");
    if (ame_ckpt_shape_matches(M, N, K)) {
        // Checkpoint-only compatibility for emulators predating the Matrix
        // exception-address fix: preserve dirty PTEs for every MSC slot.
        memset(tile_c, 0, (size_t) async_c_slots * tile_c_bytes);
#if defined(__riscv)
        asm volatile("fence rw, rw" ::: "memory");
#endif
    }
    // The checkpoint format does not preserve in-flight AME sync state.  Reset
    // sync0 before taking the ROI checkpoint so restore resumes from a clean
    // token epoch instead of blocking in msyncregreset.
    ame_ckpt_notify_start(M, N, K, 0);
    if (sparse_progress) {
        fprintf(stderr,
                "[AME_SPARSE] op=%llu phase=begin tiles=%llu interval=%lld range=%llu..%llu "
                "M=%lld N=%lld K=%lld packed_b=%d kernel=%s\n",
                (unsigned long long) sparse_op,
                (unsigned long long) sparse_total_tiles,
                (long long) sparse_interval,
                (unsigned long long) ame_sparse_progress_tile_begin(),
                (unsigned long long) ame_sparse_progress_tile_end(sparse_total_tiles),
                (long long) M,
                (long long) N,
                (long long) K,
                use_packed_b_panel,
                kernel->name);
        fflush(stderr);
    }
    AME_PANEL_PROGRESS_LOG("ame64 clear_tiles_begin");
    memset(tile_a, 0, (size_t) tile_m * tile_k * sizeof(int8_t));
    if (!use_packed_b_panel) {
        memset(tile_b, 0, tile_b_size);
    }
    AME_PANEL_PROGRESS_LOG("ame64 clear_tiles_done");

    uint64_t prof_t0 = prof ? ame_read_cycle() : 0;
    const float * xq_scales = NULL;
    AME_PANEL_PROGRESS_LOG(
        "ame64 prepare_cache_begin: src1_key=%p src1=%p graph_id=%d graph_key=%p generation=%llu",
        src1_key, src1, graph_id, graph_key, (unsigned long long) src1_generation);
    const block_q8_ame64 * xq = ame_prepare_x_q64_cache(
        src1_key,
        src1,
        K,
        N,
        src1_stride,
        graph_id,
        graph_key,
        src1_generation,
        &xq_scales,
        prof ? &prof_local : NULL);
    AME_PANEL_PROGRESS_LOG("ame64 prepare_cache_primary_done: xq=%p scales=%p", (const void *) xq, (const void *) xq_scales);
    if (xq == NULL) {
        // no graph-local cache id available; fall back to one-shot quantization by using
        // a temporary cache entry keyed to this call only
        AME_PANEL_PROGRESS_LOG("ame64 prepare_cache_fallback_begin");
        xq = ame_prepare_x_q64_cache(
            src1,
            src1,
            K,
            N,
            src1_stride,
            1,
            NULL,
            ++g_ame_x_q64_oneshot_generation,
            &xq_scales,
            prof ? &prof_local : NULL);
        if (xq == NULL) {
            AME_PANEL_PROGRESS_LOG("ame64 prepare_cache_failed");
            if (allocated_workspace) {
                ggml_aligned_free(workspace, work_size);
            }
            return;
        }
        g_ame_x_q64_cache.key = NULL;
        g_ame_x_q64_cache.graph_key = NULL;
        g_ame_x_q64_cache.graph_id = 0;
        g_ame_x_q64_cache.generation = 0;
        AME_PANEL_PROGRESS_LOG("ame64 prepare_cache_fallback_done: xq=%p scales=%p", (const void *) xq, (const void *) xq_scales);
    }
    if (prof) prof_local.cycles_prepare_cache += ame_read_cycle() - prof_t0;
    AME_PANEL_PROGRESS_LOG("ame64 prepare_cache_done");

    if (use_packed_b_panel) {
        int8_t * packed_b_panel = tile_b;
        AME_PANEL_LOG("ame64 packed_b_panel: kernel=%s M=%lld N=%lld K=%lld nb64=%lld panel_bytes=%zu",
            kernel->name, (long long) M, (long long) N, (long long) K, (long long) nb64, packed_b_panel_size);

        for (int64_t j0 = 0; j0 < N; j0 += tile_n) {
            const int jmax = (j0 + tile_n <= N) ? tile_n : (N - j0);

            prof_t0 = prof ? ame_read_cycle() : 0;
            memset(packed_b_panel, 0, packed_b_panel_size);
            for (int64_t kb = 0; kb < nb64; ++kb) {
                int8_t * panel_tile_b = packed_b_panel + kb * tile_b_size;
                for (int j = 0; j < jmax; ++j) {
                    const block_q8_ame64 * bx = &xq[(j0 + j) * nb64 + kb];
                    memcpy(&panel_tile_b[j * tile_k], bx->qs, AME_Q8_PACK_K);
                }
            }
            if (prof) prof_local.cycles_pack_b += ame_read_cycle() - prof_t0;
            AME_PANEL_LOG("ame64 packed_b_panel: j0=%lld jmax=%d packed", (long long) j0, jmax);

            if (use_whole_k_output_pipeline) {
                const size_t tile_bytes = (size_t) tile_m * tile_k * sizeof(int8_t);

                if (use_whole_k_acc_pipeline) {
                    float y_scales[AME_TILE_N_MAX];
                    const int output_tiles = (int) ((M + tile_m - 1) / tile_m);
                    for (int j = 0; j < jmax; ++j) {
                        const block_q8_ame64 * bx = &xq[(j0 + j) * nb64];
                        y_scales[j] = xq_scales != NULL ?
                            xq_scales[(j0 + j) * nb64] : GGML_FP16_TO_FP32(bx->d);
                    }
                    struct ame_whole_k_acc_pipeline_context pipeline_ctx = {
                        .out = out + j0 * M,
                        .c_slots = tile_c,
                        .src0_tile_scales = src0_tile_scales,
                        .y_scales = y_scales,
                        .nb64 = nb64,
                        .M = M,
                        .tile_c_elements = tile_c_bytes / sizeof(int32_t),
                        .tile_m = tile_m,
                        .tile_n = tile_n,
                        .jmax = jmax,
                        .transposed_c = use_transposed_c_epilogue,
                        .prof = prof ? &prof_local : NULL,
                    };

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    ggml_ame_gemm_tile_i8_i32_bT_kloop_double_buffer(
                        src0_tile_a,
                        (ptrdiff_t) ((size_t) nb64 * tile_bytes),
                        (ptrdiff_t) tile_bytes,
                        packed_b_panel,
                        (ptrdiff_t) tile_b_size,
                        (int) nb64,
                        tile_c,
                        tile_c + tile_c_bytes / sizeof(int32_t),
                        output_tiles,
                        use_transposed_c_epilogue,
                        ame_finish_whole_k_acc_pipeline_tile,
                        &pipeline_ctx);
                    if (prof) {
                        prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                        prof_local.tile_calls += (uint64_t) output_tiles * (uint64_t) nb64;
                        prof_local.whole_k_kloop_calls += (uint64_t) output_tiles;
                        prof_local.whole_k_pipeline_calls += (uint64_t) output_tiles;
                        if (jmax < tile_n || K % AME_Q8_PACK_K != 0) {
                            prof_local.edge_tile_calls += (uint64_t) output_tiles * (uint64_t) nb64;
                        } else if (M % tile_m != 0) {
                            prof_local.edge_tile_calls += (uint64_t) nb64;
                        }
                    }
                    sparse_tile += (uint64_t) output_tiles * (uint64_t) nb64;
                    AME_PANEL_LOG("ame64 whole-K acc pipeline: j0=%lld done", (long long) j0);
                    continue;
                }

                int pending_slot = -1;
                int64_t pending_i0 = 0;
                int pending_imax = 0;
                int submitted = 0;

                // All A/B panels are immutable at this point.  One fence is
                // sufficient before the stream.  Both memory slots deliberately
                // reuse acc0.  The overlap helper keeps its stack frame and C
                // address live until macquire, while its callback scales the
                // completed previous slot as the current AME submission drains.
#if defined(__riscv)
                asm volatile("fence rw, rw" ::: "memory");
#endif
                for (int64_t i0 = 0; i0 < M; i0 += tile_m) {
                    const int slot = submitted & 1;
                    const int imax = (i0 + tile_m <= M) ? tile_m : (M - i0);
                    const int64_t mt = i0 / tile_m;
                    const int8_t * packed_a_panel =
                        src0_tile_a + (size_t) mt * (size_t) nb64 * tile_bytes;
                    int32_t * c_slot = tile_c + (size_t) slot * (tile_c_bytes / sizeof(int32_t));
                    struct ame_whole_k_overlap_context overlap_ctx = {0};
                    if (pending_slot >= 0) {
                        overlap_ctx.out = out + j0 * M + pending_i0;
                        overlap_ctx.c = tile_c +
                            (size_t) pending_slot * (tile_c_bytes / sizeof(int32_t));
                        overlap_ctx.src0_tile_scales = src0_tile_scales;
                        overlap_ctx.xq = xq;
                        overlap_ctx.xq_scales = xq_scales;
                        overlap_ctx.mt = pending_i0 / tile_m;
                        overlap_ctx.j0 = j0;
                        overlap_ctx.nb64 = nb64;
                        overlap_ctx.M = M;
                        overlap_ctx.tile_m = tile_m;
                        overlap_ctx.tile_n = tile_n;
                        overlap_ctx.imax = pending_imax;
                        overlap_ctx.jmax = jmax;
                        overlap_ctx.transposed_c = use_transposed_c_epilogue;
                        overlap_ctx.prof = prof ? &prof_local : NULL;
                    }

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    ggml_ame_gemm_tile_i8_i32_bT_kloop_overlap(
                        packed_a_panel, (ptrdiff_t) tile_bytes,
                        packed_b_panel, (ptrdiff_t) tile_b_size,
                        (int) nb64, c_slot,
                        use_transposed_c_epilogue,
                        pending_slot >= 0 ? ame_finish_whole_k_pipeline_tile_callback : NULL,
                        pending_slot >= 0 ? &overlap_ctx : NULL);
                    if (prof) {
                        prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                        prof_local.tile_calls += (uint64_t) nb64;
                        prof_local.whole_k_kloop_calls++;
                        prof_local.whole_k_pipeline_calls++;
                        if (imax < tile_m || jmax < tile_n) {
                            prof_local.edge_tile_calls += (uint64_t) nb64;
                        } else if (K % AME_Q8_PACK_K != 0) {
                            prof_local.edge_tile_calls++;
                        }
                    }
                    sparse_tile += (uint64_t) nb64;
                    if (pending_slot >= 0) {
                        ame_ckpt_rebase_after_tile();
                        if (prof) {
                            prof_local.output_tile_calls++;
                        }
                    }
                    pending_slot = slot;
                    pending_i0 = i0;
                    pending_imax = imax;
                    submitted++;
                }

                if (pending_slot >= 0) {
                    ame_finish_whole_k_pipeline_tile(
                        out + j0 * M + pending_i0,
                        tile_c + (size_t) pending_slot * (tile_c_bytes / sizeof(int32_t)),
                        src0_tile_scales,
                        xq,
                        xq_scales,
                        pending_i0 / tile_m,
                        j0,
                        nb64,
                        M,
                        tile_m,
                        tile_n,
                        pending_imax,
                        jmax,
                        use_transposed_c_epilogue,
                        prof ? &prof_local : NULL);
                    ame_ckpt_rebase_after_tile();
                    if (prof) {
                        prof_local.output_tile_calls++;
                    }
                }
                AME_PANEL_LOG("ame64 whole-K output pipeline: j0=%lld done", (long long) j0);
                continue;
            }

            for (int64_t i0 = 0; i0 < M; i0 += tile_m) {
                const int imax = (i0 + tile_m <= M) ? tile_m : (M - i0);
                const int64_t mt = i0 / tile_m;
                float acc_f32[AME_TILE_M_MAX * AME_TILE_N_MAX];
                if (!use_whole_k_fused_epilogue) {
                    prof_t0 = prof ? ame_read_cycle() : 0;
                    memset(acc_f32, 0, sizeof(acc_f32));
                    if (prof) {
                        prof_local.cycles_accumulator_initialization += ame_read_cycle() - prof_t0;
                    }
                }
                AME_PANEL_PROGRESS_LOG(
                    "ame64 packed_b_panel: M=%lld N=%lld K=%lld j0=%lld i0=%lld begin",
                    (long long) M, (long long) N, (long long) K, (long long) j0, (long long) i0);

                if (use_whole_k_kloop) {
                    float x_scales[AME_TILE_M_MAX];
                    float y_scales[AME_TILE_N_MAX];
                    const size_t tile_bytes = (size_t) tile_m * tile_k * sizeof(int8_t);
                    const int8_t * packed_a_panel =
                        src0_tile_a + (size_t) mt * (size_t) nb64 * tile_bytes;

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    ggml_ame_gemm_tile_i8_i32_bT_kloop(
                        packed_a_panel,
                        (ptrdiff_t) tile_bytes,
                        packed_b_panel,
                        (ptrdiff_t) tile_b_size,
                        (int) nb64,
                        tile_c);
                    ame_ckpt_rebase_after_tile();
                    sparse_tile += (uint64_t) nb64;
                    if (prof) {
                        prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                        prof_local.tile_calls += (uint64_t) nb64;
                        prof_local.whole_k_kloop_calls++;
                        if (imax < tile_m || jmax < tile_n) {
                            prof_local.edge_tile_calls += (uint64_t) nb64;
                        } else if (K % AME_Q8_PACK_K != 0) {
                            prof_local.edge_tile_calls++;
                        }
                    }

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    const float * restrict cached_scales =
                        src0_tile_scales + (size_t) mt * (size_t) nb64 * tile_m;
                    memcpy(x_scales, cached_scales, (size_t) imax * sizeof(float));
                    for (int j = 0; j < jmax; ++j) {
                        const block_q8_ame64 * bx = &xq[(j0 + j) * nb64];
                        y_scales[j] = xq_scales != NULL ?
                            xq_scales[(j0 + j) * nb64] : GGML_FP16_TO_FP32(bx->d);
                    }
                    if (use_whole_k_fused_epilogue) {
                        ame_store_scaled_tile(
                            out + j0 * M + i0,
                            tile_c,
                            x_scales,
                            y_scales,
                            M,
                            tile_n,
                            imax,
                            jmax);
                    } else {
                        ame_accumulate_scaled_tile(
                            acc_f32,
                            tile_c,
                            x_scales,
                            y_scales,
                            tile_n,
                            imax,
                            jmax);
                    }
                    if (prof) {
                        prof_local.cycles_scale += ame_read_cycle() - prof_t0;
                    }
                } else if (async_batch > 1) {
                    unsigned long pending_targets[4] = {0};
                    uint64_t pending_ordinals[4] = {0};
                    uint64_t pending_ckpt[4] = {0};
                    int pending_sparse_log[4] = {0};
                    int32_t * pending_c[4] = {0};
                    const int8_t * pending_a[4] = {0};
                    const int8_t * pending_b[4] = {0};
                    float pending_x_scales[4][AME_TILE_M_MAX] = {{0}};
                    float pending_y_scales[4][AME_TILE_N_MAX] = {{0}};
                    uint64_t batch_start = 0;
                    int pending_count = 0;
                    const size_t tile_bytes = (size_t) tile_m * tile_k * sizeof(int8_t);

                    for (int64_t kb = 0; kb < nb64; ++kb) {
                        if (pending_count == 0 && prof) {
                            batch_start = ame_read_cycle();
                        }
                        const int slot = pending_count++;
                        const int64_t kb_for_slot = kb;
                        if ((kb & 7) == 0 || kb + 1 == nb64) {
                            AME_PANEL_PROGRESS_LOG(
                                "ame64 async: M=%lld N=%lld K=%lld j0=%lld i0=%lld kb=%lld/%lld slot=%d/%d",
                                (long long) M, (long long) N, (long long) K,
                                (long long) j0, (long long) i0, (long long) kb, (long long) nb64,
                                slot, async_batch);
                        }

                        prof_t0 = prof ? ame_read_cycle() : 0;
                        pending_a[slot] =
                            src0_tile_a + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_bytes;
                        if (prof) {
                            prof_local.cycles_pack_a += ame_read_cycle() - prof_t0;
                        }

                        for (int j = 0; j < jmax; ++j) {
                            const block_q8_ame64 * bx = &xq[(j0 + j) * nb64 + kb];
                            pending_y_scales[slot][j] =
                                xq_scales != NULL ? xq_scales[(j0 + j) * nb64 + kb] : GGML_FP16_TO_FP32(bx->d);
                        }
                        const float * restrict cached_scales =
                            src0_tile_scales + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_m;
                        memcpy(pending_x_scales[slot], cached_scales, (size_t) imax * sizeof(float));

                        pending_c[slot] = tile_c + (size_t) slot * (tile_c_bytes / sizeof(int32_t));
                        prof_t0 = prof ? ame_read_cycle() : 0;
                        if (!ame_skip_tile_c_zero()) {
                            memset(pending_c[slot], 0, tile_c_bytes);
                            if (prof) {
                                prof_local.cycles_zero_c += ame_read_cycle() - prof_t0;
                            }
                        }

                        pending_b[slot] = packed_b_panel + kb * tile_b_size;
                        const uint64_t tile_ordinal = sparse_tile++;
                        pending_ordinals[slot] = tile_ordinal;
                        pending_sparse_log[slot] = sparse_progress &&
                            ame_sparse_progress_should_log(tile_ordinal, sparse_total_tiles, sparse_interval);
                        if (pending_sparse_log[slot]) {
                            ame_sparse_progress_log(
                                sparse_op, "before", tile_ordinal, sparse_total_tiles,
                                M, N, K, i0, j0, kb_for_slot);
                        }
                        pending_ckpt[slot] = ame_ckpt_notify_tile_start(
                            M, N, K, tile_ordinal, i0, j0, kb_for_slot);
                        ggml_ame_gemm_tile_i8_i32_bT_submit_fn submit =
                            async_reg_pairs >= 2 && slot == 1 ? kernel->submit_alt : kernel->submit;
                        pending_targets[slot] = submit(
                            pending_a[slot], pending_b[slot], pending_c[slot]);

                        if (pending_count == async_batch || kb + 1 == nb64) {
                            const int batch_count = pending_count;
                            // The token is monotonic, so acquiring the newest
                            // release drains every older tile in this batch.
                            ggml_ame_sync_acquire_wait(pending_targets[batch_count - 1]);
                            if (prof) {
                                prof_local.cycles_ame_tile_call += ame_read_cycle() - batch_start;
                                prof_local.tile_calls += (uint64_t) batch_count;
                                for (int q = 0; q < batch_count; ++q) {
                                    const int64_t qkb = kb - (batch_count - 1 - q);
                                    if (imax < tile_m || jmax < tile_n ||
                                            (qkb + 1) * AME_Q8_PACK_K > K) {
                                        prof_local.edge_tile_calls++;
                                    }
                                }
                            }

                            for (int q = 0; q < batch_count; ++q) {
                                if (pending_ckpt[q] != 0) {
                                    ame_ckpt_notify_tile_stop(
                                        pending_ckpt[q], M, N, K, pending_ordinals[q],
                                        i0, j0, kb - (batch_count - 1 - q));
                                }
                                if (pending_sparse_log[q]) {
                                    ame_sparse_progress_log(
                                        sparse_op, "after", pending_ordinals[q], sparse_total_tiles,
                                        M, N, K, i0, j0, kb - (batch_count - 1 - q));
                                }
                                prof_t0 = prof ? ame_read_cycle() : 0;
                                ame_accumulate_scaled_tile(
                                    acc_f32,
                                    pending_c[q],
                                    pending_x_scales[q],
                                    pending_y_scales[q],
                                    tile_n,
                                    imax,
                                    jmax);
                                if (prof) {
                                    prof_local.cycles_scale += ame_read_cycle() - prof_t0;
                                }
                                ame_ckpt_rebase_after_tile();
                            }
                            pending_count = 0;
                        }
                    }
                } else {
                for (int64_t kb = 0; kb < nb64; ++kb) {
                    float y_scales[AME_TILE_N_MAX];
                    float x_scales[AME_TILE_M_MAX];
                    if ((kb & 7) == 0 || kb + 1 == nb64) {
                        AME_PANEL_PROGRESS_LOG(
                            "ame64 packed_b_panel: M=%lld N=%lld K=%lld j0=%lld i0=%lld kb=%lld/%lld",
                            (long long) M, (long long) N, (long long) K,
                            (long long) j0, (long long) i0, (long long) kb, (long long) nb64);
                    }

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    const int8_t * tile_a_for_gemm = tile_a;
                    if (use_packed_a_tiles) {
                        const size_t tile_bytes = (size_t) tile_m * tile_k * sizeof(int8_t);
                        tile_a_for_gemm =
                            src0_tile_a + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_bytes;
                    } else {
                        for (int i = 0; i < tile_m; ++i) {
                            if (i < imax) {
                                const block_q8_ame64 * b = &x[(i0 + i) * nb64 + kb];
                                memcpy(&tile_a[i * tile_k], b->qs, AME_Q8_PACK_K);
                            } else {
                                memset(&tile_a[i * tile_k], 0, AME_Q8_PACK_K);
                            }
                        }
                    }
                    if (prof) prof_local.cycles_pack_a += ame_read_cycle() - prof_t0;

                    for (int j = 0; j < jmax; ++j) {
                        const block_q8_ame64 * bx = &xq[(j0 + j) * nb64 + kb];
                        y_scales[j] = xq_scales != NULL ? xq_scales[(j0 + j) * nb64 + kb] : GGML_FP16_TO_FP32(bx->d);
                    }

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    if (!ame_skip_tile_c_zero()) {
                        memset(tile_c, 0, (size_t) tile_m * tile_n * sizeof(int32_t));
                        if (prof) prof_local.cycles_zero_c += ame_read_cycle() - prof_t0;
                    }

                    const int8_t * panel_tile_b = packed_b_panel + kb * tile_b_size;
                    prof_t0 = prof ? ame_read_cycle() : 0;
                    const uint64_t tile_ordinal = sparse_tile++;
                    const int log_sparse_tile = sparse_progress &&
                        ame_sparse_progress_should_log(tile_ordinal, sparse_total_tiles, sparse_interval);
                    if (log_sparse_tile) {
                        ame_sparse_progress_log(
                            sparse_op, "before", tile_ordinal, sparse_total_tiles,
                            M, N, K, i0, j0, kb);
                    }
                    const uint64_t ckpt_tile_occurrence =
                        ame_ckpt_notify_tile_start(M, N, K, tile_ordinal, i0, j0, kb);
                    kernel->gemm(tile_a_for_gemm, panel_tile_b, tile_c);
                    ame_ckpt_notify_tile_stop(
                        ckpt_tile_occurrence, M, N, K, tile_ordinal, i0, j0, kb);
                    if (log_sparse_tile) {
                        ame_sparse_progress_log(
                            sparse_op, "after", tile_ordinal, sparse_total_tiles,
                            M, N, K, i0, j0, kb);
                    }
                    ame_ckpt_rebase_after_tile();
                    if (prof) {
                        prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                        prof_local.tile_calls++;
                        if (imax < tile_m || jmax < tile_n || (kb + 1) * AME_Q8_PACK_K > K) {
                            prof_local.edge_tile_calls++;
                        }
                    }

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    if (use_packed_a_tiles) {
                        const float * restrict cached_scales =
                            src0_tile_scales + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_m;
                        memcpy(x_scales, cached_scales, (size_t) imax * sizeof(float));
                    } else {
                        for (int i = 0; i < imax; ++i) {
                            const block_q8_ame64 * bx = &x[(i0 + i) * nb64 + kb];
                            x_scales[i] = GGML_FP16_TO_FP32(bx->d);
                        }
                    }
                    ame_accumulate_scaled_tile(
                        acc_f32,
                        tile_c,
                        x_scales,
                        y_scales,
                        tile_n,
                        imax,
                        jmax);
                    if (prof) prof_local.cycles_scale += ame_read_cycle() - prof_t0;
                }
                }

                if (!use_whole_k_fused_epilogue) {
                    prof_t0 = prof ? ame_read_cycle() : 0;
                    ame_store_acc_tile(
                        out + j0 * M + i0,
                        acc_f32,
                        M,
                        tile_n,
                        imax,
                        jmax);
                }
                if (prof) {
                    if (!use_whole_k_fused_epilogue) {
                        prof_local.cycles_store += ame_read_cycle() - prof_t0;
                    }
                    prof_local.output_tile_calls++;
                }
                AME_PANEL_PROGRESS_LOG(
                    "ame64 packed_b_panel: M=%lld N=%lld K=%lld j0=%lld i0=%lld done",
                    (long long) M, (long long) N, (long long) K, (long long) j0, (long long) i0);
            }
            AME_PANEL_LOG("ame64 packed_b_panel: j0=%lld done", (long long) j0);
        }
        if (sparse_progress) {
            fprintf(stderr,
                    "[AME_SPARSE] op=%llu phase=end completed_tiles=%llu/%llu target=%lu\n",
                    (unsigned long long) sparse_op,
                    (unsigned long long) sparse_tile,
                    (unsigned long long) sparse_total_tiles,
                    ggml_ame_sync_state_global.sync0_release_target);
            fflush(stderr);
        }
        ame_ckpt_notify_stop(M, N, K);

        if (prof) {
            prof_local.cycles_packed_total = ame_read_cycle() - prof_total_start;
            ame_profile_accumulate(&prof_local);
        }

        if (allocated_workspace) {
            ggml_aligned_free(workspace, work_size);
        }

        GGML_UNUSED(ne10);
        return;
    }

    for (int64_t i0 = 0; i0 < M; i0 += tile_m) {
        const int imax = (i0 + tile_m <= M) ? tile_m : (M - i0);

        for (int64_t j0 = 0; j0 < N; j0 += tile_n) {
            const int jmax = (j0 + tile_n <= N) ? tile_n : (N - j0);
            float acc_f32[AME_TILE_M_MAX * AME_TILE_N_MAX];
            prof_t0 = prof ? ame_read_cycle() : 0;
            memset(acc_f32, 0, sizeof(acc_f32));
            if (prof) {
                prof_local.cycles_accumulator_initialization += ame_read_cycle() - prof_t0;
            }
            AME_PANEL_PROGRESS_LOG(
                "ame64 tile_output_begin: M=%lld N=%lld K=%lld i0=%lld j0=%lld imax=%d jmax=%d",
                (long long) M, (long long) N, (long long) K,
                (long long) i0, (long long) j0, imax, jmax);

            for (int64_t kb = 0; kb < nb64; ++kb) {
                float y_scales[AME_TILE_N_MAX];
                float x_scales[AME_TILE_M_MAX];

                AME_PANEL_PROGRESS_LOG(
                    "ame64 tile_k_begin: M=%lld N=%lld K=%lld i0=%lld j0=%lld kb=%lld/%lld",
                    (long long) M, (long long) N, (long long) K,
                    (long long) i0, (long long) j0, (long long) kb, (long long) nb64);

                prof_t0 = prof ? ame_read_cycle() : 0;
                for (int i = 0; i < tile_m; ++i) {
                    if (i < imax) {
                        const block_q8_ame64 * b = &x[(i0 + i) * nb64 + kb];
                        memcpy(&tile_a[i * tile_k], b->qs, AME_Q8_PACK_K);
                    } else {
                        memset(&tile_a[i * tile_k], 0, AME_Q8_PACK_K);
                    }
                }
                if (prof) prof_local.cycles_pack_a += ame_read_cycle() - prof_t0;
                AME_PANEL_PROGRESS_LOG(
                    "ame64 pack_a_done: i0=%lld j0=%lld kb=%lld",
                    (long long) i0, (long long) j0, (long long) kb);

                prof_t0 = prof ? ame_read_cycle() : 0;
                if (!ame_skip_tile_b_zero()) {
                    memset(tile_b, 0, (size_t) tile_n * tile_k * sizeof(int8_t));
                }
                for (int j = 0; j < jmax; ++j) {
                    const block_q8_ame64 * bx = &xq[(j0 + j) * nb64 + kb];
                    y_scales[j] = xq_scales != NULL ? xq_scales[(j0 + j) * nb64 + kb] : GGML_FP16_TO_FP32(bx->d);
                    memcpy(&tile_b[j * tile_k], bx->qs, AME_Q8_PACK_K);
                }
                if (prof) prof_local.cycles_pack_b += ame_read_cycle() - prof_t0;
                AME_PANEL_PROGRESS_LOG(
                    "ame64 pack_b_done: i0=%lld j0=%lld kb=%lld",
                    (long long) i0, (long long) j0, (long long) kb);

                prof_t0 = prof ? ame_read_cycle() : 0;
                if (!ame_skip_tile_c_zero()) {
                    memset(tile_c, 0, (size_t) tile_m * tile_n * sizeof(int32_t));
                    if (prof) prof_local.cycles_zero_c += ame_read_cycle() - prof_t0;
                }
                AME_PANEL_PROGRESS_LOG(
                    "ame64 before_gemm: i0=%lld j0=%lld kb=%lld tile_a=%p tile_b=%p tile_c=%p",
                    (long long) i0, (long long) j0, (long long) kb,
                    (const void *) tile_a, (const void *) tile_b, (void *) tile_c);

                prof_t0 = prof ? ame_read_cycle() : 0;
                const uint64_t tile_ordinal = sparse_tile++;
                const int log_sparse_tile = sparse_progress &&
                    ame_sparse_progress_should_log(tile_ordinal, sparse_total_tiles, sparse_interval);
                if (log_sparse_tile) {
                    ame_sparse_progress_log(
                        sparse_op, "before", tile_ordinal, sparse_total_tiles,
                        M, N, K, i0, j0, kb);
                }
                const uint64_t ckpt_tile_occurrence =
                    ame_ckpt_notify_tile_start(M, N, K, tile_ordinal, i0, j0, kb);
                kernel->gemm(tile_a, tile_b, tile_c);
                ame_ckpt_notify_tile_stop(
                    ckpt_tile_occurrence, M, N, K, tile_ordinal, i0, j0, kb);
                if (log_sparse_tile) {
                    ame_sparse_progress_log(
                        sparse_op, "after", tile_ordinal, sparse_total_tiles,
                        M, N, K, i0, j0, kb);
                }
                ame_ckpt_rebase_after_tile();
                AME_PANEL_PROGRESS_LOG(
                    "ame64 after_gemm: i0=%lld j0=%lld kb=%lld",
                    (long long) i0, (long long) j0, (long long) kb);
                if (prof) {
                    prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                    prof_local.tile_calls++;
                    if (imax < tile_m || jmax < tile_n || (kb + 1) * AME_Q8_PACK_K > K) {
                        prof_local.edge_tile_calls++;
                    }
                }

                prof_t0 = prof ? ame_read_cycle() : 0;
                for (int i = 0; i < imax; ++i) {
                    const block_q8_ame64 * bx = &x[(i0 + i) * nb64 + kb];
                    x_scales[i] = GGML_FP16_TO_FP32(bx->d);
                }
                ame_accumulate_scaled_tile(
                    acc_f32,
                    tile_c,
                    x_scales,
                    y_scales,
                    tile_n,
                    imax,
                    jmax);
                if (prof) prof_local.cycles_scale += ame_read_cycle() - prof_t0;
                AME_PANEL_PROGRESS_LOG(
                    "ame64 scale_done: i0=%lld j0=%lld kb=%lld",
                    (long long) i0, (long long) j0, (long long) kb);
            }

            prof_t0 = prof ? ame_read_cycle() : 0;
            ame_store_acc_tile(
                out + j0 * M + i0,
                acc_f32,
                M,
                tile_n,
                imax,
                jmax);
            if (prof) {
                prof_local.cycles_store += ame_read_cycle() - prof_t0;
                prof_local.output_tile_calls++;
            }
            AME_PANEL_PROGRESS_LOG(
                "ame64 tile_output_done: M=%lld N=%lld K=%lld i0=%lld j0=%lld",
                (long long) M, (long long) N, (long long) K,
                (long long) i0, (long long) j0);
        }
    }

    if (sparse_progress) {
        fprintf(stderr,
                "[AME_SPARSE] op=%llu phase=end completed_tiles=%llu/%llu target=%lu\n",
                (unsigned long long) sparse_op,
                (unsigned long long) sparse_tile,
                (unsigned long long) sparse_total_tiles,
                ggml_ame_sync_state_global.sync0_release_target);
        fflush(stderr);
    }

    ame_ckpt_notify_stop(M, N, K);

    if (prof) {
        prof_local.cycles_packed_total = ame_read_cycle() - prof_total_start;
        ame_profile_accumulate(&prof_local);
    }

    if (allocated_workspace) {
        ggml_aligned_free(workspace, work_size);
    }

    GGML_UNUSED(ne10);
}

static const ggml_bf16_t * ame_get_bf16_col_ptr(
    const void * src1,
    int64_t col,
    int64_t K,
    size_t src1_stride,
    enum ggml_type src1_type,
    const ggml_bf16_t * converted_src1
) {
    if (src1_type == GGML_TYPE_BF16) {
        return (const ggml_bf16_t *) ((const char *) src1 + col * src1_stride);
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(src1_stride);
    return converted_src1 + col * K;
}

void ggml_ame_mul_mat_bf16(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride,
    enum ggml_type src1_type,
    void * work_data,
    size_t work_size
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    const int64_t K_AME = (K / AME_TILE_K_BF16) * AME_TILE_K_BF16;

    const ggml_bf16_t * restrict a = (const ggml_bf16_t *) src0;
    float * restrict out = (float *) dst;

    ggml_bf16_t * converted_src1 = NULL;
    int src1_allocated = 0;

    const size_t tile_a_size = AME_TILE_M * AME_TILE_K_BF16 * sizeof(ggml_bf16_t);
    const size_t tile_b_size = AME_TILE_N * AME_TILE_K_BF16 * sizeof(ggml_bf16_t);
    const size_t tile_c_size = AME_TILE_M * AME_TILE_N * sizeof(float);
    const size_t acc_tile_size = AME_TILE_M * AME_TILE_N * sizeof(float);

    const size_t k_tiles = (size_t) (K_AME / AME_TILE_K_BF16);
    int use_whole_k = ame_whole_k_bf16_enabled() && K_AME == K && k_tiles > 0;
    if (use_whole_k &&
        (k_tiles > INT_MAX || k_tiles > SIZE_MAX / tile_a_size || k_tiles > SIZE_MAX / tile_b_size)) {
        use_whole_k = 0;
    }
    const size_t tile_a_alloc_size = use_whole_k ? k_tiles * tile_a_size : tile_a_size;
    const size_t tile_b_alloc_size = use_whole_k ? k_tiles * tile_b_size : tile_b_size;

    ggml_bf16_t * tile_a = (ggml_bf16_t *) ggml_aligned_malloc(tile_a_alloc_size);
    ggml_bf16_t * tile_b = (ggml_bf16_t *) ggml_aligned_malloc(tile_b_alloc_size);
    float * tile_c = (float *) ggml_aligned_malloc(tile_c_size);
    float * acc_tile = use_whole_k ? NULL : (float *) malloc(acc_tile_size);

    if (tile_a == NULL || tile_b == NULL || tile_c == NULL || (!use_whole_k && acc_tile == NULL)) {
        goto cleanup;
    }

    if (src1_type == GGML_TYPE_F32) {
        const size_t converted_size = (size_t) N * (size_t) K * sizeof(ggml_bf16_t);
        converted_src1 = (ggml_bf16_t *) ggml_aligned_malloc(converted_size);
        if (converted_src1 == NULL) {
            goto cleanup;
        }
        src1_allocated = 1;

        for (int64_t j = 0; j < N; ++j) {
            const float * src1_col = (const float *) ((const char *) src1 + j * src1_stride);
            ggml_cpu_fp32_to_bf16(src1_col, converted_src1 + j * K, K);
        }
    } else {
        GGML_ASSERT(src1_type == GGML_TYPE_BF16);
    }

    if (K_AME > 0) {
        ame_assert_phys_contiguous();
        ggml_ame_sync_begin_op();
    }

    for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
        const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (int) (N - j0);

        for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
            const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (int) (M - i0);

            if (imax != AME_TILE_M || jmax != AME_TILE_N || K_AME == 0) {
                for (int i = 0; i < imax; ++i) {
                    const ggml_bf16_t * row_a = a + (i0 + i) * K;
                    for (int j = 0; j < jmax; ++j) {
                        const ggml_bf16_t * col_b = ame_get_bf16_col_ptr(
                            src1, j0 + j, K, src1_stride, src1_type, converted_src1);
                        out[(j0 + j) * M + (i0 + i)] = ame_vec_dot_bf16_fallback(K, row_a, col_b);
                    }
                }
                continue;
            }

            if (use_whole_k) {
                for (size_t kb = 0; kb < k_tiles; ++kb) {
                    const int64_t k0 = (int64_t) kb * AME_TILE_K_BF16;
                    ggml_bf16_t * panel_a = tile_a + kb * AME_TILE_M * AME_TILE_K_BF16;
                    ggml_bf16_t * panel_b = tile_b + kb * AME_TILE_N * AME_TILE_K_BF16;

                    for (int i = 0; i < AME_TILE_M; ++i) {
                        memcpy(panel_a + i * AME_TILE_K_BF16,
                            a + (i0 + i) * K + k0,
                            AME_TILE_K_BF16 * sizeof(ggml_bf16_t));
                    }
                    for (int j = 0; j < AME_TILE_N; ++j) {
                        const ggml_bf16_t * col_b = ame_get_bf16_col_ptr(
                            src1, j0 + j, K, src1_stride, src1_type, converted_src1);
                        memcpy(panel_b + j * AME_TILE_K_BF16,
                            col_b + k0,
                            AME_TILE_K_BF16 * sizeof(ggml_bf16_t));
                    }
                }

                ggml_ame_gemm_tile_bf16_fp32_bT_kloop(
                    tile_a, (ptrdiff_t) tile_a_size,
                    tile_b, (ptrdiff_t) tile_b_size,
                    (int) k_tiles, ame_bf16_register_pairs(), tile_c);

                for (int i = 0; i < AME_TILE_M; ++i) {
                    for (int j = 0; j < AME_TILE_N; ++j) {
                        out[(j0 + j) * M + (i0 + i)] = tile_c[i * AME_TILE_N + j];
                    }
                }
                continue;
            }

            memset(acc_tile, 0, acc_tile_size);

            for (int64_t k0 = 0; k0 < K_AME; k0 += AME_TILE_K_BF16) {
                for (int i = 0; i < AME_TILE_M; ++i) {
                    memcpy(tile_a + i * AME_TILE_K_BF16,
                        a + (i0 + i) * K + k0,
                        AME_TILE_K_BF16 * sizeof(ggml_bf16_t));
                }

                for (int j = 0; j < AME_TILE_N; ++j) {
                    const ggml_bf16_t * col_b = ame_get_bf16_col_ptr(
                        src1, j0 + j, K, src1_stride, src1_type, converted_src1);
                    memcpy(tile_b + j * AME_TILE_K_BF16,
                        col_b + k0,
                        AME_TILE_K_BF16 * sizeof(ggml_bf16_t));
                }

                memset(tile_c, 0, tile_c_size);
                ggml_ame_gemm_tile_bf16_fp32_bT(tile_a, tile_b, tile_c);

                for (int i = 0; i < AME_TILE_M; ++i) {
                    for (int j = 0; j < AME_TILE_N; ++j) {
                        acc_tile[i * AME_TILE_N + j] += tile_c[i * AME_TILE_N + j];
                    }
                }
            }

            if (K_AME < K) {
                const int tail = (int) (K - K_AME);
                for (int i = 0; i < AME_TILE_M; ++i) {
                    const ggml_bf16_t * row_a = a + (i0 + i) * K + K_AME;
                    for (int j = 0; j < AME_TILE_N; ++j) {
                        const ggml_bf16_t * col_b = ame_get_bf16_col_ptr(
                            src1, j0 + j, K, src1_stride, src1_type, converted_src1) + K_AME;
                        acc_tile[i * AME_TILE_N + j] += ame_vec_dot_bf16_fallback(tail, row_a, col_b);
                    }
                }
            }

            for (int i = 0; i < AME_TILE_M; ++i) {
                for (int j = 0; j < AME_TILE_N; ++j) {
                    out[(j0 + j) * M + (i0 + i)] = acc_tile[i * AME_TILE_N + j];
                }
            }
        }
    }

cleanup:
    if (tile_a != NULL) {
        ggml_aligned_free(tile_a, tile_a_alloc_size);
    }
    if (tile_b != NULL) {
        ggml_aligned_free(tile_b, tile_b_alloc_size);
    }
    if (tile_c != NULL) {
        ggml_aligned_free(tile_c, tile_c_size);
    }
    free(acc_tile);
    if (src1_allocated) {
        ggml_aligned_free(converted_src1, (size_t) N * (size_t) K * sizeof(ggml_bf16_t));
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(work_data);
    GGML_UNUSED(work_size);
}

// Wrapper for AME-accelerated Q4_0 GEMM
// Assumes REPACKED block_q4_0_ame inputs for src0
void ggml_ame_mul_mat_q4_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    const int qk = 32;
    const int64_t nb_x = K / qk;

    // Treat src0 as REPACKED block_q4_0_ame
    // Note: The backend repacks Q4_0 to this format automatically
    // Layout matches block_q8_0 (d: f16, qs: i8[32]), so we can reuse Q8_0 kernels/logic
    const block_q4_0_ame * restrict x = (const block_q4_0_ame *)src0;
    float * restrict out = (float *)dst;

    const int64_t y_q8_size = N * nb_x;
    block_q8_0 * y_q8 = (block_q8_0 *)malloc(y_q8_size * sizeof(block_q8_0));
    if (!y_q8) return;

    for (int64_t j = 0; j < N; j++) {
        const float * src1_col = (const float *)((const char *)src1 + j * src1_stride);
        ggml_ame_quantize_row_f32_to_q8_0(src1_col, y_q8 + j * nb_x, K);
    }
    const block_q8_0 * restrict y = y_q8;

    int8_t * tile_a = (int8_t *)ggml_aligned_malloc(AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    int8_t * tile_b = (int8_t *)ggml_aligned_malloc(AME_TILE_N * AME_TILE_K * sizeof(int8_t));
    int32_t * tile_c = (int32_t *)ggml_aligned_malloc(AME_TILE_M * AME_TILE_N * sizeof(int32_t));
    if (!tile_a || !tile_b || !tile_c) {
        ggml_aligned_free(tile_a, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
        ggml_aligned_free(tile_b, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
        ggml_aligned_free(tile_c, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
        free(y_q8);
        return;
    }
    // Method A: verify physical contiguity contract before first AME instruction
    ame_assert_phys_contiguous();
    ggml_ame_sync_begin_op();

    for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
        const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);

        for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
            const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (N - j0);

#if defined(__riscv_v)
            if (imax < AME_TILE_M || jmax < AME_TILE_N) {
                 for (int i = 0; i < imax; ++i) {
                     for (int j = 0; j < jmax; ++j) {
                         const block_q4_0_ame * row_x = &x[(i0 + i) * nb_x];
                         const block_q8_0 * col_y = &y[(j0 + j) * nb_x];
                         // Repacked Q4_0 layout is compatible with Q8_0 for dot product
                         ame_vec_dot_q8_0_rvv(K, &out[(j0 + j) * M + (i0 + i)], (const void*)row_x, col_y);
                     }
                 }
                 continue;
            }
#endif

            float acc_f32[AME_TILE_M * AME_TILE_N];
            memset(acc_f32, 0, sizeof(acc_f32));

            for (int64_t kb = 0; kb < nb_x; kb++) {
                
                // Prepare Tile A (16 x 32) - Directly copy repacked data
                for (int i = 0; i < AME_TILE_M; i++) {
                     memset(&tile_a[i * AME_TILE_K], 0, AME_TILE_K);
                     if (i < imax) {
                         const block_q4_0_ame * b = &x[(i0 + i) * nb_x + kb];
                         memcpy(&tile_a[i * AME_TILE_K], b->qs, qk);
                     }
                }

                // Prepare Tile B (16 x 32)
                for (int j = 0; j < AME_TILE_N; j++) {
                     memset(&tile_b[j * AME_TILE_K], 0, AME_TILE_K);
                     if (j < jmax) {
                         const block_q8_0 * b = &y[(j0 + j) * nb_x + kb];
                         memcpy(&tile_b[j * AME_TILE_K], b->qs, qk);
                     }
                }

                memset(tile_c, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                ggml_ame_gemm_tile_i8_i32_bT(tile_a, tile_b, tile_c);

                for (int i = 0; i < imax; i++) {
                     const block_q4_0_ame * bx = &x[(i0 + i) * nb_x + kb];
                     const float d_x = GGML_FP16_TO_FP32(bx->d);
                     for (int j = 0; j < jmax; j++) {
                         const block_q8_0 * by = &y[(j0 + j) * nb_x + kb];
                         const float d_y = GGML_FP16_TO_FP32(by->d);
                         acc_f32[i * AME_TILE_N + j] += tile_c[i * AME_TILE_N + j] * (d_x * d_y);
                     }
                }
            }

            for (int i = 0; i < imax; i++) {
                for (int j = 0; j < jmax; j++) {
                    out[(j0 + j) * M + (i0 + i)] = acc_f32[i * AME_TILE_N + j];
                }
            }
        }
    }

    ggml_aligned_free(tile_a, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    ggml_aligned_free(tile_b, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
    ggml_aligned_free(tile_c, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
    free(y_q8);
}
