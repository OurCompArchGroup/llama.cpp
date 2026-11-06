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
            "occurrence=%lld roi_cycles=%llu\n",
            (long long) M, (long long) N, (long long) K, (long long) occurrence,
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

/* Q8 GEMM moved into xsai-blas plan execution. BF16/Q4 remain. */


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
