#include "ame.h"
#include "common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"

#if defined(GGML_XSAI_ALLOC)
#include "xsai_alloc.h"
#endif

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

struct ame_profile_counters {
    uint64_t baseline_calls;
    uint64_t packed_calls;
    uint64_t logical_macs;
    uint64_t tile_calls;
    uint64_t edge_tile_calls;
    uint64_t output_tile_calls;
    uint64_t cycles_baseline_total;
    uint64_t cycles_packed_total;
    uint64_t cycles_prepare_cache;
    uint64_t cycles_pack_a;
    uint64_t cycles_pack_b;
    uint64_t cycles_zero_c;
    uint64_t cycles_ame_tile_call;
    uint64_t cycles_scale;
    uint64_t cycles_store;
};

static struct ame_profile_counters g_ame_profile;

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

static void ame_ckpt_notify_start(int64_t M, int64_t N, int64_t K, int64_t j0) {
    static int emitted = 0;
    if (emitted || !ame_ckpt_shape_matches(M, N, K)) {
        return;
    }

    emitted = 1;
    fprintf(stderr,
            "[AME_CKPT] start ROI before packed-B accumulation: M=%lld N=%lld K=%lld j0=%lld\n",
            (long long) M, (long long) N, (long long) K, (long long) j0);
    fflush(stderr);
    ame_nemu_signal(0x100);
    ame_nemu_signal(0x101);
}

static void ame_ckpt_notify_stop(int64_t M, int64_t N, int64_t K) {
    static int emitted = 0;
    if (emitted || !ame_ckpt_shape_matches(M, N, K)) {
        return;
    }

    emitted = 1;
    fprintf(stderr,
            "[AME_CKPT] stop ROI after packed-B accumulation: M=%lld N=%lld K=%lld\n",
            (long long) M, (long long) N, (long long) K);
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

    fprintf(stderr,
        "[AME_PROFILE] calls baseline=%llu packed=%llu tiles=%llu edge_tiles=%llu output_tiles=%llu logical_macs=%llu logical_ops=%llu\n",
        (unsigned long long) g_ame_profile.baseline_calls,
        (unsigned long long) g_ame_profile.packed_calls,
        (unsigned long long) g_ame_profile.tile_calls,
        (unsigned long long) g_ame_profile.edge_tile_calls,
        (unsigned long long) g_ame_profile.output_tile_calls,
        (unsigned long long) g_ame_profile.logical_macs,
        (unsigned long long) ops);
    fprintf(stderr,
        "[AME_PROFILE] cycles total=%llu baseline_total=%llu packed_total=%llu prepare_cache=%llu pack_a=%llu pack_b=%llu zero_c=%llu ame_tile_call=%llu scale=%llu store=%llu\n",
        (unsigned long long) cycles_total,
        (unsigned long long) g_ame_profile.cycles_baseline_total,
        (unsigned long long) g_ame_profile.cycles_packed_total,
        (unsigned long long) g_ame_profile.cycles_prepare_cache,
        (unsigned long long) g_ame_profile.cycles_pack_a,
        (unsigned long long) g_ame_profile.cycles_pack_b,
        (unsigned long long) g_ame_profile.cycles_zero_c,
        (unsigned long long) g_ame_profile.cycles_ame_tile_call,
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
    g_ame_profile.edge_tile_calls      += local->edge_tile_calls;
    g_ame_profile.output_tile_calls    += local->output_tile_calls;
    g_ame_profile.cycles_baseline_total += local->cycles_baseline_total;
    g_ame_profile.cycles_packed_total  += local->cycles_packed_total;
    g_ame_profile.cycles_prepare_cache += local->cycles_prepare_cache;
    g_ame_profile.cycles_pack_a        += local->cycles_pack_a;
    g_ame_profile.cycles_pack_b        += local->cycles_pack_b;
    g_ame_profile.cycles_zero_c        += local->cycles_zero_c;
    g_ame_profile.cycles_ame_tile_call += local->cycles_ame_tile_call;
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

// ggml_ame_quantize_row_f32_to_q8_0 is now in ame-helper.c

static size_t ame_align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static size_t ggml_ame_q8_workspace_size(int64_t N, int64_t n_k_tiles) {
    GGML_UNUSED(N);
    GGML_UNUSED(n_k_tiles);

    size_t size = 64;
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_K * sizeof(int8_t);
    size = ame_align_up_size(size, 64);
    size += AME_TILE_N * AME_TILE_K * sizeof(int8_t);
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_N * sizeof(int32_t);
    return size;
}

static size_t ggml_ame_q8_panel_workspace_size(int64_t N, int64_t n_k_tiles) {
    GGML_UNUSED(N);

    const size_t packed_b_panel_size = (size_t) n_k_tiles * AME_TILE_N * AME_TILE_K * sizeof(int8_t);
    size_t size = 64;
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_K * sizeof(int8_t);
    size = ame_align_up_size(size, 64);
    size += packed_b_panel_size;
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_N * sizeof(int32_t);
    return size;
}

static size_t ggml_ame_q8_0_workspace_size(int64_t N, int64_t K) {
    GGML_UNUSED(N);

    size_t size = 64;
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_K * sizeof(int8_t);
    size = ame_align_up_size(size, 64);
    size += AME_TILE_N * AME_TILE_K * sizeof(int8_t);
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_N * sizeof(int32_t);
    return size;
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
    const int imax,
    const int jmax
) {
#if defined(__riscv_v)
    int j = 0;
    while (j < jmax) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (jmax - j));
        const vfloat32m8_t vy = __riscv_vle32_v_f32m8(y_scales + j, vl);

        for (int i = 0; i < imax; ++i) {
            float * restrict acc_row = acc + i * AME_TILE_N + j;
            const int32_t * restrict c_row = c + i * AME_TILE_N + j;

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
            acc[i * AME_TILE_N + j] += c[i * AME_TILE_N + j] * (d_x * y_scales[j]);
        }
    }
#endif
}

static inline void ame_store_acc_tile(
    float * restrict out,
    const float * restrict acc,
    const int64_t M,
    const int imax,
    const int jmax
) {
#if defined(__riscv_v)
    const ptrdiff_t acc_stride = (ptrdiff_t) AME_TILE_N * (ptrdiff_t) sizeof(float);
    for (int j = 0; j < jmax; ++j) {
        int i = 0;
        while (i < imax) {
            const size_t vl = __riscv_vsetvl_e32m8((size_t) (imax - i));
            const vfloat32m8_t v = __riscv_vlse32_v_f32m8(acc + i * AME_TILE_N + j, acc_stride, vl);
            __riscv_vse32_v_f32m8(out + (int64_t) j * M + i, v, vl);
            i += (int) vl;
        }
    }
#else
    for (int j = 0; j < jmax; ++j) {
        for (int i = 0; i < imax; ++i) {
            out[(int64_t) j * M + i] = acc[i * AME_TILE_N + j];
        }
    }
#endif
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
    const float ** scales_out
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

    const size_t required_wsize = ggml_ame_q8_0_workspace_size(N, K);
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
    ws_ptr += AME_TILE_M * AME_TILE_K * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_b = (int8_t *)ws_ptr;
    ws_ptr += AME_TILE_N * AME_TILE_K * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int32_t * tile_c = (int32_t *)ws_ptr;
    ws_ptr += AME_TILE_M * AME_TILE_N * sizeof(int32_t);

    if (ws_ptr > ws_end) {
        if (allocated_workspace) {
            ggml_aligned_free(workspace, work_size);
        }
        return;
    }

    // Method A: verify physical contiguity contract before first AME instruction
    ame_assert_phys_contiguous();
    memset(tile_a, 0, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    memset(tile_b, 0, AME_TILE_N * AME_TILE_K * sizeof(int8_t));

    for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
        const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);

        for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
            const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (N - j0);
            
            // Full AME Path for full tiles
            float acc_f32[AME_TILE_M * AME_TILE_N];
            memset(acc_f32, 0, sizeof(acc_f32));

            for (int64_t kb = 0; kb < nb_x; kb++) {
                float y_scales[AME_TILE_N];
                float x_scales[AME_TILE_M];
                memset(y_scales, 0, sizeof(y_scales));
                memset(x_scales, 0, sizeof(x_scales));

                // Prepare Tile A (16 x 32)
                uint64_t prof_t0 = prof ? ame_read_cycle() : 0;
                for (int i = 0; i < AME_TILE_M; i++) {
                     if (i < imax) {
                         const block_q8_0 * b = &x[(i0 + i) * nb_x + kb];
                         memcpy(&tile_a[i * AME_TILE_K], b->qs, qk);
                     } else {
                         memset(&tile_a[i * AME_TILE_K], 0, qk);
                     }
                }
                if (prof) prof_local.cycles_pack_a += ame_read_cycle() - prof_t0;

                // Prepare Tile B (16 x 32)
                 prof_t0 = prof ? ame_read_cycle() : 0;
                 if (!ame_skip_tile_b_zero()) {
                     memset(tile_b, 0, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
                 }
                 for (int j = 0; j < jmax; j++) {
                     const float * src1_col = (const float *)((const char *)src1 + (j0 + j) * src1_stride);
                     block_q8_0 tmp_block;
                     ggml_ame_quantize_row_f32_to_q8_0(src1_col + kb * qk, &tmp_block, qk);
                     y_scales[j] = GGML_FP16_TO_FP32(tmp_block.d);
                     memcpy(&tile_b[j * AME_TILE_K], tmp_block.qs, qk);
                 }
                if (prof) prof_local.cycles_pack_b += ame_read_cycle() - prof_t0;

                prof_t0 = prof ? ame_read_cycle() : 0;
                if (!ame_skip_tile_c_zero()) {
                    memset(tile_c, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                    if (prof) prof_local.cycles_zero_c += ame_read_cycle() - prof_t0;
                }

                prof_t0 = prof ? ame_read_cycle() : 0;
                ggml_ame_gemm_tile_i8_i32_bT(tile_a, tile_b, tile_c);
                if (prof) {
                    prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                    prof_local.tile_calls++;
                    if (imax < AME_TILE_M || jmax < AME_TILE_N) {
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
                    imax,
                    jmax);
                if (prof) prof_local.cycles_scale += ame_read_cycle() - prof_t0;
            }

            // Copy back
            uint64_t prof_t0 = prof ? ame_read_cycle() : 0;
            ame_store_acc_tile(
                out + j0 * M + i0,
                acc_f32,
                M,
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
    const int use_packed_b_panel = ame_use_packed_b_panel();
    const size_t tile_b_size = AME_TILE_N * AME_TILE_K * sizeof(int8_t);
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

    const size_t required_wsize = use_packed_b_panel ?
        ggml_ame_q8_panel_workspace_size(N, nb64) :
        ggml_ame_q8_workspace_size(N, nb64);
    uint8_t * workspace = (uint8_t *) work_data;
    int allocated_workspace = 0;

    if (workspace == NULL || work_size < required_wsize) {
        workspace = (uint8_t *) ggml_aligned_malloc(required_wsize);
        if (!workspace) return;
        work_size = required_wsize;
        allocated_workspace = 1;
    }

    uintptr_t ws_ptr = (uintptr_t) workspace;
    uintptr_t ws_end = ws_ptr + work_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_a = (int8_t *) ws_ptr;
    ws_ptr += AME_TILE_M * AME_TILE_K * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_b = (int8_t *) ws_ptr;
    ws_ptr += use_packed_b_panel ? packed_b_panel_size : tile_b_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int32_t * tile_c = (int32_t *) ws_ptr;
    ws_ptr += AME_TILE_M * AME_TILE_N * sizeof(int32_t);

    if (ws_ptr > ws_end) {
        if (allocated_workspace) {
            ggml_aligned_free(workspace, work_size);
        }
        return;
    }

    ame_assert_phys_contiguous();
    memset(tile_a, 0, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    if (!use_packed_b_panel) {
        memset(tile_b, 0, tile_b_size);
    }

    uint64_t prof_t0 = prof ? ame_read_cycle() : 0;
    const float * xq_scales = NULL;
    const block_q8_ame64 * xq = ame_prepare_x_q64_cache(
        src1_key,
        src1,
        K,
        N,
        src1_stride,
        graph_id,
        graph_key,
        src1_generation,
        &xq_scales);
    if (xq == NULL) {
        // no graph-local cache id available; fall back to one-shot quantization by using
        // a temporary cache entry keyed to this call only
        xq = ame_prepare_x_q64_cache(
            src1,
            src1,
            K,
            N,
            src1_stride,
            1,
            NULL,
            ++g_ame_x_q64_oneshot_generation,
            &xq_scales);
        if (xq == NULL) {
            if (allocated_workspace) {
                ggml_aligned_free(workspace, work_size);
            }
            return;
        }
        g_ame_x_q64_cache.key = NULL;
        g_ame_x_q64_cache.graph_key = NULL;
        g_ame_x_q64_cache.graph_id = 0;
        g_ame_x_q64_cache.generation = 0;
    }
    if (prof) prof_local.cycles_prepare_cache += ame_read_cycle() - prof_t0;

    const int use_packed_a_tiles = src0_tile_a != NULL && src0_tile_scales != NULL;

    if (use_packed_b_panel) {
        int8_t * packed_b_panel = tile_b;
        AME_PANEL_LOG("ame64 packed_b_panel: M=%lld N=%lld K=%lld nb64=%lld panel_bytes=%zu",
            (long long) M, (long long) N, (long long) K, (long long) nb64, packed_b_panel_size);

        for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
            const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (N - j0);

            prof_t0 = prof ? ame_read_cycle() : 0;
            memset(packed_b_panel, 0, packed_b_panel_size);
            for (int64_t kb = 0; kb < nb64; ++kb) {
                int8_t * panel_tile_b = packed_b_panel + kb * tile_b_size;
                for (int j = 0; j < jmax; ++j) {
                    const block_q8_ame64 * bx = &xq[(j0 + j) * nb64 + kb];
                    memcpy(&panel_tile_b[j * AME_TILE_K], bx->qs, AME_Q8_PACK_K);
                }
            }
            if (prof) prof_local.cycles_pack_b += ame_read_cycle() - prof_t0;
            AME_PANEL_LOG("ame64 packed_b_panel: j0=%lld jmax=%d packed", (long long) j0, jmax);
            ame_ckpt_notify_start(M, N, K, j0);

            for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
                const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);
                const int64_t mt = i0 / AME_TILE_M;
                float acc_f32[AME_TILE_M * AME_TILE_N];
                memset(acc_f32, 0, sizeof(acc_f32));
                AME_PANEL_PROGRESS_LOG(
                    "ame64 packed_b_panel: M=%lld N=%lld K=%lld j0=%lld i0=%lld begin",
                    (long long) M, (long long) N, (long long) K, (long long) j0, (long long) i0);

                for (int64_t kb = 0; kb < nb64; ++kb) {
                    float y_scales[AME_TILE_N];
                    float x_scales[AME_TILE_M];
                    if ((kb & 7) == 0 || kb + 1 == nb64) {
                        AME_PANEL_PROGRESS_LOG(
                            "ame64 packed_b_panel: M=%lld N=%lld K=%lld j0=%lld i0=%lld kb=%lld/%lld",
                            (long long) M, (long long) N, (long long) K,
                            (long long) j0, (long long) i0, (long long) kb, (long long) nb64);
                    }

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    const int8_t * tile_a_for_gemm = tile_a;
                    if (use_packed_a_tiles) {
                        const size_t tile_bytes = AME_TILE_M * AME_TILE_K * sizeof(int8_t);
                        tile_a_for_gemm =
                            src0_tile_a + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_bytes;
                    } else {
                        for (int i = 0; i < AME_TILE_M; ++i) {
                            if (i < imax) {
                                const block_q8_ame64 * b = &x[(i0 + i) * nb64 + kb];
                                memcpy(&tile_a[i * AME_TILE_K], b->qs, AME_Q8_PACK_K);
                            } else {
                                memset(&tile_a[i * AME_TILE_K], 0, AME_Q8_PACK_K);
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
                        memset(tile_c, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                        if (prof) prof_local.cycles_zero_c += ame_read_cycle() - prof_t0;
                    }

                    const int8_t * panel_tile_b = packed_b_panel + kb * tile_b_size;
                    prof_t0 = prof ? ame_read_cycle() : 0;
                    ggml_ame_gemm_tile_i8_i32_bT(tile_a_for_gemm, panel_tile_b, tile_c);
                    if (prof) {
                        prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                        prof_local.tile_calls++;
                        if (imax < AME_TILE_M || jmax < AME_TILE_N || (kb + 1) * AME_Q8_PACK_K > K) {
                            prof_local.edge_tile_calls++;
                        }
                    }

                    prof_t0 = prof ? ame_read_cycle() : 0;
                    if (use_packed_a_tiles) {
                        const float * restrict cached_scales =
                            src0_tile_scales + ((size_t) mt * (size_t) nb64 + (size_t) kb) * AME_TILE_M;
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
                        imax,
                        jmax);
                    if (prof) prof_local.cycles_scale += ame_read_cycle() - prof_t0;
                }

                prof_t0 = prof ? ame_read_cycle() : 0;
                ame_store_acc_tile(
                    out + j0 * M + i0,
                    acc_f32,
                    M,
                    imax,
                    jmax);
                if (prof) {
                    prof_local.cycles_store += ame_read_cycle() - prof_t0;
                    prof_local.output_tile_calls++;
                }
                AME_PANEL_PROGRESS_LOG(
                    "ame64 packed_b_panel: M=%lld N=%lld K=%lld j0=%lld i0=%lld done",
                    (long long) M, (long long) N, (long long) K, (long long) j0, (long long) i0);
            }
            AME_PANEL_LOG("ame64 packed_b_panel: j0=%lld done", (long long) j0);
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

    for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
        const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);

        for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
            const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (N - j0);
            float acc_f32[AME_TILE_M * AME_TILE_N];
            memset(acc_f32, 0, sizeof(acc_f32));

            for (int64_t kb = 0; kb < nb64; ++kb) {
                float y_scales[AME_TILE_N];
                float x_scales[AME_TILE_M];

                prof_t0 = prof ? ame_read_cycle() : 0;
                for (int i = 0; i < AME_TILE_M; ++i) {
                    if (i < imax) {
                        const block_q8_ame64 * b = &x[(i0 + i) * nb64 + kb];
                        memcpy(&tile_a[i * AME_TILE_K], b->qs, AME_Q8_PACK_K);
                    } else {
                        memset(&tile_a[i * AME_TILE_K], 0, AME_Q8_PACK_K);
                    }
                }
                if (prof) prof_local.cycles_pack_a += ame_read_cycle() - prof_t0;

                prof_t0 = prof ? ame_read_cycle() : 0;
                if (!ame_skip_tile_b_zero()) {
                    memset(tile_b, 0, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
                }
                for (int j = 0; j < jmax; ++j) {
                    const block_q8_ame64 * bx = &xq[(j0 + j) * nb64 + kb];
                    y_scales[j] = xq_scales != NULL ? xq_scales[(j0 + j) * nb64 + kb] : GGML_FP16_TO_FP32(bx->d);
                    memcpy(&tile_b[j * AME_TILE_K], bx->qs, AME_Q8_PACK_K);
                }
                if (prof) prof_local.cycles_pack_b += ame_read_cycle() - prof_t0;

                prof_t0 = prof ? ame_read_cycle() : 0;
                if (!ame_skip_tile_c_zero()) {
                    memset(tile_c, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                    if (prof) prof_local.cycles_zero_c += ame_read_cycle() - prof_t0;
                }

                prof_t0 = prof ? ame_read_cycle() : 0;
                ggml_ame_gemm_tile_i8_i32_bT(tile_a, tile_b, tile_c);
                if (prof) {
                    prof_local.cycles_ame_tile_call += ame_read_cycle() - prof_t0;
                    prof_local.tile_calls++;
                    if (imax < AME_TILE_M || jmax < AME_TILE_N || (kb + 1) * AME_Q8_PACK_K > K) {
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
                    imax,
                    jmax);
                if (prof) prof_local.cycles_scale += ame_read_cycle() - prof_t0;
            }

            prof_t0 = prof ? ame_read_cycle() : 0;
            ame_store_acc_tile(
                out + j0 * M + i0,
                acc_f32,
                M,
                imax,
                jmax);
            if (prof) {
                prof_local.cycles_store += ame_read_cycle() - prof_t0;
                prof_local.output_tile_calls++;
            }
        }
    }

    if (prof) {
        prof_local.cycles_packed_total = ame_read_cycle() - prof_total_start;
        ame_profile_accumulate(&prof_local);
    }

    if (allocated_workspace) {
        ggml_aligned_free(workspace, work_size);
    }

    GGML_UNUSED(ne10);
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
