#ifndef GGML_RISCV_AME_H
#define GGML_RISCV_AME_H

#include <stdint.h>
#include <stddef.h>  // for size_t

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"

// Matrix sync registers and scratch/cache state belong to a GGML worker, not
// the process. Use one spelling in both the C kernels and C++ backend.
#ifdef __cplusplus
#define GGML_AME_THREAD_LOCAL thread_local
#else
#define GGML_AME_THREAD_LOCAL _Thread_local
#endif

static inline bool ggml_ame_rmsnorm_qkv_enabled(const char * mode) {
    return mode != NULL && (strcmp(mode, "qkv") == 0 || strcmp(mode, "pipeline+qkv") == 0);
}

// AME debug logging can be forced at build time with AME_DEBUG=1 or enabled at
// runtime with GGML_AME_LOG=1 so logs are visible in QEMU serial output.
#ifndef AME_DEBUG
#define AME_DEBUG 0
#endif

static inline int ame_log_enabled(void) {
#if AME_DEBUG
    return 1;
#else
    static GGML_AME_THREAD_LOCAL int cached = -1;
    if (cached == -1) {
        const char * value = getenv("GGML_AME_LOG");
        cached = 0;
        if (value != NULL && value[0] != '\0' && strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "off") != 0 && strcmp(value, "no") != 0) {
            cached = 1;
        }
    }
    return cached;
#endif
}

#define AME_LOG(fmt, ...)                             \
    do {                                              \
        if (ame_log_enabled()) {                      \
            fprintf(stderr, "[AME] " fmt "\n", ##__VA_ARGS__); \
            fflush(stderr);                           \
        }                                             \
    } while (0)

#define AME_TILE_M 64
#define AME_TILE_K 64
#define AME_TILE_N 64
#define AME_TILE_M_MAX 128
#define AME_TILE_N_MAX 128
#define AME_TILE_K_BF16 (AME_TILE_K / (int) sizeof(ggml_bf16_t))

#define AME_Q8_BLOCK_K 32
#define AME_Q8_PACK_K 64

// Backend-private MX32 panels. Each complete panel and scale stream occupies
// whole cache lines; no masked/tail or noncompact hardware load is required.
#define AME_MXFP4_K 32
#define AME_MXFP4_TILE_BYTES (AME_TILE_M * AME_MXFP4_K / 2)
#define AME_MXFP4_SCALE_BYTES AME_TILE_M
#define AME_MXFP8_TILE_BYTES (AME_TILE_M * AME_MXFP4_K)

// Private software mcfg bytes, NOT internal MteComputeType enum values.
// Frozen spec/program_model.adoc:76,94-105 (table1, low type nibble).
#define AME_MCFG_MXFP4 0x16
#define AME_MCFG_MXFP8_E5M2 0x17
#define AME_MCFG_MXFP8_E4M3 0x18

static inline int ggml_ame_mxfp8_format_supported(int format) {
    return format == AME_MCFG_MXFP8_E4M3 || format == AME_MCFG_MXFP8_E5M2;
}

static inline size_t ggml_ame_mx_block_bytes(int format) {
    return format == AME_MCFG_MXFP4 ? 16 : (ggml_ame_mxfp8_format_supported(format) ? 32 : 0);
}

// Logical M stays the output leading dimension; only packed weight storage
// rounds up to full hardware panels. Reject before signed rounding overflow.
static inline int64_t ggml_ame_mx_padded_rows(int64_t M, int tile_mn) {
    if ((tile_mn != 64 && tile_mn != AME_TILE_M_MAX) ||
        M < tile_mn || M > INT64_MAX - (tile_mn - 1)) {
        return 0;
    }
    return ((M + tile_mn - 1) / tile_mn) * tile_mn;
}

static inline int ggml_ame_mx_shape_supported(int64_t M, int64_t N, int64_t K, int tile_mn) {
    return ggml_ame_mx_padded_rows(M, tile_mn) != 0 &&
        N > 0 && K > 0 && K <= INT32_MAX &&
        N % tile_mn == 0 && K % AME_MXFP4_K == 0;
}

static inline int ggml_ame_mxfp4_shape_supported(int64_t M, int64_t N, int64_t K) {
    return ggml_ame_mx_shape_supported(M, N, K, 64);
}

// Explicit software configuration attestation; not a fake/probed capability.
static inline int ggml_ame_mx_config_tile_mn(const char * config) {
    if (config == NULL || strcmp(config, "CUTE_2Tops_64SCP") == 0) return 64;
    return strcmp(config, "CUTE_2Tops_128SCP") == 0 ||
        strcmp(config, "CUTE_8Tops_128SCP") == 0 ? AME_TILE_M_MAX : 0;
}

// Pure selection used by the backend only AFTER its existing dual opt-in and
// metadata gate. No CSR discovery, numeric capability or occupancy attestation.
static inline int ggml_ame_mx_select_tile_k(
    int format, int64_t M, int64_t N, int64_t K, int nth, const char * k64
) {
    if (nth != 1 || ggml_ame_mx_block_bytes(format) == 0 ||
        !ggml_ame_mxfp4_shape_supported(M, N, K)) {
        return 0;
    }
    return format == AME_MCFG_MXFP4 && K % 64 == 0 &&
        k64 != NULL && strcmp(k64, "1") == 0 ? 64 : 32;
}

static inline int ggml_ame_mx_tile_k_supported(int format, int tile_k) {
    return ggml_ame_mx_block_bytes(format) != 0 &&
        (tile_k == 32 || (format == AME_MCFG_MXFP4 && tile_k == 64));
}

// Both storage layouts are composed of canonical K32 packed blocks. K64
// interleaves the two blocks within each data/scale row, not across rows.
static inline size_t ggml_ame_mx_packed_block_index(
    int64_t row, size_t kb, size_t nb32, int tile_k, int tile_mn
) {
    const size_t groups = (size_t) tile_k / AME_MXFP4_K;
    const size_t panel = (size_t) (row / tile_mn) * (nb32 / groups) + kb / groups;
    return (panel * (size_t) tile_mn + (size_t) (row % tile_mn)) * groups + kb % groups;
}

// Baseline Q8 only: retain the existing padded row64, quantizer and per-group
// epilogue; consume exactly its one Q8 group32. This is NOT a capacity probe.
static inline int ggml_ame_q8_baseline_compute_k(
    int64_t M, int64_t N, int64_t K, int tile_m, int tile_n, int nth,
    int packed, int whole_k, const char * opt_in
) {
    return opt_in != NULL && strcmp(opt_in, "1") == 0 && !packed && !whole_k &&
        nth == 1 && (tile_m == 64 || tile_m == 128) && tile_n == tile_m &&
        M > 0 && N > 0 && K > 0 && K <= INT32_MAX &&
        M % tile_m == 0 && N % tile_n == 0 && K % AME_Q8_BLOCK_K == 0 ? 32 : 64;
}

typedef struct {
    uint64_t tlenb;
    uint64_t trlenb;
    uint64_t alenb;
    uint64_t rows;
    uint64_t melen_bits;
    uint64_t arlenb;
    int int8_m_max;
    int int8_k_max;
    int int32_n_max;
    int valid;
    int fixed_tile_supported;
} ggml_ame_hw_config;

// Exact advertised geometry, not >=128: larger/different tops are unsupported.
// Existing capacity CSR decoder is unchanged. This pure predicate is testable
// with local values but production passes only the real canonical CSR result.
static inline int ggml_ame_mx_capacity_matches(const ggml_ame_hw_config * cfg, int tile_mn) {
    return cfg != NULL && cfg->valid && (tile_mn == 64 || tile_mn == 128) &&
        cfg->rows == (uint64_t) tile_mn && cfg->int8_m_max == tile_mn &&
        cfg->int32_n_max == tile_mn && cfg->int8_k_max == 64 && cfg->melen_bits == 32 &&
        cfg->trlenb == 64 && cfg->tlenb == (uint64_t) tile_mn * 64 &&
        cfg->arlenb == (uint64_t) tile_mn * 4 &&
        cfg->alenb == (uint64_t) tile_mn * (uint64_t) tile_mn * 4;
}

static inline int ggml_ame_mx_native_tile_mn(const ggml_ame_hw_config * cfg, const char * config) {
    const int tile_mn = ggml_ame_mx_config_tile_mn(config);
    return config != NULL && ggml_ame_mx_capacity_matches(cfg, tile_mn) ? tile_mn : 0;
}

typedef enum {
    GGML_AME_I8_KERNEL_NONE = 0,
    GGML_AME_I8_KERNEL_64_64_64,
    GGML_AME_I8_KERNEL_128_64_128,
} ggml_ame_i8_kernel_kind;

static inline int ggml_ame_i8_128_whole_k_enabled(void) {
    static GGML_AME_THREAD_LOCAL int cached = -1;
    if (cached == -1) {
        const char * value = getenv("GGML_AME_I8_128_WHOLE_K");
        cached = value != NULL && value[0] != '\0' &&
            strcmp(value, "0") != 0 && strcmp(value, "false") != 0 &&
            strcmp(value, "off") != 0 && strcmp(value, "no") != 0;
    }
    return cached;
}

#ifdef __cplusplus
extern "C" {
#endif

void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

void ggml_ame_gemm_tile_i8_i32_bT_q8_k32(
    const int8_t * A, const int8_t * B, int32_t * C);
void ggml_ame_gemm_tile_i8_i32_bT_128_q8_k32(
    const int8_t * A, const int8_t * B, int32_t * C);

void ggml_ame_gemm_tile_i8_i32_bT_kloop(
    const int8_t * a_tiles,
    ptrdiff_t a_tile_stride,
    const int8_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int tile_m,
    int tile_n,
    int32_t * C);

typedef void (*ggml_ame_overlap_fn)(void * opaque);

typedef void (*ggml_ame_pipeline_finish_fn)(void * opaque, int tile_index);

void ggml_ame_gemm_tile_i8_i32_bT_kloop_overlap(
    const int8_t * a_tiles,
    ptrdiff_t a_tile_stride,
    const int8_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int tile_m,
    int tile_n,
    int32_t * C,
    int transpose_c,
    ggml_ame_overlap_fn overlap_fn,
    void * overlap_opaque);

void ggml_ame_gemm_tile_i8_i32_bT_kloop_double_buffer(
    const int8_t * a_panels,
    ptrdiff_t a_panel_stride,
    ptrdiff_t a_tile_stride,
    const int8_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int tile_m,
    int tile_n,
    int32_t * C0,
    int32_t * C1,
    int output_tiles,
    int transpose_c,
    ggml_ame_pipeline_finish_fn finish_fn,
    void * finish_opaque);

unsigned long ggml_ame_gemm_tile_i8_i32_bT_submit(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

unsigned long ggml_ame_gemm_tile_i8_i32_bT_submit_alt(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

void ggml_ame_gemm_tile_i8_i32_bT_128_64_128(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

unsigned long ggml_ame_gemm_tile_i8_i32_bT_128_64_128_submit(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

unsigned long ggml_ame_gemm_tile_i8_i32_bT_128_64_128_submit_alt(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

typedef struct {
    unsigned long sync0_release_target;
} ggml_ame_sync_state;

typedef void (*ggml_ame_gemm_tile_i8_i32_bT_fn)(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

typedef unsigned long (*ggml_ame_gemm_tile_i8_i32_bT_submit_fn)(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

extern GGML_AME_THREAD_LOCAL ggml_ame_sync_state ggml_ame_sync_state_global;

#ifdef __cplusplus
}
#endif

typedef struct {
    ggml_ame_i8_kernel_kind kind;
    int tile_m;
    int tile_k;
    int tile_n;
    const char * name;
    ggml_ame_gemm_tile_i8_i32_bT_fn gemm;
    ggml_ame_gemm_tile_i8_i32_bT_submit_fn submit;
    ggml_ame_gemm_tile_i8_i32_bT_submit_fn submit_alt;
} ggml_ame_i8_kernel;

static inline const char * ggml_ame_i8_capacity_name(const ggml_ame_hw_config * cfg) {
    if (cfg->int8_m_max >= 128 && cfg->int8_k_max >= 64 && cfg->int32_n_max >= 128) {
        return "m128n128k64.i8i32";
    }
    if (cfg->int8_m_max >= 64 && cfg->int8_k_max >= 64 && cfg->int32_n_max >= 64) {
        return "m64n64k64.i8i32";
    }
    return "none";
}

static inline const ggml_ame_i8_kernel * ggml_ame_i8_kernel_for_kind(ggml_ame_i8_kernel_kind kind) {
    static const ggml_ame_i8_kernel kernel_64_64_64 = {
        GGML_AME_I8_KERNEL_64_64_64,
        64,
        64,
        64,
        "m64n64k64.i8i32",
        ggml_ame_gemm_tile_i8_i32_bT,
        ggml_ame_gemm_tile_i8_i32_bT_submit,
        ggml_ame_gemm_tile_i8_i32_bT_submit_alt,
    };
    static const ggml_ame_i8_kernel kernel_128_64_128 = {
        GGML_AME_I8_KERNEL_128_64_128,
        128,
        64,
        128,
        "m128n128k64.i8i32",
        ggml_ame_gemm_tile_i8_i32_bT_128_64_128,
        ggml_ame_gemm_tile_i8_i32_bT_128_64_128_submit,
        ggml_ame_gemm_tile_i8_i32_bT_128_64_128_submit_alt,
    };

    switch (kind) {
        case GGML_AME_I8_KERNEL_64_64_64:
            return &kernel_64_64_64;
        case GGML_AME_I8_KERNEL_128_64_128:
            return &kernel_128_64_128;
        case GGML_AME_I8_KERNEL_NONE:
        default:
            return NULL;
    }
}

static inline const ggml_ame_hw_config * ggml_ame_hw_config_get(void) {
    static GGML_AME_THREAD_LOCAL ggml_ame_hw_config cfg;
    static GGML_AME_THREAD_LOCAL int initialized = 0;
    if (initialized) {
        return &cfg;
    }
    initialized = 1;

#if defined(__riscv)
    __asm__ volatile("csrr %0, 0xcc1" : "=r"(cfg.tlenb));
    __asm__ volatile("csrr %0, 0xcc2" : "=r"(cfg.trlenb));
    __asm__ volatile("csrr %0, 0xcc3" : "=r"(cfg.alenb));
#endif

    if (cfg.tlenb == 0 || cfg.trlenb == 0 || cfg.alenb == 0 ||
            cfg.tlenb == UINT64_MAX || cfg.trlenb == UINT64_MAX || cfg.alenb == UINT64_MAX ||
            cfg.tlenb % cfg.trlenb != 0) {
        AME_LOG("hw probe invalid: tlenb=%llu trlenb=%llu alenb=%llu",
            (unsigned long long) cfg.tlenb,
            (unsigned long long) cfg.trlenb,
            (unsigned long long) cfg.alenb);
        return &cfg;
    }

    cfg.rows = cfg.tlenb / cfg.trlenb;
    if (cfg.rows == 0 || cfg.alenb % cfg.rows != 0 || cfg.rows > UINT64_MAX / cfg.rows || cfg.alenb > UINT64_MAX / 8) {
        AME_LOG("hw probe invalid: tlenb=%llu trlenb=%llu alenb=%llu rows=%llu",
            (unsigned long long) cfg.tlenb,
            (unsigned long long) cfg.trlenb,
            (unsigned long long) cfg.alenb,
            (unsigned long long) cfg.rows);
        return &cfg;
    }

    cfg.arlenb = cfg.alenb / cfg.rows;
    const uint64_t elems = cfg.rows * cfg.rows;
    const uint64_t alen_bits = cfg.alenb * 8;
    if (elems == 0 || alen_bits % elems != 0) {
        AME_LOG("hw probe invalid: tlenb=%llu trlenb=%llu alenb=%llu rows=%llu",
            (unsigned long long) cfg.tlenb,
            (unsigned long long) cfg.trlenb,
            (unsigned long long) cfg.alenb,
            (unsigned long long) cfg.rows);
        return &cfg;
    }

    cfg.melen_bits = alen_bits / elems;
    cfg.int8_m_max = (int) cfg.rows;
    cfg.int8_k_max = (int) cfg.trlenb;
    cfg.int32_n_max = (int) (cfg.arlenb / sizeof(int32_t));
    cfg.valid = 1;
    cfg.fixed_tile_supported =
        cfg.int8_m_max >= AME_TILE_M &&
        cfg.int8_k_max >= AME_TILE_K &&
        cfg.int32_n_max >= AME_TILE_N;

    AME_LOG("hw probe: tlenb=%llu trlenb=%llu alenb=%llu rows=%llu melen_bits=%llu arlenb=%llu capacity=%s fixed_tile=%d",
        (unsigned long long) cfg.tlenb,
        (unsigned long long) cfg.trlenb,
        (unsigned long long) cfg.alenb,
        (unsigned long long) cfg.rows,
        (unsigned long long) cfg.melen_bits,
        (unsigned long long) cfg.arlenb,
        ggml_ame_i8_capacity_name(&cfg),
        cfg.fixed_tile_supported);

    return &cfg;
}

static inline const ggml_ame_i8_kernel * ggml_ame_select_i8_kernel(int64_t M, int64_t N, int64_t K) {
    const ggml_ame_hw_config * cfg = ggml_ame_hw_config_get();
    if (!cfg->valid || K % AME_Q8_BLOCK_K != 0) {
        return NULL;
    }
    const char * force_64_value = getenv("GGML_AME_FORCE_I8_KERNEL_64");
    const int force_64 = force_64_value != NULL &&
        force_64_value[0] != '\0' &&
        strcmp(force_64_value, "0") != 0 &&
        strcmp(force_64_value, "false") != 0 &&
        strcmp(force_64_value, "off") != 0 &&
        strcmp(force_64_value, "no") != 0;

    if (!force_64 && M >= 128 && N >= 128 &&
            cfg->int8_m_max >= 128 && cfg->int8_k_max >= 64 && cfg->int32_n_max >= 128) {
        return ggml_ame_i8_kernel_for_kind(GGML_AME_I8_KERNEL_128_64_128);
    }
    if (M >= 64 && N >= 64 &&
            cfg->int8_m_max >= 64 && cfg->int8_k_max >= 64 && cfg->int32_n_max >= 64) {
        return ggml_ame_i8_kernel_for_kind(GGML_AME_I8_KERNEL_64_64_64);
    }
    return NULL;
}

static inline size_t ggml_ame_i8_kernel_workspace_size(const ggml_ame_i8_kernel * kernel, int64_t N, int64_t K, int use_packed_b_panel) {
    (void) N;
    if (kernel == NULL) {
        return 0;
    }

    const int tile_m = kernel->tile_m;
    const int tile_n = kernel->tile_n;
    const int tile_k = kernel->tile_k;
    const int64_t n_k_tiles = (K + tile_k - 1) / tile_k;

    size_t size = 64;
    size = (size + 63) / 64 * 64;
    size += (size_t) tile_m * tile_k * sizeof(int8_t);
    size = (size + 63) / 64 * 64;
    size += use_packed_b_panel ?
        (size_t) n_k_tiles * tile_n * tile_k * sizeof(int8_t) :
        (size_t) tile_n * tile_k * sizeof(int8_t);
    size = (size + 63) / 64 * 64;
    size += (size_t) tile_m * tile_n * sizeof(int32_t);
    return size;
}

// Partition whole output-row tiles, leaving K reduction within one worker.
// M remains the original output leading dimension, even for an edge shard.
static inline void ggml_ame_worker_rows(
        int64_t M, int tile_m, int ith, int nth, int64_t * begin, int64_t * end) {
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth && tile_m > 0);
    const int64_t tiles = (M + tile_m - 1) / tile_m;
    *begin = (tiles * ith / nth) * tile_m;
    *end = (tiles * (ith + 1) / nth) * tile_m;
    if (*begin > M) *begin = M;
    if (*end > M) *end = M;
}

static inline void ggml_ame_sync_begin_op(void) {
    ggml_ame_sync_state_global.sync0_release_target = 0;
#if defined(__riscv)
    asm volatile("msyncregreset sync0" ::: "memory");
#endif
}

// Submit a matrix task without blocking the scalar core.  The matching
// acquire can be delayed until several tiles have entered the AMU queue.
static inline unsigned long ggml_ame_sync_release(void) {
#if defined(__riscv)
    asm volatile("mrelease sync0" ::: "memory");
    return ++ggml_ame_sync_state_global.sync0_release_target;
#else
    return ++ggml_ame_sync_state_global.sync0_release_target;
#endif
}

static inline void ggml_ame_sync_acquire_wait(unsigned long acquire_target) {
#if defined(__riscv)
    asm volatile("macquire sync0, %0" :: "r"(acquire_target) : "memory");
    asm volatile("mfence" ::: "memory");
#else
    (void) acquire_target;
#endif
}

static inline unsigned long ggml_ame_sync_release_acquire_store(void) {
    const unsigned long acquire_target = ggml_ame_sync_release();
    ggml_ame_sync_acquire_wait(acquire_target);
    return acquire_target;
}

// Helper function to check if AME can be used for given dimensions.
//
// The current AME backend only accelerates Q8_0 GEMM. It only reaches the
// AME core on full MxN tiles; smaller shapes fall back to RVV/scalar helpers
// and usually lose to the generic CPU path once the extra quantize/pack cost is
// accounted for.
static inline int ggml_ame_can_use(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    return ggml_ame_select_i8_kernel(M, N, K) != NULL;
}

static inline int ggml_ame_can_use_q8(int M, int N, int K) {
    return ggml_ame_can_use(M, N, K);
}

static inline int ggml_ame_can_use_bf16(int M, int N, int K) {
    if (M < AME_TILE_M || N < AME_TILE_N || K < AME_TILE_K_BF16) {
        return 0;
    }

    const ggml_ame_hw_config * cfg = ggml_ame_hw_config_get();
    return cfg->valid &&
        cfg->int8_m_max >= AME_TILE_M &&
        cfg->int8_k_max >= AME_TILE_K &&
        cfg->int32_n_max >= AME_TILE_N;
}

// Repacked Q4_0 format for AME (pre-unpacked to int8)
// This avoids unpacking overhead during every matmul
typedef struct {
    uint16_t d;         // scale factor FP16 (same format as block_q4_0)
    int8_t qs[32];      // pre-unpacked 4-bit values to int8 [-8, 7]
} block_q4_0_ame;

typedef struct {
    uint16_t d;
    int8_t qs[AME_Q8_PACK_K];
} block_q8_ame64;

#define AME_MCFG_INT8  0x02
#define AME_MCFG_UINT8 0x03
#define AME_MCFG_INT32 0x04
#define AME_MCFG_BF16  0x0a
#define AME_MCFG_FP32  0x0c

// Matrix configuration instructions
#ifdef STC
#define MSETSEW(RD, SEW) \
    asm volatile ( \
        "msetsew %0, %1" \
        : "=r"(RD) \
        : "i"(SEW) \
        : \
    )

#define MSETINT8(RD, VAL) \
    asm volatile ( \
        "msetint8 %0, %1" \
        : "=r"(RD) \
        : "i"(VAL) \
        : \
    )

#define MSETTILEM(RD, VAL) \
    asm volatile ( \
        "msettilem %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

#define MSETTILEK(RD, VAL) \
    asm volatile ( \
        "msettilek %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

#define MSETTILEN(RD, VAL) \
    asm volatile ( \
        "msettilen %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

// Matrix accumulator zero instruction
#define MZERO(ACC) \
    asm volatile ( \
        "mzero " #ACC \
        : \
        : \
        : \
    )

// Matrix load instructions
#define MLA(REG, SRC, N) \
    asm volatile ( \
        "mla " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLB(REG, SRC, N) \
    asm volatile ( \
        "mlb " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLC(REG, SRC, N) \
    asm volatile ( \
        "mlc " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSC(REG, DST, N) \
    asm volatile ( \
        "msc " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

#define MSCT(REG, DST, N) \
    asm volatile ( \
        "msct " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMACC(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc " #ACC ", " #TR2 ", " #TR0 "\n" \
        : \
        : \
        : \
    )
#else
#define MSETSEW(RD, SEW) ((void)0) //非STC没有这条指令, 空指令

#define MSETINT8(RD, VAL) ((void)0) //非STC没有这条指令, 空指令

#define MSETTILEM(RD, VAL) \
    asm volatile ( \
        "msettilem %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

#define MSETTILEK(RD, VAL) \
    asm volatile ( \
        "msettilek %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

#define MSETTILEN(RD, VAL) \
    asm volatile ( \
        "msettilen %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

// Matrix accumulator zero instruction
#define MZERO(ACC) \
    asm volatile ( \
        "mzero " #ACC \
        : \
        : \
        : \
    )

// Matrix load instructions
#define MLA(REG, SRC, N) \
    asm volatile ( \
        "mla " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLB(REG, SRC, N) \
    asm volatile ( \
        "mlb " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLC(REG, SRC, N) \
    asm volatile ( \
        "mlc " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSC(REG, DST, N) \
    asm volatile ( \
        "msc " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

#define MSCT(REG, DST, N) \
    asm volatile ( \
        "msct " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMACC(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc " #ACC ", " #TR2 ", " #TR0 "\n" \
        : \
        : \
        : \
    )
#endif

#define MSETCFG(CFG, VAL) \
    asm volatile ( \
        "msetcfg " #CFG ", %0" \
        : \
        : "r"(VAL) \
        : "memory" \
    )

static inline void ggml_ame_config_i8_i32(void) {
#if defined(__riscv)
    const unsigned long cfg_i8 = AME_MCFG_INT8;
    const unsigned long cfg_i32 = AME_MCFG_INT32;
    MSETCFG(mcfg0, cfg_i8);
    MSETCFG(mcfg1, cfg_i8);
    MSETCFG(mcfg2, cfg_i8);
    MSETCFG(mcfg3, cfg_i8);
    MSETCFG(mcfg4, cfg_i32);
    MSETCFG(mcfg5, cfg_i32);
    MSETCFG(mcfg6, cfg_i32);
    MSETCFG(mcfg7, cfg_i32);
#endif
}

static inline void ggml_ame_config_bf16_fp32(void) {
#if defined(__riscv)
    const unsigned long cfg_bf16 = AME_MCFG_BF16;
    const unsigned long cfg_fp32 = AME_MCFG_FP32;
    MSETCFG(mcfg0, cfg_bf16);
    MSETCFG(mcfg1, cfg_bf16);
    MSETCFG(mcfg2, cfg_bf16);
    MSETCFG(mcfg3, cfg_bf16);
    MSETCFG(mcfg4, cfg_fp32);
    MSETCFG(mcfg5, cfg_fp32);
    MSETCFG(mcfg6, cfg_fp32);
    MSETCFG(mcfg7, cfg_fp32);
#endif
}

typedef struct {
    uint64_t calls;
    uint64_t output_tiles;
    uint64_t k_tiles;
    uint64_t whole_k_calls;
    uint64_t cycles_total;
    uint64_t cycles_convert_src1;
    uint64_t cycles_pack_a;
    uint64_t cycles_pack_b;
    uint64_t cycles_ame;
    uint64_t cycles_accumulate;
    uint64_t cycles_store;
} ggml_ame_bf16_profile;

#ifdef __cplusplus
extern "C" {
#endif

// Existing system-mode nemu_signal owner, also used by explicit checkpoint
// microbenchmarks. Codes retain docs/checkpoint-workload-template.md semantics.
// Not a user-mode feature probe; callers must explicitly select system ROI.
void ggml_ame_nemu_signal(int code);

// Atomic tile GEMM (16x32x16)
void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,
    const int8_t * B,
    int32_t * C
);

void ggml_ame_gemm_tile_bf16_fp32_bT(
    const ggml_bf16_t * A,
    const ggml_bf16_t * B,
    float * C
);

void ggml_ame_gemm_tile_bf16_fp32_bT_kloop(
    const ggml_bf16_t * a_tiles,
    ptrdiff_t a_tile_stride,
    const ggml_bf16_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int reg_pairs,
    float * C
);

// Shared K32 path: compact 16-byte E2M1 or 32-byte FP8 data rows and one E8M0
// byte per row. Only called after the explicit MX32 capability attestation.
void ggml_ame_gemm_tile_mx_fp32_bT_kloop(
    const uint8_t * a, const uint8_t * sa,
    const uint8_t * b, const uint8_t * sb, int64_t k_tiles, float * c, int format, int tile_k, int tile_mn,
    ggml_ame_overlap_fn overlap_fn, void * overlap_opaque);

// Lossless GGML low=k/high=k+16 -> consecutive-K nibbles, separate E8M0.
// Output sizes are nblocks*16 and nblocks bytes. Buffers must not overlap.
// No hardware instructions, quantization or scale conversion.
GGML_API bool ggml_ame_pack_mxfp4_blocks(
    uint8_t * payload, uint8_t * scales, const void * src, int64_t nblocks);

// CPU producer only: ordinary RMSNorm + gamma, real CPU MXFP4 quantizer and
// canonical AME layout. No normalized full-tensor output or matrix instruction.
// Inputs/outputs must not overlap; gamma has K contiguous F32 elements.
// F32 rows are naturally aligned; byte stride must preserve that alignment.
// Reject nonfinite normalized blocks and nonzero amax below 2^-125 before
// invoking the legacy CPU scale cast; epsilon must be finite and positive.
// On false discard all partial payload/scales. Scalar flags naturally accrue.
// Optional row_scales are reciprocal factors validated by preflight below.
// CPU-only first-panel preparation within that preflight is provisional;
// downstream consumption requires successful WHOLE-input admission. Keep
// source/gamma, rounding mode and factors unchanged through consumption.
GGML_API bool ggml_ame_rmsnorm_pack_mxfp4(
    uint8_t * payload, uint8_t * scales, const float * src, const float * gamma,
    int64_t rows, int64_t K, size_t src_stride, float eps, int tile_k, int tile_mn,
    const float * row_scales);

// Optional first-panel output in exclusive scratch, not a published tensor.
// On false discard factors AND all provisional payload/scales. profile!=0
// measures just the existing producer call on RV64 (zero on other targets).
struct ggml_ame_rmsnorm_first_panel {
    uint8_t * payload;
    uint8_t * scales;
    int tile_k;
    int profile;
    uint64_t prepare_cycles;
};

// CPU-only whole-input admission before any native issue. Writes rows F32
// factors into exclusive scratch, reuses the same normalized-block checks.
// first==NULL preserves factors-only/no-packed-output behavior. Otherwise
// prepare the first tile_mn rows ONCE using the same canonical producer, then
// validate remaining rows. No output is usable until this call succeeds.
GGML_API bool ggml_ame_rmsnorm_mxfp4_preflight(
    float * row_scales, const float * src, const float * gamma,
    int64_t rows, int64_t K, size_t src_stride, float eps, int tile_mn,
    struct ggml_ame_rmsnorm_first_panel * first);

// Native dispatch domain, deliberately stricter than the raw lossless packer:
// GGML's half-scale decoder treats e=255 as finite; MX ISA treats it as NaN.
GGML_API bool ggml_ame_mxfp4_native_domain(const void * src, int64_t nblocks);

// Backend-private preparation only: no GGML tensor/model format or public API.
// Complete K32 blocks, finite F32 sources, separate one-byte E8M0 per block.
// Unknown formats, tails and nonfinite source tensors fail before output writes.
bool ggml_ame_mxfp8_f32_domain(const float * src, int64_t nvalues, int format);
bool ggml_ame_pack_mxfp8_f32(
    uint8_t * payload, uint8_t * scales, const float * src, int64_t nvalues, int format);
bool ggml_ame_mxfp8_native_domain(
    const uint8_t * payload, const uint8_t * scales, int64_t nblocks, int format);
// Internal directed test of actual compute rejection before CSR discovery.
bool ggml_ame_mx_invalid_compute_selftest(void);
bool ggml_ame_mxfp4_k64_selftest(void);
bool ggml_ame_q8_k32_selftest(void);
bool ggml_ame_mx128_selftest(bool run_native, bool qwen_shape);
// Backend-private preparation seam: no CSR, allocation, sync or native issue.
void ggml_ame_mx_prepare_activation(
    uint8_t * payload, uint8_t * scales, const void * src,
    int64_t N, int64_t K, size_t stride, int format, int tile_k, int tile_mn);

// Software-only cache/callback regression; never probes or submits to AMU.
GGML_API bool ggml_ame_mxfp4_cache_selftest(void);

// Packed panel streams are [output_tile][K_block][row][packed_K].
struct ggml_ame_rmsnorm_input {
    const float * gamma;
    const float * row_scales; // whole-input preflight, exclusive workspace prefix
    float eps;
    int pipeline;
    int first_panel_prepared;
    uint64_t preflight_cycles;
    uint64_t prepare_cycles;
    uint64_t prepared_panels;
    uint64_t overlap_callbacks;
};
struct ggml_ame_mx_activation_reuse {
    const struct ggml_cgraph * graph;
    const struct ggml_tensor * source;
    struct ggml_tensor * qkv_norm;
    struct ggml_tensor * qkv_last;
    const void * source_data;
    const void * workspace;
    size_t workspace_bytes;
    void * interposer_work;
    size_t interposer_bytes;
    size_t stride;
    int tile_k;
    int tile_mn;
    unsigned interposers;
    unsigned qkv_dispatched;
    unsigned fcsr;
    int saved_errno;
    bool legacy;
    bool incoming;
    bool published;
    bool hit;
};

void ggml_ame_mul_mat_mx(
    const uint8_t * a, const uint8_t * sa, const void * src1, float * dst,
    int64_t M, int64_t N, int64_t K, size_t src1_stride,
    void * work_data, int ith, int nth,
    uint64_t weight_prepare_cycles, int weight_cache_hit, int format, int tile_k, int tile_mn,
    struct ggml_ame_rmsnorm_input * rms, struct ggml_ame_mx_activation_reuse * reuse);
void ggml_ame_mx_profile_finish(uint64_t elapsed_cycles, int format, int tile_k);
void ggml_ame_mx_profile_qkv_producer(uint64_t elapsed_cycles, int tile_k);

struct ggml_compute_params;
bool ggml_ame_mx_activation_interpose(
    struct ggml_ame_mx_activation_reuse * reuse, const struct ggml_tensor * node, int node_index);
bool ggml_ame_begin_rmsnorm_qkv(
    struct ggml_compute_params * params, const struct ggml_cgraph * graph, int node_index);
bool ggml_ame_try_fuse_rmsnorm_mxfp4(
    struct ggml_compute_params * params, const struct ggml_cgraph * graph, int node_index);
bool ggml_ame_rmsnorm_fusion_selftest(bool run_native, int fixture, bool benchmark);

void ggml_ame_bf16_profile_reset(void);
void ggml_ame_bf16_profile_snapshot(ggml_ame_bf16_profile * profile);

// Core AME GEMM function for INT8 matrix multiplication
// C(M×N) += A(M×K) × B(K×N), where B is transposed in memory
void ggml_ame_gemm_q8_0(
    const int8_t * A,
    const int8_t * B,
    int32_t * C,
    int M,
    int K,
    int N
);

void ggml_ame_profile_reset(void);
void ggml_ame_profile_dump_now(void);
// Pure quantizer test seam; q has ceil(k/64)*64 bytes, scale ceil(k/64)
// floats. Includes padded lanes; does not issue AME or update profiling/cache.
void ggml_ame_quantize_whole_k_test(const float * x, int64_t k, int8_t * q, float * scale);

// Quantize a row of F32 values to Q8_0 format
void ggml_ame_quantize_row_f32_to_q8_0(const float * x, void * y, int k);

// Q4_0 weight repacking (called once during set_tensor)
void ggml_ame_repack_q4_0(
    void * dst,              // Output: block_q4_0_ame array
    const void * src,        // Input: block_q4_0 array
    int64_t nblocks          // Number of Q4_0 blocks
);

// Experimental Q8_0 -> AME-native packed K64 repack.
void ggml_ame_repack_q8_0_to_ame64(
    void * dst,
    const void * src,
    int64_t nrows,
    int64_t k
);

void ggml_ame_repack_q8_0_to_ame64_whole_k(
    void * dst,
    const void * src,
    int64_t nrows,
    int64_t k
);

// GGML integration wrapper for Q8_0 quantized matrix multiplication
void ggml_ame_mul_mat_q8_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride,  // stride in bytes for src1 columns (nb[1])
    void * work_data,
    size_t work_size,
    int ith,
    int nth
);

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
    size_t work_size,
    int ith,
    int nth
);

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
    size_t work_size,
    int ith,
    int nth
);

// GGML integration wrapper for Q4_0 quantized matrix multiplication
void ggml_ame_mul_mat_q4_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride
);

#ifdef __cplusplus
}
#endif

#endif // GGML_RISCV_AME_H
