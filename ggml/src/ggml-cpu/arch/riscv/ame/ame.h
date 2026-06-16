#ifndef GGML_RISCV_AME_H
#define GGML_RISCV_AME_H

#include <stdint.h>
#include <stddef.h>  // for size_t

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// AME debug logging can be forced at build time with AME_DEBUG=1 or enabled at
// runtime with GGML_AME_LOG=1 so logs are visible in QEMU serial output.
#ifndef AME_DEBUG
#define AME_DEBUG 0
#endif

static inline int ame_log_enabled(void) {
#if AME_DEBUG
    return 1;
#else
    static int cached = -1;
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

#define AME_Q8_BLOCK_K 32
#define AME_Q8_PACK_K 64

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

typedef enum {
    GGML_AME_I8_KERNEL_NONE = 0,
    GGML_AME_I8_KERNEL_64_64_64,
    GGML_AME_I8_KERNEL_128_64_128,
} ggml_ame_i8_kernel_kind;

#ifdef __cplusplus
extern "C" {
#endif

void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

void ggml_ame_gemm_tile_i8_i32_bT_128_64_128(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

typedef struct {
    unsigned long tok0_release_target;
} ggml_ame_sync_state;

typedef void (*ggml_ame_gemm_tile_i8_i32_bT_fn)(
    const int8_t * A,
    const int8_t * B,
    int32_t * C);

extern ggml_ame_sync_state ggml_ame_sync_state_global;

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
    };
    static const ggml_ame_i8_kernel kernel_128_64_128 = {
        GGML_AME_I8_KERNEL_128_64_128,
        128,
        64,
        128,
        "m128n128k64.i8i32",
        ggml_ame_gemm_tile_i8_i32_bT_128_64_128,
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
    static ggml_ame_hw_config cfg;
    static int initialized = 0;
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
    if (M >= 128 && N >= 128 &&
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

static inline void ggml_ame_sync_begin_op(void) {
#if defined(__riscv)
    asm volatile("msyncreset tok0" ::: "memory");
#endif
    ggml_ame_sync_state_global.tok0_release_target = 0;
}

static inline unsigned long ggml_ame_sync_release_acquire_store(void) {
#if defined(__riscv)
    asm volatile("mrelease tok0" ::: "memory");
    const unsigned long acquire_target = ++ggml_ame_sync_state_global.tok0_release_target;
    asm volatile("macquire %0,tok0" :: "r"(acquire_target) : "memory");
#else
    const unsigned long acquire_target = ++ggml_ame_sync_state_global.tok0_release_target;
#endif
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
#define MZERO_ACC(ACC) \
    asm volatile ( \
        "mzero.acc.m " #ACC \
        : \
        : \
        : \
    )

// Matrix load instructions
#define MLAE8(REG, SRC, N) \
    asm volatile ( \
        "mlae8.m " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLBE8(REG, SRC, N) \
    asm volatile ( \
        "mlbe8.m " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLCE32(REG, SRC, N) \
    asm volatile ( \
        "mlce32.m " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSCE32(REG, DST, N) \
    asm volatile ( \
        "msce32.m " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMA(ACC, TR0, TR2) \
    asm volatile ( \
        "mmau.mm " #ACC ", " #TR0 ", " #TR2 "\n" \
        : \
        : \
        : \
    )

#define MQMA(ACC, TR0, TR2) \
    asm volatile ( \
        "mqma.mm " #ACC ", " #TR0 ", " #TR2 "\n" \
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
#define MZERO_ACC(ACC) \
    asm volatile ( \
        "mzero1r " #ACC \
        : \
        : \
        : \
    )

// Matrix load instructions
#define MLAE8(REG, SRC, N) \
    asm volatile ( \
        "mlae8 " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLBE8(REG, SRC, N) \
    asm volatile ( \
        "mlbe8 " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLCE32(REG, SRC, N) \
    asm volatile ( \
        "mlce32 " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSCE32(REG, DST, N) \
    asm volatile ( \
        "msce32 " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMAU(ACC, TR0, TR2) \
    asm volatile ( \
        "mmaccu.w.b" #ACC ", " #TR0 ", " #TR2 "\n" \
        : \
        : \
        : \
    )

#define MQMA(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc.w.b " #ACC ", " #TR0 ", " #TR2 "\n" \
        : \
        : \
        : \
    )
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Atomic tile GEMM (16x32x16)
void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,
    const int8_t * B,
    int32_t * C
);

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
    size_t work_size
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
    size_t work_size
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
