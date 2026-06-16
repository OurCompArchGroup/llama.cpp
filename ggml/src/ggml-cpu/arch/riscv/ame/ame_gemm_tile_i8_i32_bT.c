#include "ame.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ame_tile_progress_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        const char * value = getenv("GGML_AME_TILE_PROGRESS_LOG");
        cached = value != NULL && value[0] != '\0' && strcmp(value, "0") != 0 &&
            strcmp(value, "false") != 0 && strcmp(value, "off") != 0 && strcmp(value, "no") != 0;
    }
    return cached;
}

#define AME_TILE_PROGRESS_LOG(...) do {                         \
    if (ame_tile_progress_enabled()) {                          \
        fprintf(stderr, "[AME_TILE_PROGRESS] " __VA_ARGS__);    \
        fprintf(stderr, "\n");                                  \
        fflush(stderr);                                         \
    }                                                           \
} while (0)

// Make scalar stores to tile buffers visible before AME matrix loads read them.
#define AME_MLOAD_FENCE_BEFORE_LOAD(TAG) do {                    \
    AME_TILE_PROGRESS_LOG("fence_before_begin%s", TAG);          \
    asm volatile("fence rw, rw" ::: "memory");                   \
    AME_TILE_PROGRESS_LOG("fence_before_done%s", TAG);           \
} while (0)

// INT8 GEMM using RISC-V AME instructions
// Tile size: M=AME_TILE_M, K=AME_TILE_K, N=AME_TILE_N (atomic AME variant)
// C(MxN) = A(MxK) × B^T(NxK), where B is transposed in memory
// This function computes a single MxN tile output
#define DEFINE_AME_GEMM_TILE(FUNC, TILE_M_VALUE, TILE_K_VALUE, TILE_N_VALUE, TAG)       \
void FUNC(const int8_t * A, const int8_t * B, int32_t * C) {                            \
    const int TILE_M = (TILE_M_VALUE);                                                  \
    const int TILE_K = (TILE_K_VALUE);                                                  \
    const int TILE_N = (TILE_N_VALUE);                                                  \
                                                                                        \
    int tmp;                                                                            \
    AME_TILE_PROGRESS_LOG("mset_begin%s A=%p B=%p C=%p", TAG,                           \
        (const void *) A, (const void *) B, (void *) C);                                \
    MSETTILEM(tmp, TILE_M);                                                             \
    MSETTILEK(tmp, TILE_K);                                                             \
    MSETTILEN(tmp, TILE_N);                                                             \
    AME_TILE_PROGRESS_LOG("mset_done%s", TAG);                                         \
                                                                                        \
    int32_t * addr_c = C;                                                               \
    const int stride_c = TILE_N;                                                        \
                                                                                        \
    AME_TILE_PROGRESS_LOG("mzero_begin%s", TAG);                                       \
    MZERO_ACC(acc0);                                                                    \
    AME_TILE_PROGRESS_LOG("mzero_done%s", TAG);                                        \
    AME_MLOAD_FENCE_BEFORE_LOAD(TAG);                                                   \
                                                                                        \
    const int8_t * addr_a = A;                                                          \
    AME_TILE_PROGRESS_LOG("mlae8_begin%s A=%p stride=%d", TAG,                         \
        (const void *) addr_a, TILE_K);                                                 \
    MLAE8(tr0, addr_a, TILE_K);                                                         \
    AME_TILE_PROGRESS_LOG("mlae8_done%s", TAG);                                        \
                                                                                        \
    const int8_t * addr_b = B;                                                          \
    AME_TILE_PROGRESS_LOG("mlbe8_begin%s B=%p stride=%d", TAG,                         \
        (const void *) addr_b, TILE_K);                                                 \
    MLBE8(tr1, addr_b, TILE_K);                                                         \
    AME_TILE_PROGRESS_LOG("mlbe8_done%s", TAG);                                        \
                                                                                        \
    AME_TILE_PROGRESS_LOG("mqma_begin%s", TAG);                                        \
    MQMA(acc0, tr0, tr1);                                                               \
    AME_TILE_PROGRESS_LOG("mqma_done%s", TAG);                                         \
                                                                                        \
    AME_TILE_PROGRESS_LOG("msce32_begin%s C=%p stride_bytes=%d", TAG,                  \
        (void *) addr_c, stride_c * 4);                                                 \
    MSCE32(acc0, addr_c, stride_c * 4);                                                 \
    AME_TILE_PROGRESS_LOG("msce32_done%s", TAG);                                       \
                                                                                        \
    AME_TILE_PROGRESS_LOG("mrelease_begin%s", TAG);                                    \
    const unsigned long acquire_target = ggml_ame_sync_release_acquire_store();         \
    AME_TILE_PROGRESS_LOG("mrelease_done%s", TAG);                                     \
    AME_TILE_PROGRESS_LOG("macquire_done%s target=%lu", TAG, acquire_target);          \
}

DEFINE_AME_GEMM_TILE(ggml_ame_gemm_tile_i8_i32_bT, 64, 64, 64, "")
DEFINE_AME_GEMM_TILE(ggml_ame_gemm_tile_i8_i32_bT_128_64_128, 128, 64, 128, "_128")
