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
    ggml_ame_config_i8_i32();                                                           \
    AME_TILE_PROGRESS_LOG("mset_done%s", TAG);                                         \
                                                                                        \
    int32_t * addr_c = C;                                                               \
    const int stride_c = TILE_N;                                                        \
                                                                                        \
    AME_TILE_PROGRESS_LOG("mzero_begin%s", TAG);                                       \
    MZERO(acc0);                                                                        \
    AME_TILE_PROGRESS_LOG("mzero_done%s", TAG);                                        \
    AME_MLOAD_FENCE_BEFORE_LOAD(TAG);                                                   \
                                                                                        \
    const int8_t * addr_a = A;                                                          \
    AME_TILE_PROGRESS_LOG("mla_begin%s A=%p stride=%d", TAG,                           \
        (const void *) addr_a, TILE_K);                                                 \
    MLA(tr0, addr_a, TILE_K);                                                           \
    AME_TILE_PROGRESS_LOG("mla_done%s", TAG);                                          \
                                                                                        \
    const int8_t * addr_b = B;                                                          \
    AME_TILE_PROGRESS_LOG("mlb_begin%s B=%p stride=%d", TAG,                           \
        (const void *) addr_b, TILE_K);                                                 \
    MLB(tr1, addr_b, TILE_K);                                                           \
    AME_TILE_PROGRESS_LOG("mlb_done%s", TAG);                                          \
                                                                                        \
    AME_TILE_PROGRESS_LOG("mmacc_begin%s", TAG);                                       \
    MMACC(acc0, tr0, tr1);                                                              \
    AME_TILE_PROGRESS_LOG("mmacc_done%s", TAG);                                        \
                                                                                        \
    AME_TILE_PROGRESS_LOG("msc_begin%s C=%p stride_bytes=%d", TAG,                     \
        (void *) addr_c, stride_c * 4);                                                 \
    MSC(acc0, addr_c, stride_c * 4);                                                    \
    AME_TILE_PROGRESS_LOG("msc_done%s", TAG);                                          \
                                                                                        \
    AME_TILE_PROGRESS_LOG("mrelease_begin%s", TAG);                                    \
    const unsigned long acquire_target = ggml_ame_sync_release_acquire_store();         \
    AME_TILE_PROGRESS_LOG("mrelease_done%s", TAG);                                     \
    AME_TILE_PROGRESS_LOG("macquire_done%s target=%lu", TAG, acquire_target);          \
}

DEFINE_AME_GEMM_TILE(ggml_ame_gemm_tile_i8_i32_bT, 64, 64, 64, "")
DEFINE_AME_GEMM_TILE(ggml_ame_gemm_tile_i8_i32_bT_128_64_128, 128, 64, 128, "_128")

void ggml_ame_gemm_tile_i8_i32_bT_kloop_overlap(
    const int8_t * a_tiles,
    ptrdiff_t a_tile_stride,
    const int8_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int32_t * C,
    int transpose_c,
    ggml_ame_overlap_fn overlap_fn,
    void * overlap_opaque
) {
    if (k_tiles <= 0) {
        return;
    }

    int tmp;
    MSETTILEM(tmp, AME_TILE_M);
    MSETTILEK(tmp, AME_TILE_K);
    MSETTILEN(tmp, AME_TILE_N);
    ggml_ame_config_i8_i32();

    MZERO(acc0);
    AME_MLOAD_FENCE_BEFORE_LOAD("_kloop");

    const int8_t * addr_a = a_tiles;
    const int8_t * addr_b = b_tiles;
    int kb = 0;
    // Keep both tensor-register pairs busy and remove the per-tile parity
    // branch from the issue stream.
    for (; kb + 1 < k_tiles; kb += 2) {
        MLA(tr0, addr_a, AME_TILE_K);
        MLB(tr1, addr_b, AME_TILE_K);
        MMACC(acc0, tr0, tr1);
        addr_a += a_tile_stride;
        addr_b += b_tile_stride;

        MLA(tr2, addr_a, AME_TILE_K);
        MLB(tr3, addr_b, AME_TILE_K);
        MMACC(acc0, tr2, tr3);
        addr_a += a_tile_stride;
        addr_b += b_tile_stride;
    }
    if (kb < k_tiles) {
        MLA(tr0, addr_a, AME_TILE_K);
        MLB(tr1, addr_b, AME_TILE_K);
        MMACC(acc0, tr0, tr1);
    }

    // The AME queue is now draining loads and MMACC operations for this tile.
    // Finish the previous tile before submitting this tile's store: running
    // vector memory operations after MSC/mrelease can deadlock with the AME
    // store, while this ordering still overlaps RVV work with matrix compute.
    if (overlap_fn != NULL) {
        overlap_fn(overlap_opaque);
        asm volatile("fence rw, rw" ::: "memory");
    }

    // CUTE consumes the MSC base register after software has issued the
    // instruction. Keep it in a callee-saved register until acquire so the
    // compiler cannot recycle the address register while the store is pending.
    register int32_t * c_live __asm__("s11") = C;
    if (transpose_c) {
        MSCT(acc0, c_live, AME_TILE_M * (int) sizeof(int32_t));
    } else {
        MSC(acc0, c_live, AME_TILE_N * (int) sizeof(int32_t));
    }
    const unsigned long target = ggml_ame_sync_release();
    ggml_ame_sync_acquire_wait(target);
    asm volatile("" : "+r"(c_live) : : "memory");
}

void ggml_ame_gemm_tile_i8_i32_bT_kloop(
    const int8_t * a_tiles,
    ptrdiff_t a_tile_stride,
    const int8_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int32_t * C
) {
    ggml_ame_gemm_tile_i8_i32_bT_kloop_overlap(
        a_tiles, a_tile_stride, b_tiles, b_tile_stride, k_tiles, C, 0, NULL, NULL);
}

#define AME_KLOOP_COMPUTE(ACC) do {                                      \
    MZERO(ACC);                                                          \
    const int8_t * addr_a = a_tiles;                                     \
    const int8_t * addr_b = b_tiles;                                     \
    int kb = 0;                                                          \
    for (; kb + 1 < k_tiles; kb += 2) {                                  \
        MLA(tr0, addr_a, AME_TILE_K);                                     \
        MLB(tr1, addr_b, AME_TILE_K);                                     \
        MMACC(ACC, tr0, tr1);                                             \
        addr_a += a_tile_stride;                                          \
        addr_b += b_tile_stride;                                          \
        MLA(tr2, addr_a, AME_TILE_K);                                     \
        MLB(tr3, addr_b, AME_TILE_K);                                     \
        MMACC(ACC, tr2, tr3);                                             \
        addr_a += a_tile_stride;                                          \
        addr_b += b_tile_stride;                                          \
    }                                                                     \
    if (kb < k_tiles) {                                                    \
        MLA(tr0, addr_a, AME_TILE_K);                                     \
        MLB(tr1, addr_b, AME_TILE_K);                                     \
        MMACC(ACC, tr0, tr1);                                             \
    }                                                                     \
} while (0)

void ggml_ame_gemm_tile_i8_i32_bT_kloop_double_buffer(
    const int8_t * a_panels,
    ptrdiff_t a_panel_stride,
    ptrdiff_t a_tile_stride,
    const int8_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int32_t * C0,
    int32_t * C1,
    int output_tiles,
    int transpose_c,
    ggml_ame_pipeline_finish_fn finish_fn,
    void * finish_opaque
) {
    if (k_tiles <= 0 || output_tiles <= 0) {
        return;
    }

    int tmp;
    MSETTILEM(tmp, AME_TILE_M);
    MSETTILEK(tmp, AME_TILE_K);
    MSETTILEN(tmp, AME_TILE_N);
    ggml_ame_config_i8_i32();
    AME_MLOAD_FENCE_BEFORE_LOAD("_double_buffer");

    // CUTE may consume an MSC base register after the scalar core has issued
    // the instruction. Keep one callee-saved register per output slot live
    // across the complete stream; each slot is reused only after its token was
    // acquired and its epilogue completed.
    register int32_t * c0_live __asm__("s10") = C0;
    register int32_t * c1_live __asm__("s11") = C1;
    unsigned long pending_target = 0;

    for (int tile = 0; tile < output_tiles; ++tile) {
        const int slot = tile & 1;
        const int8_t * a_tiles = a_panels + (ptrdiff_t) tile * a_panel_stride;

        if (slot == 0) {
            AME_KLOOP_COMPUTE(acc0);
        } else {
            AME_KLOOP_COMPUTE(acc1);
        }

        // Let the current compute drain while the scalar/vector core finishes
        // the previous tile. Submit the current MSC only after that epilogue,
        // so vector memory traffic never overlaps a pending matrix store.
        if (pending_target != 0) {
            ggml_ame_sync_acquire_wait(pending_target);
            if (finish_fn != NULL) {
                finish_fn(finish_opaque, tile - 1);
                asm volatile("fence rw, rw" ::: "memory");
            }
        }

        if (slot == 0) {
            if (transpose_c) {
                MSCT(acc0, c0_live, AME_TILE_M * (int) sizeof(int32_t));
            } else {
                MSC(acc0, c0_live, AME_TILE_N * (int) sizeof(int32_t));
            }
        } else {
            if (transpose_c) {
                MSCT(acc1, c1_live, AME_TILE_M * (int) sizeof(int32_t));
            } else {
                MSC(acc1, c1_live, AME_TILE_N * (int) sizeof(int32_t));
            }
        }
        pending_target = ggml_ame_sync_release();
    }

    ggml_ame_sync_acquire_wait(pending_target);
    if (finish_fn != NULL) {
        finish_fn(finish_opaque, output_tiles - 1);
    }
    asm volatile("" : "+r"(c0_live), "+r"(c1_live) : : "memory");
}

#undef AME_KLOOP_COMPUTE

// Asynchronous variant used by the software K-tile pipeline.  The caller
// owns the output buffer until ggml_ame_sync_acquire_wait() reaches the
// returned token, so several submissions can share the AMU queue safely.
#define DEFINE_AME_GEMM_TILE_SUBMIT(FUNC, TILE_M_VALUE, TILE_K_VALUE, TILE_N_VALUE, ACC, TRA, TRB, TAG) \
unsigned long FUNC(const int8_t * A, const int8_t * B, int32_t * C) {                                   \
    const int TILE_M = (TILE_M_VALUE);                                                                  \
    const int TILE_K = (TILE_K_VALUE);                                                                  \
    const int TILE_N = (TILE_N_VALUE);                                                                  \
    int tmp;                                                                                            \
    AME_TILE_PROGRESS_LOG("async_mset_begin%s A=%p B=%p C=%p", TAG,                                   \
        (const void *) A, (const void *) B, (void *) C);                                                \
    MSETTILEM(tmp, TILE_M);                                                                             \
    MSETTILEK(tmp, TILE_K);                                                                             \
    MSETTILEN(tmp, TILE_N);                                                                             \
    ggml_ame_config_i8_i32();                                                                           \
    int32_t * addr_c = C;                                                                               \
    const int stride_c = TILE_N;                                                                        \
    MZERO(ACC);                                                                                         \
    AME_MLOAD_FENCE_BEFORE_LOAD(TAG);                                                                   \
    const int8_t * addr_a = A;                                                                          \
    MLA(TRA, addr_a, TILE_K);                                                                           \
    const int8_t * addr_b = B;                                                                          \
    MLB(TRB, addr_b, TILE_K);                                                                           \
    MMACC(ACC, TRA, TRB);                                                                               \
    MSC(ACC, addr_c, stride_c * 4);                                                                     \
    const unsigned long release_target = ggml_ame_sync_release();                                      \
    AME_TILE_PROGRESS_LOG("async_submit_done%s target=%lu", TAG, release_target);                     \
    return release_target;                                                                              \
}

DEFINE_AME_GEMM_TILE_SUBMIT(ggml_ame_gemm_tile_i8_i32_bT_submit, 64, 64, 64, acc0, tr0, tr1, "")
DEFINE_AME_GEMM_TILE_SUBMIT(ggml_ame_gemm_tile_i8_i32_bT_submit_alt, 64, 64, 64, acc1, tr2, tr3, "_alt")
DEFINE_AME_GEMM_TILE_SUBMIT(ggml_ame_gemm_tile_i8_i32_bT_128_64_128_submit, 128, 64, 128, acc0, tr0, tr1, "_128")
DEFINE_AME_GEMM_TILE_SUBMIT(ggml_ame_gemm_tile_i8_i32_bT_128_64_128_submit_alt, 128, 64, 128, acc1, tr2, tr3, "_128_alt")
