#include "ame.h"

// BF16 GEMM using Zames proposal-12 instructions.
// C(MxN) = A(MxK) x B^T(NxK), where K is 32 BF16 elements (64 bytes).
void ggml_ame_gemm_tile_bf16_fp32_bT(
    const ggml_bf16_t * A,
    const ggml_bf16_t * B,
    float * C
) {
    int tmp;
    MSETTILEM(tmp, AME_TILE_M);
    MSETTILEK(tmp, AME_TILE_K_BF16);
    MSETTILEN(tmp, AME_TILE_N);
    ggml_ame_config_bf16_fp32();

    MZERO(acc0);
    asm volatile("fence rw, rw" ::: "memory");

    MLA(tr0, A, AME_TILE_K);
    MLB(tr1, B, AME_TILE_K);
    MMACC(acc0, tr0, tr1);
    MSC(acc0, C, AME_TILE_N * (int) sizeof(float));

    ggml_ame_sync_release_acquire_store();
}

void ggml_ame_gemm_tile_bf16_fp32_bT_kloop(
    const ggml_bf16_t * a_tiles,
    ptrdiff_t a_tile_stride,
    const ggml_bf16_t * b_tiles,
    ptrdiff_t b_tile_stride,
    int k_tiles,
    int reg_pairs,
    float * C
) {
    if (k_tiles <= 0) {
        return;
    }

    int tmp;
    MSETTILEM(tmp, AME_TILE_M);
    MSETTILEK(tmp, AME_TILE_K_BF16);
    MSETTILEN(tmp, AME_TILE_N);
    ggml_ame_config_bf16_fp32();

    MZERO(acc0);
    asm volatile("fence rw, rw" ::: "memory");

    const ggml_bf16_t * addr_a = a_tiles;
    const ggml_bf16_t * addr_b = b_tiles;
    if (reg_pairs >= 2) {
        int kb = 0;
        for (; kb + 1 < k_tiles; kb += 2) {
            MLA(tr0, addr_a, AME_TILE_K);
            MLB(tr1, addr_b, AME_TILE_K);
            MMACC(acc0, tr0, tr1);
            addr_a = (const ggml_bf16_t *) ((const char *) addr_a + a_tile_stride);
            addr_b = (const ggml_bf16_t *) ((const char *) addr_b + b_tile_stride);

            MLA(tr2, addr_a, AME_TILE_K);
            MLB(tr3, addr_b, AME_TILE_K);
            MMACC(acc0, tr2, tr3);
            addr_a = (const ggml_bf16_t *) ((const char *) addr_a + a_tile_stride);
            addr_b = (const ggml_bf16_t *) ((const char *) addr_b + b_tile_stride);
        }
        if (kb < k_tiles) {
            MLA(tr0, addr_a, AME_TILE_K);
            MLB(tr1, addr_b, AME_TILE_K);
            MMACC(acc0, tr0, tr1);
        }
    } else {
        for (int kb = 0; kb < k_tiles; ++kb) {
            MLA(tr0, addr_a, AME_TILE_K);
            MLB(tr1, addr_b, AME_TILE_K);
            MMACC(acc0, tr0, tr1);
            addr_a = (const ggml_bf16_t *) ((const char *) addr_a + a_tile_stride);
            addr_b = (const ggml_bf16_t *) ((const char *) addr_b + b_tile_stride);
        }
    }

    MSC(acc0, C, AME_TILE_N * (int) sizeof(float));
    ggml_ame_sync_release_acquire_store();
}
