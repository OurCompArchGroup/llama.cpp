#include "ame.h"

// BF16 GEMM using Zames proposal-12 instructions.
// C(MxN) = A(MxK) x B^T(NxK), where K is 32 BF16 elements (64 bytes).
// a_row_stride / b_row_stride are byte strides between consecutive rows, so the
// kernel can read the original tensors in place (direct read): CUTE matrix
// loads consume whole 64-byte granules, hence both the base addresses and the
// strides must be 64-byte aligned.  Callers must guarantee that (the mul_mat
// wrapper falls back to packed tiles otherwise).
//
// tile_n 是运行时 tile 宽度（MSETTILEN/MSC 都是寄存器操作数）：
// - GEMM 走 AME_TILE_N_BF16(=128)：这一版 RTL 的 tensor 宽度固定 128，
//   MMA 的 N 必须等于 Tensor_MN，否则 CDataController 断言失败；
// - FlashAttention 仍用 64（其 KV/PV 分块逻辑按 FA_N 组织）。
void ggml_ame_gemm_tile_bf16_fp32_bT(
    const ggml_bf16_t * A,
    ptrdiff_t a_row_stride,
    const ggml_bf16_t * B,
    ptrdiff_t b_row_stride,
    int tile_n,
    float * C
) {
    int tmp;
    MSETTILEM(tmp, AME_TILE_M);
    MSETTILEK(tmp, AME_TILE_K_BF16);
    MSETTILEN(tmp, tile_n);
    ggml_ame_config_bf16_fp32();

    MZERO(acc0);
    asm volatile("fence rw, rw" ::: "memory");

    MLA(tr0, A, a_row_stride);
    MLB(tr1, B, b_row_stride);
    MMACC(acc0, tr0, tr1);
    MSC(acc0, C, tile_n * (int) sizeof(float));

    ggml_ame_sync_release_acquire_store();
}

void ggml_ame_gemm_tile_bf16_fp32_bT_kloop(
    const ggml_bf16_t * a_tiles,
    ptrdiff_t a_tile_stride,
    ptrdiff_t a_row_stride,
    const ggml_bf16_t * b_tiles,
    ptrdiff_t b_tile_stride,
    ptrdiff_t b_row_stride,
    int k_tiles,
    int reg_pairs,
    int tile_n,
    float * C
) {
    if (k_tiles <= 0) {
        return;
    }

    int tmp;
    MSETTILEM(tmp, AME_TILE_M);
    MSETTILEK(tmp, AME_TILE_K_BF16);
    MSETTILEN(tmp, tile_n);
    ggml_ame_config_bf16_fp32();

    MZERO(acc0);
    asm volatile("fence rw, rw" ::: "memory");

    const ggml_bf16_t * addr_a = a_tiles;
    const ggml_bf16_t * addr_b = b_tiles;
    if (reg_pairs >= 2) {
        int kb = 0;
        for (; kb + 1 < k_tiles; kb += 2) {
            MLA(tr0, addr_a, a_row_stride);
            MLB(tr1, addr_b, b_row_stride);
            MMACC(acc0, tr0, tr1);
            addr_a = (const ggml_bf16_t *) ((const char *) addr_a + a_tile_stride);
            addr_b = (const ggml_bf16_t *) ((const char *) addr_b + b_tile_stride);

            MLA(tr2, addr_a, a_row_stride);
            MLB(tr3, addr_b, b_row_stride);
            MMACC(acc0, tr2, tr3);
            addr_a = (const ggml_bf16_t *) ((const char *) addr_a + a_tile_stride);
            addr_b = (const ggml_bf16_t *) ((const char *) addr_b + b_tile_stride);
        }
        if (kb < k_tiles) {
            MLA(tr0, addr_a, a_row_stride);
            MLB(tr1, addr_b, b_row_stride);
            MMACC(acc0, tr0, tr1);
        }
    } else {
        for (int kb = 0; kb < k_tiles; ++kb) {
            MLA(tr0, addr_a, a_row_stride);
            MLB(tr1, addr_b, b_row_stride);
            MMACC(acc0, tr0, tr1);
            addr_a = (const ggml_bf16_t *) ((const char *) addr_a + a_tile_stride);
            addr_b = (const ggml_bf16_t *) ((const char *) addr_b + b_tile_stride);
        }
    }

    MSC(acc0, C, tile_n * (int) sizeof(float));
    ggml_ame_sync_release_acquire_store();
}
