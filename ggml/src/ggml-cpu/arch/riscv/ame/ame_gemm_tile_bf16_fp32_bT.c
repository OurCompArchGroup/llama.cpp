#include "ame.h"

#ifndef AME_MLOAD_FENCE
#define AME_MLOAD_FENCE 1
#endif

// BF16 GEMM using RISC-V AME instructions.
// C(MxN) = A(MxK) x B^T(NxK), where B is transposed in memory.
void ggml_ame_gemm_tile_bf16_fp32_bT(
    const ggml_bf16_t * A,
    const ggml_bf16_t * B,
    float * C
) {
    asm volatile("msyncreset tok0" ::: "memory");

    const int TILE_M = AME_TILE_M;
    const int TILE_K = AME_TILE_K_BF16;
    const int TILE_N = AME_TILE_N;
    const int TILE_K_BYTES = TILE_K * (int) sizeof(ggml_bf16_t);

    int tmp;
    MSETTILEM(tmp, TILE_M);
    MSETTILEK(tmp, TILE_K_BYTES);
    MSETTILEN(tmp, TILE_N);

    float * addr_c = C;
    const int stride_c = TILE_N * (int) sizeof(float);

    MZERO_ACC(acc0);
#if AME_MLOAD_FENCE
    asm volatile("fence rw, rw" ::: "memory");
#endif

    const ggml_bf16_t * addr_a = A;
    MLAE16(tr0, addr_a, TILE_K_BYTES);

    const ggml_bf16_t * addr_b = B;
    MLBE16(tr1, addr_b, TILE_K_BYTES);

    MFMACC_S_BF16(acc0, tr0, tr1);

    MSCE32(acc0, addr_c, stride_c);

    asm volatile("mrelease tok0" ::: "memory");
    int acquire_target = 1;
    asm volatile("macquire %0,tok0" :: "r"(acquire_target) : "memory");
}
