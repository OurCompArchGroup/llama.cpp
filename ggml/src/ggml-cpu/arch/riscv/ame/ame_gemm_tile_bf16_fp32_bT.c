#include "ame.h"

void ggml_ame_gemm_tile_mx_fp32_bT_kloop(
    const uint8_t * a, const uint8_t * sa,
    const uint8_t * b, const uint8_t * sb, int64_t k_tiles, float * c, int format, int tile_k, int tile_mn,
    ggml_ame_overlap_fn overlap_fn, void * overlap_opaque,
    enum ggml_ame_overlap_schedule overlap_schedule
) {
    const size_t block_bytes = ggml_ame_mx_block_bytes(format);
    const int transpose_c = format == AME_MCFG_MXFP4 && tile_mn == AME_TILE_M_MAX &&
        ggml_ame_transposed_c_epilogue_enabled();
    GGML_ASSERT(ggml_ame_mx_tile_k_supported(format, tile_k));
    GGML_ASSERT(tile_mn == 64 || tile_mn == AME_TILE_M_MAX);
    GGML_ASSERT(overlap_schedule == GGML_AME_OVERLAP_LAST_K ||
        overlap_schedule == GGML_AME_OVERLAP_EACH_K);
    const size_t blocks_per_row = (size_t) tile_k / AME_MXFP4_K;
    const size_t row_bytes = block_bytes * blocks_per_row;
    int tmp;
    MSETTILEM(tmp, tile_mn);
    MSETTILEN(tmp, tile_mn);
    MSETTILEK(tmp, tile_k);
    // mcfg table1:0x18 E4M3FN,0x17 E5M2,0x16 E2M1, not AMU type enum.
    MSETCFG(mcfg0, format);
    MSETCFG(mcfg1, format);
    MSETCFG(mcfg4, AME_MCFG_FP32);

    unsigned long saved_mfrm;
    asm volatile("csrrw %0, 0x809, zero" : "=r"(saved_mfrm) : : "memory");
    MZERO(acc0);
    asm volatile("fence rw, rw" ::: "memory");

    // The AMU may consume address GPRs after scalar issue. Keep every load
    // base live until acquire, as in the existing INT8 kloop store path.
    register const uint8_t * a_live  __asm__("s7")  = a;
    register const uint8_t * b_live  __asm__("s8")  = b;
    register const uint8_t * sa_live __asm__("s9")  = sa;
    register const uint8_t * sb_live __asm__("s10") = sb;
    register float * c_live         __asm__("s11") = c;
    for (int64_t kb = 0; kb < k_tiles; ++kb) {
        MLA(tr0, a_live, row_bytes);
        MLB(tr1, b_live, row_bytes);
        // Frozen proposal-12 encoding: rd is the scale index, NOT a GPR
        // destination. tr0 pairs with sa0, tr1 with sb1.
        asm volatile(
            ".insn r 0x2b, 1, 0x18, x0, %0, %2\n\t"
            ".insn r 0x2b, 1, 0x58, x1, %1, %2"
            : : "r"(sa_live), "r"(sb_live), "r"(blocks_per_row) : "memory");
        ggml_ame_sync_release_acquire_store();
        // Exactly ONE instruction for K64. The frozen hardware contract adds
        // block0 then block1 with per-MX32 RNE internally. No software dot/add.
        MMACC(acc0, tr0, tr1);
        // Input bases/scales above have already been acquired. Only compute
        // is pending here: reuse the canonical INT8 pre-MSC overlap ordering,
        // never run RVV memory traffic alongside a pending matrix store.
        if (overlap_fn != NULL &&
                (overlap_schedule == GGML_AME_OVERLAP_EACH_K || kb + 1 == k_tiles)) {
            asm volatile("" ::: "memory");
            overlap_fn(overlap_opaque); // CPU producer only, no AMU instructions
            asm volatile("fence rw, rw" ::: "memory");
        }
        ggml_ame_sync_release_acquire_store();
        asm volatile("" : "+r"(a_live), "+r"(b_live), "+r"(sa_live), "+r"(sb_live) : : "memory");
        a_live  += (size_t) tile_mn * row_bytes;
        b_live  += (size_t) tile_mn * row_bytes;
        sa_live += (size_t) tile_mn * blocks_per_row;
        sb_live += (size_t) tile_mn * blocks_per_row;
    }
    // Keep acc0 across K blocks: a CPU store/add would introduce a second
    // rounding and would not implement the frozen per-block-RNE contract.
    if (transpose_c) {
        MSCT(acc0, c_live, tile_mn * (int) sizeof(float));
    } else {
        MSC(acc0, c_live, tile_mn * (int) sizeof(float));
    }
    ggml_ame_sync_release_acquire_store();
    asm volatile("" : "+r"(c_live) : : "memory");
    asm volatile("csrw 0x809, %0" : : "r"(saved_mfrm) : "memory");
}

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
