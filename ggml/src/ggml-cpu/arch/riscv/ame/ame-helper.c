#include "ame.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Quantize a row of F32 values to Q8_0 format
void ggml_ame_quantize_row_f32_to_q8_0(const float * x, void * vy, int k) {
    block_q8_0 * y = (block_q8_0 *) vy;
    assert(k % 32 == 0);
    const int nb = k / 32;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            const float v = x[i * 32 + j];
            const float av = fabsf(v);
            if (av > amax) amax = av;
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 32; j++) {
            const float x0 = x[i * 32 + j] * id;
            y[i].qs[j] = roundf(x0);
        }
    }
}

// Repacks Q4_0 blocks into AME-optimized format
void ggml_ame_repack_q4_0(
    void * dst,
    const void * src,
    int64_t nblocks
) {
    const block_q4_0 * restrict src_blocks = (const block_q4_0 *) src;
    block_q4_0_ame * restrict dst_blocks = (block_q4_0_ame *) dst;

    for (int64_t i = 0; i < nblocks; i++) {
        dst_blocks[i].d = src_blocks[i].d;

        for (int j = 0; j < 16; j++) {
            uint8_t v = src_blocks[i].qs[j];
            dst_blocks[i].qs[j]      = (int8_t) (v & 0x0F) - 8;
            dst_blocks[i].qs[j + 16] = (int8_t) ((v >> 4) & 0x0F) - 8;
        }
    }
}

static inline void ggml_ame_quantize_64_f32_to_q8(const float * x, block_q8_ame64 * y) {
    float amax = 0.0f;
    for (int j = 0; j < AME_Q8_PACK_K; ++j) {
        const float av = fabsf(x[j]);
        if (av > amax) {
            amax = av;
        }
    }

    const float delta = amax / 127.0f;
    const float inv_delta = delta ? 1.0f / delta : 0.0f;
    y->d = GGML_FP32_TO_FP16(delta);

    for (int j = 0; j < AME_Q8_PACK_K; ++j) {
        y->qs[j] = roundf(x[j] * inv_delta);
    }
}

// Repacks MXFP4 blocks into AME-optimized format
// block_mxfp4 stores 32 E2M1 values packed into 16 bytes + uint8 shared exponent.
// The repacked format unpacks the 4-bit values through E2M1-to-int8 lookup,
// yielding int8 values, and stores the combined scale as FP16.
void ggml_ame_repack_mxfp4(
    void * dst,
    const void * src,
    int64_t nblocks
) {
    // E2M1 doubled values (mirrors kvalues_mxfp4 in ggml-common.h)
    static const int8_t kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    const block_mxfp4 * restrict src_blocks = (const block_mxfp4 *) src;
    block_mxfp4_ame * restrict dst_blocks = (block_mxfp4_ame *) dst;

    for (int64_t i = 0; i < nblocks; i++) {
        const float d = GGML_E8M0_TO_FP32_HALF(src_blocks[i].e);
        dst_blocks[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            dst_blocks[i].qs[j]                = kvalues[src_blocks[i].qs[j] & 0x0F];
            dst_blocks[i].qs[j + QK_MXFP4 / 2] = kvalues[src_blocks[i].qs[j] >>   4];
        }
    }
}

void ggml_ame_repack_q8_0_to_ame64(
    void * dst,
    const void * src,
    int64_t nrows,
    int64_t k
) {
    const block_q8_0 * restrict src_rows = (const block_q8_0 *) src;
    block_q8_ame64 * restrict dst_rows = (block_q8_ame64 *) dst;

    assert(k % QK8_0 == 0);

    const int64_t nb32 = k / QK8_0;
    const int64_t nb64 = (k + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const block_q8_0 * src_row = src_rows + row * nb32;
        block_q8_ame64 * dst_row = dst_rows + row * nb64;

        for (int64_t b64 = 0; b64 < nb64; ++b64) {
            float tmp[AME_Q8_PACK_K];
            memset(tmp, 0, sizeof(tmp));

            const int64_t blk0 = b64 * 2;
            if (blk0 < nb32) {
                for (int j = 0; j < QK8_0; ++j) {
                    tmp[j] = GGML_FP16_TO_FP32(src_row[blk0].d) * src_row[blk0].qs[j];
                }
            }
            if (blk0 + 1 < nb32) {
                for (int j = 0; j < QK8_0; ++j) {
                    tmp[QK8_0 + j] = GGML_FP16_TO_FP32(src_row[blk0 + 1].d) * src_row[blk0 + 1].qs[j];
                }
            }

            ggml_ame_quantize_64_f32_to_q8(tmp, &dst_row[b64]);
        }
    }
}
