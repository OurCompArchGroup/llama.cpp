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
            dst_blocks[i].qs[2 * j]     = (int8_t) (v & 0x0F) - 8;
            dst_blocks[i].qs[2 * j + 1] = (int8_t) ((v >> 4) & 0x0F) - 8;
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

void ggml_ame_repack_q8_0_to_ame64_whole_k(
    void * dst,
    const void * src,
    int64_t nrows,
    int64_t k
) {
    const block_q8_0 * restrict src_rows = (const block_q8_0 *) src;
    block_q8_ame64 * restrict dst_rows = (block_q8_ame64 *) dst;

    assert(k > 0 && k % QK8_0 == 0);

    const int64_t nb32 = k / QK8_0;
    const int64_t nb64 = (k + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const block_q8_0 * src_row = src_rows + row * nb32;
        block_q8_ame64 * dst_row = dst_rows + row * nb64;
        float amax = 0.0f;

        for (int64_t b32 = 0; b32 < nb32; ++b32) {
            const float d = GGML_FP16_TO_FP32(src_row[b32].d);
            for (int j = 0; j < QK8_0; ++j) {
                const float av = fabsf(d * src_row[b32].qs[j]);
                if (av > amax) {
                    amax = av;
                }
            }
        }

        const float delta = amax / 127.0f;
        const float inv_delta = delta ? 1.0f / delta : 0.0f;
        const ggml_fp16_t delta_fp16 = GGML_FP32_TO_FP16(delta);

        for (int64_t b64 = 0; b64 < nb64; ++b64) {
            block_q8_ame64 * out = &dst_row[b64];
            out->d = delta_fp16;

            for (int j = 0; j < AME_Q8_PACK_K; ++j) {
                const int64_t index = b64 * AME_Q8_PACK_K + j;
                if (index >= k) {
                    out->qs[j] = 0;
                    continue;
                }

                const block_q8_0 * in = &src_row[index / QK8_0];
                const float value = GGML_FP16_TO_FP32(in->d) * in->qs[index % QK8_0];
                float quantized = roundf(value * inv_delta);
                if (quantized > 127.0f) {
                    quantized = 127.0f;
                } else if (quantized < -127.0f) {
                    quantized = -127.0f;
                }
                out->qs[j] = (int8_t) quantized;
            }
        }
    }
}
