#include "ame.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu/vec.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

bool ggml_ame_mxfp4_native_domain(const void * src, int64_t nblocks) {
    if (src == NULL || nblocks <= 0 || (uint64_t) nblocks > SIZE_MAX / sizeof(block_mxfp4)) {
        return false;
    }
    const block_mxfp4 * blocks = (const block_mxfp4 *) src;
    for (int64_t b = 0; b < nblocks; ++b) {
        if (blocks[b].e == 255) {
            return false;
        }
    }
    return true;
}

bool ggml_ame_pack_mxfp4_blocks(
    uint8_t * payload, uint8_t * scales, const void * src, int64_t nblocks
) {
    if (payload == NULL || scales == NULL || src == NULL || nblocks <= 0 ||
        (uint64_t) nblocks > SIZE_MAX / sizeof(block_mxfp4)) {
        return false;
    }
    const block_mxfp4 * blocks = (const block_mxfp4 *) src;
    for (int64_t b = 0; b < nblocks; ++b) {
        // Do not decode/requantize: this also preserves the negative-zero code
        // and the raw scale byte, including 255 for negative-format tests.
        for (int j = 0; j < QK_MXFP4 / 4; ++j) {
            const uint8_t lo = blocks[b].qs[2*j];
            const uint8_t hi = blocks[b].qs[2*j + 1];
            payload[b*16 + j]     = (lo & 15) | ((hi & 15) << 4);
            payload[b*16 + j + 8] = (lo >> 4) | (hi & 0xf0);
        }
        scales[b] = blocks[b].e;
    }
    return true;
}

static bool ame_rmsnorm_source_supported(
    const float * src, const float * gamma, int64_t rows, int64_t K,
    size_t src_stride, float eps, int tile_mn
) {
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    if (src == NULL || gamma == NULL ||
        !ggml_ame_mx_shape_supported(tile_mn, rows, K, tile_mn) ||
        (eps_bits & UINT32_C(0x7fffffff)) >= UINT32_C(0x7f800000) || !(eps > 0.0f)) {
        return false;
    }
    const size_t row_bytes = (size_t) K * sizeof(float);
    const int64_t nb = K / QK_MXFP4;
    if (src_stride < row_bytes || src_stride % _Alignof(float) != 0 ||
        (uint64_t) (rows - 1) > (SIZE_MAX - row_bytes) / src_stride ||
        (uint64_t) rows > SIZE_MAX / (size_t) nb / 16) {
        return false;
    }
    return true;
}

static bool ame_rmsnorm_block(
    float * normalized, const float * x, const float * gamma, float r
) {
#if defined(__riscv_v_intrinsic)
    // Retain BOTH F32 roundings. Only fuse the scratch traffic, never r*gamma
    // or the ordered RMS reduction. The quantizer still consumes this block.
    for (size_t lane = 0; lane < QK_MXFP4;) {
        const size_t vl = __riscv_vsetvl_e32m8(QK_MXFP4 - lane);
        const vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + lane, vl);
        const vfloat32m8_t scaled = __riscv_vfmul_vf_f32m8(vx, r, vl);
        const vfloat32m8_t vg = __riscv_vle32_v_f32m8(gamma + lane, vl);
        const vfloat32m8_t weighted = __riscv_vfmul_vv_f32m8(scaled, vg, vl);
        __riscv_vse32_v_f32m8(normalized + lane, weighted, vl);
        lane += vl;
    }
#else
    memcpy(normalized, x, QK_MXFP4 * sizeof(float));
    // Preserve separate F32 rounding at RMSNorm and gamma multiply.
    ggml_vec_scale_f32(QK_MXFP4, normalized, r);
    ggml_vec_mul_f32(QK_MXFP4, normalized, normalized, gamma);
#endif
    uint32_t amax_bits = 0;
    for (int lane = 0; lane < QK_MXFP4; ++lane) {
        uint32_t bits;
        memcpy(&bits, normalized + lane, sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
            return false;
        }
        bits &= UINT32_C(0x7fffffff);
        if (bits > amax_bits) amax_bits = bits;
    }
    // Protect the legacy float-to-uint8 scale cast before invoking it.
    // 0x01000000 is 2^-125; zero is also legal. No FP/NX seeding here.
    return amax_bits == 0 || amax_bits >= UINT32_C(0x01000000);
}

bool ggml_ame_rmsnorm_mxfp4_preflight(
    float * row_scales, const float * src, const float * gamma,
    int64_t rows, int64_t K, size_t src_stride, float eps, int tile_mn,
    struct ggml_ame_rmsnorm_first_panel * first
) {
    if (row_scales == NULL ||
        !ame_rmsnorm_source_supported(src, gamma, rows, K, src_stride, eps, tile_mn) ||
        ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->from_float == NULL) {
        return false;
    }
    if (first != NULL && (first->payload == NULL || first->scales == NULL ||
        !ggml_ame_mx_tile_k_supported(AME_MCFG_MXFP4, first->tile_k) || K % first->tile_k != 0)) {
        return false;
    }
    // Keep only one reciprocal RMS per row, never the normalized tensor.
    // Validate every row/block before a caller can issue its first MMACC.
    for (int64_t row = 0; row < rows; ++row) {
        const float * x = (const float *) ((const char *) src + (size_t) row * src_stride);
        const float r = ggml_vec_rms_norm_scale_f32(K, x, eps);
        if (!(r > 0.0f) || !isfinite(r)) return false;
        row_scales[row] = r;
    }
    if (first != NULL) {
        uint64_t begin = 0, end = 0;
        first->prepare_cycles = 0;
#if defined(__riscv)
        if (first->profile) __asm__ volatile("rdcycle %0" : "=r"(begin) : : "memory");
#endif
        // Same producer does the first panel's normalized-domain checks and
        // packing. Later rows can still reject: this output is discard-only
        // until the ENTIRE preflight succeeds. Never issue AMU work here.
        if (!ggml_ame_rmsnorm_pack_mxfp4(
                first->payload, first->scales, src, gamma, tile_mn, K,
                src_stride, eps, first->tile_k, tile_mn, row_scales)) return false;
#if defined(__riscv)
        if (first->profile) __asm__ volatile("rdcycle %0" : "=r"(end) : : "memory");
#endif
        first->prepare_cycles = end - begin;
    }
    for (int64_t row = first == NULL ? 0 : tile_mn; row < rows; ++row) {
        const float * x = (const float *) ((const char *) src + (size_t) row * src_stride);
        for (int64_t k = 0; k < K; k += QK_MXFP4) {
            float normalized[QK_MXFP4];
            if (!ame_rmsnorm_block(normalized, x + k, gamma + k, row_scales[row])) return false;
        }
    }
    return true;
}

bool ggml_ame_rmsnorm_pack_mxfp4(
    uint8_t * payload, uint8_t * scales, const float * src, const float * gamma,
    int64_t rows, int64_t K, size_t src_stride, float eps, int tile_k, int tile_mn,
    const float * row_scales
) {
    if (payload == NULL || scales == NULL ||
        !ame_rmsnorm_source_supported(src, gamma, rows, K, src_stride, eps, tile_mn) ||
        !ggml_ame_mx_tile_k_supported(AME_MCFG_MXFP4, tile_k) || K % tile_k != 0) {
        return false;
    }
    const int64_t nb = K / QK_MXFP4;
    // Use the actual configured CPU entry, including its real scalar fallback.
    // Do not duplicate its encoder or manufacture incoming NX.
    const ggml_from_float_t quantize = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->from_float;
    if (quantize == NULL) {
        return false;
    }
    // Reuse ordinary AME activation preparation's bounded K32 batching:
    // independent per-block scales, with only a K32 or K64 group contiguous
    // in the packed destination. Never materialize a normalized full row.
    enum { batch_blocks = 8 };
    float normalized[batch_blocks * QK_MXFP4];
    block_mxfp4 encoded[batch_blocks];
    const int groups = tile_k / QK_MXFP4;
    for (int64_t row = 0; row < rows; ++row) {
        const float * x = (const float *) ((const char *) src + (size_t) row * src_stride);
        const float r = row_scales == NULL ? ggml_vec_rms_norm_scale_f32(K, x, eps) : row_scales[row];
        if (!(r > 0.0f) || !isfinite(r)) {
            return false;
        }
        for (int64_t kb = 0; kb < nb; kb += batch_blocks) {
            const int count = (int) (nb - kb < batch_blocks ? nb - kb : batch_blocks);
            for (int offset = 0; offset < count; ++offset) {
                if (!ame_rmsnorm_block(normalized + offset * QK_MXFP4,
                        x + (kb + offset) * QK_MXFP4,
                        gamma + (kb + offset) * QK_MXFP4, r)) {
                    return false;
                }
            }
            quantize(normalized, encoded, count * QK_MXFP4);
            if (!ggml_ame_mxfp4_native_domain(encoded, count)) {
                return false;
            }
            for (int offset = 0; offset < count; offset += groups) {
                const size_t index = ggml_ame_mx_packed_block_index(
                    row, kb + offset, nb, tile_k, tile_mn);
                if (!ggml_ame_pack_mxfp4_blocks(
                        payload + index * 16, scales + index, encoded + offset, groups)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool ggml_ame_mxfp8_f32_domain(const float * src, int64_t nvalues, int format) {
    if (src == NULL || !ggml_ame_mxfp8_format_supported(format) ||
        nvalues <= 0 || nvalues % AME_MXFP4_K != 0 ||
        (uint64_t) nvalues > SIZE_MAX / sizeof(float)) {
        return false;
    }
    for (int64_t k = 0; k < nvalues; ++k) {
        if (!isfinite(src[k])) {
            return false;
        }
    }
    return true;
}

bool ggml_ame_mxfp8_native_domain(
    const uint8_t * payload, const uint8_t * scales, int64_t nblocks, int format
) {
    if (payload == NULL || scales == NULL || nblocks <= 0 ||
        (uint64_t) nblocks > SIZE_MAX / AME_MXFP4_K || nblocks > INT64_MAX / AME_MXFP4_K ||
        !ggml_ame_mxfp8_format_supported(format)) {
        return false;
    }
    for (int64_t b = 0; b < nblocks; ++b) {
        if (scales[b] == 255) {
            return false;
        }
        for (int k = 0; k < AME_MXFP4_K; ++k) {
            const unsigned magnitude = payload[b * AME_MXFP4_K + k] & 127;
            if (format == AME_MCFG_MXFP8_E4M3 ? magnitude == 127 : magnitude >= 124) {
                return false;
            }
        }
    }
    return true;
}

// Integer RNE after exact binary exponent adjustment; independent of host
// rounding mode. E4M3FN has finite 0x78..0x7e and no Inf, E5M2 IEEE specials.
static uint8_t ame_encode_mxfp8(float value, int power, int format) {
    const int mantissa = format == AME_MCFG_MXFP8_E4M3 ? 3 : 2;
    const int bias = format == AME_MCFG_MXFP8_E4M3 ? 7 : 15;
    const unsigned sign = signbit(value) ? 128 : 0;
    if (value == 0.0f) {
        return (uint8_t) sign;
    }
    int exponent;
    frexp(fabs((double) value), &exponent);
    const int target_exp = exponent - 1 - power + bias;
    const int unit_power = target_exp > 0 ? exponent - 1 - power - mantissa :
        1 - bias - mantissa;
    const double significand = ldexp(fabs((double) value), -power - unit_power);
    unsigned rounded = (unsigned) significand;
    const double remainder = significand - rounded;
    rounded += remainder > 0.5 || (remainder == 0.5 && (rounded & 1));
    const unsigned code = (target_exp > 0 ? (unsigned) (target_exp - 1) << mantissa : 0) + rounded;
    return (uint8_t) (sign | code);
}

bool ggml_ame_pack_mxfp8_f32(
    uint8_t * payload, uint8_t * scales, const float * src, int64_t nvalues, int format
) {
    // Validate the entire source before writing even the first block. Callers
    // also preflight whole weight/activation tensors before native allocation.
    if (payload == NULL || scales == NULL || !ggml_ame_mxfp8_f32_domain(src, nvalues, format)) {
        return false;
    }
    const double max_value = format == AME_MCFG_MXFP8_E4M3 ? 448.0 : 57344.0;
    const int max_exponent = format == AME_MCFG_MXFP8_E4M3 ? 8 : 15;
    for (int64_t base = 0; base < nvalues; base += AME_MXFP4_K) {
        float amax = 0.0f;
        for (int k = 0; k < AME_MXFP4_K; ++k) {
            amax = fmaxf(amax, fabsf(src[base + k]));
        }
        int power = 0; // Canonical zero block has unit scale, preserving +/-0.
        if (amax != 0.0f) {
            power = ilogbf(amax) - max_exponent;
            power = power < -127 ? -127 : power;
            if (ldexp((double) amax, -power) > max_value) {
                power++;
            }
        }
        // Every finite F32 tensor fits the legal E8M0 range; the lower clamp
        // permits tiny values to round to signed zero, never wraps an exponent.
        assert(power >= -127 && power <= 127);
        scales[base / AME_MXFP4_K] = (uint8_t) (power + 127);
        for (int k = 0; k < AME_MXFP4_K; ++k) {
            payload[base + k] = ame_encode_mxfp8(src[base + k], power, format);
        }
    }
    return true;
}

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
