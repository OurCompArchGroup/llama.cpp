#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "arch-fallback.h"

#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#if defined(GGML_RV_MXFP4_PACKED) && defined(GGML_MXFP4_PHASE_PROFILE)
#error "GGML_RV_MXFP4_PACKED is incompatible with GGML_MXFP4_PHASE_PROFILE"
#endif

#if defined(GGML_MXFP4_SUPPLIED_BATCH8) && GGML_MXFP4_SUPPLIED_BATCH8
#if !defined(GGML_RV_MXFP4_PACKED) || !defined(GGML_AME_MXFP4_ADJACENT) || !GGML_AME_MXFP4_ADJACENT || \
    !defined(GGML_AME_MXFP4_REUSE_AMAX) || !GGML_AME_MXFP4_REUSE_AMAX
#error "GGML_MXFP4_SUPPLIED_BATCH8 requires the packed adjacent supplied-amax path"
#endif
#endif

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}

#if defined(GGML_MXFP4_PHASE_PROFILE)
#if !defined(__riscv_v) || __riscv_xlen != 64
#error "MXFP4 phase profiling requires RV64 RVV"
#endif
static _Thread_local struct ggml_mxfp4_phase_profile mxfp4_phase_profile;

void ggml_mxfp4_phase_profile_reset(void) {
    memset(&mxfp4_phase_profile, 0, sizeof(mxfp4_phase_profile));
    mxfp4_phase_profile.adjacent_tick_min = UINT64_MAX;
    for (int i = 0; i < 64; ++i) {
        const uint64_t begin = mxfp4_phase_tick(0.0f);
        const uint64_t end = mxfp4_phase_tick(0.0f);
        if (end - begin < mxfp4_phase_profile.adjacent_tick_min) {
            mxfp4_phase_profile.adjacent_tick_min = end - begin;
        }
    }
}

void ggml_mxfp4_phase_profile_get(struct ggml_mxfp4_phase_profile * result) {
    *result = mxfp4_phase_profile;
}

void ggml_mxfp4_phase_profile_rms(uint64_t stats, uint64_t copy, uint64_t apply) {
    ++mxfp4_phase_profile.rms_rows;
    mxfp4_phase_profile.rms_stats_cycles += stats;
    mxfp4_phase_profile.rms_copy_cycles += copy;
    mxfp4_phase_profile.rms_apply_cycles += apply;
}
#endif

#if defined(__riscv_v)
// Shared scale selection for the runtime quantizer and supplied-scale preflight.
// Keep the reduction, cast domain and scale expression identical at both callers.
static inline float mxfp4_block_amax(const float * values) {
    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4;) {
        const size_t vl = __riscv_vsetvl_e32m8(QK_MXFP4 - j);
        const vfloat32m8_t vx = __riscv_vle32_v_f32m8(values + j, vl);
        const vfloat32m8_t magnitude = __riscv_vfabs_v_f32m8(vx, vl);
        const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        const vfloat32m1_t maximum = __riscv_vfredmax_vs_f32m8_f32m1(magnitude, zero, vl);
        const float chunk_amax = __riscv_vfmv_f_s_f32m1_f32(maximum);
        if (amax < chunk_amax) amax = chunk_amax;
        j += (int) vl;
    }
    return amax;
}

static inline bool mxfp4_scale_cast_fallback(float amax) {
    return !isfinite(amax) || (amax > 0.0f && amax < 0x1p-125f);
}

static inline uint8_t mxfp4_scale_from_amax(float amax) {
    return amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;
}

static inline int mxfp4_threshold_search_eligible(uint8_t e, float d, float amax) {
    uint32_t search_flags;
    // Read, never clear/restore, the caller's accrued flags. FP inputs
    // keep the original scale work before this observation; the memory
    // clobber keeps the following lane loads after it. Check the actual
    // target code when changing this dependency or compiler options.
    __asm__ volatile ("frflags %0" : "=r" (search_flags) : "f" (d), "f" (amax) : "memory");
    // Bound every finite lane by 16*d explicitly. Positive FP32 bit order
    // gives this power-of-two bound without evaluating 16*d, which would
    // overflow for e252 even though that block must retain paired errors.
    const uint32_t bound_bits = (uint32_t) (e + 3) << 23;
    return (search_flags & 1u) && e <= 251 &&
        amax <= 0x1p127f && fp32_to_bits(amax) <= bound_bits;
}
#endif

#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
// One qualification body for normal input and transient RMS metadata. The
// public preflight always passes NULL; every scale/flag/admission rule below
// remains shared, including real log2f/floorf work before observing NX.
static bool mxfp4_prepare_supplied_scales(
    const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT scales, int64_t k,
    const uint32_t * GGML_RESTRICT normalized_amax_bits
) {
#else
bool ggml_mxfp4_prepare_supplied_scales(
    const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT scales, int64_t k
) {
#endif
#if defined(__riscv_v)
    if (x == NULL || scales == NULL || k <= 0 || k % QK_MXFP4 != 0 ||
        (uint64_t) k > SIZE_MAX / sizeof(float)) {
        return false;
    }
    const int64_t nb = k / QK_MXFP4;
    for (int64_t i = 0; i < nb; ++i) {
        const float * values = x + i * QK_MXFP4;
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
        float amax;
        if (normalized_amax_bits != NULL) {
            // Exact magnitude order also rejects every Inf/NaN. This value
            // came from THESE weighted lanes, not a new scale-selection rule.
            const uint32_t bits = normalized_amax_bits[i];
            if (bits >= UINT32_C(0x7f800000)) return false;
            __builtin_memcpy(&amax, &bits, sizeof(amax));
        } else {
#endif
        // Maximum-number reduction alone cannot reject NaN lanes. Inspect raw
        // bits first, without ordered FP predicates that could add scalar NV.
#if defined(GGML_MXFP4_PREFLIGHT_RVV_AMAX) && GGML_MXFP4_PREFLIGHT_RVV_AMAX
        // Unsigned absolute FP32 bits preserve finite magnitude order. One
        // reduction also detects every Inf/NaN without executing FP work.
        // Keep the original scale math and real accrued-flag gate below.
        uint32_t amax_bits = 0;
        for (int j = 0; j < QK_MXFP4;) {
            const size_t vl = __riscv_vsetvl_e32m8(QK_MXFP4 - j);
            const vfloat32m8_t raw = __riscv_vle32_v_f32m8(values + j, vl);
            const vuint32m8_t bits = __riscv_vreinterpret_v_f32m8_u32m8(raw);
            const vuint32m8_t magnitude = __riscv_vand_vx_u32m8(bits, 0x7fffffffu, vl);
            const vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, 1);
            const vuint32m1_t maximum = __riscv_vredmaxu_vs_u32m8_u32m1(magnitude, zero, vl);
            const uint32_t chunk = __riscv_vmv_x_s_u32m1_u32(maximum);
            if (amax_bits < chunk) amax_bits = chunk;
            j += (int) vl;
        }
        if (amax_bits >= 0x7f800000u) return false;
#if !defined(GGML_AME_MXFP4_REUSE_AMAX) || !GGML_AME_MXFP4_REUSE_AMAX
        float amax;
#endif
        __builtin_memcpy(&amax, &amax_bits, sizeof(amax));
#else
#if defined(GGML_MXFP4_PREFLIGHT_RVV_FINITE) && GGML_MXFP4_PREFLIGHT_RVV_FINITE
        for (int j = 0; j < QK_MXFP4;) {
            const size_t vl = __riscv_vsetvl_e32m8(QK_MXFP4 - j);
            const vfloat32m8_t raw = __riscv_vle32_v_f32m8(values + j, vl);
            const vuint32m8_t bits = __riscv_vreinterpret_v_f32m8_u32m8(raw);
            const vuint32m8_t exponent = __riscv_vand_vx_u32m8(bits, 0x7f800000u, vl);
            const vbool4_t nonfinite = __riscv_vmseq_vx_u32m8_b4(exponent, 0x7f800000u, vl);
            if (__riscv_vfirst_m_b4(nonfinite, vl) >= 0) {
                return false;
            }
            j += (int) vl;
        }
#else
        for (int j = 0; j < QK_MXFP4; ++j) {
            uint32_t bits;
#if defined(GGML_MXFP4_PREFLIGHT_INLINE_BITS) && GGML_MXFP4_PREFLIGHT_INLINE_BITS
            // Retain the alias-safe raw-bit load under AM -fno-builtin,
            // without an out-of-line four-byte memcpy for every lane.
            __builtin_memcpy(&bits, values + j, sizeof(bits));
#else
            memcpy(&bits, values + j, sizeof(bits));
#endif
            if ((bits & 0x7f800000u) == 0x7f800000u) {
                return false;
            }
        }
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
        amax = mxfp4_block_amax(values);
#else
        const float amax = mxfp4_block_amax(values);
#endif
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
        }
#endif
        if (mxfp4_scale_cast_fallback(amax)) {
            return false;
        }
        const uint8_t e = mxfp4_scale_from_amax(amax);
        if (e >= 253) {
            return false;
        }
        const float d = GGML_E8M0_TO_FP32_HALF(e);
        if (!mxfp4_threshold_search_eligible(e, d, amax)) {
            return false;
        }
        scales[i] = e;
    }
    return true;
#else
    UNUSED(x);
    UNUSED(scales);
    UNUSED(k);
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    UNUSED(normalized_amax_bits);
#endif
    return false;
#endif
}

#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
bool ggml_mxfp4_prepare_supplied_scales(
    const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT scales, int64_t k
) {
    return mxfp4_prepare_supplied_scales(x, scales, k, NULL);
}
#endif

#if defined(GGML_RV_MXFP4_PACKED) && !defined(GGML_MXFP4_PHASE_PROFILE)
#if !defined(__riscv_v) || __riscv_xlen != 64
#error "GGML_RV_MXFP4_PACKED requires the explicit matching R39 RV64 RVV production build"
#endif

static _Thread_local struct ggml_mxfp4_packed_stats mxfp4_packed_stats_state;

void ggml_mxfp4_packed_stats_reset(void) {
    memset(&mxfp4_packed_stats_state, 0, sizeof(mxfp4_packed_stats_state));
}

void ggml_mxfp4_packed_stats_get(struct ggml_mxfp4_packed_stats * result) {
    if (result != NULL) {
        *result = mxfp4_packed_stats_state;
    }
}

#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
void ggml_mxfp4_packed_stats_rms(uint64_t rows, uint64_t blocks,
    uint64_t stats, uint64_t apply, uint64_t scan, uint64_t reorder) {
    mxfp4_packed_stats_state.rms_rows += rows;
    mxfp4_packed_stats_state.rms_blocks += blocks;
    mxfp4_packed_stats_state.rms_stats_cycles += stats;
    mxfp4_packed_stats_state.rms_apply_cycles += apply;
    mxfp4_packed_stats_state.rms_scan_cycles += scan;
    mxfp4_packed_stats_state.rms_reorder_cycles += reorder;
}
#endif

// Existing accepted AM owner. The R39 production assembly consumes count
// contiguous K32 F32 blocks and one E8M0 scale per block, writes
// count * 16 adjacent-nibble bytes, and returns the final real fcsr.
extern uint64_t vector_mxfp4_pack_blocks(
    const float * values, uint8_t * adjacent, const uint8_t * scales,
    unsigned count);

enum { MXFP4_PACKED_BATCH = 8 };

#if defined(GGML_MXFP4_SUPPLIED_BATCH8) && GGML_MXFP4_SUPPLIED_BATCH8
static inline bool mxfp4_supplied_batch8_scales(const uint32_t * amax_bits, uint8_t * scales) {
    for (int offset = 0; offset < MXFP4_PACKED_BATCH; ++offset) {
        const uint32_t bits = amax_bits[offset];
        if (bits == 0u) {
            scales[offset] = 0;
        } else {
            const uint32_t exponent = bits >> 23;
            const uint32_t mantissa = bits & 0x007fffffu;
            if (exponent < 2u || exponent > 253u || mantissa < 0x100u || mantissa > 0x7fff00u) {
                return false;
            }
            scales[offset] = (uint8_t) (exponent - 2u);
        }
    }
    return true;
}
#endif

// PACK_MX orders byte j as elements 2*j and 2*j+1. block_mxfp4.qs uses
// byte j as elements j and j+16. Convert that existing packed result without
// re-running scale selection, error search or either encoder.
static void mxfp4_adjacent_to_split_half(
    block_mxfp4 * blocks, const uint8_t * adjacent, int count
) {
    for (int offset = 0; offset < count; ++offset) {
        const uint8_t * src = adjacent + offset * (QK_MXFP4 / 2);
        uint8_t * dst = blocks[offset].qs;
        for (int j = 0; j < QK_MXFP4 / 4; ++j) {
            const uint8_t lo = src[j];
            const uint8_t hi = src[j + QK_MXFP4 / 4];
            dst[2 * j + 0] = (uint8_t) ((lo & 0x0f) | ((hi & 0x0f) << 4));
            dst[2 * j + 1] = (uint8_t) ((lo >> 4) | (hi & 0xf0));
        }
    }
}

static bool quantize_row_mxfp4_packed(
    const float * x, block_mxfp4 * blocks, int64_t nb
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
    , uint8_t * adjacent_output
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    , const uint32_t * normalized_amax_bits
#endif
) {
    uint32_t incoming_flags;
#if defined(GGML_MXFP4_SUPPLIED_BATCH8) && GGML_MXFP4_SUPPLIED_BATCH8
    const bool supplied_batch8 = nb == MXFP4_PACKED_BATCH && x != NULL &&
        adjacent_output != NULL && normalized_amax_bits != NULL;
    if (supplied_batch8) {
        __asm__ volatile ("frcsr %0" : "=r" (incoming_flags) : : "memory");
    } else
#endif
    {
        __asm__ volatile ("frflags %0" : "=r" (incoming_flags) : : "memory");
    }
    // The accepted R39 caller preserves its original one-block failure path
    // when NV is already sticky. Let the unchanged scalar-threshold path run
    // instead, so this integration neither clears nor synthesizes flags.
    if (incoming_flags & 16u) {
        ++mxfp4_packed_stats_state.qualification_fallbacks;
        return false;
    }

    // Qualify the whole call before publishing packed bytes. The original
    // preflight remains authoritative for finite input, scale cast, e<=251
    // and the naturally accrued NX/amax/16*d threshold-search contract.
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
    const uint64_t qualify_begin = mxfp4_packed_phase_tick(0.0f, 0);
#endif
    for (int64_t i = 0; i < nb; i += MXFP4_PACKED_BATCH) {
        uint8_t scales[MXFP4_PACKED_BATCH];
        const int count = (int) (nb - i < MXFP4_PACKED_BATCH
            ? nb - i : MXFP4_PACKED_BATCH);
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
        if (
#if defined(GGML_MXFP4_SUPPLIED_BATCH8) && GGML_MXFP4_SUPPLIED_BATCH8
            !(supplied_batch8 && (incoming_flags & 1u) && ((incoming_flags >> 5) & 7u) < 4u &&
                mxfp4_supplied_batch8_scales(normalized_amax_bits, scales)) &&
#endif
            !mxfp4_prepare_supplied_scales(
                x + i * QK_MXFP4, scales, (int64_t) count * QK_MXFP4,
                normalized_amax_bits == NULL ? NULL : normalized_amax_bits + i)) {
#else
        if (!ggml_mxfp4_prepare_supplied_scales(
                x + i * QK_MXFP4, scales, (int64_t) count * QK_MXFP4)) {
#endif
            ++mxfp4_packed_stats_state.qualification_fallbacks;
            return false;
        }
        for (int offset = 0; offset < count; ++offset) {
            blocks[i + offset].e = scales[offset];
        }
    }
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
    // Refused qualifications retain their counters, but have no successful
    // qualification timing. Graph attribution requires zero fallbacks.
    mxfp4_packed_stats_state.qualification_cycles += mxfp4_packed_phase_tick(0.0f, 0) - qualify_begin;
#endif

    for (int64_t i = 0; i < nb; i += MXFP4_PACKED_BATCH) {
        uint8_t scales[MXFP4_PACKED_BATCH];
        uint8_t adjacent[MXFP4_PACKED_BATCH * QK_MXFP4 / 2];
        const int count = (int) (nb - i < MXFP4_PACKED_BATCH
            ? nb - i : MXFP4_PACKED_BATCH);
        for (int offset = 0; offset < count; ++offset) {
            scales[offset] = blocks[i + offset].e;
        }
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
        const uint64_t encode_begin = mxfp4_packed_phase_tick(0.0f, 0);
#endif
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
        uint8_t * const packed_output = adjacent_output == NULL ? adjacent :
            adjacent_output + i * (QK_MXFP4 / 2);
        const uint64_t fcsr = vector_mxfp4_pack_blocks(
            x + i * QK_MXFP4, packed_output, scales, (unsigned) count);
#else
        const uint64_t fcsr = vector_mxfp4_pack_blocks(
            x + i * QK_MXFP4, adjacent, scales, (unsigned) count);
#endif
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
        const uint64_t encode_end = mxfp4_packed_phase_tick(0.0f, fcsr);
        mxfp4_packed_stats_state.vector_pack_cycles += encode_end - encode_begin;
#endif
        if (fcsr & 16u) {
            ++mxfp4_packed_stats_state.packed_nv_fallbacks;
            return false;
        }
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
        const uint64_t reorder_begin = mxfp4_packed_phase_tick(0.0f, fcsr);
#endif
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
        if (adjacent_output == NULL)
#endif
        mxfp4_adjacent_to_split_half(blocks + i, adjacent, count);
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
        mxfp4_packed_stats_state.split_reorder_cycles +=
            mxfp4_packed_phase_tick(0.0f, (uint64_t) count) - reorder_begin;
#endif
    }
    ++mxfp4_packed_stats_state.packed_calls;
    mxfp4_packed_stats_state.packed_blocks += (uint64_t) nb;
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
    if (adjacent_output != NULL) {
        mxfp4_packed_stats_state.adjacent_blocks += (uint64_t) nb;
    }
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    if (normalized_amax_bits != NULL) {
        mxfp4_packed_stats_state.amax_reuse_blocks += (uint64_t) nb;
    }
#endif
    return true;
}
#endif

#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
// One body owns both the ordinary format and the opt-in packed sink. Failed
// packed qualification proceeds into the ORIGINAL fallback below exactly once.
static bool quantize_row_mxfp4_output(const float * GGML_RESTRICT x,
    void * GGML_RESTRICT y, int64_t k, uint8_t * GGML_RESTRICT adjacent_output
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    , const uint32_t * GGML_RESTRICT normalized_amax_bits
#endif
) {
#else
void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
#endif
#if defined(__riscv_v)
    assert(k % QK_MXFP4 == 0);
    block_mxfp4 * GGML_RESTRICT blocks = y;
    const int64_t nb = k / QK_MXFP4;
#if defined(GGML_RV_MXFP4_PACKED) && !defined(GGML_MXFP4_PHASE_PROFILE)
    if (nb > 0) {
        ++mxfp4_packed_stats_state.calls;
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
        if (adjacent_output != NULL) ++mxfp4_packed_stats_state.adjacent_calls;
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
        if (normalized_amax_bits != NULL) ++mxfp4_packed_stats_state.amax_reuse_calls;
#endif
        if (quantize_row_mxfp4_packed(x, blocks, nb, adjacent_output
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
                , normalized_amax_bits
#endif
        )) {
            return true;
#else
        if (quantize_row_mxfp4_packed(x, blocks, nb)) {
            return;
#endif
        }
    }
#endif
#if defined(GGML_MXFP4_PHASE_PROFILE)
    ++mxfp4_phase_profile.calls;
#endif
    for (int64_t i = 0; i < nb; ++i) {
        const float * values = x + i * QK_MXFP4;
#if defined(GGML_MXFP4_PHASE_PROFILE)
        const uint64_t phase_begin = mxfp4_phase_tick(0.0f);
#endif
        float amax = mxfp4_block_amax(values);
#if defined(GGML_MXFP4_PHASE_PROFILE)
        const uint64_t phase_amax = mxfp4_phase_tick(amax);
        // Keep the scale work after this boundary in diagnostic builds.
        __asm__ volatile ("" : "+f" (amax) : "r" (phase_amax) : "memory");
        ++mxfp4_phase_profile.blocks;
        mxfp4_phase_profile.amax_cycles += phase_amax - phase_begin;
#endif
        // Maximum-number reduction ignores NaN values; the unchanged ordered
        // search (or reference fallback) still accrues NV for their lanes.
        // Preserve final flags, not the time at which they become sticky.
        // Preserve the legacy reference outside the finite scale-cast domain.
        // This is not a new quantization/admission rule or an exponent clamp.
        if (mxfp4_scale_cast_fallback(amax)) {
            quantize_row_mxfp4_ref(values, blocks + i, QK_MXFP4);
#if defined(GGML_MXFP4_PHASE_PROFILE)
            mxfp4_phase_profile.fallback_cycles += mxfp4_phase_tick(amax) - phase_amax;
            ++mxfp4_phase_profile.fallback_blocks;
#endif
            continue;
        }
        const uint8_t e = mxfp4_scale_from_amax(amax);
        if (e >= 253) {
            quantize_row_mxfp4_ref(values, blocks + i, QK_MXFP4);
#if defined(GGML_MXFP4_PHASE_PROFILE)
            mxfp4_phase_profile.fallback_cycles += mxfp4_phase_tick(amax) - phase_amax;
            ++mxfp4_phase_profile.fallback_blocks;
#endif
            continue;
        }
        const float d = GGML_E8M0_TO_FP32_HALF(e);
        blocks[i].e = e;
        const int threshold_search = mxfp4_threshold_search_eligible(e, d, amax);
#if defined(GGML_MXFP4_PHASE_PROFILE)
        uint64_t phase_scale;
        __asm__ volatile ("rdcycle %0" : "=r" (phase_scale) :
            "f" (d), "r" (threshold_search) : "memory");
#endif
        uint8_t codes[QK_MXFP4];
        if (threshold_search) {
            // The admitted threshold search has fewer live vectors than the
            // paired-error path. Use m8 only here so a K32 block fits in one
            // strip at VLEN128; keep the paired-error register budget intact.
            for (int j = 0; j < QK_MXFP4;) {
                const size_t vl = __riscv_vsetvl_e32m8(QK_MXFP4 - j);
                const vfloat32m8_t vx = __riscv_vle32_v_f32m8(values + j, vl);
#if defined(GGML_RV_MXFP4_ENCODE)
                // Experimental custom3, not a standard RVV/Zames allocation.
                // Keep the original scale/NX/envelope gate above. Only replace
                // its lane search; paired errors, narrowing and packing stay.
                const vuint32m8_t scales = __riscv_vmv_v_x_u32m8(e, vl);
                vuint32m8_t index;
                // funct7=3 is funct6=000001 with vm=1. Same-width aliases are
                // legal; the two vector inputs establish e32m8 before issue.
                __asm__ volatile (".insn r 0x7b, 1, 3, %0, %1, %2"
                    : "=vr" (index) : "vr" (scales), "vr" (vx) : "memory");
#else
                vuint32m8_t index = __riscv_vmv_v_x_u32m8(0, vl);
                // Adjacent nonzero-center errors are exact by Sterbenz.
                // Strict midpoint comparisons retain the first minimum,
                // including code0. Removed finite errors can add only NX,
                // already sticky; real ordered comparisons retain NaN NV.
                const float midpoints[] = {0.5f, 1.5f, 2.5f, 3.5f, 5.0f, 7.0f, 10.0f};
                const vfloat32m8_t magnitude = __riscv_vfabs_v_f32m8(vx, vl);
                for (int candidate = 1; candidate < 8; ++candidate) {
                    const float threshold = midpoints[candidate - 1] * d;
                    const vbool4_t better = __riscv_vmfgt_vf_f32m8_b4(magnitude, threshold, vl);
                    index = __riscv_vmerge_vxm_u32m8(index, candidate, better, vl);
                }
                // Preserve global code0, including negative zero/tiny/ties.
                // Ordered comparisons also retain accrued NV for NaN lanes.
                const vbool4_t negative = __riscv_vmflt_vf_f32m8_b4(vx, 0.0f, vl);
                const vbool4_t nonzero = __riscv_vmsne_vx_u32m8_b4(index, 0, vl);
                const vbool4_t signed_winner = __riscv_vmand_mm_b4(negative, nonzero, vl);
                index = __riscv_vmerge_vvm_u32m8(
                    index, __riscv_vor_vx_u32m8(index, 8, vl), signed_winner, vl);
#endif
                const vuint16m4_t half = __riscv_vnsrl_wx_u16m4(index, 0, vl);
                const vuint8m2_t bytes = __riscv_vnsrl_wx_u8m2(half, 0, vl);
                __riscv_vse8_v_u8m2(codes + j, bytes, vl);
                j += (int) vl;
            }
        } else {
            for (int j = 0; j < QK_MXFP4;) {
                const size_t vl = __riscv_vsetvl_e32m4(QK_MXFP4 - j);
                const vfloat32m4_t vx = __riscv_vle32_v_f32m4(values + j, vl);
                vuint32m4_t index = __riscv_vmv_v_x_u32m4(0, vl);
                // Both errors must remain live: discarded opposite signs can
                // contribute exceptions outside the bounded NX-sticky domain.
                vfloat32m4_t best = __riscv_vfabs_v_f32m4(
                    __riscv_vfrsub_vf_f32m4(vx, 0.0f, vl), vl);
                for (int candidate = 1; candidate < 8; ++candidate) {
                    const float value = kvalues_mxfp4[candidate] * d;
                    const vfloat32m4_t positive = __riscv_vfabs_v_f32m4(
                        __riscv_vfrsub_vf_f32m4(vx, value, vl), vl);
                    const vfloat32m4_t negative = __riscv_vfabs_v_f32m4(
                        __riscv_vfrsub_vf_f32m4(vx, -value, vl), vl);
                    const vfloat32m4_t err = __riscv_vfmin_vv_f32m4(positive, negative, vl);
                    const vbool8_t negative_better =
                        __riscv_vmflt_vv_f32m4_b8(negative, positive, vl);
                    const vuint32m4_t positive_index = __riscv_vmv_v_x_u32m4(candidate, vl);
                    const vuint32m4_t candidate_index = __riscv_vmerge_vxm_u32m4(
                        positive_index, candidate + 8, negative_better, vl);
                    const vbool8_t better = __riscv_vmflt_vv_f32m4_b8(err, best, vl);
                    best = __riscv_vmerge_vvm_f32m4(best, err, better, vl);
                    index = __riscv_vmerge_vvm_u32m4(index, candidate_index, better, vl);
                }
                const vuint16m2_t half = __riscv_vnsrl_wx_u16m2(index, 0, vl);
                const vuint8m1_t bytes = __riscv_vnsrl_wx_u8m1(half, 0, vl);
                __riscv_vse8_v_u8m1(codes + j, bytes, vl);
                j += (int) vl;
            }
        }
#if defined(GGML_MXFP4_PHASE_PROFILE)
        const uint64_t phase_encode = mxfp4_phase_tick(d);
#endif
        for (int j = 0; j < QK_MXFP4 / 2;) {
            const size_t vl = __riscv_vsetvl_e8m1(QK_MXFP4 / 2 - j);
            const vuint8m1_t low = __riscv_vle8_v_u8m1(codes + j, vl);
            const vuint8m1_t high = __riscv_vle8_v_u8m1(codes + QK_MXFP4 / 2 + j, vl);
            const vuint8m1_t packed = __riscv_vor_vv_u8m1(
                low, __riscv_vsll_vx_u8m1(high, 4, vl), vl);
            __riscv_vse8_v_u8m1(blocks[i].qs + j, packed, vl);
            j += (int) vl;
        }
#if defined(GGML_MXFP4_PHASE_PROFILE)
        const uint64_t phase_pack = mxfp4_phase_tick(d);
        mxfp4_phase_profile.scale_cycles += phase_scale - phase_amax;
        mxfp4_phase_profile.encode_cycles += phase_encode - phase_scale;
        mxfp4_phase_profile.pack_cycles += phase_pack - phase_encode;
        mxfp4_phase_profile.threshold_blocks += threshold_search != 0;
        mxfp4_phase_profile.paired_blocks += threshold_search == 0;
#endif
    }
#else
    quantize_row_mxfp4_ref(x, y, k);
#endif
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
    if (adjacent_output != NULL && k > 0) {
        ++mxfp4_packed_stats_state.adjacent_fallbacks;
    }
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    if (normalized_amax_bits != NULL && k > 0) {
        ++mxfp4_packed_stats_state.amax_reuse_fallbacks;
    }
#endif
    return false;
#endif
}

#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    (void) quantize_row_mxfp4_output(x, y, k, NULL
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
        , NULL
#endif
    );
}

bool ggml_quantize_row_mxfp4_adjacent(const float * GGML_RESTRICT x,
    void * GGML_RESTRICT blocks, uint8_t * GGML_RESTRICT adjacent, int64_t k
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    , const uint32_t * GGML_RESTRICT normalized_amax_bits
#endif
) {
    return quantize_row_mxfp4_output(x, blocks, k, adjacent
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
        , normalized_amax_bits
#endif
    );
}
#endif

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q4_K * GGML_RESTRICT y = vy;
    quantize_row_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q5_K * GGML_RESTRICT y = vy;
    quantize_row_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq1_0 * GGML_RESTRICT y = vy;
    quantize_row_tq1_0_ref(x, y, k);
}

void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);

        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int sum = 0;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        for (size_t l = 0; l < 4; ++l) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[sizeof(x->qs)*5 + l*sizeof(x->qh) + j];
            }
        }

        sumf += (float) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumf += (float) sumi * d;
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const uint8_t  * GGML_RESTRICT sc = x[i].scales;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT signs = x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
}

void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

// ============================ 4-bit non-linear quants

void quantize_row_iq4_nl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row_iq4_xs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}
