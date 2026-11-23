#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML CPU internal header

#if defined(GGML_RV_MXFP4_PACKED) && defined(GGML_MXFP4_PHASE_PROFILE)
#error "GGML_RV_MXFP4_PACKED is incompatible with GGML_MXFP4_PHASE_PROFILE"
#endif
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
#if !defined(GGML_RV_MXFP4_PACKED)
#error "packed phase profiling requires the production packed quantizer"
#endif
#endif
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
#if !defined(GGML_RV_MXFP4_PACKED)
#error "adjacent output requires the production packed quantizer"
#endif
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
#if !defined(GGML_AME_MXFP4_ADJACENT) || !GGML_AME_MXFP4_ADJACENT
#error "transient amax reuse requires the original adjacent-output entry"
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

#if defined(GGML_RV_MXFP4_PACKED)
// Diagnostic counters for the explicit matching R39 RV64 RVV production build.
// They observe the existing quantize_row_mxfp4 entry without changing its ABI.
struct ggml_mxfp4_packed_stats {
    uint64_t calls;
    uint64_t packed_calls;
    uint64_t packed_blocks;
    uint64_t qualification_fallbacks;
    uint64_t packed_nv_fallbacks;
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
    uint64_t adjacent_calls, adjacent_blocks, adjacent_fallbacks;
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    uint64_t amax_reuse_calls, amax_reuse_blocks, amax_reuse_fallbacks;
#endif
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
    uint64_t qualification_cycles, vector_pack_cycles, split_reorder_cycles;
    uint64_t rms_rows, rms_blocks;
    uint64_t rms_stats_cycles, rms_apply_cycles, rms_scan_cycles, rms_reorder_cycles;
#endif
};
// Reset/get only touch thread-local counters; neither changes fcsr/frm.
void ggml_mxfp4_packed_stats_reset(void);
void ggml_mxfp4_packed_stats_get(struct ggml_mxfp4_packed_stats * result);
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
// Same runtime quantizer and one real fallback, with its normal input/size
// preconditions. true: blocks[].e and adjacent[k/2] are valid; qs is untouched.
// false: the complete legacy blocks are valid; adjacent bytes are provisional
// and must not be consumed. This is NOT a request to quantize again.
bool ggml_quantize_row_mxfp4_adjacent(const float * GGML_RESTRICT x,
    void * GGML_RESTRICT blocks, uint8_t * GGML_RESTRICT adjacent, int64_t k
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
    // Internal same-call metadata ONLY: k/32 exact sign-cleared maxima of x,
    // produced by the RMS helper's two-rounded weighted lanes. No cached or
    // caller-supplied estimate; NULL retains the ordinary input preflight.
    , const uint32_t * GGML_RESTRICT normalized_amax_bits
#endif
);
#endif
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
// Diagnostic only: keep production dispatch, math and real sticky flags.
// Both operands are compiler dependencies; the timestamp is not a fence or
// an overhead correction. Instrumented scheduling must be reported as such.
static inline uint64_t mxfp4_packed_phase_tick(float fp_dependency, uint64_t int_dependency) {
    uint64_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r" (cycles) :
        "f" (fp_dependency), "r" (int_dependency) : "memory");
    return cycles;
}
void ggml_mxfp4_packed_stats_rms(uint64_t rows, uint64_t blocks,
    uint64_t stats, uint64_t apply, uint64_t scan, uint64_t reorder);
#endif
#endif

// RVV-only supplied-scale preflight; no payload quantization or SF publication.
// x is a contiguous, nonoverlapping row of k>0 F32 elements, k divisible by 32;
// scales has k/32 bytes. Reject invalid arguments and return false on non-RVV.
// true: every block is finite, in the original scale-cast domain, with e<=251
// and the runtime quantizer's NX-sticky/amax/16*d threshold-search eligibility.
// Uses the canonical amax/log2f/floorf scale work, which may accrue FP flags;
// reads current fflags only, never seeds, clears or restores them or changes frm.
// false: caller must use the whole-row CPU fallback; partial SF bytes MUST NOT
// be consumed. Does not make the reference's unsupported input domain safe.
bool ggml_mxfp4_prepare_supplied_scales(
    const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT scales, int64_t k);

#if defined(GGML_MXFP4_PHASE_PROFILE)
// Diagnostic static builds only. Counters belong to the calling thread.
// Timestamp dependencies and counter updates perturb scheduling/register use.
struct ggml_mxfp4_phase_profile {
    uint64_t calls, blocks, threshold_blocks, paired_blocks, fallback_blocks;
    uint64_t amax_cycles, scale_cycles, encode_cycles, pack_cycles, fallback_cycles;
    uint64_t adjacent_tick_min;
    uint64_t rms_rows, rms_stats_cycles, rms_copy_cycles, rms_apply_cycles;
};
// Shared diagnostic timestamp; never present in ordinary builds. Preserve the
// original quantizer's FP dependency and compiler memory boundary.
static inline uint64_t mxfp4_phase_tick(float dependency) {
    uint64_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r" (cycles) : "f" (dependency) : "memory");
    return cycles;
}
void ggml_mxfp4_phase_profile_reset(void);
void ggml_mxfp4_phase_profile_get(struct ggml_mxfp4_phase_profile * result);
void ggml_mxfp4_phase_profile_rms(uint64_t stats, uint64_t copy, uint64_t apply);
#endif

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// Dot product
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// Generic implementation
void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
