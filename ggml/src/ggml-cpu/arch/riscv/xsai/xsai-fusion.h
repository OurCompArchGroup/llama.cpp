#pragma once

#include <xsai_blas.h>

#include <stddef.h>
#include <stdint.h>

struct ggml_cgraph;
struct ggml_tensor;

/*
 * Fusion decision vocabulary. The unified operator ABI carries only the
 * facts; the policy lives in the adapter.
 */
typedef int32_t xsai_add_rms_fusion_decision_t;
enum {
    XSAI_FUSION_ACCEPT = 0,
    XSAI_FUSION_NOT_PATTERN = 1,
    XSAI_FUSION_EXTRA_CONSUMER = 2,
    XSAI_FUSION_DTYPE_SHAPE_LAYOUT = 3,
    XSAI_FUSION_ALIAS = 4,
    XSAI_FUSION_EPS_ROUNDING = 5,
    XSAI_FUSION_NO_IMPLEMENTATION = 6,
};

/*
 * ADD -> RMS_NORM pattern check. This does not run fused math, does not
 * select a kernel, and does not rewrite the ggml graph. Availability is probed
 * with xsai_op_plan_create() for XSAI_OP_FUSED_ADD_RMS_NORM on the same call
 * metadata. On ACCEPT the caller-owned DLTensor slots and xsai call describe
 * a fused op. A miss leaves those outputs untouched so the two original ops
 * stay in place.
 */
xsai_add_rms_fusion_decision_t xsai_match_add_rms_norm(
    const struct ggml_cgraph * graph,
    const struct ggml_tensor * rms_norm,
    DLTensor inputs[3],
    DLTensor * output,
    xsai_fused_add_rms_norm_params_t * params,
    xsai_op_call_t * call);
