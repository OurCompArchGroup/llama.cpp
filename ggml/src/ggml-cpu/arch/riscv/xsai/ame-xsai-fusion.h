#pragma once

#include <xsai_blas.h>

#include <stddef.h>
#include <stdint.h>

struct ggml_cgraph;
struct ggml_tensor;

/*
 * ADD -> RMS_NORM pattern check. This does not run fused math, does not
 * select a kernel, and does not rewrite the ggml graph. On ACCEPT the
 * caller-owned DLTensor slots and xsai call describe a fused op. A miss
 * leaves those outputs untouched so the two original ops stay in place.
 */
xsai_add_rms_fusion_decision_t ame_xsai_match_add_rms_norm(
    const struct ggml_cgraph * graph,
    const struct ggml_tensor * rms_norm,
    int fused_implementation_available,
    DLTensor inputs[3],
    DLTensor * output,
    xsai_fused_add_rms_norm_params_t * params,
    xsai_op_call_t * call);
