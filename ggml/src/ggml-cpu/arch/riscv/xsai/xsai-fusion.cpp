#include "xsai-fusion.h"

#include "ggml-impl.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace {

/* Fusion facts are adapter policy now; the ABI publishes no facts struct. */
struct xsai_add_rms_fusion_facts_t {
    uint32_t struct_size;
    int32_t pattern_is_add_then_rms_norm;
    int32_t intermediate_consumer_count;
    int32_t dtype_shape_layout_match;
    int32_t alias_safe;
    int32_t eps_rounding_match;
    int32_t fused_implementation_available;
};

int64_t nelements(const ggml_tensor * tensor) {
    int64_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] < 0) {
            return -1;
        }
        if (tensor->ne[i] != 0 && n > INT64_MAX / tensor->ne[i]) {
            return -1;
        }
        n *= tensor->ne[i];
    }
    return n;
}

bool same_ne(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs->ne[i] != rhs->ne[i]) {
            return false;
        }
    }
    return true;
}

bool contiguous_f32(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) {
        return false;
    }
    size_t next = sizeof(float) * (size_t) tensor->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] < 0 || tensor->ne[i - 1] < 0) {
            return false;
        }
        if (tensor->ne[i] != 1) {
            if (tensor->nb[i] != next) {
                return false;
            }
            next *= (size_t) tensor->ne[i];
        }
    }
    return true;
}

size_t nbytes(const ggml_tensor * tensor) {
    const int64_t n = nelements(tensor);
    if (n < 0) {
        return 0;
    }
    return (size_t) n * sizeof(float);
}

bool ranges_overlap(const void * lhs, size_t lhs_n, const void * rhs, size_t rhs_n) {
    if (lhs == nullptr || rhs == nullptr || lhs_n == 0 || rhs_n == 0) {
        return false;
    }
    const uintptr_t lhs_begin = (uintptr_t) lhs;
    const uintptr_t rhs_begin = (uintptr_t) rhs;
    if (lhs_n > UINTPTR_MAX - lhs_begin || rhs_n > UINTPTR_MAX - rhs_begin) {
        return true;
    }
    return lhs_begin < rhs_begin + rhs_n && rhs_begin < lhs_begin + lhs_n;
}

bool exact_alias(const void * lhs, size_t lhs_n, const void * rhs, size_t rhs_n) {
    return lhs != nullptr && lhs == rhs && lhs_n == rhs_n && lhs_n != 0;
}

int consumer_count(const ggml_cgraph * graph, const ggml_tensor * value) {
    if (graph == nullptr || value == nullptr || graph->nodes == nullptr) {
        return 0;
    }
    int count = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node == nullptr) {
            continue;
        }
        for (int src = 0; src < GGML_MAX_SRC; ++src) {
            if (node->src[src] == value) {
                ++count;
            }
        }
    }
    if (value->flags & GGML_TENSOR_FLAG_OUTPUT) {
        ++count;
    }
    return count;
}

bool alias_safe(const ggml_tensor * output, const ggml_tensor * lhs,
                const ggml_tensor * rhs, const ggml_tensor * weight) {
    const size_t out_n = nbytes(output);
    const size_t lhs_n = nbytes(lhs);
    const size_t rhs_n = nbytes(rhs);
    const size_t weight_n = nbytes(weight);
    const void * out = output->data;
    if (ranges_overlap(out, out_n, weight->data, weight_n)) {
        return false;
    }
    const bool lhs_hit = ranges_overlap(out, out_n, lhs->data, lhs_n);
    const bool rhs_hit = ranges_overlap(out, out_n, rhs->data, rhs_n);
    if (lhs_hit && !exact_alias(out, out_n, lhs->data, lhs_n)) {
        return false;
    }
    if (rhs_hit && !exact_alias(out, out_n, rhs->data, rhs_n)) {
        return false;
    }
    if (ranges_overlap(lhs->data, lhs_n, rhs->data, rhs_n) &&
        !exact_alias(lhs->data, lhs_n, rhs->data, rhs_n)) {
        return false;
    }
    return true;
}

void bind_tensor(DLTensor * slot, const ggml_tensor * tensor) {
    *slot = DLTensor{};
    slot->data = tensor->data;
    slot->device = {kDLCPU, 0};
    slot->ndim = GGML_MAX_DIMS;
    slot->dtype = {kDLFloat, 32, 1};
    slot->shape = const_cast<int64_t *>(tensor->ne);
    slot->strides = nullptr;
    slot->byte_offset = 0;
}

xsai_add_rms_fusion_decision_t decide(const xsai_add_rms_fusion_facts_t & facts) {
    if (!facts.pattern_is_add_then_rms_norm) {
        return XSAI_FUSION_NOT_PATTERN;
    }
    if (facts.intermediate_consumer_count != 1) {
        return XSAI_FUSION_EXTRA_CONSUMER;
    }
    if (!facts.dtype_shape_layout_match) {
        return XSAI_FUSION_DTYPE_SHAPE_LAYOUT;
    }
    if (!facts.alias_safe) {
        return XSAI_FUSION_ALIAS;
    }
    if (!facts.eps_rounding_match) {
        return XSAI_FUSION_EPS_ROUNDING;
    }
    if (!facts.fused_implementation_available) {
        return XSAI_FUSION_NO_IMPLEMENTATION;
    }
    return XSAI_FUSION_ACCEPT;
}

/*
 * The library owns the decision on whether the op has a usable kernel. The
 * answer depends only on the call metadata, so one successful probe per
 * fingerprint is enough; data pointers and workspace are not part of it.
 */
std::mutex g_probe_mu;
std::unordered_set<uint64_t> g_probe_ok;

bool implementation_available(const xsai_op_call_t * call) {
    const uint64_t key = xsai_op_call_fingerprint(call);
    if (key == 0) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_probe_mu);
        if (g_probe_ok.find(key) != g_probe_ok.end()) {
            return true;
        }
    }
    xsai_op_plan_t * plan = nullptr;
    const xsai_status_t status = xsai_op_plan_create(call, &plan);
    xsai_op_plan_destroy(plan);
    if (status != XSAI_STATUS_SUCCESS) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_probe_mu);
    g_probe_ok.insert(key);
    return true;
}

}  // namespace

xsai_add_rms_fusion_decision_t xsai_match_add_rms_norm(
    const ggml_cgraph * graph,
    const ggml_tensor * rms_norm,
    DLTensor inputs[3],
    DLTensor * output,
    xsai_fused_add_rms_norm_params_t * params,
    xsai_op_call_t * call) {
    xsai_add_rms_fusion_facts_t facts = {};
    facts.struct_size = sizeof(facts);

    const ggml_tensor * intermediate =
        rms_norm != nullptr && rms_norm->op == GGML_OP_RMS_NORM ? rms_norm->src[0] : nullptr;
    const ggml_tensor * weight =
        rms_norm != nullptr ? rms_norm->src[1] : nullptr;
    const bool pattern = intermediate != nullptr && intermediate->op == GGML_OP_ADD &&
        intermediate->src[0] != nullptr && intermediate->src[1] != nullptr &&
        weight != nullptr;
    facts.pattern_is_add_then_rms_norm = pattern ? 1 : 0;
    if (!pattern) {
        return decide(facts);
    }

    const ggml_tensor * lhs = intermediate->src[0];
    const ggml_tensor * rhs = intermediate->src[1];
    facts.intermediate_consumer_count = consumer_count(graph, intermediate);

    const int64_t n = nelements(intermediate);
    const bool layout = contiguous_f32(lhs) && contiguous_f32(rhs) &&
        contiguous_f32(weight) && contiguous_f32(intermediate) &&
        contiguous_f32(rms_norm) && same_ne(lhs, rhs) &&
        same_ne(lhs, intermediate) && same_ne(lhs, rms_norm) &&
        n >= 0 && nelements(weight) == n;
    facts.dtype_shape_layout_match = layout ? 1 : 0;

    float eps = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));
    facts.eps_rounding_match = std::isfinite(eps) && eps >= 0.0f ? 1 : 0;
    facts.alias_safe = alias_safe(rms_norm, lhs, rhs, weight) ? 1 : 0;

    /* Probe on locals so a rejected match leaves the caller slots untouched. */
    DLTensor probe_inputs[3];
    DLTensor probe_output;
    bind_tensor(&probe_inputs[0], lhs);
    bind_tensor(&probe_inputs[1], rhs);
    bind_tensor(&probe_inputs[2], weight);
    bind_tensor(&probe_output, rms_norm);
    xsai_fused_add_rms_norm_params_t probe_params = {};
    probe_params.struct_size = sizeof(probe_params);
    probe_params.abi_version = XSAI_BLAS_ABI_VERSION;
    probe_params.eps = eps;
    xsai_op_call_t probe_call = {};
    probe_call.struct_size = sizeof(probe_call);
    probe_call.abi_version = XSAI_BLAS_ABI_VERSION;
    probe_call.op = XSAI_OP_FUSED_ADD_RMS_NORM;
    probe_call.inputs = probe_inputs;
    probe_call.input_count = 3;
    probe_call.outputs = &probe_output;
    probe_call.output_count = 1;
    probe_call.op_params = &probe_params;
    facts.fused_implementation_available = implementation_available(&probe_call) ? 1 : 0;

    const xsai_add_rms_fusion_decision_t decision = decide(facts);
    if (decision != XSAI_FUSION_ACCEPT || inputs == nullptr || output == nullptr ||
        params == nullptr || call == nullptr) {
        return decision;
    }

    inputs[0] = probe_inputs[0];
    inputs[1] = probe_inputs[1];
    inputs[2] = probe_inputs[2];
    *output = probe_output;
    *params = probe_params;
    *call = probe_call;
    call->inputs = inputs;
    call->outputs = output;
    call->op_params = params;
    return XSAI_FUSION_ACCEPT;
}
