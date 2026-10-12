#include "ame-xsai-fusion.h"

#include "ggml-impl.h"

#include <cmath>
#include <cstring>

namespace {

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

}  // namespace

xsai_add_rms_fusion_decision_t ame_xsai_match_add_rms_norm(
    const ggml_cgraph * graph,
    const ggml_tensor * rms_norm,
    int fused_implementation_available,
    DLTensor inputs[3],
    DLTensor * output,
    xsai_fused_add_rms_norm_params_t * params,
    xsai_op_call_t * call) {
    xsai_add_rms_fusion_facts_t facts = {};
    facts.struct_size = sizeof(facts);
    facts.fused_implementation_available = fused_implementation_available ? 1 : 0;

    const ggml_tensor * intermediate =
        rms_norm != nullptr && rms_norm->op == GGML_OP_RMS_NORM ? rms_norm->src[0] : nullptr;
    const ggml_tensor * weight =
        rms_norm != nullptr ? rms_norm->src[1] : nullptr;
    const bool pattern = intermediate != nullptr && intermediate->op == GGML_OP_ADD &&
        intermediate->src[0] != nullptr && intermediate->src[1] != nullptr &&
        weight != nullptr;
    facts.pattern_is_add_then_rms_norm = pattern ? 1 : 0;
    if (!pattern) {
        return xsai_add_rms_fusion_decide(&facts);
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

    const xsai_add_rms_fusion_decision_t decision = xsai_add_rms_fusion_decide(&facts);
    if (decision != XSAI_FUSION_ACCEPT || inputs == nullptr || output == nullptr ||
        params == nullptr || call == nullptr) {
        return decision;
    }

    bind_tensor(&inputs[0], lhs);
    bind_tensor(&inputs[1], rhs);
    bind_tensor(&inputs[2], weight);
    bind_tensor(output, rms_norm);
    *params = {};
    params->struct_size = sizeof(*params);
    params->eps = eps;
    *call = {};
    call->struct_size = sizeof(*call);
    call->abi_version = XSAI_BLAS_ABI_VERSION;
    call->op = XSAI_OP_FUSED_ADD_RMS_NORM;
    call->inputs = inputs;
    call->input_count = 3;
    call->outputs = output;
    call->output_count = 1;
    call->op_attrs = params;
    return XSAI_FUSION_ACCEPT;
}
