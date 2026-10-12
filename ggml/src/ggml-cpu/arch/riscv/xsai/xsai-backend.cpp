#include "xsai-backend.h"

#include "ame-xsai-fusion.h"
#include "ame-xsai-observe.h"
#include "ame-xsai-plan.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "traits.h"

#if defined(GGML_USE_RV_AME)
#include "../ame/ame-backend.h"
#endif

#include <xsai_blas.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

struct HostTensorGeneration {
    ggml_backend_buffer_t buffer = nullptr;
    uint64_t generation = 0;
};

struct HostGenerationState {
    uint64_t next_generation = 1;
    std::unordered_map<const ggml_tensor *, HostTensorGeneration> tensors;
    std::unordered_map<ggml_backend_buffer_t, uint64_t> buffers;
};

HostGenerationState & host_state() {
    static HostGenerationState state;
    return state;
}

const ggml_cgraph * g_graph = nullptr;
std::atomic<int> g_report_done{0};

enum class RouteKind {
    CPU,
    Q8,
    ELEMENT,
    UNARY,
    SOFTMAX,
    FUSED_RMS,
};

struct RouteNote {
    enum ggml_op op;
    const char * name;
    const char * xsai_or_cpu;
};

/* One row per ggml op the default llama graph can emit, plus the default. */
const RouteNote k_route_notes[] = {
    { GGML_OP_MUL_MAT, "MUL_MAT",
      "xsai GEMM_I8_F32 for compact Q8_0 x F32 rank-1; other matmul CPU before write" },
    { GGML_OP_MUL_MAT_ID, "MUL_MAT_ID", "CPU before write" },
    { GGML_OP_FLASH_ATTN_EXT, "FLASH_ATTN_EXT",
      "CPU before write (xsai flash is I8; the default graph is decomposed attention)" },
    { GGML_OP_SOFT_MAX, "SOFT_MAX",
      "xsai SOFTMAX when fp32, contiguous, no mask, no sink, max_bias 0; else CPU before write" },
    { GGML_OP_RMS_NORM, "RMS_NORM",
      "xsai FUSED_ADD_RMS_NORM only when the fusion matcher accepts; else CPU before write" },
    { GGML_OP_ADD, "ADD", "xsai ADD when same-shape contiguous fp32; broadcast CPU before write" },
    { GGML_OP_MUL, "MUL", "xsai MUL when same-shape contiguous fp32; broadcast CPU before write" },
    { GGML_OP_DIV, "DIV", "xsai DIV when same-shape contiguous fp32; else CPU before write" },
    { GGML_OP_UNARY, "UNARY", "xsai SILU/NEG when contiguous fp32; other unary CPU before write" },
    { GGML_OP_ROPE, "ROPE", "CPU before write" },
    { GGML_OP_GET_ROWS, "GET_ROWS", "CPU before write" },
    { GGML_OP_SCALE, "SCALE", "CPU before write" },
    { GGML_OP_CPY, "CPY", "CPU before write" },
    { GGML_OP_CONT, "CONT", "CPU before write" },
    { GGML_OP_RESHAPE, "RESHAPE", "CPU before write" },
    { GGML_OP_VIEW, "VIEW", "CPU before write" },
    { GGML_OP_PERMUTE, "PERMUTE", "CPU before write" },
    { GGML_OP_TRANSPOSE, "TRANSPOSE", "CPU before write" },
    { GGML_OP_GLU, "GLU", "CPU before write" },
    { GGML_OP_NORM, "NORM", "CPU before write" },
    { GGML_OP_NONE, "unregistered", "CPU before write" },
};

std::atomic<uint64_t> g_xsai_count[GGML_OP_COUNT];
std::atomic<uint64_t> g_cpu_count[GGML_OP_COUNT];

bool backend_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * value = std::getenv("XSAI_GGML_BACKEND");
        const bool off = value != nullptr &&
            (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
             std::strcmp(value, "off") == 0 || std::strcmp(value, "no") == 0);
        cached = off ? 0 : 1;
        if (cached) {
            ame_xsai_observe_set_enabled(1);
        }
    }
    return cached != 0;
}

void count_route(enum ggml_op op, bool xsai) {
    if (op < 0 || op >= GGML_OP_COUNT) {
        return;
    }
    if (xsai) {
        g_xsai_count[op].fetch_add(1, std::memory_order_relaxed);
    } else {
        g_cpu_count[op].fetch_add(1, std::memory_order_relaxed);
    }
}

bool dense_f32(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) {
        return false;
    }
    size_t next = sizeof(float) * (size_t) tensor->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] < 0) {
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

bool same_ne(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs->ne[i] != rhs->ne[i]) {
            return false;
        }
    }
    return true;
}

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

bool partial_overlap(const void * lhs, size_t lhs_n, const void * rhs, size_t rhs_n) {
    if (lhs == nullptr || rhs == nullptr || lhs_n == 0 || rhs_n == 0) {
        return false;
    }
    const uintptr_t lhs_begin = (uintptr_t) lhs;
    const uintptr_t rhs_begin = (uintptr_t) rhs;
    if (lhs_n > UINTPTR_MAX - lhs_begin || rhs_n > UINTPTR_MAX - rhs_begin) {
        return true;
    }
    const bool overlap = lhs_begin < rhs_begin + rhs_n && rhs_begin < lhs_begin + lhs_n;
    if (!overlap) {
        return false;
    }
    return lhs != rhs || lhs_n != rhs_n;
}

void bind_f32_1d(DLTensor * slot, void * data, int64_t * shape, int64_t n) {
    *shape = n;
    *slot = DLTensor{};
    slot->data = data;
    slot->device = {kDLCPU, 0};
    slot->ndim = 1;
    slot->dtype = {kDLFloat, 32, 1};
    slot->shape = shape;
}

struct GeneralKey {
    int32_t xsai_op;
    int32_t ndim;
    int64_t shape[4];
    float attr0;
    float attr1;
    uint32_t flags;
};

bool q8_compact(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q8_0 || tensor->ne[0] <= 0 ||
        tensor->ne[1] < 0) {
        return false;
    }
    const int64_t blocks = (tensor->ne[0] + 31) / 32;
    return tensor->nb[1] == (size_t) blocks * 34u;
}

bool q8_plan_op(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_MUL_MAT && op->type == GGML_TYPE_F32 &&
        op->src[0] != nullptr && op->src[1] != nullptr &&
        op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1]->type == GGML_TYPE_F32;
}

bool ame_owns_q8(const ggml_tensor * op) {
#if defined(GGML_USE_RV_AME)
    return q8_plan_op(op) && op->src[0]->buffer != nullptr &&
        op->src[0]->buffer->buft == ggml_backend_cpu_riscv_ame_buffer_type();
#else
    GGML_UNUSED(op);
    return false;
#endif
}

ame_xsai_q8_key q8_key(const ggml_tensor * op, uint64_t generation) {
    ame_xsai_q8_key key = {};
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * activation = op->src[1];
    key.k = weight->ne[0];
    key.n = weight->ne[1];
    key.m = activation->ne[1];
    key.act_nb0 = activation->nb[0];
    key.act_nb1 = activation->nb[1];
    key.act_nb2 = activation->nb[2];
    key.act_nb3 = activation->nb[3];
    key.weight_nb0 = weight->nb[0];
    key.weight_nb1 = weight->nb[1];
    key.weight_identity = weight;
    xsai_backend_q8_refresh(weight, generation);
    key.prepared_weight = ame_xsai_weight_plan(weight);
    key.weight_nb2 = weight->nb[2];
    key.weight_nb3 = weight->nb[3];
    key.dst_nb0 = op->nb[0];
    key.dst_nb1 = op->nb[1];
    key.dst_nb2 = op->nb[2];
    key.dst_nb3 = op->nb[3];
    key.src0_ne2 = weight->ne[2];
    key.src0_ne3 = weight->ne[3];
    key.src1_ne2 = activation->ne[2];
    key.src1_ne3 = activation->ne[3];
    key.dst_ne2 = op->ne[2];
    key.dst_ne3 = op->ne[3];
    key.weight_generation = generation;
    key.explicit_options = 0;
    return key;
}

xsai_op_type_t element_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD: return XSAI_OP_ADD;
        case GGML_OP_MUL: return XSAI_OP_MUL;
        case GGML_OP_DIV: return XSAI_OP_DIV;
        default: return -1;
    }
}

xsai_op_type_t unary_xsai_op(const ggml_tensor * op) {
    const int kind = ((const int32_t *) op->op_params)[0];
    if (kind == GGML_UNARY_OP_SILU) {
        return XSAI_OP_SILU;
    }
    if (kind == GGML_UNARY_OP_NEG) {
        return XSAI_OP_NEG;
    }
    return -1;
}

bool element_eligible(const ggml_tensor * op, int64_t * n_out) {
    if (element_op(op->op) < 0 || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    if (!dense_f32(op) || !dense_f32(op->src[0]) || !dense_f32(op->src[1]) ||
        !same_ne(op, op->src[0]) || !same_ne(op, op->src[1])) {
        return false;
    }
    const int64_t n = nelements(op);
    if (n < 0) {
        return false;
    }
    const size_t bytes = (size_t) n * sizeof(float);
    if (partial_overlap(op->data, bytes, op->src[0]->data, bytes) ||
        partial_overlap(op->data, bytes, op->src[1]->data, bytes)) {
        return false;
    }
    *n_out = n;
    return true;
}

bool unary_eligible(const ggml_tensor * op, xsai_op_type_t * xsai_op, int64_t * n_out) {
    if (op->op != GGML_OP_UNARY || op->src[0] == nullptr) {
        return false;
    }
    const xsai_op_type_t mapped = unary_xsai_op(op);
    if (mapped < 0 || !dense_f32(op) || !dense_f32(op->src[0]) || !same_ne(op, op->src[0])) {
        return false;
    }
    const int64_t n = nelements(op);
    if (n < 0) {
        return false;
    }
    const size_t bytes = (size_t) n * sizeof(float);
    if (partial_overlap(op->data, bytes, op->src[0]->data, bytes)) {
        return false;
    }
    *xsai_op = mapped;
    *n_out = n;
    return true;
}

bool softmax_eligible(const ggml_tensor * op, float * scale_out, int64_t * rows_out, int64_t * cols_out) {
    if (op->op != GGML_OP_SOFT_MAX || op->src[0] == nullptr || op->src[1] != nullptr ||
        op->src[2] != nullptr) {
        return false;
    }
    float scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, (const float *) op->op_params + 0, sizeof(scale));
    std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(max_bias));
    if (max_bias != 0.0f || !dense_f32(op) || !dense_f32(op->src[0]) || !same_ne(op, op->src[0])) {
        return false;
    }
    if (op->ne[0] <= 0) {
        return false;
    }
    const int64_t n = nelements(op);
    if (n <= 0 || n % op->ne[0] != 0) {
        return false;
    }
    *scale_out = scale;
    *cols_out = op->ne[0];
    *rows_out = n / op->ne[0];
    return true;
}

struct BuiltCall {
    GeneralKey key;
    bool ok = false;
    xsai_op_call_t call = {};
    DLTensor inputs[3] = {};
    DLTensor output = {};
    int64_t shapes[4] = {};
    xsai_softmax_params_t softmax = {};
    xsai_fused_add_rms_norm_params_t fused = {};
};

bool build_element_call(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    int64_t n = 0;
    if (!element_eligible(op, &n)) {
        return false;
    }
    std::memset(&built->key, 0, sizeof(built->key));
    built->key.xsai_op = element_op(op->op);
    built->key.ndim = 1;
    built->key.shape[0] = n;
    bind_f32_1d(&built->inputs[0], with_data ? op->src[0]->data : nullptr, &built->shapes[0], n);
    bind_f32_1d(&built->inputs[1], with_data ? op->src[1]->data : nullptr, &built->shapes[1], n);
    bind_f32_1d(&built->output, with_data ? op->data : nullptr, &built->shapes[2], n);
    built->call = {};
    built->call.struct_size = sizeof(built->call);
    built->call.abi_version = XSAI_BLAS_ABI_VERSION;
    built->call.op = built->key.xsai_op;
    built->call.input_count = 2;
    built->call.inputs = built->inputs;
    built->call.output_count = 1;
    built->call.outputs = &built->output;
    built->ok = true;
    return true;
}

bool build_unary_call(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    xsai_op_type_t xsai_op = -1;
    int64_t n = 0;
    if (!unary_eligible(op, &xsai_op, &n)) {
        return false;
    }
    std::memset(&built->key, 0, sizeof(built->key));
    built->key.xsai_op = xsai_op;
    built->key.ndim = 1;
    built->key.shape[0] = n;
    built->key.flags = (uint32_t) ((const int32_t *) op->op_params)[0];
    bind_f32_1d(&built->inputs[0], with_data ? op->src[0]->data : nullptr, &built->shapes[0], n);
    bind_f32_1d(&built->output, with_data ? op->data : nullptr, &built->shapes[1], n);
    built->call = {};
    built->call.struct_size = sizeof(built->call);
    built->call.abi_version = XSAI_BLAS_ABI_VERSION;
    built->call.op = xsai_op;
    built->call.input_count = 1;
    built->call.inputs = built->inputs;
    built->call.output_count = 1;
    built->call.outputs = &built->output;
    built->ok = true;
    return true;
}

bool build_softmax_call(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    float scale = 1.0f;
    int64_t rows = 0;
    int64_t cols = 0;
    if (!softmax_eligible(op, &scale, &rows, &cols)) {
        return false;
    }
    std::memset(&built->key, 0, sizeof(built->key));
    built->key.xsai_op = XSAI_OP_SOFTMAX;
    built->key.ndim = 2;
    built->key.shape[0] = rows;
    built->key.shape[1] = cols;
    built->key.attr0 = scale;
    built->shapes[0] = rows;
    built->shapes[1] = cols;
    built->inputs[0] = {};
    built->inputs[0].data = with_data ? op->src[0]->data : nullptr;
    built->inputs[0].device = {kDLCPU, 0};
    built->inputs[0].ndim = 2;
    built->inputs[0].dtype = {kDLFloat, 32, 1};
    built->inputs[0].shape = built->shapes;
    built->output = built->inputs[0];
    built->output.data = with_data ? op->data : nullptr;
    built->output.shape = built->shapes + 2;
    built->shapes[2] = rows;
    built->shapes[3] = cols;
    built->softmax = {};
    built->softmax.struct_size = sizeof(built->softmax);
    built->softmax.axis = 1;
    built->softmax.scale = scale;
    built->call = {};
    built->call.struct_size = sizeof(built->call);
    built->call.abi_version = XSAI_BLAS_ABI_VERSION;
    built->call.op = XSAI_OP_SOFTMAX;
    built->call.input_count = 1;
    built->call.inputs = built->inputs;
    built->call.output_count = 1;
    built->call.outputs = &built->output;
    built->call.op_attrs = &built->softmax;
    built->ok = true;
    return true;
}

bool build_fused_call(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    if (op->op != GGML_OP_RMS_NORM) {
        return false;
    }
    const xsai_add_rms_fusion_decision_t decision = ame_xsai_match_add_rms_norm(
        g_graph, op, 1, built->inputs, &built->output, &built->fused, &built->call);
    if (decision != XSAI_FUSION_ACCEPT) {
        return false;
    }
    if (!with_data) {
        for (int i = 0; i < 3; ++i) {
            built->inputs[i].data = nullptr;
        }
        built->output.data = nullptr;
    }
    std::memset(&built->key, 0, sizeof(built->key));
    built->key.xsai_op = XSAI_OP_FUSED_ADD_RMS_NORM;
    built->key.ndim = built->output.ndim;
    for (int i = 0; i < 4 && i < built->output.ndim; ++i) {
        built->key.shape[i] = built->output.shape[i];
    }
    built->key.attr0 = built->fused.eps;
    built->ok = true;
    return true;
}

bool build_general(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    *built = {};
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return build_element_call(op, with_data, built);
        case GGML_OP_UNARY:
            return build_unary_call(op, with_data, built);
        case GGML_OP_SOFT_MAX:
            return build_softmax_call(op, with_data, built);
        case GGML_OP_RMS_NORM:
            return build_fused_call(op, with_data, built);
        default:
            return false;
    }
}

xsai_op_plan_t * general_plan(const ggml_tensor * op, const BuiltCall * built, bool create) {
    if (create) {
        return ame_xsai_general_cache_get_or_create(op, &built->key, sizeof(built->key), &built->call);
    }
    return ame_xsai_general_cache_lookup(op, &built->key, sizeof(built->key));
}

}  // namespace

extern "C" void xsai_backend_note_graph(const struct ggml_cgraph * graph) {
    g_graph = graph;
}

extern "C" uint64_t xsai_backend_host_generation(const struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return 0;
    }
    HostGenerationState & state = host_state();
    uint64_t generation = 0;
    const auto bit = state.buffers.find(tensor->buffer);
    if (bit != state.buffers.end()) {
        generation = bit->second;
    }
    const auto tit = state.tensors.find(tensor);
    if (tit != state.tensors.end() && tit->second.buffer == tensor->buffer) {
        generation = std::max(generation, tit->second.generation);
    }
    return generation;
}

extern "C" void xsai_backend_q8_refresh(const struct ggml_tensor * tensor, uint64_t generation) {
    if (!q8_compact(tensor) || tensor->data == nullptr) {
        return;
    }
    if (ame_xsai_weight_plan(tensor) != nullptr &&
        ame_xsai_weight_generation(tensor) == generation) {
        return;
    }
    ame_xsai_weight_refresh_q8(tensor, tensor->data, tensor->ne[1], tensor->ne[0], generation);
}

extern "C" void xsai_backend_q8_release(const struct ggml_tensor * tensor) {
    ame_xsai_weight_release(tensor);
}

extern "C" void ggml_ame_host_buffer_on_write(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    size_t offset,
    size_t size) {
    GGML_UNUSED(offset);
    if (buffer == nullptr || size == 0) {
        return;
    }
    HostGenerationState & state = host_state();
    const uint64_t generation = ++state.next_generation;
    state.buffers[buffer] = generation;
    if (tensor != nullptr) {
        state.tensors[tensor] = { buffer, generation };
        xsai_backend_q8_refresh(tensor, generation);
    }
}

extern "C" void ggml_ame_host_buffer_on_free(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return;
    }
    HostGenerationState & state = host_state();
    state.buffers.erase(buffer);
    for (auto it = state.tensors.begin(); it != state.tensors.end(); ) {
        if (it->second.buffer == buffer) {
            xsai_backend_q8_release(it->first);
            it = state.tensors.erase(it);
        } else {
            ++it;
        }
    }
}

bool xsai_backend_q8_work_size(const struct ggml_tensor * op, uint64_t weight_generation, size_t & size) {
    if (!backend_enabled() || !q8_plan_op(op)) {
        return false;
    }
    const ame_xsai_q8_key key = q8_key(op, weight_generation);
    xsai_op_plan_t * plan = ame_xsai_q8_cache_get_or_create(op, &key);
    if (plan == nullptr) {
        return false;
    }
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    if (ame_xsai_plan_read_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
        return false;
    }
    /* ggml's compute buffer is not guaranteed to be 64-byte aligned. */
    size = requirements.workspace_size;
    if (requirements.workspace_size != 0) {
        size += requirements.workspace_alignment;
    }
    return true;
}

bool align_workspace(const xsai_op_plan_t * plan, void * wdata, size_t wsize,
                     void ** aligned_out, size_t * size_out) {
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    if (ame_xsai_plan_read_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
        return false;
    }
    if (requirements.workspace_size == 0) {
        *aligned_out = wdata;
        *size_out = wsize;
        return true;
    }
    const size_t alignment = requirements.workspace_alignment == 0 ? 1 : requirements.workspace_alignment;
    const uintptr_t base = (uintptr_t) wdata;
    const uintptr_t aligned = alignment <= 1 ? base : (base + alignment - 1) & ~(uintptr_t) (alignment - 1);
    const size_t pad = (size_t) (aligned - base);
    if (wdata == nullptr || pad > wsize || wsize - pad < requirements.workspace_size) {
        return false;
    }
    *aligned_out = (void *) aligned;
    *size_out = wsize - pad;
    return true;
}

bool xsai_backend_q8_compute(struct ggml_compute_params * params, struct ggml_tensor * op,
                             uint64_t weight_generation) {
    if (!backend_enabled() || !q8_plan_op(op)) {
        return false;
    }
    const ame_xsai_q8_key key = q8_key(op, weight_generation);
    xsai_op_plan_t * plan = ame_xsai_q8_cache_lookup(op, &key);
    if (plan == nullptr) {
        if (params->ith == 0) {
            count_route(op->op, false);
        }
        return false;
    }
    if (params->ith != 0) {
        return true;
    }
    void * workspace = nullptr;
    size_t workspace_size = 0;
    if (!align_workspace(plan, params->wdata, params->wsize, &workspace, &workspace_size)) {
        count_route(op->op, false);
        return false;
    }
    const ame_xsai_q8_bind bind = {
        op->src[1]->data,
        op->src[0]->data,
        op->data,
        workspace,
        workspace_size,
    };
    const xsai_status_t status = ame_xsai_q8_cache_execute(plan, &key, &bind);
    /* GEMM reports these before it stores output. */
    if (status != XSAI_STATUS_SUCCESS) {
        count_route(op->op, false);
        return false;
    }
    count_route(op->op, true);
    return true;
}

bool xsai_backend_work_size(int n_threads, const struct ggml_tensor * op, size_t & size) {
    if (!backend_enabled() || op == nullptr) {
        return false;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        if (ame_owns_q8(op)) {
            return false;
        }
        const uint64_t generation = op->src[0] != nullptr ? xsai_backend_host_generation(op->src[0]) : 0;
        return xsai_backend_q8_work_size(op, generation, size);
    }
    BuiltCall built;
    if (!build_general(op, false, &built)) {
        return false;
    }
    xsai_op_plan_t * plan = general_plan(op, &built, true);
    if (plan == nullptr) {
        return false;
    }
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    if (ame_xsai_plan_read_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
        return false;
    }
    size = requirements.workspace_size;
    if (requirements.workspace_size != 0) {
        size += requirements.workspace_alignment;
    }
    if (op->op == GGML_OP_SOFT_MAX) {
        const size_t cpu_scratch = sizeof(float) * (size_t) op->ne[0] * (size_t) std::max(n_threads, 1);
        if (cpu_scratch > size) {
            size = cpu_scratch;
        }
    }
    return true;
}

bool xsai_backend_compute(struct ggml_compute_params * params, struct ggml_tensor * op) {
    if (!backend_enabled() || op == nullptr || params == nullptr) {
        return false;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        if (ame_owns_q8(op)) {
            return false;
        }
        if (!q8_plan_op(op)) {
            if (params->ith == 0) {
                count_route(op->op, false);
            }
            return false;
        }
        const uint64_t generation = op->src[0] != nullptr ? xsai_backend_host_generation(op->src[0]) : 0;
        return xsai_backend_q8_compute(params, op, generation);
    }
    BuiltCall built;
    if (!build_general(op, true, &built)) {
        if (params->ith == 0) {
            count_route(op->op, false);
        }
        return false;
    }
    xsai_op_plan_t * plan = general_plan(op, &built, false);
    if (plan == nullptr) {
        if (params->ith == 0) {
            count_route(op->op, false);
        }
        return false;
    }
    if (params->ith != 0) {
        return true;
    }
    void * workspace = nullptr;
    size_t workspace_size = 0;
    if (!align_workspace(plan, params->wdata, params->wsize, &workspace, &workspace_size)) {
        count_route(op->op, false);
        return false;
    }
    built.call.workspace = workspace;
    built.call.workspace_size = workspace_size;
    const xsai_status_t status = ame_xsai_general_cache_execute(plan, &built.call);
    if (status != XSAI_STATUS_SUCCESS) {
        count_route(op->op, false);
        return false;
    }
    count_route(op->op, true);
    return true;
}

extern "C" void xsai_backend_print_report(void) {
    if (g_report_done.exchange(1) != 0) {
        return;
    }
    std::fprintf(stderr, "xsai-backend enabled=%d\n", backend_enabled() ? 1 : 0);
    std::fprintf(stderr, "xsai-backend map:\n");
    for (const RouteNote & row : k_route_notes) {
        std::fprintf(stderr, "  %-16s %s\n", row.name, row.xsai_or_cpu);
    }
    std::fprintf(stderr, "xsai-backend routes:\n");
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        const uint64_t xsai = g_xsai_count[op].load(std::memory_order_relaxed);
        const uint64_t cpu = g_cpu_count[op].load(std::memory_order_relaxed);
        if (xsai == 0 && cpu == 0) {
            continue;
        }
        const char * name = "OTHER";
        for (const RouteNote & row : k_route_notes) {
            if (row.op == op) {
                name = row.name;
                break;
            }
        }
        std::fprintf(stderr, "  op=%d %-16s xsai=%" PRIu64 " cpu_fallback=%" PRIu64 "\n",
                     op, name, xsai, cpu);
    }
    ame_xsai_observe_export_t observed = {};
    ame_xsai_observe_export(&observed);
    std::fprintf(stderr,
                 "xsai-backend plan_create=%" PRIu64 " key_miss=%" PRIu64
                 " cache_hit=%" PRIu64 " execute=%" PRIu64 " prepared_bytes=%" PRIu64 "\n",
                 (uint64_t) ame_xsai_plan_create_count(),
                 (uint64_t) ame_xsai_plan_key_miss_count(),
                 (uint64_t) ame_xsai_plan_cache_hit_count(),
                 (uint64_t) ame_xsai_plan_execute_count(),
                 (uint64_t) ame_xsai_prepared_bytes());
    std::fprintf(stderr,
                 "xsai-backend observe prepare_count=%" PRIu64 " prepare_ns=%" PRIu64
                 " plan_create_count=%" PRIu64 " plan_create_ns=%" PRIu64
                 " workspace_count=%" PRIu64 " workspace_ns=%" PRIu64
                 " execute_count=%" PRIu64 " execute_ns=%" PRIu64
                 " kernel_count=%" PRIu64 " kernel_ns=%" PRIu64 "\n",
                 (uint64_t) observed.prepare_count, (uint64_t) observed.prepare_ns,
                 (uint64_t) observed.plan_create_count, (uint64_t) observed.plan_create_ns,
                 (uint64_t) observed.workspace_count, (uint64_t) observed.workspace_ns,
                 (uint64_t) observed.execute_count, (uint64_t) observed.execute_ns,
                 (uint64_t) observed.kernel_count, (uint64_t) observed.kernel_ns);
}

namespace {

struct xsai_buffer_context {
    void * base = nullptr;
    size_t base_size = 0;
    std::vector<ggml_tensor *> tensors;
};

xsai_buffer_context * xsai_buffer_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<xsai_buffer_context *>(buffer->context);
}

void buffer_free(ggml_backend_buffer_t buffer) {
    xsai_buffer_context * ctx = xsai_buffer_ctx(buffer);
    for (ggml_tensor * tensor : ctx->tensors) {
        xsai_backend_q8_release(tensor);
    }
    if (ctx->base != nullptr) {
        ggml_aligned_free(ctx->base, ctx->base_size);
    }
    delete ctx;
}

void * buffer_get_base(ggml_backend_buffer_t buffer) {
    return xsai_buffer_ctx(buffer)->base;
}

enum ggml_status buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    xsai_buffer_ctx(buffer)->tensors.push_back(tensor);
    return GGML_STATUS_SUCCESS;
}

void buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                       const void * data, size_t offset, size_t size) {
    std::memcpy((char *) tensor->data + offset, data, size);
    if (size != 0) {
        xsai_backend_q8_refresh(tensor, xsai_backend_host_generation(tensor));
    }
    GGML_UNUSED(buffer);
}

void buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                       void * data, size_t offset, size_t size) {
    std::memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

void buffer_memset(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                   uint8_t value, size_t offset, size_t size) {
    std::memset((char *) tensor->data + offset, value, size);
    if (size != 0) {
        xsai_backend_q8_refresh(tensor, xsai_backend_host_generation(tensor));
    }
    GGML_UNUSED(buffer);
}

void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    xsai_buffer_context * ctx = xsai_buffer_ctx(buffer);
    std::memset(ctx->base, value, buffer->size);
    GGML_UNUSED(value);
}

const ggml_backend_buffer_i xsai_buffer_interface = {
    /* .free_buffer     = */ buffer_free,
    /* .get_base        = */ buffer_get_base,
    /* .init_tensor     = */ buffer_init_tensor,
    /* .memset_tensor   = */ buffer_memset,
    /* .set_tensor      = */ buffer_set_tensor,
    /* .get_tensor      = */ buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ buffer_clear,
    /* .reset           = */ nullptr,
};

const char * buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "XSAI";
}

ggml_backend_buffer_t buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == nullptr) {
        return nullptr;
    }
    xsai_buffer_context * ctx = new xsai_buffer_context();
    ctx->base = data;
    ctx->base_size = size;
    return ggml_backend_buffer_init(buft, xsai_buffer_interface, ctx, size);
}

size_t buffer_type_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

size_t buffer_type_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

bool buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

}  // namespace

namespace ggml::cpu::xsai {

class tensor_traits : public ggml::cpu::tensor_traits {
public:
    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        return xsai_backend_work_size(n_threads, op, size);
    }
    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        return xsai_backend_compute(params, op);
    }
};

static tensor_traits * traits() {
    static tensor_traits object;
    return &object;
}

class extra_buffer_type : ggml::cpu::extra_buffer_type {
public:
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor *) override {
        /* Storage stays ordinary host bytes. Scheduling must not block CPU fallback. */
        return true;
    }
    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor *) override {
        if (!backend_enabled()) {
            return nullptr;
        }
        return traits();
    }
};

}  // namespace ggml::cpu::xsai

static void xsai_report_at_exit() {
    xsai_backend_print_report();
}

ggml_backend_buffer_type_t ggml_backend_cpu_xsai_buffer_type(void) {
    static bool report_armed = false;
    if (!report_armed) {
        std::atexit(xsai_report_at_exit);
        report_armed = true;
    }
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ buffer_type_name,
            /* .alloc_buffer     = */ buffer_type_alloc,
            /* .get_alignment    = */ buffer_type_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ buffer_type_alloc_size,
            /* .is_host          = */ buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::xsai::extra_buffer_type(),
    };
    return &buft;
}
