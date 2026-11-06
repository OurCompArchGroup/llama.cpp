#include "xsai-backend.h"

#include "xsai-fusion.h"
#include "xsai-observe.h"
#include "xsai-plan.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "traits.h"

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
      "Q8_0 MUL_MAT -> xsai-blas GEMM; other types stay with AME/CPU" },
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
std::atomic<uint64_t> g_q8_probe_claims{0};
std::atomic<uint64_t> g_q8_probe_declines{0};
std::atomic<uint64_t> g_q8_foreign_declines{0};

bool env_disabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr &&
        (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
         std::strcmp(value, "off") == 0 || std::strcmp(value, "no") == 0);
}

bool backend_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = env_disabled("XSAI_GGML_BACKEND") ? 0 : 1;
        if (cached) {
            xsai_observe_set_enabled(1);
        }
    }
    return cached != 0;
}

/* Q8 weight placement and execution switch. Off keeps Q8_0 MUL_MAT on AME/CPU. */
bool q8_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = env_disabled("XSAI_GGML_Q8") ? 0 : 1;
    }
    return cached != 0;
}

/*
 * Status policy from the operator ABI: UNSUPPORTED and PREPARATION_REQUIRED
 * let the framework keep the CPU node. INVALID_ARGUMENT means the adapter
 * built a malformed call, and any execute-stage failure is a runtime error.
 * Neither is allowed to fall back and hide the defect.
 */
bool fallback_status(xsai_status_t status) {
    return status == XSAI_STATUS_UNSUPPORTED ||
        status == XSAI_STATUS_PREPARATION_REQUIRED;
}

void adapter_error(const char * stage, xsai_status_t status) {
    std::fprintf(stderr, "xsai-backend: adapter ABI error at %s: status=%d\n",
                 stage, (int) status);
    GGML_ABORT("xsai adapter ABI error");
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

bool q8_compact(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q8_0 || tensor->ne[0] <= 0 ||
        tensor->ne[1] <= 0) {
        return false;
    }
    const int64_t blocks = (tensor->ne[0] + 31) / 32;
    return tensor->nb[0] == 34 && tensor->nb[1] == (size_t) blocks * 34u;
}

/*
 * The adapter maps a ggml MUL_MAT onto a compact logical contract: A [M, K]
 * from src1 [K, M], B [K, N] from the row-major Q8_0 weight [N, K], and C
 * [M, N] over dst [N, M]. Anything else stays on the CPU.
 */
bool q8_plan_op(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32 ||
        op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * activation = op->src[1];
    if (weight->type != GGML_TYPE_Q8_0 || activation->type != GGML_TYPE_F32) {
        return false;
    }
    if (!q8_compact(weight) || weight->ne[2] != 1 || weight->ne[3] != 1) {
        return false;
    }
    if (!dense_f32(activation) || !dense_f32(op)) {
        return false;
    }
    if (activation->ne[2] != 1 || activation->ne[3] != 1 ||
        op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    return activation->ne[0] == weight->ne[0] &&
        activation->ne[1] >= 1 &&
        op->ne[0] == weight->ne[1] &&
        op->ne[1] == activation->ne[1];
}

/* Q8 GEMM reads the weight on the host and keys its plan cache on the host
 * generation counter, so only host buffers qualify (the plain CPU buffer and
 * this adapter's own host buffer). A weight in any other buffer type belongs
 * to a backend that manages its own memory and invalidation: decline, so a
 * failed attempt there is not retried here and the host generation stays
 * meaningful. Raw host memory (buffer == nullptr, as in the unit tests)
 * counts as host. */
bool q8_foreign_weight(const ggml_tensor * op) {
    if (!q8_plan_op(op)) {
        return false;
    }
    const ggml_backend_buffer_t buffer = op->src[0]->buffer;
    if (buffer == nullptr) {
        return false;
    }
    const struct ggml_backend_buffer_type_i * iface = &buffer->buft->iface;
    return iface->is_host == nullptr || !iface->is_host(buffer->buft);
}

/* Builds and seals the plan identity. False means this node stays on the CPU. */
bool q8_key_build(const ggml_tensor * op, uint64_t generation, xsai_q8_key * key) {
    if (!q8_plan_op(op)) {
        return false;
    }
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * activation = op->src[1];
    xsai_backend_q8_refresh(weight, generation);
    const xsai_weight_plan_t * prepared = xsai_weight_plan(weight);
    if (prepared == nullptr) {
        return false;
    }
    *key = xsai_q8_key{};
    key->op = XSAI_OP_GEMM;
    key->m = activation->ne[1];
    key->k = weight->ne[0];
    key->n = weight->ne[1];
    key->blocks = (key->k + 31) / 32;
    key->weight_generation = generation;
    key->weight_identity = weight;
    key->prepared_weight = prepared;
    return xsai_q8_key_seal(key);
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
    uint64_t key = 0; /* template call fingerprint */
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
    bind_f32_1d(&built->inputs[0], with_data ? op->src[0]->data : nullptr, &built->shapes[0], n);
    bind_f32_1d(&built->inputs[1], with_data ? op->src[1]->data : nullptr, &built->shapes[1], n);
    bind_f32_1d(&built->output, with_data ? op->data : nullptr, &built->shapes[2], n);
    built->call = {};
    built->call.struct_size = sizeof(built->call);
    built->call.abi_version = XSAI_BLAS_ABI_VERSION;
    built->call.op = element_op(op->op);
    built->call.input_count = 2;
    built->call.inputs = built->inputs;
    built->call.output_count = 1;
    built->call.outputs = &built->output;
    built->key = xsai_op_call_fingerprint(&built->call);
    built->ok = built->key != 0;
    return built->ok;
}

bool build_unary_call(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    xsai_op_type_t xsai_op = -1;
    int64_t n = 0;
    if (!unary_eligible(op, &xsai_op, &n)) {
        return false;
    }
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
    built->key = xsai_op_call_fingerprint(&built->call);
    built->ok = built->key != 0;
    return built->ok;
}

bool build_softmax_call(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    float scale = 1.0f;
    int64_t rows = 0;
    int64_t cols = 0;
    if (!softmax_eligible(op, &scale, &rows, &cols)) {
        return false;
    }
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
    built->softmax.abi_version = XSAI_BLAS_ABI_VERSION;
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
    built->call.op_params = &built->softmax;
    built->key = xsai_op_call_fingerprint(&built->call);
    built->ok = built->key != 0;
    return built->ok;
}

bool build_fused_call(const ggml_tensor * op, bool with_data, BuiltCall * built) {
    if (op->op != GGML_OP_RMS_NORM) {
        return false;
    }
    const xsai_add_rms_fusion_decision_t decision = xsai_match_add_rms_norm(
        g_graph, op, built->inputs, &built->output, &built->fused, &built->call);
    if (decision != XSAI_FUSION_ACCEPT) {
        return false;
    }
    if (!with_data) {
        for (int i = 0; i < 3; ++i) {
            built->inputs[i].data = nullptr;
        }
        built->output.data = nullptr;
    }
    built->key = xsai_op_call_fingerprint(&built->call);
    built->ok = built->key != 0;
    return built->ok;
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

xsai_op_plan_t * general_plan(const ggml_tensor * op, const BuiltCall * built,
                              bool create, xsai_status_t * status_out) {
    if (status_out != nullptr) {
        *status_out = XSAI_STATUS_SUCCESS;
    }
    if (create) {
        return xsai_general_cache_get_or_create(op, built->key, &built->call, status_out);
    }
    return xsai_general_cache_lookup(op, built->key);
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
    if (xsai_weight_plan(tensor) != nullptr &&
        xsai_weight_generation(tensor) == generation) {
        return;
    }
    xsai_weight_refresh_q8(tensor, tensor->data, tensor->ne[1], tensor->ne[0], generation);
}

extern "C" void xsai_backend_q8_release(const struct ggml_tensor * tensor) {
    xsai_weight_release(tensor);
}

/* Shared by the default CPU buffer hook and the XSAI buffer's own writes: a
   write to a host-resident weight bumps its generation and refreshes any
   prepared weight plan bound to it. */
static void note_host_write(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, size_t size) {
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

static void forget_buffer(ggml_backend_buffer_t buffer) {
    HostGenerationState & state = host_state();
    state.buffers.erase(buffer);
    for (auto it = state.tensors.begin(); it != state.tensors.end(); ) {
        if (it->second.buffer == buffer) {
            it = state.tensors.erase(it);
        } else {
            ++it;
        }
    }
}

extern "C" void ggml_xsai_host_buffer_on_write(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    size_t offset,
    size_t size) {
    GGML_UNUSED(offset);
    note_host_write(buffer, tensor, size);
}

extern "C" void ggml_xsai_host_buffer_on_free(ggml_backend_buffer_t buffer) {
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
    xsai_q8_key key;
    if (!q8_key_build(op, weight_generation, &key)) {
        return false;
    }
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    xsai_op_plan_t * plan = xsai_q8_cache_get_or_create(op, &key, &status);
    if (plan == nullptr) {
        if (!fallback_status(status)) {
            adapter_error("q8 plan_create", status);
        }
        return false;
    }
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    if (xsai_plan_read_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
        return false;
    }
    /* ggml's compute buffer is not guaranteed to be 64-byte aligned. */
    size = requirements.workspace_size;
    if (requirements.workspace_size != 0) {
        size += requirements.workspace_alignment;
    }
    return true;
}

static bool align_workspace(const xsai_op_plan_t * plan, void * wdata, size_t wsize,
                            void ** aligned_out, size_t * size_out) {
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    if (xsai_plan_read_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
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
    xsai_q8_key key;
    if (!q8_key_build(op, weight_generation, &key)) {
        if (params->ith == 0) {
            count_route(op->op, false);
        }
        return false;
    }
    xsai_op_plan_t * plan = xsai_q8_cache_lookup(op, &key);
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
    const xsai_q8_bind bind = {
        op->src[1]->data,
        op->data,
        workspace,
        workspace_size,
    };
    const xsai_status_t status = xsai_q8_cache_execute(plan, &key, &bind);
    if (status != XSAI_STATUS_SUCCESS) {
        adapter_error("q8 execute", status);
    }
    count_route(op->op, true);
    return true;
}

bool xsai_backend_work_size(int n_threads, const struct ggml_tensor * op, size_t & size) {
    if (!backend_enabled() || op == nullptr) {
        return false;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        if (q8_foreign_weight(op)) {
            return false;
        }
        const uint64_t generation = op->src[0] != nullptr ? xsai_backend_host_generation(op->src[0]) : 0;
        return xsai_backend_q8_work_size(op, generation, size);
    }
    BuiltCall built;
    if (!build_general(op, false, &built)) {
        return false;
    }
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    xsai_op_plan_t * plan = general_plan(op, &built, true, &status);
    if (plan == nullptr) {
        if (!fallback_status(status)) {
            adapter_error("plan_create", status);
        }
        return false;
    }
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    if (xsai_plan_read_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
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
        if (q8_foreign_weight(op)) {
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
    xsai_op_plan_t * plan = general_plan(op, &built, false, nullptr);
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
    const xsai_status_t status = xsai_general_cache_execute(plan, &built.call);
    if (status != XSAI_STATUS_SUCCESS) {
        adapter_error("execute", status);
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
    std::fprintf(stderr,
                 "xsai-backend q8 probe_claims=%" PRIu64 " probe_declines=%" PRIu64
                 " foreign_declines=%" PRIu64 "\n",
                 (uint64_t) g_q8_probe_claims.load(std::memory_order_relaxed),
                 (uint64_t) g_q8_probe_declines.load(std::memory_order_relaxed),
                 (uint64_t) g_q8_foreign_declines.load(std::memory_order_relaxed));
    xsai_observe_export_t observed = {};
    xsai_observe_export(&observed);
    std::fprintf(stderr,
                 "xsai-backend plan_create=%" PRIu64 " key_miss=%" PRIu64
                 " cache_hit=%" PRIu64 " execute=%" PRIu64
                 " prepared_bytes=%" PRIu64 " prepared_peak_bytes=%" PRIu64 "\n",
                 (uint64_t) xsai_plan_create_count(),
                 (uint64_t) xsai_plan_key_miss_count(),
                 (uint64_t) xsai_plan_cache_hit_count(),
                 (uint64_t) xsai_plan_execute_count(),
                 (uint64_t) xsai_prepared_bytes(),
                 (uint64_t) xsai_prepared_peak_bytes());
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
    forget_buffer(buffer);
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
    /* ggml only calls the weak host-buffer hooks for the default CPU buffer,
       so this buffer type must note its own writes. */
    note_host_write(buffer, tensor, size);
}

void buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                       void * data, size_t offset, size_t size) {
    std::memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

void buffer_memset(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                   uint8_t value, size_t offset, size_t size) {
    std::memset((char *) tensor->data + offset, value, size);
    note_host_write(buffer, tensor, size);
}

void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    xsai_buffer_context * ctx = xsai_buffer_ctx(buffer);
    std::memset(ctx->base, value, buffer->size);
    note_host_write(buffer, nullptr, buffer->size);
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

/*
 * Weight placement probe. weight_buft_supported() creates a zero-size dummy
 * buffer of the candidate type and attaches it to the weight before asking the
 * device. Only such a dummy (size 0, no context) counts as a placement query;
 * real buffers keep the permissive runtime answer so scheduling never blocks
 * the CPU fallback. Only src0 is inspected: MUL_MAT/GET_ROWS weights live
 * there, while a bias on src1 must not turn an ADD into a Q8 probe. The dummy
 * has an empty iface, so q8_foreign_weight() must not be called on it.
 */
bool xsai_backend_placement_supports(ggml_backend_buffer_type_t self, const ggml_tensor * op) {
    if (op == nullptr) {
        return true;
    }
    const ggml_tensor * src0 = op->src[0];
    if (src0 == nullptr || src0->buffer == nullptr || src0->buffer->buft != self ||
        src0->buffer->size != 0 || src0->buffer->context != nullptr) {
        return true;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return true;
    }
    const bool supported = backend_enabled() && q8_enabled() && q8_plan_op(op);
    if (supported) {
        g_q8_probe_claims.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_q8_probe_declines.fetch_add(1, std::memory_order_relaxed);
    }
    return supported;
}

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
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        return xsai_backend_placement_supports(ggml_backend_cpu_xsai_buffer_type(), op);
    }
    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (!backend_enabled()) {
            return nullptr;
        }
        if (op->op == GGML_OP_MUL_MAT) {
            if (!q8_enabled() || !q8_plan_op(op)) {
                return nullptr;
            }
            if (q8_foreign_weight(op)) {
                g_q8_foreign_declines.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
        }
        return traits();
    }
};

}  // namespace ggml::cpu::xsai

static void xsai_report_at_exit() {
    xsai_backend_print_report();
}

ggml_backend_buffer_type_t ggml_backend_cpu_xsai_buffer_type(void) {
    /* Buffer allocation happens at model load, before the first graph plan.
       Resolve the runtime switches here so observation and weight placement
       cover load-time decisions. */
    (void) backend_enabled();
    (void) q8_enabled();
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
