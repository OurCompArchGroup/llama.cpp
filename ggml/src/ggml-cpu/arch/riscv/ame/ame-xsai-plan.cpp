#include "ame-xsai-plan.h"

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>

namespace {

std::mutex g_mu;
std::unordered_map<const void *, std::pair<ame_xsai_q8_key, xsai_op_plan_t *>> g_cache;
uint64_t g_create_count = 0;
uint64_t g_hit_count = 0;

bool key_equal(const ame_xsai_q8_key & lhs, const ame_xsai_q8_key & rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

void destroy_plan(xsai_op_plan_t * plan) {
    xsai_op_plan_destroy(plan);
}

bool layout_supported(const ame_xsai_q8_key * key) {
    if (key == nullptr || key->m <= 0 || key->n <= 0 || key->k <= 0) {
        return false;
    }
    if (key->src0_ne2 != 1 || key->src0_ne3 != 1 || key->src1_ne2 != 1 ||
        key->src1_ne3 != 1 || key->dst_ne2 != 1 || key->dst_ne3 != 1) {
        return false;
    }
    const int64_t blocks = (key->k + 31) / 32;
    const uint64_t row_bytes = (uint64_t) blocks * 34u;
    if (key->act_nb0 != sizeof(float) ||
        key->act_nb1 != (uint64_t) key->k * sizeof(float)) {
        return false;
    }
    if (key->weight_nb0 != 34 || key->weight_nb1 != row_bytes) {
        return false;
    }
    const bool native_out = key->dst_nb0 == (uint64_t) key->m * sizeof(float) &&
        key->dst_nb1 == sizeof(float);
    const bool ggml_out = key->dst_nb0 == sizeof(float) &&
        key->dst_nb1 == (uint64_t) key->n * sizeof(float);
    return native_out || ggml_out;
}

xsai_status_t build_call(const ame_xsai_q8_key * key, const ame_xsai_q8_bind * bind,
                          DLTensor inputs[2], DLTensor * output,
                          int64_t shapes[6], int64_t out_strides[2],
                          xsai_op_call_t * call) {
    const int64_t blocks = (key->k + 31) / 32;
    shapes[0] = key->m;
    shapes[1] = key->k;
    shapes[2] = key->n;
    shapes[3] = blocks * 34;
    shapes[4] = key->n;
    shapes[5] = key->m;
    inputs[0] = DLTensor{};
    inputs[0].data = bind ? const_cast<void *>(bind->activation) : nullptr;
    inputs[0].device = {kDLCPU, 0};
    inputs[0].ndim = 2;
    inputs[0].dtype = {kDLFloat, 32, 1};
    inputs[0].shape = shapes;
    inputs[1] = DLTensor{};
    inputs[1].data = bind ? const_cast<void *>(bind->weight) : nullptr;
    inputs[1].device = {kDLCPU, 0};
    inputs[1].ndim = 2;
    inputs[1].dtype = {kDLUInt, 8, 1};
    inputs[1].shape = shapes + 2;
    *output = DLTensor{};
    output->data = bind ? bind->output : nullptr;
    output->device = {kDLCPU, 0};
    output->ndim = 2;
    output->dtype = {kDLFloat, 32, 1};
    output->shape = shapes + 4;
    if (key->dst_nb0 == sizeof(float)) {
        out_strides[0] = 1;
        out_strides[1] = key->n;
        output->strides = out_strides;
    }
    *call = xsai_op_call_t{};
    call->struct_size = sizeof(*call);
    call->abi_version = XSAI_BLAS_ABI_VERSION;
    call->op = XSAI_OP_GEMM_I8_F32;
    call->input_count = 2;
    call->inputs = inputs;
    call->output_count = 1;
    call->outputs = output;
    if (bind != nullptr) {
        call->workspace = bind->workspace;
        call->workspace_size = bind->workspace_size;
    }
    return XSAI_STATUS_SUCCESS;
}

void drop_locked(const void * op) {
    const auto it = g_cache.find(op);
    if (it == g_cache.end()) {
        return;
    }
    destroy_plan(it->second.second);
    g_cache.erase(it);
}

}  // namespace

xsai_op_plan_t * ame_xsai_q8_cache_get_or_create(const void * op,
                                                 const ame_xsai_q8_key * key) {
    if (op == nullptr || key == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_cache.find(op);
    if (it != g_cache.end() && key_equal(it->second.first, *key)) {
        ++g_hit_count;
        return it->second.second;
    }
    if (it != g_cache.end()) {
        drop_locked(op);
    }
    if (!layout_supported(key)) {
        return nullptr;
    }

    DLTensor inputs[2];
    DLTensor output;
    int64_t shapes[6];
    int64_t out_strides[2];
    xsai_op_call_t call;
    build_call(key, nullptr, inputs, &output, shapes, out_strides, &call);
    xsai_op_plan_t * plan = nullptr;
    const xsai_status_t status = xsai_op_plan_create(&call, &plan);
    if (status != XSAI_STATUS_SUCCESS || plan == nullptr) {
        xsai_op_plan_destroy(plan);
        return nullptr;
    }
    g_cache.emplace(op, std::make_pair(*key, plan));
    ++g_create_count;
    return plan;
}

xsai_op_plan_t * ame_xsai_q8_cache_lookup(const void * op,
                                          const ame_xsai_q8_key * key) {
    if (op == nullptr || key == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_cache.find(op);
    if (it == g_cache.end() || !key_equal(it->second.first, *key)) {
        return nullptr;
    }
    return it->second.second;
}

xsai_status_t ame_xsai_q8_cache_execute(xsai_op_plan_t * plan,
                                        const ame_xsai_q8_key * key,
                                        const ame_xsai_q8_bind * bind) {
    if (plan == nullptr || key == nullptr || bind == nullptr) {
        return XSAI_STATUS_INVALID_ARGUMENT;
    }
    DLTensor inputs[2];
    DLTensor output;
    int64_t shapes[6];
    int64_t out_strides[2];
    xsai_op_call_t call;
    build_call(key, bind, inputs, &output, shapes, out_strides, &call);
    return xsai_op_plan_execute(plan, &call);
}

uint64_t ame_xsai_plan_create_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_create_count;
}

uint64_t ame_xsai_plan_cache_hit_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_hit_count;
}

void ame_xsai_plan_cache_reset_counts(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_create_count = 0;
    g_hit_count = 0;
}

void ame_xsai_plan_cache_clear(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto & entry : g_cache) {
        destroy_plan(entry.second.second);
    }
    g_cache.clear();
}

size_t ame_xsai_plan_cache_live(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_cache.size();
}

extern "C" void ame_xsai_plan_cache_retain(const void * const * ops, size_t count) {
    std::lock_guard<std::mutex> lock(g_mu);
    std::unordered_set<const void *> live;
    for (size_t i = 0; i < count; ++i) {
        if (ops != nullptr) {
            live.insert(ops[i]);
        }
    }
    for (auto it = g_cache.begin(); it != g_cache.end(); ) {
        if (live.find(it->first) == live.end()) {
            destroy_plan(it->second.second);
            it = g_cache.erase(it);
        } else {
            ++it;
        }
    }
}
