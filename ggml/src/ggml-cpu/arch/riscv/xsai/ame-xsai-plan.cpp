#include "ame-xsai-plan.h"
#include "ame-xsai-observe.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ObserveSpan {
    int phase;
    uint64_t token;
    explicit ObserveSpan(int phase_in) : phase(phase_in), token(UINT64_MAX) {
        ame_xsai_observe_span_begin(phase, &token);
    }
    ~ObserveSpan() {
        ame_xsai_observe_span_end(phase, token);
    }
};

namespace {

std::mutex g_mu;
std::condition_variable g_cv;
const void * g_refreshing = nullptr;

struct CacheEntry {
    ame_xsai_q8_key key;
    xsai_op_plan_t * plan = nullptr;
    std::shared_ptr<std::atomic<int>> inflight = std::make_shared<std::atomic<int>>(0);
    bool general = false;
    std::vector<uint8_t> general_key;
};

std::unordered_map<const void *, CacheEntry> g_cache;
uint64_t g_create_count = 0;
uint64_t g_hit_count = 0;
uint64_t g_key_miss_count = 0;
uint64_t g_execute_count = 0;
uint64_t g_seq = 0;
uint64_t g_last_op_seq = 0;
uint64_t g_last_weight_seq = 0;
uint64_t g_op_destroy_count = 0;
uint64_t g_weight_destroy_count = 0;
int g_order_violations = 0;

struct WeightSlot {
    xsai_weight_plan_t * plan = nullptr;
    void * arena = nullptr;
    uint64_t generation = 0;
    size_t arena_bytes = 0;
};

std::mutex g_weight_mu;
std::unordered_map<const void *, WeightSlot> g_weights;

bool key_equal(const ame_xsai_q8_key & lhs, const ame_xsai_q8_key & rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

void note_op_destroy_locked() {
    g_last_op_seq = ++g_seq;
    ++g_op_destroy_count;
}

void note_weight_destroy(bool require_op) {
    std::lock_guard<std::mutex> lock(g_mu);
    const uint64_t seq = ++g_seq;
    if (require_op && (g_last_op_seq == 0 || g_last_op_seq > seq)) {
        ++g_order_violations;
    }
    g_last_weight_seq = seq;
    ++g_weight_destroy_count;
}

void destroy_plan_locked(xsai_op_plan_t * plan) {
    xsai_op_plan_destroy(plan);
    note_op_destroy_locked();
}

void wait_and_destroy(CacheEntry entry) {
    if (entry.plan == nullptr) {
        return;
    }
    std::unique_lock<std::mutex> lock(g_mu);
    g_cv.wait(lock, [&] { return entry.inflight->load() == 0; });
    destroy_plan_locked(entry.plan);
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
    call->prepared_weight = key->prepared_weight;
    if (bind != nullptr) {
        call->workspace = bind->workspace;
        call->workspace_size = bind->workspace_size;
    }
    return XSAI_STATUS_SUCCESS;
}

int retire_weight_locked_copy(const void * tensor, std::vector<CacheEntry> * retired) {
    int count = 0;
    for (auto it = g_cache.begin(); it != g_cache.end(); ) {
        if (it->second.key.weight_identity == tensor) {
            retired->push_back(std::move(it->second));
            it = g_cache.erase(it);
            ++count;
        } else {
            ++it;
        }
    }
    return count;
}

void destroy_weight_slot(const void * tensor, bool require_op) {
    WeightSlot slot;
    {
        std::lock_guard<std::mutex> lock(g_weight_mu);
        const auto it = g_weights.find(tensor);
        if (it == g_weights.end()) {
            return;
        }
        slot = it->second;
        g_weights.erase(it);
    }
    if (slot.plan != nullptr) {
        xsai_weight_plan_destroy(slot.plan);
        note_weight_destroy(require_op);
    }
    std::free(slot.arena);
}

}  // namespace

xsai_op_plan_t * ame_xsai_q8_cache_get_or_create(const void * op,
                                                 const ame_xsai_q8_key * key) {
    if (op == nullptr || key == nullptr) {
        return nullptr;
    }
    CacheEntry replaced;
    bool have_replaced = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (key->weight_identity != nullptr && key->weight_identity == g_refreshing) {
            return nullptr;
        }
        const auto it = g_cache.find(op);
        if (it != g_cache.end() && !it->second.general && key_equal(it->second.key, *key)) {
            ++g_hit_count;
            return it->second.plan;
        }
        if (it != g_cache.end()) {
            replaced = std::move(it->second);
            g_cache.erase(it);
            have_replaced = true;
        }
    }
    if (have_replaced) {
        wait_and_destroy(std::move(replaced));
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
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_key_miss_count;
    }
    {
        ObserveSpan span(AME_XSAI_OBS_PLAN_CREATE);
        status = xsai_op_plan_create(&call, &plan);
    }
    if (status != XSAI_STATUS_SUCCESS || plan == nullptr) {
        xsai_op_plan_destroy(plan);
        return nullptr;
    }
    CacheEntry stale;
    bool have_stale = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (key->weight_identity != nullptr && key->weight_identity == g_refreshing) {
            destroy_plan_locked(plan);
            return nullptr;
        }
        const auto existing = g_cache.find(op);
        CacheEntry created;
        created.key = *key;
        created.plan = plan;
        created.inflight = std::make_shared<std::atomic<int>>(0);
        if (existing != g_cache.end()) {
            stale = std::move(existing->second);
            existing->second = std::move(created);
            have_stale = true;
        } else {
            g_cache.emplace(op, std::move(created));
        }
        ++g_create_count;
    }
    if (have_stale) {
        wait_and_destroy(std::move(stale));
    }
    return plan;
}

xsai_op_plan_t * ame_xsai_q8_cache_lookup(const void * op,
                                          const ame_xsai_q8_key * key) {
    if (op == nullptr || key == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_cache.find(op);
    if (it == g_cache.end() || it->second.general || !key_equal(it->second.key, *key)) {
        return nullptr;
    }
    return it->second.plan;
}

xsai_status_t ame_xsai_q8_cache_execute(xsai_op_plan_t * plan,
                                        const ame_xsai_q8_key * key,
                                        const ame_xsai_q8_bind * bind) {
    if (plan == nullptr || key == nullptr || bind == nullptr) {
        return XSAI_STATUS_INVALID_ARGUMENT;
    }
    std::shared_ptr<std::atomic<int>> inflight;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        for (auto & entry : g_cache) {
            if (entry.second.plan == plan) {
                inflight = entry.second.inflight;
                inflight->fetch_add(1);
                break;
            }
        }
    }
    if (!inflight) {
        return XSAI_STATUS_UNSUPPORTED;
    }
    DLTensor inputs[2];
    DLTensor output;
    int64_t shapes[6];
    int64_t out_strides[2];
    xsai_op_call_t call;
    build_call(key, bind, inputs, &output, shapes, out_strides, &call);
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    {
        ObserveSpan span(AME_XSAI_OBS_EXECUTE);
        status = xsai_op_plan_execute(plan, &call);
    }
    if (status == XSAI_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_execute_count;
    }
    inflight->fetch_sub(1);
    g_cv.notify_all();
    return status;
}

static bool general_key_equal(const CacheEntry & entry, const void * key, size_t key_nbytes) {
    return entry.general && entry.general_key.size() == key_nbytes &&
        (key_nbytes == 0 || std::memcmp(entry.general_key.data(), key, key_nbytes) == 0);
}

xsai_op_plan_t * ame_xsai_general_cache_get_or_create(const void * op,
                                                      const void * key,
                                                      size_t key_nbytes,
                                                      const xsai_op_call_t * call_template) {
    if (op == nullptr || call_template == nullptr || (key == nullptr && key_nbytes != 0)) {
        return nullptr;
    }
    CacheEntry replaced;
    bool have_replaced = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_cache.find(op);
        if (it != g_cache.end() && general_key_equal(it->second, key, key_nbytes)) {
            ++g_hit_count;
            return it->second.plan;
        }
        if (it != g_cache.end()) {
            replaced = std::move(it->second);
            g_cache.erase(it);
            have_replaced = true;
        }
    }
    if (have_replaced) {
        wait_and_destroy(std::move(replaced));
    }

    xsai_op_plan_t * plan = nullptr;
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_key_miss_count;
    }
    {
        ObserveSpan span(AME_XSAI_OBS_PLAN_CREATE);
        status = xsai_op_plan_create(call_template, &plan);
    }
    if (status != XSAI_STATUS_SUCCESS || plan == nullptr) {
        xsai_op_plan_destroy(plan);
        return nullptr;
    }

    CacheEntry created;
    created.plan = plan;
    created.inflight = std::make_shared<std::atomic<int>>(0);
    created.general = true;
    if (key_nbytes != 0) {
        const auto * bytes = static_cast<const uint8_t *>(key);
        created.general_key.assign(bytes, bytes + key_nbytes);
    }
    CacheEntry stale;
    bool have_stale = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto existing = g_cache.find(op);
        if (existing != g_cache.end()) {
            stale = std::move(existing->second);
            existing->second = std::move(created);
            have_stale = true;
        } else {
            g_cache.emplace(op, std::move(created));
        }
        ++g_create_count;
    }
    if (have_stale) {
        wait_and_destroy(std::move(stale));
    }
    return plan;
}

xsai_op_plan_t * ame_xsai_general_cache_lookup(const void * op,
                                               const void * key,
                                               size_t key_nbytes) {
    if (op == nullptr || (key == nullptr && key_nbytes != 0)) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_cache.find(op);
    if (it == g_cache.end() || !general_key_equal(it->second, key, key_nbytes)) {
        return nullptr;
    }
    return it->second.plan;
}

xsai_status_t ame_xsai_general_cache_execute(xsai_op_plan_t * plan,
                                             const xsai_op_call_t * call) {
    if (plan == nullptr || call == nullptr) {
        return XSAI_STATUS_INVALID_ARGUMENT;
    }
    std::shared_ptr<std::atomic<int>> inflight;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        for (auto & entry : g_cache) {
            if (entry.second.plan == plan && entry.second.general) {
                inflight = entry.second.inflight;
                inflight->fetch_add(1);
                break;
            }
        }
    }
    if (!inflight) {
        return XSAI_STATUS_UNSUPPORTED;
    }
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    {
        ObserveSpan span(AME_XSAI_OBS_EXECUTE);
        status = xsai_op_plan_execute(plan, call);
    }
    if (status == XSAI_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_execute_count;
    }
    inflight->fetch_sub(1);
    g_cv.notify_all();
    return status;
}

xsai_status_t ame_xsai_plan_read_requirements(
    const xsai_op_plan_t * plan, xsai_op_requirements_t * requirements) {
    ObserveSpan span(AME_XSAI_OBS_WORKSPACE);
    return xsai_op_plan_get_requirements(plan, requirements);
}

uint64_t ame_xsai_plan_create_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_create_count;
}

uint64_t ame_xsai_plan_cache_hit_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_hit_count;
}

uint64_t ame_xsai_plan_key_miss_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_key_miss_count;
}

uint64_t ame_xsai_plan_execute_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_execute_count;
}

uint64_t ame_xsai_prepared_bytes(void) {
    std::lock_guard<std::mutex> lock(g_weight_mu);
    uint64_t bytes = 0;
    for (const auto & entry : g_weights) {
        bytes += entry.second.arena_bytes;
    }
    return bytes;
}

void ame_xsai_plan_cache_reset_counts(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_create_count = 0;
    g_hit_count = 0;
    g_key_miss_count = 0;
    g_execute_count = 0;
}

void ame_xsai_plan_cache_clear(void) {
    std::vector<CacheEntry> entries;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        for (auto & entry : g_cache) {
            entries.push_back(std::move(entry.second));
        }
        g_cache.clear();
    }
    for (auto & entry : entries) {
        wait_and_destroy(std::move(entry));
    }
}

size_t ame_xsai_plan_cache_live(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_cache.size();
}

extern "C" void ame_xsai_plan_cache_retain(const void * const * ops, size_t count) {
    std::vector<CacheEntry> dropped;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        std::unordered_set<const void *> live;
        for (size_t i = 0; i < count; ++i) {
            if (ops != nullptr) {
                live.insert(ops[i]);
            }
        }
        for (auto it = g_cache.begin(); it != g_cache.end(); ) {
            if (live.find(it->first) == live.end()) {
                dropped.push_back(std::move(it->second));
                it = g_cache.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto & entry : dropped) {
        wait_and_destroy(std::move(entry));
    }
}

xsai_status_t ame_xsai_weight_refresh_q8(
    const void * tensor,
    const void * q8_data,
    int64_t rows,
    int64_t logical_k,
    uint64_t generation) {
    if (tensor == nullptr || rows < 0 || logical_k <= 0) {
        return XSAI_STATUS_INVALID_ARGUMENT;
    }
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_refreshing = tensor;
    }
    std::vector<CacheEntry> retired;
    int retired_count = 0;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        retired_count = retire_weight_locked_copy(tensor, &retired);
    }
    for (auto & entry : retired) {
        wait_and_destroy(std::move(entry));
    }
    destroy_weight_slot(tensor, retired_count > 0);

    const int64_t blocks = (logical_k + 31) / 32;
    int64_t shape[2] = {rows, blocks * 34};
    DLTensor input{};
    input.data = const_cast<void *>(q8_data);
    input.device = {kDLCPU, 0};
    input.ndim = 2;
    input.dtype = {kDLUInt, 8, 1};
    input.shape = shape;
    xsai_weight_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = XSAI_BLAS_ABI_VERSION;
    desc.inputs = &input;
    desc.input_count = 1;
    desc.generation = generation;
    desc.source_format = XSAI_WEIGHT_SOURCE_Q8_0;
    desc.logical_k = logical_k;
    desc.native_capabilities = XSAI_CAP_RVV | XSAI_CAP_AME;

    xsai_weight_plan_t * plan = nullptr;
    xsai_status_t status = xsai_weight_plan_create(&desc, &plan);
    void * arena = nullptr;
    size_t arena_bytes = 0;
    if (status == XSAI_STATUS_SUCCESS) {
        xsai_weight_requirements_t requirements{};
        requirements.struct_size = sizeof(requirements);
        requirements.abi_version = XSAI_BLAS_ABI_VERSION;
        status = xsai_weight_plan_get_requirements(plan, &requirements);
        if (status == XSAI_STATUS_SUCCESS && requirements.arena_size) {
            if (posix_memalign(&arena, requirements.arena_alignment,
                               requirements.arena_size) != 0) {
                arena = nullptr;
                status = XSAI_STATUS_OUT_OF_MEMORY;
            }
        }
        if (status == XSAI_STATUS_SUCCESS) {
            arena_bytes = requirements.arena_size;
            ObserveSpan span(AME_XSAI_OBS_PREPARE);
            status = xsai_weight_plan_prepare(plan, arena, requirements.arena_size);
        }
    }
    if (status != XSAI_STATUS_SUCCESS) {
        xsai_weight_plan_destroy(plan);
        std::free(arena);
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_refreshing == tensor) {
            g_refreshing = nullptr;
        }
        return status;
    }
    {
        std::lock_guard<std::mutex> lock(g_weight_mu);
        g_weights[tensor] = WeightSlot{plan, arena, generation, arena_bytes};
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_refreshing == tensor) {
        g_refreshing = nullptr;
    }
    return XSAI_STATUS_SUCCESS;
}

void ame_xsai_weight_release(const void * tensor) {
    if (tensor == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_refreshing = tensor;
    }
    std::vector<CacheEntry> retired;
    int retired_count = 0;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        retired_count = retire_weight_locked_copy(tensor, &retired);
    }
    for (auto & entry : retired) {
        wait_and_destroy(std::move(entry));
    }
    destroy_weight_slot(tensor, retired_count > 0);
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_refreshing == tensor) {
        g_refreshing = nullptr;
    }
}

const xsai_weight_plan_t * ame_xsai_weight_plan(const void * tensor) {
    std::lock_guard<std::mutex> lock(g_weight_mu);
    const auto it = g_weights.find(tensor);
    if (it == g_weights.end()) {
        return nullptr;
    }
    return it->second.plan;
}

uint64_t ame_xsai_weight_generation(const void * tensor) {
    std::lock_guard<std::mutex> lock(g_weight_mu);
    const auto it = g_weights.find(tensor);
    if (it == g_weights.end()) {
        return 0;
    }
    return it->second.generation;
}

uint64_t ame_xsai_op_plan_destroy_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_op_destroy_count;
}

uint64_t ame_xsai_weight_plan_destroy_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_weight_destroy_count;
}

int ame_xsai_destroy_order_ok(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_order_violations == 0 && g_weight_destroy_count > 0 &&
        g_last_op_seq != 0 && g_last_op_seq < g_last_weight_seq;
}

void ame_xsai_destroy_order_reset(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_seq = 0;
    g_last_op_seq = 0;
    g_last_weight_seq = 0;
    g_op_destroy_count = 0;
    g_weight_destroy_count = 0;
    g_order_violations = 0;
}
