#include "xsai-plan.h"
#include "xsai-observe.h"

#include <algorithm>
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
        xsai_observe_span_begin(phase, &token);
    }
    ~ObserveSpan() {
        xsai_observe_span_end(phase, token);
    }
};

namespace {

std::mutex g_mu;
std::condition_variable g_cv;
const void * g_refreshing = nullptr;

struct CacheEntry {
    bool general = false;
    xsai_q8_key key{};
    uint64_t general_key = 0;
    xsai_op_plan_t * plan = nullptr;
    std::shared_ptr<std::atomic<int>> inflight = std::make_shared<std::atomic<int>>(0);
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
    size_t arena_bytes = 0;
    void * payload = nullptr;
    void * scale = nullptr;
    uint64_t generation = 0;
};

std::mutex g_weight_mu;
std::unordered_map<const void *, WeightSlot> g_weights;
uint64_t g_prepared_peak_bytes = 0;

bool key_equal(const xsai_q8_key & lhs, const xsai_q8_key & rhs) {
    return lhs.fingerprint == rhs.fingerprint && lhs.op == rhs.op &&
        lhs.m == rhs.m && lhs.k == rhs.k && lhs.n == rhs.n &&
        lhs.blocks == rhs.blocks &&
        lhs.weight_generation == rhs.weight_generation &&
        lhs.weight_identity == rhs.weight_identity &&
        lhs.prepared_weight == rhs.prepared_weight;
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

bool layout_supported(const xsai_q8_key * key) {
    if (key == nullptr || key->prepared_weight == nullptr) {
        return false;
    }
    if (key->fingerprint == 0 || key->m <= 0 || key->n <= 0 || key->k <= 0) {
        return false;
    }
    if (key->blocks != (key->k + 31) / 32) {
        return false;
    }
    return key->op == XSAI_OP_GEMM;
}

}  // namespace

bool xsai_q8_key_seal(xsai_q8_key * key) {
    if (key == nullptr || key->blocks <= 0) {
        return false;
    }
    xsai_q8_call parts;
    xsai_q8_call_init(key, nullptr, &parts);
    key->fingerprint = xsai_op_call_fingerprint(&parts.call);
    return key->fingerprint != 0;
}

void xsai_q8_call_init(const xsai_q8_key * key, const xsai_q8_bind * bind,
                       xsai_q8_call * parts) {
    std::memset(parts, 0, sizeof(*parts));
    const bool vector = key->m == 1;

    parts->a_shape[0] = vector ? key->k : key->m;
    parts->a_shape[1] = key->k;
    parts->b_shape[0] = key->k;
    parts->b_shape[1] = key->n;
    parts->b_strides[0] = 1;
    parts->b_strides[1] = key->k;
    parts->scale_shape[0] = key->blocks;
    parts->scale_shape[1] = key->n;
    parts->scale_strides[0] = 1;
    parts->scale_strides[1] = key->blocks;
    parts->c_shape[0] = vector ? key->n : key->m;
    parts->c_shape[1] = key->n;

    parts->inputs[0].data = bind ? const_cast<void *>(bind->activation) : nullptr;
    parts->inputs[0].device = {kDLCPU, 0};
    parts->inputs[0].ndim = vector ? 1 : 2;
    parts->inputs[0].dtype = {kDLFloat, 32, 1};
    parts->inputs[0].shape = parts->a_shape;

    /* Metadata only: the prepared weight publishes these views at execute. */
    parts->inputs[1].data = nullptr;
    parts->inputs[1].device = {kDLCPU, 0};
    parts->inputs[1].ndim = 2;
    parts->inputs[1].dtype = {kDLInt, 8, 1};
    parts->inputs[1].shape = parts->b_shape;
    parts->inputs[1].strides = parts->b_strides;

    parts->inputs[2].data = nullptr;
    parts->inputs[2].device = {kDLCPU, 0};
    parts->inputs[2].ndim = 2;
    parts->inputs[2].dtype = {kDLFloat, 16, 1};
    parts->inputs[2].shape = parts->scale_shape;
    parts->inputs[2].strides = parts->scale_strides;

    parts->output.data = bind ? bind->output : nullptr;
    parts->output.device = {kDLCPU, 0};
    parts->output.ndim = vector ? 1 : 2;
    parts->output.dtype = {kDLFloat, 32, 1};
    parts->output.shape = parts->c_shape;

    parts->params.struct_size = sizeof(parts->params);
    parts->params.abi_version = XSAI_BLAS_ABI_VERSION;
    parts->params.b_format = XSAI_QUANT_I8_GROUP32;

    parts->call.struct_size = sizeof(parts->call);
    parts->call.abi_version = XSAI_BLAS_ABI_VERSION;
    parts->call.op = key->op;
    parts->call.input_count = 3;
    parts->call.inputs = parts->inputs;
    parts->call.output_count = 1;
    parts->call.outputs = &parts->output;
    parts->call.op_params = &parts->params;
    parts->call.prepared_weight = key->prepared_weight;
    if (bind != nullptr) {
        parts->call.workspace = bind->workspace;
        parts->call.workspace_size = bind->workspace_size;
    }
}

xsai_op_plan_t * xsai_q8_cache_get_or_create(const void * op,
                                             const xsai_q8_key * key,
                                             xsai_status_t * status_out) {
    if (status_out != nullptr) {
        *status_out = XSAI_STATUS_SUCCESS;
    }
    if (op == nullptr || key == nullptr) {
        if (status_out != nullptr) {
            *status_out = XSAI_STATUS_INVALID_ARGUMENT;
        }
        return nullptr;
    }
    CacheEntry replaced;
    bool have_replaced = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (key->weight_identity != nullptr && key->weight_identity == g_refreshing) {
            if (status_out != nullptr) {
                *status_out = XSAI_STATUS_PREPARATION_REQUIRED;
            }
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
        if (status_out != nullptr) {
            *status_out = XSAI_STATUS_UNSUPPORTED;
        }
        return nullptr;
    }

    xsai_q8_call parts;
    xsai_q8_call_init(key, nullptr, &parts);
    xsai_op_plan_t * plan = nullptr;
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_key_miss_count;
    }
    {
        ObserveSpan span(XSAI_OBS_PLAN_CREATE);
        status = xsai_op_plan_create(&parts.call, &plan);
    }
    if (status != XSAI_STATUS_SUCCESS || plan == nullptr) {
        xsai_op_plan_destroy(plan);
        if (status_out != nullptr) {
            *status_out = status != XSAI_STATUS_SUCCESS ? status
                                                        : XSAI_STATUS_UNSUPPORTED;
        }
        return nullptr;
    }
    CacheEntry stale;
    bool have_stale = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (key->weight_identity != nullptr && key->weight_identity == g_refreshing) {
            destroy_plan_locked(plan);
            if (status_out != nullptr) {
                *status_out = XSAI_STATUS_PREPARATION_REQUIRED;
            }
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

xsai_op_plan_t * xsai_q8_cache_lookup(const void * op,
                                      const xsai_q8_key * key) {
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

xsai_status_t xsai_q8_cache_execute(xsai_op_plan_t * plan,
                                    const xsai_q8_key * key,
                                    const xsai_q8_bind * bind) {
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
    xsai_q8_call parts;
    xsai_q8_call_init(key, bind, &parts);
    xsai_status_t status = XSAI_STATUS_SUCCESS;
    {
        ObserveSpan span(XSAI_OBS_EXECUTE);
        status = xsai_op_plan_execute(plan, &parts.call);
    }
    if (status == XSAI_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_execute_count;
    }
    inflight->fetch_sub(1);
    g_cv.notify_all();
    return status;
}

xsai_op_type_t xsai_plan_cached_op(const void * op) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_cache.find(op);
    if (it == g_cache.end() || it->second.general) {
        return (xsai_op_type_t) -1;
    }
    return it->second.key.op;
}

xsai_op_plan_t * xsai_general_cache_get_or_create(const void * op,
                                                  uint64_t key,
                                                  const xsai_op_call_t * call_template,
                                                  xsai_status_t * status_out) {
    if (status_out != nullptr) {
        *status_out = XSAI_STATUS_SUCCESS;
    }
    if (op == nullptr || call_template == nullptr || key == 0) {
        if (status_out != nullptr) {
            *status_out = XSAI_STATUS_INVALID_ARGUMENT;
        }
        return nullptr;
    }
    CacheEntry replaced;
    bool have_replaced = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_cache.find(op);
        if (it != g_cache.end() && it->second.general &&
            it->second.general_key == key) {
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
        ObserveSpan span(XSAI_OBS_PLAN_CREATE);
        status = xsai_op_plan_create(call_template, &plan);
    }
    if (status != XSAI_STATUS_SUCCESS || plan == nullptr) {
        xsai_op_plan_destroy(plan);
        if (status_out != nullptr) {
            *status_out = status != XSAI_STATUS_SUCCESS ? status
                                                        : XSAI_STATUS_UNSUPPORTED;
        }
        return nullptr;
    }

    CacheEntry created;
    created.plan = plan;
    created.inflight = std::make_shared<std::atomic<int>>(0);
    created.general = true;
    created.general_key = key;
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

xsai_op_plan_t * xsai_general_cache_lookup(const void * op, uint64_t key) {
    if (op == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_cache.find(op);
    if (it == g_cache.end() || !it->second.general ||
        it->second.general_key != key) {
        return nullptr;
    }
    return it->second.plan;
}

xsai_status_t xsai_general_cache_execute(xsai_op_plan_t * plan,
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
        ObserveSpan span(XSAI_OBS_EXECUTE);
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

xsai_status_t xsai_plan_read_requirements(
    const xsai_op_plan_t * plan, xsai_op_requirements_t * requirements) {
    ObserveSpan span(XSAI_OBS_WORKSPACE);
    return xsai_op_plan_get_requirements(plan, requirements);
}

uint64_t xsai_plan_create_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_create_count;
}

uint64_t xsai_plan_cache_hit_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_hit_count;
}

uint64_t xsai_plan_key_miss_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_key_miss_count;
}

uint64_t xsai_plan_execute_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_execute_count;
}

uint64_t xsai_prepared_bytes(void) {
    std::lock_guard<std::mutex> lock(g_weight_mu);
    uint64_t bytes = 0;
    for (const auto & entry : g_weights) {
        bytes += entry.second.arena_bytes;
    }
    return bytes;
}

uint64_t xsai_prepared_peak_bytes(void) {
    std::lock_guard<std::mutex> lock(g_weight_mu);
    return g_prepared_peak_bytes;
}

void xsai_plan_cache_reset_counts(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_create_count = 0;
    g_hit_count = 0;
    g_key_miss_count = 0;
    g_execute_count = 0;
}

void xsai_plan_cache_clear(void) {
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

size_t xsai_plan_cache_live(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_cache.size();
}

extern "C" void xsai_plan_cache_retain(const void * const * ops, size_t count) {
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

namespace {

/* Frees the adapter-owned conversion buffers behind one weight plan. */
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
    std::free(slot.payload);
    std::free(slot.scale);
}

int retire_weight_locked_copy(const void * tensor, std::vector<CacheEntry> * retired) {
    int count = 0;
    for (auto it = g_cache.begin(); it != g_cache.end(); ) {
        if (!it->second.general && it->second.key.weight_identity == tensor) {
            retired->push_back(std::move(it->second));
            it = g_cache.erase(it);
            ++count;
        } else {
            ++it;
        }
    }
    return count;
}

}  // namespace

xsai_status_t xsai_weight_refresh_q8(
    const void * tensor,
    const void * q8_data,
    int64_t rows,
    int64_t logical_k,
    uint64_t generation) {
    if (tensor == nullptr || q8_data == nullptr || rows <= 0 || logical_k <= 0) {
        return XSAI_STATUS_INVALID_ARGUMENT;
    }
    const int64_t blocks = (logical_k + 31) / 32;
    if ((uint64_t)rows > SIZE_MAX / (uint64_t)logical_k ||
        (uint64_t)rows > SIZE_MAX / ((uint64_t)blocks * sizeof(uint16_t))) {
        return XSAI_STATUS_INVALID_ARGUMENT;
    }
    const size_t payload_bytes = (size_t)rows * (size_t)logical_k;
    const size_t scale_bytes = (size_t)rows * (size_t)blocks * sizeof(uint16_t);

    /*
     * Q8_0 packs one fp16 scale and 32 int8 quants per 34-byte block. The
     * target I8_GROUP32 layout is [N, K] int8 and [N, ceil(K/32)] fp16, so a
     * row copy is a block copy: the scale bytes and code bytes move verbatim.
     */
    auto * payload = static_cast<int8_t *>(std::malloc(payload_bytes));
    auto * scale = static_cast<uint16_t *>(std::malloc(scale_bytes));
    if (payload == nullptr || scale == nullptr) {
        std::free(payload);
        std::free(scale);
        return XSAI_STATUS_OUT_OF_MEMORY;
    }
    const auto * source = static_cast<const uint8_t *>(q8_data);
    for (int64_t row = 0; row < rows; ++row) {
        const uint8_t * source_row = source + (size_t)row * (size_t)blocks * 34u;
        int8_t * payload_row = payload + (size_t)row * (size_t)logical_k;
        uint16_t * scale_row = scale + (size_t)row * (size_t)blocks;
        for (int64_t block = 0; block < blocks; ++block) {
            const uint8_t * source_block = source_row + (size_t)block * 34u;
            std::memcpy(&scale_row[block], source_block, sizeof(uint16_t));
            const int64_t offset = block * 32;
            const int64_t take = logical_k - offset < 32 ? logical_k - offset : 32;
            std::memcpy(payload_row + offset, source_block + 2, (size_t)take);
        }
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

    int64_t payload_shape[2] = {logical_k, rows};
    int64_t payload_strides[2] = {1, logical_k};
    int64_t scale_shape[2] = {blocks, rows};
    int64_t scale_strides[2] = {1, blocks};
    DLTensor payload_tensor{};
    payload_tensor.data = payload;
    payload_tensor.device = {kDLCPU, 0};
    payload_tensor.ndim = 2;
    payload_tensor.dtype = {kDLInt, 8, 1};
    payload_tensor.shape = payload_shape;
    payload_tensor.strides = payload_strides;
    DLTensor scale_tensor{};
    scale_tensor.data = scale;
    scale_tensor.device = {kDLCPU, 0};
    scale_tensor.ndim = 2;
    scale_tensor.dtype = {kDLFloat, 16, 1};
    scale_tensor.shape = scale_shape;
    scale_tensor.strides = scale_strides;
    xsai_weight_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = XSAI_BLAS_ABI_VERSION;
    desc.payload = &payload_tensor;
    desc.scale = &scale_tensor;
    desc.format = XSAI_QUANT_I8_GROUP32;
    desc.generation = generation;
    desc.native_capabilities = XSAI_CAP_AME;

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
            const size_t alignment =
                requirements.arena_alignment ? requirements.arena_alignment : 1;
            if (posix_memalign(&arena, alignment, requirements.arena_size) != 0) {
                arena = nullptr;
                status = XSAI_STATUS_OUT_OF_MEMORY;
            }
        }
        if (status == XSAI_STATUS_SUCCESS) {
            arena_bytes = requirements.arena_size;
            ObserveSpan span(XSAI_OBS_PREPARE);
            status = xsai_weight_plan_prepare(plan, arena, requirements.arena_size);
        }
    }
    if (status != XSAI_STATUS_SUCCESS) {
        xsai_weight_plan_destroy(plan);
        std::free(arena);
        std::free(payload);
        std::free(scale);
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_refreshing == tensor) {
            g_refreshing = nullptr;
        }
        return status;
    }
    {
        std::lock_guard<std::mutex> lock(g_weight_mu);
        WeightSlot slot;
        slot.plan = plan;
        slot.arena = arena;
        slot.arena_bytes = arena_bytes;
        slot.payload = payload;
        slot.scale = scale;
        slot.generation = generation;
        g_weights[tensor] = slot;
        uint64_t live = 0;
        for (const auto & entry : g_weights) {
            live += entry.second.arena_bytes;
        }
        g_prepared_peak_bytes = std::max(g_prepared_peak_bytes, live);
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_refreshing == tensor) {
        g_refreshing = nullptr;
    }
    return XSAI_STATUS_SUCCESS;
}

void xsai_weight_release(const void * tensor) {
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

const xsai_weight_plan_t * xsai_weight_plan(const void * tensor) {
    std::lock_guard<std::mutex> lock(g_weight_mu);
    const auto it = g_weights.find(tensor);
    if (it == g_weights.end()) {
        return nullptr;
    }
    return it->second.plan;
}

uint64_t xsai_weight_generation(const void * tensor) {
    std::lock_guard<std::mutex> lock(g_weight_mu);
    const auto it = g_weights.find(tensor);
    if (it == g_weights.end()) {
        return 0;
    }
    return it->second.generation;
}

uint64_t xsai_op_plan_destroy_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_op_destroy_count;
}

uint64_t xsai_weight_plan_destroy_count(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_weight_destroy_count;
}

int xsai_destroy_order_ok(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_order_violations == 0 && g_weight_destroy_count > 0 &&
        g_last_op_seq != 0 && g_last_op_seq < g_last_weight_seq;
}

void xsai_destroy_order_reset(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_seq = 0;
    g_last_op_seq = 0;
    g_last_weight_seq = 0;
    g_op_destroy_count = 0;
    g_weight_destroy_count = 0;
    g_order_violations = 0;
}
