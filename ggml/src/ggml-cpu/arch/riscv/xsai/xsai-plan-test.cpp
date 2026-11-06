#include "xsai-plan.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

static int g_failures = 0;

static void expect(bool cond, const char * name) {
    if (cond) {
        std::printf("PASS %s\n", name);
    } else {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

/* Q8_0 row: one fp16 scale followed by 32 int8 codes per 34-byte block. */
static void pack_row(uint8_t * dst, uint16_t scale_bits, int8_t code, int64_t k) {
    const int64_t blocks = (k + 31) / 32;
    std::memset(dst, 0, (size_t) blocks * 34u);
    for (int64_t block = 0; block < blocks; ++block) {
        std::memcpy(dst + block * 34, &scale_bits, sizeof(scale_bits));
        const int64_t base = block * 32;
        const int64_t valid = k - base < 32 ? k - base : 32;
        std::memset(dst + block * 34 + 2, code, (size_t) valid);
    }
}

static xsai_q8_key make_key(xsai_op_type_t op, int64_t m, int64_t n, int64_t k,
                            uint64_t generation, const void * identity,
                            const xsai_weight_plan_t * prepared) {
    xsai_q8_key key = {};
    key.op = op;
    key.m = m;
    key.n = n;
    key.k = k;
    key.blocks = (k + 31) / 32;
    key.weight_generation = generation;
    key.weight_identity = identity;
    key.prepared_weight = prepared;
    xsai_q8_key_seal(&key);
    return key;
}

static void * align_workspace(std::vector<unsigned char> & storage,
                              size_t alignment) {
    const size_t effective = alignment == 0 ? 1 : alignment;
    return reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(storage.data()) + effective - 1) &
        ~(uintptr_t) (effective - 1));
}

static bool workspace_from_requirements(const xsai_op_plan_t * plan,
                                        std::vector<unsigned char> & storage,
                                        size_t & size) {
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    if (xsai_op_plan_get_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
        return false;
    }
    const size_t alignment = requirements.workspace_alignment == 0
        ? 1 : requirements.workspace_alignment;
    storage.assign(requirements.workspace_size + alignment, 0);
    size = requirements.workspace_size;
    return true;
}

int main() {
    const int64_t n = 2;
    const int64_t k = 32;
    const int64_t m = 2;
    float activation[m * k];
    float single_activation[k];
    uint8_t packed[n * 34];
    for (int64_t i = 0; i < m * k; ++i) {
        activation[i] = 1.0f;
    }
    for (int64_t i = 0; i < k; ++i) {
        single_activation[i] = 1.0f;
    }
    for (int64_t row = 0; row < n; ++row) {
        pack_row(packed + row * 34, 0x3c00, 1, k); /* fp16 1.0 */
    }

    int weight_tensor = 0;
    int vector_tensor = 0;
    int wide_tensor = 0;
    int other_op = 0;

    xsai_plan_cache_clear();
    xsai_plan_cache_reset_counts();
    expect(xsai_weight_refresh_q8(&weight_tensor, packed, n, k, 1) ==
               XSAI_STATUS_SUCCESS,
           "Q8 refresh converts and prepares the weight");
    const xsai_weight_plan_t * prepared = xsai_weight_plan(&weight_tensor);
    expect(prepared != nullptr, "refresh publishes a prepared weight");
    expect(xsai_prepared_bytes() > 0, "prepared weight reports owned bytes");

    xsai_q8_key key = make_key(XSAI_OP_GEMM, m, n, k, 1, &weight_tensor, prepared);
    expect(key.fingerprint != 0, "template call fingerprints");
    expect(xsai_q8_cache_get_or_create(&other_op, &key, nullptr) != nullptr,
           "first work_size creates a plan");
    xsai_op_plan_t * plan = xsai_q8_cache_lookup(&other_op, &key);
    expect(plan != nullptr, "lookup finds the cached plan");
    expect(xsai_plan_cached_op(&other_op) == XSAI_OP_GEMM,
           "M > 1 caches a GEMM plan");

    std::vector<unsigned char> storage;
    size_t workspace_size = 0;
    expect(workspace_from_requirements(plan, storage, workspace_size) &&
               workspace_size > 0,
           "dynamic I8 plan reports workspace");
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    xsai_op_plan_get_requirements(plan, &requirements);
    void * aligned = align_workspace(storage, requirements.workspace_alignment);

    float out_a[m * n];
    float out_b[m * n];
    std::memset(out_a, 0, sizeof(out_a));
    std::memset(out_b, 0, sizeof(out_b));
    xsai_q8_bind bind = {activation, out_a, aligned, workspace_size};
    expect(xsai_q8_cache_execute(plan, &key, &bind) == XSAI_STATUS_SUCCESS,
           "first execute succeeds");
    /* amax=1 -> scale 1/127 and codes 127; weight code 1 scale 1; K=32. */
    bool values_ok = true;
    for (float value : out_a) {
        values_ok = values_ok && value == 32.0f;
    }
    expect(values_ok, "GEMM output matches the dequantized product");

    const uint64_t creates_before_reuse = xsai_plan_create_count();
    expect(xsai_q8_cache_get_or_create(&other_op, &key, nullptr) == plan,
           "second work_size returns the same plan");
    expect(xsai_plan_create_count() == creates_before_reuse,
           "second work_size does not create");
    expect(xsai_plan_cache_hit_count() == 1, "second work_size hits the cache");

    bind.output = out_b;
    expect(xsai_q8_cache_execute(plan, &key, &bind) == XSAI_STATUS_SUCCESS,
           "output pointer refresh executes the same plan");
    bool refreshed_ok = true;
    for (int i = 0; i < m * n; ++i) {
        refreshed_ok = refreshed_ok && out_b[i] == out_a[i];
    }
    expect(refreshed_ok, "refreshed output matches the first execute");

    float sentinel[m * n];
    for (float & value : sentinel) {
        value = 7.0f;
    }
    bind.output = sentinel;
    bind.workspace_size = 0;
    expect(xsai_q8_cache_execute(plan, &key, &bind) ==
               XSAI_STATUS_INSUFFICIENT_WORKSPACE,
           "short workspace is rejected");
    bool untouched = true;
    for (float value : sentinel) {
        untouched = untouched && value == 7.0f;
    }
    expect(untouched, "short workspace does not write output");

    /* M = 1 is the vector form through the same prepared weight. */
    xsai_q8_key vector =
        make_key(XSAI_OP_GEMM, 1, n, k, 1, &weight_tensor, prepared);
    expect(vector.fingerprint != key.fingerprint,
           "vector and matrix keys differ");
    xsai_op_plan_t * vector_plan =
        xsai_q8_cache_get_or_create(&vector_tensor, &vector, nullptr);
    expect(vector_plan != nullptr, "M = 1 creates a plan");
    expect(xsai_plan_cached_op(&vector_tensor) == XSAI_OP_GEMM,
           "M = 1 caches a GEMM plan");
    std::vector<unsigned char> vector_storage;
    size_t vector_workspace = 0;
    expect(workspace_from_requirements(vector_plan, vector_storage,
                                       vector_workspace),
           "vector plan reports workspace");
    requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    xsai_op_plan_get_requirements(vector_plan, &requirements);
    void * vector_aligned = align_workspace(vector_storage,
                                          requirements.workspace_alignment);
    float vector_out[n];
    std::memset(vector_out, 0, sizeof(vector_out));
    xsai_q8_bind vector_bind = {single_activation, vector_out, vector_aligned,
                              vector_workspace};
    expect(xsai_q8_cache_execute(vector_plan, &vector, &vector_bind) ==
               XSAI_STATUS_SUCCESS,
           "vector form executes over the prepared weight");
    bool vector_ok = true;
    for (float value : vector_out) {
        vector_ok = vector_ok && value == 32.0f;
    }
    expect(vector_ok, "vector output matches the dequantized product");

    /* A different K needs its own prepared weight and its own plan. */
    const int64_t wide_k = 40;
    uint8_t wide_packed[n * 2 * 34];
    for (int64_t row = 0; row < n; ++row) {
        pack_row(wide_packed + row * 2 * 34, 0x3c00, 1, wide_k);
    }
    expect(xsai_weight_refresh_q8(&wide_tensor, wide_packed, n, wide_k, 1) ==
               XSAI_STATUS_SUCCESS,
           "K = 40 weight prepares");
    xsai_q8_key wide = make_key(XSAI_OP_GEMM, m, n, wide_k, 1, &wide_tensor,
                                xsai_weight_plan(&wide_tensor));
    const uint64_t creates_before_wide = xsai_plan_create_count();
    xsai_op_plan_t * wide_plan =
        xsai_q8_cache_get_or_create(&vector_tensor, &wide, nullptr);
    expect(wide_plan != nullptr && wide_plan != vector_plan,
           "changing K creates a new plan");
    expect(xsai_plan_create_count() == creates_before_wide + 1,
           "changing K increments plan create");

    /* Repreparing the weight retires every op plan before the weight plan. */
    xsai_destroy_order_reset();
    expect(xsai_weight_refresh_q8(&weight_tensor, packed, n, k, 2) ==
               XSAI_STATUS_SUCCESS,
           "generation refresh replaces the weight plan");
    expect(xsai_destroy_order_ok(),
           "refresh destroys op plans before the weight plan");
    const xsai_weight_plan_t * regenerated = xsai_weight_plan(&weight_tensor);
    expect(regenerated != nullptr && regenerated != prepared,
           "refresh publishes a new prepared weight");
    xsai_q8_key life = make_key(XSAI_OP_GEMM, m, n, k, 2, &weight_tensor,
                                regenerated);
    const uint64_t creates_before_life = xsai_plan_create_count();
    xsai_op_plan_t * life_plan =
        xsai_q8_cache_get_or_create(&other_op, &life, nullptr);
    expect(life_plan != nullptr &&
               xsai_plan_create_count() == creates_before_life + 1,
           "the refreshed weight creates a new op plan");
    expect(xsai_q8_cache_lookup(&other_op, &key) == nullptr,
           "the old generation key no longer matches");
    std::vector<unsigned char> life_storage;
    size_t life_workspace = 0;
    expect(workspace_from_requirements(life_plan, life_storage, life_workspace),
           "refreshed plan reports workspace");
    requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    xsai_op_plan_get_requirements(life_plan, &requirements);
    void * life_aligned = align_workspace(life_storage,
                                          requirements.workspace_alignment);
    float life_out[m * n];
    std::memset(life_out, 0, sizeof(life_out));
    xsai_q8_bind life_bind = {activation, life_out, life_aligned, life_workspace};
    expect(xsai_q8_cache_execute(life_plan, &life, &life_bind) ==
               XSAI_STATUS_SUCCESS,
           "refreshed prepared weight executes");

    /* Verbatim copy: code 3 with fp16 scale 0.5 gives 3 * 0.5 * K. */
    int scale_tensor = 0;
    const int64_t scale_k = 40;
    uint8_t scale_packed[2 * 34];
    pack_row(scale_packed, 0x3800, 3, scale_k); /* fp16 0.5 */
    expect(xsai_weight_refresh_q8(&scale_tensor, scale_packed, 1, scale_k, 1) ==
               XSAI_STATUS_SUCCESS,
           "fractional fp16 scale prepares");
    float scale_activation[scale_k];
    for (float & value : scale_activation) {
        value = 1.0f;
    }
    xsai_q8_key scaled = make_key(XSAI_OP_GEMM, 1, 1, scale_k, 1, &scale_tensor,
                                  xsai_weight_plan(&scale_tensor));
    xsai_op_plan_t * scale_plan =
        xsai_q8_cache_get_or_create(&vector_tensor, &scaled, nullptr);
    std::vector<unsigned char> scale_storage;
    size_t scale_workspace = 0;
    expect(scale_plan != nullptr &&
               workspace_from_requirements(scale_plan, scale_storage,
                                           scale_workspace),
           "scaled weight creates a plan");
    requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    xsai_op_plan_get_requirements(scale_plan, &requirements);
    void * scale_aligned = align_workspace(scale_storage,
                                           requirements.workspace_alignment);
    float scale_out = 0.0f;
    xsai_q8_bind scale_bind = {scale_activation, &scale_out, scale_aligned,
                               scale_workspace};
    expect(xsai_q8_cache_execute(scale_plan, &scaled, &scale_bind) ==
               XSAI_STATUS_SUCCESS &&
               scale_out == 3.0f * 0.5f * (float) scale_k,
           "codes and fp16 scales are copied verbatim");

    const size_t live_before = xsai_plan_cache_live();
    const void * keep[] = {&other_op};
    xsai_plan_cache_retain(keep, 1);
    expect(xsai_plan_cache_live() == 1 && live_before > 1,
           "graph rebuild drops plans for nodes that are gone");

    xsai_plan_cache_clear();
    expect(xsai_plan_cache_live() == 0, "cache clear destroys plans");

    xsai_weight_release(&weight_tensor);
    xsai_weight_release(&wide_tensor);
    xsai_weight_release(&scale_tensor);
    expect(xsai_weight_plan(&weight_tensor) == nullptr,
           "release drops the weight plan");
    expect(xsai_destroy_order_ok(),
           "release destroys op plans before the weight plan");
    return g_failures == 0 ? 0 : 1;
}
