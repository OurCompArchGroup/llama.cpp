#include "ame-xsai-plan.h"

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

static ame_xsai_q8_key make_key(int64_t m, int64_t n, int64_t k,
                                uint64_t generation) {
    ame_xsai_q8_key key = {};
    key.m = m;
    key.n = n;
    key.k = k;
    key.act_nb0 = sizeof(float);
    key.act_nb1 = (uint64_t) k * sizeof(float);
    key.weight_nb0 = 34;
    key.weight_nb1 = (uint64_t) ((k + 31) / 32) * 34u;
    key.dst_nb0 = sizeof(float);
    key.dst_nb1 = (uint64_t) n * sizeof(float);
    key.src0_ne2 = 1;
    key.src0_ne3 = 1;
    key.src1_ne2 = 1;
    key.src1_ne3 = 1;
    key.dst_ne2 = 1;
    key.dst_ne3 = 1;
    key.weight_generation = generation;
    return key;
}

static void pack_row(uint8_t * dst, float scale_is_one, const int8_t * codes,
                     int64_t k) {
    (void) scale_is_one;
    const uint16_t one = 0x3c00;
    const int64_t blocks = (k + 31) / 32;
    std::memset(dst, 0, (size_t) blocks * 34u);
    for (int64_t block = 0; block < blocks; ++block) {
        std::memcpy(dst + block * 34, &one, sizeof(one));
        const int64_t base = block * 32;
        const int64_t valid = k - base < 32 ? k - base : 32;
        std::memcpy(dst + block * 34 + 2, codes + base, (size_t) valid);
    }
}

int main() {
    const int64_t m = 2;
    const int64_t n = 2;
    const int64_t k = 32;
    float activation[2 * 32];
    int8_t codes[2 * 32];
    uint8_t packed[2 * 34];
    for (int64_t i = 0; i < m * k; ++i) {
        activation[i] = 1.0f;
    }
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            codes[row * k + col] = 1;
        }
        pack_row(packed + row * 34, 1.0f, codes + row * k, k);
    }

    int sentinel_op = 1;
    int other_op = 2;
    ame_xsai_q8_key key = make_key(m, n, k, 1);
    ame_xsai_plan_cache_clear();
    ame_xsai_plan_cache_reset_counts();

    xsai_op_plan_t * plan = ame_xsai_q8_cache_get_or_create(&sentinel_op, &key);
    expect(plan != nullptr, "first work_size creates a plan");
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    expect(xsai_op_plan_get_requirements(plan, &requirements) == XSAI_STATUS_SUCCESS &&
               requirements.workspace_size > 0,
           "requirements come from the cached plan");

    xsai_op_plan_t * again = ame_xsai_q8_cache_get_or_create(&sentinel_op, &key);
    expect(again == plan, "second work_size returns the same plan");
    expect(ame_xsai_plan_create_count() == 1, "second work_size does not create");
    expect(ame_xsai_plan_cache_hit_count() == 1, "second work_size hits the cache");

    std::vector<unsigned char> workspace(
        requirements.workspace_size + requirements.workspace_alignment);
    void * aligned = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(workspace.data()) +
         requirements.workspace_alignment - 1) &
        ~(uintptr_t) (requirements.workspace_alignment - 1));
    float out_a[4];
    float out_b[4];
    std::memset(out_a, 0, sizeof(out_a));
    std::memset(out_b, 0, sizeof(out_b));
    ame_xsai_q8_bind bind = {activation, packed, out_a, aligned,
                             requirements.workspace_size};
    expect(ame_xsai_q8_cache_execute(plan, &key, &bind) == XSAI_STATUS_SUCCESS,
           "first execute succeeds");
    /* amax=1, scale_a=1/127, codes=127, weight code=1, scale_b=1, K=32.
     * each output is 32 * 127 * (1/127) = 32. ggml store is m-major. */
    bool values_ok = true;
    for (float value : out_a) {
        values_ok = values_ok && value == 32.0f;
    }
    expect(values_ok, "Q8 plan matches the scalar reference");

    const uint64_t creates_before_refresh = ame_xsai_plan_create_count();
    bind.output = out_b;
    expect(ame_xsai_q8_cache_lookup(&sentinel_op, &key) == plan,
           "lookup finds the plan without creating");
    expect(ame_xsai_q8_cache_execute(plan, &key, &bind) == XSAI_STATUS_SUCCESS,
           "refreshed output pointer executes the same plan");
    expect(ame_xsai_plan_create_count() == creates_before_refresh,
           "refreshing the output pointer does not select again");
    bool refreshed_ok = true;
    for (int i = 0; i < 4; ++i) {
        refreshed_ok = refreshed_ok && out_b[i] == out_a[i];
    }
    expect(refreshed_ok, "refreshed output matches the first execute");

    float sentinel[4];
    for (float & value : sentinel) {
        value = 7.0f;
    }
    bind.output = sentinel;
    bind.workspace_size = 0;
    expect(ame_xsai_q8_cache_execute(plan, &key, &bind) ==
               XSAI_STATUS_INSUFFICIENT_WORKSPACE,
           "short workspace is rejected");
    bool untouched = true;
    for (float value : sentinel) {
        untouched = untouched && value == 7.0f;
    }
    expect(untouched, "short workspace does not write output");

    ame_xsai_q8_key changed_k = key;
    changed_k.k = 64;
    changed_k.act_nb1 = 64 * sizeof(float);
    changed_k.weight_nb1 = 2 * 34;
    expect(ame_xsai_q8_cache_lookup(&sentinel_op, &changed_k) == nullptr,
           "lookup misses after K changes");
    xsai_op_plan_t * replanned =
        ame_xsai_q8_cache_get_or_create(&sentinel_op, &changed_k);
    expect(replanned != nullptr && replanned != plan,
           "changing K creates a new plan");
    expect(ame_xsai_plan_create_count() == creates_before_refresh + 1,
           "changing K increments plan create");

    ame_xsai_q8_key changed_gen = changed_k;
    changed_gen.weight_generation = 9;
    xsai_op_plan_t * generated =
        ame_xsai_q8_cache_get_or_create(&sentinel_op, &changed_gen);
    expect(generated != nullptr && generated != replanned,
           "changing weight generation creates a new plan");
    expect(ame_xsai_plan_create_count() == creates_before_refresh + 2,
           "changing weight generation increments plan create");

    ame_xsai_q8_key other = make_key(m, n, k, 1);
    expect(ame_xsai_q8_cache_get_or_create(&other_op, &other) != nullptr,
           "a second ggml op gets its own plan");
    const size_t live_before = ame_xsai_plan_cache_live();
    const void * keep[] = {&sentinel_op};
    ame_xsai_plan_cache_retain(keep, 1);
    expect(ame_xsai_plan_cache_live() == 1 && live_before == 2,
           "graph rebuild drops plans for nodes that are gone");

    const uint64_t creates_before_options = ame_xsai_plan_create_count();
    ame_xsai_q8_key options = changed_gen;
    options.explicit_options = 1;
    expect(ame_xsai_q8_cache_get_or_create(&sentinel_op, &options) != nullptr &&
               ame_xsai_plan_create_count() == creates_before_options + 1,
           "changing explicit options creates a new plan");

    ame_xsai_plan_cache_clear();
    expect(ame_xsai_plan_cache_live() == 0, "cache clear destroys plans");
    return g_failures == 0 ? 0 : 1;
}
