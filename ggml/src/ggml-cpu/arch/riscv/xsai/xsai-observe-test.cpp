#include "xsai-observe.h"
#include "xsai-plan.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

static bool selection_equal(const xsai_op_requirements_t & lhs,
                            const xsai_op_requirements_t & rhs) {
    return lhs.selected_policy == rhs.selected_policy &&
        lhs.selected_capabilities == rhs.selected_capabilities &&
        lhs.selected_implementation == rhs.selected_implementation &&
        lhs.selected_tactic == rhs.selected_tactic &&
        lhs.workspace_size == rhs.workspace_size &&
        lhs.workspace_alignment == rhs.workspace_alignment;
}

int main() {
    xsai_observe_export_t sample = {};
    xsai_observe_export(&sample);
    expect(xsai_observe_enabled() == 0, "observation starts disabled");
    expect(sample.prepare_count == 0 && sample.plan_create_count == 0 &&
               sample.workspace_count == 0 && sample.execute_count == 0 &&
               sample.kernel_count == 0,
           "disabled export is empty");

    int weight_tensor = 0;
    int op = 1;
    uint8_t q8_row[34];
    const uint16_t one = 0x3c00;
    std::memset(q8_row, 0, sizeof(q8_row));
    std::memcpy(q8_row, &one, sizeof(one));
    for (int i = 2; i < 34; ++i) {
        q8_row[i] = 1;
    }
    float activation[32];
    for (float & value : activation) {
        value = 1.0f;
    }

    xsai_plan_cache_clear();
    expect(xsai_weight_refresh_q8(&weight_tensor, q8_row, 1, 32, 1) ==
               XSAI_STATUS_SUCCESS,
           "prepare succeeds while observation is off");
    xsai_q8_key key = make_key(XSAI_OP_GEMM, 1, 1, 32, 1, &weight_tensor,
                               xsai_weight_plan(&weight_tensor));
    xsai_op_plan_t * quiet =
        xsai_q8_cache_get_or_create(&op, &key);
    xsai_op_requirements_t quiet_requirements = {};
    quiet_requirements.struct_size = sizeof(quiet_requirements);
    quiet_requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    expect(quiet != nullptr &&
               xsai_plan_read_requirements(quiet, &quiet_requirements) ==
                   XSAI_STATUS_SUCCESS,
           "selection while observation is off");
    xsai_observe_export(&sample);
    expect(sample.prepare_count == 0 && sample.plan_create_count == 0 &&
               sample.workspace_count == 0 && sample.execute_count == 0,
           "disabled hooks do not record");
    xsai_plan_cache_clear();
    xsai_weight_release(&weight_tensor);

    xsai_observe_set_enabled(1);
    xsai_observe_reset();
    expect(xsai_weight_refresh_q8(&weight_tensor, q8_row, 1, 32, 1) ==
               XSAI_STATUS_SUCCESS,
           "prepare succeeds while observation is on");
    key = make_key(XSAI_OP_GEMM, 1, 1, 32, 1, &weight_tensor,
                   xsai_weight_plan(&weight_tensor));
    xsai_op_plan_t * plan = xsai_q8_cache_get_or_create(&op, &key);
    xsai_op_requirements_t requirements = {};
    requirements.struct_size = sizeof(requirements);
    requirements.abi_version = XSAI_BLAS_ABI_VERSION;
    expect(plan != nullptr &&
               xsai_plan_read_requirements(plan, &requirements) ==
                   XSAI_STATUS_SUCCESS,
           "plan create and workspace read");
    expect(selection_equal(quiet_requirements, requirements),
           "observation does not change the frozen selection");
    xsai_op_plan_t * again = xsai_q8_cache_get_or_create(&op, &key);
    expect(again == plan, "cache hit does not create another plan");
    expect(xsai_plan_read_requirements(plan, &requirements) ==
               XSAI_STATUS_SUCCESS,
           "second workspace read");

    std::vector<unsigned char> workspace(
        requirements.workspace_size + requirements.workspace_alignment);
    void * aligned = workspace.data();
    if (requirements.workspace_alignment > 1) {
        aligned = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(workspace.data()) +
             requirements.workspace_alignment - 1) &
            ~(uintptr_t) (requirements.workspace_alignment - 1));
    }
    float out = 0.0f;
    xsai_q8_bind bind = {activation, &out, aligned,
                             requirements.workspace_size};
    expect(xsai_q8_cache_execute(plan, &key, &bind) == XSAI_STATUS_SUCCESS &&
               out == 32.0f,
           "first execute");
    out = 0.0f;
    expect(xsai_q8_cache_execute(plan, &key, &bind) == XSAI_STATUS_SUCCESS &&
               out == 32.0f,
           "second execute");

    xsai_observe_export(&sample);
    std::printf(
        "prepare count=%llu ns=%llu\n"
        "plan_create count=%llu ns=%llu\n"
        "workspace count=%llu ns=%llu\n"
        "execute count=%llu ns=%llu\n"
        "kernel count=%llu ns=%llu\n",
        (unsigned long long) sample.prepare_count,
        (unsigned long long) sample.prepare_ns,
        (unsigned long long) sample.plan_create_count,
        (unsigned long long) sample.plan_create_ns,
        (unsigned long long) sample.workspace_count,
        (unsigned long long) sample.workspace_ns,
        (unsigned long long) sample.execute_count,
        (unsigned long long) sample.execute_ns,
        (unsigned long long) sample.kernel_count,
        (unsigned long long) sample.kernel_ns);
    expect(sample.prepare_count == 1, "prepare counted once");
    expect(sample.plan_create_count == 1, "plan create ignores the cache hit");
    expect(sample.workspace_count == 2, "workspace requirements counted twice");
    expect(sample.execute_count == 2, "execute counted twice");
    expect(sample.kernel_count == sample.execute_count &&
               sample.kernel_ns == sample.execute_ns,
           "kernel sample is execute only");
    expect(sample.kernel_count !=
               sample.prepare_count + sample.plan_create_count +
                   sample.workspace_count + sample.execute_count,
           "one-time prepare and plan create are outside kernel_count");
    expect(sample.prepare_ns > 0 && sample.plan_create_ns > 0 &&
               sample.workspace_ns > 0 && sample.execute_ns > 0,
           "each phase sampled monotonic time");

    const uint64_t kernel_before = sample.kernel_ns;
    xsai_weight_refresh_q8(&weight_tensor, q8_row, 1, 32, 2);
    xsai_observe_export(&sample);
    expect(sample.prepare_count == 2 && sample.kernel_ns == kernel_before &&
               sample.execute_count == 2,
           "another prepare does not move kernel time");

    xsai_weight_release(&weight_tensor);
    xsai_plan_cache_clear();
    if (g_failures != 0) {
        std::printf("xsai-observe-test failures=%d\n", g_failures);
        return 1;
    }
    std::printf("xsai-observe-test pass\n");
    return 0;
}
