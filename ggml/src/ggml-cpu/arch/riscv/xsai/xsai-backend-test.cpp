#include "xsai-backend.h"

#include "xsai-observe.h"
#include "xsai-plan.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu-impl.h"

#include <cstdio>
#include <cstdlib>
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

static void fill_f32(ggml_tensor * tensor, int64_t n0, int64_t n1, void * data) {
    std::memset(tensor, 0, sizeof(*tensor));
    tensor->type = GGML_TYPE_F32;
    tensor->ne[0] = n0;
    tensor->ne[1] = n1;
    tensor->ne[2] = 1;
    tensor->ne[3] = 1;
    tensor->nb[0] = sizeof(float);
    tensor->nb[1] = sizeof(float) * (size_t) n0;
    tensor->nb[2] = tensor->nb[1] * (size_t) n1;
    tensor->nb[3] = tensor->nb[2];
    tensor->data = data;
}

static void test_add_reuses_plan() {
    ggml_tensor lhs;
    ggml_tensor rhs;
    ggml_tensor dst;
    float lhs_data[4] = {1.f, 2.f, 3.f, 4.f};
    float rhs_data[4] = {10.f, 20.f, 30.f, 40.f};
    float dst_data[4] = {9.f, 9.f, 9.f, 9.f};
    fill_f32(&lhs, 4, 1, lhs_data);
    fill_f32(&rhs, 4, 1, rhs_data);
    fill_f32(&dst, 4, 1, dst_data);
    dst.op = GGML_OP_ADD;
    dst.src[0] = &lhs;
    dst.src[1] = &rhs;

    xsai_plan_cache_clear();
    xsai_plan_cache_reset_counts();
    xsai_observe_reset();

    size_t bytes = 99;
    expect(xsai_backend_work_size(1, &dst, bytes), "add work_size creates a plan");
    expect(bytes == 0, "add workspace is empty");
    const uint64_t created = xsai_plan_create_count();
    const uint64_t misses = xsai_plan_key_miss_count();
    expect(created == 1 && misses == 1 && created == misses, "first add plan-create equals key miss");

    size_t again = 7;
    expect(xsai_backend_work_size(1, &dst, again), "second add work_size hits");
    expect(xsai_plan_create_count() == 1, "second work_size does not create");
    expect(xsai_plan_cache_hit_count() == 1, "second work_size is a cache hit");
    expect(xsai_plan_key_miss_count() == 1, "hit is not a key miss");

    ggml_compute_params params = {};
    params.ith = 0;
    params.nth = 1;
    expect(xsai_backend_compute(&params, &dst), "add compute executes the plan");
    expect(dst_data[0] == 11.f && dst_data[1] == 22.f && dst_data[2] == 33.f && dst_data[3] == 44.f,
           "add compute wrote the sum");
    expect(xsai_backend_compute(&params, &dst), "second add compute reuses the plan");
    expect(xsai_plan_execute_count() == 2, "two executes share one plan");
    expect(xsai_plan_create_count() == xsai_plan_key_miss_count(),
           "plan-create still equals key miss after reuse");

    xsai_observe_export_t observed = {};
    xsai_observe_export(&observed);
    expect(observed.plan_create_count == xsai_plan_key_miss_count(),
           "observe plan_create matches key miss");
    expect(observed.execute_count == 2 && observed.kernel_count == observed.execute_count,
           "observe execute is the kernel sample");
}

static void test_fallback_does_not_write() {
    ggml_tensor src;
    ggml_tensor dst;
    float src_data[2] = {1.f, 2.f};
    float dst_data[2] = {7.f, 8.f};
    fill_f32(&src, 2, 1, src_data);
    fill_f32(&dst, 2, 1, dst_data);
    dst.op = GGML_OP_ROPE;
    dst.src[0] = &src;

    size_t bytes = 123;
    expect(!xsai_backend_work_size(1, &dst, bytes), "unregistered rope work_size declines");
    expect(bytes == 123, "declined work_size leaves the caller size alone");

    ggml_compute_params params = {};
    params.ith = 0;
    params.nth = 1;
    expect(!xsai_backend_compute(&params, &dst), "unregistered rope compute falls back");
    expect(dst_data[0] == 7.f && dst_data[1] == 8.f, "fallback does not write the output");

    ggml_tensor rhs;
    float rhs_data[1] = {1.f};
    fill_f32(&rhs, 1, 1, rhs_data);
    dst.op = GGML_OP_ADD;
    dst.src[1] = &rhs;
    dst_data[0] = 7.f;
    dst_data[1] = 8.f;
    expect(!xsai_backend_compute(&params, &dst), "broadcast add falls back before write");
    expect(dst_data[0] == 7.f && dst_data[1] == 8.f, "broadcast add leaves the output unchanged");
}

static void test_q8_work_and_compute() {
    ggml_tensor weight;
    ggml_tensor activation;
    ggml_tensor dst;
    std::memset(&weight, 0, sizeof(weight));
    std::memset(&activation, 0, sizeof(activation));
    std::memset(&dst, 0, sizeof(dst));

    unsigned char q8[34];
    std::memset(q8, 0, sizeof(q8));
    q8[0] = 0x00;
    q8[1] = 0x3c; /* fp16 1.0, little endian */
    for (int i = 0; i < 32; ++i) {
        q8[2 + i] = 1;
    }
    float act[32];
    for (int i = 0; i < 32; ++i) {
        act[i] = 1.f;
    }
    float out = 42.f;

    weight.type = GGML_TYPE_Q8_0;
    weight.ne[0] = 32;
    weight.ne[1] = 1;
    weight.ne[2] = 1;
    weight.ne[3] = 1;
    weight.nb[0] = 34;
    weight.nb[1] = 34;
    weight.nb[2] = 34;
    weight.nb[3] = 34;
    weight.data = q8;

    activation.type = GGML_TYPE_F32;
    activation.ne[0] = 32;
    activation.ne[1] = 1;
    activation.ne[2] = 1;
    activation.ne[3] = 1;
    activation.nb[0] = sizeof(float);
    activation.nb[1] = 32 * sizeof(float);
    activation.nb[2] = activation.nb[1];
    activation.nb[3] = activation.nb[1];
    activation.data = act;

    dst.type = GGML_TYPE_F32;
    dst.op = GGML_OP_MUL_MAT;
    dst.ne[0] = 1;
    dst.ne[1] = 1;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = sizeof(float);
    dst.nb[1] = sizeof(float);
    dst.nb[2] = sizeof(float);
    dst.nb[3] = sizeof(float);
    dst.data = &out;
    dst.src[0] = &weight;
    dst.src[1] = &activation;

    const uint64_t before = xsai_plan_create_count();
    size_t bytes = 0;
    const bool planned = xsai_backend_work_size(1, &dst, bytes);
    expect(planned, "q8 work_size creates a plan");
    if (!planned) {
        return;
    }
    expect(xsai_plan_create_count() == before + 1, "q8 plan-create counted");
    expect(xsai_plan_cached_op(&dst) == XSAI_OP_GEMM,
           "M = 1 activation uses the vector form");
    ggml_compute_params params = {};
    params.ith = 0;
    params.nth = 1;
    params.wdata = nullptr;
    params.wsize = bytes;
    void * scratch = nullptr;
    if (bytes) {
        if (posix_memalign(&scratch, 64, bytes) != 0) {
            scratch = nullptr;
        }
        params.wdata = scratch;
    }
    expect(xsai_backend_compute(&params, &dst), "q8 compute executes");
    expect(out > 31.f && out < 33.f, "q8 compute is near the reference dot");
    std::printf("q8 vector out=%g workspace=%zu prepared_bytes=%llu\n", out, bytes,
                (unsigned long long) xsai_prepared_bytes());
    expect(xsai_prepared_bytes() > 0, "q8 prepare owns arena bytes");
    std::free(scratch);

    /* M = 2 reuses the same prepared weight through a GEMM plan. */
    float act2[2 * 32];
    float out2[2] = {0.f, 0.f};
    for (int i = 0; i < 2 * 32; ++i) {
        act2[i] = 1.f;
    }
    activation.ne[1] = 2;
    activation.nb[1] = 32 * sizeof(float);
    activation.nb[2] = activation.nb[1] * 2;
    activation.nb[3] = activation.nb[2];
    activation.data = act2;
    dst.ne[1] = 2;
    dst.nb[1] = sizeof(float);
    dst.nb[2] = sizeof(float) * 2;
    dst.nb[3] = dst.nb[2];
    dst.data = out2;

    size_t gemm_bytes = 0;
    const uint64_t gemm_before = xsai_plan_create_count();
    expect(xsai_backend_work_size(1, &dst, gemm_bytes), "q8 gemm work_size creates a plan");
    expect(xsai_plan_create_count() == gemm_before + 1, "gemm plan-create counted");
    expect(xsai_plan_cached_op(&dst) == XSAI_OP_GEMM,
           "M > 1 activation routes to GEMM");
    ggml_compute_params gemm_params = {};
    gemm_params.ith = 0;
    gemm_params.nth = 1;
    void * gemm_scratch = nullptr;
    if (gemm_bytes) {
        if (posix_memalign(&gemm_scratch, 64, gemm_bytes) != 0) {
            gemm_scratch = nullptr;
        }
        gemm_params.wdata = gemm_scratch;
    }
    gemm_params.wsize = gemm_bytes;
    expect(xsai_backend_compute(&gemm_params, &dst), "q8 gemm compute executes");
    expect(out2[0] > 31.f && out2[0] < 33.f && out2[1] > 31.f && out2[1] < 33.f,
           "q8 gemm writes every row");
    std::free(gemm_scratch);
}

static void test_placement_probe() {
    ggml_backend_buffer_type dummy_buft = {};
    ggml_backend_buffer dummy = {};
    dummy.buft = &dummy_buft;
    dummy.context = nullptr;
    dummy.size = 0;

    unsigned char q8[34] = {};
    q8[0] = 0x00;
    q8[1] = 0x3c; /* fp16 1.0 */
    ggml_tensor weight;
    std::memset(&weight, 0, sizeof(weight));
    weight.type = GGML_TYPE_Q8_0;
    weight.ne[0] = 32;
    weight.ne[1] = 1;
    weight.ne[2] = 1;
    weight.ne[3] = 1;
    weight.nb[0] = 34;
    weight.nb[1] = 34;
    weight.nb[2] = 34;
    weight.nb[3] = 34;
    weight.data = q8;
    weight.buffer = &dummy;

    float act[32] = {};
    float out = 0.f;
    ggml_tensor activation;
    ggml_tensor dst;
    fill_f32(&activation, 32, 1, act);
    fill_f32(&dst, 1, 1, &out);
    dst.op = GGML_OP_MUL_MAT;
    dst.src[0] = &weight;
    dst.src[1] = &activation;

    expect(xsai_backend_placement_supports(&dummy_buft, &dst), "q8 mul_mat placement claims");

    weight.type = GGML_TYPE_F16;
    expect(!xsai_backend_placement_supports(&dummy_buft, &dst),
           "non-q8 mul_mat placement declines");
    weight.type = GGML_TYPE_Q8_0;

    dst.op = GGML_OP_GET_ROWS;
    expect(xsai_backend_placement_supports(&dummy_buft, &dst),
           "non mul_mat placement stays permissive");
    dst.op = GGML_OP_MUL_MAT;

    dummy.size = 34;
    weight.type = GGML_TYPE_F16;
    expect(xsai_backend_placement_supports(&dummy_buft, &dst),
           "real buffer placement stays permissive");
    dummy.size = 0;
    weight.type = GGML_TYPE_Q8_0;

    dummy.buft = nullptr;
    expect(xsai_backend_placement_supports(&dummy_buft, &dst),
           "foreign buffer is not a placement probe");
    dummy.buft = &dummy_buft;
}

int main() {
    test_add_reuses_plan();
    test_fallback_does_not_write();
    test_q8_work_and_compute();
    test_placement_probe();
    xsai_plan_cache_clear();
    if (g_failures) {
        std::printf("xsai-backend-test failures=%d\n", g_failures);
        return 1;
    }
    std::printf("xsai-backend-test pass\n");
    return 0;
}
