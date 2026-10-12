#include "ame-xsai-fusion.h"

#include "ggml-impl.h"

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

struct graph_fixture {
    ggml_tensor lhs;
    ggml_tensor rhs;
    ggml_tensor weight;
    ggml_tensor add;
    ggml_tensor rms;
    ggml_tensor extra;
    ggml_tensor * nodes[3];
    ggml_cgraph graph;
    float lhs_data[4];
    float rhs_data[4];
    float weight_data[4];
    float add_data[4];
    float rms_data[4];
};

static void make_fixture(graph_fixture * fx) {
    std::memset(fx, 0, sizeof(*fx));
    for (int i = 0; i < 4; ++i) {
        fx->lhs_data[i] = (float) (i + 1);
        fx->rhs_data[i] = 1.0f;
        fx->weight_data[i] = 1.0f;
        fx->add_data[i] = 0.0f;
        fx->rms_data[i] = 9.0f;
    }
    fill_f32(&fx->lhs, 4, 1, fx->lhs_data);
    fill_f32(&fx->rhs, 4, 1, fx->rhs_data);
    fill_f32(&fx->weight, 4, 1, fx->weight_data);
    fill_f32(&fx->add, 4, 1, fx->add_data);
    fill_f32(&fx->rms, 4, 1, fx->rms_data);
    fx->add.op = GGML_OP_ADD;
    fx->add.src[0] = &fx->lhs;
    fx->add.src[1] = &fx->rhs;
    fx->rms.op = GGML_OP_RMS_NORM;
    fx->rms.src[0] = &fx->add;
    fx->rms.src[1] = &fx->weight;
    const float eps = 1.0e-5f;
    std::memcpy(fx->rms.op_params, &eps, sizeof(eps));
    fx->nodes[0] = &fx->add;
    fx->nodes[1] = &fx->rms;
    fx->graph.n_nodes = 2;
    fx->graph.nodes = fx->nodes;
}

int main() {
    graph_fixture fx;
    make_fixture(&fx);
    const enum ggml_op add_op = fx.add.op;
    const enum ggml_op rms_op = fx.rms.op;

    DLTensor inputs[3];
    DLTensor output;
    xsai_fused_add_rms_norm_params_t params;
    xsai_op_call_t call;
    std::memset(&call, 0, sizeof(call));
    call.op = (xsai_op_type_t) 99;

    xsai_add_rms_fusion_decision_t decision = ame_xsai_match_add_rms_norm(
        &fx.graph, &fx.rms, 1, inputs, &output, &params, &call);
    expect(decision == XSAI_FUSION_ACCEPT, "all five conditions accept");
    expect(call.op == XSAI_OP_FUSED_ADD_RMS_NORM, "accepted call is the fused op");
    expect(call.input_count == 3, "fused call has lhs rhs and weight");
    expect(params.eps == 1.0e-5f, "eps is copied unchanged");
    expect(fx.add.op == add_op && fx.rms.op == rms_op, "graph ops are not rewritten");

    xsai_op_plan_t * plan = nullptr;
    expect(xsai_op_plan_create(&call, &plan) == XSAI_STATUS_SUCCESS,
           "accepted call has a fused implementation");
    fx.add.op = GGML_OP_NONE;
    float executed[4] = {9, 9, 9, 9};
    output.data = executed;
    expect(xsai_op_plan_execute(plan, &call) == XSAI_STATUS_SUCCESS,
           "execute does not trace the previous add");
    expect(executed[0] != 9.0f, "fused execute wrote the output");
    xsai_op_plan_destroy(plan);
    fx.add.op = add_op;

    make_fixture(&fx);
    fill_f32(&fx.extra, 4, 1, fx.add_data);
    fx.extra.op = GGML_OP_MUL;
    fx.extra.src[0] = &fx.add;
    fx.nodes[2] = &fx.extra;
    fx.graph.n_nodes = 3;
    call.op = (xsai_op_type_t) 99;
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_EXTRA_CONSUMER, "extra consumer does not fuse");
    expect(call.op == (xsai_op_type_t) 99, "rejected match leaves the call untouched");

    make_fixture(&fx);
    fx.add.flags = GGML_TENSOR_FLAG_OUTPUT;
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_EXTRA_CONSUMER, "graph output is an extra consumer");

    make_fixture(&fx);
    fx.lhs.ne[1] = 2;
    fx.rhs.ne[1] = 2;
    fx.add.ne[1] = 2;
    fx.rms.ne[1] = 2;
    fx.weight.ne[0] = 8;
    fx.weight.nb[1] = sizeof(float) * 8;
    fx.lhs.nb[1] = sizeof(float) * 8;
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_DTYPE_SHAPE_LAYOUT, "strided layout does not fuse");

    make_fixture(&fx);
    fx.weight.type = GGML_TYPE_F16;
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_DTYPE_SHAPE_LAYOUT, "dtype mismatch does not fuse");

    make_fixture(&fx);
    fx.rms.data = fx.weight.data;
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_ALIAS, "alias onto the weight does not fuse");

    make_fixture(&fx);
    float bad_eps = -1.0f;
    std::memcpy(fx.rms.op_params, &bad_eps, sizeof(bad_eps));
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_EPS_ROUNDING, "negative eps does not fuse");

    make_fixture(&fx);
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 0, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_NO_IMPLEMENTATION,
           "missing fused implementation does not fuse");

    make_fixture(&fx);
    fx.add.op = GGML_OP_MUL;
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_NOT_PATTERN, "mul then rms_norm is not the pattern");

    make_fixture(&fx);
    fx.rms.data = fx.lhs.data;
    decision = ame_xsai_match_add_rms_norm(&fx.graph, &fx.rms, 1, inputs, &output,
                                           &params, &call);
    expect(decision == XSAI_FUSION_ACCEPT, "exact alias onto an add input is safe");

    if (g_failures != 0) {
        std::printf("ame-xsai-fusion-test failures=%d\n", g_failures);
        return 1;
    }
    std::printf("ame-xsai-fusion-test pass\n");
    return 0;
}
