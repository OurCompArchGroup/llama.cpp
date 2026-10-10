#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ame-backend.h"
#include "ame.h"
#if defined(GGML_USE_XSAI_BLAS)
#include "ame-xsai-plan.h"
#endif

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "traits.h"

#if defined(GGML_USE_XSAI_BLAS)
#include <xsai_blas.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <vector>

// AME_DEBUG/AME_LOG now defined in ame.h
static bool ame_backend_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0 &&
        strcmp(value, "false") != 0 && strcmp(value, "off") != 0 && strcmp(value, "no") != 0;
}

static inline uint64_t ame_backend_read_cycle() {
#if defined(__riscv)
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#else
    return 0;
#endif
}

static int64_t ame_backend_env_i64(const char * name, int64_t fallback) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char * end = nullptr;
    const long long parsed = strtoll(value, &end, 0);
    return end != value ? (int64_t) parsed : fallback;
}

static bool ame_backend_can_use_f16_via_bf16(const ggml_tensor * op) {
    if (!ame_backend_env_enabled("GGML_AME_F16_BF16") || getenv("GGML_AME_SCALAR") != nullptr ||
        op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (ame_backend_env_enabled("GGML_AME_F16_BF16_REQUIRE_AME_BUFFER") &&
        (src0->buffer == nullptr || src0->buffer->buft != ggml_backend_cpu_riscv_ame_buffer_type())) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[2] <= 0 || src0->ne[3] <= 0 ||
        src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    if (src0->nb[0] != sizeof(ggml_fp16_t) || src1->nb[0] != sizeof(float) ||
        op->nb[0] != sizeof(float) || op->nb[1] != (size_t) op->ne[0] * sizeof(float)) {
        return false;
    }
    if (src0->nb[1] < (size_t) src0->ne[0] * sizeof(ggml_fp16_t) ||
        src1->nb[1] < (size_t) src1->ne[0] * sizeof(float)) {
        return false;
    }

    return ggml_ame_can_use_bf16(src0->ne[1], src1->ne[1], src0->ne[0]);
}


static void reference_mul_mat_bf16_f32(
    const void * src0_data,
    ggml_type src1_type,
    const void * src1_data,
    float * dst_data,
    int64_t M, int64_t N, int64_t K,
    size_t src1_stride_bytes
) {
    const ggml_bf16_t * a = (const ggml_bf16_t *) src0_data;
    std::vector<ggml_bf16_t> src1_bf16;

    if (src1_type == GGML_TYPE_F32) {
        src1_bf16.resize((size_t) N * (size_t) K);
        for (int64_t n = 0; n < N; ++n) {
            const float * src1_col = (const float *) ((const char *) src1_data + n * src1_stride_bytes);
            ggml_cpu_fp32_to_bf16(src1_col, src1_bf16.data() + n * K, K);
        }
    }

    for (int64_t n = 0; n < N; ++n) {
        const ggml_bf16_t * b_col = src1_type == GGML_TYPE_BF16
            ? (const ggml_bf16_t *) ((const char *) src1_data + n * src1_stride_bytes)
            : src1_bf16.data() + n * K;
        for (int64_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += GGML_BF16_TO_FP32(a[m * K + k]) * GGML_BF16_TO_FP32(b_col[k]);
            }
            dst_data[n * M + m] = sum;
        }
    }
}

// Check if AME can accelerate this operation
static bool qtype_has_ame_kernels(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_BF16;
}

static size_t ame_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static bool ame_parse_env_bool(const char * name) {
    const char * value = getenv(name);
    if (value == NULL) {
        return false;
    }
    if (value[0] == '\0') {
        return false;
    }
    if (strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "off") == 0 || strcmp(value, "no") == 0) {
        return false;
    }
    return true;
}


struct ame_packed_q8_info {
    void * data = nullptr;
    size_t size = 0;
    void * tile_a = nullptr;
    size_t tile_a_size = 0;
    float * tile_scales = nullptr;
    size_t tile_scales_count = 0;
};

struct ame_buffer_context {
    void * base = nullptr;
    size_t base_size = 0;
    uint64_t generation = 1;
    std::unordered_map<const ggml_tensor *, ame_packed_q8_info> packed_q8;
    std::vector<ggml_tensor *> tensors;
};

static inline ame_buffer_context * ame_buffer_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<ame_buffer_context *>(buffer->context);
}

struct ame_host_tensor_generation {
    ggml_backend_buffer_t buffer = nullptr;
    uint64_t generation = 0;
};

struct ame_host_generation_state {
    uint64_t next_generation = 1;
    std::unordered_map<const ggml_tensor *, ame_host_tensor_generation> tensors;
    std::unordered_map<ggml_backend_buffer_t, uint64_t> buffers;
};

static ame_host_generation_state & ame_host_generations() {
    static ame_host_generation_state state;
    return state;
}

extern "C" void ggml_ame_host_buffer_on_write(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    size_t offset,
    size_t size
) {
    GGML_UNUSED(offset);
    if (buffer == nullptr || size == 0) {
        return;
    }

    ame_host_generation_state & state = ame_host_generations();
    const uint64_t generation = ++state.next_generation;
    state.buffers[buffer] = generation;
    if (tensor != nullptr) {
        state.tensors[tensor] = { buffer, generation };
    }
}

extern "C" void ggml_ame_host_buffer_on_free(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return;
    }

    ame_host_generation_state & state = ame_host_generations();
    state.buffers.erase(buffer);
    for (auto it = state.tensors.begin(); it != state.tensors.end(); ) {
        if (it->second.buffer == buffer) {
            it = state.tensors.erase(it);
        } else {
            ++it;
        }
    }
}

static uint64_t ame_tensor_content_generation(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return 0;
    }

    if (tensor->buffer->buft == ggml_backend_cpu_riscv_ame_buffer_type() &&
        tensor->buffer->context != nullptr) {
        return ame_buffer_ctx(tensor->buffer)->generation;
    }

    ame_host_generation_state & state = ame_host_generations();
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


static void ame_convert_f16_rows_to_bf16(
    const void * src,
    size_t src_row_stride,
    ggml_bf16_t * dst,
    float * f32_scratch,
    int64_t rows,
    int64_t cols
) {
    for (int64_t i = 0; i < rows; ++i) {
        const ggml_fp16_t * src_row = reinterpret_cast<const ggml_fp16_t *>(
            static_cast<const char *>(src) + i * src_row_stride);
        ggml_cpu_fp16_to_fp32(src_row, f32_scratch + i * cols, cols);
    }
    ggml_cpu_fp32_to_bf16(f32_scratch, dst, rows * cols);
}

static void ggml_backend_ame_mul_mat_f16_via_bf16(ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ame_backend_can_use_f16_via_bf16(dst));

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];
    const int64_t batches = src1->ne[2] * src1->ne[3];
    const size_t converted_a_size = (size_t) M * (size_t) K * sizeof(ggml_bf16_t);
    const size_t f32_scratch_offset = ame_align_up(converted_a_size, GGML_MEM_ALIGN);
    const size_t f32_scratch_size = (size_t) M * (size_t) K * sizeof(float);
    const size_t conversion_workspace_size = f32_scratch_offset + f32_scratch_size;
    void * conversion_workspace = ggml_aligned_malloc(conversion_workspace_size);
    GGML_ASSERT(conversion_workspace != nullptr);
    ggml_bf16_t * converted_a = static_cast<ggml_bf16_t *>(conversion_workspace);
    float * f32_scratch = reinterpret_cast<float *>(
        static_cast<char *>(conversion_workspace) + f32_scratch_offset);

    const bool profile = ame_backend_env_enabled("GGML_AME_BF16_PROFILE");
    const bool progress = ame_backend_env_enabled("GGML_AME_BF16_PROGRESS");
    const uint64_t total_start = profile ? ame_backend_read_cycle() : 0;
    uint64_t convert_a_cycles = 0;
    uint64_t kernel_cycles = 0;
    int64_t converted_a_slices = 0;
    int64_t last_i02 = -1;
    int64_t last_i03 = -1;

    if (progress) {
        fprintf(stderr,
            "[AME_F16_BF16_PROGRESS] phase=begin name=%s M=%lld N=%lld K=%lld batches=%lld "
            "src0_ne2=%lld src1_ne2=%lld src0_nb1=%zu src0_nb2=%zu src1_nb1=%zu src1_nb2=%zu\n",
            dst->name,
            (long long) M,
            (long long) N,
            (long long) K,
            (long long) batches,
            (long long) src0->ne[2],
            (long long) src1->ne[2],
            src0->nb[1],
            src0->nb[2],
            src1->nb[1],
            src1->nb[2]);
        fflush(stderr);
    }

    for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
        for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
            const int64_t i02 = i12 / r2;
            const int64_t i03 = i13 / r3;
            const void * src0_batch = static_cast<const char *>(src0->data) + i02 * src0->nb[2] + i03 * src0->nb[3];

            if (i02 != last_i02 || i03 != last_i03) {
                const uint64_t start = profile ? ame_backend_read_cycle() : 0;
                ame_convert_f16_rows_to_bf16(
                    src0_batch, src0->nb[1], converted_a, f32_scratch, M, K);
                if (profile) {
                    convert_a_cycles += ame_backend_read_cycle() - start;
                }
                ++converted_a_slices;
                last_i02 = i02;
                last_i03 = i03;
            }

            const void * src1_batch = static_cast<const char *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            void * dst_batch = static_cast<char *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];
            if (progress) {
                fprintf(stderr,
                    "[AME_F16_BF16_PROGRESS] phase=batch_begin name=%s batch=%lld/%lld src0_slice=%lld,%lld\n",
                    dst->name,
                    (long long) (i13 * src1->ne[2] + i12 + 1),
                    (long long) batches,
                    (long long) i02,
                    (long long) i03);
                fflush(stderr);
            }
            const uint64_t start = profile ? ame_backend_read_cycle() : 0;
            ggml_ame_mul_mat_bf16(
                converted_a,
                src1_batch,
                dst_batch,
                K, M,
                K, N,
                src1->nb[1],
                src1->type,
                params->wdata,
                params->wsize
            );
            if (profile) {
                kernel_cycles += ame_backend_read_cycle() - start;
            }
            if (progress) {
                fprintf(stderr,
                    "[AME_F16_BF16_PROGRESS] phase=batch_end name=%s batch=%lld/%lld\n",
                    dst->name,
                    (long long) (i13 * src1->ne[2] + i12 + 1),
                    (long long) batches);
                fflush(stderr);
            }
        }
    }

    if (profile) {
        fprintf(stderr,
            "[AME_F16_BF16_PROFILE] name=%s M=%lld N=%lld K=%lld batches=%lld a_slices=%lld "
            "whole_k=%d reg_pairs=%lld convert_a_cycles=%llu kernel_cycles=%llu total_cycles=%llu\n",
            dst->name,
            (long long) M,
            (long long) N,
            (long long) K,
            (long long) batches,
            (long long) converted_a_slices,
            ame_backend_env_enabled("GGML_AME_WHOLE_K_BF16") ? 1 : 0,
            (long long) (ame_backend_env_i64("GGML_AME_BF16_REG_PAIRS", 2) >= 2 ? 2 : 1),
            (unsigned long long) convert_a_cycles,
            (unsigned long long) kernel_cycles,
            (unsigned long long) (ame_backend_read_cycle() - total_start));
    }
    if (progress) {
        fprintf(stderr, "[AME_F16_BF16_PROGRESS] phase=end name=%s\n", dst->name);
        fflush(stderr);
    }

    ggml_aligned_free(conversion_workspace, conversion_workspace_size);
}

// Compute forward for AME operations
static void ggml_backend_ame_mul_mat(ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    if (src0->type == GGML_TYPE_F16) {
        ggml_backend_ame_mul_mat_f16_via_bf16(params, dst);
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    GGML_ASSERT(ne00 == ne10);

    if (src0->type == GGML_TYPE_Q8_0) {
        /* Q8 GEMM is executed by ame_xsai_plan_execute. This entry only keeps
         * the non-Q8 kernels. */
        GGML_UNUSED(ne00);
        GGML_UNUSED(ne01);
        GGML_UNUSED(ne10);
        GGML_UNUSED(ne11);
        return;
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_BF16);
        GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16);

        AME_LOG("backend_ame_mul_mat: dispatching to BF16 kernel");
        ggml_ame_mul_mat_bf16(
            src0->data,
            src1->data,
            dst->data,
            ne00, ne01,
            ne10, ne11,
            src1->nb[1],
            src1->type,
            params->wdata,
            params->wsize
        );
    }

    GGML_UNUSED(params);
}

#if defined(GGML_USE_XSAI_BLAS)
static bool ame_q8_plan_op(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_MUL_MAT && op->type == GGML_TYPE_F32 &&
        op->src[0] != nullptr && op->src[1] != nullptr &&
        op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1]->type == GGML_TYPE_F32;
}

static ame_xsai_q8_key ame_q8_plan_key(const ggml_tensor * op) {
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
    key.weight_generation = ame_tensor_content_generation(weight);
    key.explicit_options = 0;
    return key;
}

static xsai_op_plan_t * ame_xsai_plan_get_or_create(const ggml_tensor * op) {
    if (!ame_q8_plan_op(op)) {
        return nullptr;
    }
    const ame_xsai_q8_key key = ame_q8_plan_key(op);
    return ame_xsai_q8_cache_get_or_create(op, &key);
}

static xsai_op_plan_t * ame_xsai_plan_lookup(const ggml_tensor * op) {
    if (!ame_q8_plan_op(op)) {
        return nullptr;
    }
    const ame_xsai_q8_key key = ame_q8_plan_key(op);
    return ame_xsai_q8_cache_lookup(op, &key);
}

static xsai_status_t ame_xsai_plan_execute(xsai_op_plan_t * plan, const ggml_tensor * op,
                                           void * workspace, size_t workspace_size) {
    const ame_xsai_q8_key key = ame_q8_plan_key(op);
    const ame_xsai_q8_bind bind = {
        op->src[1]->data,
        op->src[0]->data,
        op->data,
        workspace,
        workspace_size,
    };
    return ame_xsai_q8_cache_execute(plan, &key, &bind);
}
#endif

// AME tensor_traits implementation
namespace ggml::cpu::riscv_ame {

class tensor_traits : public ggml::cpu::tensor_traits {
public:
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
#if defined(GGML_USE_XSAI_BLAS)
        if (op->src[0]->type == GGML_TYPE_Q8_0) {
            xsai_op_plan_t * plan = ame_xsai_plan_get_or_create(op);
            if (plan == nullptr) {
                return false;
            }
            xsai_op_requirements_t requirements = {};
            requirements.struct_size = sizeof(requirements);
            requirements.abi_version = XSAI_BLAS_ABI_VERSION;
            if (xsai_op_plan_get_requirements(plan, &requirements) != XSAI_STATUS_SUCCESS) {
                return false;
            }
            size = requirements.workspace_size;
            return true;
        }
#else
        if (op->src[0]->type == GGML_TYPE_Q8_0) {
            return false;
        }
#endif

        if (op->src[0]->type == GGML_TYPE_BF16 &&
            (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_BF16) &&
            ggml_ame_can_use_bf16(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0])) {
            size = 0;
            return true;
        }

        if (ame_backend_can_use_f16_via_bf16(op)) {
            size = 0;
            return true;
        }

        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        static bool scalar_mode = (getenv("GGML_AME_SCALAR") != nullptr);

#if defined(GGML_USE_XSAI_BLAS)
        if (op->src[0] != nullptr && op->src[0]->type == GGML_TYPE_Q8_0) {
            xsai_op_plan_t * plan = ame_xsai_plan_lookup(op);
            if (plan == nullptr) {
                return false;
            }
            if (params->ith != 0) {
                return true;
            }
            const xsai_status_t status = ame_xsai_plan_execute(
                plan, op, params->wdata, params->wsize);
            /* UNSUPPORTED is returned before any output write. Let ggml CPU
             * run that op. Any other execute failure may already have written
             * dst, so it must not fall through. */
            if (status == XSAI_STATUS_UNSUPPORTED) {
                return false;
            }
            return true;
        }
#endif

        if (scalar_mode) {
            if (params->ith != 0) {
                return true;
            }
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];
            const int64_t M = src0->ne[1];
            const int64_t N = src1->ne[1];
            const int64_t K = src0->ne[0];

            if (src0->type == GGML_TYPE_BF16 &&
                (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16)) {
                reference_mul_mat_bf16_f32(
                    src0->data,
                    src1->type,
                    src1->data,
                    (float *) op->data,
                    M, N, K,
                    src1->nb[1]
                );
                return true;
            }
            return false;
        }

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const bool f16_via_bf16 = ame_backend_can_use_f16_via_bf16(op);
        if (src0->type == GGML_TYPE_F16) {
            if (!f16_via_bf16) {
                return false;
            }
        } else if (src0->type == GGML_TYPE_Q8_0) {
            return false;
        } else if (src0->type == GGML_TYPE_BF16) {
            if ((src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_BF16) ||
                !ggml_ame_can_use_bf16(src0->ne[1], src1->ne[1], src0->ne[0])) {
                return false;
            }
        } else {
            return false;
        }

        if (params->ith != 0) {
            return true;
        }

        AME_LOG("tensor_traits::compute_forward: calling AME mul_mat");
        ggml_backend_ame_mul_mat(params, op);
        AME_LOG("tensor_traits::compute_forward: AME mul_mat completed");
        return true;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}

}  // namespace ggml::cpu::riscv_ame

// AME buffer interface
static void ggml_backend_ame_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ame_buffer_context * ctx = ame_buffer_ctx(buffer);
    for (auto & it : ctx->packed_q8) {
        if (it.second.data != nullptr) {
            ggml_aligned_free(it.second.data, it.second.size);
        }
        if (it.second.tile_a != nullptr) {
            ggml_aligned_free(it.second.tile_a, it.second.tile_a_size);
        }
        if (it.second.tile_scales != nullptr) {
            ggml_aligned_free(it.second.tile_scales, it.second.tile_scales_count * sizeof(float));
        }
    }
    if (ctx->base != nullptr) {
        ggml_aligned_free(ctx->base, ctx->base_size);
    }
    delete ctx;
}

static void * ggml_backend_ame_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ame_buffer_ctx(buffer)->base;
}

static enum ggml_status ggml_backend_ame_buffer_init_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor
) {
    ame_buffer_ctx(buffer)->tensors.push_back(tensor);
    tensor->extra = (void *)ggml::cpu::riscv_ame::get_tensor_traits(buffer, tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_ame_buffer_memset_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    uint8_t value,
    size_t offset,
    size_t size
) {
    memset((char *)tensor->data + offset, value, size);
    if (size != 0) {
        ame_buffer_ctx(buffer)->generation++;
    }
}

static void ggml_backend_ame_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size
) {
    memcpy((char *) tensor->data + offset, data, size);
    if (size != 0) {
        ame_buffer_ctx(buffer)->generation++;
    }
    GGML_UNUSED(data);
}

static void ggml_backend_ame_buffer_get_tensor(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size
) {
    memcpy((char *)data, (const char *)tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_ame_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ame_buffer_context * ctx = ame_buffer_ctx(buffer);
    memset(ctx->base, value, buffer->size);
    if (buffer->size != 0) {
        ctx->generation++;
    }

    GGML_UNUSED(value);
}

static ggml_backend_buffer_i ggml_backend_ame_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_ame_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ame_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_ame_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_ame_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_ame_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_ame_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_ame_buffer_clear,
    /* .reset           = */ nullptr,
};

// Buffer type interface
static const char * ggml_backend_ame_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "RISCV_AME";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_ame_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft,
    size_t size
) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    ame_buffer_context * ctx = new ame_buffer_context();
    ctx->base = data;
    ctx->base_size = size;
    return ggml_backend_buffer_init(buft, ggml_backend_ame_buffer_interface, ctx, size);
}

static size_t ggml_backend_ame_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;  // 64-byte alignment for RISC-V cache lines
    GGML_UNUSED(buft);
}

static size_t ggml_backend_ame_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft,
    const ggml_tensor * tensor
) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

// Extra buffer type for operation support checking
namespace ggml::cpu::riscv_ame {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
public:
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // AME buffers keep standard ggml tensor bytes, so the generic CPU backend
        // can still execute unsupported ops directly from the same storage.  This
        // method therefore only decides whether the AME-specialized path may run;
        // it must not block CPU fallback at scheduling time.

        if (op->op != GGML_OP_MUL_MAT) {
            return true;
        }

        auto is_contiguous_2d = [](const struct ggml_tensor * t) {
            return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
        };

        AME_LOG("supports_op: checking op=%d type0=%d type1=%d", op->op, 
                op->src[0] ? op->src[0]->type : -1, 
                op->src[1] ? op->src[1]->type : -1);

        if (!is_contiguous_2d(op->src[0])) {
            AME_LOG("supports_op: fallback (src0 not contiguous 2d)");
            return true;
        }

        if (!is_contiguous_2d(op->src[1])) {
            AME_LOG("supports_op: fallback (src1 not contiguous 2d)");
            return true;
        }

        if (!op->src[0]->buffer) {
            AME_LOG("supports_op: fallback (src0 has no buffer)");
            return true;
        }

        if (op->src[0]->buffer->buft != ggml_backend_cpu_riscv_ame_buffer_type()) {
            AME_LOG("supports_op: fallback (src0 not in AME buffer)");
            return true;
        }

        if (!qtype_has_ame_kernels(op->src[0]->type)) {
            AME_LOG("supports_op: fallback (src0 type not supported)");
            return true;
        }

        /* Q8 eligibility is decided by xsai-blas plan_create. Do not
         * pre-guess tile shape or AME capability here. */
        const bool shape_supported = op->src[0]->type == GGML_TYPE_Q8_0 ||
            ggml_ame_can_use_bf16(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0]);
        if (!shape_supported) {
            AME_LOG("supports_op: fallback (shape not AME-friendly)");
            return true;
        }
            
        // src1 must be host buffer
        if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
            AME_LOG("supports_op: fallback (src1 not host)");
            return true;
        }
        const bool src1_supported =
            (op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1]->type == GGML_TYPE_F32) ||
            (op->src[0]->type == GGML_TYPE_BF16 &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_BF16));
        if (src1_supported) {
            AME_LOG("supports_op: accept M=%lld N=%lld K=%lld", (long long) op->src[0]->ne[1], (long long) op->src[1]->ne[1], (long long) op->src[0]->ne[0]);
            return true;
        }

        AME_LOG("supports_op: fallback (src1 type not supported)");
        return true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT &&
            op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_cpu_riscv_ame_buffer_type()) {
            return (ggml::cpu::tensor_traits *)op->src[0]->extra;
        }
        if (ame_backend_can_use_f16_via_bf16(op)) {
            return ggml::cpu::riscv_ame::get_tensor_traits(nullptr, nullptr);
        }
        return nullptr;
    }
};

}  // namespace ggml::cpu::riscv_ame

// Public buffer type getter. Q8 capability is owned by xsai-blas; this
// registration no longer probes CSR tile limits or picks an i8 kernel.
ggml_backend_buffer_type_t ggml_backend_cpu_riscv_ame_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv_ame = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_ame_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_ame_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_ame_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ ggml_backend_ame_buffer_type_get_alloc_size,
            /* .is_host          = */ nullptr,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::riscv_ame::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_riscv_ame;
}
