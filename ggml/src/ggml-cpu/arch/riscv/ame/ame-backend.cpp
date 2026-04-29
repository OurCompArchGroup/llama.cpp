#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ame-backend.h"
#include "ame.h"

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "traits.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

// AME_DEBUG/AME_LOG now defined in ame.h
// Simple scalar reference implementation for Q8_0 x F32 matmul
// Now using tiled approach with ggml_ame_gemm_tile_i8_i32_bT SCALAR version
static void reference_mul_mat_q8_0_f32(
    const void * src0_data,
    const void * src1_data,
    float * dst_data,
    int64_t M, int64_t N, int64_t K,
    size_t src1_stride_bytes
) {
    const block_q8_0 * x = (const block_q8_0 *)src0_data;
    const int64_t nb = K / QK8_0;
    
    // Quantize src1 to Q8_0
    block_q8_0 * y = (block_q8_0 *)malloc(N * nb * sizeof(block_q8_0));
    
    for (int64_t n = 0; n < N; n++) {
        const float * src1_col = (const float *)((const char *)src1_data + n * src1_stride_bytes);
        ggml_ame_quantize_row_f32_to_q8_0(src1_col, y + n * nb, K);
    }
    
    // Heap-allocated via ggml_aligned_malloc so the buffers come from the xsai
    // pool and are physically contiguous (required by CUTE AMU PA-offset addressing).
    int8_t  * tile_A = (int8_t  *)ggml_aligned_malloc(AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    int8_t  * tile_B = (int8_t  *)ggml_aligned_malloc(AME_TILE_N * AME_TILE_K * sizeof(int8_t));
    int32_t * tile_C = (int32_t *)ggml_aligned_malloc(AME_TILE_M * AME_TILE_N * sizeof(int32_t));
    if (!tile_A || !tile_B || !tile_C) {
        ggml_aligned_free(tile_A, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
        ggml_aligned_free(tile_B, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
        ggml_aligned_free(tile_C, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
        free(y);
        return;
    }

    memset(dst_data, 0, M * N * sizeof(float));

    for (int64_t m0 = 0; m0 < M; m0 += AME_TILE_M) {
        for (int64_t n0 = 0; n0 < N; n0 += AME_TILE_N) {
            
            // Loop over blocks (K dimension)
            for (int64_t b = 0; b < nb; b++) {
                // Pack A tile: [AME_TILE_M, AME_TILE_K]
                for (int i = 0; i < AME_TILE_M; i++) {
                    if (m0 + i < M) {
                         const block_q8_0 * blk = &x[b + (m0 + i) * nb];
                         memset(&tile_A[i * AME_TILE_K], 0, AME_TILE_K);
                         memcpy(&tile_A[i * AME_TILE_K], blk->qs, QK8_0);
                    } else {
                         memset(&tile_A[i * AME_TILE_K], 0, AME_TILE_K);
                    }
                }

                // Pack B tile: [AME_TILE_N, AME_TILE_K]
                // Note: y is column-major logic (N x nb), but stored linear.
                // y[b + n*nb] is block for col 'n'.
                // We want tile_B to have rows corresponding to 'n' (transposed B).
                for (int j = 0; j < AME_TILE_N; j++) {
                    if (n0 + j < N) {
                         const block_q8_0 * blk = &y[b + (n0 + j) * nb];
                         memset(&tile_B[j * AME_TILE_K], 0, AME_TILE_K);
                         memcpy(&tile_B[j * AME_TILE_K], blk->qs, QK8_0);
                    } else {
                         memset(&tile_B[j * AME_TILE_K], 0, AME_TILE_K);
                    }
                }

                // Compute Tile
                memset(tile_C, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                ggml_ame_gemm_tile_i8_i32_bT(tile_A, tile_B, tile_C);

                // Accumulate to destination with scales
                for (int i = 0; i < AME_TILE_M; i++) {
                    if (m0 + i >= M) continue;
                    const float d_a = GGML_FP16_TO_FP32(x[b + (m0 + i) * nb].d);
                    
                    for (int j = 0; j < AME_TILE_N; j++) {
                        if (n0 + j >= N) continue;
                        const float d_b = GGML_FP16_TO_FP32(y[b + (n0 + j) * nb].d);
                        
                        float val = (float)tile_C[i * AME_TILE_N + j];
                        dst_data[(m0 + i) + (n0 + j) * M] += val * d_a * d_b;
                    }
                }
            }
        }
    }
    
    ggml_aligned_free(tile_A, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    ggml_aligned_free(tile_B, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
    ggml_aligned_free(tile_C, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
    free(y);
}

static float reference_dot_bf16(const ggml_bf16_t * a, const ggml_bf16_t * b, int64_t K) {
    float sum = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        sum += GGML_BF16_TO_FP32(a[k]) * GGML_BF16_TO_FP32(b[k]);
    }
    return sum;
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
            dst_data[n * M + m] = reference_dot_bf16(a + m * K, b_col, K);
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

static bool ame_diff_enabled() {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_parse_env_bool("GGML_AME_DIFF") ? 1 : 0;
    }
    return cached == 1;
}

static void ame_report_diff(
    const char * tag,
    const float * got,
    const float * ref,
    int64_t M,
    int64_t N,
    int64_t K,
    float threshold
) {
    float max_abs = 0.0f;
    int64_t max_i = 0;
    int64_t max_j = 0;
    float got_v = 0.0f;
    float ref_v = 0.0f;

    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            const float cur_got = got[j * M + i];
            const float cur_ref = ref[j * M + i];
            const float cur_abs = fabsf(cur_got - cur_ref);
            if (cur_abs > max_abs) {
                max_abs = cur_abs;
                max_i = i;
                max_j = j;
                got_v = cur_got;
                ref_v = cur_ref;
            }
        }
    }

    fprintf(stderr,
            "[AME-DIFF] %s M=%lld N=%lld K=%lld max_abs=%.6f at (%lld,%lld) got=%.6f ref=%.6f%s\n",
            tag,
            (long long) M,
            (long long) N,
            (long long) K,
            max_abs,
            (long long) max_i,
            (long long) max_j,
            got_v,
            ref_v,
            max_abs > threshold ? " <!>" : "");
}

static bool ame_use_packed_q8() {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_parse_env_bool("GGML_AME_PACKED_Q8") ? 1 : 0;
    }
    return cached == 1;
}

struct ame_packed_q8_info {
    void * data = nullptr;
    size_t size = 0;
};

struct ame_buffer_context {
    void * base = nullptr;
    size_t base_size = 0;
    std::unordered_map<const ggml_tensor *, ame_packed_q8_info> packed_q8;
    std::vector<ggml_tensor *> tensors;
};

static inline ame_buffer_context * ame_buffer_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<ame_buffer_context *>(buffer->context);
}

static size_t ggml_ame_packed_q8_64_size(const ggml_tensor * tensor) {
    const int64_t nb64 = (tensor->ne[0] + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    return (size_t) ggml_nrows(tensor) * (size_t) nb64 * sizeof(block_q8_ame64);
}

static const block_q8_ame64 * ame_get_packed_q8_weight(const ggml_tensor * tensor) {
    if (!ame_use_packed_q8()) {
        return nullptr;
    }
    if (tensor->buffer == nullptr || tensor->buffer->context == nullptr) {
        return nullptr;
    }

    ame_buffer_context * ctx = ame_buffer_ctx(tensor->buffer);
    const auto it = ctx->packed_q8.find(tensor);
    if (it == ctx->packed_q8.end()) {
        return nullptr;
    }
    return static_cast<const block_q8_ame64 *>(it->second.data);
}

static void ame_store_packed_q8_weight(
    ggml_backend_buffer_t buffer,
    const ggml_tensor * tensor,
    const void * data
) {
    ame_buffer_context * ctx = ame_buffer_ctx(buffer);
    auto & entry = ctx->packed_q8[tensor];
    const size_t packed_size = ggml_ame_packed_q8_64_size(tensor);

    if (entry.data == nullptr || entry.size != packed_size) {
        if (entry.data != nullptr) {
            ggml_aligned_free(entry.data, entry.size);
        }
        entry.data = ggml_aligned_malloc(packed_size);
        entry.size = packed_size;
    }

    if (entry.data != nullptr) {
        ggml_ame_repack_q8_0_to_ame64(entry.data, data, ggml_nrows(tensor), tensor->ne[0]);
    }
}

static size_t ame_refresh_packed_q8_weights(ggml_backend_buffer_t buffer) {
    if (!ame_use_packed_q8()) {
        return 0;
    }

    ame_buffer_context * ctx = ame_buffer_ctx(buffer);
    size_t repacked = 0;

    for (ggml_tensor * tensor : ctx->tensors) {
        if (tensor == nullptr || tensor->data == nullptr) {
            continue;
        }
        if (tensor->type != GGML_TYPE_Q8_0) {
            continue;
        }

        ame_store_packed_q8_weight(buffer, tensor, tensor->data);
        ++repacked;
    }

    return repacked;
}

static size_t ggml_backend_ame_desired_wsize(const ggml_tensor * op) {
    if (op->src[0]->type == GGML_TYPE_BF16) {
        return 0;
    }

    const int64_t K = op->src[0]->ne[0];
    const int64_t N = op->src[1]->ne[1];
    const int64_t nb_x = ame_use_packed_q8() ? ((K + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K) : (K / QK8_0);
    const size_t packed_b_panel_size = (size_t) nb_x * AME_TILE_N * AME_TILE_K * sizeof(int8_t);

    size_t size = 64;
    size = ame_align_up(size, 64);
    size += (size_t) N * (size_t) nb_x * sizeof(ggml_fp16_t);
    size = ame_align_up(size, 64);
    size += AME_TILE_M * AME_TILE_K * sizeof(int8_t);
    size = ame_align_up(size, 64);
    size += packed_b_panel_size;
    size = ame_align_up(size, 64);
    size += AME_TILE_M * AME_TILE_N * sizeof(int32_t);
    return size;
}

// Compute forward for AME operations
static void ggml_backend_ame_mul_mat(ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    GGML_ASSERT(ne00 == ne10);

    if (src0->type == GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        const block_q8_ame64 * packed_w = ame_get_packed_q8_weight(src0);
        if (packed_w != nullptr) {
            AME_LOG("backend_ame_mul_mat: dispatching to packed Q8_64 kernel");
            ggml_ame_mul_mat_q8_0_ame64(
                packed_w,
                src1,
                src1->data,
                dst->data,
                ne00, ne01,
                ne10, ne11,
                src1->nb[1],
                ggml_threadpool_graph_id(params->threadpool),
                params->wdata,
                params->wsize
            );
        } else {
            AME_LOG("backend_ame_mul_mat: dispatching to baseline Q8_0 kernel");
            ggml_ame_mul_mat_q8_0(
                src0->data,
                src1->data,
                dst->data,
                ne00, ne01,
                ne10, ne11,
                src1->nb[1],
                params->wdata,
                params->wsize
            );
        }
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

    if (ame_diff_enabled()) {
        std::vector<float> ref((size_t) ne01 * (size_t) ne11, 0.0f);
        const float threshold = 0.01f;

        if (src0->type == GGML_TYPE_BF16) {
            reference_mul_mat_bf16_f32(
                src0->data,
                src1->type,
                src1->data,
                ref.data(),
                ne01, ne11, ne00,
                src1->nb[1]
            );
            ame_report_diff("bf16", (const float *) dst->data, ref.data(), ne01, ne11, ne00, threshold);
        }
    }

    GGML_UNUSED(params);
}

// AME tensor_traits implementation
namespace ggml::cpu::riscv_ame {

class tensor_traits : public ggml::cpu::tensor_traits {
public:
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        if (op->src[0]->type == GGML_TYPE_Q8_0) {
            if (op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            if (!ggml_ame_can_use_q8(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0])) {
                return false;
            }
            size = ggml_backend_ame_desired_wsize(op);
            return true;
        }

        if (op->src[0]->type == GGML_TYPE_BF16 &&
            (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_BF16) &&
            ggml_ame_can_use_bf16(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0])) {
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

        if (scalar_mode) {
            if (params->ith != 0) {
                return true;
            }

            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];
            const int64_t M = src0->ne[1];
            const int64_t N = src1->ne[1];
            const int64_t K = src0->ne[0];

            if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
                reference_mul_mat_q8_0_f32(
                    src0->data,
                    src1->data,
                    (float *) op->data,
                    M, N, K,
                    src1->nb[1]
                );
                return true;
            }
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
        if (src0->type == GGML_TYPE_Q8_0) {
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (!ggml_ame_can_use_q8(src0->ne[1], src1->ne[1], src0->ne[0])) {
                return false;
            }
        } else if (src0->type == GGML_TYPE_BF16) {
            if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_BF16) {
                return false;
            }
            if (!ggml_ame_can_use_bf16(src0->ne[1], src1->ne[1], src0->ne[0])) {
                return false;
            }
        } else {
            return false;
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
    GGML_UNUSED(buffer);
}

static void ggml_backend_ame_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size
) {
    memcpy((char *) tensor->data + offset, data, size);
    if (ame_use_packed_q8() && tensor->type == GGML_TYPE_Q8_0 && offset == 0 && size == ggml_nbytes(tensor)) {
        ame_store_packed_q8_weight(buffer, tensor, data);
    }
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

    if (value == 0) {
        const size_t repacked = ame_refresh_packed_q8_weights(buffer);
        if (repacked > 0) {
            AME_LOG("buffer_clear: synthesized %zu packed Q8_64 tensors after zero-fill", repacked);
        }
    }
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

        const bool shape_supported =
            op->src[0]->type == GGML_TYPE_Q8_0
                ? ggml_ame_can_use_q8(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0])
                : ggml_ame_can_use_bf16(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0]);
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
            (op->src[0]->type == GGML_TYPE_BF16 && (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_BF16));
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
        return nullptr;
    }
};

}  // namespace ggml::cpu::riscv_ame

// Runtime AME availability check
static bool ggml_ame_available() {
#ifdef GGML_USE_RV_AME
    // TODO: Add runtime detection by trying to execute an AME instruction
    AME_LOG("ggml_ame_available: returning true (build has GGML_USE_RV_AME)");
    return true;
#else
    AME_LOG("ggml_ame_available: returning false (GGML_USE_RV_AME not set)");
    return false;
#endif
}

// Public buffer type getter
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

    if (!ggml_ame_available()) {
        AME_LOG("buffer_type: AME not available, returning nullptr");
        return nullptr;
    }

    AME_LOG("buffer_type: returning AME buffer type");

    return &ggml_backend_cpu_buffer_type_riscv_ame;
}
