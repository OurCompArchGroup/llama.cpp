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
    ggml_ame_sync_begin_op();

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

static bool ame_use_packed_q8() {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_parse_env_bool("GGML_AME_PACKED_Q8") ? 1 : 0;
    }
    return cached == 1;
}

static bool ame_use_whole_k_q8() {
    static int cached = -1;
    if (cached == -1) {
        cached = ame_parse_env_bool("GGML_AME_WHOLE_K_Q8") ? 1 : 0;
    }
    return cached == 1;
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

static size_t ggml_ame_packed_q8_64_size(const ggml_tensor * tensor) {
    const int64_t nb64 = (tensor->ne[0] + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    return (size_t) ggml_nrows(tensor) * (size_t) nb64 * sizeof(block_q8_ame64);
}

static const block_q8_ame64 * ame_get_packed_q8_weight(
    const ggml_tensor * tensor,
    const int8_t ** tile_a_out,
    const float ** tile_scales_out
) {
    if (tile_a_out != nullptr) {
        *tile_a_out = nullptr;
    }
    if (tile_scales_out != nullptr) {
        *tile_scales_out = nullptr;
    }
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
    if (tile_a_out != nullptr) {
        *tile_a_out = static_cast<const int8_t *>(it->second.tile_a);
    }
    if (tile_scales_out != nullptr) {
        *tile_scales_out = it->second.tile_scales;
    }
    return static_cast<const block_q8_ame64 *>(it->second.data);
}

static void ame_build_packed_q8_tile_a_cache(
    ame_packed_q8_info & entry,
    const ggml_tensor * tensor
) {
    const int64_t K = tensor->ne[0];
    const int64_t M = tensor->ne[1];
    const int64_t nb64 = (K + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    const int64_t m_tiles = (M + AME_TILE_M - 1) / AME_TILE_M;
    const size_t tile_bytes = AME_TILE_M * AME_TILE_K * sizeof(int8_t);
    const size_t tile_a_size = (size_t) m_tiles * (size_t) nb64 * tile_bytes;
    const size_t tile_scales_count = (size_t) m_tiles * (size_t) nb64 * AME_TILE_M;

    if (entry.tile_a == nullptr || entry.tile_a_size != tile_a_size) {
        if (entry.tile_a != nullptr) {
            ggml_aligned_free(entry.tile_a, entry.tile_a_size);
        }
        entry.tile_a = ggml_aligned_malloc(tile_a_size);
        entry.tile_a_size = tile_a_size;
    }

    if (entry.tile_scales == nullptr || entry.tile_scales_count != tile_scales_count) {
        if (entry.tile_scales != nullptr) {
            ggml_aligned_free(entry.tile_scales, entry.tile_scales_count * sizeof(float));
        }
        entry.tile_scales = static_cast<float *>(ggml_aligned_malloc(tile_scales_count * sizeof(float)));
        entry.tile_scales_count = tile_scales_count;
    }

    if (entry.tile_a == nullptr || entry.tile_scales == nullptr || entry.data == nullptr) {
        return;
    }

    const block_q8_ame64 * blocks = static_cast<const block_q8_ame64 *>(entry.data);
    for (int64_t mt = 0; mt < m_tiles; ++mt) {
        const int64_t i0 = mt * AME_TILE_M;
        const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);

        for (int64_t kb = 0; kb < nb64; ++kb) {
            int8_t * dst_tile =
                static_cast<int8_t *>(entry.tile_a) + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_bytes;
            float * dst_scales =
                entry.tile_scales + ((size_t) mt * (size_t) nb64 + (size_t) kb) * AME_TILE_M;

            for (int i = 0; i < AME_TILE_M; ++i) {
                if (i < imax) {
                    const block_q8_ame64 * b = &blocks[(i0 + i) * nb64 + kb];
                    memcpy(&dst_tile[i * AME_TILE_K], b->qs, AME_Q8_PACK_K);
                    dst_scales[i] = GGML_FP16_TO_FP32(b->d);
                } else {
                    memset(&dst_tile[i * AME_TILE_K], 0, AME_Q8_PACK_K);
                    dst_scales[i] = 0.0f;
                }
            }
        }
    }
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
        if (ame_use_whole_k_q8()) {
            ggml_ame_repack_q8_0_to_ame64_whole_k(
                entry.data, data, ggml_nrows(tensor), tensor->ne[0]);
        } else {
            ggml_ame_repack_q8_0_to_ame64(
                entry.data, data, ggml_nrows(tensor), tensor->ne[0]);
        }
        ame_build_packed_q8_tile_a_cache(entry, tensor);
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
    const int64_t K = op->src[0]->ne[0];
    const int64_t M = op->src[0]->ne[1];
    const int64_t N = op->src[1]->ne[1];
    // Kernel dispatch enables packed-B solely through this flag. Workspace
    // planning must use the same condition; otherwise larger M shapes fall
    // back to a per-call allocation even though they still execute packed-B.
    const bool use_packed_b_panel = ame_backend_env_enabled("GGML_AME_USE_PACKED_B_PANEL");
    const ggml_ame_i8_kernel * kernel = ggml_ame_select_i8_kernel(M, N, K);
    return ggml_ame_i8_kernel_workspace_size(kernel, N, K, use_packed_b_panel ? 1 : 0);
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
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        const int8_t * packed_w_tile_a = nullptr;
        const float * packed_w_tile_scales = nullptr;
        const block_q8_ame64 * packed_w = ame_get_packed_q8_weight(src0, &packed_w_tile_a, &packed_w_tile_scales);
        if (packed_w != nullptr) {
            AME_LOG("backend_ame_mul_mat: dispatching to packed Q8_64 kernel whole_k=%d",
                ame_use_whole_k_q8() ? 1 : 0);
            ggml_ame_mul_mat_q8_0_ame64(
                packed_w,
                src1,
                src1->data,
                dst->data,
                ne00, ne01,
                ne10, ne11,
                src1->nb[1],
                packed_w_tile_a,
                packed_w_tile_scales,
                ggml_threadpool_graph_id(params->threadpool),
                params->threadpool,
                ame_tensor_content_generation(src1),
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
            if (op->src[1]->type != GGML_TYPE_F32 ||
                !ggml_ame_can_use_q8(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0])) {
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
        const bool f16_via_bf16 = ame_backend_can_use_f16_via_bf16(op);
        if (src0->type == GGML_TYPE_F16) {
            if (!f16_via_bf16) {
                return false;
            }
        } else if (src0->type == GGML_TYPE_Q8_0) {
            if (src1->type != GGML_TYPE_F32 ||
                !ggml_ame_can_use_q8(src0->ne[1], src1->ne[1], src0->ne[0])) {
                return false;
            }
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
    if (buffer->size != 0) {
        ctx->generation++;
    }

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

        const bool shape_supported = op->src[0]->type == GGML_TYPE_Q8_0
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

// Runtime AME availability check
static bool ggml_ame_available() {
#ifdef GGML_USE_RV_AME
    const ggml_ame_hw_config * cfg = ggml_ame_hw_config_get();
    if (ggml_ame_select_i8_kernel(64, 64, 64) == NULL) {
        AME_LOG("ggml_ame_available: returning false (hw probe valid=%d tlenb=%llu trlenb=%llu alenb=%llu capacity=%s)",
            cfg->valid,
            (unsigned long long) cfg->tlenb,
            (unsigned long long) cfg->trlenb,
            (unsigned long long) cfg->alenb,
            ggml_ame_i8_capacity_name(cfg));
        return false;
    }
    AME_LOG("ggml_ame_available: returning true (capacity=%s)", ggml_ame_i8_capacity_name(cfg));
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
