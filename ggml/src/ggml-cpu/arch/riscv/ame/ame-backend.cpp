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
                         memcpy(&tile_A[i * AME_TILE_K], blk->qs, AME_TILE_K);
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
                         memcpy(&tile_B[j * AME_TILE_K], blk->qs, AME_TILE_K);
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

// Check if AME can accelerate this operation
static bool qtype_has_ame_kernels(ggml_type type) {
    // AME supports Q8_0 (native int8) and Q4_0 (dequantized to int8)
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0;
}

// Compute forward for AME operations
static void ggml_backend_ame_mul_mat(ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // M
    const int64_t ne10 = src1->ne[0];  // K
    const int64_t ne11 = src1->ne[1];  // N

    GGML_ASSERT(ne00 == ne10);  // K dimensions must match

    AME_LOG("backend_ame_mul_mat: src0_type=%d M=%lld N=%lld K=%lld", src0->type, (long long)ne01, (long long)ne11, (long long)ne00);

    // Call AME implementation
    if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        AME_LOG("backend_ame_mul_mat: dispatching to Q8_0 kernel");
        // For Q8_0 x F32, we need to quantize src1 first
        // This is typically done in a separate step
        // For now, call the Q8_0 x Q8_0 kernel
        ggml_ame_mul_mat_q8_0(
            src0->data,
            src1->data,
            dst->data,
            ne00, ne01,
            ne10, ne11,
            src1->nb[1]  // pass stride
        );
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        AME_LOG("backend_ame_mul_mat: dispatching to Q4_0 kernel");
        // Q4_0 x F32: dequantize Q4_0 to int8, then use AME
        ggml_ame_mul_mat_q4_0(
            src0->data,
            src1->data,
            dst->data,
            ne00, ne01,
            ne10, ne11,
            src1->nb[1]
        );
    }

    GGML_UNUSED(params);
}

// AME tensor_traits implementation
namespace ggml::cpu::riscv_ame {

class tensor_traits : public ggml::cpu::tensor_traits {
public:
    bool work_size(int /* n_threads */, const struct ggml_tensor * /* op */, size_t & size) override {
        // AME doesn't need extra workspace for now
        size = 0;
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            static bool scalar_mode = (getenv("GGML_AME_SCALAR") != nullptr);
            static bool diff_mode = (getenv("GGML_AME_DIFF") != nullptr);
            
            if (scalar_mode) {
                // Scalar mode: only thread 0 computes, others skip
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
                        (float *)op->data,
                        M, N, K,
                        src1->nb[1]
                    );
                    return true;
                } else {
                    return false;
                }
            }
            
            if (diff_mode && params->ith == 0) {
                // DIFF mode: run AME, then reference, then compare
                const ggml_tensor * src0 = op->src[0];
                const ggml_tensor * src1 = op->src[1];
                const int64_t M = src0->ne[1];
                const int64_t N = src1->ne[1];
                const int64_t K = src0->ne[0];
                
                if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32) {
                    // Not supported for diff, just run AME
                    ggml_backend_ame_mul_mat(params, op);
                    return true;
                }
                
                // Allocate buffer for reference result
                const size_t result_size = M * N * sizeof(float);
                float * ref_data = (float *)malloc(result_size);
                float * ame_data = (float *)malloc(result_size);
                if (!ref_data || !ame_data) {
                    free(ref_data);
                    free(ame_data);
                    ggml_backend_ame_mul_mat(params, op);
                    return true;
                }
                
                // Save original output pointer
                float * original_dst = (float *)op->data;
                
                // 1. Run AME
                ggml_backend_ame_mul_mat(params, op);
                memcpy(ame_data, original_dst, result_size);
                
                // 2. Run reference
                reference_mul_mat_q8_0_f32(
                    src0->data,
                    src1->data,
                    ref_data,
                    M, N, K,
                    src1->nb[1]
                );
                
                // 3. Compare first few elements
                static int diff_count = 0;
                if (diff_count < 3) {
                    fprintf(stderr, "\n[DIFF] Matmul #%d: M=%lld N=%lld K=%lld\n", 
                            diff_count, (long long)M, (long long)N, (long long)K);
                    fprintf(stderr, "[DIFF] First 10 elements comparison:\n");
                    fprintf(stderr, "  Index | AME Result | REF Result | Diff\n");
                    for (int i = 0; i < 10 && i < M * N; i++) {
                        float diff = ame_data[i] - ref_data[i];
                        fprintf(stderr, "  %5d | %10.6f | %10.6f | %+.6f %s\n", 
                                i, ame_data[i], ref_data[i], diff,
                                (fabsf(diff) > 0.01f) ? "<!>" : "");
                    }
                    diff_count++;
                }
                
                // Use reference result as output
                memcpy(original_dst, ref_data, result_size);
                
                free(ref_data);
                free(ame_data);
                return true;
            }
            
            AME_LOG("tensor_traits::compute_forward: calling AME mul_mat");
            ggml_backend_ame_mul_mat(params, op);
            AME_LOG("tensor_traits::compute_forward: AME mul_mat completed");
            return true;
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}

}  // namespace ggml::cpu::riscv_ame

// AME buffer interface
static void ggml_backend_ame_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void * ggml_backend_ame_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)(buffer->context);
}

static enum ggml_status ggml_backend_ame_buffer_init_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor
) {
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
    // Repack Q4_0 weights to AME-optimized format (pre-unpacked int8)
    // This follows the AMX/SpaceMit pattern of preprocessing weights once
    if (tensor->type == GGML_TYPE_Q4_0 && offset == 0) {
        // Calculate number of Q4_0 blocks
        const int64_t ne0 = tensor->ne[0];
        const int64_t nblocks = ne0 / 32;  // Q4_0 block size is 32
        const int64_t total_blocks = nblocks * ggml_nrows(tensor);
        
        // Repack from block_q4_0 to block_q4_0_ame
        ggml_ame_repack_q4_0(tensor->data, data, total_blocks);
    } else {
        // For Q8_0 and other types, direct copy
        memcpy((char *)tensor->data + offset, data, size);
    }
    GGML_UNUSED(buffer);
}

static void ggml_backend_ame_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_ame_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_ame_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ame_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_ame_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_ame_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_ame_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
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

    return ggml_backend_buffer_init(buft, ggml_backend_ame_buffer_interface, data, size);
}

static size_t ggml_backend_ame_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;  // 64-byte alignment for RISC-V cache lines
    GGML_UNUSED(buft);
}

static size_t ggml_backend_ame_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft,
    const ggml_tensor * tensor
) {
    // Q4_0 needs extra space for repacked format
    // block_q4_0: 2 bytes (d) + 16 bytes (qs) = 18 bytes
    // block_q4_0_ame: 2 bytes (d) + 32 bytes (qs) = 34 bytes
    // Size increase: 34/18 = 1.889x
    if (tensor->type == GGML_TYPE_Q4_0) {
        const int64_t ne0 = tensor->ne[0];
        const int64_t nblocks = ne0 / 32;  // Q4_0 block size is 32
        const int64_t nrows = ggml_nrows(tensor);
        return nrows * nblocks * sizeof(block_q4_0_ame);
    }
    
    // For other types, return standard size
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

// Extra buffer type for operation support checking
namespace ggml::cpu::riscv_ame {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
public:
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // Handle only 2D matrix multiply for now
        auto is_contiguous_2d = [](const struct ggml_tensor * t) {
            return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
        };

        AME_LOG("supports_op: checking op=%d type0=%d type1=%d", op->op, 
                op->src[0] ? op->src[0]->type : -1, 
                op->src[1] ? op->src[1]->type : -1);

        if (op->op != GGML_OP_MUL_MAT) {
            AME_LOG("supports_op: reject (not MUL_MAT)");
            return false;
        }

        if (!is_contiguous_2d(op->src[0])) {
            AME_LOG("supports_op: reject (src0 not contiguous 2d)");
            return false;
        }

        if (!is_contiguous_2d(op->src[1])) {
            AME_LOG("supports_op: reject (src1 not contiguous 2d)");
            return false;
        }

        if (!op->src[0]->buffer) {
            AME_LOG("supports_op: reject (src0 has no buffer)");
            return false;
        }

        // TEMPORARY: Accept both AME buffer and host buffer for testing
        // In production, only AME buffer should be accepted for optimal performance
        bool is_ame_buffer = (op->src[0]->buffer->buft == ggml_backend_cpu_riscv_ame_buffer_type());
        bool is_host_buffer = ggml_backend_buft_is_host(op->src[0]->buffer->buft);
        
        if (!is_ame_buffer && !is_host_buffer) {
            AME_LOG("supports_op: reject (src0 buffer type mismatch: %p vs %p, name=%s)", 
                    (void*)op->src[0]->buffer->buft, 
                    (void*)ggml_backend_cpu_riscv_ame_buffer_type(),
                    op->src[0]->buffer->buft ? ggml_backend_buft_name(op->src[0]->buffer->buft) : "NULL");
            return false;
        }
        
        if (is_host_buffer) {
            AME_LOG("supports_op: accepting host buffer (testing mode)");
        }

        if (!qtype_has_ame_kernels(op->src[0]->type)) {
            AME_LOG("supports_op: reject (src0 type not supported)");
            return false;
        }
            
        // src1 must be host buffer
        if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
            AME_LOG("supports_op: reject (src1 not host)");
            return false;
        }
        // src1 must be float32
        if (op->src[1]->type == GGML_TYPE_F32) {
            AME_LOG("supports_op: accept M=%lld N=%lld K=%lld", (long long) op->src[0]->ne[1], (long long) op->src[1]->ne[1], (long long) op->src[0]->ne[0]);
            return true;
        }

        AME_LOG("supports_op: reject (src1 not F32)");
        return false;
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
