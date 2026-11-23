#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ame-backend.h"
#include "ame.h"

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu/vec.h"
#include "ggml-quants.h"
#include "ops.h"
#include "traits.h"
#if (defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE) || \
    (defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT)
#include "ggml-cpu/quants.h"
#endif
#if defined(GGML_XSAI_ALLOC)
#include "xsai_alloc.h"
#endif

#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__) && defined(__riscv)
#include <sched.h>
#endif

// The current XSAI Linux only enables first-use matrix state; it does not
// save/restore in-flight matrix registers on migration. Keep a worker on one
// hart until all its matrix submissions have drained, then restore the caller's
// affinity. Honor an existing strict GGML singleton mask.
class ame_worker_affinity {
#if defined(__linux__) && defined(__riscv)
    cpu_set_t saved;
    bool changed = false;
#endif
public:
    ame_worker_affinity(int ith, int nth) {
#if defined(__linux__) && defined(__riscv)
        GGML_ASSERT(sched_getaffinity(0, sizeof(saved), &saved) == 0);
        const int available = CPU_COUNT(&saved);
        GGML_ASSERT(available > 0);
        GGML_ASSERT(available == 1 || nth <= available);
        if (available > 1) {
            int ordinal = 0;
            int cpu = 0;
            for (; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &saved) && ordinal++ == ith) break;
            }
            GGML_ASSERT(cpu < CPU_SETSIZE);
            cpu_set_t selected;
            CPU_ZERO(&selected);
            CPU_SET(cpu, &selected);
            GGML_ASSERT(sched_setaffinity(0, sizeof(selected), &selected) == 0);
            changed = true;
        }
        AME_LOG("worker_binding ith=%d nth=%d cpu=%d", ith, nth, sched_getcpu());
#else
        GGML_UNUSED(ith);
        GGML_UNUSED(nth);
#endif
    }
    ~ame_worker_affinity() {
#if defined(__linux__) && defined(__riscv)
        if (changed) {
            GGML_ASSERT(sched_setaffinity(0, sizeof(saved), &saved) == 0);
        }
#endif
    }
};

// AME_DEBUG/AME_LOG now defined in ame.h
static bool ame_backend_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0 &&
        strcmp(value, "false") != 0 && strcmp(value, "off") != 0 && strcmp(value, "no") != 0;
}

static int ame_rmsnorm_mode() {
    const char * mode = getenv("GGML_AME_RMSNORM_MXFP4");
    return mode != nullptr && strcmp(mode, "serial") == 0 ? 1 :
        mode != nullptr && strcmp(mode, "pipeline") == 0 ? 2 :
        mode != nullptr && strcmp(mode, "pipeline-chunk") == 0 ? 3 : 0;
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
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_BF16 ||
        type == GGML_TYPE_MXFP4 || type == GGML_TYPE_F32;
}

static size_t ame_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static int ame_backend_mx_format(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) {
        return 0;
    }
    if (op->src[0]->type == GGML_TYPE_MXFP4) {
        return AME_MCFG_MXFP4;
    }
    if (op->src[0]->type != GGML_TYPE_F32) {
        return 0;
    }
    const char * format = getenv("GGML_AME_MXFP8_FORMAT");
    if (format == nullptr || strcmp(format, "e4m3") == 0) {
        return AME_MCFG_MXFP8_E4M3;
    }
    return strcmp(format, "e5m2") == 0 ? AME_MCFG_MXFP8_E5M2 : 0;
}

static bool ame_backend_mx_metadata_supported(const ggml_tensor * op) {
    // These are manual experimental configuration attestations, NOT capability
    // discovery. Legacy capacity CSRs do not report scaled-MMA support.
    const int format = ame_backend_mx_format(op);
    const int tile_mn = ggml_ame_mx_config_tile_mn(getenv("GGML_AME_MX_CONFIG"));
    const char * enable = getenv(format == AME_MCFG_MXFP4 ? "GGML_AME_MXFP4" : "GGML_AME_MXFP8");
    const char * verified = getenv("GGML_AME_MX32_VERIFIED");
    if (format == 0 || enable == nullptr || strcmp(enable, "1") != 0 ||
        verified == nullptr || strcmp(verified, "1") != 0 ||
        getenv("GGML_AME_SCALAR") != nullptr ||
        op == nullptr || op->op != GGML_OP_MUL_MAT ||
        op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (b->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(op) ||
        a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
        op->ne[2] != 1 || op->ne[3] != 1 || a->ne[0] != b->ne[0] ||
        op->ne[0] != a->ne[1] || op->ne[1] != b->ne[1] ||
        !ggml_ame_mx_shape_supported(a->ne[1], b->ne[1], a->ne[0], tile_mn)) {
        return false;
    }
    const size_t nb = (size_t) a->ne[0] / AME_MXFP4_K;
    // Include generic CPU Q8 activation workspace for allocation-failure
    // fallback, plus room for alignment and the native FP32 output tile.
    const size_t weight_block = ggml_ame_mx_block_bytes(format) + 1;
    const size_t activation_block = std::max(weight_block, sizeof(block_q8_0));
    if ((uint64_t) a->ne[1] > SIZE_MAX / nb / std::max(weight_block, sizeof(float) * AME_MXFP4_K) ||
        (uint64_t) ggml_ame_mx_padded_rows(a->ne[1], tile_mn) > SIZE_MAX / nb / weight_block ||
        (uint64_t) b->ne[1] > (SIZE_MAX - 65536) / nb / activation_block ||
        (uint64_t) a->ne[1] > UINT64_MAX / (uint64_t) b->ne[1] / (uint64_t) a->ne[0] / 2) {
        return false;
    }
    return true;
}

static bool ame_backend_can_use_mx(const ggml_tensor * op) {
    // Historical absent-config plans are software-only. Refuse native before
    // this function's buffer/capacity discovery; backend registration elsewhere
    // may still have its own canonical probe.
    if (!ame_backend_mx_metadata_supported(op) || getenv("GGML_AME_MX_CONFIG") == nullptr) {
        return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (a->buffer == nullptr || a->buffer->buft != ggml_backend_cpu_riscv_ame_buffer_type() ||
        (b->buffer != nullptr && b->buffer->buft != a->buffer->buft &&
            !ggml_backend_buft_is_host(b->buffer->buft))) {
        return false;
    }
    const ggml_ame_hw_config * cfg = ggml_ame_hw_config_get();
    return ggml_ame_mx_native_tile_mn(cfg, getenv("GGML_AME_MX_CONFIG")) != 0;
}

static int ame_backend_mx_tile_k(const ggml_tensor * op, int nth) {
    if (!ame_backend_mx_metadata_supported(op)) {
        return 0;
    }
    return ggml_ame_mx_select_tile_k(ame_backend_mx_format(op),
        op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0],
        nth, getenv("GGML_AME_MXFP4_K64"));
}

static size_t ame_mx_worker_wsize(const ggml_tensor * op) {
    const size_t tile_mn = (size_t) ggml_ame_mx_config_tile_mn(getenv("GGML_AME_MX_CONFIG"));
    const size_t blocks = (size_t) op->src[1]->ne[1] * ((size_t) op->src[0]->ne[0] / AME_MXFP4_K);
    const size_t native = blocks * (ggml_ame_mx_block_bytes(ame_backend_mx_format(op)) + 1) +
        tile_mn * tile_mn * sizeof(float);
    size_t size = std::max(native, blocks * sizeof(block_q8_0));
    if (op->src[0]->type == GGML_TYPE_MXFP4 && ame_rmsnorm_mode() != 0) {
        // Preserve both the old full-activation budget and generic fallback.
        // The fused path uses at most two panels plus this aligned prefix.
        const uint64_t rows = (uint64_t) op->src[1]->ne[1];
        if (rows > (SIZE_MAX - 63) / sizeof(float)) return 0;
        const size_t factors = ame_align_up((size_t) rows * sizeof(float), 64);
        if (size > SIZE_MAX - factors - 63) return 0;
        size += factors;
    }
    return ame_align_up(size, 64);
}

// Software half of the existing work_size path, independently testable without
// treating a capacity CSR as proof of scaled arithmetic support.
static bool ame_mx_work_size_plan(int n_threads, const ggml_tensor * op, size_t & size) {
    if (ame_backend_mx_tile_k(op, n_threads) == 0) {
        return false;
    }
    const size_t worker_size = ame_mx_worker_wsize(op);
    if (worker_size == 0 || worker_size > (SIZE_MAX - 63) / (size_t) n_threads) {
        return false;
    }
    size = worker_size * (size_t) n_threads + 63;
    return true;
}

static bool ame_mxfp4_activations_supported(const ggml_tensor * tensor) {
    const float * x = static_cast<const float *>(tensor->data);
    // The canonical legacy quantizer casts its scale exponent directly to
    // uint8_t. Do not invoke it on NaN/Inf or a nonzero block whose exponent
    // would be negative. Make the same decision on every worker before issue.
    for (int64_t base = 0; base < ggml_nelements(tensor); base += AME_MXFP4_K) {
        float amax = 0.0f;
        for (int k = 0; k < AME_MXFP4_K; ++k) {
            if (!std::isfinite(x[base + k])) {
                return false;
            }
            amax = std::max(amax, std::fabs(x[base + k]));
        }
        if (amax != 0.0f && amax < std::ldexp(1.0f, -125)) {
            return false;
        }
    }
    return true;
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
    static const bool cached = ame_parse_env_bool("GGML_AME_PACKED_Q8");
    return cached;
}

static bool ame_use_whole_k_q8() {
    static const bool cached = ame_parse_env_bool("GGML_AME_WHOLE_K_Q8");
    return cached;
}

struct ame_packed_q8_info {
    void * data = nullptr;
    size_t size = 0;
    void * tile_a = nullptr;
    size_t tile_a_size = 0;
    float * tile_scales = nullptr;
    size_t tile_scales_count = 0;
    int tile_m = 0;
};

struct ame_packed_mx_info {
    void * data = nullptr;
    size_t size = 0;
    uint64_t generation = 0;
    const void * source = nullptr;
    int64_t M = 0;
    int64_t padded_M = 0;
    int64_t K = 0;
    int format = 0;
    int tile_k = 32;
    int tile_mn = 64;
    size_t source_row_stride = 0;
    ggml_type source_type = GGML_TYPE_COUNT;
    bool domain_rejected = false;
    // MX32/E8M0, K32 or paired K64 panels. Format/layout/source store identity.
};

struct ame_buffer_context {
    void * base = nullptr;
    size_t base_size = 0;
    uint64_t generation = 1;
    std::unordered_map<const ggml_tensor *, ame_packed_q8_info> packed_q8;
    std::unordered_map<const ggml_tensor *, ame_packed_mx_info> packed_mxfp4;
    std::mutex mxfp4_mutex;
    uint64_t mxfp4_allocation_attempts = 0;
    // Same canonical allocator in production; injectable only by the software
    // selftest on its isolated buffer, to prove OOM fallback without huge allocs.
    void * (*mx_allocate)(size_t) = ggml_aligned_malloc;
    std::vector<ggml_tensor *> tensors;
};

static inline ame_buffer_context * ame_buffer_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<ame_buffer_context *>(buffer->context);
}

static bool ame_mx_pack_block(
    uint8_t * payload, uint8_t * scale, const ggml_tensor * tensor, size_t block, int format
) {
    if (format == AME_MCFG_MXFP4) {
        return ggml_ame_pack_mxfp4_blocks(
            payload, scale, static_cast<const block_mxfp4 *>(tensor->data) + block, 1);
    }
    return ggml_ame_pack_mxfp8_f32(
        payload, scale, static_cast<const float *>(tensor->data) + block * AME_MXFP4_K, AME_MXFP4_K, format);
}

static bool ame_mxfp4_cache_matches(const ame_packed_mx_info & entry, const ggml_tensor * tensor) {
    // CPU graph ops and callers can write tensor->data without a buffer
    // callback. Generation alone is not an immutable-content guarantee.
    // Validate exactly (no hash/collision assumption), using the SAME packer
    // rather than a second representation implementation. This conservative
    // correctness scan is charged to weight_prepare/full elapsed profiling.
    const size_t nb = (size_t) entry.K / AME_MXFP4_K;
    const size_t block_bytes = ggml_ame_mx_block_bytes(entry.format);
    const auto * payload = static_cast<const uint8_t *>(entry.data);
    const uint8_t * scales = payload + (size_t) entry.padded_M * nb * block_bytes;
    for (int64_t row = 0; row < entry.M; ++row) {
        for (size_t kb = 0; kb < nb; ++kb) {
            const size_t index = ggml_ame_mx_packed_block_index(row, kb, nb, entry.tile_k, entry.tile_mn);
            uint8_t expected[32], scale;
            if (!ame_mx_pack_block(expected, &scale, tensor, (size_t) row * nb + kb, entry.format) ||
                scale != scales[index] ||
                memcmp(expected, payload + index * block_bytes, block_bytes) != 0) {
                return false;
            }
        }
    }
    return true;
}

static const ame_packed_mx_info * ame_get_packed_mx_weight(
    const ggml_tensor * tensor, bool * cache_hit, int format, int tile_k = 32, int tile_mn = 64
) {
    *cache_hit = false;
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr ||
        !ggml_ame_mx_tile_k_supported(format, tile_k) ||
        tensor->type != (format == AME_MCFG_MXFP4 ? GGML_TYPE_MXFP4 : GGML_TYPE_F32) ||
        !ggml_is_contiguous(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
        !ggml_ame_mx_shape_supported(tensor->ne[1], tile_mn, tensor->ne[0], tile_mn) ||
        tensor->ne[0] % tile_k != 0 ||
        (uint64_t) tensor->ne[1] > SIZE_MAX / (uint64_t) tensor->ne[0] / sizeof(float) ||
        (uint64_t) ggml_ame_mx_padded_rows(tensor->ne[1], tile_mn) >
            SIZE_MAX / ((uint64_t) tensor->ne[0] / AME_MXFP4_K) / (ggml_ame_mx_block_bytes(format) + 1) ||
        tensor->ne[1] > INT64_MAX / tensor->ne[0]) {
        return nullptr;
    }
    ame_buffer_context * ctx = ame_buffer_ctx(tensor->buffer);
    const std::lock_guard<std::mutex> lock(ctx->mxfp4_mutex);
    *cache_hit = false;
    auto & entry = ctx->packed_mxfp4[tensor];
    const int64_t M = tensor->ne[1];
    const int64_t padded_M = ggml_ame_mx_padded_rows(M, tile_mn);
    const int64_t K = tensor->ne[0];
    const size_t nb = (size_t) K / AME_MXFP4_K;
    if (entry.generation == ctx->generation && entry.source == tensor->data &&
        entry.M == M && entry.padded_M == padded_M && entry.K == K && entry.format == format &&
        entry.tile_k == tile_k && entry.tile_mn == tile_mn && entry.source_row_stride == tensor->nb[1] &&
        entry.source_type == tensor->type) {
        if (entry.data == nullptr && entry.domain_rejected) {
            return nullptr; // Sticky invalid domain; CPU reads current original source bytes.
        }
        if (entry.data != nullptr && ame_mxfp4_cache_matches(entry, tensor)) {
            *cache_hit = true;
            return &entry;
        }
        // Direct CPU mutation or transient OOM: retry the same validation/fill
        // path. An allocation failure must never poison this identity.
    }
    // All buffer writes increment generation, including partial set/memset
    // and clear(nonzero). Rebuild under the same owning buffer, not a global
    // pointer cache. Only invalid-domain rejection is sticky for the current
    // generation/source/shape/format;
    // a transient allocation failure is retried on the next same-identity query.
    if (entry.data != nullptr) {
        ggml_aligned_free(entry.data, entry.size);
    }
    entry.data = nullptr;
    entry.size = 0;
    entry.generation = ctx->generation;
    entry.source = tensor->data;
    entry.M = M;
    entry.padded_M = padded_M;
    entry.K = K;
    entry.format = format;
    entry.tile_k = tile_k;
    entry.tile_mn = tile_mn;
    entry.source_row_stride = tensor->nb[1];
    entry.source_type = tensor->type;
    entry.domain_rejected = false;
    // Scan the WHOLE weight tensor before allocation or any native issue,
    // including all-zero blocks: GGML e255 decodes finite, but MX ISA e255
    // denotes NaN. The raw packer must still preserve 255 for format tests.
    const bool valid_domain = format == AME_MCFG_MXFP4 ?
        ggml_ame_mxfp4_native_domain(tensor->data, (int64_t) ((size_t) M * nb)) :
        ggml_ame_mxfp8_f32_domain(static_cast<const float *>(tensor->data), M * K, format);
    if (!valid_domain) {
        entry.domain_rejected = true;
        AME_LOG("mx_weight_reject format=0x%x reason=invalid_domain M=%lld K=%lld generation=%llu",
            format, (long long) M, (long long) K, (unsigned long long) ctx->generation);
        return nullptr;
    }
    const size_t block_bytes = ggml_ame_mx_block_bytes(format);
    entry.size = (size_t) padded_M * nb * (block_bytes + 1);
    ctx->mxfp4_allocation_attempts++;
    entry.data = ctx->mx_allocate(entry.size);
    if (entry.data == nullptr) {
        entry.size = 0;
        return nullptr;
    }
    uint8_t * payload = static_cast<uint8_t *>(entry.data);
    uint8_t * scales = payload + (size_t) padded_M * nb * block_bytes;
    for (int64_t mt = 0; mt < padded_M / tile_mn; ++mt) {
        for (size_t kb = 0; kb < nb; ++kb) {
            for (int row = 0; row < tile_mn; ++row) {
                const size_t index = ggml_ame_mx_packed_block_index(mt * tile_mn + row, kb, nb, tile_k, tile_mn);
                if (mt * tile_mn + row < M) {
                    GGML_ASSERT(ame_mx_pack_block(
                        payload + index * block_bytes, scales + index,
                        tensor, ((size_t) mt * (size_t) tile_mn + row) * nb + kb, format));
                } else {
                    // Padding is not source data. Never read beyond logical M
                    // and never use E8M0=255 (NaN) for an inactive row.
                    memset(payload + index * block_bytes, 0, block_bytes);
                    scales[index] = 127;
                }
            }
        }
    }
    AME_LOG("mx_weight_pack format=0x%x tile_k=%d tile_mn=%d M=%lld K=%lld generation=%llu bytes=%zu",
        format, tile_k, tile_mn, (long long) M, (long long) K, (unsigned long long) ctx->generation, entry.size);
    return &entry;
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
    const ggml_ame_i8_kernel * kernel,
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
    const bool matching_layout = kernel != nullptr && it->second.tile_m == kernel->tile_m;
    if (tile_a_out != nullptr && matching_layout) {
        *tile_a_out = static_cast<const int8_t *>(it->second.tile_a);
    }
    if (tile_scales_out != nullptr && matching_layout) {
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
    const ggml_ame_i8_kernel * kernel = ggml_ame_select_i8_kernel(M, AME_TILE_N_MAX, K);
    const int tile_m = ggml_ame_i8_128_whole_k_enabled() && kernel != nullptr ?
        kernel->tile_m : AME_TILE_M;
    const int64_t m_tiles = (M + tile_m - 1) / tile_m;
    const size_t tile_bytes = (size_t) tile_m * AME_TILE_K * sizeof(int8_t);
    const size_t tile_a_size = (size_t) m_tiles * (size_t) nb64 * tile_bytes;
    const size_t tile_scales_count = (size_t) m_tiles * (size_t) nb64 * tile_m;
    entry.tile_m = 0;

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
        const int64_t i0 = mt * tile_m;
        const int imax = (i0 + tile_m <= M) ? tile_m : (M - i0);

        for (int64_t kb = 0; kb < nb64; ++kb) {
            int8_t * dst_tile =
                static_cast<int8_t *>(entry.tile_a) + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_bytes;
            float * dst_scales =
                entry.tile_scales + ((size_t) mt * (size_t) nb64 + (size_t) kb) * tile_m;

            for (int i = 0; i < tile_m; ++i) {
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
    entry.tile_m = tile_m;
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
                params->wsize,
                params->ith, params->nth
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
    const ame_worker_affinity affinity(params->ith, params->nth);
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

    // GGML gives every worker the same wdata. Slice it before invoking the
    // existing kernels: no two harts may pack into the same A/B/C tiles.
    void * worker_data = nullptr;
    size_t worker_size = 0;
    if (src0->type == GGML_TYPE_Q8_0) {
        worker_size = ame_align_up(ggml_backend_ame_desired_wsize(dst), 64);
        GGML_ASSERT(params->wdata != nullptr && worker_size <= params->wsize / params->nth);
        worker_data = static_cast<char *>(params->wdata) + params->ith * worker_size;
    }

    if (src0->type == GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        const int8_t * packed_w_tile_a = nullptr;
        const float * packed_w_tile_scales = nullptr;
        const block_q8_ame64 * packed_w = ame_get_packed_q8_weight(
            src0, ggml_ame_select_i8_kernel(ne01, ne11, ne00),
            &packed_w_tile_a, &packed_w_tile_scales);
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
                worker_data,
                worker_size,
                params->ith, params->nth
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
                worker_data,
                worker_size,
                params->ith, params->nth
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
            params->wsize,
            params->ith, params->nth
        );
    }

    GGML_UNUSED(params);
}

static bool ame_disjoint_ranges(const void * a, size_t na, const void * b, size_t nb) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a), pb = reinterpret_cast<uintptr_t>(b);
    if (a == nullptr || b == nullptr || na == 0 || nb == 0 ||
        na > UINTPTR_MAX - pa || nb > UINTPTR_MAX - pb) return false;
    return pa + na <= pb || pb + nb <= pa;
}

static GGML_AME_THREAD_LOCAL uint64_t g_rms_fused_calls;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_prepared_panels;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_admitted_first_panels;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_overlap_callbacks;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_last_preflight_cycles;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_last_prepare_cycles;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_chunk_callback_visits;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_prepared_chunks;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_drain_chunks;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_last_callback_prepare_cycles;
static GGML_AME_THREAD_LOCAL uint64_t g_rms_last_drain_prepare_cycles;

// A declined fusion attempt is speculative CPU work, not part of the original
// graph. Restore exactly its entering scalar environment on decline (including
// every incoming sticky flag); successful work naturally accrues its own flags.
class ame_scalar_transaction {
    std::fenv_t saved;
    bool active;
public:
    bool valid;
    explicit ame_scalar_transaction(bool enabled) : active(enabled),
        valid(!enabled || std::fegetenv(&saved) == 0) {}
    void commit() { active = false; }
    ~ame_scalar_transaction() {
        if (active && valid) GGML_ASSERT(std::fesetenv(&saved) == 0);
    }
};

static bool ame_compute_mx_forward(
    ggml_compute_params * params, ggml_tensor * op,
    const ggml_tensor * rms_src = nullptr, const ggml_tensor * gamma = nullptr,
    float eps = 0.0f, int rms_mode = 0
) {
    const int format = ame_backend_mx_format(op);
    const bool profile = ame_backend_env_enabled("GGML_AME_PROFILE");
    const uint64_t op_start = profile ? ame_backend_read_cycle() : 0;
    const bool fused = rms_src != nullptr;
    ame_scalar_transaction scalar_try(fused);
    if (!scalar_try.valid || params->nth != 1 || !ame_backend_mx_metadata_supported(op)) return false;
    const int tile_k = ame_backend_mx_tile_k(op, params->nth);
    const int tile_mn = ggml_ame_mx_config_tile_mn(getenv("GGML_AME_MX_CONFIG"));
    if (tile_k == 0 || op->src[0]->data == nullptr || op->src[1]->data == nullptr) return false;
    const size_t worker_size = ame_mx_worker_wsize(op);
    const size_t offset = (64 - (uintptr_t) params->wdata % 64) % 64;
    if (worker_size == 0 || params->wdata == nullptr || params->wsize < offset ||
        worker_size > params->wsize - offset) return false;
    char * work = static_cast<char *>(params->wdata) + offset;
    ggml_ame_rmsnorm_input rms = {};
    const uint64_t preflight_start = profile ? ame_backend_read_cycle() : 0;
    // Domain admission stays before capacity discovery, packed allocation and
    // any AMU issue. Never inspect the not-yet-computed gamma-MUL data.
    if (fused) {
        if (format != AME_MCFG_MXFP4 || gamma == nullptr || rms_mode == 0) return false;
        const size_t N = (size_t) rms_src->ne[1], nb = (size_t) rms_src->ne[0] / AME_MXFP4_K;
        const size_t factor_bytes = ame_align_up(N * sizeof(float), 64);
        const size_t buffered_rows = std::min(N, (size_t) (2 * tile_mn));
        auto * b = reinterpret_cast<uint8_t *>(work) + factor_bytes;
        ggml_ame_rmsnorm_first_panel first = {
            b, b + buffered_rows * nb * 16, tile_k, profile, 0
        };
        if (!ggml_ame_rmsnorm_mxfp4_preflight(
                reinterpret_cast<float *>(work), static_cast<const float *>(rms_src->data),
                static_cast<const float *>(gamma->data), rms_src->ne[1], rms_src->ne[0],
                rms_src->nb[1], eps, tile_mn, &first)) return false;
        rms.gamma = static_cast<const float *>(gamma->data);
        rms.row_scales = reinterpret_cast<const float *>(work);
        rms.eps = eps;
        rms.pipeline = rms_mode - 1; // serial=0, legacy pipeline=1, chunk=2
        rms.first_panel_prepared = 1;
        rms.prepare_cycles = first.prepare_cycles;
        rms.prepared_panels = 1;
        rms.preflight_cycles = profile ? ame_backend_read_cycle() - preflight_start - first.prepare_cycles : 0;
    } else if (!(format == AME_MCFG_MXFP4 ? ame_mxfp4_activations_supported(op->src[1]) :
            ggml_ame_mxfp8_f32_domain(static_cast<const float *>(op->src[1]->data),
                ggml_nelements(op->src[1]), format))) {
        AME_LOG("mx_compute_reject format=0x%x reason=activation_domain", format);
        return false;
    }
    if (!(format == AME_MCFG_MXFP4 ?
            ggml_ame_mxfp4_native_domain(op->src[0]->data, ggml_nelements(op->src[0]) / AME_MXFP4_K) :
            ggml_ame_mxfp8_f32_domain(static_cast<const float *>(op->src[0]->data),
                ggml_nelements(op->src[0]), format))) {
        AME_LOG("mx_compute_reject format=0x%x reason=weight_domain", format);
        return false;
    }
    if (!ame_backend_can_use_mx(op)) return false;
    const uint64_t weight_start = profile ? ame_backend_read_cycle() : 0;
    bool cache_hit;
    const ame_packed_mx_info * packed = ame_get_packed_mx_weight(
        op->src[0], &cache_hit, format, tile_k, tile_mn);
    const uint64_t weight_cycles = profile ? ame_backend_read_cycle() - weight_start : 0;
    if (packed == nullptr) return false;
    if (fused && (!ame_disjoint_ranges(work, worker_size, packed->data, packed->size) ||
        !ame_disjoint_ranges(op->data, ggml_nbytes(op), packed->data, packed->size))) return false;
#if defined(GGML_XSAI_ALLOC)
    // Anonymous host-test mode is simulator-only, never physical AMU DMA.
    if ((!xsai_pool_phys_contiguous() || !xsai_in_pool(packed->data) ||
         !xsai_in_pool(params->wdata)) && !xsai_alloc_host_test_mode()) return false;
#else
    return false;
#endif
    {
        const ame_worker_affinity affinity(params->ith, params->nth);
        const uint8_t * a = static_cast<const uint8_t *>(packed->data);
        const size_t blocks = (size_t) packed->padded_M * ((size_t) packed->K / AME_MXFP4_K);
        AME_LOG("dispatch=%s_mx32 format=0x%x tile_k=%d tile_mn=%d ith=%d nth=%d M=%lld N=%lld K=%lld",
            format == AME_MCFG_MXFP4 ? "mxfp4" : "mxfp8", format, tile_k, tile_mn,
            params->ith, params->nth, (long long) packed->M,
            (long long) op->src[1]->ne[1], (long long) packed->K);
        scalar_try.commit(); // All bool-false paths precede native submission.
        ggml_ame_mul_mat_mx(
            a, a + blocks * ggml_ame_mx_block_bytes(format),
            fused ? rms_src->data : op->src[1]->data, static_cast<float *>(op->data),
            packed->M, op->src[1]->ne[1], packed->K, fused ? rms_src->nb[1] : op->src[1]->nb[1],
            work, params->ith, params->nth, weight_cycles, cache_hit ? 1 : 0, format, tile_k, tile_mn,
            fused ? &rms : nullptr);
    }
    if (profile) {
        const uint64_t elapsed = ame_backend_read_cycle() - op_start;
        ggml_ame_mx_profile_finish(elapsed, format, tile_k);
    }
    if (fused) {
        g_rms_fused_calls++;
        g_rms_prepared_panels += rms.prepared_panels;
        g_rms_admitted_first_panels += rms.first_panel_prepared != 0;
        g_rms_overlap_callbacks += rms.overlap_callbacks;
        g_rms_last_preflight_cycles = rms.preflight_cycles;
        g_rms_last_prepare_cycles = rms.prepare_cycles;
        g_rms_chunk_callback_visits += rms.chunk_callback_visits;
        g_rms_prepared_chunks += rms.prepared_chunks;
        g_rms_drain_chunks += rms.drain_chunks;
        g_rms_last_callback_prepare_cycles = rms.callback_prepare_cycles;
        g_rms_last_drain_prepare_cycles = rms.drain_prepare_cycles;
    }
    return true;
}

static bool ame_rmsnorm_graph_supported(const ggml_cgraph * graph, int index) {
    if (graph == nullptr || index < 0 || index > graph->n_nodes - 3 ||
        graph->use_counts == nullptr || graph->visited_hash_set.size == 0 ||
        graph->visited_hash_set.keys == nullptr || graph->visited_hash_set.used == nullptr ||
        graph->grads != nullptr || graph->grad_accs != nullptr) return false;
    const ggml_op ops[] = {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT};
    const int outputs[] = {index + 2};
    if (!ggml_can_fuse_subgraph(graph, index, 3, ops, outputs, 1)) return false;
    const ggml_tensor * norm = graph->nodes[index], * mul = graph->nodes[index + 1];
    const ggml_tensor * op = graph->nodes[index + 2];
    if (norm->src[0] == nullptr || mul->src[0] != norm || mul->src[1] == nullptr ||
        op->src[1] != mul || op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_MXFP4 ||
        ((norm->flags | mul->flags) & GGML_TENSOR_FLAG_INPUT) != 0 ||
        !ame_backend_mx_metadata_supported(op)) return false;
    const ggml_tensor * x = norm->src[0], * gamma = mul->src[1];
    if (x->type != GGML_TYPE_F32 || gamma->type != GGML_TYPE_F32 ||
        norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(gamma) ||
        !ggml_is_contiguous(norm) || !ggml_is_contiguous(mul) ||
        !ggml_are_same_shape(x, norm) || !ggml_are_same_shape(norm, mul) ||
        x->ne[2] != 1 || x->ne[3] != 1 || gamma->ne[0] != x->ne[0] ||
        gamma->ne[1] != 1 || gamma->ne[2] != 1 || gamma->ne[3] != 1 ||
        x->view_src != nullptr || gamma->view_src != nullptr ||
        op->view_src != nullptr || op->src[0]->view_src != nullptr) return false;
    for (int src = 1; src < GGML_MAX_SRC; ++src) if (norm->src[src] != nullptr) return false;
    for (int src = 2; src < GGML_MAX_SRC; ++src) {
        if (mul->src[src] != nullptr || op->src[src] != nullptr) return false;
    }
    return true;
}

extern "C" bool ggml_ame_try_fuse_rmsnorm_mxfp4(
    ggml_compute_params * params, const ggml_cgraph * graph, int node_index
) {
    const int mode = ame_rmsnorm_mode();
    if (mode == 0 || params == nullptr || params->ith != 0 || params->nth != 1 ||
        !ame_rmsnorm_graph_supported(graph, node_index)) return false;
    ggml_tensor * norm = graph->nodes[node_index], * mul = graph->nodes[node_index + 1];
    ggml_tensor * op = graph->nodes[node_index + 2];
    const ggml_tensor * x = norm->src[0], * gamma = mul->src[1], * w = op->src[0];
    if (x->data == nullptr || gamma->data == nullptr || w->data == nullptr ||
        norm->data == nullptr || mul->data == nullptr || op->data == nullptr ||
        params->wdata == nullptr) return false;
    // Gallocr can alias dead intermediates without view_src. Those tensors
    // are elided, so they need not be mutually disjoint; LIVE reads may not
    // alias either writer (Y or workspace), even partially.
    for (const ggml_tensor * read : {x, gamma, w}) {
        if (!ame_disjoint_ranges(read->data, ggml_nbytes(read), op->data, ggml_nbytes(op)) ||
            !ame_disjoint_ranges(read->data, ggml_nbytes(read), params->wdata, params->wsize)) return false;
    }
    if (!ame_disjoint_ranges(op->data, ggml_nbytes(op), params->wdata, params->wsize)) return false;
    auto hidden_observer = [&](const ggml_tensor * other) {
        return other != op && other != norm && other != mul &&
            (other->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) &&
            other->data != nullptr &&
            (!ame_disjoint_ranges(other->data, ggml_nbytes(other), norm->data, ggml_nbytes(norm)) ||
             !ame_disjoint_ranges(other->data, ggml_nbytes(other), mul->data, ggml_nbytes(mul)));
    };
    // Scheduler graph views have no leaf list and expose only slice nodes.
    // Their canonical visited set still contains the COMPLETE graph, including
    // observable INPUT/OUTPUT aliases outside this execution slice.
    const auto & visited = graph->visited_hash_set;
    for (size_t i = 0; i < visited.size; ++i) {
        if (ggml_bitset_get(visited.used, i) &&
            (visited.keys[i] == nullptr || hidden_observer(visited.keys[i]))) return false;
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(eps));
    return ame_compute_mx_forward(params, op, x, gamma, eps, mode);
}

// AME tensor_traits implementation
namespace ggml::cpu::riscv_ame {

class tensor_traits : public ggml::cpu::tensor_traits {
public:
    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        if (op->src[0]->type == GGML_TYPE_MXFP4 || op->src[0]->type == GGML_TYPE_F32) {
            // First scaled-state bring-up is T1. The existing worker affinity
            // helper cannot prove that multiple singleton masks are different
            // harts, and Linux does not yet preserve the complete scale context.
            if (!ame_mx_work_size_plan(n_threads, op, size)) {
                return false;
            }
            return ame_backend_can_use_mx(op);
        }
        if (op->src[0]->type == GGML_TYPE_Q8_0) {
            if (op->src[1]->type != GGML_TYPE_F32 ||
                !ggml_ame_can_use_q8(op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0])) {
                return false;
            }
            const size_t worker_size = ame_align_up(ggml_backend_ame_desired_wsize(op), 64);
            GGML_ASSERT(n_threads > 0 && worker_size <= SIZE_MAX / (size_t) n_threads);
            size = worker_size * (size_t) n_threads;
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

        if (op->src[0]->type == GGML_TYPE_MXFP4 || op->src[0]->type == GGML_TYPE_F32) {
            return ame_compute_mx_forward(params, op);
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
    for (auto & it : ctx->packed_mxfp4) {
        ggml_aligned_free(it.second.data, it.second.size);
    }
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

        const bool shape_supported = (op->src[0]->type == GGML_TYPE_MXFP4 || op->src[0]->type == GGML_TYPE_F32)
            ? ame_backend_can_use_mx(op) : op->src[0]->type == GGML_TYPE_Q8_0
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
            (op->src[0]->type == GGML_TYPE_MXFP4 && op->src[1]->type == GGML_TYPE_F32) ||
            (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
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

extern "C" bool ggml_ame_rmsnorm_fusion_selftest(bool run_native, int fixture, bool benchmark) {
    const char * names[] = {"GGML_AME_MX_CONFIG", "GGML_AME_MXFP4", "GGML_AME_MX32_VERIFIED",
        "GGML_AME_MXFP4_K64", "GGML_AME_SCALAR", "GGML_AME_RMSNORM_MXFP4",
        "GGML_XSAI_OP_PROFILE_DETAIL", "GGML_XSAI_OP_PROGRESS_LOG"};
    std::vector<std::string> saved;
    std::vector<bool> present;
    for (const char * name : names) {
        const char * value = getenv(name);
        saved.emplace_back(value == nullptr ? "" : value);
        present.push_back(value != nullptr);
        unsetenv(name);
    }
    std::fenv_t original_env;
    GGML_ASSERT(std::fegetenv(&original_env) == 0);
    GGML_ASSERT(std::fesetround(FE_TONEAREST) == 0);
    auto seed = []() {
        GGML_ASSERT(std::feclearexcept(FE_ALL_EXCEPT) == 0);
        GGML_ASSERT(std::feraiseexcept(FE_DIVBYZERO) == 0);
    };
    setenv("GGML_AME_MX_CONFIG", "CUTE_2Tops_128SCP", 1);
    setenv("GGML_AME_MXFP4", "1", 1);
    setenv("GGML_AME_MX32_VERIFIED", "1", 1);
    GGML_ASSERT(fixture >= 0 && fixture <= 10);
    // Keep fixtures 0..3 unchanged; add each legal single-Ktile boundary
    // with 1/2/3 token panels, including two-slot reuse on the third panel.
    // Fixture 10 is the explicit large performance shape, not a K-tail case.
    const bool target_fixture = fixture == 10;
    // The large perf entry keeps its timed output/work/light gates. Full
    // allocator/abort/OOM acceptance remains in the native-only entry.
    // Never change the historical fixtures' validation schedule.
    const bool screen_only = benchmark && target_fixture;
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
    const bool expect_unfused_adjacent = run_native && screen_only &&
        ame_backend_env_enabled("GGML_AME_RMS_TEST_NATIVE_UNFUSED");
#endif
    const bool chunk_fixture = fixture >= 4 && fixture <= 9;
    const int K = target_fixture ? 256 : chunk_fixture ? (fixture >= 7 ? 64 : 32) : 128;
    const int nb = K / 32;
    const int tile_k = chunk_fixture ? K : 64;
    setenv("GGML_AME_MXFP4_K64", tile_k == 64 ? "1" : "0", 1);
    setenv("GGML_AME_RMSNORM_MXFP4", chunk_fixture ? "pipeline-chunk" : "pipeline", 1);
    const int M = target_fixture ? 512 : fixture == 3 ? 288 : fixture != 0 ? 256 : 128;
    const int N = target_fixture ? 2048 : chunk_fixture ? 128 * (1 + (fixture - 4) % 3) :
        fixture >= 2 ? 384 : fixture == 1 ? 128 : 256;
    const unsigned n_panels = unsigned(N / 128);
    constexpr uint32_t poison = 0x7fc56789;
    auto fill_poison = [&](std::vector<float> & values) {
        for (float & v : values) memcpy(&v, &poison, sizeof(v));
    };
    auto is_poison = [&](const float * values, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            uint32_t bits; memcpy(&bits, values + i, 4);
            if (bits != poison) return false;
        }
        return true;
    };
    ggml_context * meta = ggml_init({1048576, nullptr, true});
    GGML_ASSERT(meta != nullptr);
    ggml_tensor * w = ggml_new_tensor_2d(meta, GGML_TYPE_MXFP4, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(meta, GGML_TYPE_F32, K, N);
    ggml_tensor * g = ggml_new_tensor_1d(meta, GGML_TYPE_F32, K);
    ggml_set_input(x); ggml_set_input(g);
    ggml_tensor * norm = ggml_rms_norm(meta, x, 0x1p-10f);
    ggml_tensor * mul = ggml_mul(meta, norm, g);
    ggml_tensor * dst = ggml_mul_mat(meta, w, mul);
    ggml_set_output(dst);
    ggml_cgraph * graph = ggml_new_graph(meta);
    ggml_build_forward_expand(graph, dst);
    GGML_ASSERT(graph->n_nodes == 3);
    auto buffer = ggml_backend_ame_buffer_type_alloc_buffer(
        ggml_backend_cpu_riscv_ame_buffer_type(), ggml_nbytes(w));
    GGML_ASSERT(buffer != nullptr);
    w->buffer = buffer; w->data = ggml_backend_buffer_get_base(buffer);
    ggml_backend_buffer_init_tensor(buffer, w);
    std::vector<block_mxfp4> weights(M * nb);
    for (int row = 0; row < M; ++row) for (int kb = 0; kb < nb; ++kb) {
        auto & b = weights[row * nb + kb];
        b.e = uint8_t(124 + (row + kb) % 5);
        for (int lane = 0; lane < 32; ++lane)
            b.qs[lane % 16] |= ((row * 3 + kb * 5 + lane) & 15) << (lane < 16 ? 0 : 4);
    }
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    std::vector<float> input(N * K), gamma(K);
    std::vector<float> norm_data(N * K + 32), mul_data(N * K + 32), output(M * N + 32);
    for (int row = 0; row < N; ++row) for (int k = 0; k < K; ++k)
        input[row * K + k] = float((row * 17 + k * 13) % 127 - 63) * 0x1p-5f;
    for (int k = 0; k < K; ++k) gamma[k] = float(32 + k % 65) * 0x1p-6f;
    const auto source_before = input, gamma_before = gamma;
    const auto weight_source_before = weights;
    x->data = input.data(); g->data = gamma.data();
    norm->data = norm_data.data() + 16; mul->data = mul_data.data() + 16;
    dst->data = output.data() + 16;
    size_t work_size = 0;
    GGML_ASSERT(ame_mx_work_size_plan(1, dst, work_size));
    void * work = ggml_aligned_malloc(work_size);
    GGML_ASSERT(work != nullptr);
    ggml_compute_params params = {}; params.nth = 1; params.wdata = work; params.wsize = work_size;
    bool ok = ame_rmsnorm_graph_supported(graph, 0);
    unsigned rejected = 0;
    auto reject = [&]() {
        const auto before_input = input, before_gamma = gamma;
        fill_poison(norm_data); fill_poison(mul_data); fill_poison(output);
        const auto calls = g_rms_fused_calls;
        seed();
        const bool refused = !ggml_ame_try_fuse_rmsnorm_mxfp4(&params, graph, 0);
        const bool flags = std::fetestexcept(FE_ALL_EXCEPT) == FE_DIVBYZERO;
        ok &= refused && flags && calls == g_rms_fused_calls && is_poison(output.data(), output.size()) &&
            is_poison(norm_data.data(), norm_data.size()) && is_poison(mul_data.data(), mul_data.size()) &&
            memcmp(input.data(), before_input.data(), input.size() * sizeof(float)) == 0 &&
            memcmp(gamma.data(), before_gamma.data(), gamma.size() * sizeof(float)) == 0;
        ++rejected;
    };
    norm->flags |= GGML_TENSOR_FLAG_OUTPUT; reject(); norm->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    mul->flags |= GGML_TENSOR_FLAG_INPUT; reject(); mul->flags &= ~GGML_TENSOR_FLAG_INPUT;
    norm->flags &= ~GGML_TENSOR_FLAG_COMPUTE; reject(); norm->flags |= GGML_TENSOR_FLAG_COMPUTE;
    params.nth = 2; reject(); params.nth = 1;
    const size_t old_wsize = params.wsize; params.wsize = 1; reject(); params.wsize = old_wsize;
    void * old_dst = dst->data; dst->data = input.data() + 1; reject(); dst->data = old_dst;
    params.wdata = input.data(); reject(); params.wdata = work;
    x->view_src = g; reject(); x->view_src = nullptr;
    g->ne[1] = 2; reject(); g->ne[1] = 1;
    const float eps = 0x1p-10f, zero_eps = 0.0f;
    memcpy(norm->op_params, &zero_eps, sizeof(float)); reject();
    memcpy(norm->op_params, &eps, sizeof(float));
    uint32_t qnan = 0x7fc12345;
    memcpy(input.data() + input.size() - 1, &qnan, 4); reject(); input = source_before;
    std::fill(input.begin(), input.end(), 1.0f);
    std::fill(input.end() - K, input.end(), 0x1p-20f);
    std::fill(gamma.begin(), gamma.end(), 0x1p-124f);
    const float three = 3.0f; memcpy(norm->op_params, &three, 4); reject();
    input = source_before; gamma = gamma_before; memcpy(norm->op_params, &eps, 4);
    // Full use counts survive a graph view; an out-of-slice second consumer
    // must still prevent elision. Do not manufacture a local use count.
    ggml_cgraph * fanout = ggml_new_graph(meta);
    ggml_build_forward_expand(fanout, dst);
    ggml_build_forward_expand(fanout, ggml_dup(meta, norm));
    ggml_cgraph slice = ggml_graph_view(fanout, 0, 3);
    ok &= !ame_rmsnorm_graph_supported(fanout, 0) && !ame_rmsnorm_graph_supported(&slice, 0);
    ggml_cgraph partial = ggml_graph_view(graph, 0, 2);
    ok &= !ame_rmsnorm_graph_supported(&partial, 0);
    unsigned observer_rejects = 0;
    auto reject_observer = [&](const ggml_cgraph * candidate) {
        const auto calls = g_rms_fused_calls;
        std::vector<uint8_t> before_x(ggml_nbytes(x));
        memcpy(before_x.data(), x->data, before_x.size());
        for (size_t i = 0; i < ggml_nelements(dst); ++i)
            memcpy(static_cast<float *>(dst->data) + i, &poison, sizeof(poison));
        seed();
        const bool refused = !ggml_ame_try_fuse_rmsnorm_mxfp4(&params, candidate, 0);
        ok &= refused && g_rms_fused_calls == calls &&
            std::fetestexcept(FE_ALL_EXCEPT) == FE_DIVBYZERO &&
            memcmp(before_x.data(), x->data, before_x.size()) == 0 &&
            gamma == gamma_before && is_poison(static_cast<const float *>(dst->data), M * N);
        ++observer_rejects;
    };
    if (ok) {
        // The REAL allocator may reuse an owned INPUT leaf for RMS/MUL.
        // That observable storage must prevent elision in BOTH a full graph
        // and the scheduler-shaped graph_view (which has no leaf list).
        ggml_tensor * tensors[] = {x, norm, mul, dst};
        void * data[4];
        ggml_backend_buffer_t buffers[4];
        for (int i = 0; i < 4; ++i) {
            data[i] = tensors[i]->data; buffers[i] = tensors[i]->buffer;
            tensors[i]->data = nullptr; tensors[i]->buffer = nullptr;
        }
        auto galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, graph));
        const bool input_reused = x->data == norm->data && norm->data == mul->data;
        ok &= input_reused && (x->flags & GGML_TENSOR_FLAG_INPUT) &&
            x->view_src == nullptr && norm->view_src == nullptr && mul->view_src == nullptr;
        memcpy(x->data, input.data(), ggml_nbytes(x));
        ggml_cgraph view = ggml_graph_view(graph, 0, 3);
        ok &= view.n_leafs == 0 && view.leafs == nullptr;
        reject_observer(graph);
        reject_observer(&view);
        printf("AME_RMS_OBSERVER_INPUT_CASE real_allocator=yes input_norm_mul_reused=%s "
            "fullgraph_and_view_decline=%s issue_count=0\n",
            input_reused ? "yes" : "no", ok ? "PASS" : "FAIL");
        for (int i = 0; i < 4; ++i) {
            tensors[i]->data = data[i]; tensors[i]->buffer = buffers[i];
        }
        ggml_gallocr_free(galloc);
    }
    if (ok) {
        // An independent OUTPUT object outside the slice can also observe a
        // partial range of the elided intermediate, without a dependency edge
        // to it. The canonical full visited set is the common authority.
        ggml_tensor * observer = ggml_dup(meta, g);
        ggml_set_output(observer);
        observer->data = static_cast<float *>(mul->data) + 1;
        ggml_cgraph * with_observer = ggml_new_graph(meta);
        ggml_build_forward_expand(with_observer, dst);
        ggml_build_forward_expand(with_observer, observer);
        ggml_cgraph view = ggml_graph_view(with_observer, 0, 3);
        ok &= with_observer->n_nodes == 4 && view.n_leafs == 0 &&
            ame_rmsnorm_graph_supported(with_observer, 0) &&
            ame_rmsnorm_graph_supported(&view, 0);
        const auto before_mul = mul_data;
        reject_observer(with_observer);
        reject_observer(&view);
        ok &= memcmp(before_mul.data(), mul_data.data(), mul_data.size() * sizeof(float)) == 0;
        printf("AME_RMS_OBSERVER_OUTPUT_CASE outside_slice=yes partial_alias=yes "
            "fullgraph_and_view_decline=%s issue_count=0\n", ok ? "PASS" : "FAIL");
    }
    unsigned native_variants = 0;
    if (ok && run_native) {
        if (benchmark) {
            fflush(stdout);
            fprintf(stderr, "AME_RMS_MEASUREMENT warmups_per_path=1 measured_per_path=3 "
            "mode_order=rotating cold_weight_pack_each_sample=yes oracle=once_outside_timers "
            "profile=%s timing=whole_graph platform=caller_defined "
            "fixture=%d M=%d N=%d K=%d logical_ops=%llu "
            "oracle_reuse=%s\n",
            ame_backend_env_enabled("GGML_AME_PROFILE") ? "enabled" : "disabled",
            fixture, M, N, K, (unsigned long long) (2ULL * M * N * K),
            target_fixture ? "packed_row_bitidentity" : "none");
        }
        auto pool_params = ggml_threadpool_params_default(1);
        auto pool = ggml_threadpool_new(&pool_params);
        ggml_cplan plan = ggml_graph_plan(graph, 1, pool);
        // The graph planner also owns thread cache-line padding. Allocate its
        // actual result, not just the AME operator's narrower software budget.
        if (plan.work_size > work_size) {
            ggml_aligned_free(work, work_size);
            work_size = plan.work_size;
            work = ggml_aligned_malloc(work_size);
            GGML_ASSERT(work != nullptr);
        }
        plan.work_data = static_cast<uint8_t *>(work); plan.work_size = work_size;
        std::vector<float> baseline, weighted, oracle(M * N);
        int baseline_flags = 0;
        bool oracle_ready = false;
        unsigned long saved_mfrm;
        asm volatile("csrr %0, 0x809" : "=r"(saved_mfrm));
        const char * modes[] = {"off", "serial", "pipeline", "pipeline-chunk"};
        const int mode_count = int(sizeof(modes) / sizeof(modes[0]));
        const int rounds = benchmark ? 4 : 1;
        for (int invocation = 0; invocation < mode_count * rounds && ok; ++invocation) {
            const int sample = invocation / mode_count, mode_index = invocation % mode_count;
            const char * mode = modes[(mode_index + sample) % mode_count];
            const char * phase = !benchmark ? "correctness" : sample == 0 ? "warmup" : "measured";
            setenv("GGML_AME_RMSNORM_MXFP4", mode, 1);
            fill_poison(norm_data); fill_poison(mul_data); fill_poison(output);
            const uint64_t calls = g_rms_fused_calls, panels = g_rms_prepared_panels;
            const uint64_t initial_panels = g_rms_admitted_first_panels;
            const uint64_t callbacks = g_rms_overlap_callbacks;
            const uint64_t visits = g_rms_chunk_callback_visits;
            const uint64_t chunks = g_rms_prepared_chunks, drains = g_rms_drain_chunks;
            const uint64_t bounded_before = ggml_ame_rmsnorm_bounded_preflight_count();
            // Same cold packed-weight identity for ALL variants. Do
            // not compare the first cold baseline to later cache-hit fusion.
            // Source provisioning is outside every timer; actual repacking
            // and allocation remain inside graph compute.
            ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
            const uint64_t allocations = ame_buffer_ctx(buffer)->mxfp4_allocation_attempts;
            // Existing counters are scoped to ONE graph invocation in this
            // test only. Reset/dump and all oracle/checker work are outside
            // its timer; the graph still includes cache allocation/packing.
            if (benchmark) ggml_ame_profile_reset();
#if (defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE) || \
    (defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT)
            ggml_mxfp4_packed_stats_reset();
#endif
            seed();
            const uint64_t begin = ame_backend_read_cycle();
            const auto status = ggml_graph_compute(graph, &plan);
            const uint64_t elapsed = ame_backend_read_cycle() - begin;
            const int flags = std::fetestexcept(FE_ALL_EXCEPT);
            unsigned long after_mfrm;
            asm volatile("csrr %0, 0x809" : "=r"(after_mfrm));
#if (defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE) || \
    (defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT)
            struct ggml_mxfp4_packed_stats packed_phase;
            ggml_mxfp4_packed_stats_get(&packed_phase); // Before the untimed oracle.
#endif
            const bool fused = strcmp(mode, "off") != 0;
            const bool chunk = strcmp(mode, "pipeline-chunk") == 0;
            const uint64_t expected_chunks = chunk ? uint64_t(n_panels - 1) * (K / tile_k) : 0;
            const uint64_t expected_callbacks = chunk ? expected_chunks :
                strcmp(mode, "pipeline") == 0 ? n_panels - 1 : 0U;
            const uint64_t actual_calls = g_rms_fused_calls - calls;
            const uint64_t actual_allocations = ame_buffer_ctx(buffer)->mxfp4_allocation_attempts - allocations;
            const uint64_t actual_panels = g_rms_prepared_panels - panels;
            const uint64_t actual_first_panels = g_rms_admitted_first_panels - initial_panels;
            const uint64_t actual_callbacks = g_rms_overlap_callbacks - callbacks;
            const uint64_t actual_visits = g_rms_chunk_callback_visits - visits;
            const uint64_t actual_chunks = g_rms_prepared_chunks - chunks;
            const uint64_t actual_drains = g_rms_drain_chunks - drains;
            const uint64_t bounded_preflights =
                ggml_ame_rmsnorm_bounded_preflight_count() - bounded_before;
#if defined(GGML_AME_RMS_PREFLIGHT_BOUNDS) && GGML_AME_RMS_PREFLIGHT_BOUNDS
            const unsigned bounds_compiled = 1;
#else
            const unsigned bounds_compiled = 0;
#endif
            const bool bounds_ok = !target_fixture ||
                bounded_preflights == (fused ? bounds_compiled : 0U);
            const bool status_ok = status == GGML_STATUS_SUCCESS;
            const bool mfrm_ok = after_mfrm == saved_mfrm;
            const bool counters_ok = actual_calls == unsigned(fused) && actual_allocations == 1 &&
                actual_panels == (fused ? n_panels : 0U) &&
                actual_first_panels == unsigned(fused) && actual_callbacks == expected_callbacks &&
                actual_visits == expected_chunks && actual_chunks == expected_chunks && actual_drains == 0;
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
            const uint64_t off_calls = expect_unfused_adjacent ? uint64_t(N) * ((nb + 7) / 8) : 0;
            const uint64_t off_blocks = expect_unfused_adjacent ? uint64_t(N) * nb : 0;
            ok &= packed_phase.adjacent_fallbacks == 0 && (fused ?
                packed_phase.adjacent_calls > 0 && packed_phase.adjacent_blocks > 0 :
                packed_phase.adjacent_calls == off_calls && packed_phase.adjacent_blocks == off_blocks);
            if (!fused && expect_unfused_adjacent) {
                ok &= packed_phase.calls == off_calls && packed_phase.packed_calls == off_calls &&
                    packed_phase.packed_blocks == off_blocks && packed_phase.qualification_fallbacks == 0 &&
                    packed_phase.packed_nv_fallbacks == 0;
            }
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
            ok &= packed_phase.amax_reuse_calls == (fused ? packed_phase.adjacent_calls : 0) &&
                packed_phase.amax_reuse_blocks == (fused ? packed_phase.adjacent_blocks : 0) &&
                packed_phase.amax_reuse_fallbacks == 0;
#endif
            const bool prior_ok = ok;
            const bool first_oracle = !oracle_ready;
            if (first_oracle) {
                GGML_ASSERT(!fused); // The first warmup/control is always off.
                baseline.assign(output.begin() + 16, output.end() - 16);
                weighted.assign(mul_data.begin() + 16, mul_data.end() - 16);
                baseline_flags = flags;
                if (target_fixture) {
                    // Independent original ordered RMS, not another GGML
                    // graph compiled with the same candidate reduction flag.
                    // This fixture's exact F32-square inputs also require
                    // bitexact normalized/weighted values before its oracle.
                    uint32_t max_factor_ulp = 0;
                    bool reference_match = true;
                    for (int row = 0; row < N; ++row) {
                        const float * xr = input.data() + row * K;
                        const float reference_r = ggml_vec_rms_norm_scale_f32_reference(K, xr, eps);
                        const float candidate_r = ggml_vec_rms_norm_scale_f32(K, xr, eps);
                        uint32_t reference_bits, candidate_bits;
                        memcpy(&reference_bits, &reference_r, sizeof(reference_bits));
                        memcpy(&candidate_bits, &candidate_r, sizeof(candidate_bits));
                        const uint32_t ulp = reference_bits >= candidate_bits ?
                            reference_bits - candidate_bits : candidate_bits - reference_bits;
                        max_factor_ulp = std::max(max_factor_ulp, ulp);
                        reference_match &= reference_r > 0 && candidate_r > 0 &&
                            std::isfinite(reference_r) && std::isfinite(candidate_r);
                        for (int k = 0; k < K; ++k) {
                            volatile float reference_norm = xr[k] * reference_r;
                            volatile float reference_weighted = reference_norm * gamma[k];
                            const float rn = reference_norm, rw = reference_weighted;
                            reference_match &= memcmp(&rn, norm_data.data() + 16 + row * K + k,
                                sizeof(float)) == 0 &&
                                memcmp(&rw, weighted.data() + row * K + k, sizeof(float)) == 0;
                        }
                    }
                    const bool reference_ok = reference_match && max_factor_ulp <= 3;
                    ok &= reference_ok;
                    fprintf(stderr, "AME_RMS_REFERENCE_CHECK fixture=10 rows=%d K=%d "
                        "ordered_reference=yes max_factor_ulp=%u factor_ulp_limit=3 "
                        "normalized_weighted=fixed_fixture_bitexact timing=outside_samples "
                        "exact=%s full_model_quality=not_inferred\n",
                        N, K, max_factor_ulp, reference_ok ? "PASS" : "FAIL");
                }
                // Independent logical GGML blocks, exact double products,
                // explicit F32 rounding after every K32, never a whole-K dot.
                std::vector<block_mxfp4> activations(N * nb);
                const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->from_float;
                for (int row = 0; row < N; ++row)
                    quantize(weighted.data() + row * K, activations.data() + row * nb, K);
                const double mag[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
                auto decode = [&](const block_mxfp4 & b, int lane) {
                    const unsigned code = (b.qs[lane % 16] >> (lane < 16 ? 0 : 4)) & 15;
                    return std::ldexp((code & 8 ? -1.0 : 1.0) * mag[code & 7], int(b.e) - 127);
                };
                for (int col = 0; col < N; ++col) {
                    // The deterministic input generator repeats every 127 rows.
                    // Reuse only the untimed reference, and only after proving
                    // the actual packed rows equal; mismatches use the full dot.
                    if (target_fixture && col >= 127 && memcmp(
                            activations.data() + col * nb,
                            activations.data() + (col % 127) * nb,
                            size_t(nb) * sizeof(block_mxfp4)) == 0) {
                        memcpy(oracle.data() + col * M, oracle.data() + (col % 127) * M,
                            size_t(M) * sizeof(float));
                        continue;
                    }
                    for (int row = 0; row < M; ++row) {
                        // Weight exponents/codes repeat every lcm(5,16)=80 rows.
                        // No assumption about generator identity substitutes for
                        // checking the packed bytes used by this oracle.
                        if (target_fixture && row >= 80 && memcmp(
                                weights.data() + row * nb,
                                weights.data() + (row % 80) * nb,
                                size_t(nb) * sizeof(block_mxfp4)) == 0) {
                            oracle[col * M + row] = oracle[col * M + row % 80];
                            continue;
                        }
                        float acc = 0;
                        for (int kb = 0; kb < nb; ++kb) {
                            double sum = 0;
                            for (int lane = 0; lane < 32; ++lane)
                                sum += decode(weights[row * nb + kb], lane) * decode(activations[col * nb + kb], lane);
                            volatile float rounded = float(double(acc) + sum);
                            acc = rounded;
                        }
                        oracle[col * M + row] = acc;
                    }
                }
                oracle_ready = true;
            }
            size_t mismatch_index = SIZE_MAX;
            uint32_t mismatch_actual = 0, mismatch_expected = 0;
            for (size_t i = 0; i < oracle.size(); ++i) {
                uint32_t actual_bits, expected_bits;
                memcpy(&actual_bits, output.data() + 16 + i, sizeof(actual_bits));
                memcpy(&expected_bits, oracle.data() + i, sizeof(expected_bits));
                if (actual_bits != expected_bits) {
                    mismatch_index = i;
                    mismatch_actual = actual_bits;
                    mismatch_expected = expected_bits;
                    break;
                }
            }
            const bool oracle_match = mismatch_index == SIZE_MAX;
            const bool flags_ok = first_oracle || flags == baseline_flags;
            const bool norm_canary_ok = !fused || is_poison(norm_data.data(), norm_data.size());
            const bool mul_canary_ok = !fused || is_poison(mul_data.data(), mul_data.size());
            const bool output_canary_ok = is_poison(output.data(), 16) &&
                is_poison(output.data() + 16 + M * N, 16);
            const bool input_unchanged = input == source_before;
            const bool gamma_unchanged = gamma == gamma_before;
            const bool weight_source_unchanged = memcmp(
                weights.data(), weight_source_before.data(), weights.size() * sizeof(weights[0])) == 0;
            const bool iteration_ok = status_ok && mfrm_ok && counters_ok && oracle_match && flags_ok &&
                norm_canary_ok && mul_canary_ok && output_canary_ok && input_unchanged && gamma_unchanged &&
                bounds_ok;
            ok &= iteration_ok;
            if (!iteration_ok) {
                fprintf(stderr,
                    "AME_RMS_FUSION_DIAG mode=%s phase=%s sample=%d prior_ok=%d "
                    "status_ok=%d mfrm_before=%lx mfrm_after=%lx mfrm_ok=%d "
                    "calls=%llu/%u allocations=%llu/1 panels=%llu/%u first_panels=%llu/%u "
                    "callbacks=%llu/%llu visits=%llu/%llu chunks=%llu/%llu drains=%llu/0 "
                    "oracle_match=%d mismatch_index=%zu actual=0x%08x expected=0x%08x "
                    "flags=0x%x expected_flags=0x%x flags_ok=%d "
                    "norm_canary=%d mul_canary=%d output_canary=%d input_unchanged=%d "
                    "gamma_unchanged=%d weight_source_unchanged=%d\n",
                    mode, phase, sample, prior_ok ? 1 : 0, status_ok ? 1 : 0,
                    saved_mfrm, after_mfrm, mfrm_ok ? 1 : 0,
                    (unsigned long long) actual_calls, unsigned(fused),
                    (unsigned long long) actual_allocations,
                    (unsigned long long) actual_panels, fused ? n_panels : 0U,
                    (unsigned long long) actual_first_panels, unsigned(fused),
                    (unsigned long long) actual_callbacks, (unsigned long long) expected_callbacks,
                    (unsigned long long) actual_visits, (unsigned long long) expected_chunks,
                    (unsigned long long) actual_chunks, (unsigned long long) expected_chunks,
                    (unsigned long long) actual_drains,
                    oracle_match ? 1 : 0, mismatch_index, mismatch_actual, mismatch_expected,
                    flags, baseline_flags, flags_ok ? 1 : 0,
                    norm_canary_ok ? 1 : 0, mul_canary_ok ? 1 : 0, output_canary_ok ? 1 : 0,
                    input_unchanged ? 1 : 0, gamma_unchanged ? 1 : 0,
                    weight_source_unchanged ? 1 : 0);
            }
            fprintf(benchmark ? stderr : stdout,
                "AME_RMS_FUSION_CASE mode=%s phase=%s sample=%d seq=%d hidden=%d output=%d cycles=%llu "
                "scope=whole_graph_RMS_gamma_quant_pack_GEMM_output flags=%x "
                "weight_pack=cold_all_variants packed_allocations=1 "
                "fused_calls=%llu panels=%llu callbacks=%llu norm_gamma_preflight=%llu panel_quant_pack=%llu "
                "first_panels_admitted=%llu "
                "activation_slots=%u slot_rewrites=%u output_channel_tiles=%d "
                "chunk_callback_visits=%llu prepared_chunks=%llu drain_chunks=%llu "
                "callback_prepare_cycles=%llu drain_prepare_cycles=%llu prepare_subsets_not_additive=yes "
                "bounded_preflights=%llu bounds_compiled=%u "
                "exact=%s performance_platform=caller_defined\n",
                mode, phase, sample, N, K, M, (unsigned long long) elapsed, flags,
                (unsigned long long) (g_rms_fused_calls - calls),
                (unsigned long long) (g_rms_prepared_panels - panels),
                (unsigned long long) (g_rms_overlap_callbacks - callbacks),
                (unsigned long long) (fused ? g_rms_last_preflight_cycles : 0),
                (unsigned long long) (fused ? g_rms_last_prepare_cycles : 0),
                (unsigned long long) (g_rms_admitted_first_panels - initial_panels),
                fused ? std::min(n_panels, 2U) : 0U, fused && n_panels > 2 ? n_panels - 2 : 0U,
                (M + 127) / 128,
                (unsigned long long) (g_rms_chunk_callback_visits - visits),
                (unsigned long long) (g_rms_prepared_chunks - chunks),
                (unsigned long long) (g_rms_drain_chunks - drains),
                (unsigned long long) (fused ? g_rms_last_callback_prepare_cycles : 0),
                (unsigned long long) (fused ? g_rms_last_drain_prepare_cycles : 0),
                (unsigned long long) bounded_preflights, bounds_compiled,
                ok ? "PASS" : "FAIL");
            if (benchmark) {
                fprintf(stderr, "AME_RMS_SAMPLE_PROFILE_BEGIN mode=%s phase=%s sample=%d calls_expected=1\n",
                    mode, phase, sample);
                ggml_ame_profile_dump_now();
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
                fprintf(stderr, "AME_RMS_PACKED_PHASE mode=%s phase=%s sample=%d "
                    "calls=%llu packed_calls=%llu packed_blocks=%llu qualification_fallbacks=%llu packed_nv_fallbacks=%llu "
                    "rms_rows=%llu rms_blocks=%llu rms_stats=%llu rms_apply=%llu rms_finite_scan=%llu "
                    "packed_qualification=%llu vector_pack=%llu split_reorder=%llu rms_AME_reorder=%llu "
                    "scope=production_packed_instrumented raw_cycles_no_overhead_subtraction=yes "
                    "helper_counters_exclude_unfused_RMS=yes rejected_qualification_cycles_not_counted=yes\n",
                    mode, phase, sample,
                    (unsigned long long) packed_phase.calls, (unsigned long long) packed_phase.packed_calls,
                    (unsigned long long) packed_phase.packed_blocks,
                    (unsigned long long) packed_phase.qualification_fallbacks,
                    (unsigned long long) packed_phase.packed_nv_fallbacks,
                    (unsigned long long) packed_phase.rms_rows, (unsigned long long) packed_phase.rms_blocks,
                    (unsigned long long) packed_phase.rms_stats_cycles,
                    (unsigned long long) packed_phase.rms_apply_cycles,
                    (unsigned long long) packed_phase.rms_scan_cycles,
                    (unsigned long long) packed_phase.qualification_cycles,
                    (unsigned long long) packed_phase.vector_pack_cycles,
                    (unsigned long long) packed_phase.split_reorder_cycles,
                    (unsigned long long) packed_phase.rms_reorder_cycles);
#endif
#if defined(GGML_AME_MXFP4_ADJACENT) && GGML_AME_MXFP4_ADJACENT
                fprintf(stderr, "AME_RMS_ADJACENT_OUTPUT mode=%s phase=%s sample=%d "
                    "calls=%llu packed_blocks=%llu fallbacks=%llu generic_layout_unchanged=yes "
                    "quant_calls=%llu vector_calls=%llu vector_blocks=%llu "
                    "qualification_fallbacks=%llu packed_nv_fallbacks=%llu\n",
                    mode, phase, sample,
                    (unsigned long long) packed_phase.adjacent_calls,
                    (unsigned long long) packed_phase.adjacent_blocks,
                    (unsigned long long) packed_phase.adjacent_fallbacks,
                    (unsigned long long) packed_phase.calls,
                    (unsigned long long) packed_phase.packed_calls,
                    (unsigned long long) packed_phase.packed_blocks,
                    (unsigned long long) packed_phase.qualification_fallbacks,
                    (unsigned long long) packed_phase.packed_nv_fallbacks);
#endif
#if defined(GGML_AME_MXFP4_REUSE_AMAX) && GGML_AME_MXFP4_REUSE_AMAX
                fprintf(stderr, "AME_RMS_AMAX_REUSE mode=%s phase=%s sample=%d "
                    "calls=%llu packed_blocks=%llu fallbacks=%llu same_call_weighted_lanes=yes\n",
                    mode, phase, sample,
                    (unsigned long long) packed_phase.amax_reuse_calls,
                    (unsigned long long) packed_phase.amax_reuse_blocks,
                    (unsigned long long) packed_phase.amax_reuse_fallbacks);
#endif
                fprintf(stderr, "AME_RMS_SAMPLE_PROFILE_END mode=%s phase=%s sample=%d\n", mode, phase, sample);
            }
            if (!ok) break;
            ++native_variants;
        }
        if (benchmark) {
            // The following allocation/abort/OOM correctness cases are NOT
            // samples. Keep them out of every delimited sample profile.
            ggml_ame_profile_reset();
            if (screen_only) {
                fprintf(stderr, "AME_RMS_MEASUREMENT_END measured_graphs=%u warmup_graphs=%d "
                    "extra_correctness_follows=no full_acceptance=deferred exact=%s\n",
                    native_variants >= unsigned(mode_count) ? native_variants - mode_count : 0,
                    mode_count, ok ? "PASS" : "FAIL");
            } else {
                fprintf(stderr, "AME_RMS_MEASUREMENT_END measured_graphs=%u warmup_graphs=%d "
                    "extra_correctness_follows=yes exact=%s\n",
                    native_variants >= unsigned(mode_count) ? native_variants - mode_count : 0,
                    mode_count, ok ? "PASS" : "FAIL");
            }
        }
        for (const char * allocator_mode : {"pipeline", "pipeline-chunk"}) {
            if (!ok || screen_only) break;
            // Exercise the REAL allocator seam: RMS/MUL may share an owned
            // allocation despite both having view_src == NULL.
            void * old_norm = norm->data, * old_mul = mul->data;
            norm->data = mul->data = dst->data = nullptr;
            auto galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, graph));
            const bool reused = norm->data == mul->data;
            for (size_t i = 0; i < ggml_nelements(norm); ++i)
                memcpy(static_cast<float *>(norm->data) + i, &poison, 4);
            for (size_t i = 0; i < ggml_nelements(dst); ++i)
                memcpy(static_cast<float *>(dst->data) + i, &poison, 4);
            setenv("GGML_AME_RMSNORM_MXFP4", allocator_mode, 1);
            const uint64_t calls = g_rms_fused_calls;
            seed();
            const auto status = ggml_graph_compute(graph, &plan);
            const int flags = std::fetestexcept(FE_ALL_EXCEPT);
            ok &= reused && norm->view_src == nullptr && mul->view_src == nullptr &&
                status == GGML_STATUS_SUCCESS && g_rms_fused_calls == calls + 1 &&
                flags == baseline_flags && is_poison(static_cast<const float *>(norm->data), N * K) &&
                memcmp(dst->data, oracle.data(), oracle.size() * sizeof(float)) == 0 &&
                input == source_before && gamma == gamma_before;
            printf("AME_RMS_GALLOC_CASE real_allocator=yes norm_mul_reused=%s "
                "fused_calls=%llu exact=%s\n", reused ? "yes" : "no",
                (unsigned long long) (g_rms_fused_calls - calls), ok ? "PASS" : "FAIL");
            norm->data = old_norm; mul->data = old_mul; dst->data = old_dst;
            norm->buffer = mul->buffer = dst->buffer = nullptr;
            ggml_gallocr_free(galloc);
        }
        for (const char * abort_mode : {"pipeline", "pipeline-chunk"}) {
            if (!ok || screen_only) break;
            // A real abort after norm must not execute gamma or write Y.
            setenv("GGML_AME_RMSNORM_MXFP4", abort_mode, 1);
            int abort_checks = 0;
            plan.abort_callback = [](void * opaque) {
                ++*static_cast<int *>(opaque);
                return true;
            };
            plan.abort_callback_data = &abort_checks;
            fill_poison(norm_data); fill_poison(mul_data); fill_poison(output);
            const auto calls = g_rms_fused_calls;
            seed();
            ok &= ggml_graph_compute(graph, &plan) == GGML_STATUS_ABORTED &&
                abort_checks == 1 && g_rms_fused_calls == calls &&
                !is_poison(norm_data.data() + 16, N * K) &&
                is_poison(mul_data.data(), mul_data.size()) && is_poison(output.data(), output.size());
            plan.abort_callback = nullptr; plan.abort_callback_data = nullptr;
            printf("AME_RMS_ABORT_CASE checks=%d elided_nodes=0 output_untouched=%s\n",
                abort_checks, ok ? "PASS" : "FAIL");
        }
        if (ok && !screen_only) {
            // Canonical allocation failure seam, on this test's OWN buffer.
            // Both controls must execute the real generic CPU fallback, not
            // compare fallback Q8 activations against successful W4A4 output.
            auto * ctx = ame_buffer_ctx(buffer);
            const auto allocate = ctx->mx_allocate;
            ctx->mx_allocate = [](size_t) -> void * { return nullptr; };
            ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
            std::vector<float> fallback;
            int fallback_flags = 0;
            for (const char * mode : {"off", "pipeline", "pipeline-chunk"}) {
                setenv("GGML_AME_RMSNORM_MXFP4", mode, 1);
                fill_poison(norm_data); fill_poison(mul_data); fill_poison(output);
                const auto calls = g_rms_fused_calls;
                seed();
                const auto status = ggml_graph_compute(graph, &plan);
                const int flags = std::fetestexcept(FE_ALL_EXCEPT);
                ok &= status == GGML_STATUS_SUCCESS && calls == g_rms_fused_calls &&
                    !is_poison(norm_data.data() + 16, N * K) &&
                    !is_poison(mul_data.data() + 16, N * K) &&
                    is_poison(output.data(), 16) && is_poison(output.data() + 16 + M * N, 16);
                if (strcmp(mode, "off") == 0) {
                    fallback.assign(output.begin() + 16, output.end() - 16);
                    fallback_flags = flags;
                } else {
                    ok &= flags == fallback_flags &&
                        memcmp(output.data() + 16, fallback.data(), fallback.size() * sizeof(float)) == 0;
                }
            }
            ctx->mx_allocate = allocate;
            ok &= input == source_before && gamma == gamma_before;
            printf("AME_RMS_OOM_CASE real_CPU_fallback_controls=3 fused_calls=0 "
                "full_output_flags_exact=%s\n", ok ? "PASS" : "FAIL");
        }
        ggml_threadpool_free(pool);
    }
    ggml_aligned_free(work, work_size);
    ggml_backend_buffer_free(buffer);
    ggml_free(meta);
    for (size_t i = 0; i < saved.size(); ++i) {
        if (present[i]) setenv(names[i], saved[i].c_str(), 1); else unsetenv(names[i]);
    }
    GGML_ASSERT(std::fesetenv(&original_env) == 0);
    if (ok && screen_only) {
        printf("AME_RMS_FULL_ACCEPTANCE_DEFERRED fixture=10 "
            "post_sample_allocator_abort_oom=not_run original_native_fixture_retained=yes\n");
    }
    if (ok) printf("AME_RMS_FUSION_%s_PASS rejected=%u observer_rejects=%u full_graph_variants=%u "
        "whole_domain_before_issue=yes scalar_decline_transaction=pass fanout_and_slice=reject "
        "live_partial_alias=reject input_unchanged=yes full_output_bitexact=%s "
        "native_performance=not_inferred\n",
        run_native ? (screen_only ? "SCREEN" : "NATIVE") : "SOFTWARE", rejected, observer_rejects, native_variants,
        run_native ? "yes" : "not_run");
    return ok;
}

extern "C" bool ggml_ame_mx128_selftest(bool run_native, bool qwen_shape) {
    const char * names[] = {"GGML_AME_MX_CONFIG", "GGML_AME_MXFP4", "GGML_AME_MXFP8",
        "GGML_AME_MX32_VERIFIED", "GGML_AME_MXFP8_FORMAT", "GGML_AME_MXFP4_K64", "GGML_AME_SCALAR"};
    std::vector<std::string> saved;
    std::vector<bool> present;
    for (const char * name : names) {
        const char * value = getenv(name);
        saved.emplace_back(value == nullptr ? "" : value);
        present.push_back(value != nullptr);
        unsetenv(name);
    }
    setenv("GGML_AME_MX_CONFIG", "CUTE_8Tops_128SCP", 1);
    setenv("GGML_AME_MXFP4", "1", 1); setenv("GGML_AME_MXFP8", "1", 1);
    setenv("GGML_AME_MX32_VERIFIED", "1", 1);
    ggml_context * meta = ggml_init({262144, nullptr, true});
    bool ok = meta != nullptr;
    unsigned queries = 0, rejects = 0, injected_oom = 0, layouts = 0, plans = 0, cpu_checks = 0, refused_plans = 0;
    unsigned config_plan_checks = 0;
    // Independent format decoder, not the production FP8 encoder/packer.
    auto decode8 = [](uint8_t code, uint8_t scale, int format) {
        const int bits = format == AME_MCFG_MXFP8_E4M3 ? 3 : 2;
        const int bias = bits == 3 ? 7 : 15;
        const int exp = (code & 127) >> bits, mant = code & ((1 << bits) - 1);
        const double magnitude = std::ldexp(exp ? 1.0 + std::ldexp((double) mant, -bits) :
            std::ldexp((double) mant, -bits), (exp ? exp - bias : 1 - bias) + (int) scale - 127);
        return code & 128 ? -magnitude : magnitude;
    };
    if (meta != nullptr) {
        for (const auto & dims : std::vector<std::vector<int64_t>>{
                {32, 128, 128}, {64, 256, 128}, {96, 128, 256},
                {33, 128, 128}, {64, 64, 128}, {64, 128, 192}, {64, 127, 128},
                {64, 129, 128}, {64, 191, 128}, {64, 192, 128}, {64, 255, 128},
                {64, 257, 128}, {64, 288, 128}, {64, 383, 128}, {288, 288, 256}}) {
            ggml_tensor * w = ggml_new_tensor_2d(meta, GGML_TYPE_F32, dims[0], dims[1]);
            ggml_tensor * x = ggml_new_tensor_2d(meta, GGML_TYPE_F32, dims[0], dims[2]);
            ggml_tensor * op = ggml_mul_mat(meta, w, x);
            size_t size = 0;
            const bool admit = dims[0] % 32 == 0 && dims[1] >= 128 && dims[2] % 128 == 0;
            for (const char * config : {"CUTE_2Tops_128SCP", "CUTE_8Tops_128SCP"}) {
                setenv("GGML_AME_MX_CONFIG", config, 1);
                ok &= ame_mx_work_size_plan(1, op, size) == admit;
                ok &= !ame_mx_work_size_plan(2, op, size);
                refused_plans += !admit;
                config_plan_checks++;
            }
            for (const char * invalid : {"", "128", "CUTE_8Tops_256SCP", "unknown"}) {
                setenv("GGML_AME_MX_CONFIG", invalid, 1);
                ok &= !ame_mx_work_size_plan(1, op, size);
                refused_plans++;
            }
            setenv("GGML_AME_MX_CONFIG", "CUTE_8Tops_128SCP", 1);
        }
    }
    for (int format : {AME_MCFG_MXFP4, AME_MCFG_MXFP8_E4M3, AME_MCFG_MXFP8_E5M2}) {
        if (!ok || meta == nullptr) break;
        setenv("GGML_AME_MXFP8_FORMAT", format == AME_MCFG_MXFP8_E5M2 ? "e5m2" : "e4m3", 1);
        for (int tk : {32, 64}) {
            if (!ggml_ame_mx_tile_k_supported(format, tk)) continue;
            setenv("GGML_AME_MXFP4_K64", tk == 64 ? "1" : "0", 1);
          for (int M : {128, 129, 191, 192, 255, 256, 257, 288, 383}) {
            constexpr int N = 256, K = 128, nb = K / 32;
            ggml_tensor * w = ggml_new_tensor_2d(meta, format == AME_MCFG_MXFP4 ? GGML_TYPE_MXFP4 : GGML_TYPE_F32, K, M);
            ggml_tensor * x = ggml_new_tensor_2d(meta, GGML_TYPE_F32, K, N);
            ggml_tensor * op = ggml_mul_mat(meta, w, x);
            ggml_backend_buffer_t buffer = ggml_backend_ame_buffer_type_alloc_buffer(nullptr, ggml_nbytes(w) + 128);
            if (buffer == nullptr) { ok = false; break; }
            auto * source_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
            memset(source_base, 0xa5, ggml_nbytes(w) + 128);
            w->buffer = buffer; w->data = source_base + 64;
            ggml_backend_buffer_init_tensor(buffer, w);
            auto * ctx = ame_buffer_ctx(buffer);
            std::vector<block_mxfp4> blocks(M * nb);
            std::vector<float> f32(std::max(M, N) * K);
            for (int row = 0; row < std::max(M, N); ++row) for (int kb = 0; kb < nb; ++kb) {
                if (row < M) blocks[row * nb + kb].e = (uint8_t) ((row + kb * 37) % 255);
                for (int k = 0; k < 32; ++k) {
                    if (row < M) blocks[row * nb + kb].qs[k % 16] |= ((row * 3 + kb * 7 + k) & 15) << (k < 16 ? 0 : 4);
                    // Different row AND adjacent K32 envelopes make scale0/1
                    // swaps and cross-row scale reuse observable in the oracle.
                    f32[row * K + kb * 32 + k] = std::ldexp(
                        (row + kb + k) & 1 ? -1.0f : 1.0f,
                        -(k % 4) + (row % 5) * 3 + kb * 5 - 12);
                }
            }
            ggml_backend_tensor_set(w, format == AME_MCFG_MXFP4 ?
                static_cast<void *>(blocks.data()) : static_cast<void *>(f32.data()), 0, ggml_nbytes(w));
            auto layout = [&](const uint8_t * payload, const uint8_t * scales, bool activation) {
                const int rows = activation ? N : ((M + 127) / 128) * 128;
                for (int row = 0; row < rows; ++row) for (int k = 0; k < K; ++k) {
                    const size_t panel = (size_t) (row / 128 * (K / tk) + k / tk);
                    const size_t scale_idx = panel * 128 * (tk / 32) + (row % 128) * (tk / 32) + k % tk / 32;
                    const size_t row_bytes = (size_t) tk / (format == AME_MCFG_MXFP4 ? 2 : 1);
                    const size_t offset = panel * 128 * row_bytes + (row % 128) * row_bytes +
                        (size_t) (k % tk) / (format == AME_MCFG_MXFP4 ? 2 : 1);
                    if (!activation && row >= M) {
                        if (payload[offset] != 0 || scales[scale_idx] != 127) return false;
                    } else if (format == AME_MCFG_MXFP4) {
                        block_mxfp4 encoded;
                        const block_mxfp4 * source = static_cast<const block_mxfp4 *>(w->data) + row * nb + k / 32;
                        if (activation) {
                            quantize_row_mxfp4_ref(f32.data() + row * K + k / 32 * 32, &encoded, 32);
                            source = &encoded;
                        }
                        const uint8_t code = (source->qs[k % 16] >> (k % 32 < 16 ? 0 : 4)) & 15;
                        if (((payload[offset] >> (4 * (k & 1))) & 15) != code || scales[scale_idx] != source->e) return false;
                    } else {
                        const float source = activation ? f32[row * K + k] : static_cast<float *>(w->data)[row * K + k];
                        if (decode8(payload[offset], scales[scale_idx], format) != source) return false;
                    }
                }
                layouts++;
                return true;
            };
            auto query = [&](int mn, bool accepted, bool cached, unsigned allocations) {
                bool hit = false;
                const auto before = ctx->mxfp4_allocation_attempts;
                const auto * entry = ame_get_packed_mx_weight(w, &hit, format, tk, mn);
                queries++; rejects += !accepted;
                const auto & record = ctx->packed_mxfp4.at(w);
                bool valid = (entry != nullptr) == accepted && hit == cached && record.tile_mn == mn &&
                    record.tile_k == tk && ctx->mxfp4_allocation_attempts == before + allocations;
                if (!accepted) valid &= record.data == nullptr && record.size == 0;
                if (entry != nullptr) {
                    const size_t padded_M = size_t((M + mn - 1) / mn * mn);
                    valid &= entry->M == M && entry->padded_M == int64_t(padded_M) &&
                        entry->size == padded_M * nb * (ggml_ame_mx_block_bytes(format) + 1);
                }
                if (entry != nullptr && mn == 128) {
                    const auto * payload = static_cast<const uint8_t *>(entry->data);
                    const size_t padded_M = size_t((M + 127) / 128 * 128);
                    valid &= layout(payload, payload + padded_M * nb * ggml_ame_mx_block_bytes(format), false);
                }
                return valid;
            };
            ok &= query(128, true, false, 1); ok &= query(128, true, true, 0);
            ok &= query(64, true, false, 1); ok &= query(128, true, false, 1);
            if (format == AME_MCFG_MXFP4) static_cast<block_mxfp4 *>(w->data)[5].qs[2] ^= 0x81;
            else static_cast<float *>(w->data)[5] *= -1.0f;
            ok &= query(128, true, false, 1);
            const size_t invalid_offset = format == AME_MCFG_MXFP4 ?
                ggml_nbytes(w) - sizeof(block_mxfp4) : ggml_nbytes(w) - sizeof(float);
            if (format == AME_MCFG_MXFP4) static_cast<block_mxfp4 *>(w->data)[M * nb - 1].e = 255;
            else static_cast<float *>(w->data)[M * K - 1] = NAN;
            ok &= query(128, false, false, 0); ok &= query(128, false, false, 0);
            if (format == AME_MCFG_MXFP4) static_cast<block_mxfp4 *>(w->data)[M * nb - 1].e = 127;
            else static_cast<float *>(w->data)[M * K - 1] = -1.0f;
            ok &= query(128, false, false, 0);
            uint8_t repair = 127; float repair_f32 = -1.0f;
            ggml_backend_tensor_set(w, format == AME_MCFG_MXFP4 ? static_cast<void *>(&repair) :
                static_cast<void *>(&repair_f32), invalid_offset, format == AME_MCFG_MXFP4 ? 1 : sizeof(float));
            const auto generation = ctx->generation;
            ctx->mx_allocate = [](size_t) -> void * { return nullptr; };
            injected_oom++; ok &= query(128, false, false, 1);
            ctx->mx_allocate = ggml_aligned_malloc;
            ok &= query(128, true, false, 1) && ctx->generation == generation;
            ok &= query(128, true, true, 0);
            size_t planned = 0;
            ok &= ame_mx_work_size_plan(1, op, planned) && ame_backend_mx_tile_k(op, 1) == tk;
            const size_t payload_size = N * nb * ggml_ame_mx_block_bytes(format);
            const size_t native_size = payload_size + N * nb + 65536;
            ok &= planned == ame_align_up(std::max(native_size, (size_t) (N * nb * sizeof(block_q8_0))), 64) + 63;
            ok &= !ame_mx_work_size_plan(2, op, planned);
            std::vector<uint8_t> work(native_size + 128, 0xa5);
            ggml_ame_mx_prepare_activation(work.data() + 64, work.data() + 64 + payload_size,
                f32.data(), N, K, K * sizeof(float), format, tk, 128);
            ok &= layout(work.data() + 64, work.data() + 64 + payload_size, true);
            const uint8_t * activation_scales = work.data() + 64 + payload_size;
            ok &= activation_scales[0] != activation_scales[tk / 32]; // Next row.
            ok &= activation_scales[0] != activation_scales[tk == 64 ? 1 : 128]; // Next K32.
            for (size_t i = 0; i < 64; ++i) ok &= work[i] == 0xa5;
            for (size_t i = 64 + payload_size + N * nb; i < work.size(); ++i) ok &= work[i] == 0xa5;
            for (size_t i = 0; i < 64; ++i)
                ok &= source_base[i] == 0xa5 && source_base[64 + ggml_nbytes(w) + i] == 0xa5;
            plans++;
            ggml_backend_buffer_free(buffer);
          }
        }
    }
    if (ok && meta != nullptr) {
        setenv("GGML_AME_MXFP8_FORMAT", "e4m3", 1);
        ggml::cpu::riscv_ame::tensor_traits traits;
        auto pp = ggml_threadpool_params_default(1);
        ggml_threadpool * pool = ggml_threadpool_new(&pp);
        ggml_cplan plan = {}; plan.n_threads = 1; plan.threadpool = pool;
        ok &= ggml_graph_compute(ggml_new_graph_custom(meta, 1, false), &plan) == GGML_STATUS_SUCCESS;
        for (int m : {64, 128}) {
            const int n = 128, k = 128;
            ggml_tensor * w = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k, m);
            ggml_tensor * x = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k, n);
            ggml_tensor * dst = ggml_mul_mat(meta, w, x);
            std::vector<float> a(m * k, 0.25f), b(n * k, 0.25f), output(m * n), control(m * n);
            w->data = a.data(); x->data = b.data(); dst->data = output.data();
            std::vector<uint8_t> workspace(ame_mx_worker_wsize(dst) + 63);
            ggml_compute_params params = {}; params.nth = 1; params.threadpool = pool;
            params.wdata = workspace.data(); params.wsize = workspace.size();
            for (int invalid = 0; invalid < 3; ++invalid) {
                a[0] = invalid == 0 ? NAN : 0.25f;
                b[0] = invalid == 1 ? INFINITY : 0.25f;
                if (invalid < 2 || m == 64) {
                    ok &= !traits.compute_forward(&params, dst); // Shape/domain before CSR.
                } else {
                    size_t admitted = 0;
                    ok &= ame_mx_work_size_plan(1, dst, admitted); // No capability fake/probe.
                }
                setenv("GGML_AME_MXFP8", "0", 1);
                unsetenv("GGML_AME_MX_CONFIG");
                ggml_compute_forward_mul_mat(&params, dst);
                control = output;
                // Same canonical CPU operator, same input under exact128 config.
                setenv("GGML_AME_MX_CONFIG", "CUTE_8Tops_128SCP", 1);
                ggml_compute_forward_mul_mat(&params, dst);
                ok &= memcmp(control.data(), output.data(), output.size() * sizeof(float)) == 0;
                for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
                    const float value = output[j * m + i];
                    if (invalid == 0 && i == 0) ok &= std::isnan(value);
                    else if (invalid == 1 && j == 0) ok &= std::isinf(value);
                    else ok &= value == 8.0f;
                }
                cpu_checks++; setenv("GGML_AME_MXFP8", "1", 1);
            }
        }
        ggml_threadpool_free(pool);
    }
    unsigned native_calls = 0, native_outputs = 0;
    if (ok && meta != nullptr && run_native) {
        // Explicit opt-in only. The old software-only mode never probes CSRs
        // or submits AMU work. No capability or allocator result is fabricated.
        setenv("GGML_AME_MX_CONFIG", "CUTE_2Tops_128SCP", 1);
        ok &= std::fegetround() == FE_TONEAREST;
        ggml::cpu::riscv_ame::tensor_traits traits;
        auto pp = ggml_threadpool_params_default(1);
        ggml_threadpool * pool = ggml_threadpool_new(&pp);
        ggml_cplan plan = {}; plan.n_threads = 1; plan.threadpool = pool;
        ok &= ggml_graph_compute(ggml_new_graph_custom(meta, 1, false), &plan) == GGML_STATUS_SUCCESS;
        std::vector<std::vector<int>> cases;
        for (int format : {AME_MCFG_MXFP4, AME_MCFG_MXFP8_E4M3, AME_MCFG_MXFP8_E5M2}) {
            if (qwen_shape) {
                cases.push_back({format, format == AME_MCFG_MXFP4 ? 64 : 32, 1024, 3072, 256});
                continue;
            }
            for (int k : {32, 64, 128}) {
                cases.push_back({format, 32, k, 128, 128});
                if (format == AME_MCFG_MXFP4 && k >= 64) cases.push_back({format, 64, k, 128, 128});
            }
            for (int m : {129, 191, 192, 255, 257, 288, 383})
                cases.push_back({format, format == AME_MCFG_MXFP4 ? 64 : 32, 64, m, 128});
        }
        if (!qwen_shape) {
            cases.push_back({AME_MCFG_MXFP4, 64, 128, 256, 256});
            cases.push_back({AME_MCFG_MXFP4, 32, 288, 288, 256});
        }
        constexpr double fp4_magnitude[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        auto decode4 = [&](unsigned code, unsigned scale) {
            const double value = std::ldexp(fp4_magnitude[code & 7], int(scale) - 127);
            return code & 8 ? -value : value;
        };
        for (const auto & item : cases) {
            if (!ok) break;
            const int format = item[0], tk = item[1], k = item[2], m = item[3], n = item[4], nb = k / 32;
            setenv("GGML_AME_MXFP4_K64", tk == 64 ? "1" : "0", 1);
            setenv("GGML_AME_MXFP8_FORMAT", format == AME_MCFG_MXFP8_E5M2 ? "e5m2" : "e4m3", 1);
            ggml_tensor * w = ggml_new_tensor_2d(meta, format == AME_MCFG_MXFP4 ? GGML_TYPE_MXFP4 : GGML_TYPE_F32, k, m);
            ggml_tensor * x = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k, n);
            ggml_tensor * dst = ggml_mul_mat(meta, w, x);
            auto buffer = ggml_backend_ame_buffer_type_alloc_buffer(
                ggml_backend_cpu_riscv_ame_buffer_type(), ggml_nbytes(w));
            if (buffer == nullptr) { ok = false; break; }
            w->buffer = buffer; w->data = ggml_backend_buffer_get_base(buffer);
            ggml_backend_buffer_init_tensor(buffer, w);
            std::vector<block_mxfp4> blocks(m * nb), quantized_b(n * nb);
            std::vector<float> weights(m * k), activations(n * k), output(m * n + 32), expected(m * n);
            for (int row = 0; row < m; ++row) for (int kb = 0; kb < nb; ++kb) {
                auto & block = blocks[row * nb + kb];
                block.e = uint8_t((kb & 1 ? 110 : 127) + row % 5 - 2);
                for (int lane = 0; lane < 32; ++lane) {
                    const unsigned code = (row * 3 + kb * 5 + lane) & 15;
                    block.qs[lane % 16] |= code << (lane < 16 ? 0 : 4);
                    weights[row * k + kb * 32 + lane] = std::ldexp(
                        float((row * 3 + kb + lane) % 15 - 7) * 0.5f,
                        row % 3 + (qwen_shape ? kb % 4 : kb) - 2);
                }
            }
            for (int row = 0; row < n; ++row) {
                for (int lane = 0; lane < k; ++lane)
                    activations[row * k + lane] = std::ldexp(
                        float((row * 5 + lane) % 15 - 7) * 0.5f,
                        row % 3 + (qwen_shape ? (lane / 32) % 4 : lane / 32) - 2);
                quantize_row_mxfp4_ref(activations.data() + row * k, quantized_b.data() + row * nb, k);
            }
            ggml_backend_tensor_set(w, format == AME_MCFG_MXFP4 ?
                static_cast<void *>(blocks.data()) : static_cast<void *>(weights.data()), 0, ggml_nbytes(w));
            x->data = activations.data(); dst->data = output.data() + 16;
            constexpr uint32_t poison = 0x7fc56789;
            for (auto & value : output) memcpy(&value, &poison, sizeof(value));
            size_t work_size = 0;
            const bool admitted = traits.work_size(1, dst, work_size);
            if (!admitted) {
                fprintf(stderr, "AME_MX128_NATIVE_REJECT format=0x%x M=%d N=%d K=%d tile_k=%d\n", format, m, n, k, tk);
                ok = false;
            }
            void * work = admitted ? ggml_aligned_malloc(work_size) : nullptr;
            ok &= work != nullptr;
            if (ok) {
                ggml_compute_params params = {}; params.nth = 1; params.threadpool = pool;
                params.wdata = work; params.wsize = work_size;
                // This is the production dispatch, including preparation,
                // real capacity/allocator gates, native waits and stores.
                ok &= traits.compute_forward(&params, dst);
                if (ok) ++native_calls;
                bool hit = false;
                const auto * packed = ame_get_packed_mx_weight(w, &hit, format, tk, 128);
                ok &= packed != nullptr && hit;
                if (ok) {
                    const size_t b_size = size_t(n) * nb * ggml_ame_mx_block_bytes(format);
                    std::vector<uint8_t> prepared_b(b_size + n * nb);
                    ggml_ame_mx_prepare_activation(prepared_b.data(), prepared_b.data() + b_size,
                        activations.data(), n, k, k * sizeof(float), format, tk, 128);
                    auto read = [&](const uint8_t * data, int rows, int row, int lane) {
                        // Independent physical panel indexing, not the packer's
                        // packed_block_index helper or the kernel's C layout.
                        const size_t panel_row = size_t(row / 128 * (k / tk) + lane / tk) * 128 + row % 128;
                        const size_t row_bytes = tk / (format == AME_MCFG_MXFP4 ? 2 : 1);
                        unsigned code = data[panel_row * row_bytes + (lane % tk) /
                            (format == AME_MCFG_MXFP4 ? 2 : 1)];
                        if (format == AME_MCFG_MXFP4) code = (code >> (4 * (lane & 1))) & 15;
                        const auto * scales = data + size_t(rows) * nb * ggml_ame_mx_block_bytes(format);
                        const unsigned scale = scales[panel_row * (tk / 32) + lane % tk / 32];
                        return format == AME_MCFG_MXFP4 ? decode4(code, scale) : decode8(code, scale, format);
                    };
                    std::vector<double> a(m * k), b(n * k);
                    for (int row = 0; row < m; ++row) for (int lane = 0; lane < k; ++lane) {
                        a[row * k + lane] = read(static_cast<const uint8_t *>(packed->data),
                            (m + 127) / 128 * 128, row, lane);
                        const auto & block = blocks[row * nb + lane / 32];
                        const unsigned code = (block.qs[lane % 16] >> (lane % 32 < 16 ? 0 : 4)) & 15;
                        ok &= a[row * k + lane] == (format == AME_MCFG_MXFP4 ?
                            decode4(code, block.e) : double(weights[row * k + lane]));
                    }
                    for (int row = 0; row < n; ++row) for (int lane = 0; lane < k; ++lane) {
                        b[row * k + lane] = read(prepared_b.data(), n, row, lane);
                        const auto & block = quantized_b[row * nb + lane / 32];
                        const unsigned code = (block.qs[lane % 16] >> (lane % 32 < 16 ? 0 : 4)) & 15;
                        ok &= b[row * k + lane] == (format == AME_MCFG_MXFP4 ?
                            decode4(code, block.e) : double(activations[row * k + lane]));
                    }
                    // Bounded finite inputs make each 32-product sum exact in
                    // double. Explicit volatile FP32 stores round after EACH
                    // K32 block; never use a whole-K float/GGML dot as oracle.
                    for (int col = 0; col < n; ++col) for (int row = 0; row < m; ++row) {
                        float acc = 0.0f;
                        for (int kb = 0; kb < nb; ++kb) {
                            double sum = 0;
                            for (int lane = 0; lane < 32; ++lane)
                                sum += a[row * k + kb * 32 + lane] * b[col * k + kb * 32 + lane];
                            volatile float rounded = float(double(acc) + sum);
                            acc = rounded;
                        }
                        expected[col * m + row] = acc;
                    }
                    const bool equal = memcmp(dst->data, expected.data(), expected.size() * sizeof(float)) == 0;
                    if (!equal) {
                        for (size_t i = 0; i < expected.size(); ++i) {
                            uint32_t want, got;
                            memcpy(&want, &expected[i], 4); memcpy(&got, static_cast<float *>(dst->data) + i, 4);
                            if (want != got) {
                                fprintf(stderr, "AME_MX128_NATIVE_MISMATCH format=0x%x M=%d N=%d K=%d tile_k=%d "
                                    "index=%zu expected=%08x actual=%08x\n", format, m, n, k, tk, i, want, got);
                                break;
                            }
                        }
                    }
                    ok &= equal;
                    for (int i = 0; i < 16; ++i) {
                        uint32_t before, after;
                        memcpy(&before, &output[i], 4); memcpy(&after, &output[m * n + 16 + i], 4);
                        ok &= before == poison && after == poison;
                    }
                    if (ok) {
                        native_outputs += m * n;
                        printf("AME_MX128_NATIVE_CASE format=0x%x M=%d N=%d K=%d tile_k=%d outputs=%d exact=PASS\n",
                            format, m, n, k, tk, m * n);
                    }
                }
            }
            if (work != nullptr) ggml_aligned_free(work, work_size);
            ggml_backend_buffer_free(buffer);
        }
        ggml_threadpool_free(pool);
        ok &= native_calls == (qwen_shape ? 3U : 34U) &&
            native_outputs == (qwen_shape ? 2359296U : 970368U);
    }
    if (meta != nullptr) ggml_free(meta);
    for (size_t i = 0; i < saved.size(); ++i) {
        if (present[i]) setenv(names[i], saved[i].c_str(), 1); else unsetenv(names[i]);
    }
    ok &= queries == 396 && rejects == 144 && injected_oom == 36 && plans == 36 && cpu_checks == 6 &&
        refused_plans == 68 && config_plan_checks == 30;
    if (ok && run_native) printf("%s native_dispatch_calls=%u exact_outputs=%u "
        "config=CUTE_2Tops_128SCP ordered_K32_RNE=pass fallback=not-used performance=not-measured\n",
        qwen_shape ? "AME_MX128_QWEN_NATIVE_PASS" : "AME_MX128_NATIVE_PASS",
        native_calls, native_outputs);
    if (ok && !run_native) printf("AME_MX128_SOFTWARE_PASS queries=%u rejected=%u injected_oom=%u "
        "independent_layouts=%u software_plans=%u CPU_full_output_checks=%u refused_plans=%u tile_mn_identity=pass "
        "config_plan_checks=%u exact_configs=2Tops128,8Tops128 "
        "C_workspace_bytes=65536 MX4_K64_scale_bytes=256 M_tail_shapes=9 padding=zero_scale127 "
        "activation_scale_order=distinct_exact same_identity_oom_retry=pass "
        "actual_native_issues=0 CSR_AMU=not-issued native_numeric=SKIP\n",
        queries, rejects, injected_oom, layouts, plans, cpu_checks, refused_plans, config_plan_checks);
    return ok;
}

extern "C" bool ggml_ame_mxfp4_k64_selftest(void) {
    ggml_init_params init = {16384, nullptr, true};
    ggml_context * meta = ggml_init(init);
    if (meta == nullptr) return false;
    ggml_tensor * weight = ggml_new_tensor_2d(meta, GGML_TYPE_MXFP4, 128, 128);
    ggml_tensor * activation = ggml_new_tensor_2d(meta, GGML_TYPE_F32, 128, 64);
    ggml_tensor * op = ggml_mul_mat(meta, weight, activation);
    const size_t size = ggml_nbytes(weight);
    ggml_backend_buffer_t buffer = ggml_backend_ame_buffer_type_alloc_buffer(nullptr, size);
    if (buffer == nullptr) {
        ggml_free(meta);
        return false;
    }
    weight->buffer = buffer;
    weight->data = ggml_backend_buffer_get_base(buffer);
    ggml_backend_buffer_init_tensor(buffer, weight);
    auto * ctx = ame_buffer_ctx(buffer);
    std::vector<block_mxfp4> blocks(size / sizeof(block_mxfp4));
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i].e = (uint8_t) (i % 255);
        for (int k = 0; k < 32; ++k) {
            blocks[i].qs[k % 16] |= ((i * 7 + k) & 15) << (k < 16 ? 0 : 4);
        }
    }
    ggml_backend_tensor_set(weight, blocks.data(), 0, size);
    unsigned queries = 0, rejected = 0;
    auto expect = [&](int tile_k, bool accepted, bool cached, uint64_t attempts) {
        const uint64_t before = ctx->mxfp4_allocation_attempts;
        bool hit = false;
        const auto * result = ame_get_packed_mx_weight(weight, &hit, AME_MCFG_MXFP4, tile_k);
        const auto & entry = ctx->packed_mxfp4.at(weight);
        queries++;
        rejected += !accepted;
        return (result != nullptr) == accepted && hit == cached &&
            entry.tile_k == tile_k && entry.format == AME_MCFG_MXFP4 &&
            entry.generation == ctx->generation && entry.source == weight->data &&
            entry.source_type == GGML_TYPE_MXFP4 && entry.source_row_stride == weight->nb[1] &&
            (accepted || (entry.data == nullptr && entry.size == 0)) &&
            ctx->mxfp4_allocation_attempts == before + attempts;
    };
    auto layout = [&]() {
        const auto * payload = static_cast<const uint8_t *>(ctx->packed_mxfp4.at(weight).data);
        const auto * scales = payload + blocks.size() * 16;
        const auto * source = static_cast<const block_mxfp4 *>(weight->data);
        bool valid = true;
        // Independent direct byte-index oracle (not the production index or
        // packer); verifies actual getter's panels, every nibble and both scales.
        for (size_t row = 0; row < 128; ++row) {
            for (size_t k = 0; k < 128; ++k) {
                const size_t panel = (row / 64) * 2 + k / 64;
                const size_t offset = panel * 2048 + row % 64 * 32 + k % 64 / 2;
                const size_t scale_index = panel * 128 + row % 64 * 2 + k % 64 / 32;
                const auto & block = source[row * 4 + k / 32];
                const uint8_t expected = (block.qs[k % 16] >> (k % 32 < 16 ? 0 : 4)) & 15;
                valid &= ((payload[offset] >> (4 * (k & 1))) & 15) == expected;
                valid &= scales[scale_index] == block.e;
            }
        }
        return valid;
    };
    bool ok = expect(32, true, false, 1);
    ok &= expect(32, true, true, 0);
    ok &= expect(64, true, false, 1) && layout();
    ok &= std::memcmp(weight->data, blocks.data(), size) == 0;
    ok &= expect(64, true, true, 0);
    ok &= expect(32, true, false, 1);
    ok &= expect(64, true, false, 1) && layout();
    auto * raw = static_cast<block_mxfp4 *>(weight->data);
    raw[5].qs[3] ^= 0x81;
    blocks[5].qs[3] ^= 0x81;
    ok &= expect(64, true, false, 1) && layout();
    raw[5].e = 254; // Raw scale change within legal domain must also rebuild.
    blocks[5].e = 254;
    ok &= expect(64, true, false, 1) && layout();
    raw[blocks.size() - 1].e = 255;
    ok &= expect(64, false, false, 0);
    ok &= expect(64, false, false, 0);
    raw[blocks.size() - 1].e = 127;
    ok &= expect(64, false, false, 0);
    const size_t last = (blocks.size() - 1) * sizeof(block_mxfp4);
    ggml_backend_tensor_memset(weight, 127, last, 1);
    ok &= expect(64, true, false, 1) && layout();
    // Real allocation failure then recovery under EXACT same cache identity.
    ggml_backend_tensor_memset(weight, 126, last, 1);
    blocks.back().e = 126;
    const uint64_t generation = ctx->generation;
    ctx->mx_allocate = [](size_t) -> void * { return nullptr; };
    ok &= expect(64, false, false, 1);
    ok &= !ctx->packed_mxfp4.at(weight).domain_rejected;
    ctx->mx_allocate = ggml_aligned_malloc;
    ok &= expect(64, true, false, 1) && layout();
    ok &= ctx->generation == generation;
    ok &= expect(64, true, true, 0);
    std::vector<uint8_t> current(size);
    ggml_backend_tensor_get(weight, current.data(), 0, size);
    ok &= std::memcmp(current.data(), blocks.data(), size) == 0;

    const char * env_names[] = {"GGML_AME_MXFP4", "GGML_AME_MX32_VERIFIED", "GGML_AME_MXFP4_K64", "GGML_AME_SCALAR"};
    std::vector<std::string> saved;
    std::vector<bool> present;
    for (const char * name : env_names) {
        const char * value = getenv(name);
        saved.emplace_back(value == nullptr ? "" : value);
        present.push_back(value != nullptr);
        unsetenv(name);
    }
    unsigned select32 = 0, select64 = 0, refuse = 0;
    auto selection = [&](ggml_tensor * graph_op, int threads, int expected) {
        size_t workspace = 0;
        const int actual = ame_backend_mx_tile_k(graph_op, threads);
        const bool plan = ame_mx_work_size_plan(threads, graph_op, workspace);
        select32 += actual == 32;
        select64 += actual == 64;
        refuse += actual == 0;
        return actual == expected && plan == (expected != 0);
    };
    ok &= selection(op, 1, 0);
    setenv("GGML_AME_MXFP4_K64", "1", 1);
    ok &= selection(op, 1, 0);
    setenv("GGML_AME_MXFP4", "1", 1);
    ok &= selection(op, 1, 0); // No VERIFIED: neither K32 nor K64 allowed.
    setenv("GGML_AME_MX32_VERIFIED", "1", 1);
    ok &= selection(op, 1, 64);
    ok &= selection(op, 2, 0);
    unsetenv("GGML_AME_MXFP4_K64");
    ok &= selection(op, 1, 32); // Default must stay K32.
    for (const char * knob : {"0", "off", "true", "64", "unknown", "1"}) {
        setenv("GGML_AME_MXFP4_K64", knob, 1);
        ok &= selection(op, 1, strcmp(knob, "1") == 0 ? 64 : 32);
    }
    uint64_t planned_mma = 0, planned_blocks = 0;
    for (const auto & dims : std::vector<std::vector<int64_t>>{
            {64, 64, 64}, {128, 128, 128}, {32, 64, 64}, {96, 64, 64},
            {64, 63, 64}, {64, 64, 65}, {33, 64, 64}}) {
        const int64_t k = dims[0], m = dims[1], n = dims[2];
        ggml_tensor * a = ggml_new_tensor_2d(meta, GGML_TYPE_MXFP4, k == 33 ? 64 : k, m);
        ggml_tensor * b = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k == 33 ? 64 : k, n);
        ggml_tensor * dst = ggml_mul_mat(meta, a, b);
        // A GGML MXFP4 tensor cannot represent K33; use F32 K33 to prove the
        // shared shape selector rejects it without constructing invalid GGML.
        if (k == 33) {
            a = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k, m);
            b = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k, n);
            dst = ggml_mul_mat(meta, a, b);
        }
        const int expected = m % 64 || n % 64 || k % 32 ? 0 : k % 64 ? 32 : 64;
        ok &= selection(dst, 1, expected);
        if (expected != 0) {
            const uint64_t outputs = (uint64_t) (m / 64) * (uint64_t) (n / 64);
            planned_mma += outputs * (uint64_t) (k / expected);
            planned_blocks += outputs * (uint64_t) (k / 32);
        }
    }
    setenv("GGML_AME_SCALAR", "1", 1);
    ok &= selection(op, 1, 0);
    for (size_t i = 0; i < saved.size(); ++i) {
        if (present[i]) setenv(env_names[i], saved[i].c_str(), 1);
        else unsetenv(env_names[i]);
    }
    ok &= queries == 15 && rejected == 4;
    ok &= select32 == 8 && select64 == 4 && refuse == 8;
    ok &= planned_mma == 13 && planned_blocks == 22;
    ggml_backend_buffer_free(buffer);
    ggml_free(meta);
    if (ok) {
        printf("AME_MXFP4_K64_CACHE_PASS queries=%u rejected=%u injected_oom=1 "
               "layout_retry=pass source_mutation=pass two_scales=exact same_identity_oom_retry=pass "
               "selected_K32=%u selected_K64=%u refused=%u planned_mmacc=%llu planned_K32_blocks=%llu "
               "actual_native_issues=0 CSR_AMU=not-issued numeric=not-run\n",
               queries, rejected, select32, select64, refuse,
               (unsigned long long) planned_mma, (unsigned long long) planned_blocks);
    }
    return ok;
}

extern "C" bool ggml_ame_mx_invalid_compute_selftest(void) {
    ggml_init_params init = {16384, nullptr, true};
    ggml_context * meta = ggml_init(init);
    if (meta == nullptr) return false;
    ggml_tensor * weight = ggml_new_tensor_2d(meta, GGML_TYPE_F32, 64, 64);
    ggml_tensor * activation = ggml_new_tensor_2d(meta, GGML_TYPE_F32, 64, 64);
    ggml_tensor * op = ggml_mul_mat(meta, weight, activation);
    const size_t size = ggml_nbytes(weight);
    ggml_backend_buffer_t buffer = ggml_backend_ame_buffer_type_alloc_buffer(nullptr, size);
    if (buffer == nullptr) {
        ggml_free(meta);
        return false;
    }
    weight->buffer = buffer;
    weight->data = ggml_backend_buffer_get_base(buffer);
    ggml_backend_buffer_init_tensor(buffer, weight);
    auto * ctx = ame_buffer_ctx(buffer);
    std::vector<float> input(size / sizeof(float), 0.25f);
    std::vector<float> output(input.size(), -4567.0f);
    ggml_backend_tensor_set(weight, input.data(), 0, size);
    auto * weight_data = static_cast<float *>(weight->data);
    for (size_t row = 0; row < 64; ++row) weight_data[row * 64] += 1.0f / 1024;
    activation->data = input.data();
    op->data = output.data();
    // Reuse the real CPU operator and its real T1 threadpool. An empty graph
    // initializes threadpool barrier state without invoking extra-buffer
    // discovery/CSR probes; no CPU kernel or scheduler is reimplemented.
    auto pool_params = ggml_threadpool_params_default(1);
    ggml_threadpool * pool = ggml_threadpool_new(&pool_params);
    ggml_cplan cpu_plan = {};
    cpu_plan.n_threads = 1;
    cpu_plan.threadpool = pool;
    ggml_cgraph * empty = ggml_new_graph_custom(meta, 1, false);
    bool ok = ggml_graph_compute(empty, &cpu_plan) == GGML_STATUS_SUCCESS;

    const char * env_names[] = {"GGML_AME_MXFP8", "GGML_AME_MX32_VERIFIED", "GGML_AME_MXFP8_FORMAT", "GGML_AME_SCALAR"};
    std::vector<std::string> saved;
    std::vector<bool> present;
    for (const char * name : env_names) {
        const char * value = getenv(name);
        saved.emplace_back(value == nullptr ? "" : value);
        present.push_back(value != nullptr);
    }
    setenv("GGML_AME_MXFP8", "1", 1);
    setenv("GGML_AME_MX32_VERIFIED", "1", 1);
    unsetenv("GGML_AME_SCALAR");
    ggml::cpu::riscv_ame::tensor_traits traits;
    unsigned queries = 0;
    unsigned cpu_checks = 0;
    unsigned shape_checks = 0;
    uint64_t finite_values = 0, nonfinite_values = 0;
    for (const char * format : {"e4m3", "e5m2"}) {
        setenv("GGML_AME_MXFP8_FORMAT", format, 1);
        // Generic workspace allocation is allowed. Give compute sufficient
        // workspace so neither a missing buffer nor a small size can mask the
        // actual whole-tensor domain preflight being exercised.
        std::vector<uint8_t> workspace(ame_mx_worker_wsize(op) + 63, 0xa5);
        const auto untouched_workspace = workspace;
        ggml_compute_params params = {};
        params.nth = 1;
        params.wdata = workspace.data();
        params.wsize = workspace.size();
        params.threadpool = pool;
        ok &= ame_backend_mx_metadata_supported(op);
        size_t planned = 0;
        ok &= ame_mx_work_size_plan(1, op, planned) && planned == workspace.size();
        for (float * source : {weight_data, input.data()}) {
            for (size_t bad : {size_t(0), input.size() / 2, input.size() - 1}) {
                for (float invalid : {NAN, INFINITY, -INFINITY}) {
                    const float previous = source[bad];
                    source[bad] = invalid;
                    for (int repeat = 0; repeat < 2; ++repeat) {
                        ok &= !traits.compute_forward(&params, op);
                        ok &= ctx->mxfp4_allocation_attempts == 0 && ctx->packed_mxfp4.empty();
                        ok &= workspace == untouched_workspace;
                        for (float value : output) ok &= value == -4567.0f;
                        queries++;
                    }
                    // Invoke the canonical generic F32 operator after fallback.
                    // Finite rows retain the 2^-12 contribution that FP8 would
                    // discard; affected rows/columns keep original NaN/Inf.
                    // Direct generic operator never consults specialization;
                    // additionally disable dispatch around it as an explicit
                    // regression guard, restoring both-opt-in for next query.
                    setenv("GGML_AME_MXFP8", "0", 1);
                    ggml_compute_forward_mul_mat(&params, op);
                    setenv("GGML_AME_MXFP8", "1", 1);
                    for (size_t n = 0; n < 64; ++n) {
                        for (size_t m = 0; m < 64; ++m) {
                            const bool affected = source == weight_data ? m == bad / 64 : n == bad / 64;
                            const float value = output[n * 64 + m];
                            finite_values += std::isfinite(value);
                            nonfinite_values += !std::isfinite(value);
                            if (!affected) ok &= value == 4.0f + 1.0f / 4096;
                            else if (std::isnan(invalid)) ok &= std::isnan(value);
                            else ok &= std::isinf(value) && std::signbit(value) == std::signbit(invalid);
                        }
                    }
                    cpu_checks++;
                    source[bad] = previous;
                    std::fill(output.begin(), output.end(), -4567.0f);
                    std::fill(workspace.begin(), workspace.end(), 0xa5);
                }
            }
        }

        // Build consistent graphs, not mutated mismatching dimensions. Every
        // stride/nb is produced by GGML and both K operands exactly agree.
        for (const auto & dims : std::vector<std::vector<int64_t>>{{33, 64, 64}, {64, 64, 65}, {64, 63, 64}}) {
            const int64_t k = dims[0], m = dims[1], n = dims[2];
            ggml_tensor * a = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k, m);
            ggml_tensor * b = ggml_new_tensor_2d(meta, GGML_TYPE_F32, k, n);
            ggml_tensor * dst = ggml_mul_mat(meta, a, b);
            std::vector<float> a_data(k * m, 0.25f), b_data(k * n, 0.25f), result(m * n, -4567.0f);
            for (int64_t row = 0; row < m; ++row) a_data[row * k] += 1.0f / 1024;
            a->data = a_data.data();
            b->data = b_data.data();
            dst->data = result.data();
            ok &= ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst);
            ok &= a->ne[0] == b->ne[0] && dst->ne[0] == a->ne[1] && dst->ne[1] == b->ne[1];
            ok &= !ame_backend_mx_metadata_supported(dst) && !ame_mx_work_size_plan(1, dst, planned);
            ok &= !traits.compute_forward(&params, dst);
            setenv("GGML_AME_MXFP8", "0", 1);
            ggml_compute_forward_mul_mat(&params, dst);
            setenv("GGML_AME_MXFP8", "1", 1);
            for (float value : result) ok &= value == (float) k / 16 + 1.0f / 4096;
            ok &= ctx->mxfp4_allocation_attempts == 0 && ctx->packed_mxfp4.empty();
            shape_checks++;
        }
    }
    ok &= queries == 72 && cpu_checks == 36 && shape_checks == 6 &&
        finite_values == 145152 && nonfinite_values == 2304;
    for (float value : output) ok &= value == -4567.0f;
    for (size_t i = 0; i < saved.size(); ++i) {
        if (present[i]) setenv(env_names[i], saved[i].c_str(), 1);
        else unsetenv(env_names[i]);
    }
    ggml_backend_buffer_free(buffer);
    ggml_threadpool_free(pool);
    ggml_free(meta);
    if (ok) {
        printf("AME_MX_INVALID_COMPUTE_PASS queries=%u dual_optin=1 shapes=64x64x64 T=1 "
               "activation_nonfinite=fallback weight_nonfinite=fallback packed_allocations=0 "
               "preflight_workspace_output=untouched cache=empty CPU_F32_checks=%u "
               "consistent_shape_checks=%u finite_values=%llu nonfinite_values=%llu "
               "native_metadata_plan=admit CSR_AMU=not-issued\n",
               queries, cpu_checks, shape_checks,
               (unsigned long long) finite_values, (unsigned long long) nonfinite_values);
    }
    return ok;
}

static bool ame_mxfp8_cache_selftest(ggml_context * meta) {
    ggml_tensor * weight = ggml_new_tensor_2d(meta, GGML_TYPE_F32, 64, 64);
    const size_t size = ggml_nbytes(weight);
    ggml_backend_buffer_t buffer = ggml_backend_ame_buffer_type_alloc_buffer(nullptr, size);
    if (buffer == nullptr) {
        return false;
    }
    weight->buffer = buffer;
    weight->data = ggml_backend_buffer_get_base(buffer);
    ggml_backend_buffer_init_tensor(buffer, weight);
    auto * ctx = ame_buffer_ctx(buffer);
    std::vector<float> input(size / sizeof(float));
    for (size_t k = 0; k < input.size(); ++k) {
        input[k] = ((int) (k % 29) - 14) * 0.25f;
    }
    unsigned queries = 0, rejected = 0, failed_allocations = 0;
    auto expect = [&](int format, bool accepted, bool cached, uint64_t attempts) {
        const uint64_t before = ctx->mxfp4_allocation_attempts;
        bool hit = false;
        const auto * packed = ame_get_packed_mx_weight(weight, &hit, format);
        const auto & entry = ctx->packed_mxfp4.at(weight);
        queries++;
        rejected += !accepted;
        return (packed != nullptr) == accepted && hit == cached &&
            entry.format == format && entry.generation == ctx->generation &&
            entry.source == weight->data && entry.M == weight->ne[1] && entry.K == weight->ne[0] &&
            (accepted || (entry.data == nullptr && entry.size == 0)) &&
            ctx->mxfp4_allocation_attempts == before + attempts;
    };
    bool ok = true;
    for (int format : {AME_MCFG_MXFP8_E4M3, AME_MCFG_MXFP8_E5M2}) {
        ggml_backend_tensor_set(weight, input.data(), 0, size);
        auto * raw = static_cast<float *>(weight->data);
        raw[input.size() - 1] = NAN; // Last value rejects the whole tensor, not just one tile.
        ok &= expect(format, false, false, 0);
        ok &= expect(format, false, false, 0);
        raw[input.size() - 1] = 0.0f;
        ok &= expect(format, false, false, 0); // Sticky for current generation/format.
        ggml_backend_tensor_memset(weight, 0, size - sizeof(float), sizeof(float));
        ok &= expect(format, true, false, 1);
        ok &= expect(format, true, true, 0);
        // Check the exact [output_tile][K_block][row][K] and scale wiring.
        const auto & packed = ctx->packed_mxfp4.at(weight);
        const auto * payload = static_cast<const uint8_t *>(packed.data);
        const auto * scales = payload + 64 * 2 * 32;
        for (size_t row = 0; row < 64; ++row) {
            for (size_t kb = 0; kb < 2; ++kb) {
                uint8_t expected[32], scale;
                ok &= ggml_ame_pack_mxfp8_f32(expected, &scale, raw + row * 64 + kb * 32, 32, format);
                ok &= std::memcmp(payload + (kb * 64 + row) * 32, expected, 32) == 0;
                ok &= scales[kb * 64 + row] == scale;
            }
        }
        ok &= ggml_ame_mxfp8_native_domain(payload, scales, 128, format);
        raw[0] = 32.0f;
        ok &= expect(format, true, false, 1); // Direct CPU write invalidates positive hit.
        ok &= expect(format, true, true, 0);
        raw[input.size() - 1] = INFINITY;
        ok &= expect(format, false, false, 0);
        ok &= expect(format, false, false, 0);
        ggml_backend_tensor_memset(weight, 0, size - sizeof(float), sizeof(float));
        ok &= expect(format, true, false, 1);

        ctx->mx_allocate = [](size_t) -> void * { return nullptr; };
        ggml_backend_tensor_memset(weight, 0, 0, sizeof(float));
        const uint64_t oom_generation = ctx->generation;
        const void * oom_source = weight->data;
        ok &= expect(format, false, false, 1); // Real getter, deterministic OOM injection.
        failed_allocations++;
        ok &= !ctx->packed_mxfp4.at(weight).domain_rejected;
        ctx->mx_allocate = ggml_aligned_malloc;
        // No setter, generation increment, source/shape/format change or cache
        // reset is allowed between OOM and this successful retry.
        ok &= expect(format, true, false, 1);
        ok &= ctx->generation == oom_generation && weight->data == oom_source;
        uint8_t repaired[32], repaired_scale;
        ok &= ggml_ame_pack_mxfp8_f32(repaired, &repaired_scale, raw, 32, format);
        const auto * repaired_payload = static_cast<const uint8_t *>(ctx->packed_mxfp4.at(weight).data);
        ok &= repaired_payload != nullptr && std::memcmp(repaired_payload, repaired, 32) == 0 &&
            repaired_payload[128 * 32] == repaired_scale;
        ok &= expect(format, true, true, 0);
        const int other = format == AME_MCFG_MXFP8_E4M3 ? AME_MCFG_MXFP8_E5M2 : AME_MCFG_MXFP8_E4M3;
        ok &= expect(other, true, false, 1); // Same generation/source/shape, different format.
        ok &= expect(other, true, true, 0);
    }
    const uint64_t allocations = ctx->mxfp4_allocation_attempts;
    for (int unknown : {0, 0x0e, 0x10, 0xff}) {
        bool hit = true;
        ok &= ame_get_packed_mx_weight(weight, &hit, unknown) == nullptr && !hit;
    }
    weight->ne[0] = 33;
    bool hit = true;
    ok &= ame_get_packed_mx_weight(weight, &hit, AME_MCFG_MXFP8_E4M3) == nullptr && !hit;
    weight->ne[0] = 64;
    ok &= ctx->mxfp4_allocation_attempts == allocations;

    // Exercise actual dispatch guards, not merely a pure packing predicate.
    // Each case must short-circuit before the capacity CSR probe.
    ggml_tensor * activation = ggml_new_tensor_2d(meta, GGML_TYPE_F32, 64, 64);
    activation->data = input.data();
    ggml_tensor * op = ggml_mul_mat(meta, weight, activation);
    const char * env_names[] = {"GGML_AME_MXFP8", "GGML_AME_MX32_VERIFIED", "GGML_AME_MXFP8_FORMAT", "GGML_AME_SCALAR"};
    std::vector<std::string> saved;
    std::vector<bool> present;
    for (const char * name : env_names) {
        const char * value = getenv(name);
        saved.emplace_back(value == nullptr ? "" : value);
        present.push_back(value != nullptr);
        unsetenv(name);
    }
    ggml::cpu::riscv_ame::tensor_traits traits;
    size_t work_size = 0;
    ggml_compute_params params = {};
    params.nth = 1;
    ok &= !traits.work_size(1, op, work_size) && !traits.compute_forward(&params, op);
    setenv("GGML_AME_MXFP8", "1", 1);
    ok &= !ame_backend_can_use_mx(op); // Enable alone is insufficient.
    unsetenv("GGML_AME_MXFP8");
    setenv("GGML_AME_MX32_VERIFIED", "1", 1);
    ok &= !ame_backend_can_use_mx(op); // Attestation alone is insufficient.
    setenv("GGML_AME_MXFP8", "1", 1);
    setenv("GGML_AME_MXFP8_FORMAT", "unknown", 1);
    ok &= !ame_backend_can_use_mx(op);
    setenv("GGML_AME_MXFP8_FORMAT", "e5m2", 1);
    ok &= ame_backend_mx_format(op) == AME_MCFG_MXFP8_E5M2;
    setenv("GGML_AME_MXFP8_FORMAT", "e4m3", 1);
    ok &= ame_backend_mx_format(op) == AME_MCFG_MXFP8_E4M3;
    params.nth = 2;
    ok &= !traits.work_size(2, op, work_size) && !traits.compute_forward(&params, op);
    // Consistent K33/N65/M63 graph and CPU checks live in the separate
    // actual-compute selftest; never claim mutated mismatching dims prove it.
    setenv("GGML_AME_SCALAR", "1", 1);
    ok &= !ame_backend_can_use_mx(op);
    for (size_t i = 0; i < saved.size(); ++i) {
        if (present[i]) setenv(env_names[i], saved[i].c_str(), 1);
        else unsetenv(env_names[i]);
    }
    ok &= ctx->mxfp4_allocation_attempts == allocations;
    ggml_backend_tensor_set(weight, input.data(), 0, size);
    ok &= expect(AME_MCFG_MXFP8_E4M3, true, false, 1);
    std::vector<float> unchanged(input.size());
    ggml_backend_tensor_get(weight, unchanged.data(), 0, size);
    ok &= std::memcmp(input.data(), unchanged.data(), size) == 0;
    ggml_backend_buffer_free(buffer);
    ok &= queries == 31 && rejected == 12 && failed_allocations == 2;
    if (ok) {
        printf("AME_MXFP8_CACHE_PASS queries=%u rejected=%u injected_oom=%u "
               "domain_reject_pack_allocations=0 same_identity_oom_retry=pass "
               "format_retry=pass raw_write=pass scale_layout=pass "
               "dual_optin=pass shape_test=separate_compute T2_fallback=pass CPU_storage=unchanged native=not-issued\n",
               queries, rejected, failed_allocations);
    }
    return ok;
}

// Exercise the real cache and buffer callbacks without registering a buffer
// type, probing CSRs, or invoking tensor_traits/native matrix execution.
// Called only by the existing pure-software quantization test.
extern "C" bool ggml_ame_mxfp4_cache_selftest(void) {
    ggml_init_params params = {8192, nullptr, true};
    ggml_context * meta = ggml_init(params);
    if (meta == nullptr) {
        return false;
    }
    ggml_tensor * tensor = ggml_new_tensor_2d(meta, GGML_TYPE_MXFP4, 64, 64);
    const size_t size = ggml_nbytes(tensor);
    ggml_backend_buffer_t buffer = ggml_backend_ame_buffer_type_alloc_buffer(nullptr, 2 * size + 64);
    if (buffer == nullptr) {
        ggml_free(meta);
        return false;
    }
    tensor->buffer = buffer;
    tensor->data = ggml_backend_buffer_get_base(buffer);
    ggml_backend_buffer_init_tensor(buffer, tensor);
    ame_buffer_context * ctx = ame_buffer_ctx(buffer);
    ggml_backend_buffer_clear(buffer, 0);
    unsigned queries = 0;
    unsigned rejected = 0;
    auto expect_tensor = [&](ggml_tensor * query, bool accepted, bool cached) {
        const uint64_t allocations = ctx->mxfp4_allocation_attempts;
        bool hit = false;
        const ame_packed_mx_info * packed = ame_get_packed_mx_weight(query, &hit, AME_MCFG_MXFP4);
        const auto & entry = ctx->packed_mxfp4.at(query);
        queries++;
        rejected += !accepted;
        return (packed != nullptr) == accepted && hit == cached &&
            entry.generation == ctx->generation && entry.source == query->data &&
            entry.M == query->ne[1] && entry.K == query->ne[0] &&
            (accepted || (entry.data == nullptr && entry.size == 0)) &&
            ctx->mxfp4_allocation_attempts == allocations + (accepted && !cached ? 1 : 0);
    };
    auto expect = [&](bool accepted, bool cached) {
        return expect_tensor(tensor, accepted, cached);
    };
    std::vector<block_mxfp4> blocks(size / sizeof(block_mxfp4));
    for (auto & block : blocks) {
        block.e = 127;
    }
    // All payloads are zero. A bad LAST block must reject the whole tensor.
    blocks.back().e = 255;
    ggml_backend_tensor_set(tensor, blocks.data(), 0, size);
    bool ok = expect(false, false);
    ok &= expect(false, false);
    // Rejection remains sticky for the recorded generation even after a raw
    // repair: conservative CPU fallback is safe until an observed write.
    static_cast<block_mxfp4 *>(tensor->data)[blocks.size() - 1].e = 127;
    ok &= expect(false, false);
    const size_t last_scale = (blocks.size() - 1) * sizeof(block_mxfp4);
    ggml_backend_tensor_memset(tensor, 127, last_scale, 1);
    ok &= expect(true, false);
    ok &= expect(true, true);

    // Positive cache hits must also observe direct CPU writes that bypass all
    // callbacks. A changed payload is rebuilt; a raw e255 must reject before
    // native issue, even though generation and tensor identity are unchanged.
    auto * raw = static_cast<block_mxfp4 *>(tensor->data);
    raw[0].qs[0] = 0x21;
    ok &= expect(true, false);
    const auto * packed_bytes = static_cast<const uint8_t *>(ctx->packed_mxfp4.at(tensor).data);
    ok &= packed_bytes != nullptr && packed_bytes[0] == 1 && packed_bytes[8] == 2;
    ok &= expect(true, true);
    raw[blocks.size() - 1].e = 255;
    ok &= expect(false, false);
    raw[blocks.size() - 1].e = 127;
    ok &= expect(false, false); // Rejection remains conservative until an observed write.
    ggml_backend_tensor_memset(tensor, 127, last_scale, 1);
    ok &= expect(true, false);

    const uint8_t bad_scale = 255;
    ggml_backend_tensor_set(tensor, &bad_scale, last_scale, 1);
    ok &= expect(false, false);
    ggml_backend_tensor_memset(tensor, 127, last_scale, 1);
    ok &= expect(true, false);

    ggml_backend_buffer_clear(buffer, 0xff);
    ok &= expect(false, false);
    ok &= expect(false, false);
    ggml_backend_buffer_clear(buffer, 0);
    ok &= expect(true, false);
    ok &= expect(true, true);
    // Rejection is per tensor identity, not just per pointer/generation.
    auto * zero_blocks = static_cast<block_mxfp4 *>(tensor->data);
    zero_blocks[0].e = 255;
    ok &= expect(false, false);
    ok &= expect(false, false);
    zero_blocks[0].e = 0;
    ok &= expect(false, false);
    ggml_tensor * alias = ggml_new_tensor_2d(meta, GGML_TYPE_MXFP4, 64, 64);
    alias->buffer = buffer;
    alias->data = tensor->data;
    ggml_backend_buffer_init_tensor(buffer, alias);
    ok &= expect_tensor(alias, true, false);
    ok &= expect_tensor(alias, true, true);
    ok &= expect(false, false); // Alias acceptance must not clear this rejection.

    // The source pointer also belongs to the rejected cache identity. Extra
    // buffer space keeps this shifted all-zero source completely in bounds.
    tensor->data = static_cast<uint8_t *>(tensor->data) + 64;
    ok &= expect(true, false);
    tensor->data = static_cast<uint8_t *>(tensor->data) - 64;
    ok &= expect(true, false);

    // Change K and M independently on the same tensor object, after a sticky
    // rejection and raw repair, preserving contiguous GGML block strides.
    auto shape = [&](int64_t k, int64_t m) {
        tensor->ne[0] = k;
        tensor->ne[1] = m;
        tensor->nb[1] = (size_t) k / QK_MXFP4 * sizeof(block_mxfp4);
        tensor->nb[2] = tensor->nb[1] * (size_t) m;
        tensor->nb[3] = tensor->nb[2];
    };
    zero_blocks[0].e = 255;
    ok &= expect(false, false);
    zero_blocks[0].e = 0;
    ok &= expect(false, false);
    shape(32, 64);
    ok &= expect(true, false);
    ok &= expect(true, true);
    shape(64, 64);
    ok &= expect(true, false);
    zero_blocks[0].e = 255;
    ok &= expect(false, false);
    zero_blocks[0].e = 0;
    ok &= expect(false, false);
    shape(64, 128);
    ok &= expect(true, false);
    ok &= expect(true, true);
    shape(64, 64);
    ok &= expect(true, false);
    std::vector<uint8_t> original(size, 0xff);
    ggml_backend_tensor_get(tensor, original.data(), 0, size);
    for (uint8_t byte : original) {
        ok &= byte == 0;
    }
    ggml_backend_buffer_free(buffer);
    ok &= ame_mxfp8_cache_selftest(meta);
    ggml_free(meta);
    if (ok) {
        printf("AME_MXFP4_CACHE_DETAIL queries=%u rejected=%u invalid_pack_allocations=0 "
               "tensor_identity=pass source_retry=pass M_retry=pass K_retry=pass native=not-issued\n",
               queries, rejected);
    }
    return ok;
}

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
