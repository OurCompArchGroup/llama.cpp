#include "ame.h"
#include "common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "vec.h"

#if defined(GGML_XSAI_ALLOC)
#include "xsai_alloc.h"
#endif

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/* Method A safety gate: CUTE AMU computes per-row addresses as
 *   PA_row = PA_base + row * stride
 * without issuing a TLB re-query for each virtual page boundary.
 * This is only correct when the tile buffers are physically contiguous.
 * tile_a = AME_TILE_M * AME_TILE_K = 128*64 = 8192 bytes spans 3 virtual
 * pages, so non-contiguous allocation silently corrupts rows >= 63 and >=127.
 */
static void ame_assert_phys_contiguous(void) {
#if defined(GGML_USE_RV_AME) && defined(GGML_XSAI_ALLOC)
    static int checked = 0;
    if (checked) return;
    checked = 1;
    if (!xsai_pool_phys_contiguous()) {
        fprintf(stderr,
            "[AME] FATAL: xsai memory pool is not physically contiguous.\n"
            "             CUTE AMU row-address formula: PA_row = PA_base + row*stride\n"
            "             (no per-row TLB re-query, LocalMMU TODO not yet implemented).\n"
            "             tile_a (8192 B) spans 3 virtual pages; rows >=63 and >=127\n"
            "             will read wrong physical addresses => silent data corruption.\n"
            "             Fix: boot kernel with hugepages=512, or define\n"
            "             RESERVED_PHYS_BASE_ADDR for /dev/mem-backed pool.\n");
        abort();
    }
#endif
}

// Forward declaration of atomic GEMM function
extern void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,      // Input matrix A: MxK
    const int8_t * B,      // Input matrix B (transposed): NxK
    int32_t * C            // Output matrix C: MxN
);

extern void ggml_ame_gemm_tile_bf16_fp32_bT(
    const ggml_bf16_t * A,
    const ggml_bf16_t * B,
    float * C
);

#if defined(__riscv_v)
#include <riscv_vector.h>

// Dot product using RVV for Q8_0 blocks
// Compatible with block_q8_0 layout
static void ame_vec_dot_q8_0_rvv(int n, float * s, const void * vx, const void * vy) {
    const int qk = 32;
    const int nb = n / qk;
    const block_q8_0 * restrict x = (const block_q8_0 *)vx;
    const block_q8_0 * restrict y = (const block_q8_0 *)vy;
    
    float sumf = 0;
    
    for (int i = 0; i < nb; ++i) {
        // Each Q8_0 block has 32 int8 elements
        // Use vsetvli to handle any VLEN (128, 256, 512, 1024, etc.)
        size_t vl = __riscv_vsetvl_e8m2(qk);
        
        int sumi = 0;
        size_t offset = 0;
        
        // Process block in chunks that fit in vector registers
        while (offset < qk) {
            vl = __riscv_vsetvl_e8m2(qk - offset);
            
            // Load elements
            vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[i].qs + offset, vl);
            vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[i].qs + offset, vl);

            // Widen multiply: int8 * int8 -> int16
            vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);

            // Reduce sum: int16 -> int32
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);

            sumi += __riscv_vmv_x_s_i32m1_i32(v_sum);
            offset += vl;
        }

        sumf += sumi * (GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));
    }
    *s = sumf;
}

// Dot product for Q4_0 (standard format) using RVV
static void ame_vec_dot_q4_0_rvv(int n, float * s, const void * vx, const void * vy) {
    const int qk = 32;
    const int nb = n / qk;
    const block_q4_0 * restrict x = (const block_q4_0 *)vx;
    const block_q8_0 * restrict y = (const block_q8_0 *)vy; // y is always quantized to Q8_0 in our gemm
    
    float sumf = 0;
    
    for (int i = 0; i < nb; ++i) {
        // Unpack Q4_0 block (4-bit to 8-bit)
        int8_t x_unpacked[32];
        for (int j = 0; j < 16; j++) {
            uint8_t v = x[i].qs[j];
            x_unpacked[j] = (int8_t)(v & 0x0F) - 8;
            x_unpacked[j+16] = (int8_t)((v >> 4) & 0x0F) - 8;
        }

        // Now compute dot product with proper vsetvl
        int sumi = 0;
        size_t offset = 0;
        
        while (offset < qk) {
            size_t vl = __riscv_vsetvl_e8m2(qk - offset);
            
            vint8m2_t bx = __riscv_vle8_v_i8m2(x_unpacked + offset, vl);
            vint8m2_t by = __riscv_vle8_v_i8m2(y[i].qs + offset, vl);
            
            vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx, by, vl);
            
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);
            
            sumi += __riscv_vmv_x_s_i32m1_i32(v_sum);
            offset += vl;
        }
        
        sumf += sumi * (GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));
    }
    *s = sumf;
}
#endif

static float ame_vec_dot_bf16_scalar(int n, const ggml_bf16_t * x, const ggml_bf16_t * y) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_BF16_TO_FP32(x[i]) * GGML_BF16_TO_FP32(y[i]);
    }
    return sum;
}

static float ame_vec_dot_bf16_fallback(int n, const ggml_bf16_t * x, const ggml_bf16_t * y) {
#if defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfwma)
    float sum = 0.0f;
    ggml_vec_dot_bf16(n, &sum, 0, (ggml_bf16_t *) x, 0, (ggml_bf16_t *) y, 0, 1);
    return sum;
#else
    return ame_vec_dot_bf16_scalar(n, x, y);
#endif
}

// ggml_ame_quantize_row_f32_to_q8_0 is now in ame-helper.c

static size_t ame_align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static size_t ggml_ame_q8_workspace_size(int64_t N, int64_t n_k_tiles) {
    const size_t packed_b_panel_size = (size_t) n_k_tiles * AME_TILE_N * AME_TILE_K * sizeof(int8_t);
    size_t size = 64;
    size = ame_align_up_size(size, 64);
    size += (size_t) N * (size_t) n_k_tiles * sizeof(ggml_fp16_t);
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_K * sizeof(int8_t);
    size = ame_align_up_size(size, 64);
    size += packed_b_panel_size;
    size = ame_align_up_size(size, 64);
    size += AME_TILE_M * AME_TILE_N * sizeof(int32_t);
    return size;
}

static size_t ggml_ame_q8_0_workspace_size(int64_t N, int64_t K) {
    return ggml_ame_q8_workspace_size(N, K / QK8_0);
}

static inline void ggml_ame_quantize_block_f32_to_q8_0(const float * x, block_q8_0 * y) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float v = x[j];
        const float av = fabsf(v);
        if (av > amax) {
            amax = av;
        }
    }

    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;

    y->d = GGML_FP32_TO_FP16(d);
    for (int j = 0; j < 32; ++j) {
        y->qs[j] = roundf(x[j] * id);
    }
}

static inline void ggml_ame_quantize_block_f32_to_q8_64(const float * x, int valid, block_q8_ame64 * y) {
    float tmp[AME_Q8_PACK_K];
    memset(tmp, 0, sizeof(tmp));
    if (valid > 0) {
        memcpy(tmp, x, (size_t) valid * sizeof(float));
    }

    float amax = 0.0f;
    for (int j = 0; j < AME_Q8_PACK_K; ++j) {
        const float av = fabsf(tmp[j]);
        if (av > amax) {
            amax = av;
        }
    }

    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;

    y->d = GGML_FP32_TO_FP16(d);
    for (int j = 0; j < AME_Q8_PACK_K; ++j) {
        y->qs[j] = roundf(tmp[j] * id);
    }
}

struct ame_x_q64_cache {
    const void * key;
    int graph_id;
    size_t stride;
    int64_t K;
    int64_t N;
    block_q8_ame64 * blocks;
    size_t blocks_count;
};

static struct ame_x_q64_cache g_ame_x_q64_cache = {0};

static const block_q8_ame64 * ame_prepare_x_q64_cache(
    const void * src1_key,
    const void * src1,
    int64_t K,
    int64_t N,
    size_t src1_stride,
    int graph_id
) {
    if (graph_id <= 0) {
        return NULL;
    }

    const int64_t nb64 = (K + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    const size_t blocks_count = (size_t) N * (size_t) nb64;

    if (g_ame_x_q64_cache.key == src1_key &&
        g_ame_x_q64_cache.graph_id == graph_id &&
        g_ame_x_q64_cache.stride == src1_stride &&
        g_ame_x_q64_cache.K == K &&
        g_ame_x_q64_cache.N == N &&
        g_ame_x_q64_cache.blocks != NULL) {
        return g_ame_x_q64_cache.blocks;
    }

    const size_t alloc_size = blocks_count * sizeof(block_q8_ame64);
    if (g_ame_x_q64_cache.blocks == NULL || g_ame_x_q64_cache.blocks_count != blocks_count) {
        if (g_ame_x_q64_cache.blocks != NULL) {
            ggml_aligned_free(g_ame_x_q64_cache.blocks, g_ame_x_q64_cache.blocks_count * sizeof(block_q8_ame64));
        }
        g_ame_x_q64_cache.blocks = (block_q8_ame64 *) ggml_aligned_malloc(alloc_size);
        if (g_ame_x_q64_cache.blocks == NULL) {
            memset(&g_ame_x_q64_cache, 0, sizeof(g_ame_x_q64_cache));
            return NULL;
        }
        g_ame_x_q64_cache.blocks_count = blocks_count;
    }

    for (int64_t j = 0; j < N; ++j) {
        const float * src1_col = (const float *) ((const char *) src1 + j * src1_stride);
        for (int64_t kb = 0; kb < nb64; ++kb) {
            const int64_t base = kb * AME_Q8_PACK_K;
            const int valid = (base + AME_Q8_PACK_K <= K) ? AME_Q8_PACK_K : (K > base ? (int) (K - base) : 0);
            block_q8_ame64 * dst = &g_ame_x_q64_cache.blocks[j * nb64 + kb];
            ggml_ame_quantize_block_f32_to_q8_64(src1_col + base, valid, dst);
        }
    }

    g_ame_x_q64_cache.key = src1_key;
    g_ame_x_q64_cache.graph_id = graph_id;
    g_ame_x_q64_cache.stride = src1_stride;
    g_ame_x_q64_cache.K = K;
    g_ame_x_q64_cache.N = N;
    return g_ame_x_q64_cache.blocks;
}

// Wrapper for AME-accelerated Q8_0 GEMM
void ggml_ame_mul_mat_q8_0(
    const void * src0,  // Weight matrix (Q8_0)
    const void * src1,  // Input matrix (F32)
    void * dst,         // Output (F32)
    int64_t ne00,       // K
    int64_t ne01,       // M
    int64_t ne10,       // K (unused)
    int64_t ne11,       // N
    size_t src1_stride,
    void * work_data,
    size_t work_size
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    
    const int qk = 32;
    const int64_t nb_x = K / qk;
    const size_t packed_b_tile_size = AME_TILE_N * AME_TILE_K * sizeof(int8_t);
    const size_t packed_b_panel_size = (size_t) nb_x * packed_b_tile_size;

    const block_q8_0 * restrict x = (const block_q8_0 *)src0;
    float * restrict out = (float *)dst;

    const size_t required_wsize = ggml_ame_q8_0_workspace_size(N, K);
    uint8_t * workspace = (uint8_t *)work_data;
    int allocated_workspace = 0;

    if (workspace == NULL || work_size < required_wsize) {
        workspace = (uint8_t *)ggml_aligned_malloc(required_wsize);
        if (!workspace) return;
        work_size = required_wsize;
        allocated_workspace = 1;
    }

    uintptr_t ws_ptr = (uintptr_t)workspace;
    uintptr_t ws_end = ws_ptr + work_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    const int64_t y_scale_count = N * nb_x;
    ggml_fp16_t * y_scales = (ggml_fp16_t *)ws_ptr;
    ws_ptr += y_scale_count * sizeof(ggml_fp16_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_a = (int8_t *)ws_ptr;
    ws_ptr += AME_TILE_M * AME_TILE_K * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * packed_b_panel = (int8_t *)ws_ptr;
    ws_ptr += packed_b_panel_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int32_t * tile_c = (int32_t *)ws_ptr;
    ws_ptr += AME_TILE_M * AME_TILE_N * sizeof(int32_t);

    if (ws_ptr > ws_end) {
        if (allocated_workspace) {
            ggml_aligned_free(workspace, work_size);
        }
        return;
    }

    // Method A: verify physical contiguity contract before first AME instruction
    ame_assert_phys_contiguous();
    memset(tile_a, 0, AME_TILE_M * AME_TILE_K * sizeof(int8_t));

    for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
        const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (N - j0);

        memset(packed_b_panel, 0, packed_b_panel_size);
        for (int64_t kb = 0; kb < nb_x; kb++) {
            int8_t * tile_b = packed_b_panel + kb * packed_b_tile_size;
            for (int j = 0; j < jmax; j++) {
                 const float * src1_col = (const float *)((const char *)src1 + (j0 + j) * src1_stride);
                 block_q8_0 tmp_block;
                 ggml_ame_quantize_block_f32_to_q8_0(src1_col + kb * qk, &tmp_block);
                 y_scales[(j0 + j) * nb_x + kb] = tmp_block.d;
                 memcpy(&tile_b[j * AME_TILE_K], tmp_block.qs, qk);
            }
        }

        for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
            const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);
            
            // Full AME Path for full tiles
            float acc_f32[AME_TILE_M * AME_TILE_N];
            memset(acc_f32, 0, sizeof(acc_f32));

            for (int64_t kb = 0; kb < nb_x; kb++) {
                // Prepare Tile A (16 x 32)
                for (int i = 0; i < AME_TILE_M; i++) {
                     if (i < imax) {
                         const block_q8_0 * b = &x[(i0 + i) * nb_x + kb];
                         memcpy(&tile_a[i * AME_TILE_K], b->qs, qk);
                     } else {
                         memset(&tile_a[i * AME_TILE_K], 0, qk);
                     }
                }

                // Prepare Tile B (16 x 32)
                const int8_t * tile_b = packed_b_panel + kb * packed_b_tile_size;

                memset(tile_c, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                ggml_ame_gemm_tile_i8_i32_bT(tile_a, tile_b, tile_c);

                // Accumulate scaling factors
                for (int i = 0; i < imax; i++) {
                     const block_q8_0 * bx = &x[(i0 + i) * nb_x + kb];
                     const float d_x = GGML_FP16_TO_FP32(bx->d);
                     for (int j = 0; j < jmax; j++) {
                         const float d_y = GGML_FP16_TO_FP32(y_scales[(j0 + j) * nb_x + kb]);
                         
                         acc_f32[i * AME_TILE_N + j] += tile_c[i * AME_TILE_N + j] * (d_x * d_y);
                     }
                }
            }

            // Copy back
            for (int i = 0; i < imax; i++) {
                for (int j = 0; j < jmax; j++) {
                    out[(j0 + j) * M + (i0 + i)] = acc_f32[i * AME_TILE_N + j];
                }
            }
        }
    }

    if (allocated_workspace) {
        ggml_aligned_free(workspace, work_size);
    }
}

void ggml_ame_mul_mat_q8_0_ame64(
    const void * src0,
    const void * src1_key,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride,
    int graph_id,
    void * work_data,
    size_t work_size
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    const int64_t nb64 = (K + AME_Q8_PACK_K - 1) / AME_Q8_PACK_K;
    const size_t packed_b_tile_size = AME_TILE_N * AME_TILE_K * sizeof(int8_t);
    const size_t packed_b_panel_size = (size_t) nb64 * packed_b_tile_size;

    const block_q8_ame64 * restrict x = (const block_q8_ame64 *) src0;
    float * restrict out = (float *) dst;

    const size_t required_wsize = ggml_ame_q8_workspace_size(N, nb64);
    uint8_t * workspace = (uint8_t *) work_data;
    int allocated_workspace = 0;

    if (workspace == NULL || work_size < required_wsize) {
        workspace = (uint8_t *) ggml_aligned_malloc(required_wsize);
        if (!workspace) return;
        work_size = required_wsize;
        allocated_workspace = 1;
    }

    uintptr_t ws_ptr = (uintptr_t) workspace;
    uintptr_t ws_end = ws_ptr + work_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    const int64_t y_scale_count = N * nb64;
    ggml_fp16_t * y_scales = (ggml_fp16_t *) ws_ptr;
    ws_ptr += y_scale_count * sizeof(ggml_fp16_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * tile_a = (int8_t *) ws_ptr;
    ws_ptr += AME_TILE_M * AME_TILE_K * sizeof(int8_t);

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int8_t * packed_b_panel = (int8_t *) ws_ptr;
    ws_ptr += packed_b_panel_size;

    ws_ptr = ame_align_up_size(ws_ptr, 64);
    int32_t * tile_c = (int32_t *) ws_ptr;
    ws_ptr += AME_TILE_M * AME_TILE_N * sizeof(int32_t);

    if (ws_ptr > ws_end) {
        if (allocated_workspace) {
            ggml_aligned_free(workspace, work_size);
        }
        return;
    }

    ame_assert_phys_contiguous();
    memset(tile_a, 0, AME_TILE_M * AME_TILE_K * sizeof(int8_t));

    const block_q8_ame64 * xq = ame_prepare_x_q64_cache(src1_key, src1, K, N, src1_stride, graph_id);
    if (xq == NULL) {
        // no graph-local cache id available; fall back to one-shot quantization by using
        // a temporary cache entry keyed to this call only
        xq = ame_prepare_x_q64_cache(src1, src1, K, N, src1_stride, 1);
        if (xq == NULL) {
            if (allocated_workspace) {
                ggml_aligned_free(workspace, work_size);
            }
            return;
        }
        g_ame_x_q64_cache.key = NULL;
        g_ame_x_q64_cache.graph_id = 0;
    }

    for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
        const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (N - j0);

        memset(packed_b_panel, 0, packed_b_panel_size);
        for (int64_t kb = 0; kb < nb64; ++kb) {
            int8_t * tile_b = packed_b_panel + kb * packed_b_tile_size;
            for (int j = 0; j < jmax; ++j) {
                const block_q8_ame64 * bx = &xq[(j0 + j) * nb64 + kb];
                y_scales[(j0 + j) * nb64 + kb] = bx->d;
                memcpy(&tile_b[j * AME_TILE_K], bx->qs, AME_Q8_PACK_K);
            }
        }

        for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
            const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);
            float acc_f32[AME_TILE_M * AME_TILE_N];
            memset(acc_f32, 0, sizeof(acc_f32));

            for (int64_t kb = 0; kb < nb64; ++kb) {
                for (int i = 0; i < AME_TILE_M; ++i) {
                    if (i < imax) {
                        const block_q8_ame64 * b = &x[(i0 + i) * nb64 + kb];
                        memcpy(&tile_a[i * AME_TILE_K], b->qs, AME_Q8_PACK_K);
                    } else {
                        memset(&tile_a[i * AME_TILE_K], 0, AME_Q8_PACK_K);
                    }
                }

                const int8_t * tile_b = packed_b_panel + kb * packed_b_tile_size;
                memset(tile_c, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                ggml_ame_gemm_tile_i8_i32_bT(tile_a, tile_b, tile_c);

                for (int i = 0; i < imax; ++i) {
                    const block_q8_ame64 * bx = &x[(i0 + i) * nb64 + kb];
                    const float d_x = GGML_FP16_TO_FP32(bx->d);
                    for (int j = 0; j < jmax; ++j) {
                        const float d_y = GGML_FP16_TO_FP32(y_scales[(j0 + j) * nb64 + kb]);
                        acc_f32[i * AME_TILE_N + j] += tile_c[i * AME_TILE_N + j] * (d_x * d_y);
                    }
                }
            }

            for (int i = 0; i < imax; ++i) {
                for (int j = 0; j < jmax; ++j) {
                    out[(j0 + j) * M + (i0 + i)] = acc_f32[i * AME_TILE_N + j];
                }
            }
        }
    }

    if (allocated_workspace) {
        ggml_aligned_free(workspace, work_size);
    }

    GGML_UNUSED(ne10);
}

static const ggml_bf16_t * ame_get_bf16_col_ptr(
    const void * src1,
    int64_t col,
    int64_t K,
    size_t src1_stride,
    enum ggml_type src1_type,
    const ggml_bf16_t * converted_src1
) {
    if (src1_type == GGML_TYPE_BF16) {
        return (const ggml_bf16_t *) ((const char *) src1 + col * src1_stride);
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(src1_stride);
    return converted_src1 + col * K;
}

void ggml_ame_mul_mat_bf16(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride,
    enum ggml_type src1_type,
    void * work_data,
    size_t work_size
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    const int64_t K_AME = (K / AME_TILE_K_BF16) * AME_TILE_K_BF16;

    const ggml_bf16_t * restrict a = (const ggml_bf16_t *) src0;
    float * restrict out = (float *) dst;

    ggml_bf16_t * converted_src1 = NULL;
    int src1_allocated = 0;

    const size_t tile_a_size = AME_TILE_M * AME_TILE_K_BF16 * sizeof(ggml_bf16_t);
    const size_t tile_b_size = AME_TILE_N * AME_TILE_K_BF16 * sizeof(ggml_bf16_t);
    const size_t tile_c_size = AME_TILE_M * AME_TILE_N * sizeof(float);
    const size_t acc_tile_size = AME_TILE_M * AME_TILE_N * sizeof(float);

    ggml_bf16_t * tile_a = (ggml_bf16_t *) ggml_aligned_malloc(tile_a_size);
    ggml_bf16_t * tile_b = (ggml_bf16_t *) ggml_aligned_malloc(tile_b_size);
    float * tile_c = (float *) ggml_aligned_malloc(tile_c_size);
    float * acc_tile = (float *) malloc(acc_tile_size);

    if (tile_a == NULL || tile_b == NULL || tile_c == NULL || acc_tile == NULL) {
        goto cleanup;
    }

    if (src1_type == GGML_TYPE_F32) {
        converted_src1 = (ggml_bf16_t *) ggml_aligned_malloc((size_t) N * (size_t) K * sizeof(ggml_bf16_t));
        if (converted_src1 == NULL) {
            goto cleanup;
        }
        src1_allocated = 1;

        for (int64_t j = 0; j < N; ++j) {
            const float * src1_col = (const float *) ((const char *) src1 + j * src1_stride);
            ggml_cpu_fp32_to_bf16(src1_col, converted_src1 + j * K, K);
        }
    } else {
        GGML_ASSERT(src1_type == GGML_TYPE_BF16);
    }

    if (K_AME > 0) {
        ame_assert_phys_contiguous();
    }

    for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
        const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (int) (N - j0);

        for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
            const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (int) (M - i0);

            if (imax != AME_TILE_M || jmax != AME_TILE_N || K_AME == 0) {
                for (int i = 0; i < imax; ++i) {
                    const ggml_bf16_t * row_a = a + (i0 + i) * K;
                    for (int j = 0; j < jmax; ++j) {
                        const ggml_bf16_t * col_b = ame_get_bf16_col_ptr(src1, j0 + j, K, src1_stride, src1_type, converted_src1);
                        out[(j0 + j) * M + (i0 + i)] = ame_vec_dot_bf16_fallback(K, row_a, col_b);
                    }
                }
                continue;
            }

            memset(acc_tile, 0, acc_tile_size);

            for (int64_t k0 = 0; k0 < K_AME; k0 += AME_TILE_K_BF16) {
                for (int i = 0; i < AME_TILE_M; ++i) {
                    memcpy(tile_a + i * AME_TILE_K_BF16, a + (i0 + i) * K + k0, AME_TILE_K_BF16 * sizeof(ggml_bf16_t));
                }

                for (int j = 0; j < AME_TILE_N; ++j) {
                    const ggml_bf16_t * col_b = ame_get_bf16_col_ptr(src1, j0 + j, K, src1_stride, src1_type, converted_src1);
                    memcpy(tile_b + j * AME_TILE_K_BF16, col_b + k0, AME_TILE_K_BF16 * sizeof(ggml_bf16_t));
                }

                memset(tile_c, 0, tile_c_size);
                ggml_ame_gemm_tile_bf16_fp32_bT(tile_a, tile_b, tile_c);

                for (int i = 0; i < AME_TILE_M; ++i) {
                    for (int j = 0; j < AME_TILE_N; ++j) {
                        acc_tile[i * AME_TILE_N + j] += tile_c[i * AME_TILE_N + j];
                    }
                }
            }

            if (K_AME < K) {
                const int tail = (int) (K - K_AME);
                for (int i = 0; i < AME_TILE_M; ++i) {
                    const ggml_bf16_t * row_a = a + (i0 + i) * K + K_AME;
                    for (int j = 0; j < AME_TILE_N; ++j) {
                        const ggml_bf16_t * col_b = ame_get_bf16_col_ptr(src1, j0 + j, K, src1_stride, src1_type, converted_src1) + K_AME;
                        acc_tile[i * AME_TILE_N + j] += ame_vec_dot_bf16_fallback(tail, row_a, col_b);
                    }
                }
            }

            for (int i = 0; i < AME_TILE_M; ++i) {
                for (int j = 0; j < AME_TILE_N; ++j) {
                    out[(j0 + j) * M + (i0 + i)] = acc_tile[i * AME_TILE_N + j];
                }
            }
        }
    }

cleanup:
    if (tile_a != NULL) {
        ggml_aligned_free(tile_a, tile_a_size);
    }
    if (tile_b != NULL) {
        ggml_aligned_free(tile_b, tile_b_size);
    }
    if (tile_c != NULL) {
        ggml_aligned_free(tile_c, tile_c_size);
    }
    free(acc_tile);
    if (src1_allocated) {
        ggml_aligned_free(converted_src1, (size_t) N * (size_t) K * sizeof(ggml_bf16_t));
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(work_data);
    GGML_UNUSED(work_size);
}

// Wrapper for AME-accelerated Q4_0 GEMM
// Assumes REPACKED block_q4_0_ame inputs for src0
void ggml_ame_mul_mat_q4_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    const int qk = 32;
    const int64_t nb_x = K / qk;

    // Treat src0 as REPACKED block_q4_0_ame
    // Note: The backend repacks Q4_0 to this format automatically
    // Layout matches block_q8_0 (d: f16, qs: i8[32]), so we can reuse Q8_0 kernels/logic
    const block_q4_0_ame * restrict x = (const block_q4_0_ame *)src0;
    float * restrict out = (float *)dst;

    const int64_t y_q8_size = N * nb_x;
    block_q8_0 * y_q8 = (block_q8_0 *)malloc(y_q8_size * sizeof(block_q8_0));
    if (!y_q8) return;

    for (int64_t j = 0; j < N; j++) {
        const float * src1_col = (const float *)((const char *)src1 + j * src1_stride);
        ggml_ame_quantize_row_f32_to_q8_0(src1_col, y_q8 + j * nb_x, K);
    }
    const block_q8_0 * restrict y = y_q8;

    int8_t * tile_a = (int8_t *)ggml_aligned_malloc(AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    int8_t * tile_b = (int8_t *)ggml_aligned_malloc(AME_TILE_N * AME_TILE_K * sizeof(int8_t));
    int32_t * tile_c = (int32_t *)ggml_aligned_malloc(AME_TILE_M * AME_TILE_N * sizeof(int32_t));
    if (!tile_a || !tile_b || !tile_c) {
        ggml_aligned_free(tile_a, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
        ggml_aligned_free(tile_b, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
        ggml_aligned_free(tile_c, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
        free(y_q8);
        return;
    }
    // Method A: verify physical contiguity contract before first AME instruction
    ame_assert_phys_contiguous();

    for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
        const int imax = (i0 + AME_TILE_M <= M) ? AME_TILE_M : (M - i0);

        for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
            const int jmax = (j0 + AME_TILE_N <= N) ? AME_TILE_N : (N - j0);

#if defined(__riscv_v)
            if (imax < AME_TILE_M || jmax < AME_TILE_N) {
                 for (int i = 0; i < imax; ++i) {
                     for (int j = 0; j < jmax; ++j) {
                         const block_q4_0_ame * row_x = &x[(i0 + i) * nb_x];
                         const block_q8_0 * col_y = &y[(j0 + j) * nb_x];
                         // Repacked Q4_0 layout is compatible with Q8_0 for dot product
                         ame_vec_dot_q8_0_rvv(K, &out[(j0 + j) * M + (i0 + i)], (const void*)row_x, col_y);
                     }
                 }
                 continue;
            }
#endif

            float acc_f32[AME_TILE_M * AME_TILE_N];
            memset(acc_f32, 0, sizeof(acc_f32));

            for (int64_t kb = 0; kb < nb_x; kb++) {
                
                // Prepare Tile A (16 x 32) - Directly copy repacked data
                for (int i = 0; i < AME_TILE_M; i++) {
                     memset(&tile_a[i * AME_TILE_K], 0, AME_TILE_K);
                     if (i < imax) {
                         const block_q4_0_ame * b = &x[(i0 + i) * nb_x + kb];
                         memcpy(&tile_a[i * AME_TILE_K], b->qs, qk);
                     }
                }

                // Prepare Tile B (16 x 32)
                for (int j = 0; j < AME_TILE_N; j++) {
                     memset(&tile_b[j * AME_TILE_K], 0, AME_TILE_K);
                     if (j < jmax) {
                         const block_q8_0 * b = &y[(j0 + j) * nb_x + kb];
                         memcpy(&tile_b[j * AME_TILE_K], b->qs, qk);
                     }
                }

                memset(tile_c, 0, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
                ggml_ame_gemm_tile_i8_i32_bT(tile_a, tile_b, tile_c);

                for (int i = 0; i < imax; i++) {
                     const block_q4_0_ame * bx = &x[(i0 + i) * nb_x + kb];
                     const float d_x = GGML_FP16_TO_FP32(bx->d);
                     for (int j = 0; j < jmax; j++) {
                         const block_q8_0 * by = &y[(j0 + j) * nb_x + kb];
                         const float d_y = GGML_FP16_TO_FP32(by->d);
                         acc_f32[i * AME_TILE_N + j] += tile_c[i * AME_TILE_N + j] * (d_x * d_y);
                     }
                }
            }

            for (int i = 0; i < imax; i++) {
                for (int j = 0; j < jmax; j++) {
                    out[(j0 + j) * M + (i0 + i)] = acc_f32[i * AME_TILE_N + j];
                }
            }
        }
    }

    ggml_aligned_free(tile_a, AME_TILE_M * AME_TILE_K * sizeof(int8_t));
    ggml_aligned_free(tile_b, AME_TILE_N * AME_TILE_K * sizeof(int8_t));
    ggml_aligned_free(tile_c, AME_TILE_M * AME_TILE_N * sizeof(int32_t));
    free(y_q8);
}
