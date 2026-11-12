#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ame-flash-attn.h"

#include "ame.h"
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "vec.h"

#if defined(__riscv_v)
#include <riscv_vector.h>
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int FA_M = AME_TILE_M;
constexpr int FA_K = AME_TILE_K_BF16;
constexpr size_t FA_ALIGN = 64;

struct ame_fa_workspace {
    ggml_bf16_t * q_panels;
    ggml_bf16_t * k_panels;
    ggml_bf16_t * p_panels;
    ggml_bf16_t * v_panels;
    ggml_bf16_t * v_rows;
    float * scores;
    float * pv;
    float * out;
    float * row_max;
    float * row_sum;
};

struct ame_fa_profile {
    uint64_t pack_q;
    uint64_t pack_k;
    uint64_t qk_ame;
    uint64_t softmax;
    uint64_t pack_v;
    uint64_t pack_pv;
    uint64_t pv_ame;
    uint64_t accumulate;
    uint64_t store;
    uint64_t qk_tiles;
    uint64_t pv_tiles;
};

static inline size_t align_up(size_t value) {
    return (value + FA_ALIGN - 1) & ~(FA_ALIGN - 1);
}

static inline uint64_t read_cycle() {
#if defined(__riscv)
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#else
    return 0;
#endif
}

static bool env_disabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr &&
        (strcmp(value, "0") == 0 || strcmp(value, "false") == 0 ||
         strcmp(value, "off") == 0 || strcmp(value, "no") == 0);
}

static bool env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && !env_disabled(name);
}

static bool implementation_enabled() {
    static const bool enabled = !env_disabled("GGML_AME_FLASH_ATTN");
    return enabled;
}

// FA 的 N tile 必须恰好等于硬件的 tensor 宽度：RTL 的 CDataController 在每次
// MMA 时断言 MatrixRegWorkingTensor_N === Tensor_MN（"not Full!"），N 小于
// 物理宽度会导致紧凑地址布局错位。Tensor_MN = tlenb / trlenb 由 CSR 探测
// 得到：minimal (CUTE_2Tops) 为 64，default (CUTE_8Tops_128SCP) 为 128。
// 因此 FA_N 不是编译期常量，而是按运行时硬件配置选择。
static int fa_tile_n() {
    const ggml_ame_hw_config * cfg = ggml_ame_hw_config_get();
    if (cfg == nullptr || !cfg->valid || cfg->rows == 0 || cfg->rows > (uint64_t) INT_MAX) {
        return 0;
    }
    const int fa_n = (int) cfg->rows;
    // K/P/V panel 均按 FA_K 列打包，N 必须是 FA_K 的整数倍。
    if (fa_n % FA_K != 0) {
        return 0;
    }
    return fa_n;
}

static bool shape_supported(const ggml_tensor * dst, int fa_n) {
    if (fa_n <= 0 || dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    return q->type == GGML_TYPE_F32 &&
        (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) &&
        k->type == v->type &&
        q->ne[0] == k->ne[0] &&
        q->ne[1] >= FA_M && k->ne[1] >= fa_n &&
        k->ne[0] > 0 && k->ne[0] % FA_K == 0 &&
        v->ne[0] > 0 && v->ne[0] % fa_n == 0 &&
        q->ne[2] % k->ne[2] == 0 && q->ne[3] % k->ne[3] == 0 &&
        q->ne[2] % v->ne[2] == 0 && q->ne[3] % v->ne[3] == 0 &&
        q->nb[0] == sizeof(float) &&
        k->nb[0] == ggml_type_size(k->type) &&
        v->nb[0] == ggml_type_size(v->type) &&
        dst->type == GGML_TYPE_F32 && dst->ne[0] == v->ne[0] &&
        dst->nb[0] == sizeof(float) && dst->nb[1] == (size_t) v->ne[0] * sizeof(float);
}

static size_t workspace_size_for_dims(int64_t dk, int64_t dv, int fa_n) {
    const size_t dk_tiles = (size_t) dk / FA_K;
    const size_t n_panels = (size_t) fa_n / FA_K;
    size_t size = FA_ALIGN;
    const auto add = [&size](size_t bytes) {
        size = align_up(size);
        size += bytes;
    };

    add(dk_tiles * FA_M * FA_K * sizeof(ggml_bf16_t));          // Q panels
    add(dk_tiles * (size_t) fa_n * FA_K * sizeof(ggml_bf16_t)); // K panels
    add(n_panels * FA_M * FA_K * sizeof(ggml_bf16_t));          // P panels
    add(n_panels * (size_t) fa_n * FA_K * sizeof(ggml_bf16_t)); // V panels
    add((size_t) fa_n * (size_t) dv * sizeof(ggml_bf16_t));     // row-major V tile
    add((size_t) FA_M * (size_t) fa_n * sizeof(float));         // QK scores
    add((size_t) FA_M * (size_t) fa_n * sizeof(float));         // PV result
    add(FA_M * (size_t) dv * sizeof(float));                    // output accumulator
    add(FA_M * sizeof(float));                                  // online max
    add(FA_M * sizeof(float));                                  // online sum
    return align_up(size);
}

template<typename T>
static T * carve(char *& cursor, size_t count) {
    uintptr_t address = (uintptr_t) cursor;
    address = (address + FA_ALIGN - 1) & ~(uintptr_t) (FA_ALIGN - 1);
    T * result = reinterpret_cast<T *>(address);
    cursor = reinterpret_cast<char *>(result + count);
    return result;
}

static ame_fa_workspace make_workspace(void * data, int64_t dk, int64_t dv, int fa_n) {
    char * cursor = static_cast<char *>(data);
    const size_t dk_tiles = (size_t) dk / FA_K;
    const size_t n_panels = (size_t) fa_n / FA_K;
    ame_fa_workspace ws;
    ws.q_panels = carve<ggml_bf16_t>(cursor, dk_tiles * FA_M * FA_K);
    ws.k_panels = carve<ggml_bf16_t>(cursor, dk_tiles * (size_t) fa_n * FA_K);
    ws.p_panels = carve<ggml_bf16_t>(cursor, n_panels * FA_M * FA_K);
    ws.v_panels = carve<ggml_bf16_t>(cursor, n_panels * (size_t) fa_n * FA_K);
    ws.v_rows   = carve<ggml_bf16_t>(cursor, (size_t) fa_n * (size_t) dv);
    ws.scores   = carve<float>(cursor, (size_t) FA_M * (size_t) fa_n);
    ws.pv       = carve<float>(cursor, (size_t) FA_M * (size_t) fa_n);
    ws.out      = carve<float>(cursor, FA_M * (size_t) dv);
    ws.row_max  = carve<float>(cursor, FA_M);
    ws.row_sum  = carve<float>(cursor, FA_M);
    return ws;
}

#if defined(__riscv_v)
static inline void store_bf16_rne(vfloat32m8_t values, ggml_bf16_t * dst, size_t vl) {
    vuint32m8_t bits = __riscv_vreinterpret_v_f32m8_u32m8(values);
    const vuint32m8_t lsb = __riscv_vand_vx_u32m8(__riscv_vsrl_vx_u32m8(bits, 16, vl), 1, vl);
    bits = __riscv_vadd_vx_u32m8(bits, 0x7fff, vl);
    bits = __riscv_vadd_vv_u32m8(bits, lsb, vl);
    const vuint16m4_t narrowed = __riscv_vnsrl_wx_u16m4(bits, 16, vl);
    __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t *>(dst), narrowed, vl);
}

static inline void store_bf16_rne(vfloat32m4_t values, ggml_bf16_t * dst, size_t vl) {
    vuint32m4_t bits = __riscv_vreinterpret_v_f32m4_u32m4(values);
    const vuint32m4_t lsb = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(bits, 16, vl), 1, vl);
    bits = __riscv_vadd_vx_u32m4(bits, 0x7fff, vl);
    bits = __riscv_vadd_vv_u32m4(bits, lsb, vl);
    const vuint16m2_t narrowed = __riscv_vnsrl_wx_u16m2(bits, 16, vl);
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t *>(dst), narrowed, vl);
}
#endif

static void f32_to_bf16(const float * src, ggml_bf16_t * dst, int64_t n) {
    int64_t offset = 0;
#if defined(__riscv_v)
    while (offset < n) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (n - offset));
        const vfloat32m8_t values = __riscv_vle32_v_f32m8(src + offset, vl);
        store_bf16_rne(values, dst + offset, vl);
        offset += (int64_t) vl;
    }
#else
    for (; offset < n; ++offset) {
        dst[offset] = GGML_FP32_TO_BF16(src[offset]);
    }
#endif
}

static void q8_to_bf16(const block_q8_0 * src, ggml_bf16_t * dst, int64_t n) {
    GGML_ASSERT(n % QK8_0 == 0);
    for (int64_t block = 0; block < n / QK8_0; ++block) {
        const float scale = GGML_FP16_TO_FP32(src[block].d);
#if defined(__riscv_v)
        int offset = 0;
        while (offset < QK8_0) {
            // Keep widening below LMUL=8. The FPGA RVV implementation faults
            // on the otherwise legal m2 -> m4 -> m8 conversion/store chain.
            const size_t vl = __riscv_vsetvl_e8m1(QK8_0 - offset);
            const vint8m1_t q8 = __riscv_vle8_v_i8m1(src[block].qs + offset, vl);
            const vint16m2_t q16 = __riscv_vwcvt_x_x_v_i16m2(q8, vl);
            vfloat32m4_t values = __riscv_vfwcvt_f_x_v_f32m4(q16, vl);
            values = __riscv_vfmul_vf_f32m4(values, scale, vl);
            store_bf16_rne(values, dst + block * QK8_0 + offset, vl);
            offset += (int) vl;
        }
#else
        for (int i = 0; i < QK8_0; ++i) {
            dst[block * QK8_0 + i] = GGML_FP32_TO_BF16(scale * src[block].qs[i]);
        }
#endif
    }
}

static void row_to_bf16(const void * src, ggml_type type, ggml_bf16_t * dst, int64_t n) {
    switch (type) {
        case GGML_TYPE_F32:
            f32_to_bf16(static_cast<const float *>(src), dst, n);
            break;
        case GGML_TYPE_F16: {
            const ggml_fp16_t * values = static_cast<const ggml_fp16_t *>(src);
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = GGML_FP32_TO_BF16(GGML_FP16_TO_FP32(values[i]));
            }
        } break;
        case GGML_TYPE_Q8_0:
            q8_to_bf16(static_cast<const block_q8_0 *>(src), dst, n);
            break;
        default:
            GGML_ABORT("unsupported AME FlashAttention row type");
    }
}

static const void * row_segment(const void * row, ggml_type type, int64_t element) {
    if (type == GGML_TYPE_Q8_0) {
        GGML_ASSERT(element % QK8_0 == 0);
        return static_cast<const block_q8_0 *>(row) + element / QK8_0;
    }
    return static_cast<const char *>(row) + element * ggml_type_size(type);
}

static void scale_row(float * row, int64_t n, float scale) {
    int64_t offset = 0;
#if defined(__riscv_v)
    while (offset < n) {
        const size_t vl = __riscv_vsetvl_e32m8((size_t) (n - offset));
        vfloat32m8_t values = __riscv_vle32_v_f32m8(row + offset, vl);
        values = __riscv_vfmul_vf_f32m8(values, scale, vl);
        __riscv_vse32_v_f32m8(row + offset, values, vl);
        offset += (int64_t) vl;
    }
#else
    for (; offset < n; ++offset) {
        row[offset] *= scale;
    }
#endif
}

static float softmax_row(float * row, int valid, int fa_n, float max_value) {
    const float sum = (float) ggml_vec_soft_max_f32(valid, row, row, max_value);
    for (int i = valid; i < fa_n; ++i) {
        row[i] = 0.0f;
    }
    return sum;
}

static float row_max(const float * row, int valid) {
    float result = -INFINITY;
    ggml_vec_max_f32(valid, &result, row);
    return result;
}

static void pack_q_panels(
        const ggml_tensor * q,
        int64_t iq1,
        int64_t iq2,
        int64_t iq3,
        int rows,
        ame_fa_workspace & ws) {
    const int dk_tiles = (int) q->ne[0] / FA_K;
    for (int kb = 0; kb < dk_tiles; ++kb) {
        ggml_bf16_t * panel = ws.q_panels + (size_t) kb * FA_M * FA_K;
        for (int tq = 0; tq < rows; ++tq) {
            const float * row = reinterpret_cast<const float *>(
                static_cast<const char *>(q->data) +
                (iq1 + tq) * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
            f32_to_bf16(row + kb * FA_K, panel + tq * FA_K, FA_K);
        }
        memset(panel + rows * FA_K, 0, (FA_M - rows) * FA_K * sizeof(ggml_bf16_t));
    }
}

static void pack_k_panels(
        const ggml_tensor * k,
        int64_t ic,
        int64_t ik2,
        int64_t ik3,
        int rows,
        int fa_n,
        ame_fa_workspace & ws) {
    const int dk_tiles = (int) k->ne[0] / FA_K;
    for (int kb = 0; kb < dk_tiles; ++kb) {
        ggml_bf16_t * panel = ws.k_panels + (size_t) kb * fa_n * FA_K;
        for (int tk = 0; tk < rows; ++tk) {
            const void * row = static_cast<const char *>(k->data) +
                (ic + tk) * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
            row_to_bf16(row_segment(row, k->type, kb * FA_K), k->type, panel + tk * FA_K, FA_K);
        }
        memset(panel + rows * FA_K, 0, (fa_n - rows) * FA_K * sizeof(ggml_bf16_t));
    }
}

static void pack_v_rows(
        const ggml_tensor * v,
        int64_t ic,
        int64_t iv2,
        int64_t iv3,
        int rows,
        int fa_n,
        ame_fa_workspace & ws) {
    const int64_t dv = v->ne[0];
    for (int tk = 0; tk < rows; ++tk) {
        const void * row = static_cast<const char *>(v->data) +
            (ic + tk) * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
        row_to_bf16(row, v->type, ws.v_rows + (size_t) tk * dv, dv);
    }
    memset(ws.v_rows + (size_t) rows * dv, 0,
        (size_t) (fa_n - rows) * (size_t) dv * sizeof(ggml_bf16_t));
}

static void pack_probability_panels(const float * scores, int rows, int fa_n, ggml_bf16_t * panels) {
    const int n_panels = fa_n / FA_K;
    for (int panel_idx = 0; panel_idx < n_panels; ++panel_idx) {
        ggml_bf16_t * panel = panels + (size_t) panel_idx * FA_M * FA_K;
        for (int tq = 0; tq < rows; ++tq) {
            f32_to_bf16(scores + (size_t) tq * fa_n + panel_idx * FA_K, panel + tq * FA_K, FA_K);
        }
        memset(panel + rows * FA_K, 0, (FA_M - rows) * FA_K * sizeof(ggml_bf16_t));
    }
}

static void pack_v_panels(const ggml_bf16_t * rows, int64_t dv, int64_t dv0, int fa_n, ggml_bf16_t * panels) {
    const int n_panels = fa_n / FA_K;
    for (int panel_idx = 0; panel_idx < n_panels; ++panel_idx) {
        ggml_bf16_t * panel = panels + (size_t) panel_idx * fa_n * FA_K;
        for (int n = 0; n < fa_n; ++n) {
            for (int k = 0; k < FA_K; ++k) {
                panel[n * FA_K + k] = rows[(panel_idx * FA_K + k) * dv + dv0 + n];
            }
        }
    }
}

static void print_profile(
        const ggml_tensor * k,
        const ame_fa_profile & profile,
        uint64_t total_cycles) {
    fprintf(stderr,
        "[AME_FLASH_ATTN_PROFILE] kv=%s qk_tiles=%llu pv_tiles=%llu "
        "cycles total=%llu pack_q=%llu pack_k=%llu qk_ame=%llu softmax=%llu "
        "pack_v=%llu pack_pv=%llu pv_ame=%llu accumulate=%llu store=%llu\n",
        ggml_type_name(k->type),
        (unsigned long long) profile.qk_tiles,
        (unsigned long long) profile.pv_tiles,
        (unsigned long long) total_cycles,
        (unsigned long long) profile.pack_q,
        (unsigned long long) profile.pack_k,
        (unsigned long long) profile.qk_ame,
        (unsigned long long) profile.softmax,
        (unsigned long long) profile.pack_v,
        (unsigned long long) profile.pack_pv,
        (unsigned long long) profile.pv_ame,
        (unsigned long long) profile.accumulate,
        (unsigned long long) profile.store);
    fflush(stderr);
}

} // namespace

extern "C" size_t ggml_ame_flash_attn_ext_work_size(const ggml_tensor * dst) {
    const int fa_n = fa_tile_n();
    if (fa_n == 0 || !shape_supported(dst, fa_n)) {
        return 0;
    }
    return workspace_size_for_dims(dst->src[1]->ne[0], dst->src[2]->ne[0], fa_n);
}

extern "C" int ggml_ame_flash_attn_ext_compute(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int fa_n = fa_tile_n();
    if (!implementation_enabled() || params->nth != 1 || params->ith != 0 || fa_n == 0 ||
        !shape_supported(dst, fa_n) || !ggml_ame_can_use_bf16(FA_M, fa_n, FA_K)) {
        return 0;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int64_t dk = k->ne[0];
    const int64_t dv = v->ne[0];
    const size_t required = workspace_size_for_dims(dk, dv, fa_n);
    GGML_ASSERT(params->wdata != nullptr && params->wsize >= required);
    ame_fa_workspace ws = make_workspace(params->wdata, dk, dv, fa_n);

    // 预脏化两个 MSC 目标（QK 的 scores、PV 的 pv）。
    // Svade 语义下 AME 的 MSC 写一个 CPU 从未写过的页时，DTLB 会因 PTE D=0
    // 报 store page fault；旧版 EMU 的 MLS 异常地址通路缺失（LSQWrapper 的
    // exceptionAddr 只从 store/load queue 取值，不含 mlsQueue），stval 会是
    // store queue 的陈旧地址，guest 内核按无 VMA 处理直接 SIGSEGV。
    // 这里用 CPU 先写一遍，把 PTE 的 A/D 置上，首个 MSC 便不再触发该异常；
    // 与 ame-matmul.c 里 tile_c 的预清零同源。两个 buffer 随后都会被 AME
    // 整块覆盖，这里的零值不影响结果。
    {
        const size_t msc_target_bytes = (size_t) FA_M * (size_t) fa_n * sizeof(float);
        memset(ws.scores, 0, msc_target_bytes);
        memset(ws.pv,     0, msc_target_bytes);
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = (uint32_t) q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -max_bias / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int64_t rk2 = q->ne[2] / k->ne[2];
    const int64_t rk3 = q->ne[3] / k->ne[3];
    const int64_t rv2 = q->ne[2] / v->ne[2];
    const int64_t rv3 = q->ne[3] / v->ne[3];
    const int dk_tiles = (int) dk / FA_K;
    const bool profile_enabled = env_enabled("GGML_AME_FLASH_ATTN_PROFILE");
    const uint64_t total_start = profile_enabled ? read_cycle() : 0;
    ame_fa_profile profile = {};

    static bool logged = false;
    if (!logged) {
        AME_LOG("FlashAttention backend active: q=f32 kv=%s DK=%lld DV=%lld Q=%lld KV=%lld FA_N=%d",
            ggml_type_name(k->type), (long long) dk, (long long) dv,
            (long long) q->ne[1], (long long) k->ne[1], fa_n);
        logged = true;
    }

    // 每个 AME op 入口重置 sync0 令牌计数，与 Q8 / BF16 mul_mat 路径一致。
    // 这也让 NEMU/EMU checkpoint 恢复时硬件 mtokr（不是 checkpoint 状态，
    // 恢复后为 0）与软件 target 计数从同一起点开始，避免恢复后死等令牌。
    ggml_ame_sync_begin_op();

    // 调试用：在 FA op 入口（令牌已重置、尚未发任何 AME kernel）触发 NEMU
    // manual one-shot checkpoint；仅设置 GGML_AME_FA_CPT_TRAP 时生效。
    if (getenv("GGML_AME_FA_CPT_TRAP") != NULL) {
#if defined(__riscv)
        asm volatile(
            "li a0, 0x105\n\t"
            ".insn r 0x6B, 0, 0, x0, x0, x0\n\t"
            :
            :
            : "a0", "memory");
#endif
    }

    for (int64_t iq3 = 0; iq3 < q->ne[3]; ++iq3) {
        for (int64_t iq2 = 0; iq2 < q->ne[2]; ++iq2) {
            const int64_t ik2 = iq2 / rk2;
            const int64_t ik3 = iq3 / rk3;
            const int64_t iv2 = iq2 / rv2;
            const int64_t iv3 = iq3 / rv3;
            const uint32_t h = (uint32_t) iq2;
            const float slope = max_bias > 0.0f ?
                (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1)) : 1.0f;

            for (int64_t iq1 = 0; iq1 < q->ne[1]; iq1 += FA_M) {
                const int q_rows = (int) std::min<int64_t>(FA_M, q->ne[1] - iq1);
                uint64_t start = profile_enabled ? read_cycle() : 0;
                pack_q_panels(q, iq1, iq2, iq3, q_rows, ws);
                if (profile_enabled) profile.pack_q += read_cycle() - start;

                memset(ws.out, 0, FA_M * (size_t) dv * sizeof(float));
                for (int tq = 0; tq < FA_M; ++tq) {
                    ws.row_max[tq] = -INFINITY;
                    ws.row_sum[tq] = 0.0f;
                }

                for (int64_t ic = 0; ic < k->ne[1]; ic += fa_n) {
                    const int kv_rows = (int) std::min<int64_t>(fa_n, k->ne[1] - ic);

                    bool all_masked = mask != nullptr;
                    if (mask != nullptr) {
                        for (int tq = 0; tq < q_rows && all_masked; ++tq) {
                            const ggml_fp16_t * mask_row = reinterpret_cast<const ggml_fp16_t *>(
                                static_cast<const char *>(mask->data) +
                                (iq1 + tq) * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                                (iq3 % mask->ne[3]) * mask->nb[3]);
                            for (int tk = 0; tk < kv_rows; ++tk) {
                                if (GGML_FP16_TO_FP32(mask_row[ic + tk]) != -INFINITY) {
                                    all_masked = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (all_masked) {
                        continue;
                    }

                    start = profile_enabled ? read_cycle() : 0;
                    pack_k_panels(k, ic, ik2, ik3, kv_rows, fa_n, ws);
                    if (profile_enabled) profile.pack_k += read_cycle() - start;

                    start = profile_enabled ? read_cycle() : 0;
                    ggml_ame_gemm_tile_bf16_fp32_bT_kloop(
                        ws.q_panels, FA_M * FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                        FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                        ws.k_panels, fa_n * FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                        FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                        dk_tiles, 2, fa_n, ws.scores);
                    if (profile_enabled) {
                        profile.qk_ame += read_cycle() - start;
                        ++profile.qk_tiles;
                    }

                    start = profile_enabled ? read_cycle() : 0;
                    bool any_row = false;
                    for (int tq = 0; tq < q_rows; ++tq) {
                        float * score_row = ws.scores + (size_t) tq * fa_n;
                        const ggml_fp16_t * mask_row = mask ? reinterpret_cast<const ggml_fp16_t *>(
                            static_cast<const char *>(mask->data) +
                            (iq1 + tq) * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                            (iq3 % mask->ne[3]) * mask->nb[3]) : nullptr;

                        for (int tk = 0; tk < kv_rows; ++tk) {
                            float value = score_row[tk] * scale;
                            if (logit_softcap != 0.0f) {
                                value = tanhf(value) * logit_softcap;
                            }
                            if (mask_row != nullptr) {
                                value += slope * GGML_FP16_TO_FP32(mask_row[ic + tk]);
                            }
                            score_row[tk] = value;
                        }
                        for (int tk = kv_rows; tk < fa_n; ++tk) {
                            score_row[tk] = -INFINITY;
                        }

                        const float tile_max = row_max(score_row, kv_rows);
                        if (tile_max == -INFINITY) {
                            memset(score_row, 0, (size_t) fa_n * sizeof(float));
                            continue;
                        }

                        any_row = true;
                        const float old_max = ws.row_max[tq];
                        const float new_max = std::max(old_max, tile_max);
                        if (new_max > old_max) {
                            const float old_scale = expf(old_max - new_max);
                            scale_row(ws.out + tq * dv, dv, old_scale);
                            ws.row_sum[tq] *= old_scale;
                        }
                        ws.row_max[tq] = new_max;
                        ws.row_sum[tq] += softmax_row(score_row, kv_rows, fa_n, new_max);
                    }
                    for (int tq = q_rows; tq < FA_M; ++tq) {
                        memset(ws.scores + (size_t) tq * fa_n, 0, (size_t) fa_n * sizeof(float));
                    }
                    if (profile_enabled) profile.softmax += read_cycle() - start;
                    if (!any_row) {
                        continue;
                    }

                    start = profile_enabled ? read_cycle() : 0;
                    pack_v_rows(v, ic, iv2, iv3, kv_rows, fa_n, ws);
                    if (profile_enabled) profile.pack_v += read_cycle() - start;

                    start = profile_enabled ? read_cycle() : 0;
                    pack_probability_panels(ws.scores, q_rows, fa_n, ws.p_panels);
                    if (profile_enabled) profile.pack_pv += read_cycle() - start;

                    for (int64_t dv0 = 0; dv0 < dv; dv0 += fa_n) {
                        start = profile_enabled ? read_cycle() : 0;
                        pack_v_panels(ws.v_rows, dv, dv0, fa_n, ws.v_panels);
                        if (profile_enabled) profile.pack_pv += read_cycle() - start;

                        start = profile_enabled ? read_cycle() : 0;
                        ggml_ame_gemm_tile_bf16_fp32_bT_kloop(
                            ws.p_panels, FA_M * FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                            FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                            ws.v_panels, fa_n * FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                            FA_K * (ptrdiff_t) sizeof(ggml_bf16_t),
                            fa_n / FA_K, 2, fa_n, ws.pv);
                        if (profile_enabled) {
                            profile.pv_ame += read_cycle() - start;
                            ++profile.pv_tiles;
                        }

                        start = profile_enabled ? read_cycle() : 0;
                        for (int tq = 0; tq < q_rows; ++tq) {
                            float * out_row = ws.out + tq * dv + dv0;
                            const float * pv_row = ws.pv + (size_t) tq * fa_n;
                            for (int n = 0; n < fa_n; ++n) {
                                out_row[n] += pv_row[n];
                            }
                        }
                        if (profile_enabled) profile.accumulate += read_cycle() - start;
                    }
                }

                if (sinks != nullptr) {
                    const float sink = static_cast<const float *>(sinks->data)[h];
                    for (int tq = 0; tq < q_rows; ++tq) {
                        float max_scale = 1.0f;
                        float sink_scale = 1.0f;
                        if (sink > ws.row_max[tq]) {
                            max_scale = expf(ws.row_max[tq] - sink);
                            scale_row(ws.out + tq * dv, dv, max_scale);
                        } else {
                            sink_scale = expf(sink - ws.row_max[tq]);
                        }
                        ws.row_sum[tq] = ws.row_sum[tq] * max_scale + sink_scale;
                    }
                }

                start = profile_enabled ? read_cycle() : 0;
                for (int tq = 0; tq < q_rows; ++tq) {
                    const float inverse = ws.row_sum[tq] == 0.0f ? 0.0f : 1.0f / ws.row_sum[tq];
                    scale_row(ws.out + tq * dv, dv, inverse);
                    void * output = static_cast<char *>(dst->data) +
                        (iq3 * dst->ne[2] * dst->ne[1] + iq2 + (iq1 + tq) * dst->ne[1]) * dst->nb[1];
                    memcpy(output, ws.out + tq * dv, (size_t) dv * sizeof(float));
                }
                if (profile_enabled) profile.store += read_cycle() - start;
            }
        }
    }

    if (profile_enabled) {
        print_profile(k, profile, read_cycle() - total_start);
    }
    return 1;
}
