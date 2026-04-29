#include "ame.h"

#include "ggml-impl.h"

#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <string.h>

static sigjmp_buf g_ame_bf16_sigill_jmp;
static volatile sig_atomic_t g_ame_bf16_sigill_active = 0;

static void ggml_ame_bf16_sigill_handler(int signo) {
    if (signo == SIGILL && g_ame_bf16_sigill_active) {
        siglongjmp(g_ame_bf16_sigill_jmp, 1);
    }
}

static float ggml_ame_bf16_scalar_dot(const ggml_bf16_t * a, const ggml_bf16_t * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_BF16_TO_FP32(a[i]) * GGML_BF16_TO_FP32(b[i]);
    }
    return sum;
}

int ggml_ame_bf16_smoke_once(void) {
    static int cached = -1;
    if (cached != -1) {
        return cached;
    }

    const int tile_k = AME_TILE_K_BF16;
    const size_t tile_a_size = AME_TILE_M * tile_k * sizeof(ggml_bf16_t);
    const size_t tile_b_size = AME_TILE_N * tile_k * sizeof(ggml_bf16_t);
    const size_t tile_c_size = AME_TILE_M * AME_TILE_N * sizeof(float);

    ggml_bf16_t * tile_a = NULL;
    ggml_bf16_t * tile_b = NULL;
    float * tile_c = NULL;

    struct sigaction sa_new;
    struct sigaction sa_old;
    memset(&sa_new, 0, sizeof(sa_new));
    sa_new.sa_handler = ggml_ame_bf16_sigill_handler;
    sigemptyset(&sa_new.sa_mask);

    if (sigaction(SIGILL, &sa_new, &sa_old) != 0) {
        cached = 1;
        return cached;
    }

    int rc = 0;
    g_ame_bf16_sigill_active = 1;
    if (sigsetjmp(g_ame_bf16_sigill_jmp, 1) != 0) {
        rc = 2;
        goto done;
    }

    tile_a = (ggml_bf16_t *) ggml_aligned_malloc(tile_a_size);
    tile_b = (ggml_bf16_t *) ggml_aligned_malloc(tile_b_size);
    tile_c = (float *) ggml_aligned_malloc(tile_c_size);
    if (tile_a == NULL || tile_b == NULL || tile_c == NULL) {
        rc = 3;
        goto done;
    }

    memset(tile_a, 0, tile_a_size);
    memset(tile_b, 0, tile_b_size);
    memset(tile_c, 0, tile_c_size);

    tile_a[0 * tile_k + 0] = GGML_FP32_TO_BF16(1.0f);
    tile_a[0 * tile_k + 1] = GGML_FP32_TO_BF16(-2.0f);
    tile_a[0 * tile_k + (tile_k - 1)] = GGML_FP32_TO_BF16(0.75f);
    tile_a[1 * tile_k + 0] = GGML_FP32_TO_BF16(0.5f);
    tile_a[1 * tile_k + 1] = GGML_FP32_TO_BF16(4.0f);
    tile_a[1 * tile_k + (tile_k - 1)] = GGML_FP32_TO_BF16(-1.25f);

    tile_b[0 * tile_k + 0] = GGML_FP32_TO_BF16(3.0f);
    tile_b[0 * tile_k + 1] = GGML_FP32_TO_BF16(0.25f);
    tile_b[0 * tile_k + (tile_k - 1)] = GGML_FP32_TO_BF16(-0.5f);
    tile_b[1 * tile_k + 0] = GGML_FP32_TO_BF16(-1.5f);
    tile_b[1 * tile_k + 1] = GGML_FP32_TO_BF16(2.0f);
    tile_b[1 * tile_k + (tile_k - 1)] = GGML_FP32_TO_BF16(1.5f);

    ggml_ame_gemm_tile_bf16_fp32_bT(tile_a, tile_b, tile_c);

    const float expected00 = ggml_ame_bf16_scalar_dot(&tile_a[0 * tile_k], &tile_b[0 * tile_k], tile_k);
    const float expected01 = ggml_ame_bf16_scalar_dot(&tile_a[0 * tile_k], &tile_b[1 * tile_k], tile_k);
    const float expected10 = ggml_ame_bf16_scalar_dot(&tile_a[1 * tile_k], &tile_b[0 * tile_k], tile_k);
    const float expected11 = ggml_ame_bf16_scalar_dot(&tile_a[1 * tile_k], &tile_b[1 * tile_k], tile_k);

    if (fabsf(tile_c[0 * AME_TILE_N + 0] - expected00) > 1e-3f ||
        fabsf(tile_c[0 * AME_TILE_N + 1] - expected01) > 1e-3f ||
        fabsf(tile_c[1 * AME_TILE_N + 0] - expected10) > 1e-3f ||
        fabsf(tile_c[1 * AME_TILE_N + 1] - expected11) > 1e-3f) {
        rc = 4;
    }

done:
    g_ame_bf16_sigill_active = 0;
    sigaction(SIGILL, &sa_old, NULL);

    if (tile_a != NULL) {
        ggml_aligned_free(tile_a, tile_a_size);
    }
    if (tile_b != NULL) {
        ggml_aligned_free(tile_b, tile_b_size);
    }
    if (tile_c != NULL) {
        ggml_aligned_free(tile_c, tile_c_size);
    }

    cached = rc;
    return cached;
}
