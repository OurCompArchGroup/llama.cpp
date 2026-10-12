#include "ame-xsai-observe.h"

#include <atomic>
#include <mutex>
#include <time.h>

namespace {

std::atomic<int> g_enabled{0};
std::mutex g_mu;
uint64_t g_count[4];
uint64_t g_ns[4];

uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t) ts.tv_sec * UINT64_C(1000000000) + (uint64_t) ts.tv_nsec;
}

}  // namespace

void ame_xsai_observe_set_enabled(int enabled) {
    g_enabled.store(enabled ? 1 : 0, std::memory_order_release);
}

int ame_xsai_observe_enabled(void) {
    return g_enabled.load(std::memory_order_acquire);
}

void ame_xsai_observe_reset(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (int i = 0; i < 4; ++i) {
        g_count[i] = 0;
        g_ns[i] = 0;
    }
}

void ame_xsai_observe_export(ame_xsai_observe_export_t * out) {
    if (out == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    out->prepare_count = g_count[AME_XSAI_OBS_PREPARE];
    out->prepare_ns = g_ns[AME_XSAI_OBS_PREPARE];
    out->plan_create_count = g_count[AME_XSAI_OBS_PLAN_CREATE];
    out->plan_create_ns = g_ns[AME_XSAI_OBS_PLAN_CREATE];
    out->workspace_count = g_count[AME_XSAI_OBS_WORKSPACE];
    out->workspace_ns = g_ns[AME_XSAI_OBS_WORKSPACE];
    out->execute_count = g_count[AME_XSAI_OBS_EXECUTE];
    out->execute_ns = g_ns[AME_XSAI_OBS_EXECUTE];
    out->kernel_count = g_count[AME_XSAI_OBS_EXECUTE];
    out->kernel_ns = g_ns[AME_XSAI_OBS_EXECUTE];
}

void ame_xsai_observe_span_begin(int phase, uint64_t * token) {
    if (token == nullptr) {
        return;
    }
    *token = UINT64_MAX;
    if (phase < AME_XSAI_OBS_PREPARE || phase > AME_XSAI_OBS_EXECUTE) {
        return;
    }
    if (!g_enabled.load(std::memory_order_acquire)) {
        return;
    }
    *token = monotonic_ns();
}

void ame_xsai_observe_span_end(int phase, uint64_t token) {
    if (token == UINT64_MAX) {
        return;
    }
    if (phase < AME_XSAI_OBS_PREPARE || phase > AME_XSAI_OBS_EXECUTE) {
        return;
    }
    const uint64_t end = monotonic_ns();
    const uint64_t dt = end >= token ? end - token : 0;
    std::lock_guard<std::mutex> lock(g_mu);
    g_count[phase] += 1;
    g_ns[phase] += dt;
}
