#pragma once

#include <stdint.h>

/*
 * Optional counts and monotonic timestamps for the XSAI adapter.
 * Disabled until xsai_observe_set_enabled(1). The samples are not read
 * by plan creation or kernel selection.
 *
 * kernel_count and kernel_ns copy the execute sample only. prepare and
 * plan_create stay in their own fields.
 */

enum {
    XSAI_OBS_PREPARE = 0,
    XSAI_OBS_PLAN_CREATE = 1,
    XSAI_OBS_WORKSPACE = 2,
    XSAI_OBS_EXECUTE = 3,
};

typedef struct xsai_observe_export {
    uint64_t prepare_count;
    uint64_t prepare_ns;
    uint64_t plan_create_count;
    uint64_t plan_create_ns;
    uint64_t workspace_count;
    uint64_t workspace_ns;
    uint64_t execute_count;
    uint64_t execute_ns;
    uint64_t kernel_count;
    uint64_t kernel_ns;
} xsai_observe_export_t;

#ifdef __cplusplus
extern "C" {
#endif

void xsai_observe_set_enabled(int enabled);
int xsai_observe_enabled(void);
void xsai_observe_reset(void);
void xsai_observe_export(xsai_observe_export_t * out);

/* No-op while observation is disabled. token comes back as UINT64_MAX. */
void xsai_observe_span_begin(int phase, uint64_t * token);
void xsai_observe_span_end(int phase, uint64_t token);

#ifdef __cplusplus
}
#endif
