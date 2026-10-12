#pragma once

#include <stdint.h>

/*
 * Optional counts and monotonic timestamps for the XSAI adapter.
 * Disabled until ame_xsai_observe_set_enabled(1). The samples are not read
 * by plan creation or kernel selection.
 *
 * kernel_count and kernel_ns copy the execute sample only. prepare and
 * plan_create stay in their own fields.
 */

enum {
    AME_XSAI_OBS_PREPARE = 0,
    AME_XSAI_OBS_PLAN_CREATE = 1,
    AME_XSAI_OBS_WORKSPACE = 2,
    AME_XSAI_OBS_EXECUTE = 3,
};

typedef struct ame_xsai_observe_export {
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
} ame_xsai_observe_export_t;

#ifdef __cplusplus
extern "C" {
#endif

void ame_xsai_observe_set_enabled(int enabled);
int ame_xsai_observe_enabled(void);
void ame_xsai_observe_reset(void);
void ame_xsai_observe_export(ame_xsai_observe_export_t * out);

/* No-op while observation is disabled. token comes back as UINT64_MAX. */
void ame_xsai_observe_span_begin(int phase, uint64_t * token);
void ame_xsai_observe_span_end(int phase, uint64_t token);

#ifdef __cplusplus
}
#endif
