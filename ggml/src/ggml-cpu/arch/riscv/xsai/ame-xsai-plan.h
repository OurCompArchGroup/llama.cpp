#pragma once

#include <xsai_blas.h>

#include <stddef.h>
#include <stdint.h>

/*
 * Stable identity of one Q8 GEMM plan. Data pointers are deliberately absent:
 * refreshing them reuses the plan. Shape, stride, batch rank, weight
 * generation, and explicit options are part of the key, so a change destroys
 * the previous plan and creates another one.
 */
struct ame_xsai_q8_key {
    int64_t m;
    int64_t n;
    int64_t k;
    uint64_t act_nb0;
    uint64_t act_nb1;
    uint64_t act_nb2;
    uint64_t act_nb3;
    uint64_t weight_nb0;
    uint64_t weight_nb1;
    uint64_t weight_nb2;
    uint64_t weight_nb3;
    uint64_t dst_nb0;
    uint64_t dst_nb1;
    uint64_t dst_nb2;
    uint64_t dst_nb3;
    int64_t src0_ne2;
    int64_t src0_ne3;
    int64_t src1_ne2;
    int64_t src1_ne3;
    int64_t dst_ne2;
    int64_t dst_ne3;
    uint64_t weight_generation;
    uint32_t explicit_options;
    uint32_t reserved;
    /* Tensor whose bytes this plan executes. NULL for callers that do not
     * own a prepared weight. prepared_weight is the xsai handle, not a
     * copy of the bytes. */
    const void * weight_identity;
    const xsai_weight_plan_t * prepared_weight;
};

struct ame_xsai_q8_bind {
    const void * activation;
    const void * weight;
    void * output;
    void * workspace;
    size_t workspace_size;
};

xsai_op_plan_t * ame_xsai_q8_cache_get_or_create(const void * op,
                                                 const ame_xsai_q8_key * key);
xsai_op_plan_t * ame_xsai_q8_cache_lookup(const void * op,
                                          const ame_xsai_q8_key * key);
xsai_status_t ame_xsai_q8_cache_execute(xsai_op_plan_t * plan,
                                        const ame_xsai_q8_key * key,
                                        const ame_xsai_q8_bind * bind);

/* Reads the frozen workspace record. Does not select a kernel. */
xsai_status_t ame_xsai_plan_read_requirements(
    const xsai_op_plan_t * plan,
    xsai_op_requirements_t * requirements);

uint64_t ame_xsai_plan_create_count(void);
uint64_t ame_xsai_plan_cache_hit_count(void);
/* plan_create calls. Equals key misses: a hit does not create. */
uint64_t ame_xsai_plan_key_miss_count(void);
/* Successful plan_execute calls. Larger than create when a plan is reused. */
uint64_t ame_xsai_plan_execute_count(void);
/* Sum of prepared-weight arena bytes still owned by the cache. */
uint64_t ame_xsai_prepared_bytes(void);
void ame_xsai_plan_cache_reset_counts(void);
void ame_xsai_plan_cache_clear(void);
size_t ame_xsai_plan_cache_live(void);

/* Q8_0 bytes -> I8_GROUP32 weight plan. Retires op plans for this tensor,
 * waits until their execute calls return, destroys those op plans, destroys
 * the previous weight plan, then prepares a new one. The arena is owned here.
 * xsai destroy does not free it. */
xsai_status_t ame_xsai_weight_refresh_q8(
    const void * tensor,
    const void * q8_data,
    int64_t rows,
    int64_t logical_k,
    uint64_t generation);
void ame_xsai_weight_release(const void * tensor);
const xsai_weight_plan_t * ame_xsai_weight_plan(const void * tensor);
uint64_t ame_xsai_weight_generation(const void * tensor);

uint64_t ame_xsai_op_plan_destroy_count(void);
uint64_t ame_xsai_weight_plan_destroy_count(void);
int ame_xsai_destroy_order_ok(void);
void ame_xsai_destroy_order_reset(void);

#ifdef __cplusplus
extern "C" {
#endif

void ame_xsai_plan_cache_retain(const void * const * ops, size_t count);

#ifdef __cplusplus
}
#endif

/* Same cache as the Q8 entries. The key bytes are the miss identity.
 * call_template is read only during plan_create. Data pointers may be null. */
xsai_op_plan_t * ame_xsai_general_cache_get_or_create(const void * op,
                                                      const void * key,
                                                      size_t key_nbytes,
                                                      const xsai_op_call_t * call_template);
xsai_op_plan_t * ame_xsai_general_cache_lookup(const void * op,
                                               const void * key,
                                               size_t key_nbytes);
xsai_status_t ame_xsai_general_cache_execute(xsai_op_plan_t * plan,
                                             const xsai_op_call_t * call);
