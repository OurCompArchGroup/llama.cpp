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

uint64_t ame_xsai_plan_create_count(void);
uint64_t ame_xsai_plan_cache_hit_count(void);
void ame_xsai_plan_cache_reset_counts(void);
void ame_xsai_plan_cache_clear(void);
size_t ame_xsai_plan_cache_live(void);

#ifdef __cplusplus
extern "C" {
#endif

void ame_xsai_plan_cache_retain(const void * const * ops, size_t count);

#ifdef __cplusplus
}
#endif
