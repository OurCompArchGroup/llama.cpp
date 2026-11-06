#pragma once

#include <xsai_blas.h>

#include <stddef.h>
#include <stdint.h>

/*
 * Stable identity of one GEMM plan over a Q8_0-derived I8_GROUP32 weight.
 * The cache identity is the template call fingerprint: it covers the
 * operator, both counts, every tensor's metadata, the params, the prepared
 * weight fingerprint, and the dispatch options, while ignoring data pointers
 * and workspace. The remaining fields rebuild the execute-time call and guard
 * the fingerprint against a collision.
 */
struct xsai_q8_key {
    uint64_t fingerprint;
    xsai_op_type_t op;      /* XSAI_OP_GEMM */
    int64_t m;              /* 1 for the vector form */
    int64_t k;
    int64_t n;
    int64_t blocks;         /* ceil(k / 32) */
    uint64_t weight_generation;
    /* Tensor whose converted bytes this plan executes. The adapter owns the
     * payload and scale storage; prepared_weight is the xsai handle, not a
     * copy of the bytes. */
    const void * weight_identity;
    const xsai_weight_plan_t * prepared_weight;
};

struct xsai_q8_bind {
    const void * activation;
    void * output;
    void * workspace;
    size_t workspace_size;
};

/*
 * One call's tensor metadata. Shape and stride arrays are embedded so the
 * structure owns every pointer a DLTensor borrows for the duration of one
 * library call.
 */
struct xsai_q8_call {
    DLTensor inputs[3];
    DLTensor output;
    int64_t a_shape[2];
    int64_t b_shape[2];
    int64_t b_strides[2];
    int64_t scale_shape[2];
    int64_t scale_strides[2];
    int64_t c_shape[2];
    xsai_gemm_params_t params;
    xsai_op_call_t call;
};

/* Formats the template call for key and stores its fingerprint. Returns
 * false when the key cannot describe a call the library would fingerprint. */
bool xsai_q8_key_seal(xsai_q8_key * key);

/* Fills parts for key. bind may be null at plan-create time; the prepared
 * weight then supplies the payload and scale tensors at execute. */
void xsai_q8_call_init(const xsai_q8_key * key, const xsai_q8_bind * bind,
                       xsai_q8_call * parts);

xsai_op_plan_t * xsai_q8_cache_get_or_create(const void * op,
                                             const xsai_q8_key * key,
                                             xsai_status_t * status_out = nullptr);
xsai_op_plan_t * xsai_q8_cache_lookup(const void * op,
                                      const xsai_q8_key * key);
xsai_status_t xsai_q8_cache_execute(xsai_op_plan_t * plan,
                                    const xsai_q8_key * key,
                                    const xsai_q8_bind * bind);

/* Op type of the plan cached for op, or -1. Diagnostic and test hook. */
xsai_op_type_t xsai_plan_cached_op(const void * op);

/* Reads the frozen workspace record. Does not select a kernel. */
xsai_status_t xsai_plan_read_requirements(
    const xsai_op_plan_t * plan,
    xsai_op_requirements_t * requirements);

uint64_t xsai_plan_create_count(void);
uint64_t xsai_plan_cache_hit_count(void);
/* plan_create calls. Equals key misses: a hit does not create. */
uint64_t xsai_plan_key_miss_count(void);
/* Successful plan_execute calls. Larger than create when a plan is reused. */
uint64_t xsai_plan_execute_count(void);
/* Sum of prepared-weight arena bytes still owned by the cache. */
uint64_t xsai_prepared_bytes(void);
/* High-water mark of the same sum; survives weights being released. */
uint64_t xsai_prepared_peak_bytes(void);
void xsai_plan_cache_reset_counts(void);
void xsai_plan_cache_clear(void);
size_t xsai_plan_cache_live(void);

/*
 * Q8_0 bytes -> I8_GROUP32 weight plan. Load-time conversion copies the int8
 * codes and the fp16 block scales verbatim into adapter-owned [N, K] and
 * [N, ceil(K/32)] buffers; nothing is requantized. Retires op plans for this
 * tensor, waits until their execute calls return, destroys those op plans,
 * destroys the previous weight plan, then prepares a new one. The arena and
 * the converted buffers are owned here. xsai destroy does not free them.
 */
xsai_status_t xsai_weight_refresh_q8(
    const void * tensor,
    const void * q8_data,
    int64_t rows,
    int64_t logical_k,
    uint64_t generation);
void xsai_weight_release(const void * tensor);
const xsai_weight_plan_t * xsai_weight_plan(const void * tensor);
uint64_t xsai_weight_generation(const void * tensor);

uint64_t xsai_op_plan_destroy_count(void);
uint64_t xsai_weight_plan_destroy_count(void);
int xsai_destroy_order_ok(void);
void xsai_destroy_order_reset(void);

#ifdef __cplusplus
extern "C" {
#endif

void xsai_plan_cache_retain(const void * const * ops, size_t count);

#ifdef __cplusplus
}
#endif

/* Same cache as the GEMM entries. The key is the template call
 * fingerprint. call_template is read only during plan_create. Data pointers
 * may be null. */
xsai_op_plan_t * xsai_general_cache_get_or_create(const void * op,
                                                  uint64_t key,
                                                  const xsai_op_call_t * call_template,
                                                  xsai_status_t * status_out = nullptr);
xsai_op_plan_t * xsai_general_cache_lookup(const void * op, uint64_t key);
xsai_status_t xsai_general_cache_execute(xsai_op_plan_t * plan,
                                         const xsai_op_call_t * call);
