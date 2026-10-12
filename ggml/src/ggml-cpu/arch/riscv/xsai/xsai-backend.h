#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

struct ggml_compute_params;

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_xsai_buffer_type(void);

/* Graph planning notifies the adapter. The pointer is not copied. */
void xsai_backend_note_graph(const struct ggml_cgraph * graph);

/* Drop cached plans whose nodes are not in this graph. */
void ame_xsai_plan_cache_retain(const void * const * ops, size_t count);

/* One-shot route, plan, and observe summary. Safe to call more than once. */
void xsai_backend_print_report(void);

void xsai_backend_q8_refresh(const struct ggml_tensor * tensor, uint64_t generation);
void xsai_backend_q8_release(const struct ggml_tensor * tensor);
uint64_t xsai_backend_host_generation(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
bool xsai_backend_q8_work_size(const struct ggml_tensor * op, uint64_t weight_generation, size_t & size);
bool xsai_backend_q8_compute(struct ggml_compute_params * params, struct ggml_tensor * op, uint64_t weight_generation);
bool xsai_backend_work_size(int n_threads, const struct ggml_tensor * op, size_t & size);
bool xsai_backend_compute(struct ggml_compute_params * params, struct ggml_tensor * op);
#endif
