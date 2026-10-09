#pragma once

#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv_ame_buffer_type(void);

/* Drop cached xsai plans whose ggml nodes are not in this graph.
 * The cplan itself stores only scratch, never an xsai plan. */
void ame_xsai_plan_cache_retain(const void * const * ops, size_t count);

#ifdef __cplusplus
}
#endif
