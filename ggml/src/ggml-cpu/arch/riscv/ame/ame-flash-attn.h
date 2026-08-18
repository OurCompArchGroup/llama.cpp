#pragma once

#include "ggml.h"

#include <stddef.h>

struct ggml_compute_params;

#ifdef __cplusplus
extern "C" {
#endif

size_t ggml_ame_flash_attn_ext_work_size(const struct ggml_tensor * dst);

// Returns non-zero when the operation was handled by the AME implementation.
int ggml_ame_flash_attn_ext_compute(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
