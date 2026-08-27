#ifndef GGML_RISCV_AME_MEM_TRACE_H
#define GGML_RISCV_AME_MEM_TRACE_H

#include "ggml.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// NEMU memory-trace protocol. Keep these values aligned with NEMU's special
// instruction handler without depending on NEMU-private headers.
#define GGML_AME_MEM_TRACE_BEGIN 0x103
#define GGML_AME_MEM_TRACE_END   0x104

static inline const char * ggml_ame_mem_trace_target_name(void) {
    const char * name = getenv("GGML_XSAI_MEM_TRACE_NAME");
    return name != NULL && name[0] != '\0' ? name : "Qcur";
}

static inline const char * ggml_ame_mem_trace_target_op(void) {
    const char * op = getenv("GGML_XSAI_MEM_TRACE_OP");
    return op != NULL && op[0] != '\0' ? op : "MUL_MAT";
}

static inline int ggml_ame_mem_trace_target_layer(void) {
    const char * value = getenv("GGML_XSAI_MEM_TRACE_LAYER");
    if (value == NULL || value[0] == '\0') {
        return -1;
    }

    char * end = NULL;
    const long layer = strtol(value, &end, 10);
    return end != value && *end == '\0' && layer >= 0 && layer <= 0x7fffffff ? (int) layer : -1;
}

static inline bool ggml_ame_mem_trace_name_matches(const char * tensor_name) {
    const char * target = ggml_ame_mem_trace_target_name();
    const int layer = ggml_ame_mem_trace_target_layer();

    if (layer >= 0) {
        const size_t target_len = strlen(target);
        const size_t tensor_len = strlen(tensor_name);
        if (tensor_len <= target_len + 1 || strncmp(tensor_name, target, target_len) != 0 ||
                tensor_name[target_len] != '-') {
            return false;
        }

        char * end = NULL;
        const long parsed = strtol(tensor_name + target_len + 1, &end, 10);
        return end != tensor_name + target_len + 1 && *end == '\0' && parsed == layer;
    }

    if (strcmp(tensor_name, target) == 0) {
        return true;
    }

    const size_t target_len = strlen(target);
    if (strncmp(tensor_name, target, target_len) != 0 || tensor_name[target_len] != '-') {
        return false;
    }

    const char * suffix = tensor_name + target_len + 1;
    if (*suffix == '\0') {
        return false;
    }
    for (; *suffix != '\0'; ++suffix) {
        if (*suffix < '0' || *suffix > '9') {
            return false;
        }
    }
    return true;
}

static inline bool ggml_ame_mem_trace_is_target(const struct ggml_tensor * tensor) {
    return tensor != NULL &&
        (strcmp(ggml_ame_mem_trace_target_op(), "*") == 0 ||
         strcmp(ggml_op_name(tensor->op), ggml_ame_mem_trace_target_op()) == 0) &&
        ggml_ame_mem_trace_name_matches(tensor->name);
}

static inline void ggml_ame_mem_trace_signal(int signal_id) {
#if defined(__riscv)
    __asm__ __volatile__(
        "mv a0, %0\n\t"
        ".insn r 0x6B, 0, 0, x0, x0, x0\n\t"
        :
        : "r"(signal_id)
        : "a0", "memory");
#else
    (void) signal_id;
#endif
}

#endif // GGML_RISCV_AME_MEM_TRACE_H
