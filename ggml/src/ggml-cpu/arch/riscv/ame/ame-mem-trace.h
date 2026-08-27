#ifndef GGML_RISCV_AME_MEM_TRACE_H
#define GGML_RISCV_AME_MEM_TRACE_H

#include "ggml.h"

#include <stdbool.h>
#include <string.h>

// NEMU memory-trace protocol. Keep these values aligned with NEMU's special
// instruction handler without depending on NEMU-private headers.
#define GGML_AME_MEM_TRACE_BEGIN 0x103
#define GGML_AME_MEM_TRACE_END   0x104

static inline bool ggml_ame_mem_trace_is_q_projection(const struct ggml_tensor * tensor) {
    return tensor != NULL && tensor->op == GGML_OP_MUL_MAT &&
        (strcmp(tensor->name, "Qcur") == 0 || strncmp(tensor->name, "Qcur-", 5) == 0);
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
