#include "ame.h"

#include <stdio.h>

static void ame_probe_emit_fp2pack4_once(void) {
    static volatile int emitted = 0;
    if (__sync_bool_compare_and_swap(&emitted, 0, 1)) {
        fprintf(stderr, "[AME-PROBE] AME native int8 x fp2pack4 tile kernel executed\n");
        fflush(stderr);
    }
}

/*
 * ecall73/NEMU feat-bitnet defines FP2PACK4 as table-0 type-code 13.  The
 * current assembler does not yet expose msetcfg, so encode it directly:
 *   msetcfg mcfg[rd], rs1
 * with rs1 fixed to t0.  This sequence configures tr0=INT8, tr1=FP2PACK4,
 * and acc0=INT32.  Keep this in the native opt-in kernel: QEMU's older AME
 * model does not implement FP2PACK4.
 */
static inline void ame_configure_i8_fp2pack4_i32(void) {
    asm volatile(
        "li t0, 2\n"          // MTYPECODE_INT8
        ".word 0x4202802b\n"  // msetcfg tr0, t0
        "li t0, 13\n"         // MTYPECODE_FP2PACK4
        ".word 0x420280ab\n"  // msetcfg tr1, t0
        "li t0, 4\n"          // MTYPECODE_INT32
        ".word 0x4202822b\n"  // msetcfg acc0, t0
        ::: "t0", "memory");
}

/*
 * NEMU feat-bitnet uses the current AME opcode subgroups (funct3=4 for
 * sync, 1 for loads/stores, 2 for MMA, and 3 for matrix misc).  LLVM's AME
 * mnemonics still emit the older funct3=0 forms, so this opt-in NEMU kernel
 * carries the matching raw instructions.  FP2PACK4 has no QEMU support, so
 * this does not change the QEMU-compatible AME kernels.
 */
static inline void ame_nemu_msyncreset_tok0(void) {
    asm volatile(".word 0x8000402b" ::: "memory");
}

static inline void ame_nemu_mzero_acc0(void) {
    asm volatile(".word 0x0000322b" ::: "memory");
}

static inline void ame_nemu_mlae8_tr0(const int8_t * a) {
    register const int8_t * a0 asm("a0") = a;
    register int a3 asm("a3") = AME_I2_NATIVE_TILE_K;
    asm volatile(".word 0x00d5102b" :: "r"(a0), "r"(a3) : "memory");
}

static inline void ame_nemu_mlbe8_tr1(const uint8_t * b) {
    register const uint8_t * a1 asm("a1") = b;
    register int a5 asm("a5") = AME_I2_NATIVE_TILE_K / 4;
    asm volatile(".word 0x10f590ab" :: "r"(a1), "r"(a5) : "memory");
}

static inline void ame_nemu_mqma_acc0_tr0_tr1(void) {
    asm volatile(".word 0x0010222b" ::: "memory");
}

static inline void ame_nemu_msce32_acc0(int32_t * c) {
    register int32_t * a2 asm("a2") = c;
    register int a0 asm("a0") = AME_I2_NATIVE_TILE_N * (int) sizeof(*c);
    asm volatile(".word 0x22a6122b" :: "r"(a2), "r"(a0) : "memory");
}

static inline void ame_nemu_mrelease_tok0(void) {
    asm volatile(".word 0x9000402b" ::: "memory");
}

static inline void ame_nemu_macquire_tok0(int value) {
    register int a0 asm("a0") = value;
    asm volatile(".word 0xa005402b" :: "r"(a0) : "memory");
}

void ggml_ame_gemm_tile_i8_i32_fp2pack4_bT(
    const int8_t * A,
    const uint8_t * B,
    int32_t * C
) {
    ame_nemu_msyncreset_tok0();

    int tmp;
    MSETTILEM(tmp, AME_I2_NATIVE_TILE_M);
    MSETTILEK(tmp, AME_I2_NATIVE_TILE_K);
    MSETTILEN(tmp, AME_I2_NATIVE_TILE_N);
    ame_configure_i8_fp2pack4_i32();

    ame_nemu_mzero_acc0();
    asm volatile("fence rw, rw" ::: "memory");

    // NEMU's FP2PACK4 B load requires exactly 16 bytes for its 64 logical K
    // elements.  The operand is already arranged as [128 output rows][16].
    ame_nemu_mlae8_tr0(A);
    ame_nemu_mlbe8_tr1(B);

    ame_nemu_mqma_acc0_tr0_tr1();
    ame_probe_emit_fp2pack4_once();

    ame_nemu_msce32_acc0(C);

    ame_nemu_mrelease_tok0();
    ame_nemu_macquire_tok0(1);
}
