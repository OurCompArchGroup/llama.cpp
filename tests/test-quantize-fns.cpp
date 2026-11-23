// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "../ggml/src/ggml-cpu/arch/riscv/ame/ame.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <cfenv>
#include <limits>
#include <string>
#include <vector>

#if defined(__unix__)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};

static void test_ame_q8_k32_packing() {
    unsigned selections = 0;
    for (int tm : {64, 128}) {
        for (const char * knob : {static_cast<const char *>(nullptr), "", "0", "off", "true", "32", "unknown", "1"}) {
            for (int64_t K : {32, 64, 96, 1024}) {
                assert(ggml_ame_q8_baseline_compute_k(3072, 256, K, tm, tm, 1, 0, 0, knob) ==
                    (knob != nullptr && std::strcmp(knob, "1") == 0 ? 32 : 64));
                selections++;
            }
        }
        for (const auto & dims : std::vector<std::vector<int64_t>>{
                {63, 256, 1024}, {3072, 65, 1024}, {3072, 256, 33}, {0, 256, 1024}, {3072, 256, 0}}) {
            assert(ggml_ame_q8_baseline_compute_k(dims[0], dims[1], dims[2], tm, tm, 1, 0, 0, "1") == 64);
            selections++;
        }
        for (int packed : {0, 1}) for (int whole : {0, 1}) for (int nth : {1, 2}) {
            assert(ggml_ame_q8_baseline_compute_k(3072, 256, 1024, tm, tm, nth, packed, whole, "1") ==
                (!packed && !whole && nth == 1 ? 32 : 64));
            selections++;
        }
    }
    assert(ggml_ame_q8_baseline_compute_k(3072, 256, 1024, 256, 256, 1, 0, 0, "1") == 64);
    assert(ggml_ame_q8_baseline_compute_k(3072, 256, INT64_MAX, 128, 128, 1, 0, 0, "1") == 64);
    selections += 2;
    uint32_t random = 0x513729ab;
    float old_accumulator = 0.0f, new_accumulator = 0.0f;
    for (int group = 0; group < 256; ++group) {
        float a[32], b[32];
        for (int k = 0; k < 32; ++k) {
            random = random * 1664525u + 1013904223u;
            a[k] = std::ldexp((float) ((int) (random % 255) - 127), group % 9 - 9);
            random = random * 1664525u + 1013904223u;
            b[k] = std::ldexp((float) ((int) (random % 255) - 127), group % 7 - 8);
        }
        block_q8_0 qa, qb;
        ggml_ame_quantize_row_f32_to_q8_0(a, &qa, 32);
        ggml_ame_quantize_row_f32_to_q8_0(b, &qb, 32);
        if (group == 0) std::memset(qa.qs, 0, sizeof(qa.qs));
        if (group == 1) { qa.qs[0] = -128; qb.qs[0] = 127; }
        int8_t padded_a[64] = {}, padded_b[64] = {};
        std::memcpy(padded_a, qa.qs, 32);
        std::memcpy(padded_b, qb.qs, 32);
        int32_t old_dot = 0, new_dot = 0;
        for (int k = 0; k < 64; ++k) old_dot += padded_a[k] * padded_b[k];
        for (int k = 0; k < 32; ++k) new_dot += qa.qs[k] * qb.qs[k];
        assert(old_dot == new_dot);
        const float scale = ggml_fp16_to_fp32(qa.d) * ggml_fp16_to_fp32(qb.d);
        old_accumulator += old_dot * scale;
        new_accumulator += new_dot * scale;
        assert(std::memcmp(&old_accumulator, &new_accumulator, sizeof(float)) == 0);
    }
    assert(selections == 92 && std::isfinite(new_accumulator) && new_accumulator != 0.0f);
    printf("AME_Q8_K32_PACK_PASS groups=256 selectors=%u group_k=32 row_stride=64 "
        "padding=zero quantizer=unchanged dot_i32=exact scale_accumulator=bit_identical "
        "unknown_edge_T2_packed_whole=fallback actual_native_mmacc=0\n", selections);
}

// Independent bit-field decoder: no production LUT or E8M0 helper. This is a
// packing oracle, not a claim that native hardware satisfies FP32 MX reduction.
static float ame_test_e2m1(uint8_t code, uint8_t scale) {
    if (scale == 255) {
        return NAN;
    }
    const int exponent = (code >> 1) & 3;
    const int mantissa = code & 1;
    float value = exponent == 0 ? mantissa * 0.5f :
        std::ldexp(1.0f + mantissa * 0.5f, exponent - 1);
    value = std::ldexp(value, (int) scale - 127);
    return code & 8 ? -value : value;
}

static void test_mxfp4_quant_weights() {
    constexpr int rows = 3;
    constexpr int columns = 64;
    constexpr int blocks = rows * columns / QK_MXFP4;
    const size_t bytes = rows * ggml_row_size(GGML_TYPE_MXFP4, columns);
    std::vector<float> input(rows * columns, 0.0f);
    std::vector<float> weights(columns, 1.0f);
    // The same column weights apply to every row, not to a packed flat tensor.
    // One block has an unimportant outlier; the next must retain its outlier.
    weights[0] = 0.0f;
    weights[32] = 1000.0f;
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < columns; ++column) {
            input[row * columns + column] =
                (row ? -1.0f : 1.0f) * (column % QK_MXFP4 == 0 ? 64.0f : 2.0f);
        }
    }
    const auto original_input = input;
    const auto original_weights = weights;
    std::vector<block_mxfp4> reference(blocks);
    quantize_row_mxfp4_ref(input.data(), reference.data(), rows * columns);
    std::vector<uint8_t> encoded(bytes + 2, 0xa5);
    assert(ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), encoded.data() + 1,
        0, rows, columns, nullptr) == bytes);
    assert(std::memcmp(encoded.data() + 1, reference.data(), bytes) == 0);
    assert(encoded.front() == 0xa5 && encoded.back() == 0xa5);

    assert(ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), encoded.data() + 1,
        0, rows, columns, weights.data()) == bytes);
    const auto * weighted = reinterpret_cast<const block_mxfp4 *>(encoded.data() + 1);
    double reference_error = 0.0;
    double weighted_error = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const int index = row * columns + column;
            const int lane = column % QK_MXFP4;
            const int block = index / QK_MXFP4;
            assert(weighted[block].e < 255);
            const int shift = lane < QK_MXFP4 / 2 ? 0 : 4;
            const int slot = lane % (QK_MXFP4 / 2);
            const float actual = ame_test_e2m1(
                (weighted[block].qs[slot] >> shift) & 15, weighted[block].e);
            const float before = ame_test_e2m1(
                (reference[block].qs[slot] >> shift) & 15, reference[block].e);
            const double delta = double(actual) - input[index];
            const double old_delta = double(before) - input[index];
            weighted_error += weights[column] * delta * delta;
            reference_error += weights[column] * old_delta * old_delta;
        }
    }
    assert(weighted_error < reference_error);
    assert(encoded.front() == 0xa5 && encoded.back() == 0xa5);
    assert(input == original_input && weights == original_weights);

    std::vector<uint8_t> chunked(bytes + 2, 0xa5);
    for (int row = 0; row < rows; ++row) {
        assert(ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), chunked.data() + 1,
            row * columns, 1, columns, weights.data()) == bytes / rows);
    }
    assert(chunked == encoded);
    std::fill(weights.begin(), weights.end(), 0.0f);
    assert(ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), encoded.data() + 1,
        0, rows, columns, weights.data()) == bytes);
    assert(std::memcmp(encoded.data() + 1, reference.data(), bytes) == 0);

    uint32_t random = 0x6397ac51;
    unsigned checked_blocks = 0;
    for (int trial = 0; trial < 128; ++trial) {
        for (int column = 0; column < columns; ++column) {
            random = random * 1664525u + 1013904223u;
            weights[column] = column < QK_MXFP4 ? 0.0f :
                std::ldexp(float(1 + random % 16), int(random % 13) - 6);
        }
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; ++column) {
                random = random * 1664525u + 1013904223u;
                input[row * columns + column] = std::ldexp(
                    float(int(random % 65) - 32), trial % 81 - 40);
            }
        }
        quantize_row_mxfp4_ref(input.data(), reference.data(), rows * columns);
        assert(ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), encoded.data() + 1,
            0, rows, columns, weights.data()) == bytes);
        const auto * result = reinterpret_cast<const block_mxfp4 *>(encoded.data() + 1);
        for (int block = 0; block < blocks; ++block) {
            double before = 0.0, after = 0.0;
            assert(result[block].e < 255);
            for (int lane = 0; lane < QK_MXFP4; ++lane) {
                const int index = block * QK_MXFP4 + lane;
                const int shift = lane < QK_MXFP4 / 2 ? 0 : 4;
                const int slot = lane % (QK_MXFP4 / 2);
                const double old_value = ame_test_e2m1(
                    (reference[block].qs[slot] >> shift) & 15, reference[block].e);
                const double new_value = ame_test_e2m1(
                    (result[block].qs[slot] >> shift) & 15, result[block].e);
                const double delta0 = old_value - input[index];
                const double delta1 = new_value - input[index];
                before += weights[index % columns] * delta0 * delta0;
                after += weights[index % columns] * delta1 * delta1;
            }
            assert(std::isfinite(after) && after <= before);
            if (block % (columns / QK_MXFP4) == 0) {
                assert(std::memcmp(&result[block], &reference[block], sizeof(block_mxfp4)) == 0);
            }
            ++checked_blocks;
        }
        for (int row = 0; row < rows; ++row) {
            assert(ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), chunked.data() + 1,
                row * columns, 1, columns, weights.data()) == bytes / rows);
        }
        assert(chunked == encoded && encoded.front() == 0xa5 && encoded.back() == 0xa5);
    }
    // Exercise the legal reference exponent endpoints without changing the
    // no-calibration quantizer's existing behavior.
    for (int exponent : {-125, 127}) {
        std::fill(input.begin(), input.end(), std::ldexp(1.0f, exponent));
        std::fill(weights.begin(), weights.end(), 1.0f);
        assert(ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), encoded.data() + 1,
            0, rows, columns, weights.data()) == bytes);
        const auto * result = reinterpret_cast<const block_mxfp4 *>(encoded.data() + 1);
        for (int block = 0; block < blocks; ++block) {
            assert(result[block].e < 255);
            for (int lane = 0; lane < QK_MXFP4; ++lane) {
                const int shift = lane < QK_MXFP4 / 2 ? 0 : 4;
                const int slot = lane % (QK_MXFP4 / 2);
                assert(ame_test_e2m1((result[block].qs[slot] >> shift) & 15,
                    result[block].e) == input[block * QK_MXFP4 + lane]);
            }
        }
    }
    unsigned rejected_inputs = 0;
#if defined(__unix__)
    // The CLI validates imatrix files too, but the low-level optional path
    // must reject invalid calibration/source values instead of emitting bytes.
    for (int bad = 0; bad < 7; ++bad) {
        std::fill(input.begin(), input.end(), 1.0f);
        std::fill(weights.begin(), weights.end(), 1.0f);
        if (bad == 0) weights[0] = -1.0f;
        if (bad == 1) weights[0] = std::numeric_limits<float>::quiet_NaN();
        if (bad == 2) weights[0] = std::numeric_limits<float>::infinity();
        if (bad == 3) input[columns + 1] = std::numeric_limits<float>::quiet_NaN();
        if (bad == 4) input[columns + 1] = std::numeric_limits<float>::infinity();
        if (bad == 5) std::fill(input.begin(), input.end(), std::numeric_limits<float>::denorm_min());
        std::fflush(nullptr);
        const pid_t child = fork();
        assert(child >= 0);
        if (child == 0) {
            if (setenv("GGML_NO_BACKTRACE", "1", 1) != 0) _exit(2);
            if (!std::freopen("/dev/null", "w", stderr)) _exit(2);
            if (bad == 6) {
                quantize_mxfp4(input.data(), encoded.data() + 1, 2, 16, weights.data());
            } else {
                ggml_quantize_chunk(GGML_TYPE_MXFP4, input.data(), encoded.data() + 1,
                    0, rows, columns, weights.data());
            }
            _exit(0);
        }
        int status = 0;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
        ++rejected_inputs;
    }
#endif
    printf("MXFP4_IMATRIX_QUANT_PASS rows=%d columns=%d reference_error=%.9g "
        "weighted_error=%.9g null_zero_weights=legacy_exact chunking=exact "
        "canaries=pass random_blocks=%u invalid_inputs_rejected=%u "
        "hardware=not-run model_quality=not-run\n",
        rows, columns, reference_error, weighted_error, checked_blocks, rejected_inputs);
}

#if defined(GGML_RV_MXFP4_ENCODE)
// Direct ISA gate in the existing quantization fixture, independent of CPU
// scale selection. The strict16 Double oracle is not the hardware comparator.
static uint32_t mxfp4_vector_oracle(uint32_t bits, uint32_t e, bool & invalid) {
    invalid = e > 251 || (bits & 0x7f800000u) == 0x7f800000u;
    if (invalid) return 0;
    float x;
    std::memcpy(&x, &bits, sizeof(x));
    const double d = std::ldexp(1.0, int(e) - 128);
    invalid = std::fabs(double(x)) > 16.0 * d;
    if (invalid) return 0;
    const int centers[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    double best = std::numeric_limits<double>::infinity();
    uint32_t winner = 0;
    for (uint32_t code = 0; code < 16; ++code) {
        const double error = std::fabs(double(x) - centers[code] * d);
        if (error < best) { best = error; winner = code; }
    }
    return winner;
}

static __attribute__((noinline)) void mxfp4_vector_probe(
    const uint32_t * values, const uint32_t * scales, const uint32_t * old,
    uint32_t * result, unsigned vl, uint8_t mask, bool vm, int alias,
    unsigned seed, unsigned & flags, unsigned & rounding
) {
    // Explicitly set the complete vector state on each call. Early-clobber
    // scalar outputs must not alias pointers consumed by the final store.
#define MXFP4_VECTOR_PROBE(FUNCT7, DEST) \
    __asm__ volatile ( \
        ".option push\n\t.option arch, +v\n\t" \
        "vsetivli zero, 4, e32, m1, tu, mu\n\t" \
        "vle32.v v8, (%[x])\n\tvle32.v v12, (%[e])\n\t" \
        "vle32.v v16, (%[old])\n\tvlm.v v0, (%[mask])\n\t" \
        "csrw fflags, %[seed]\n\t" \
        "vsetvli zero, %[vl], e32, m1, tu, mu\n\t" \
        ".insn r 0x7b, 1, " FUNCT7 ", " DEST ", v12, v8\n\t" \
        "frflags %[flags]\n\tfrrm %[rounding]\n\t" \
        "vsetivli zero, 4, e32, m1, tu, mu\n\tvse32.v " DEST ", (%[out])\n\t.option pop" \
        : [flags] "=&r" (flags), [rounding] "=&r" (rounding) \
        : [x] "r" (values), [e] "r" (scales), [old] "r" (old), \
          [mask] "r" (&mask), [seed] "r" (seed), [vl] "r" (vl), [out] "r" (result) \
        : "v0", "v8", "v12", "v16", "memory")
    if (vm) {
        if (alias == 1) { MXFP4_VECTOR_PROBE("3", "v8"); }
        else if (alias == 2) { MXFP4_VECTOR_PROBE("3", "v12"); }
        else { MXFP4_VECTOR_PROBE("3", "v16"); }
    } else {
        if (alias == 1) { MXFP4_VECTOR_PROBE("2", "v8"); }
        else if (alias == 2) { MXFP4_VECTOR_PROBE("2", "v12"); }
        else { MXFP4_VECTOR_PROBE("2", "v16"); }
    }
#undef MXFP4_VECTOR_PROBE
}

static void test_mxfp4_vector_encode() {
    unsigned saved_fcsr;
    __asm__ volatile ("frcsr %0" : "=r" (saved_fcsr));
    unsigned cases = 0;
    const auto check = [&](const std::array<uint32_t, 4> & values,
                           const std::array<uint32_t, 4> & scales,
                           unsigned vl, uint8_t mask, bool vm, int alias, unsigned seed) {
        const std::array<uint32_t, 4> old = {0xa5a50101, 0xa5a50202, 0xa5a50303, 0xa5a50404};
        std::array<uint32_t, 6> result = {0x19631751, 0, 0, 0, 0, 0x75131963};
        std::array<uint32_t, 4> expected = alias == 1 ? values : alias == 2 ? scales : old;
        unsigned expected_flags = seed;
        for (unsigned lane = 0; lane < vl; ++lane) {
            if (!vm && !(mask & (1u << lane))) continue;
            bool invalid;
            expected[lane] = mxfp4_vector_oracle(values[lane], scales[lane], invalid);
            if (invalid) expected_flags |= 16;
        }
        unsigned before_rounding, flags, rounding;
        __asm__ volatile ("frrm %0" : "=r" (before_rounding));
        mxfp4_vector_probe(values.data(), scales.data(), old.data(), result.data() + 1,
            vl, mask, vm, alias, seed, flags, rounding);
        assert(std::equal(expected.begin(), expected.end(), result.begin() + 1));
        assert(result.front() == 0x19631751 && result.back() == 0x75131963);
        assert(flags == expected_flags && rounding == before_rounding);
        ++cases;
    };
    for (int rm : {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD}) {
        assert(std::fesetround(rm) == 0);
        for (unsigned e = 0; e <= 251; ++e) {
            for (double midpoint : {0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0}) {
                const float tie = float(std::ldexp(midpoint, int(e) - 128));
                for (float value : {std::nextafter(tie, 0.0f), tie,
                                    std::nextafter(tie, std::numeric_limits<float>::infinity())}) {
                    std::array<uint32_t, 4> bits;
                    std::memcpy(&bits[0], &value, sizeof(value));
                    bits[1] = bits[0] | 0x80000000u;
                    bits[2] = 0; bits[3] = 0x80000000u;
                    check(bits, {e, e, e, e}, 4, 15, true, 0, 0);
                }
            }
        }
        for (unsigned seed = 0; seed < 32; ++seed) {
            for (int alias = 0; alias < 3; ++alias) {
                check({0x3fc00000, 0xc0a00000, 0x00000001, 0x7f000000},
                      {128, 129, 0, 251}, 4, 15, true, alias, seed);
                for (unsigned vl : {0u, 1u, 3u, 4u}) {
                    // All bad inputs are either inactive or individually
                    // active; unrelated lanes must not inherit their NV.
                    check({0x3fc00000, 0x7fc12345, 0x7f800001, 0x7f800000},
                          {128, 256, 252, 0xffffffff}, vl, 1, false, alias, seed);
                    check({0x3fc00000, 0x7fc12345, 0x7f800001, 0x7f800000},
                          {128, 256, 252, 0xffffffff}, vl, 15, true, alias, seed);
                }
            }
        }
    }
    __asm__ volatile ("fscsr %0" : : "r" (saved_fcsr) : "memory");
    printf("MXFP4_VECTOR_ENCODE_PASS cases=%u scale_range=0..251 oracle=strict16_double "
           "midpoint_neighbors=exact full_u32_e=checked aliases=3 mask_tail_vl0=checked "
           "incoming_flags=32 frm=4 scalar_flags=exact canaries=exact packed_output=no\n", cases);
}
#endif

static void test_mxfp4_runtime_bytes() {
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    assert(cpu && cpu->from_float);
    std::fenv_t saved_environment;
    assert(std::fegetenv(&saved_environment) == 0);
    unsigned checked_blocks = 0;
    unsigned checked_rows = 0;
    unsigned seeded_rows = 0;
    unsigned amax_rows = 0;
    uint32_t random = 0x6bc913a5;
    const auto seed_flags = [](int flags) {
        assert((flags & ~FE_ALL_EXCEPT) == 0);
        assert(std::feclearexcept(FE_ALL_EXCEPT) == 0);
        assert(std::feraiseexcept(flags) == 0);
        // Some libm implementations also raise NX when asked to raise OF/UF.
        // Seed exactly the requested architectural state, then verify it
        // before either quantizer. This helper is test-only.
        const int extra = std::fetestexcept(FE_ALL_EXCEPT) & ~flags;
        assert(std::feclearexcept(extra) == 0);
        const int observed = std::fetestexcept(FE_ALL_EXCEPT);
        if (observed != flags) {
            fprintf(stderr, "MXFP4_RUNTIME_SEED_MISMATCH expected=%x actual=%x\n", flags, observed);
            assert(false);
        }
    };
    const auto check = [&](const std::vector<float> & values, int initial_flags = 0) {
        assert(!values.empty() && values.size() % QK_MXFP4 == 0);
        // Neither the F32 row nor the byte-aligned block payload requires a
        // vector-aligned address. Preserve source bytes, including NaN payloads.
        std::vector<float> input(values.size() + 2, -123.25f);
        std::memcpy(input.data() + 1, values.data(), values.size() * sizeof(float));
        const auto unchanged = input;
        const size_t bytes = values.size() / QK_MXFP4 * sizeof(block_mxfp4);
        std::vector<uint8_t> expected(bytes + 2, 0xa5);
        std::vector<uint8_t> actual(bytes + 2, 0xa5);
        seed_flags(initial_flags);
        quantize_row_mxfp4_ref(input.data() + 1,
            reinterpret_cast<block_mxfp4 *>(expected.data() + 1), values.size());
        const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
        seed_flags(initial_flags);
        cpu->from_float(input.data() + 1, actual.data() + 1, values.size());
        const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
        if (actual_flags != expected_flags) {
            fprintf(stderr, "MXFP4_RUNTIME_FLAGS_MISMATCH round=%d row=%u seed=%x expected=%x actual=%x\n",
                std::fegetround(), checked_rows, initial_flags, expected_flags, actual_flags);
            assert(false);
        }
        if (actual != expected) {
            const size_t byte = std::mismatch(actual.begin(), actual.end(), expected.begin()).first - actual.begin();
            fprintf(stderr, "MXFP4_RUNTIME_BYTE_MISMATCH round=%d row=%u seed=%x byte=%zu expected=%02x actual=%02x\n",
                std::fegetround(), checked_rows, initial_flags, byte, expected[byte], actual[byte]);
            assert(false);
        }
        assert(std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0);
        checked_blocks += values.size() / QK_MXFP4;
        ++checked_rows;
    };
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        assert(std::fesetround(rounding) == 0);
        uint8_t untouched[] = {0xa5, 0xa5, 0xa5};
        cpu->from_float(nullptr, untouched + 1, 0);
        assert(untouched[0] == 0xa5 && untouched[1] == 0xa5 && untouched[2] == 0xa5);
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<float> values(QK_MXFP4);
            for (int lane = 0; lane < QK_MXFP4; ++lane) {
                values[lane] = pattern == 0 ? 0.0f : pattern == 1 ? -0.0f :
                    pattern == 2 ? (lane & 1 ? -0.0f : 0.0f) :
                    std::ldexp(float(lane - 16), -3);
            }
            check(values);
        }
        // Exercise every finite exponent, including legacy tiny-block fallback
        // and log2f rounding immediately around powers of two.
        for (int exponent = -149; exponent <= 127; ++exponent) {
            std::vector<float> values(3 * QK_MXFP4);
            for (float & value : values) {
                random = random * 1664525u + 1013904223u;
                value = std::ldexp(float(int(random & 0xffff) - 32768), exponent - 15);
            }
            const float anchor = std::ldexp(1.0f, exponent);
            values[0] = anchor;
            values[QK_MXFP4] = std::nextafter(anchor, 0.0f);
            values[2 * QK_MXFP4] = std::nextafter(anchor, INFINITY);
            check(values);
        }
        // Each exact midpoint and its adjacent FP32 values, for both signs.
        // The anchor fixes amax without deriving expected codes from the DUT.
        const float midpoints[] = {0.5f, 1.5f, 2.5f, 3.5f, 5.0f, 7.0f, 10.0f};
        for (int exponent = -125; exponent <= 127; ++exponent) {
            const float d = std::ldexp(1.0f, exponent - 3);
            for (int sign : {-1, 1}) for (int neighbor : {-1, 0, 1}) {
                std::vector<float> values(QK_MXFP4);
                values[0] = 12.0f * d;
                for (int lane = 1; lane < QK_MXFP4; ++lane) {
                    float value = midpoints[(lane - 1) % 7] * d;
                    if (neighbor) value = std::nextafter(value, neighbor < 0 ? 0.0f : INFINITY);
                    values[lane] = sign * value;
                }
                check(values);
                // Explicit NX eligibility at the subnormal-scale endpoints
                // and e251. The signed anchor also covers both saturation
                // directions without depending on log2f to imply a bound.
                if (exponent == -125 || exponent == -124 || exponent == 126) {
                    values[0] = sign * 14.0f * d;
                    check(values, FE_INEXACT);
                    ++seeded_rows;
                }
            }
        }
        // Explicit high-end scale/fallback cases followed by a normal block.
        // Do not rely on a random block to reach log2f's rounding to 128.
        // This also catches a block-local fallback accidentally returning the
        // entire row, and code8 substitutions for reference zero-code ties.
        {
            std::vector<float> values(3 * QK_MXFP4, 0.0f);
            const float largest = std::numeric_limits<float>::max();
            const float before_largest = std::nextafter(largest, 0.0f);
            values[0] = largest;
            values[1] = -largest;
            values[QK_MXFP4] = before_largest;
            values[QK_MXFP4 + 1] = -before_largest;
            const int base = 2 * QK_MXFP4;
            values[base] = 8.0f;
            values[base + 1] = -0.0f;
            values[base + 2] = std::numeric_limits<float>::denorm_min();
            values[base + 3] = -std::numeric_limits<float>::denorm_min();
            values[base + 4] = std::nextafter(std::numeric_limits<float>::min(), 0.0f);
            values[base + 5] = -values[base + 4];
            values[base + 6] = std::numeric_limits<float>::min();
            values[base + 7] = -std::numeric_limits<float>::min();
            values[base + 8] = -0.5f;
            values[base + 9] = 0.5f;
            check(values);
        }
        // Removing opposite-sign errors can preserve every encoded byte while
        // losing accrued overflow/inexact. Keep explicit non-fallback e252
        // blocks, plus a normal-scale rounding case, in the same flags oracle.
        for (float value : {-0x1p127f, 0x1p127f}) {
            check(std::vector<float>(QK_MXFP4, value));
        }
        {
            std::vector<float> values(QK_MXFP4, 0.0f);
            values[0] = 8.0f;
            values[1] = -std::nextafter(8.0f, 0.0f);
            check(values);
        }
        // Mixed finite bit patterns cover large dynamic ranges and subnormal
        // lanes in normal blocks, not just zero-filled fake-model activations.
        for (int trial = 0; trial < 128; ++trial) {
            std::vector<float> values(5 * QK_MXFP4);
            for (float & value : values) {
                random = random * 1664525u + 1013904223u;
                uint32_t bits = random;
                if ((bits & 0x7f800000u) == 0x7f800000u) bits &= 0xff7fffffu;
                std::memcpy(&value, &bits, sizeof(value));
            }
            check(values);
        }
        // A finite anchor makes scale selection defined while NaN lanes retain
        // the original first-minimum behavior. Infinite-amax casts remain the
        // unchanged reference's legacy behavior, not a new numeric contract.
        for (uint32_t bits : {0x7fc12345u, 0xffc12345u, 0x7f800001u}) {
            std::vector<float> values(QK_MXFP4, 0.0f);
            values[0] = 1.0f;
            std::memcpy(&values[3], &bits, sizeof(bits));
            values[16] = -0.0f;
            check(values);
        }
        // A quiet NaN in the first amax lane need not raise NV in fmax.
        // Do not let a later amax comparison or a preseeded NV hide loss of
        // the search's ordered-comparison exception.
        for (uint32_t bits : {0x7fc12345u, 0xffc12345u}) {
            std::vector<float> values(QK_MXFP4, 0.0f);
            std::memcpy(&values[0], &bits, sizeof(bits));
            values[1] = 8.0f;
            check(values, FE_INEXACT);
            ++seeded_rows;
            assert(std::fetestexcept(FE_ALL_EXCEPT) == (FE_INEXACT | FE_INVALID));
        }
        // Max reduction must retain every input lane and reset its seed for
        // each K32 block. Keep normal, subnormal-scale and e252 maxima in
        // one row, with the unique maximum moving through all 32 lanes.
        for (int initial_flags : {0, FE_INEXACT}) {
            for (int maximum_lane = 0; maximum_lane < QK_MXFP4; ++maximum_lane) {
                std::vector<float> values(3 * QK_MXFP4, -0.0f);
                const float sign = maximum_lane & 1 ? -1.0f : 1.0f;
                values[maximum_lane] = sign * 8.0f;
                values[QK_MXFP4 + maximum_lane] = sign * 0x1p-125f;
                values[2 * QK_MXFP4 + maximum_lane] = sign * 0x1p127f;
                check(values, initial_flags);
                seeded_rows += initial_flags != 0;
                ++amax_rows;
            }
            // Maximum-number reduction may raise NV later than the scalar
            // ordered scan for qNaN. The complete quantizer must still
            // report the same final flags and preserve reference code0.
            for (uint32_t bits : {0x7fc12345u, 0xffc12345u, 0x7f800001u, 0xff800001u}) {
                for (int nan_lane = 0; nan_lane < QK_MXFP4; ++nan_lane) {
                    std::vector<float> values(QK_MXFP4, 0.0f);
                    values[(nan_lane + 1) % QK_MXFP4] = 8.0f;
                    std::memcpy(&values[nan_lane], &bits, sizeof(bits));
                    check(values, initial_flags);
                    seeded_rows += initial_flags != 0;
                    ++amax_rows;
                }
                // No finite anchor: the legacy zero amax gives e0 without
                // calling log2f. Both sides must accrue NV, not NX/OF/UF.
                std::vector<float> values(QK_MXFP4);
                for (float & value : values) {
                    std::memcpy(&value, &bits, sizeof(bits));
                }
                check(values, initial_flags);
                seeded_rows += initial_flags != 0;
                ++amax_rows;
                assert(std::fetestexcept(FE_ALL_EXCEPT) == (initial_flags | FE_INVALID));
            }
        }
        // Exercise every caller-visible sticky state, including NX alone and
        // states without NX. These are correctness inputs, not benchmark
        // repetitions. The unchanged R5 paired search is the first baseline;
        // a future conditional search must pass the same complete byte/flag
        // oracle, not inherit a previous binary's PASS.
        const int exceptions[] = {FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW, FE_UNDERFLOW, FE_INEXACT};
        for (unsigned state = 0; state < 32; ++state) {
            int initial_flags = 0;
            for (unsigned bit = 0; bit < 5; ++bit) {
                if (state & (1u << bit)) initial_flags |= exceptions[bit];
            }
            uint8_t empty[] = {0xa5, 0xa5, 0xa5};
            seed_flags(initial_flags);
            cpu->from_float(nullptr, empty + 1, 0);
            assert(std::fetestexcept(FE_ALL_EXCEPT) == initial_flags);
            assert(empty[0] == 0xa5 && empty[1] == 0xa5 && empty[2] == 0xa5);

            check(std::vector<float>(QK_MXFP4, -0.0f), initial_flags);
            ++seeded_rows;
            {
                std::vector<float> values(QK_MXFP4, 0.0f);
                values[0] = 8.0f;
                values[1] = -std::nextafter(8.0f, 0.0f);
                values[2] = -0.5f;
                values[3] = -0.0f;
                values[15] = std::numeric_limits<float>::denorm_min();
                values[16] = -std::numeric_limits<float>::denorm_min();
                values[30] = std::numeric_limits<float>::min();
                values[31] = -std::numeric_limits<float>::min();
                check(values, initial_flags);
                ++seeded_rows;
            }
            // Exact powers fix e0/e1/e2/e251/e252. In particular, NX seeded
            // before an e252 block must not hide a missing accrued OF.
            for (int exponent : {-125, -124, -123, 126, 127}) {
                for (int sign : {-1, 1}) {
                    check(std::vector<float>(QK_MXFP4, std::ldexp(float(sign), exponent)), initial_flags);
                    ++seeded_rows;
                }
            }
            {
                std::vector<float> values(3 * QK_MXFP4);
                for (int lane = 0; lane < QK_MXFP4; ++lane) {
                    values[lane] = lane & 1 ? -0x1p126f : 0x1p126f;
                    values[QK_MXFP4 + lane] = -0x1p127f;
                    values[2 * QK_MXFP4 + lane] = float(lane - 16) * 0.5f;
                }
                check(values, initial_flags);
                ++seeded_rows;
            }
            // At VLEN128/m4, lane31 is in the second strip. Keep NaN payloads
            // intact and require actual ordered-comparison NV on both signs.
            for (uint32_t bits : {0x7fc12345u, 0xffc12345u, 0x7f800001u, 0xff800001u}) {
                std::vector<float> values(QK_MXFP4, 0.0f);
                values[0] = 8.0f;
                values[16] = -0.0f;
                std::memcpy(&values[31], &bits, sizeof(bits));
                check(values, initial_flags);
                ++seeded_rows;
            }
        }
    }
    assert(std::fesetenv(&saved_environment) == 0);
    printf("MXFP4_RUNTIME_BYTES_PASS rows=%u blocks=%u rounding_modes=4 "
        "reference=unchanged_ref runtime=cpu_type_traits all_payload_and_scale_bytes=exact fp_exceptions=exact "
        "midpoint_neighbors=pass signed_zero=pass finite_exponents=all "
        "misalignment_and_canaries=pass initial_flag_states=32 seeded_rows=%u "
        "amax_lane_rows=%u native_matrix=not-issued\n",
        checked_rows, checked_blocks, seeded_rows, amax_rows);
}

// Supplied-scale admission only: no matrix issue, payload encoder or perf loop.
static void test_mxfp4_supplied_scales() {
#if defined(_WIN32) && defined(GGML_BACKEND_SHARED)
    // Like the existing CMake internal-function tests, do not invent an export
    // or private declaration for a Windows shared-library internal symbol.
    printf("MXFP4_SUPPLIED_PREFLIGHT_SKIP reason=Windows_shared_internal_API_not_exported\n");
#else
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    assert(cpu && cpu->from_float);
    // Query the compiled CPU backend, not this test TU's private -march flags.
    const bool rvv = ggml_cpu_has_riscv_v() != 0;
    std::fenv_t saved_environment;
    assert(std::fegetenv(&saved_environment) == 0);
    unsigned rows = 0, admitted = 0, rejected = 0, fallbacks = 0;
    unsigned partial_rejects = 0, unsafe_ref_skipped = 0, invalid_arguments = 0;
    const auto seed_flags = [](int flags) {
        assert(std::feclearexcept(FE_ALL_EXCEPT) == 0);
        assert(std::feraiseexcept(flags) == 0);
        assert(std::feclearexcept(std::fetestexcept(FE_ALL_EXCEPT) & ~flags) == 0);
        assert(std::fetestexcept(FE_ALL_EXCEPT) == flags);
    };
    const auto check = [&](const std::vector<float> & values, int initial_flags,
            bool rvv_accepts, int rvv_added_flags, bool reference_safe, bool partial) {
        assert(!values.empty() && values.size() % QK_MXFP4 == 0);
        const size_t nb = values.size() / QK_MXFP4, bytes = nb * sizeof(block_mxfp4);
        const int rounding = std::fegetround();
        std::vector<float> input(values.size() + 2, -123.25f);
        std::memcpy(input.data() + 1, values.data(), values.size() * sizeof(float));
        const auto unchanged = input;
        std::vector<uint8_t> scales(nb + 2, 0xa5);
        std::vector<uint8_t> fallback(bytes + 2, 0xa5);
        std::vector<uint8_t> runtime(bytes + 2, 0xa5), reference(bytes + 2, 0xa5);
        seed_flags(initial_flags);
        // Positive admission must prove the API itself executes scale work.
        // No synthetic NX seed or external log2f call may make it pass.
        if (rvv_accepts) assert((std::fetestexcept(FE_ALL_EXCEPT) & FE_INEXACT) == 0);
        const bool accepted = ggml_mxfp4_prepare_supplied_scales(
            input.data() + 1, scales.data() + 1, values.size());
        const int prepare_flags = std::fetestexcept(FE_ALL_EXCEPT);
        const int expected_flags = initial_flags | (rvv ? rvv_added_flags : 0);
        if (accepted != (rvv && rvv_accepts) || prepare_flags != expected_flags) {
            fprintf(stderr, "MXFP4_SUPPLIED_PREFLIGHT_MISMATCH row=%u round=%d seed=%x "
                "rvv=%d accepted=%d expected=%d flags=%x expected_flags=%x\n",
                rows, rounding, initial_flags, rvv, accepted, rvv && rvv_accepts,
                prepare_flags, expected_flags);
            assert(false);
        }
        assert(std::fegetround() == rounding);
        assert(scales.front() == 0xa5 && scales.back() == 0xa5);
        assert(std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0);
        if (!rvv) for (uint8_t byte : scales) assert(byte == 0xa5);
        const uint8_t first_scale = scales[1];
        if (partial && rvv) {
            assert(!accepted && nb == 2 && first_scale != 0xa5);
            ++partial_rejects;
        }
        int fallback_flags = 0;
        if (!accepted) {
            // SF output is not transactional. Discard ALL partial scale bytes,
            // then really quantize the complete original row without clearing
            // the accrued preflight flags. Never fall back only the suffix.
            std::fill(scales.begin() + 1, scales.end() - 1, 0xcc);
            if (reference_safe) {
                cpu->from_float(input.data() + 1, fallback.data() + 1, values.size());
                fallback_flags = std::fetestexcept(FE_ALL_EXCEPT);
                assert(std::fegetround() == rounding);
                assert(fallback.front() == 0xa5 && fallback.back() == 0xa5);
                assert(std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0);
                ++fallbacks;
            }
            ++rejected;
        } else {
            ++admitted;
        }
        if (reference_safe) {
            // Independent test invocations start from the SAME incoming state.
            // These resets are fixture-only; the fallback above had no reset.
            seed_flags(initial_flags);
            quantize_row_mxfp4_ref(input.data() + 1,
                reinterpret_cast<block_mxfp4 *>(reference.data() + 1), values.size());
            const int reference_flags = std::fetestexcept(FE_ALL_EXCEPT);
            assert(std::fegetround() == rounding);
            assert(reference.front() == 0xa5 && reference.back() == 0xa5);
            assert(std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0);
            seed_flags(initial_flags);
            cpu->from_float(input.data() + 1, runtime.data() + 1, values.size());
            const int runtime_flags = std::fetestexcept(FE_ALL_EXCEPT);
            assert(std::fegetround() == rounding);
            assert(runtime.front() == 0xa5 && runtime.back() == 0xa5);
            assert(runtime == reference && runtime_flags == reference_flags);
            if (accepted) {
                assert(prepare_flags == reference_flags);
                for (size_t block = 0; block < nb; ++block) {
                    assert(scales[1 + block] == reference[1 + block * sizeof(block_mxfp4)]);
                    assert(scales[1 + block] == runtime[1 + block * sizeof(block_mxfp4)]);
                }
                assert(nb == 2 && scales[1] != scales[2]);
            } else {
                assert(fallback == reference && fallback_flags == reference_flags);
                if (partial && rvv) assert(first_scale == reference[1]);
            }
        } else {
            // Inf-amax/tiny-only casts are outside the unchanged reference's
            // supported scale-cast domain. Rejection is NOT a fallback PASS.
            assert(!accepted);
            ++unsafe_ref_skipped;
        }
        assert(scales.front() == 0xa5 && scales.back() == 0xa5);
        assert(std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0);
        ++rows;
    };
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        assert(std::fesetround(rounding) == 0);
        // Every initial sticky combination WITHOUT NX, not just a clean seed.
        // 3/6/12 are non-powers: their real canonical log2f must naturally
        // accrue NX before the API's frflags observation and true return.
        const int non_nx[] = {FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW, FE_UNDERFLOW};
        for (unsigned state = 0; state < 16; ++state) {
            int seed = 0;
            for (unsigned bit = 0; bit < 4; ++bit) if (state & (1u << bit)) seed |= non_nx[bit];
            for (const auto & maxima : std::vector<std::vector<float>>{
                    {3.0f, 6.0f}, {6.0f, 12.0f}, {12.0f, 3.0f}}) {
                std::vector<float> values(2 * QK_MXFP4, 0.0f);
                for (int block = 0; block < 2; ++block) {
                    for (int lane = 0; lane < QK_MXFP4; ++lane) {
                        values[block * QK_MXFP4 + lane] =
                            maxima[block] * float(lane % 15 - 7) * 0.125f;
                    }
                    values[block * QK_MXFP4 + (block ? 31 : 0)] = maxima[block];
                    values[block * QK_MXFP4 + 16] = -0.0f;
                }
                check(values, seed, true, FE_INEXACT, true, false);
            }
            // Exact scale work does NOT earn eligibility from this fixture.
            // Keep initial NX clear; zero/exact powers must really fall back.
            std::vector<float> powers(2 * QK_MXFP4, 8.0f);
            std::fill(powers.begin() + QK_MXFP4, powers.end(), 16.0f);
            check(powers, seed, false, 0, true, false);
            std::vector<float> zeros(2 * QK_MXFP4, 0.0f);
            for (size_t lane = 0; lane < zeros.size(); lane += 2) zeros[lane] = -0.0f;
            check(zeros, seed, false, 0, true, false);
        }
        // All 32 incoming states are correctness inputs for invalid arguments
        // and the non-RVV no-write/no-flags contract, never positive NX seeds.
        const int exceptions[] = {FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW, FE_UNDERFLOW, FE_INEXACT};
        for (unsigned state = 0; state < 32; ++state) {
            int seed = 0;
            for (unsigned bit = 0; bit < 5; ++bit) if (state & (1u << bit)) seed |= exceptions[bit];
            for (int bad = 0; bad < 4; ++bad) {
                std::vector<float> input(QK_MXFP4 + 2, 8.0f);
                input.front() = input.back() = -123.25f;
                const auto unchanged = input;
                uint8_t scales[] = {0xa5, 0xa5, 0xa5};
                seed_flags(seed);
                const bool accepted = ggml_mxfp4_prepare_supplied_scales(
                    bad == 0 ? nullptr : input.data() + 1,
                    bad == 1 ? nullptr : scales + 1,
                    bad == 2 ? 0 : (bad == 3 ? 31 : QK_MXFP4));
                assert(!accepted && std::fetestexcept(FE_ALL_EXCEPT) == seed);
                assert(std::fegetround() == rounding);
                for (uint8_t byte : scales) assert(byte == 0xa5);
                assert(std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0);
                ++invalid_arguments;
            }
        }
        for (int seed : {0, FE_INEXACT}) {
            for (uint32_t bits : {0x7fc12345u, 0xffc12345u, 0x7f800001u,
                    0xff800001u, 0x7f800000u, 0xff800000u}) {
                for (int lane : {0, 31}) {
                    std::vector<float> values(QK_MXFP4, 0.0f);
                    values[lane == 0 ? 1 : 0] = 8.0f; // Defined scale for NaN fallback.
                    std::memcpy(&values[lane], &bits, sizeof(bits));
                    check(values, seed, false, 0, (bits & 0x7fffffu) != 0, false);
                }
            }
            check(std::vector<float>(QK_MXFP4, 0x1p127f), seed, false, 0, true, false);
        }
        for (float tiny : {0x1p-126f, std::numeric_limits<float>::denorm_min()}) {
            check(std::vector<float>(QK_MXFP4, tiny), 0, false, 0, false, false);
        }
        // First SF really written by natural-NX scale work; second K32 rejects.
        // The safe whole-row fallback must ignore that otherwise valid prefix.
        for (uint32_t bits : {0x7fc12345u, 0x7f800001u, 0x7f000000u}) {
            std::vector<float> values(2 * QK_MXFP4, 0.0f);
            values[0] = 3.0f;
            values[QK_MXFP4] = 8.0f;
            std::memcpy(&values[2 * QK_MXFP4 - 1], &bits, sizeof(bits));
            check(values, 0, false, FE_INEXACT, true, true);
        }
    }
    assert(std::fesetenv(&saved_environment) == 0);
    printf("MXFP4_SUPPLIED_PREFLIGHT_PASS rows=%u admitted=%u rejected=%u "
        "whole_row_fallbacks=%u partial_rejects=%u unsafe_reference_not_invoked=%u "
        "invalid_arguments=%u rounding_modes=4 invalid_initial_states=32 positive_initial_states=16_without_NX "
        "positive_initial_NX=0 natural_log2_NX=required rvv=%d "
        "scales=canonical_runtime_and_ref canaries_input_frm_flags=exact native_matrix=not-issued\n",
        rows, admitted, rejected, fallbacks, partial_rejects, unsafe_ref_skipped, invalid_arguments, rvv);
#endif
}

static void test_ame_mxfp4_packing() {
    for (int nblocks : {1, 2, 3, 64, 129}) {
        // Deliberately unaligned source (the byte representation permits it).
        std::vector<uint8_t> source(17 * nblocks + 2, 0xa5);
        auto * blocks = reinterpret_cast<block_mxfp4 *>(source.data() + 1);
        for (int b = 0; b < nblocks; ++b) {
            blocks[b].e = (uint8_t) (126 + b % 4);
            for (int j = 0; j < 16; ++j) {
                blocks[b].qs[j] = (uint8_t) (((j + b) & 15) | (((15 - j + b) & 15) << 4));
            }
        }
        const auto unchanged = source;
        std::vector<uint8_t> payload(nblocks * 16 + 2, 0xa5);
        std::vector<uint8_t> scales(nblocks + 2, 0xa5);
        assert(ggml_ame_pack_mxfp4_blocks(payload.data() + 1, scales.data() + 1, blocks, nblocks));
        assert(source == unchanged && payload.front() == 0xa5 && payload.back() == 0xa5);
        assert(scales.front() == 0xa5 && scales.back() == 0xa5);
        std::vector<block_mxfp4> restored(nblocks);
        for (int b = 0; b < nblocks; ++b) {
            restored[b].e = scales[b + 1];
            for (int k = 0; k < 32; ++k) {
                const uint8_t byte = payload[1 + b * 16 + k / 2];
                const uint8_t code = (byte >> (4 * (k & 1))) & 15;
                restored[b].qs[k % 16] |= code << (k < 16 ? 0 : 4);
                const uint8_t expected = (blocks[b].qs[k % 16] >> (k < 16 ? 0 : 4)) & 15;
                assert(code == expected);
                assert(ame_test_e2m1(code, scales[b + 1]) == ame_test_e2m1(expected, blocks[b].e));
            }
        }
        assert(std::memcmp(restored.data(), blocks, nblocks * sizeof(block_mxfp4)) == 0);
    }
    // Exhaust all raw E8M0 bytes and all E2M1 codes. Compare finite-scale
    // values with canonical CPU dequantization, separately retain raw 255.
    for (int scale = 0; scale < 256; ++scale) {
        block_mxfp4 block = {};
        block.e = (uint8_t) scale;
        for (int j = 0; j < 16; ++j) {
            block.qs[j] = (uint8_t) (j | ((15 - j) << 4));
        }
        uint8_t payload[16], e;
        assert(ggml_ame_pack_mxfp4_blocks(payload, &e, &block, 1));
        assert(e == scale);
        float reference[32];
        if (scale != 255) {
            dequantize_row_mxfp4(&block, reference, 32);
        }
        for (int k = 0; k < 32; ++k) {
            const uint8_t code = (payload[k / 2] >> (4 * (k & 1))) & 15;
            const float value = ame_test_e2m1(code, e);
            if (scale == 255) {
                assert(std::isnan(value));
            } else {
                assert(value == reference[k]);
            }
        }
    }
    uint8_t sentinel = 0xa5;
    block_mxfp4 block = {};
    assert(!ggml_ame_pack_mxfp4_blocks(&sentinel, &sentinel, &block, 0));
    assert(!ggml_ame_pack_mxfp4_blocks(nullptr, &sentinel, &block, 1));
    assert(sentinel == 0xa5);
    assert(ggml_ame_mxfp4_shape_supported(64, 128, 96));
    assert(ggml_ame_mxfp4_shape_supported(128, 64, 32));
    for (int edge : {0, 1, 15, 16, 17, 31, 32, 33, 63, 65, 127, 129}) {
        assert(ggml_ame_mxfp4_shape_supported(edge, 64, 64) == (edge >= 64));
        assert(!ggml_ame_mxfp4_shape_supported(64, edge, 64));
    }
    for (int tail : {0, 1, 15, 16, 17, 31, 33, 63, 65}) {
        assert(!ggml_ame_mxfp4_shape_supported(64, 64, tail));
    }
    printf("AME_MXFP4_PACK_PASS roundtrip=exact scales=256 codes=16 edge_shape=fallback\n");
}

static void test_ame_mxfp4_k64_packing() {
    unsigned rows_checked = 0, selections = 0;
    for (int64_t M : {64, 128}) {
        for (int64_t K : {64, 128, 256}) {
            const size_t nb = (size_t) K / 32;
            std::vector<block_mxfp4> source((size_t) M * nb);
            for (int64_t row = 0; row < M; ++row) {
                for (size_t kb = 0; kb < nb; ++kb) {
                    auto & block = source[(size_t) row * nb + kb];
                    block.e = (uint8_t) ((row * 3 + (int64_t) kb * 37) % 256);
                    for (int k = 0; k < 32; ++k) {
                        const uint8_t code = (uint8_t) ((row * 5 + (int64_t) kb * 7 + k) & 15);
                        block.qs[k % 16] |= code << (k < 16 ? 0 : 4);
                    }
                }
            }
            const auto unchanged = source;
            std::vector<uint8_t> payload((size_t) M * nb * 16 + 2, 0xa5);
            std::vector<uint8_t> scales((size_t) M * nb + 2, 0xa5);
            for (int64_t row = 0; row < M; ++row) {
                for (size_t kb = 0; kb < nb; kb += 2) {
                    const size_t index = ggml_ame_mx_packed_block_index(row, kb, nb, 64, 64);
                    assert(ggml_ame_pack_mxfp4_blocks(payload.data() + 1 + index * 16,
                        scales.data() + 1 + index, source.data() + (size_t) row * nb + kb, 2));
                }
            }
            assert(std::memcmp(source.data(), unchanged.data(), source.size() * sizeof(block_mxfp4)) == 0);
            assert(payload.front() == 0xa5 && payload.back() == 0xa5);
            assert(scales.front() == 0xa5 && scales.back() == 0xa5);
            // Independent panel/row/K byte index, not the production helper.
            // Check all nibbles, both scales and the resulting consecutive K
            // ordering across the K32 boundary and across output-row panels.
            for (int64_t row = 0; row < M; ++row) {
                for (int64_t k = 0; k < K; ++k) {
                    const size_t panel = (size_t) (row / 64 * (K / 64) + k / 64);
                    const size_t byte = panel * 2048 + (size_t) (row % 64) * 32 + (size_t) (k % 64) / 2;
                    const size_t scale = panel * 128 + (size_t) (row % 64) * 2 + (size_t) (k % 64) / 32;
                    const auto & block = source[(size_t) row * nb + (size_t) k / 32];
                    const uint8_t expected = (block.qs[k % 16] >> (k % 32 < 16 ? 0 : 4)) & 15;
                    assert(((payload[1 + byte] >> (4 * (k & 1))) & 15) == expected);
                    assert(scales[1 + scale] == block.e); // Raw255 deliberately preserved.
                }
                rows_checked++;
            }
        }
    }
    // All raw scale pairs and E2M1 code values retain their bits, including
    // negative zero and255; native-domain rejection remains a separate gate.
    for (int e = 0; e < 256; ++e) {
        block_mxfp4 blocks[2] = {};
        blocks[0].e = (uint8_t) e;
        blocks[1].e = (uint8_t) (255 - e);
        for (int j = 0; j < 16; ++j) {
            blocks[0].qs[j] = (uint8_t) (j | ((15 - j) << 4));
            blocks[1].qs[j] = (uint8_t) ((15 - j) | (j << 4));
        }
        uint8_t payload[32], scales[2];
        assert(ggml_ame_pack_mxfp4_blocks(payload, scales, blocks, 2));
        assert(scales[0] == e && scales[1] == 255 - e);
        assert(ggml_ame_mxfp4_native_domain(blocks, 2) == (e != 0 && e != 255));
        for (int k = 0; k < 64; ++k) {
            const uint8_t expected = (blocks[k / 32].qs[k % 16] >> (k % 32 < 16 ? 0 : 4)) & 15;
            assert(((payload[k / 2] >> (4 * (k & 1))) & 15) == expected);
        }
    }
    for (const char * knob : {static_cast<const char *>(nullptr), "", "0", "off", "true", "64", "garbage", "1"}) {
        for (int K : {32, 64, 96, 128}) {
            const int expected = knob != nullptr && std::strcmp(knob, "1") == 0 && K % 64 == 0 ? 64 : 32;
            assert(ggml_ame_mx_select_tile_k(AME_MCFG_MXFP4, 64, 64, K, 1, knob) == expected);
            for (int format : {AME_MCFG_MXFP8_E4M3, AME_MCFG_MXFP8_E5M2}) {
                assert(ggml_ame_mx_select_tile_k(format, 64, 64, K, 1, knob) == 32);
            }
            selections += 3;
        }
    }
    for (int edge : {0, 1, 31, 33, 63, 65, 127}) {
        assert(ggml_ame_mx_select_tile_k(AME_MCFG_MXFP4, edge, 64, 64, 1, "1") == (edge >= 64 ? 64 : 0));
        assert(ggml_ame_mx_select_tile_k(AME_MCFG_MXFP4, 64, edge, 64, 1, "1") == 0);
        assert(ggml_ame_mx_select_tile_k(AME_MCFG_MXFP4, 64, 64, edge, 1, "1") == 0);
    }
    assert(ggml_ame_mx_select_tile_k(AME_MCFG_MXFP4, 64, 64, 64, 2, "1") == 0);
    assert(ggml_ame_mx_select_tile_k(0x10, 64, 64, 64, 1, "1") == 0);
    assert(!ggml_ame_mx_tile_k_supported(AME_MCFG_MXFP8_E4M3, 64));
    printf("AME_MXFP4_K64_PACK_PASS rows=%u raw_scale_pairs=256 all_codes=16 "
           "two_scale_order=exact nibble_permutation=exact panel_shape=64,128x64,128,256 "
           "selections=%u K32_default=pass edge_tail=fallback hardware=not-run\n",
           rows_checked, selections);
}

static void test_ame_rmsnorm_quant() {
    std::fenv_t saved;
    assert(std::fegetenv(&saved) == 0);
    const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->from_float;
    assert(quantize != nullptr);
    unsigned cases = 0, checked_blocks = 0;
    auto seed = [](int round, int flags) {
        assert(std::fesetround(round) == 0);
        assert(std::feclearexcept(FE_ALL_EXCEPT) == 0);
        assert(std::feraiseexcept(flags) == 0);
        assert(std::feclearexcept(FE_ALL_EXCEPT & ~flags) == 0);
        assert(std::fetestexcept(FE_ALL_EXCEPT) == flags);
    };
    auto check = [&](int rows, int K, int tk, int mn, int round, int flags, int pattern = 0) {
        const int stride = K + 5, nb = K / 32;
        const float eps = pattern == 0 ? 0x1p-10f :
            pattern == 7 ? 3.0f - 3.0f / K : 3.0f;
        std::vector<float> input(size_t(rows) * stride + 2, 0x1.5p+8f);
        std::vector<float> gamma(K + 2, -0x1.3p+7f);
        for (int row = 0; row < rows; ++row) {
            for (int k = 0; k < K; ++k) {
                input[1 + size_t(row) * stride + k] = pattern != 0 ?
                    (pattern == 3 && row == rows - 1 ? 2.0f : 1.0f) : row % 11 == 0 ? -0.0f :
                    float((row * 17 + k * 13) % 127 - 63) * 0x1p-5f;
            }
        }
        for (int k = 0; k < K; ++k) {
            gamma[1 + k] = pattern == 1 ? 0x1p-124f :
                pattern == 2 || pattern == 4 ? (k < (pattern == 4 ? 256 : 32) ? 1.0f : 1.5f) :
                pattern >= 3 ? 1.0f : float(32 + (k * 7) % 65) * 0x1p-6f;
        }
        if (pattern == 5 || pattern == 6) {
            // Exact RMS=.5, then a REAL underflow/inexact gamma multiply
            // late in this batch or the next. Other lanes anchor its scale.
            // Unlike gamma=1.5, this tests norm-generated NX reordering.
            gamma[1 + (pattern == 5 ? 7 : 8) * 32 + 3] = 0x1p-149f;
        }
        if (pattern == 7) {
            // Power-of-two K makes mean+eps exactly 4 in every frm.
            // (2*.5)*minsub is EXACT; 2*(.5*minsub) would add UF/NX.
            assert(K == 256);
            for (int row = 0; row < rows; ++row)
                input[1 + size_t(row) * stride + 7 * 32 + 3] = 2.0f;
            gamma[1 + 7 * 32 + 3] = 0x1p-149f;
        }
        const auto input_before = input, gamma_before = gamma;
        const size_t blocks = size_t(rows) * nb;
        std::vector<uint8_t> expected(blocks * 16 + 2, 0xc3), actual(expected);
        std::vector<uint8_t> expected_sf(blocks + 2, 0xa7), actual_sf(expected_sf);
        std::vector<block_mxfp4> encoded(blocks);
        ggml_init_params init = { 4 * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(init);
        assert(ctx != nullptr);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, rows);
        ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
        x->data = input.data() + 1;
        x->nb[1] = size_t(stride) * sizeof(float);
        x->nb[2] = x->nb[1] * rows;
        x->nb[3] = x->nb[2];
        g->data = gamma.data() + 1;
        ggml_tensor * normalized = ggml_rms_norm(ctx, x, eps);
        ggml_tensor * weighted = ggml_mul(ctx, normalized, g);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, weighted);
        seed(round, flags);
        assert(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS);
        if (pattern == 5 || pattern == 6) {
            const int norm_flags = std::fetestexcept(FE_ALL_EXCEPT);
            assert((norm_flags & (FE_UNDERFLOW | FE_INEXACT)) == (FE_UNDERFLOW | FE_INEXACT));
        }
        if (pattern == 7) {
            assert(std::fetestexcept(FE_ALL_EXCEPT) == flags);
        }
        for (int row = 0; row < rows; ++row) {
            quantize(static_cast<const float *>(weighted->data) + size_t(row) * K,
                encoded.data() + size_t(row) * nb, K);
        }
        // Existing packer, with an independently expressed panel index.
        for (int row = 0; row < rows; ++row) {
            for (int kb = 0; kb < nb; ++kb) {
                const size_t groups = tk / 32;
                const size_t index = (size_t(row / mn) * (nb / groups) + kb / groups) * mn * groups +
                    size_t(row % mn) * groups + kb % groups;
                assert(ggml_ame_pack_mxfp4_blocks(expected.data() + 1 + index * 16,
                    expected_sf.data() + 1 + index, encoded.data() + size_t(row) * nb + kb, 1));
            }
        }
        const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
        seed(round, flags);
        assert(ggml_ame_rmsnorm_pack_mxfp4(actual.data() + 1, actual_sf.data() + 1,
            input.data() + 1, gamma.data() + 1, rows, K, size_t(stride) * sizeof(float),
            eps, tk, mn, nullptr));
        const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
        if (actual != expected || actual_sf != expected_sf || actual_flags != expected_flags) {
            fprintf(stderr, "AME_RMS_QUANT_MISMATCH rows=%d K=%d tk=%d mn=%d round=%d "
                "seed=%x flags=%x/%x payload=%d scales=%d\n", rows, K, tk, mn, round,
                flags, actual_flags, expected_flags, actual == expected, actual_sf == expected_sf);
            assert(false);
        }
        assert(std::fegetround() == round);
        // Whole-input preflight and subsequent token panels must be exactly
        // equivalent too; no partial panel can authorize native execution.
        std::vector<float> factors(rows + 2, -123.0f);
        std::fill(actual.begin(), actual.end(), 0xc3);
        std::fill(actual_sf.begin(), actual_sf.end(), 0xa7);
        seed(round, flags);
        assert(ggml_ame_rmsnorm_mxfp4_preflight(factors.data() + 1,
            input.data() + 1, gamma.data() + 1, rows, K, size_t(stride) * sizeof(float), eps, mn, nullptr));
        for (int panel = 0; panel < rows; panel += mn) {
            assert(ggml_ame_rmsnorm_pack_mxfp4(actual.data() + 1 + size_t(panel) * nb * 16,
                actual_sf.data() + 1 + size_t(panel) * nb,
                input.data() + 1 + size_t(panel) * stride, gamma.data() + 1, mn, K,
                size_t(stride) * sizeof(float), eps, tk, mn, factors.data() + 1 + panel));
        }
        assert(actual == expected && actual_sf == expected_sf);
        assert(std::fetestexcept(FE_ALL_EXCEPT) == expected_flags && std::fegetround() == round);
        // Provisional first-panel preparation is reusable ONLY after all
        // rows pass admission. The remaining panels use unchanged factors.
        std::fill(actual.begin(), actual.end(), 0xc3);
        std::fill(actual_sf.begin(), actual_sf.end(), 0xa7);
        seed(round, flags);
        ggml_ame_rmsnorm_first_panel first = { actual.data() + 1, actual_sf.data() + 1, tk, 0, 0 };
        assert(ggml_ame_rmsnorm_mxfp4_preflight(factors.data() + 1,
            input.data() + 1, gamma.data() + 1, rows, K,
            size_t(stride) * sizeof(float), eps, mn, &first));
        for (int panel = mn; panel < rows; panel += mn) {
            assert(ggml_ame_rmsnorm_pack_mxfp4(actual.data() + 1 + size_t(panel) * nb * 16,
                actual_sf.data() + 1 + size_t(panel) * nb,
                input.data() + 1 + size_t(panel) * stride, gamma.data() + 1, mn, K,
                size_t(stride) * sizeof(float), eps, tk, mn, factors.data() + 1 + panel));
        }
        assert(actual == expected && actual_sf == expected_sf && first.prepare_cycles == 0);
        assert(std::fetestexcept(FE_ALL_EXCEPT) == expected_flags && std::fegetround() == round);
        assert(factors.front() == -123.0f && factors.back() == -123.0f);
        assert(std::memcmp(input.data(), input_before.data(), input.size() * sizeof(float)) == 0);
        assert(std::memcmp(gamma.data(), gamma_before.data(), gamma.size() * sizeof(float)) == 0);
        ggml_free(ctx);
        ++cases;
        checked_blocks += unsigned(blocks);
    };
    // Two token panels: real graph RMSNorm+gamma versus direct packed producer.
    for (int mn : {64, 128}) {
        for (int tk : {32, 64}) {
            check(256, 128, tk, mn, FE_TONEAREST, 0);
        }
    }
    const int exceptions[] = {FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW, FE_UNDERFLOW, FE_INEXACT};
    for (int round : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        // A full eight-block batch, one batch plus a K64 tail, and two
        // batches plus a tail. The original full-row quantizer is the oracle.
        for (int K : {256, 320, 576}) {
            for (int tk : {32, 64}) check(64, K, tk, 64, round, 0);
        }
        check(64, 288, 32, 64, round, 0); // eight plus one K32, odd tail
        for (int tk : {32, 64}) check(256, 320, tk, 128, round, 0);
        for (int tk : {32, 64}) check(64, 320, tk, 64, round, 0, 4);
        for (unsigned mask = 0; mask < 32; ++mask) {
            int flags = 0;
            for (unsigned bit = 0; bit < 5; ++bit) {
                if (mask & (1u << bit)) flags |= exceptions[bit];
            }
            for (int tk : {32, 64}) check(64, 64, tk, 64, round, flags);
            for (int pattern : {5, 6}) {
                for (int tk : {32, 64}) check(64, 320, tk, 64, round, flags, pattern);
            }
            for (int tk : {32, 64}) check(64, 256, tk, 64, round, flags, 7);
        }
        // Exact RMS factor .5: defined scale-cast boundary, then cases
        // where NX first arises in a later K32 block or a later token row.
        for (int pattern : {1, 2, 3}) {
            for (int tk : {32, 64}) check(64, 64, tk, 64, round, 0, pattern);
        }
    }
    // Odd K32 count is legal for K32 tiles, but K64 must reject before
    // touching outputs or FP state, without encoding an incomplete pair.
    std::vector<float> odd_input(64 * 288, 1.0f), odd_gamma(288, 1.0f);
    std::vector<uint8_t> odd_payload(64 * 9 * 16 + 2, 0xc3), odd_scales(64 * 9 + 2, 0xa7);
    seed(FE_TONEAREST, 0);
    assert(!ggml_ame_rmsnorm_pack_mxfp4(odd_payload.data() + 1, odd_scales.data() + 1,
        odd_input.data(), odd_gamma.data(), 64, 288, 288 * sizeof(float),
        3.0f, 64, 64, nullptr));
    assert(std::fetestexcept(FE_ALL_EXCEPT) == 0);
    assert(std::all_of(odd_payload.begin(), odd_payload.end(), [](uint8_t v) { return v == 0xc3; }));
    assert(std::all_of(odd_scales.begin(), odd_scales.end(), [](uint8_t v) { return v == 0xa7; }));
    // First output is provisional: even a fully prepared first panel cannot
    // authorize a late normalized-domain failure in the next token panel.
    std::vector<float> late_input(128 * 64, 1.0f), late_gamma(64, 0x1p-124f);
    std::fill(late_input.end() - 64, late_input.end(), 0x1p-20f);
    std::vector<float> late_factors(130, -123.0f);
    std::vector<uint8_t> late_payload(64 * 2 * 16 + 2, 0xc3), late_sf(64 * 2 + 2, 0xa7);
    ggml_ame_rmsnorm_first_panel late_first = {
        late_payload.data() + 1, late_sf.data() + 1, 64, 0, 0
    };
    seed(FE_TONEAREST, 0);
    assert(!ggml_ame_rmsnorm_mxfp4_preflight(late_factors.data() + 1,
        late_input.data(), late_gamma.data(), 128, 64, 64 * sizeof(float), 3.0f, 64, &late_first));
    assert(late_sf[1] != 0xa7); // actual first panel was prepared, then discarded
    assert(late_payload.front() == 0xc3 && late_payload.back() == 0xc3);
    assert(late_sf.front() == 0xa7 && late_sf.back() == 0xa7);
    assert(late_factors.front() == -123.0f && late_factors.back() == -123.0f);
    for (int bad = 0; bad < 3; ++bad) {
        std::fill(late_payload.begin(), late_payload.end(), 0xc3);
        std::fill(late_sf.begin(), late_sf.end(), 0xa7);
        std::fill(late_factors.begin(), late_factors.end(), -123.0f);
        ggml_ame_rmsnorm_first_panel invalid_first = {
            bad == 0 ? nullptr : late_payload.data() + 1,
            bad == 1 ? nullptr : late_sf.data() + 1, bad == 2 ? 16 : 64, 0, 123
        };
        seed(FE_TONEAREST, 0);
        assert(!ggml_ame_rmsnorm_mxfp4_preflight(late_factors.data() + 1,
            late_input.data(), late_gamma.data(), 128, 64, 64 * sizeof(float), 3.0f, 64, &invalid_first));
        assert(std::fetestexcept(FE_ALL_EXCEPT) == 0 && invalid_first.prepare_cycles == 123);
        assert(std::all_of(late_payload.begin(), late_payload.end(), [](uint8_t v) { return v == 0xc3; }));
        assert(std::all_of(late_sf.begin(), late_sf.end(), [](uint8_t v) { return v == 0xa7; }));
        assert(std::all_of(late_factors.begin(), late_factors.end(), [](float v) { return v == -123.0f; }));
    }
    // Source shape accepts K96, but a requested first K64 panel must reject
    // before factors, packed scratch, descriptor timing or FP state changes.
    ggml_ame_rmsnorm_first_panel incomplete_pair = {
        odd_payload.data() + 1, odd_scales.data() + 1, 64, 0, 123
    };
    seed(FE_TONEAREST, 0);
    assert(!ggml_ame_rmsnorm_mxfp4_preflight(late_factors.data() + 1,
        odd_input.data(), odd_gamma.data(), 128, 96, 96 * sizeof(float), 3.0f, 64, &incomplete_pair));
    assert(std::fetestexcept(FE_ALL_EXCEPT) == 0 && incomplete_pair.prepare_cycles == 123);
    assert(std::all_of(odd_payload.begin(), odd_payload.end(), [](uint8_t v) { return v == 0xc3; }));
    assert(std::all_of(odd_scales.begin(), odd_scales.end(), [](uint8_t v) { return v == 0xa7; }));
    assert(std::all_of(late_factors.begin(), late_factors.end(), [](float v) { return v == -123.0f; }));
    std::vector<float> input(64 * 64, 1.0f), gamma(64, 1.0f);
    std::vector<uint8_t> payload(64 * 32 + 2, 0xc3), scales(64 * 2 + 2, 0xa7);
    for (int bad = 0; bad < 11; ++bad) {
        seed(FE_TONEAREST, 0);
        assert(!ggml_ame_rmsnorm_pack_mxfp4(bad == 0 ? nullptr : payload.data() + 1,
            bad == 1 ? nullptr : scales.data() + 1, bad == 2 ? nullptr : input.data(),
            bad == 3 ? nullptr : gamma.data(), bad == 4 ? 0 : 64, bad == 5 ? 31 : 64,
            bad == 6 ? 4 : (bad == 10 ? 64 * sizeof(float) + 1 : 64 * sizeof(float)),
            bad == 7 ? 0.0f : 0x1p-10f,
            bad == 8 ? 16 : 32, bad == 9 ? 32 : 64, nullptr));
        assert(std::fetestexcept(FE_ALL_EXCEPT) == 0);
        assert(std::all_of(payload.begin(), payload.end(), [](uint8_t v) { return v == 0xc3; }));
        assert(std::all_of(scales.begin(), scales.end(), [](uint8_t v) { return v == 0xa7; }));
    }
    // Late numeric rejection may leave a partial packed result: discard it.
    // Do not invoke the CPU quantizer on an Inf/NaN reference domain.
    gamma[63] = INFINITY;
    seed(FE_TONEAREST, 0);
    assert(!ggml_ame_rmsnorm_pack_mxfp4(payload.data() + 1, scales.data() + 1,
        input.data(), gamma.data(), 64, 64, 64 * sizeof(float), 0x1p-10f, 32, 64, nullptr));
    // A failed batch need not publish its earlier valid blocks. Partial
    // packed scratch, if any, remains discard-only under the API contract.
    assert(scales[65] == 0xa7);
    assert(payload.front() == 0xc3 && payload.back() == 0xc3);
    assert(scales.front() == 0xa7 && scales.back() == 0xa7);
    // Finite is not sufficient for the legacy scale cast: 2^-127 would
    // produce a negative floating-point value before its uint8 conversion.
    // Reject before calling that quantizer; deliberately no unsafe oracle.
    std::fill(gamma.begin(), gamma.end(), 0x1p-126f);
    std::fill(payload.begin(), payload.end(), 0xc3);
    std::fill(scales.begin(), scales.end(), 0xa7);
    seed(FE_TONEAREST, 0);
    assert(!ggml_ame_rmsnorm_pack_mxfp4(payload.data() + 1, scales.data() + 1,
        input.data(), gamma.data(), 64, 64, 64 * sizeof(float), 3.0f, 32, 64, nullptr));
    assert(std::fetestexcept(FE_ALL_EXCEPT) == 0);
    assert(std::all_of(payload.begin(), payload.end(), [](uint8_t v) { return v == 0xc3; }));
    assert(std::all_of(scales.begin(), scales.end(), [](uint8_t v) { return v == 0xa7; }));
    assert(std::fesetenv(&saved) == 0);
    printf("AME_RMS_QUANT_PASS cases=%u blocks=%u full_shape=256x128 token_panels=2 "
        "small_flags=4frm_x32states_x2layouts invalid_arguments=12 invalid_first_descriptors=4 partial_reject=discard "
        "tiny_scale_cast=pre_reject defined_boundary=exact late_NX=block_row_and_batch "
        "quant_batch_blocks=8 batch_tail_K=256_288_320_576 natural_norm_UFNX=kb7_kb8 "
        "two_round_norm_no_new_flags=exact "
        "reference=real_GGML_RMSNorm_gamma_CPU_quant_pack payload_scales_flags=exact "
        "whole_input_preflight_then_panels=exact first_panel_admission_reuse=exact provisional_late_reject=discard "
        "normalized_tensor_materialized_in_candidate=no "
        "hardware=not_run\n", cases, checked_blocks);
}

static void test_ame_mxfp4_native_domain() {
    for (int scale = 0; scale < 256; ++scale) {
        block_mxfp4 block = {};
        block.e = (uint8_t) scale; // All-zero payload must NOT exempt e255.
        assert(ggml_ame_mxfp4_native_domain(&block, 1) == (scale != 255));
    }
    for (int nblocks : {1, 2, 3, 64, 129}) {
        std::vector<block_mxfp4> blocks(nblocks);
        for (auto & block : blocks) {
            block.e = 127;
        }
        assert(ggml_ame_mxfp4_native_domain(blocks.data(), nblocks));
        for (int bad : {0, nblocks / 2, nblocks - 1}) {
            blocks[bad].e = 255;
            const auto unchanged = blocks;
            assert(!ggml_ame_mxfp4_native_domain(blocks.data(), nblocks));
            // CPU semantics really differ even for an all-zero block: its
            // output is finite zero; native scale255 would propagate NaN.
            float cpu[32];
            dequantize_row_mxfp4(&blocks[bad], cpu, 32);
            for (float value : cpu) {
                assert(std::isfinite(value) && value == 0.0f);
            }
            uint8_t payload[16], scale;
            assert(ggml_ame_pack_mxfp4_blocks(payload, &scale, &blocks[bad], 1));
            assert(scale == 255); // Domain rejection must NOT change the packer.
            assert(std::memcmp(blocks.data(), unchanged.data(), blocks.size() * sizeof(block_mxfp4)) == 0);
            blocks[bad].e = 127;
            assert(ggml_ame_mxfp4_native_domain(blocks.data(), nblocks));
        }
    }
    assert(!ggml_ame_mxfp4_native_domain(nullptr, 1));
    block_mxfp4 block = {};
    assert(!ggml_ame_mxfp4_native_domain(&block, 0));
    printf("AME_MXFP4_DOMAIN_PASS scales=256 bad_block=first,middle,last zero_payload255=reject raw_packer255=preserved\n");
}

// Independent mathematical decoder, not the production conversion algorithm.
static double ame_test_fp8(uint8_t code, int format) {
    const bool e4 = format == AME_MCFG_MXFP8_E4M3;
    const int bits = e4 ? 3 : 2;
    const int exponent = (code & 127) >> bits;
    const int mantissa = code & ((1 << bits) - 1);
    if (e4 ? (code & 127) == 127 : exponent == 31) {
        return NAN;
    }
    const double value = std::ldexp(exponent == 0 ? mantissa : mantissa + (1 << bits),
        (exponent == 0 ? 1 : exponent) - (e4 ? 7 : 15) - bits);
    return code & 128 ? -value : value;
}

static void test_ame_mxfp8_packing() {
    assert(AME_MCFG_MXFP8_E4M3 == 0x18 && AME_MCFG_MXFP8_E5M2 == 0x17);
    for (int format : {AME_MCFG_MXFP8_E4M3, AME_MCFG_MXFP8_E5M2}) {
        const bool e4 = format == AME_MCFG_MXFP8_E4M3;
        const uint8_t max_code = e4 ? 0x7e : 0x7b;
        const float max_value = e4 ? 448.0f : 57344.0f;
        uint8_t payload[32] = {}, scale;
        // All 256 raw byte classifications, including signed zero, E4M3FN's
        // finite top exponent, E5M2 Inf/sNaN/qNaN and negative counterparts.
        for (int code = 0; code < 256; ++code) {
            payload[31] = (uint8_t) code;
            scale = 127;
            const double value = ame_test_fp8((uint8_t) code, format);
            assert(ggml_ame_mxfp8_native_domain(payload, &scale, 1, format) == std::isfinite(value));
            if (!std::isfinite(value)) {
                continue;
            }
            for (int power : {-127, 0, e4 ? 119 : 111}) {
                float input[32] = {};
                input[0] = std::ldexp(max_value, power); // Anchor exactly selects power.
                input[1] = (float) std::ldexp(value, power);
                const auto original = std::vector<float>(input, input + 32);
                assert(ggml_ame_pack_mxfp8_f32(payload, &scale, input, 32, format));
                assert(scale == power + 127 && payload[0] == max_code && payload[1] == code);
                assert(std::memcmp(input, original.data(), sizeof(input)) == 0);
                assert(ggml_ame_mxfp8_native_domain(payload, &scale, 1, format));
            }
        }
        // Exhaust scale domain independently of F32's narrower representable
        // range. Domain checking does not reinterpret or mutate scale bytes.
        std::memset(payload, 0, sizeof(payload));
        for (int e = 0; e < 256; ++e) {
            scale = (uint8_t) e;
            assert(ggml_ame_mxfp8_native_domain(payload, &scale, 1, format) == (e != 255));
            assert(scale == e);
        }
        uint8_t multi_payload[96] = {}, multi_scales[3] = {127, 127, 127};
        for (int bad_block : {0, 1, 2}) {
            multi_scales[bad_block] = 255;
            assert(!ggml_ame_mxfp8_native_domain(multi_payload, multi_scales, 3, format));
            assert(multi_scales[bad_block] == 255); // Rejection cannot rewrite a raw scale.
            multi_scales[bad_block] = 127;
            multi_payload[bad_block * 32 + 31] = e4 ? 0xff : 0xfd;
            assert(!ggml_ame_mxfp8_native_domain(multi_payload, multi_scales, 3, format));
            multi_payload[bad_block * 32 + 31] = 0;
            assert(ggml_ame_mxfp8_native_domain(multi_payload, multi_scales, 3, format));
        }
        for (int nblocks : {1, 2, 3, 64, 129}) {
            std::vector<float> source(nblocks * 32 + 2, 12345.0f);
            std::vector<uint8_t> bytes(nblocks * 32 + 2, 0xa5);
            std::vector<uint8_t> scales(nblocks + 2, 0xa5);
            std::vector<uint8_t> expected(nblocks * 32);
            for (int b = 0; b < nblocks; ++b) {
                const int power = -10 + (b % 5) * 5;
                for (int k = 0; k < 32; ++k) {
                    const uint8_t code = k == 0 ? max_code : (uint8_t) ((b + k) % (max_code + 1));
                    expected[b * 32 + k] = code | (k % 3 == 0 && k != 0 ? 128 : 0);
                    source[1 + b * 32 + k] = (float) std::ldexp(
                        ame_test_fp8(expected[b * 32 + k], format), power);
                }
            }
            const auto original = source;
            assert(ggml_ame_pack_mxfp8_f32(bytes.data() + 1, scales.data() + 1,
                source.data() + 1, nblocks * 32, format));
            assert(std::memcmp(source.data(), original.data(), source.size() * sizeof(float)) == 0);
            assert(bytes.front() == 0xa5 && bytes.back() == 0xa5);
            assert(scales.front() == 0xa5 && scales.back() == 0xa5);
            assert(std::memcmp(bytes.data() + 1, expected.data(), expected.size()) == 0);
            for (int b = 0; b < nblocks; ++b) {
                assert(scales[b + 1] == 127 - 10 + (b % 5) * 5);
            }
            assert(ggml_ame_mxfp8_native_domain(bytes.data() + 1, scales.data() + 1, nblocks, format));
        }
        // RNE ties and subnormal ties must not inherit the host rounding mode.
        const int old_round = std::fegetround();
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD}) {
            assert(std::fesetround(rounding) == 0);
            float input[32] = {};
            input[0] = max_value;
            input[1] = e4 ? 1.0625f : 1.125f;
            input[2] = e4 ? 1.1875f : 1.375f;
            input[3] = -input[1];
            input[4] = std::ldexp(1.0f, e4 ? -10 : -17); // 0.5*min subnormal.
            input[5] = -input[4];
            assert(ggml_ame_pack_mxfp8_f32(payload, &scale, input, 32, format));
            assert(scale == 127);
            assert(payload[1] == (e4 ? 0x38 : 0x3c) && payload[2] == (e4 ? 0x3a : 0x3e));
            assert(payload[3] == (e4 ? 0xb8 : 0xbc) && payload[4] == 0 && payload[5] == 128);
        }
        assert(std::fesetround(old_round) == 0);
        // Exhaust adjacent finite-code midpoints with an independent decoder.
        // Anchor max_value fixes E8M0=127; test both signs and one F32 ULP on
        // either side. Includes zero/subnormal and normal exponent carries.
        unsigned midpoint_cases = 0;
        for (unsigned code = 0; code < max_code; ++code) {
            const float midpoint = (float) ((ame_test_fp8(code, format) + ame_test_fp8(code + 1, format)) * 0.5);
            for (int direction : {-1, 0, 1}) {
                const float value = direction < 0 ? std::nextafter(midpoint, -INFINITY) :
                    direction > 0 ? std::nextafter(midpoint, INFINITY) : midpoint;
                const unsigned expected = direction < 0 ? code : direction > 0 ? code + 1 :
                    code + (code & 1);
                float input[32] = {};
                input[0] = max_value;
                input[1] = value;
                input[2] = -value;
                assert(ggml_ame_pack_mxfp8_f32(payload, &scale, input, 32, format));
                assert(scale == 127 && payload[1] == expected && payload[2] == (expected | 128));
                midpoint_cases += 2;
            }
        }
        unsigned envelope_cases = 0;
        for (int power : {-127, -126, -1, 0, 1, 50, e4 ? 119 : 111}) {
            const float threshold = std::ldexp(max_value, power);
            for (int direction : {-1, 0, 1}) {
                float input[32] = {};
                input[0] = direction < 0 ? std::nextafter(threshold, 0.0f) :
                    direction > 0 ? std::nextafter(threshold, INFINITY) : threshold;
                assert(ggml_ame_pack_mxfp8_f32(payload, &scale, input, 32, format));
                assert(scale == power + 127 + (direction > 0 ? 1 : 0));
                // Above envelope doubles scale; max/2 has one-smaller
                // exponent code but identical mantissa (no special or wrap).
                assert(payload[0] == (direction > 0 ? max_code - (e4 ? 8 : 4) : max_code));
                envelope_cases++;
            }
        }
        float zeros[32] = {};
        zeros[1] = -0.0f;
        assert(ggml_ame_pack_mxfp8_f32(payload, &scale, zeros, 32, format));
        assert(scale == 127 && payload[0] == 0 && payload[1] == 128);
        for (float finite : {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()}) {
            zeros[0] = finite;
            assert(ggml_ame_pack_mxfp8_f32(payload, &scale, zeros, 32, format));
            assert(scale != 255 && ggml_ame_mxfp8_native_domain(payload, &scale, 1, format));
            if (finite == std::numeric_limits<float>::denorm_min()) {
                assert(scale == 0 && payload[0] == 0); // Lower scale clamp, no exponent wrap.
            } else {
                assert(scale == (e4 ? 247 : 240) && payload[0] == 0x78);
            }
        }
        for (int bad : {0, 32, 63}) {
            for (float invalid : {NAN, INFINITY, -INFINITY}) {
                float input[64] = {};
                input[bad] = invalid;
                uint8_t untouched[64], untouched_scales[2];
                std::memset(untouched, 0xa5, sizeof(untouched));
                std::memset(untouched_scales, 0xa5, sizeof(untouched_scales));
                assert(!ggml_ame_mxfp8_f32_domain(input, 64, format));
                assert(!ggml_ame_pack_mxfp8_f32(untouched, untouched_scales, input, 64, format));
                for (uint8_t value : untouched) assert(value == 0xa5);
                for (uint8_t value : untouched_scales) assert(value == 0xa5);
            }
        }
        for (int tail : {0, 1, 31, 33, 63, 65}) {
            assert(!ggml_ame_pack_mxfp8_f32(payload, &scale, zeros, tail, format));
        }
        for (int unknown : {0, 0x0e, 0x10, 0x16, 0xff}) {
            assert(!ggml_ame_pack_mxfp8_f32(payload, &scale, zeros, 32, unknown));
            assert(!ggml_ame_mxfp8_native_domain(payload, &scale, 1, unknown));
        }
        assert(!ggml_ame_pack_mxfp8_f32(nullptr, &scale, zeros, 32, format));
        assert(!ggml_ame_pack_mxfp8_f32(payload, nullptr, zeros, 32, format));
        assert(!ggml_ame_mxfp8_f32_domain(nullptr, 32, format));
        printf("AME_MXFP8_PACK_PASS format=%s mcfg=0x%x bytes=256 scales=256 "
               "K32_blocks=1,2,3,64,129 scale_separation=exact zero=pass RNE=pass "
               "midpoint_ULP_cases=%u scale_envelope_cases=%u clamp=pass "
               "nonfinite_tensor=reject tail=reject hardware=not-run\n",
               e4 ? "e4m3fn" : "e5m2", format, midpoint_cases, envelope_cases);
    }
}

static void test_ame_mx128_packing() {
    unsigned panels = 0, predicates = 0, parser_cases = 0;
    // Test-first M-tail gate: the production cache must pad the final
    // physical weight panel without rejecting logical Stories15M M288.
    assert(ggml_ame_mx_shape_supported(288, 256, 288, 128));
    const struct { const char * name; int tile_mn; } configs[] = {
        {nullptr, 64}, {"", 0}, {"128", 0}, {"unknown", 0},
        {"CUTE_2Tops_64SCP", 64}, {"CUTE_2Tops_128SCP", 128},
        {"CUTE_8Tops_128SCP", 128}, {"CUTE_8Tops_64SCP", 0},
        {"CUTE_8Tops_256SCP", 0}, {"cute_8Tops_128SCP", 0},
        {"CUTE_8Tops_128SCP ", 0}, {"CUTE_8Tops_128SCP-extra", 0},
    };
    for (const auto & config : configs) {
        assert(ggml_ame_mx_config_tile_mn(config.name) == config.tile_mn);
        parser_cases++;
    }
    for (int mn : {64, 128, 256}) {
        ggml_ame_hw_config cfg = {};
        cfg.valid = 1; cfg.rows = mn; cfg.tlenb = (uint64_t) mn * 64;
        cfg.trlenb = 64; cfg.alenb = (uint64_t) mn * mn * 4;
        cfg.arlenb = mn * 4; cfg.melen_bits = 32;
        cfg.int8_m_max = mn; cfg.int8_k_max = 64; cfg.int32_n_max = mn;
        for (const auto & config : configs) {
            const int expected = config.name != nullptr && config.tile_mn == mn ? mn : 0;
            assert(ggml_ame_mx_native_tile_mn(&cfg, config.name) == expected);
            predicates++;
        }
        for (int field = 0; field < 10; ++field) {
            auto bad = cfg;
            if (field == 0) bad.valid = 0;
            if (field == 1) bad.rows++;
            if (field == 2) bad.tlenb++;
            if (field == 3) bad.trlenb++;
            if (field == 4) bad.alenb--;
            if (field == 5) bad.arlenb++;
            if (field == 6) bad.melen_bits = 16;
            if (field == 7) bad.int8_k_max = 32;
            if (field == 8) bad.int8_m_max++;
            if (field == 9) bad.int32_n_max++;
            for (const char * config : {"CUTE_2Tops_128SCP", "CUTE_8Tops_128SCP"}) {
                assert(ggml_ame_mx_native_tile_mn(&bad, config) == 0);
                predicates++;
            }
        }
    }
    for (const char * config : {"CUTE_2Tops_128SCP", "CUTE_8Tops_128SCP"}) {
        assert(ggml_ame_mx_native_tile_mn(nullptr, config) == 0);
        predicates++;
    }
    for (int edge : {0, 64, 127, 129, 192, 255}) {
        assert(ggml_ame_mx_shape_supported(edge, 128, 64, 128) == (edge >= 128));
        assert(!ggml_ame_mx_shape_supported(128, edge, 64, 128));
    }
    const struct { int64_t logical, padded; } row_shapes[] = {
        {128, 128}, {129, 256}, {191, 256}, {192, 256}, {255, 256},
        {256, 256}, {257, 384}, {288, 384}, {383, 384},
    };
    for (const auto & shape : row_shapes) {
        assert(ggml_ame_mx_padded_rows(shape.logical, 128) == shape.padded);
        assert(ggml_ame_mx_shape_supported(shape.logical, 256, 288, 128));
        assert(ggml_ame_mx_select_tile_k(AME_MCFG_MXFP4, shape.logical, 256, 288, 1, "1") == 32);
        assert(ggml_ame_mx_select_tile_k(AME_MCFG_MXFP4, shape.logical, 256, 128, 1, "1") == 64);
        for (int nth : {1, 2, 3, 5}) {
            int64_t previous = 0;
            for (int ith = 0; ith < nth; ++ith) {
                int64_t begin, end;
                ggml_ame_worker_rows(shape.logical, 128, ith, nth, &begin, &end);
                assert(begin == previous && end >= begin && end <= shape.logical);
                assert(begin == shape.logical || begin % 128 == 0);
                assert(end == shape.logical || end % 128 == 0);
                previous = end;
            }
            assert(previous == shape.logical);
        }
    }
    for (int64_t invalid : {int64_t(-1), int64_t(0), int64_t(127), INT64_MAX, INT64_MAX - 126}) {
        assert(ggml_ame_mx_padded_rows(invalid, 128) == 0);
        assert(!ggml_ame_mx_shape_supported(invalid, 128, 64, 128));
    }
    assert(ggml_ame_mx_padded_rows(INT64_MAX - 127, 128) == INT64_MAX - 127);
    assert(!ggml_ame_mx_shape_supported(288, 256, INT64_MAX, 128));
    assert(ggml_ame_mx_padded_rows(288, 0) == 0 && ggml_ame_mx_padded_rows(288, 256) == 0);
    for (int format : {AME_MCFG_MXFP4, AME_MCFG_MXFP8_E4M3, AME_MCFG_MXFP8_E5M2}) {
        for (int tk : {32, 64}) {
            if (!ggml_ame_mx_tile_k_supported(format, tk)) continue;
            const size_t bytes = ggml_ame_mx_block_bytes(format);
            for (const auto & shape : row_shapes) for (int64_t K : {32, 64, 96, 128, 256}) {
                if (K % tk) continue;
                const int64_t M = shape.logical, padded_M = shape.padded;
                const size_t nb = (size_t) K / 32, groups = tk / 32;
                std::vector<block_mxfp4> blocks((size_t) M * nb);
                std::vector<float> fp8_source((size_t) M * (size_t) K);
                const uint8_t top = format == AME_MCFG_MXFP8_E4M3 ? 126 : 123;
                for (int64_t row = 0; row < M; ++row) for (size_t kb = 0; kb < nb; ++kb) {
                    blocks[(size_t) row * nb + kb].e = (uint8_t) ((row * 7 + kb * 31) & 255);
                    for (int k = 0; k < 32; ++k) {
                        const uint8_t nibble = (uint8_t) ((row * 3 + kb * 7 + k) & 15);
                        blocks[(size_t) row * nb + kb].qs[k % 16] |= nibble << (k < 16 ? 0 : 4);
                        const uint8_t code = k == 31 ? top :
                            (uint8_t) (((row * 3 + (int64_t) kb * 7 + k) % (top + 1)) | (k & 1 ? 128 : 0));
                        if (format != AME_MCFG_MXFP4)
                            fp8_source[(size_t) row * K + kb * 32 + k] = (float) ame_test_fp8(code, format);
                    }
                }
                const auto unchanged = blocks;
                const auto unchanged_fp8 = fp8_source;
                std::vector<uint8_t> payload((size_t) padded_M * nb * bytes + 2, 0xa5);
                std::vector<uint8_t> scales((size_t) padded_M * nb + 2, 0xa5);
                for (int64_t row = 0; row < padded_M; ++row) for (size_t kb = 0; kb < nb; ++kb) {
                    const size_t index = ggml_ame_mx_packed_block_index(row, kb, nb, tk, 128);
                    if (row >= M) {
                        std::memset(payload.data() + 1 + index * bytes, 0, bytes);
                        scales[1 + index] = 127;
                    } else if (format == AME_MCFG_MXFP4) {
                        assert(ggml_ame_pack_mxfp4_blocks(payload.data() + 1 + index * bytes,
                            scales.data() + 1 + index, blocks.data() + (size_t) row * nb + kb, 1));
                    } else {
                        assert(ggml_ame_pack_mxfp8_f32(payload.data() + 1 + index * bytes,
                            scales.data() + 1 + index, fp8_source.data() + (size_t) row * K + kb * 32, 32, format));
                    }
                }
                // Independent direct panel/row/K oracle, not production index.
                for (int64_t row = 0; row < padded_M; ++row) for (int64_t k = 0; k < K; ++k) {
                    const size_t panel = (size_t) (row / 128 * (K / tk) + k / tk);
                    const size_t scale = panel * 128 * groups + (size_t) (row % 128) * groups + (k % tk) / 32;
                    const size_t offset = panel * 128 * bytes * groups +
                        (size_t) (row % 128) * bytes * groups + (size_t) (k % tk) / (format == AME_MCFG_MXFP4 ? 2 : 1);
                    if (row >= M) {
                        assert(payload[1 + offset] == 0 && scales[1 + scale] == 127);
                    } else if (format == AME_MCFG_MXFP4) {
                        assert(((payload[1 + offset] >> (4 * (k & 1))) & 15) ==
                            ((row * 3 + k / 32 * 7 + k % 32) & 15));
                        assert(scales[1 + scale] == (uint8_t) ((row * 7 + k / 32 * 31) & 255));
                    } else {
                        const uint8_t expected = k % 32 == 31 ? top :
                            (uint8_t) (((row * 3 + k / 32 * 7 + k % 32) % (top + 1)) | (k & 1 ? 128 : 0));
                        assert(payload[1 + offset] == expected && scales[1 + scale] == 127);
                    }
                }
                assert(payload.front() == 0xa5 && payload.back() == 0xa5 &&
                    scales.front() == 0xa5 && scales.back() == 0xa5);
                assert(std::memcmp(blocks.data(), unchanged.data(), blocks.size() * sizeof(block_mxfp4)) == 0);
                assert(std::memcmp(fp8_source.data(), unchanged_fp8.data(), fp8_source.size() * sizeof(float)) == 0);
                panels += (unsigned) (padded_M / 128 * (K / tk));
            }
        }
    }
    assert(parser_cases == 12 && predicates == 98 && panels == 1220);
    printf("AME_MX128_PACK_PASS panels=%u capacity_predicates=%u parser_cases=%u "
        "exact_configs=2Tops64,2Tops128,8Tops128 formats=MX4,E4M3,E5M2 "
        "tile_MN=128 K32_default_K64_optin=1 nibble_payload_scales=independent_exact "
        "MX4_K64_scale_bytes=256 M_tail_shapes=9 padding=zero_scale127 "
        "N_tail=fallback overflow=reject CSR_AMU=not-issued raw255=preserved\n", panels, predicates, parser_cases);
#if defined(GGML_TEST_RV_AME)
    // Exercise the real preparation seam, not just the lossless packer above.
    // The oracle keeps the old nt/kb/row traversal and unchanged scalar encoder.
    // Physical bytes below are derived from logical K, without production
    // packed_block_index or a call to the tested lossless packing helper.
    std::fenv_t saved_environment;
    assert(std::fegetenv(&saved_environment) == 0);
    unsigned prepare_cases = 0, prepare_blocks = 0, prepare_negatives = 0;
    const auto seed_flags = [](int flags) {
        assert(std::feclearexcept(FE_ALL_EXCEPT) == 0);
        assert(std::feraiseexcept(flags) == 0);
        assert(std::feclearexcept(std::fetestexcept(FE_ALL_EXCEPT) & ~flags) == 0);
        assert(std::fetestexcept(FE_ALL_EXCEPT) == flags);
    };
    const auto check_prepare = [&](int mn, int tk, int K, int N, int padding, int initial_flags) {
        const int nb = K / 32, stride = K + padding;
        std::vector<float> input(size_t(N) * stride + 2, -123.25f);
        for (int row = 0; row < N; ++row) {
            std::fill(input.begin() + 1 + row * stride + K,
                input.begin() + 1 + (row + 1) * stride, std::numeric_limits<float>::quiet_NaN());
            for (int kb = 0; kb < nb; ++kb) {
                const int exponents[] = {-125, -124, 0, 126, 127, 3};
                const int exponent = exponents[(row + kb) % 6];
                const float anchor = std::ldexp((row + kb) & 1 ? -1.0f : 1.0f, exponent);
                float * block = input.data() + 1 + row * stride + kb * 32;
                for (int k = 0; k < 32; ++k) {
                    block[k] = anchor * (float(k % 13 - 6) * 0.125f);
                }
                block[0] = anchor;
                block[1] = -std::nextafter(std::fabs(anchor), 0.0f);
                block[15] = std::numeric_limits<float>::denorm_min();
                block[16] = -std::numeric_limits<float>::denorm_min();
                block[31] = -0.0f;
                if ((row + kb) % 17 == 0) {
                    for (int k = 0; k < 32; ++k) block[k] = k & 1 ? -0.0f : 0.0f;
                }
            }
        }
        const auto unchanged = input;
        std::vector<block_mxfp4> encoded(size_t(N) * nb);
        const size_t bytes = size_t(N) * nb * 16, scale_bytes = size_t(N) * nb;
        std::vector<uint8_t> expected(bytes + 2, 0), expected_scales(scale_bytes + 2, 0xa5);
        expected.front() = expected.back() = 0xa5;
        seed_flags(initial_flags);
        for (int nt = 0; nt < N / mn; ++nt) for (int kb = 0; kb < nb; ++kb) {
            for (int row = nt * mn; row < (nt + 1) * mn; ++row) {
                quantize_row_mxfp4_ref(input.data() + 1 + row * stride + kb * 32,
                    &encoded[size_t(row) * nb + kb], 32);
            }
        }
        const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
        for (int row = 0; row < N; ++row) for (int k = 0; k < K; ++k) {
            const block_mxfp4 & block = encoded[size_t(row) * nb + k / 32];
            const unsigned code = (block.qs[k % 16] >> (k % 32 < 16 ? 0 : 4)) & 15;
            const size_t panel = size_t(row / mn) * (K / tk) + k / tk;
            const size_t offset = (panel * mn + row % mn) * (tk / 2) + k % tk / 2;
            const size_t scale = (panel * mn + row % mn) * (tk / 32) + k % tk / 32;
            expected[1 + offset] |= code << (4 * (k & 1));
            expected_scales[1 + scale] = block.e;
        }
        std::vector<uint8_t> actual(bytes + 2, 0xa5), actual_scales(scale_bytes + 2, 0xa5);
        seed_flags(initial_flags);
        ggml_ame_mx_prepare_activation(actual.data() + 1, actual_scales.data() + 1,
            input.data() + 1, N, K, size_t(stride) * sizeof(float), AME_MCFG_MXFP4, tk, mn);
        const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
        if (actual_flags != expected_flags || actual != expected || actual_scales != expected_scales) {
            fprintf(stderr, "AME_MX_ACTIVATION_PREP_MISMATCH mn=%d tk=%d K=%d N=%d padding=%d "
                "round=%d seed=%x flags_expected=%x flags_actual=%x\n",
                mn, tk, K, N, padding, std::fegetround(), initial_flags, expected_flags, actual_flags);
            assert(false);
        }
        assert(std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0);
        auto corrupt = actual;
        corrupt[bytes] ^= 1;
        assert(corrupt != expected); // Last logical row/panel must be observed.
        corrupt = actual_scales;
        corrupt[scale_bytes] ^= 1;
        assert(corrupt != expected_scales);
        prepare_negatives += 2;
        prepare_blocks += N * nb;
        ++prepare_cases;
    };
    assert(std::fesetround(FE_TONEAREST) == 0);
    for (const auto & shape : std::vector<std::vector<int>>{
            {64, 32, 32, 64, 1}, {64, 64, 64, 128, 17},
            {128, 32, 224, 128, 3}, {128, 64, 256, 128, 0},
            {128, 32, 288, 256, 17}, {128, 64, 320, 128, 1},
            {128, 64, 576, 128, 17}, {128, 64, 768, 128, 3},
            {128, 64, 1024, 128, 1}, {128, 64, 3072, 128, 0}}) {
        check_prepare(shape[0], shape[1], shape[2], shape[3], shape[4], 0);
    }
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        assert(std::fesetround(rounding) == 0);
        for (int seed : {0, FE_INEXACT}) {
            check_prepare(64, 32, 288, 64, 17, seed);
            // K64 8+2 tail, two N panels and padded source stride together.
            // Both values of the only branch-selecting flag are explicit;
            // the smaller case below exhausts all 32 accrued-flag states.
            check_prepare(128, 64, 320, 256, 17, seed);
        }
        const int exceptions[] = {FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW, FE_UNDERFLOW, FE_INEXACT};
        for (unsigned state = 0; state < 32; ++state) {
            int seed = 0;
            for (unsigned bit = 0; bit < 5; ++bit) if (state & (1u << bit)) seed |= exceptions[bit];
            check_prepare(64, 64, 64, 64, 1, seed);
        }
    }
    assert(std::fesetenv(&saved_environment) == 0);
    printf("AME_MX_ACTIVATION_PREP_PASS cases=%u K32_blocks=%u negative_checks=%u "
        "source_and_canaries=exact padded_stride_poison=unread "
        "payload_and_scale=independent_exact accrued_flags=exact rounding_modes=4 "
        "initial_flag_states=32 MN=64,128 tile_K=32,64 CSR_AMU=not-issued\n",
        prepare_cases, prepare_blocks, prepare_negatives);
#endif
}

static void test_ame_mxfp4_activation_error() {
    // Separate the lossy activation change from the bit-exact weight reorder.
    // Deterministic finite inputs, different per-block dynamic ranges, K=96.
    constexpr int M = 9, N = 7, K = 96;
    std::vector<float> weights(M*K), input(N*K), w(M*K), mx(N*K), q8(N*K);
    for (int i = 0; i < M*K; ++i) {
        weights[i] = std::sin(i * 0.17f) * (1 + (i / 32) % 3);
    }
    for (int i = 0; i < N*K; ++i) {
        input[i] = std::cos(i * 0.11f + 0.3f) * (1 + (i / 32) % 4);
    }
    std::vector<block_mxfp4> encoded_w(M*K/32), encoded_x(N*K/32);
    std::vector<block_q8_0> encoded_q8(N*K/32);
    quantize_row_mxfp4_ref(weights.data(), encoded_w.data(), M*K);
    quantize_row_mxfp4_ref(input.data(), encoded_x.data(), N*K);
    quantize_row_q8_0_ref(input.data(), encoded_q8.data(), N*K);
    dequantize_row_mxfp4(encoded_w.data(), w.data(), M*K);
    dequantize_row_mxfp4(encoded_x.data(), mx.data(), N*K);
    dequantize_row_q8_0(encoded_q8.data(), q8.data(), N*K);
    std::vector<uint8_t> payload(encoded_x.size()*16), scales(encoded_x.size());
    assert(ggml_ame_pack_mxfp4_blocks(payload.data(), scales.data(), encoded_x.data(), encoded_x.size()));
    for (int i = 0; i < N*K; ++i) {
        const uint8_t code = (payload[i / 2] >> (4 * (i & 1))) & 15;
        assert(ame_test_e2m1(code, scales[i/32]) == mx[i]);
    }
    for (int baseline = 0; baseline < 2; ++baseline) {
        double sq_error = 0, sq_ref = 0, dot = 0, sq_mx = 0, max_error = 0;
        const float * ref_x = baseline == 0 ? input.data() : q8.data();
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                double ref = 0, candidate = 0;
                for (int k = 0; k < K; ++k) {
                    ref += (double) w[m*K+k] * ref_x[n*K+k];
                    candidate += (double) w[m*K+k] * mx[n*K+k];
                }
                sq_error += (candidate - ref) * (candidate - ref);
                sq_ref += ref * ref;
                dot += candidate * ref;
                sq_mx += candidate * candidate;
                max_error = std::max(max_error, std::fabs(candidate - ref));
            }
        }
        const double nmse = sq_error / sq_ref;
        const double alpha = dot / sq_mx; // Least-squares gain, reported only.
        printf("AME_MXFP4_ACTIVATION_ERROR reference=weight_mxfp4_x_%s alpha=%.9g max_abs=%.9g nmse=%.9g hardware=not-run\n",
            baseline == 0 ? "f32" : "q8_0", alpha, max_error, nmse);
        assert(std::isfinite(nmse) && nmse < 0.1); // Synthetic smoke only, not model acceptance.
    }
}


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    bool mxfp4_runtime_only = false;
    bool mxfp4_vector_only = false;
    bool ame_mxfp4_only = false;
    bool ame_invalid_compute_only = false;
    bool ame_k64_only = false;
    bool ame_q8_k32_only = false;
    bool ame_mx128_only = false;
    bool ame_mx128_native_only = false;
    bool ame_mx128_qwen_native_only = false;
    bool ame_rmsnorm_quant_only = false;
    bool ame_rmsnorm_fusion_only = false;
    bool ame_rmsnorm_fusion_native = false;
    bool ame_rmsnorm_benchmark = false;
    int ame_rmsnorm_fixture = 0;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else if (arg == "--mxfp4-runtime-only") {
            mxfp4_runtime_only = true;
        } else if (arg == "--mxfp4-vector-only") {
            mxfp4_vector_only = true;
        } else if (arg == "--ame-mxfp4-only" || arg == "--ame-mx-only") {
            ame_mxfp4_only = true;
        } else if (arg == "--ame-mx-invalid-compute-only") {
            ame_invalid_compute_only = true;
        } else if (arg == "--ame-mxfp4-k64-only") {
            ame_k64_only = true;
        } else if (arg == "--ame-q8-k32-only") {
            ame_q8_k32_only = true;
        } else if (arg == "--ame-mx128-only") {
            ame_mx128_only = true;
        } else if (arg == "--ame-mx128-native-only") {
            ame_mx128_native_only = true;
        } else if (arg == "--ame-mx128-qwen-native-only") {
            ame_mx128_native_only = true;
            ame_mx128_qwen_native_only = true;
        } else if (arg == "--ame-rmsnorm-quant-only") {
            ame_rmsnorm_quant_only = true;
        } else if (arg == "--ame-rmsnorm-fusion-only" || arg == "--ame-rmsnorm-fusion-native-only") {
            ame_rmsnorm_fusion_only = true;
            ame_rmsnorm_fusion_native = arg == "--ame-rmsnorm-fusion-native-only";
        } else if (arg == "--ame-rmsnorm-fusion-perf-only") {
            ame_rmsnorm_fusion_only = ame_rmsnorm_fusion_native = ame_rmsnorm_benchmark = true;
        } else if (arg == "--ame-rmsnorm-fusion-reuse-only") {
            ame_rmsnorm_fusion_only = ame_rmsnorm_fusion_native = true;
            ame_rmsnorm_fixture = 1;
        } else if (arg == "--ame-rmsnorm-fusion-slots-only" || arg == "--ame-rmsnorm-fusion-tail-only") {
            ame_rmsnorm_fusion_only = ame_rmsnorm_fusion_native = true;
            ame_rmsnorm_fixture = arg == "--ame-rmsnorm-fusion-tail-only" ? 3 : 2;
        } else if (arg == "--ame-rmsnorm-fusion-qkv-only" || arg == "--ame-rmsnorm-fusion-qkv-qwen-only") {
            ame_rmsnorm_fusion_only = ame_rmsnorm_fusion_native = true;
            ame_rmsnorm_fixture = arg == "--ame-rmsnorm-fusion-qkv-qwen-only" ? 5 : 4;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    ggml_cpu_init();
    if (mxfp4_vector_only) {
#if defined(GGML_RV_MXFP4_ENCODE)
        test_mxfp4_vector_encode();
        return 0;
#else
        fprintf(stderr, "MXFP4_VECTOR_ENCODE_UNSUPPORTED reason=experimental_option_off\n");
        return 77;
#endif
    }
    if (ame_rmsnorm_fusion_only) {
#if defined(GGML_TEST_RV_AME)
        return ggml_ame_rmsnorm_fusion_selftest(
            ame_rmsnorm_fusion_native, ame_rmsnorm_fixture, ame_rmsnorm_benchmark) ? 0 : 1;
#else
        GGML_UNUSED(ame_rmsnorm_fusion_native);
        GGML_UNUSED(ame_rmsnorm_fixture);
        GGML_UNUSED(ame_rmsnorm_benchmark);
        fprintf(stderr, "AME_RMS_FUSION_UNSUPPORTED reason=AME_backend_not_built\n");
        return 77;
#endif
    }
    if (ame_rmsnorm_quant_only) {
        test_ame_rmsnorm_quant();
        return 0;
    }
    test_mxfp4_runtime_bytes();
    if (mxfp4_runtime_only) {
        test_mxfp4_supplied_scales();
        return 0;
    }

    if (ame_mx128_only || ame_mx128_native_only) {
        test_ame_mx128_packing();
#if defined(GGML_TEST_RV_AME)
        return ggml_ame_mx128_selftest(ame_mx128_native_only, ame_mx128_qwen_native_only) ? 0 : 1;
#else
        if (ame_mx128_native_only) {
            fprintf(stderr, "AME_MX128_NATIVE_UNSUPPORTED reason=AME_backend_not_built\n");
            return 77;
        }
        printf("AME_MX128_CACHE_SKIP reason=AME_backend_not_built native_numeric=SKIP\n");
        return 0;
#endif
    }
    if (ame_q8_k32_only) {
        test_ame_q8_k32_packing();
#if defined(GGML_TEST_RV_AME)
        return ggml_ame_q8_k32_selftest() ? 0 : 1;
#else
        printf("AME_Q8_K32_SOFTWARE_SKIP reason=AME_backend_not_built native_numeric=SKIP\n");
        return 0;
#endif
    }
    if (ame_invalid_compute_only) {
#if defined(GGML_TEST_RV_AME)
        return ggml_ame_mx_invalid_compute_selftest() ? 0 : 1;
#else
        printf("AME_MX_INVALID_COMPUTE_SKIP reason=AME_backend_not_built\n");
        return 77;
#endif
    }
    if (ame_k64_only) {
        test_ame_mxfp4_k64_packing();
#if defined(GGML_TEST_RV_AME)
        return ggml_ame_mxfp4_k64_selftest() ? 0 : 1;
#else
        printf("AME_MXFP4_K64_CACHE_SKIP reason=AME_backend_not_built\n");
        return 0;
#endif
    }
    test_mxfp4_quant_weights();
    test_ame_mxfp4_packing();
    test_ame_mxfp4_k64_packing();
    test_ame_mxfp4_native_domain();
    test_ame_mxfp8_packing();
    test_ame_mxfp4_activation_error();
#if defined(GGML_TEST_RV_AME)
    assert(ggml_ame_mxfp4_cache_selftest());
    printf("AME_MXFP4_CACHE_PASS zero255=reject sticky=pass generation_retry=pass raw_payload_write=pass raw_scale255=reject partial_set=pass memset=pass clear=pass tensor_identity=pass source_retry=pass M_retry=pass K_retry=pass invalid_pack_allocations=0 native=not-issued\n");
#else
    printf("AME_MXFP4_CACHE_SKIP reason=AME_backend_not_built domain_test=executed\n");
#endif
    if (ame_mxfp4_only) {
        return 0;
    }
    int num_failed = 0;
    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS : MAX_QUANTIZATION_TOTAL_ERROR;
            failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                                            type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                                          ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                                          : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0
                                          ? MAX_DOT_PRODUCT_ERROR_TERNARY
                                          : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
