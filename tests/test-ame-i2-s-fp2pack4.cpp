#include "ggml-quants.h"
#include "quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Host I2_S stores four 2-bit values from four 32-element groups in one byte.
static uint8_t code_at(const uint8_t * packed, int64_t index) {
    const uint8_t byte = packed[(size_t) (index / 128) * 32 + (size_t) (index % 32)];
    return (byte >> (6 - 2 * ((index % 128) / 32))) & 0x3;
}

static uint8_t i2_to_fp2(uint8_t code) {
    return code == 0 ? 3 : (code == 2 ? 1 : 0);
}

static void pack_fp2pack4_row64(const uint8_t * i2_row, int64_t k0, uint8_t * packed) {
    for (int64_t k = 0; k < 64; k += 4) {
        packed[k / 4] = i2_to_fp2(code_at(i2_row, k0 + k + 0)) |
                        (i2_to_fp2(code_at(i2_row, k0 + k + 1)) << 2) |
                        (i2_to_fp2(code_at(i2_row, k0 + k + 2)) << 4) |
                        (i2_to_fp2(code_at(i2_row, k0 + k + 3)) << 6);
    }
}

static bool test_host_and_fp2_equivalence() {
    constexpr int64_t M = 129;
    constexpr int64_t N = 67;
    constexpr int64_t K = 1152;
    static_assert(M != N, "the layout regression test must use an asymmetric shape");
    const size_t row_bytes = (size_t) K / 4;

    std::vector<float> source((size_t) M * K);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            const int group = (int) ((k % 128) / 32);
            source[(size_t) m * K + k] = group == 0 ? -2.5f : group == 2 ? 2.5f : 0.0f;
        }
    }

    std::vector<uint8_t> weights((size_t) M * row_bytes + 32, 0);
    if (quantize_i2_s(source.data(), weights.data(), M, K, nullptr) != weights.size()) {
        std::fprintf(stderr, "unexpected I2_S storage size\n");
        return false;
    }
    const float weight_scale = reinterpret_cast<const float *>(weights.data() + (size_t) M * row_bytes)[0];

    if (i2_to_fp2(0) != 3 || i2_to_fp2(1) != 0 || i2_to_fp2(2) != 1 || i2_to_fp2(3) != 0) {
        std::fprintf(stderr, "I2_S to FP2PACK4 code map is incorrect\n");
        return false;
    }

    // Check the GGML byte order and the exact FP2 lane conversion independently
    // of the matrix dimensions.
    for (int64_t k = 0; k < 32; ++k) {
        if (weights[(size_t) k] != 0x19 || i2_to_fp2(code_at(weights.data(), k)) != 3) {
            std::fprintf(stderr, "I2_S/FP2 pack mismatch at lane %lld: byte=0x%02x\n",
                         (long long) k, weights[(size_t) k]);
            return false;
        }
    }
    uint8_t fp2_panel[16] = {};
    pack_fp2pack4_row64(weights.data(), 0, fp2_panel);
    for (int byte = 0; byte < 16; ++byte) {
        const uint8_t expected = byte < 8 ? 0xff : 0x00;
        if (fp2_panel[byte] != expected) {
            std::fprintf(stderr, "FP2PACK4 lane order mismatch at byte=%d: got=0x%02x expected=0x%02x\n",
                         byte, fp2_panel[byte], expected);
            return false;
        }
    }

    std::vector<float> activations((size_t) N * K);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            activations[(size_t) n * K + k] =
                (float) (((n * 13 + k * 3) % 37) - 18) / 7.0f;
        }
    }

    std::vector<int8_t> q((size_t) K);
    std::vector<float> host_output((size_t) M * N);
    std::vector<float> native_output((size_t) M * N);
    std::vector<float> row_major_output((size_t) M * N);
    for (int64_t n = 0; n < N; ++n) {
        float act_scale = 0.0f;
        int32_t act_sum = 0;
        quantize_row_i8_s(activations.data() + (size_t) n * K, q.data(), K, &act_scale, &act_sum);

        for (int64_t m = 0; m < M; ++m) {
            float raw_dot = 0.0f;
            ggml_vec_dot_i2_i8_s(
                (int) K, &raw_dot, 0,
                weights.data() + (size_t) m * row_bytes, row_bytes,
                q.data(), 0, 1);

            int32_t fp2_dot = 0;
            for (int64_t k = 0; k < K; ++k) {
                const uint8_t code = code_at(weights.data() + (size_t) m * row_bytes, k);
                fp2_dot += (int32_t) q[(size_t) k] *
                           (code == 0 ? -1 : code == 2 ? 1 : 0);
            }
            const float host = (raw_dot - (float) act_sum) / act_scale * weight_scale;
            const float native = (float) fp2_dot / act_scale * weight_scale;
            if (std::abs(host - native) > 1e-5f * std::max(1.0f, std::abs(host))) {
                std::fprintf(stderr, "host/FP2 mismatch at m=%lld n=%lld: %.9g vs %.9g\n",
                             (long long) m, (long long) n, host, native);
                return false;
            }
            host_output[(size_t) n * M + m] = host;
            native_output[(size_t) n * M + m] = native;
            row_major_output[(size_t) m * N + n] = native;
        }
    }
    for (size_t index = 0; index < host_output.size(); ++index) {
        if (std::abs(host_output[index] - native_output[index]) >
            1e-5f * std::max(1.0f, std::abs(host_output[index]))) {
            std::fprintf(stderr, "native output layout mismatch at %zu\n", index);
            return false;
        }
    }
    if (host_output == row_major_output) {
        std::fprintf(stderr, "row-major output unexpectedly matches GGML [M, N] layout\n");
        return false;
    }
    return true;
}

static bool test_flat_tail_decode() {
    constexpr int64_t M = 3;
    constexpr int64_t K = 4304; // deliberately not a multiple of 128
    const int64_t elements = M * K;
    std::vector<uint8_t> packed((size_t) ((elements + 127) / 128) * 32 + 32, 0);

    for (int64_t index = 0; index < elements; ++index) {
        const uint8_t code = (uint8_t) ((index % 3) == 0 ? 0 : (index % 3) == 1 ? 1 : 2);
        packed[(size_t) (index / 128) * 32 + (size_t) (index % 32)] |=
            (uint8_t) (code << (6 - 2 * ((index % 128) / 32)));
    }

    for (int64_t row = 0; row < M; ++row) {
        for (int64_t k = 0; k < K; ++k) {
            const int64_t index = row * K + k;
            const uint8_t expected = (uint8_t) ((index % 3) == 0 ? 0 : (index % 3) == 1 ? 1 : 2);
            if (code_at(packed.data(), index) != expected) {
                std::fprintf(stderr, "flat I2_S tail decode mismatch at row=%lld k=%lld\n",
                             (long long) row, (long long) k);
                return false;
            }
        }
    }
    return true;
}

int main() {
    if (!test_host_and_fp2_equivalence() || !test_flat_tail_decode()) {
        return 1;
    }
    std::puts("AME I2_S/FP2PACK4 host equivalence: PASS");
    return 0;
}
