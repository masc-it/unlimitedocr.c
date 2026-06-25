#include "backend/cpu_ref/uocr_cpu_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

static int nearly_equal(float actual, float expected, float tol) {
    return fabsf(actual - expected) <= tol;
}

static int test_dtype_conversions(void) {
    CHECK(strcmp(uocr_cpu_ref_backend_name(), "cpu-ref") == 0);

    CHECK(uocr_cpu_ref_bf16_bits_to_f32(0x3f80u) == 1.0f);
    CHECK(uocr_cpu_ref_bf16_bits_to_f32(0xc020u) == -2.5f);
    CHECK(uocr_cpu_ref_f16_bits_to_f32(0x3c00u) == 1.0f);
    CHECK(uocr_cpu_ref_f16_bits_to_f32(0xc000u) == -2.0f);
    CHECK(uocr_cpu_ref_f32_to_f16_bits(0.0f) == 0x0000u);
    CHECK(uocr_cpu_ref_f32_to_f16_bits(-0.0f) == 0x8000u);
    CHECK(uocr_cpu_ref_f32_to_f16_bits(1.0f) == 0x3c00u);
    CHECK(uocr_cpu_ref_f32_to_f16_bits(-2.0f) == 0xc000u);
    CHECK(uocr_cpu_ref_f32_to_f16_bits(65504.0f) == 0x7bffu);
    CHECK(uocr_cpu_ref_f32_to_f16_bits(INFINITY) == 0x7c00u);
    CHECK((uocr_cpu_ref_f32_to_f16_bits(NAN) & 0x7c00u) == 0x7c00u);

    const uint16_t bf16_values[3] = {0x3f80u, 0x4000u, 0xc040u};
    float f32_values[3] = {0.0f, 0.0f, 0.0f};
    CHECK(uocr_cpu_ref_bf16_to_f32_array(bf16_values, 3u, f32_values) == 1);
    CHECK(f32_values[0] == 1.0f);
    CHECK(f32_values[1] == 2.0f);
    CHECK(f32_values[2] == -3.0f);
    CHECK(uocr_cpu_ref_bf16_to_f32_array(NULL, 3u, f32_values) == 0);

    const float src[4] = {1.0f, -2.0f, 0.5f, 65504.0f};
    uint16_t f16_values[4] = {0u, 0u, 0u, 0u};
    CHECK(uocr_cpu_ref_f32_to_f16_array(src, 4u, f16_values) == 1);
    CHECK(f16_values[0] == 0x3c00u);
    CHECK(f16_values[1] == 0xc000u);
    CHECK(f16_values[2] == 0x3800u);
    CHECK(f16_values[3] == 0x7bffu);

    float roundtrip[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    CHECK(uocr_cpu_ref_f16_to_f32_array(f16_values, 4u, roundtrip) == 1);
    CHECK(roundtrip[0] == 1.0f);
    CHECK(roundtrip[1] == -2.0f);
    CHECK(roundtrip[2] == 0.5f);
    CHECK(roundtrip[3] == 65504.0f);
    CHECK(uocr_cpu_ref_f32_to_f16_array(src, 4u, NULL) == 0);
    return 0;
}

static int test_rmsnorm(void) {
    const float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, -1.0f, 1.0f, 0.5f};
    const float weight[4] = {1.0f, 0.5f, -1.0f, 2.0f};
    float out[8] = {0.0f};
    CHECK(uocr_cpu_ref_rmsnorm_f32(input, weight, 2u, 4u, 1.0e-6f, out) == 1);

    for (uint32_t row = 0u; row < 2u; ++row) {
        float sum_squares = 0.0f;
        for (uint32_t col = 0u; col < 4u; ++col) {
            const float value = input[(size_t)row * 4u + col];
            sum_squares += value * value;
        }
        const float inv_rms = 1.0f / sqrtf(sum_squares / 4.0f + 1.0e-6f);
        for (uint32_t col = 0u; col < 4u; ++col) {
            const float expected = input[(size_t)row * 4u + col] * inv_rms * weight[col];
            CHECK(nearly_equal(out[(size_t)row * 4u + col], expected, 1.0e-6f));
        }
    }
    CHECK(uocr_cpu_ref_rmsnorm_f32(NULL, weight, 2u, 4u, 1.0e-6f, out) == 0);
    CHECK(uocr_cpu_ref_rmsnorm_f32(input, weight, 0u, 4u, 1.0e-6f, out) == 0);
    CHECK(uocr_cpu_ref_rmsnorm_f32(input, weight, 2u, 4u, -1.0f, out) == 0);
    return 0;
}

static int test_rope_split_half(void) {
    const float q[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
    };
    const float k[8] = {
        -1.0f, -2.0f, -3.0f, -4.0f,
        -1.0f, -2.0f, -3.0f, -4.0f,
    };
    float q_out[8] = {0.0f};
    float k_out[8] = {0.0f};
    CHECK(uocr_cpu_ref_rope_split_half_f32(q, k, 2u, 1u, 4u, 0u, 10000.0f, q_out, k_out) == 1);

    for (uint32_t index = 0u; index < 4u; ++index) {
        CHECK(q_out[index] == q[index]);
        CHECK(k_out[index] == k[index]);
    }

    const float cos0 = cosf(1.0f);
    const float sin0 = sinf(1.0f);
    const float cos1 = cosf(0.01f);
    const float sin1 = sinf(0.01f);
    const float expected_q[4] = {
        1.0f * cos0 - 3.0f * sin0,
        2.0f * cos1 - 4.0f * sin1,
        3.0f * cos0 + 1.0f * sin0,
        4.0f * cos1 + 2.0f * sin1,
    };
    const float expected_k[4] = {
        -1.0f * cos0 - (-3.0f) * sin0,
        -2.0f * cos1 - (-4.0f) * sin1,
        -3.0f * cos0 + (-1.0f) * sin0,
        -4.0f * cos1 + (-2.0f) * sin1,
    };
    for (uint32_t index = 0u; index < 4u; ++index) {
        CHECK(nearly_equal(q_out[4u + index], expected_q[index], 1.0e-6f));
        CHECK(nearly_equal(k_out[4u + index], expected_k[index], 1.0e-6f));
    }

    CHECK(uocr_cpu_ref_rope_split_half_f32(q, k, 2u, 1u, 3u, 0u, 10000.0f, q_out, k_out) == 0);
    CHECK(uocr_cpu_ref_rope_split_half_f32(q, k, 2u, 1u, 4u, 0u, 0.0f, q_out, k_out) == 0);
    return 0;
}

static float dot2(const float *q, const float *k, uint32_t query_token, uint32_t key_token) {
    return q[(size_t)query_token * 2u] * k[(size_t)key_token * 2u] +
           q[(size_t)query_token * 2u + 1u] * k[(size_t)key_token * 2u + 1u];
}

static float sdpa_expected_dim(const float *q,
                               const float *k,
                               const float *v,
                               uint32_t query_token,
                               uint32_t dim) {
    float max_score = -INFINITY;
    for (uint32_t key_token = 0u; key_token <= query_token; ++key_token) {
        const float score = dot2(q, k, query_token, key_token);
        if (score > max_score) {
            max_score = score;
        }
    }
    float denom = 0.0f;
    for (uint32_t key_token = 0u; key_token <= query_token; ++key_token) {
        denom += expf(dot2(q, k, query_token, key_token) - max_score);
    }
    float out = 0.0f;
    for (uint32_t key_token = 0u; key_token <= query_token; ++key_token) {
        const float weight = expf(dot2(q, k, query_token, key_token) - max_score) / denom;
        out += weight * v[(size_t)key_token * 2u + dim];
    }
    return out;
}

static int test_causal_sdpa(void) {
    const float q[6] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float k[6] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float v[6] = {10.0f, 0.0f, 0.0f, 20.0f, 30.0f, 40.0f};
    float out[6] = {0.0f};
    CHECK(uocr_cpu_ref_causal_sdpa_f32(q, k, v, 3u, 1u, 2u, 1.0f, out) == 1);

    for (uint32_t token = 0u; token < 3u; ++token) {
        for (uint32_t dim = 0u; dim < 2u; ++dim) {
            const float expected = sdpa_expected_dim(q, k, v, token, dim);
            CHECK(nearly_equal(out[(size_t)token * 2u + dim], expected, 1.0e-6f));
        }
    }
    CHECK(out[0] == 10.0f);
    CHECK(out[1] == 0.0f);
    CHECK(uocr_cpu_ref_causal_sdpa_f32(NULL, k, v, 3u, 1u, 2u, 1.0f, out) == 0);
    CHECK(uocr_cpu_ref_causal_sdpa_f32(q, k, v, 0u, 1u, 2u, 1.0f, out) == 0);
    return 0;
}

int main(void) {
    if (test_dtype_conversions() != 0) return 1;
    if (test_rmsnorm() != 0) return 1;
    if (test_rope_split_half() != 0) return 1;
    if (test_causal_sdpa() != 0) return 1;
    return 0;
}
