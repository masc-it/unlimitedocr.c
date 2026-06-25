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

static float test_silu(float value) {
    return value / (1.0f + expf(-value));
}

static int test_dense_swiglu(void) {
    enum { ROWS = 2u, HIDDEN = 2u, INTERMEDIATE = 3u };
    const float input[ROWS * HIDDEN] = {
        1.0f, 2.0f,
        -1.0f, 0.5f,
    };
    const float gate_weight[INTERMEDIATE * HIDDEN] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, -1.0f,
    };
    const float up_weight[INTERMEDIATE * HIDDEN] = {
        2.0f, 0.0f,
        0.0f, 3.0f,
        1.0f, 1.0f,
    };
    const float down_weight[HIDDEN * INTERMEDIATE] = {
        1.0f, 0.0f, 1.0f,
        0.5f, -1.0f, 2.0f,
    };
    float workspace[ROWS * INTERMEDIATE] = {0.0f};
    float out[ROWS * HIDDEN] = {0.0f};

    CHECK(uocr_cpu_ref_dense_swiglu_f32(input,
                                        gate_weight,
                                        up_weight,
                                        down_weight,
                                        ROWS,
                                        HIDDEN,
                                        INTERMEDIATE,
                                        workspace,
                                        out) == 1);

    float expected_workspace[ROWS * INTERMEDIATE] = {0.0f};
    for (uint32_t row = 0u; row < ROWS; ++row) {
        const float x0 = input[(size_t)row * HIDDEN];
        const float x1 = input[(size_t)row * HIDDEN + 1u];
        const float gates[INTERMEDIATE] = {x0, x1, x0 - x1};
        const float ups[INTERMEDIATE] = {2.0f * x0, 3.0f * x1, x0 + x1};
        for (uint32_t inter = 0u; inter < INTERMEDIATE; ++inter) {
            expected_workspace[(size_t)row * INTERMEDIATE + inter] = test_silu(gates[inter]) * ups[inter];
            CHECK(nearly_equal(workspace[(size_t)row * INTERMEDIATE + inter],
                               expected_workspace[(size_t)row * INTERMEDIATE + inter],
                               1.0e-6f));
        }
    }

    for (uint32_t row = 0u; row < ROWS; ++row) {
        const float m0 = expected_workspace[(size_t)row * INTERMEDIATE];
        const float m1 = expected_workspace[(size_t)row * INTERMEDIATE + 1u];
        const float m2 = expected_workspace[(size_t)row * INTERMEDIATE + 2u];
        const float expected0 = m0 + m2;
        const float expected1 = 0.5f * m0 - m1 + 2.0f * m2;
        CHECK(nearly_equal(out[(size_t)row * HIDDEN], expected0, 1.0e-6f));
        CHECK(nearly_equal(out[(size_t)row * HIDDEN + 1u], expected1, 1.0e-6f));
    }

    CHECK(uocr_cpu_ref_dense_swiglu_f32(NULL,
                                        gate_weight,
                                        up_weight,
                                        down_weight,
                                        ROWS,
                                        HIDDEN,
                                        INTERMEDIATE,
                                        workspace,
                                        out) == 0);
    CHECK(uocr_cpu_ref_dense_swiglu_f32(input,
                                        gate_weight,
                                        up_weight,
                                        down_weight,
                                        0u,
                                        HIDDEN,
                                        INTERMEDIATE,
                                        workspace,
                                        out) == 0);
    CHECK(uocr_cpu_ref_dense_swiglu_f32(input,
                                        gate_weight,
                                        up_weight,
                                        down_weight,
                                        ROWS,
                                        HIDDEN,
                                        INTERMEDIATE,
                                        NULL,
                                        out) == 0);
    return 0;
}

static void fill_kv_token(float *k, float *v, uint32_t value, uint32_t stride) {
    for (uint32_t index = 0u; index < stride; ++index) {
        k[index] = (float)(value * 100u + index);
        v[index] = (float)(value * 100u + 1000u + index);
    }
}

static int check_kv_token_value(const float *k,
                                const float *v,
                                uint32_t value,
                                uint32_t stride) {
    for (uint32_t index = 0u; index < stride; ++index) {
        CHECK(k[index] == (float)(value * 100u + index));
        CHECK(v[index] == (float)(value * 100u + 1000u + index));
    }
    return 0;
}

static int test_kv_cache_ring(void) {
    enum { PROMPT_CAP = 4u, RING = 3u, HEADS = 2u, HEAD_DIM = 2u, PROMPT_LEN = 3u, STRIDE = HEADS * HEAD_DIM };
    uocr_cpu_ref_kv_cache_layout layout;
    CHECK(uocr_cpu_ref_kv_cache_layout_init(PROMPT_CAP, RING, HEADS, HEAD_DIM, &layout) == 1);
    CHECK(layout.cache_token_capacity == PROMPT_CAP + RING);
    CHECK(layout.token_stride_floats == STRIDE);
    CHECK(layout.total_floats == (PROMPT_CAP + RING) * STRIDE);
    CHECK(uocr_cpu_ref_kv_cache_layout_init(0u, RING, HEADS, HEAD_DIM, &layout) == 0);
    CHECK(uocr_cpu_ref_kv_cache_layout_init(PROMPT_CAP, 0u, HEADS, HEAD_DIM, &layout) == 0);

    float k_cache[(PROMPT_CAP + RING) * STRIDE];
    float v_cache[(PROMPT_CAP + RING) * STRIDE];
    for (uint32_t index = 0u; index < (PROMPT_CAP + RING) * STRIDE; ++index) {
        k_cache[index] = -1.0f;
        v_cache[index] = -2.0f;
    }

    float k_token[STRIDE];
    float v_token[STRIDE];
    for (uint32_t position = 0u; position < 8u; ++position) {
        fill_kv_token(k_token, v_token, position, STRIDE);
        CHECK(uocr_cpu_ref_kv_cache_write_token_f32(k_cache, v_cache, &layout, PROMPT_LEN, position, k_token, v_token) == 1);
    }

    uint32_t cache_token = UINT32_MAX;
    CHECK(uocr_cpu_ref_kv_cache_token_for_position(PROMPT_LEN, &layout, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_cpu_ref_kv_cache_token_for_position(PROMPT_LEN, &layout, 2u, &cache_token) == 1);
    CHECK(cache_token == 2u);
    CHECK(uocr_cpu_ref_kv_cache_token_for_position(PROMPT_LEN, &layout, 3u, &cache_token) == 1);
    CHECK(cache_token == 3u);
    CHECK(uocr_cpu_ref_kv_cache_token_for_position(PROMPT_LEN, &layout, 4u, &cache_token) == 1);
    CHECK(cache_token == 4u);
    CHECK(uocr_cpu_ref_kv_cache_token_for_position(PROMPT_LEN, &layout, 5u, &cache_token) == 1);
    CHECK(cache_token == 5u);
    CHECK(uocr_cpu_ref_kv_cache_token_for_position(PROMPT_LEN, &layout, 6u, &cache_token) == 1);
    CHECK(cache_token == 3u);
    CHECK(uocr_cpu_ref_kv_cache_token_for_position(PROMPT_CAP + 1u, &layout, 0u, &cache_token) == 0);

    float k_read[STRIDE];
    float v_read[STRIDE];
    CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, 0u, k_read, v_read) == 1);
    CHECK(check_kv_token_value(k_read, v_read, 0u, STRIDE) == 0);
    CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, 1u, k_read, v_read) == 1);
    CHECK(check_kv_token_value(k_read, v_read, 1u, STRIDE) == 0);
    CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, 2u, k_read, v_read) == 1);
    CHECK(check_kv_token_value(k_read, v_read, 2u, STRIDE) == 0);
    CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, 3u, k_read, v_read) == 1);
    CHECK(check_kv_token_value(k_read, v_read, 6u, STRIDE) == 0);
    CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, 4u, k_read, v_read) == 1);
    CHECK(check_kv_token_value(k_read, v_read, 7u, STRIDE) == 0);
    CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, 5u, k_read, v_read) == 1);
    CHECK(check_kv_token_value(k_read, v_read, 5u, STRIDE) == 0);
    CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, layout.cache_token_capacity, k_read, v_read) == 0);

    uocr_cpu_ref_decode_attention_plan plan;
    CHECK(uocr_cpu_ref_kv_cache_decode_attention_plan(PROMPT_LEN, &layout, 5u, &plan) == 1);
    CHECK(plan.generated_count == 5u);
    CHECK(plan.live_generated == RING);
    CHECK(plan.first_generated_index == 2u);
    CHECK(plan.first_generated_position == 5u);
    CHECK(plan.query_position == 7u);
    CHECK(plan.attention_length == PROMPT_LEN + RING);

    const uint32_t expected_attention_tokens[PROMPT_LEN + RING] = {0u, 1u, 2u, 5u, 3u, 4u};
    const uint32_t expected_attention_values[PROMPT_LEN + RING] = {0u, 1u, 2u, 5u, 6u, 7u};
    for (uint32_t attention_index = 0u; attention_index < plan.attention_length; ++attention_index) {
        CHECK(uocr_cpu_ref_kv_cache_decode_attention_index_to_token(&plan, attention_index, &cache_token) == 1);
        CHECK(cache_token == expected_attention_tokens[attention_index]);
        CHECK(uocr_cpu_ref_kv_cache_read_token_f32(k_cache, v_cache, &layout, cache_token, k_read, v_read) == 1);
        CHECK(check_kv_token_value(k_read, v_read, expected_attention_values[attention_index], STRIDE) == 0);
    }
    CHECK(uocr_cpu_ref_kv_cache_decode_attention_index_to_token(&plan, plan.attention_length, &cache_token) == 0);

    CHECK(uocr_cpu_ref_kv_cache_decode_attention_plan(PROMPT_LEN, &layout, 0u, &plan) == 1);
    CHECK(plan.live_generated == 0u);
    CHECK(plan.attention_length == PROMPT_LEN);
    CHECK(plan.query_position == PROMPT_LEN);
    CHECK(uocr_cpu_ref_kv_cache_decode_attention_plan(0u, &layout, 1u, &plan) == 0);
    CHECK(uocr_cpu_ref_kv_cache_decode_attention_plan(PROMPT_CAP + 1u, &layout, 1u, &plan) == 0);
    return 0;
}

int main(void) {
    if (test_dtype_conversions() != 0) return 1;
    if (test_rmsnorm() != 0) return 1;
    if (test_rope_split_half() != 0) return 1;
    if (test_causal_sdpa() != 0) return 1;
    if (test_dense_swiglu() != 0) return 1;
    if (test_kv_cache_ring() != 0) return 1;
    return 0;
}
