#include "backend/metal/uocr_metal.h"
#include "model/uocr_model_file.h"
#include "runtime/uocr_memory.h"
#include "unlimitedocr.h"

#include "uocr_test_model_file.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef UOCR_TEST_METAL_RESOURCE_PATH
#define UOCR_TEST_METAL_RESOURCE_PATH NULL
#endif

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static int test_metal_smoke(void) {
    if (!uocr_metal_is_available()) {
        printf("Metal device not available; skipping Metal smoke test\n");
        return 0;
    }

    const uint64_t recommended = uocr_metal_recommended_working_set_size();
    CHECK(recommended > 0u);
    const uint64_t default_budget = uocr_metal_default_memory_budget_bytes(recommended);
    CHECK(default_budget > 0u);
    CHECK(default_budget <= recommended);

    char error[1024];
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_smoke_test(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    return 0;
}

static int test_metal_named_scratch_buffers(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) == 0u);
    CHECK(uocr_metal_context_total_scratch_high_watermark(ctx) == 0u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 1024u, 0, error, sizeof(error)) == 1);
    const uint64_t first_capacity = uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(first_capacity >= 1024u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 512u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, first_capacity + 1u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) >= first_capacity + 1u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_LOGITS, 4096u, 1, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS) >= 4096u);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) >=
          uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) +
              uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS));

    uocr_metal_context_release_scratch(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == 0u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16u;
    uint32_t bits = 0u;
    uint32_t mantissa = (uint32_t)h & 0x03ffu;
    int exponent = (int)((h >> 10u) & 0x001fu);
    if (exponent == 0) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            mantissa &= ~0x0400u;
            bits = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
    }
    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static int test_metal_get_rows_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TABLE_ROWS = 5, ROW_WIDTH = 8, OUT_ROWS = 4 };
    const uint16_t values[] = {
        0x0000u, /* 0.0 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x3800u, /* 0.5 */
        0xb800u, /* -0.5 */
        0x4200u, /* 3.0 */
        0xc200u, /* -3.0 */
        0x4400u  /* 4.0 */
    };
    uint16_t table[TABLE_ROWS * ROW_WIDTH];
    for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * ROW_WIDTH); ++i) {
        table[i] = values[(i * 3u + 1u) % (sizeof(values) / sizeof(values[0]))];
    }
    const int32_t row_ids[OUT_ROWS] = {3, 0, 4, 1};

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t out_f16[OUT_ROWS * ROW_WIDTH];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          row_ids,
                                          OUT_ROWS,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                          out_f16,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)ROW_WIDTH; ++col) {
            const uint16_t expected = table[(uint32_t)row_ids[row] * (uint32_t)ROW_WIDTH + col];
            CHECK(out_f16[row * (uint32_t)ROW_WIDTH + col] == expected);
        }
    }

    float out_f32[OUT_ROWS * ROW_WIDTH];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          row_ids,
                                          OUT_ROWS,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F32,
                                          out_f32,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)ROW_WIDTH; ++col) {
            const uint16_t expected_bits = table[(uint32_t)row_ids[row] * (uint32_t)ROW_WIDTH + col];
            CHECK(out_f32[row * (uint32_t)ROW_WIDTH + col] == f16_bits_to_f32(expected_bits));
        }
    }

    const int32_t bad_row_ids[1] = {TABLE_ROWS};
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          bad_row_ids,
                                          1u,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                          out_f16,
                                          error,
                                          sizeof(error)) == 0);
    CHECK(strstr(error, "outside table rows") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_prompt_assembly_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TABLE_ROWS = 6, HIDDEN = 8, TEXT_TOKENS = 4, IMAGE_TOKENS = 2, PROMPT_TOKENS = 6 };
    uint16_t table[TABLE_ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)TABLE_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            table[row * (uint32_t)HIDDEN + col] = (uint16_t)(0x3c00u + row * 0x40u + col);
        }
    }
    uint16_t image_features[IMAGE_TOKENS * HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
        image_features[i] = (uint16_t)(0x5000u + i);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const int32_t text_ids[TEXT_TOKENS] = {0, 3, 5, 1};
    uint16_t text_out[TEXT_TOKENS * HIDDEN];
    memset(text_out, 0, sizeof(text_out));
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 text_ids,
                                                 TEXT_TOKENS,
                                                 UINT32_MAX,
                                                 0u,
                                                 NULL,
                                                 text_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)TEXT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(text_out[token * (uint32_t)HIDDEN + col] == table[(uint32_t)text_ids[token] * (uint32_t)HIDDEN + col]);
        }
    }

    const int32_t prompt_ids[PROMPT_TOKENS] = {0, 2, 12345, 12346, 4, 1};
    uint16_t prompt_out[PROMPT_TOKENS * HIDDEN];
    memset(prompt_out, 0, sizeof(prompt_out));
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 prompt_ids,
                                                 PROMPT_TOKENS,
                                                 2u,
                                                 IMAGE_TOKENS,
                                                 image_features,
                                                 prompt_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)PROMPT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint16_t expected = 0u;
            if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
            } else {
                expected = table[(uint32_t)prompt_ids[token] * (uint32_t)HIDDEN + col];
            }
            CHECK(prompt_out[token * (uint32_t)HIDDEN + col] == expected);
        }
    }

    const int32_t bad_text_ids[1] = {TABLE_ROWS};
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 bad_text_ids,
                                                 1u,
                                                 UINT32_MAX,
                                                 0u,
                                                 NULL,
                                                 text_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "outside embedding rows") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_rmsnorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, HIDDEN = 1280 };
    const uint16_t input_values[] = {
        0x3800u, /* 0.5 */
        0xb800u, /* -0.5 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x3400u, /* 0.25 */
        0xb400u  /* -0.25 */
    };
    const uint16_t weight_values[] = {
        0x3c00u, /* 1.0 */
        0x3800u, /* 0.5 */
        0x4000u, /* 2.0 */
        0x3e00u, /* 1.5 */
        0x3a00u  /* 0.75 */
    };
    uint16_t input[ROWS * HIDDEN];
    uint16_t weight[HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        input[i] = input_values[(i * 5u + 3u) % (sizeof(input_values) / sizeof(input_values[0]))];
    }
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        weight[i] = weight_values[(i * 7u + 1u) % (sizeof(weight_values) / sizeof(weight_values[0]))];
    }

    const float eps = 1.0e-6f;
    float expected[ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            sum += x * x;
        }
        const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + eps);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            const float w = f16_bits_to_f32(weight[col]);
            expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    float out_f32[ROWS * HIDDEN];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         eps,
                                         UOCR_METAL_RMSNORM_OUTPUT_F32,
                                         out_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-4f);
    }

    uint16_t out_f16[ROWS * HIDDEN];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         eps,
                                         UOCR_METAL_RMSNORM_OUTPUT_F16,
                                         out_f16,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 4.0e-3f);
    }

    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         0.0f,
                                         UOCR_METAL_RMSNORM_OUTPUT_F32,
                                         out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "eps must be positive") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_dense_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, IN_FEATURES = 1280, OUT_FEATURES = 96 };
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint16_t weight_values[] = {
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint16_t bias_values[] = {
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t bias_value_count = (uint32_t)(sizeof(bias_values) / sizeof(bias_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)ROWS * IN_FEATURES * sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)malloc((size_t)OUT_FEATURES * IN_FEATURES * sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)malloc((size_t)OUT_FEATURES * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(uint16_t));
    CHECK(input != NULL && weight != NULL && bias != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(ROWS * IN_FEATURES); ++i) {
        input[i] = input_values[(i * 17u + i / 29u + 1u) % input_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)(OUT_FEATURES * IN_FEATURES); ++i) {
        weight[i] = weight_values[(i * 11u + i / 37u + 3u) % weight_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        bias[i] = bias_values[(i * 5u + 1u) % bias_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
            float sum = f16_bits_to_f32(bias[col]);
            for (uint32_t k = 0u; k < (uint32_t)IN_FEATURES; ++k) {
                sum += f16_bits_to_f32(input[row * (uint32_t)IN_FEATURES + k]) *
                       f16_bits_to_f32(weight[col * (uint32_t)IN_FEATURES + k]);
            }
            expected[row * (uint32_t)OUT_FEATURES + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       ROWS,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * OUT_FEATURES); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.5e-3f);
    }

    memset(out_f16, 0, (size_t)ROWS * OUT_FEATURES * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       1u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                       out_f16,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 5.0e-2f);
    }

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       NULL,
                                       1u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
        CHECK(fabsf(out_f32[col] - (expected[col] - f16_bits_to_f32(bias[col]))) < 2.5e-3f);
    }

    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       ROWS,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       (uocr_metal_dense_output_type)99,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal dense output type") != NULL);
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       0u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal dense request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_attention_qkvo_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = 1280, PROJECTIONS = 4 };
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint16_t weight_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *weights[PROJECTIONS];
    float *expected[PROJECTIONS];
    float *out_f32[PROJECTIONS];
    uint16_t *out_f16[PROJECTIONS];
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        weights[p] = (uint16_t *)malloc((size_t)HIDDEN * HIDDEN * sizeof(uint16_t));
        expected[p] = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
        out_f32[p] = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
        out_f16[p] = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
        CHECK(weights[p] != NULL && expected[p] != NULL && out_f32[p] != NULL && out_f16[p] != NULL);
    }
    CHECK(input != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 19u + i / 23u + 2u) % input_value_count];
    }
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)(HIDDEN * HIDDEN); ++i) {
            weights[p][i] = weight_values[(i * (7u + p * 2u) + i / 41u + p * 3u) % weight_value_count];
        }
    }

    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                float sum = 0.0f;
                for (uint32_t k = 0u; k < (uint32_t)HIDDEN; ++k) {
                    sum += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k]) *
                           f16_bits_to_f32(weights[p][col * (uint32_t)HIDDEN + k]);
                }
                expected[p][token * (uint32_t)HIDDEN + col] = sum;
            }
        }
        memset(out_f32[p], 0, (size_t)TOKENS * HIDDEN * sizeof(float));
        memset(out_f16[p], 0, (size_t)HIDDEN * sizeof(uint16_t));
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                TOKENS,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
            CHECK(fabsf(out_f32[p][i] - expected[p][i]) < 4.0e-3f);
        }
    }

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                1u,
                                                UOCR_METAL_DENSE_OUTPUT_F16,
                                                out_f16[0],
                                                out_f16[1],
                                                out_f16[2],
                                                out_f16[3],
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
            CHECK(fabsf(f16_bits_to_f32(out_f16[p][i]) - expected[p][i]) < 1.5e-1f);
        }
    }

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                0u,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal attention projection request") != NULL);
    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                TOKENS,
                                                (uocr_metal_dense_output_type)99,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal attention projection output type") != NULL);

    uocr_metal_context_destroy(ctx);
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        free(out_f16[p]);
        free(out_f32[p]);
        free(expected[p]);
        free(weights[p]);
    }
    free(input);
    return 0;
}

static int test_metal_recent_decoder_primitives_stress(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint16_t half_pool[] = {
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x4200u, /* 3.0 */
        0x5000u  /* 32.0; catches accidental fp16 RMS variance overflow */
    };
    const uint32_t half_pool_count = (uint32_t)(sizeof(half_pool) / sizeof(half_pool[0]));

    {
        enum { TABLE_ROWS = 257, HIDDEN = 1280, OUT_ROWS = 17 };
        uint16_t table[TABLE_ROWS * HIDDEN];
        uint16_t out[OUT_ROWS * HIDDEN];
        const int32_t row_ids[OUT_ROWS] = {256, 0, 128, 7, 7, 255, 1, 64, 200, 31, 16, 127, 2, 254, 3, 129, 42};
        for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * HIDDEN); ++i) {
            table[i] = half_pool[(i * 13u + i / 17u + 5u) % half_pool_count];
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_get_rows_f16(ctx,
                                              table,
                                              TABLE_ROWS,
                                              HIDDEN,
                                              row_ids,
                                              OUT_ROWS,
                                              UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                              out,
                                              error,
                                              sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                CHECK(out[row * (uint32_t)HIDDEN + col] == table[(uint32_t)row_ids[row] * (uint32_t)HIDDEN + col]);
            }
        }
    }

    {
        enum { TABLE_ROWS = 64, HIDDEN = 1280, TOKENS = 9, IMAGE_TOKENS = 3 };
        uint16_t table[TABLE_ROWS * HIDDEN];
        uint16_t image_features[IMAGE_TOKENS * HIDDEN];
        uint16_t out[TOKENS * HIDDEN];
        const int32_t ids[TOKENS] = {0, 17, 999999, -7, 123456, 63, 4, 1, 2};
        for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * HIDDEN); ++i) {
            table[i] = half_pool[(i * 7u + 3u) % half_pool_count];
        }
        for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
            image_features[i] = (uint16_t)(0x6000u + (i % 1024u));
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                uint16_t expected = 0u;
                if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                    expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
                } else {
                    expected = table[(uint32_t)ids[token] * (uint32_t)HIDDEN + col];
                }
                CHECK(out[token * (uint32_t)HIDDEN + col] == expected);
            }
        }

        const int32_t span_at_start_ids[TOKENS] = {-123, 999999, 3, 4, 5, 6, 7, 8, 9};
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     span_at_start_ids,
                                                     TOKENS,
                                                     0u,
                                                     2u,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(out[col] == image_features[col]);
            CHECK(out[(uint32_t)HIDDEN + col] == image_features[(uint32_t)HIDDEN + col]);
        }

        const int32_t span_at_end_ids[TOKENS] = {0, 1, 2, 3, 4, 5, 6, -123, 999999};
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     span_at_end_ids,
                                                     TOKENS,
                                                     7u,
                                                     2u,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(out[7u * (uint32_t)HIDDEN + col] == image_features[col]);
            CHECK(out[8u * (uint32_t)HIDDEN + col] == image_features[(uint32_t)HIDDEN + col]);
        }

        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     0u,
                                                     0u,
                                                     NULL,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "UINT32_MAX image span start") != NULL);

        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     NULL,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "requires image features") != NULL);

        const int32_t invalid_outside_span[TOKENS] = {0, TABLE_ROWS, 999999, -7, 123456, 2, 3, 4, 5};
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     invalid_outside_span,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "outside embedding rows") != NULL);
    }

    {
        enum { ROWS = 2, HIDDEN = 1280 };
        uint16_t input[ROWS * HIDDEN];
        uint16_t weight[HIDDEN];
        float out[ROWS * HIDDEN];
        float expected[ROWS * HIDDEN];
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            input[col] = 0x5000u; /* 32.0: fp16 sum of squares would overflow. */
            weight[col] = half_pool[(col * 5u + 3u) % half_pool_count];
        }
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            input[(uint32_t)HIDDEN + col] = half_pool[(col * 11u + 1u) % (half_pool_count - 1u)];
        }
        const float eps = 1.0e-6f;
        for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
            float sum = 0.0f;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
                sum += x * x;
            }
            const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + eps);
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
                const float w = f16_bits_to_f32(weight[col]);
                expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
            }
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                             input,
                                             weight,
                                             ROWS,
                                             HIDDEN,
                                             eps,
                                             UOCR_METAL_RMSNORM_OUTPUT_F32,
                                             out,
                                             error,
                                             sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
            CHECK(fabsf(out[i] - expected[i]) < 3.0e-4f);
        }
    }

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_runtime_arenas(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 16u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    uint64_t expected_kv = 0u;
    uint64_t expected_prompt = 0u;
    uint64_t expected_decoder = 0u;
    uint64_t expected_router_topk = 0u;
    uint64_t expected_moe_intermediate = 0u;
    uint64_t expected_vision = 0u;
    uint64_t expected_logits = 0u;
    CHECK(uocr_estimate_kv_cache_bytes(1u, 16u, &expected_kv) == UOCR_OK);
    CHECK(uocr_estimate_prompt_embedding_bytes(1u, 16u, &expected_prompt) == UOCR_OK);
    CHECK(uocr_estimate_decoder_scratch_bytes(1u, 16u, &expected_decoder) == UOCR_OK);
    CHECK(uocr_estimate_moe_router_topk_bytes(1u, 16u, &expected_router_topk) == UOCR_OK);
    CHECK(uocr_estimate_moe_intermediate_bytes(1u, 16u, &expected_moe_intermediate) == UOCR_OK);
    CHECK(uocr_estimate_vision_scratch_bytes(&expected_vision) == UOCR_OK);
    CHECK(uocr_estimate_logits_readback_bytes(1u, &expected_logits) == UOCR_OK);

    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_KV_CACHE) == expected_kv);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_PROMPT_EMBEDDINGS) == expected_prompt);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_HIDDEN_PINGPONG) == expected_decoder);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_ROUTER_TOPK) == expected_router_topk);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_MOE_INTERMEDIATE) == expected_moe_intermediate);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_VISION_SCRATCH) == expected_vision);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_LOGITS_READBACK) == expected_logits);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == expected_kv + expected_prompt + expected_decoder +
                                                               expected_router_topk + expected_moe_intermediate +
                                                               expected_vision + expected_logits);

    uocr_metal_context_release_runtime_arenas(ctx);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_model_mapping(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_two_tensor_uocr_model(path) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_model_view_count(ctx) == 1u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 2u);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_model_view_info view_info;
    memset(&view_info, 0, sizeof(view_info));
    CHECK(uocr_metal_context_get_model_view_info(ctx, 0u, &view_info) == 1);
    CHECK(view_info.file_offset <= model.tensors[0].payload_offset);
    CHECK(view_info.length >= UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_tensor_binding binding;
    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 1u, &binding) == 1);
    CHECK(binding.tensor_id == 1u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[0].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 16u);

    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 2u, &binding) == 1);
    CHECK(binding.tensor_id == 2u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[1].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 32u);
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 999u, &binding) == 0);

    CHECK(uocr_metal_context_warmup_model_views(ctx, 128u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_last_warmup_bytes(ctx) == 128u);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_TRANSIENT) >= sizeof(uint32_t));

    uocr_metal_context_unmap_model(ctx);
    CHECK(uocr_metal_context_model_view_count(ctx) == 0u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == 0u);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_public_engine_open_initializes_metal(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);

    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.recommended_working_set_bytes == uocr_metal_recommended_working_set_size());
    CHECK(report.memory_budget_bytes == uocr_metal_default_memory_budget_bytes(report.recommended_working_set_bytes));
    CHECK(report.memory_budget_bytes > 0u);
    CHECK(report.category_live_bytes[UOCR_MEMORY_KV_CACHE] == report.estimated_kv_cache_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_PROMPT_EMBEDDINGS] == report.estimated_prompt_embeddings_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_VISION_SCRATCH] == report.estimated_vision_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_DECODER_SCRATCH] == report.estimated_decoder_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_MOE_SCRATCH] == report.estimated_moe_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_LOGITS_READBACK] == report.estimated_logits_readback_bytes);
    CHECK(report.total_live_bytes == report.estimated_kv_cache_bytes +
                                     report.estimated_prompt_embeddings_bytes +
                                     report.estimated_vision_scratch_bytes +
                                     report.estimated_decoder_scratch_bytes +
                                     report.estimated_moe_scratch_bytes +
                                     report.estimated_logits_readback_bytes);

    uocr_engine_close(engine);
    return 0;
}

int main(void) {
    CHECK(strcmp(uocr_metal_backend_name(), "metal") == 0);
    if (test_metal_smoke() != 0) return 1;
    if (test_metal_named_scratch_buffers() != 0) return 1;
    if (test_metal_get_rows_f16() != 0) return 1;
    if (test_metal_prompt_assembly_f16() != 0) return 1;
    if (test_metal_rmsnorm_f16() != 0) return 1;
    if (test_metal_dense_f16() != 0) return 1;
    if (test_metal_attention_qkvo_f16() != 0) return 1;
    if (test_metal_recent_decoder_primitives_stress() != 0) return 1;
    if (test_metal_runtime_arenas() != 0) return 1;
    if (test_metal_model_mapping() != 0) return 1;
    if (test_public_engine_open_initializes_metal() != 0) return 1;
    return 0;
}
