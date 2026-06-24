#include "backend/metal/uocr_metal.h"
#include "model/uocr_constants.h"
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

static uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0u) {
            return (uint16_t)(sign | 0x7c00u);
        }
        return (uint16_t)(sign | 0x7e00u);
    }

    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return (uint16_t)sign;
        }
        uint32_t mant = mantissa | 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mant >> shift;
        const uint32_t round_bit = 1u << (shift - 1u);
        const uint32_t remainder = mant & (round_bit - 1u);
        if ((mant & round_bit) != 0u && (remainder != 0u || (rounded & 1u) != 0u)) {
            ++rounded;
        }
        return (uint16_t)(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13u;
    const uint32_t round_bit = 0x00001000u;
    const uint32_t remainder = mantissa & (round_bit - 1u);
    if ((mantissa & round_bit) != 0u && (remainder != 0u || (rounded_mantissa & 1u) != 0u)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0u;
            ++half_exponent;
            if (half_exponent >= 31) {
                return (uint16_t)(sign | 0x7c00u);
            }
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10u) | rounded_mantissa);
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

static int test_metal_final_rmsnorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, HIDDEN = UOCR_HIDDEN_SIZE };
    uint16_t input[ROWS * HIDDEN];
    uint16_t weight[HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)((row * 13u + col * 5u) % 17u) - 8;
            input[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.0625f);
        }
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        weight[col] = f32_to_f16_bits(0.5f + (float)((col * 3u) % 11u) * 0.03125f);
    }

    float expected[ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            sum += x * x;
        }
        const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + UOCR_RMS_NORM_EPS);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            const float w = f16_bits_to_f32(weight[col]);
            expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
        }
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_single_final_norm_uocr_model(path, weight) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    float out_f32[ROWS * HIDDEN];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
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
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F16,
                                               out_f16,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 4.0e-3f);
    }

    uocr_metal_context_unmap_model(ctx);
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F32,
                                               out_f32,
                                               error,
                                               sizeof(error)) == 0);
    CHECK(strstr(error, "mapped model views") != NULL);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_lm_head_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROW_COUNT = 4, HIDDEN = UOCR_HIDDEN_SIZE };
    const uint32_t rows[ROW_COUNT] = {0u, 1u, UOCR_TOKEN_IMAGE, UOCR_VOCAB_SIZE - 1u};
    uint16_t input[HIDDEN];
    uint16_t row_weights[ROW_COUNT * HIDDEN];
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const int centered = (int)((col * 7u) % 13u) - 6;
        input[col] = f32_to_f16_bits((float)centered * 0.03125f);
    }
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)(((row + 3u) * 11u + col * 5u) % 17u) - 8;
            row_weights[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.015625f);
        }
    }

    float expected[ROW_COUNT];
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            sum += f16_bits_to_f32(input[col]) * f16_bits_to_f32(row_weights[row * (uint32_t)HIDDEN + col]);
        }
        expected[row] = sum;
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_sparse_lm_head_uocr_model(path, rows, row_weights, ROW_COUNT) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(logits != NULL);
    memset(logits, 0x7f, (size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(uocr_metal_context_lm_head_f16(ctx, input, 1u, logits, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        CHECK(fabsf(logits[rows[row]] - expected[row]) < 2.0e-4f);
    }
    CHECK(fabsf(logits[UOCR_TOKEN_PAD]) < 1.0e-7f);
    CHECK(fabsf(logits[42u]) < 1.0e-7f);

    free(logits);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_argmax_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, VOCAB = UOCR_VOCAB_SIZE };
    float *logits = (float *)malloc((size_t)ROWS * (size_t)VOCAB * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = -1000.0f;
    }
    logits[0u * (uint32_t)VOCAB + 17u] = 3.25f;
    logits[0u * (uint32_t)VOCAB + 42u] = 3.25f; /* tie: lower token id wins */
    logits[1u * (uint32_t)VOCAB + UOCR_TOKEN_PAD] = -0.5f;
    logits[1u * (uint32_t)VOCAB + UOCR_TOKEN_IMAGE] = 12.0f;
    logits[1u * (uint32_t)VOCAB + (uint32_t)VOCAB - 1u] = 11.9f;
    for (uint32_t col = 0u; col < (uint32_t)VOCAB; ++col) {
        logits[2u * (uint32_t)VOCAB + col] = -INFINITY;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f, 0.0f};
    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        logits,
                                        ROWS,
                                        VOCAB,
                                        token_ids,
                                        scores,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(token_ids[0] == 17u);
    CHECK(scores[0] == 3.25f);
    CHECK(token_ids[1] == (uint32_t)UOCR_TOKEN_IMAGE);
    CHECK(scores[1] == 12.0f);
    CHECK(token_ids[2] == 0u);
    CHECK(isinf(scores[2]) && scores[2] < 0.0f);

    const float small_logits[10] = {
        -1.0f, 0.5f, 0.5f, -2.0f, 0.25f,
        -INFINITY, -4.0f, -3.0f, -2.0f, -1.0f,
    };
    uint32_t small_ids[2] = {UINT32_MAX, UINT32_MAX};
    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        small_logits,
                                        2u,
                                        5u,
                                        small_ids,
                                        NULL,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(small_ids[0] == 1u);
    CHECK(small_ids[1] == 4u);

    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        logits,
                                        ROWS,
                                        0u,
                                        token_ids,
                                        scores,
                                        error,
                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal argmax request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(logits);
    return 0;
}

static int test_metal_select_greedy_with_no_repeat_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, VOCAB = UOCR_VOCAB_SIZE };
    float *logits = (float *)malloc((size_t)ROWS * (size_t)VOCAB * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = -1000.0f;
    }
    logits[0u * (uint32_t)VOCAB + 3u] = 100.0f;
    logits[0u * (uint32_t)VOCAB + 4u] = 90.0f;
    logits[1u * (uint32_t)VOCAB + 8u] = 50.0f;
    logits[1u * (uint32_t)VOCAB + 9u] = 40.0f;

    const int32_t sequence0[] = {1, 2, 3, 1, 2}; /* current prefix [1,2] bans 3 */
    const uocr_no_repeat_ngram_config no_repeat[ROWS] = {
        {sequence0, 5u, 3u, 0u},
        {NULL, 0u, 0u, 0u},
    };

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f};
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               ROWS,
                                               VOCAB,
                                               no_repeat,
                                               token_ids,
                                               scores,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(isinf(logits[3u]) && logits[3u] < 0.0f);
    CHECK(token_ids[0] == 4u);
    CHECK(scores[0] == 90.0f);
    CHECK(token_ids[1] == 8u);
    CHECK(scores[1] == 50.0f);

    logits[0u] = 1.0f;
    token_ids[0] = UINT32_MAX;
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               VOCAB,
                                               NULL,
                                               token_ids,
                                               NULL,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(token_ids[0] == 4u);

    const uocr_no_repeat_ngram_config invalid[1] = {{NULL, 5u, 3u, 0u}};
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               VOCAB,
                                               invalid,
                                               token_ids,
                                               scores,
                                               error,
                                               sizeof(error)) == 0);
    CHECK(strstr(error, "failed to apply no-repeat-ngram bans") != NULL);

    uocr_metal_context_destroy(ctx);
    free(logits);
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

static int test_metal_attention_output_residual_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = 1280 };
    const uint16_t context_values[] = {
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0xa400u, /* -0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint16_t weight_values[] = {
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0xa400u, /* -0.015625 */
        0x2800u, /* 0.03125 */
        0xa800u  /* -0.03125 */
    };
    const uint16_t residual_values[] = {
        0xb800u, /* -0.5 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint32_t context_value_count = (uint32_t)(sizeof(context_values) / sizeof(context_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *context = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)malloc((size_t)HIDDEN * HIDDEN * sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(context != NULL && weight != NULL && residual != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        context[i] = context_values[(i * 11u + i / 17u + 3u) % context_value_count];
        residual[i] = residual_values[(i * 7u + i / 19u + 1u) % residual_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)(HIDDEN * HIDDEN); ++i) {
        weight[i] = weight_values[(i * 13u + i / 23u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            float sum = f16_bits_to_f32(residual[token * (uint32_t)HIDDEN + col]);
            for (uint32_t k = 0u; k < (uint32_t)HIDDEN; ++k) {
                sum += f16_bits_to_f32(context[token * (uint32_t)HIDDEN + k]) *
                       f16_bits_to_f32(weight[col * (uint32_t)HIDDEN + k]);
            }
            expected[token * (uint32_t)HIDDEN + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           TOKENS,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.0e-4f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           1u,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           out_f16,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 8.0e-3f);
    }

    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           TOKENS,
                                                           (uocr_metal_dense_output_type)99,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal attention output type") != NULL);
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           0u,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal attention output request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(residual);
    free(weight);
    free(context);
    return 0;
}

static int test_metal_dense_swiglu_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_DENSE_LAYER0_INTERMEDIATE };
    const uint16_t input_values[] = {
        0xa800u, /* -0.03125 */
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint16_t weight_values[] = {
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0xa000u, /* -0.0078125 */
        0x2400u, /* 0.015625 */
        0xa400u  /* -0.015625 */
    };
    const uint16_t residual_values[] = {
        0xa800u, /* -0.03125 */
        0xa400u, /* -0.015625 */
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOKENS * INTERMEDIATE * sizeof(float));
    float *expected_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_no_residual = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && residual != NULL &&
          mid != NULL && expected_residual != NULL && expected_no_residual != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 11u + i / 29u + 2u) % input_value_count];
        residual[i] = residual_values[(i * 7u + i / 17u + 1u) % residual_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
        const uint32_t k0 = (row * 37u) % (uint32_t)HIDDEN;
        const uint32_t k1 = (row * 37u + 113u) % (uint32_t)HIDDEN;
        const uint32_t u0 = (row * 53u + 7u) % (uint32_t)HIDDEN;
        const uint32_t u1 = (row * 53u + 211u) % (uint32_t)HIDDEN;
        gate_weight[row * (uint32_t)HIDDEN + k0] = weight_values[(row + 1u) % weight_value_count];
        gate_weight[row * (uint32_t)HIDDEN + k1] = weight_values[(row * 3u + 2u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u0] = weight_values[(row * 5u + 3u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u1] = weight_values[(row * 7u + 4u) % weight_value_count];
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const uint32_t d0 = (col * 41u) % (uint32_t)INTERMEDIATE;
        const uint32_t d1 = (col * 41u + 977u) % (uint32_t)INTERMEDIATE;
        const uint32_t d2 = (col * 41u + 2222u) % (uint32_t)INTERMEDIATE;
        down_weight[col * (uint32_t)INTERMEDIATE + d0] = weight_values[(col + 2u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d1] = weight_values[(col * 3u + 1u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d2] = weight_values[(col * 5u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            float gate = 0.0f;
            float up = 0.0f;
            const uint32_t k0 = (row * 37u) % (uint32_t)HIDDEN;
            const uint32_t k1 = (row * 37u + 113u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (row * 53u + 7u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (row * 53u + 211u) % (uint32_t)HIDDEN;
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k0]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + k0]);
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k1]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + k1]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u0]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u0]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u1]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u1]);
            const float silu = gate / (1.0f + expf(-gate));
            mid[token * (uint32_t)INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const uint32_t d0 = (col * 41u) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (col * 41u + 977u) % (uint32_t)INTERMEDIATE;
            const uint32_t d2 = (col * 41u + 2222u) % (uint32_t)INTERMEDIATE;
            float sum = 0.0f;
            sum += mid[token * (uint32_t)INTERMEDIATE + d0] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d0]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d1] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d1]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d2] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d2]);
            if (token == 0u) {
                expected_no_residual[col] = sum;
            }
            expected_residual[token * (uint32_t)HIDDEN + col] =
                sum + f16_bits_to_f32(residual[token * (uint32_t)HIDDEN + col]);
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              TOKENS,
                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_residual[i]) < 2.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              NULL,
                                              1u,
                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                              out_f16,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_no_residual[i]) < 2.0e-5f);
    }

    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              TOKENS,
                                              (uocr_metal_dense_output_type)99,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal dense SwiGLU output type") != NULL);
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              0u,
                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal dense SwiGLU request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected_no_residual);
    free(expected_residual);
    free(mid);
    free(residual);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_shared_experts_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_SHARED_INTERMEDIATE };
    const uint16_t input_values[] = {
        0xb400u, /* -0.25 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t weight_values[] = {
        0x2800u, /* 0.03125 */
        0xa800u, /* -0.03125 */
        0x2c00u, /* 0.0625 */
        0xac00u, /* -0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOKENS * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && mid != NULL &&
          expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 13u + i / 19u + 4u) % input_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
        const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
        const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
        const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
        const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
        gate_weight[row * (uint32_t)HIDDEN + g0] = weight_values[(row + 1u) % weight_value_count];
        gate_weight[row * (uint32_t)HIDDEN + g1] = weight_values[(row * 3u + 2u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u0] = weight_values[(row * 5u + 3u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u1] = weight_values[(row * 7u + 4u) % weight_value_count];
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
        const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
        const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
        down_weight[col * (uint32_t)INTERMEDIATE + d0] = weight_values[(col + 2u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d1] = weight_values[(col * 3u + 1u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d2] = weight_values[(col * 5u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
            const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
            float gate = 0.0f;
            float up = 0.0f;
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g0]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + g0]);
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g1]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + g1]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u0]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u0]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u1]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u1]);
            const float silu = gate / (1.0f + expf(-gate));
            mid[token * (uint32_t)INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
            const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
            float sum = 0.0f;
            sum += mid[token * (uint32_t)INTERMEDIATE + d0] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d0]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d1] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d1]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d2] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d2]);
            expected[token * (uint32_t)HIDDEN + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    TOKENS,
                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    1u,
                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                    out_f16,
                                                    error,
                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 3.0e-4f);
    }

    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    TOKENS,
                                                    (uocr_metal_dense_output_type)99,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE shared experts output type") != NULL);
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    0u,
                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE shared experts request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(mid);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static void compute_router_topk_expected(const float *probs,
                                          uint32_t token,
                                          uint32_t experts,
                                          uint32_t top_k,
                                          uint32_t *top_ids,
                                          float *top_weights) {
    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        float best = -INFINITY;
        uint32_t best_expert = 0u;
        for (uint32_t expert = 0u; expert < experts; ++expert) {
            int already_selected = 0;
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if (top_ids[token * top_k + prev] == expert) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected) {
                continue;
            }
            const float value = probs[token * experts + expert];
            if (value > best || (value == best && expert < best_expert)) {
                best = value;
                best_expert = expert;
            }
        }
        top_ids[token * top_k + rank] = best_expert;
        top_weights[token * top_k + rank] = best;
    }
}

static int test_metal_moe_router_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = UOCR_HIDDEN_SIZE, EXPERTS = UOCR_ROUTED_EXPERTS, TOP_K = UOCR_MOE_TOP_K };
    uint16_t *input = (uint16_t *)calloc((size_t)TOKENS * HIDDEN, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc((size_t)EXPERTS * HIDDEN, sizeof(uint16_t));
    float *expected_logits = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    float *expected_probs = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    uint32_t *expected_top_ids = (uint32_t *)calloc((size_t)TOKENS * TOP_K, sizeof(uint32_t));
    float *expected_top_weights = (float *)calloc((size_t)TOKENS * TOP_K, sizeof(float));
    float *logits = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    float *probs = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    uint32_t *top_ids = (uint32_t *)malloc((size_t)TOKENS * TOP_K * sizeof(uint32_t));
    float *top_weights = (float *)malloc((size_t)TOKENS * TOP_K * sizeof(float));
    uint32_t *top_ids_optional = (uint32_t *)malloc((size_t)TOP_K * sizeof(uint32_t));
    float *top_weights_optional = (float *)malloc((size_t)TOP_K * sizeof(float));
    CHECK(input != NULL && weight != NULL && expected_logits != NULL && expected_probs != NULL &&
          expected_top_ids != NULL && expected_top_weights != NULL && logits != NULL && probs != NULL &&
          top_ids != NULL && top_weights != NULL && top_ids_optional != NULL && top_weights_optional != NULL);

    const float token_values[TOKENS][8] = {
        {0.50f, -0.25f, 0.125f, 1.00f, -0.50f, 0.25f, 0.75f, -0.125f},
        {-0.75f, 0.50f, -0.125f, 0.25f, 0.625f, -0.375f, 0.125f, 0.875f},
        {0.25f, 0.75f, 0.50f, -0.50f, 0.375f, 0.125f, -0.625f, 0.25f},
    };
    const uint32_t feature_cols[8] = {0u, 1u, 2u, 3u, 17u, 113u, 509u, 1021u};
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t i = 0u; i < 8u; ++i) {
            input[token * (uint32_t)HIDDEN + feature_cols[i]] = f32_to_f16_bits(token_values[token][i]);
        }
    }

    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        const float weights[8] = {
            ((float)expert - 31.5f) / 16.0f,
            ((float)((expert * 7u) % 17u) - 8.0f) / 32.0f,
            ((float)((expert * 11u) % 23u) - 11.0f) / 48.0f,
            ((float)((expert * 13u) % 29u) - 14.0f) / 64.0f,
            ((float)((expert * 5u) % 19u) - 9.0f) / 40.0f,
            ((float)((expert * 3u) % 31u) - 15.0f) / 80.0f,
            ((float)((expert * 17u) % 37u) - 18.0f) / 96.0f,
            ((float)((expert * 23u) % 41u) - 20.0f) / 112.0f,
        };
        for (uint32_t i = 0u; i < 8u; ++i) {
            weight[expert * (uint32_t)HIDDEN + feature_cols[i]] = f32_to_f16_bits(weights[i]);
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        float max_logit = -INFINITY;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            float sum = 0.0f;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                sum += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + col]) *
                       f16_bits_to_f32(weight[expert * (uint32_t)HIDDEN + col]);
            }
            expected_logits[token * (uint32_t)EXPERTS + expert] = sum;
            if (sum > max_logit) {
                max_logit = sum;
            }
        }
        float denom = 0.0f;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            const float value = expf(expected_logits[token * (uint32_t)EXPERTS + expert] - max_logit);
            expected_probs[token * (uint32_t)EXPERTS + expert] = value;
            denom += value;
        }
        float prob_sum = 0.0f;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            expected_probs[token * (uint32_t)EXPERTS + expert] /= denom;
            prob_sum += expected_probs[token * (uint32_t)EXPERTS + expert];
        }
        CHECK(fabsf(prob_sum - 1.0f) < 2.0e-6f);
        compute_router_topk_expected(expected_probs,
                                     token,
                                     (uint32_t)EXPERTS,
                                     (uint32_t)TOP_K,
                                     expected_top_ids,
                                     expected_top_weights);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(logits, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(probs, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(top_ids, 0xff, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * EXPERTS); ++i) {
        CHECK(fabsf(logits[i] - expected_logits[i]) < 2.0e-6f);
        CHECK(fabsf(probs[i] - expected_probs[i]) < 2.0e-6f);
    }
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * TOP_K); ++i) {
        CHECK(top_ids[i] == expected_top_ids[i]);
        CHECK(fabsf(top_weights[i] - expected_top_weights[i]) < 2.0e-6f);
    }

    memset(top_ids_optional, 0xff, (size_t)TOP_K * sizeof(uint32_t));
    memset(top_weights_optional, 0, (size_t)TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            1u,
                                            NULL,
                                            NULL,
                                            top_ids_optional,
                                            top_weights_optional,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)TOP_K; ++i) {
        CHECK(top_ids_optional[i] == expected_top_ids[i]);
        CHECK(fabsf(top_weights_optional[i] - expected_top_weights[i]) < 2.0e-6f);
    }

    memset(weight, 0, (size_t)EXPERTS * HIDDEN * sizeof(uint16_t));
    memset(logits, 1, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(probs, 1, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(top_ids, 0xff, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * EXPERTS); ++i) {
        CHECK(fabsf(logits[i]) < 2.0e-7f);
        CHECK(fabsf(probs[i] - (1.0f / (float)EXPERTS)) < 2.0e-7f);
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            CHECK(top_ids[token * (uint32_t)TOP_K + rank] == rank);
            CHECK(fabsf(top_weights[token * (uint32_t)TOP_K + rank] - (1.0f / (float)EXPERTS)) < 2.0e-7f);
        }
    }

    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            NULL,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE router request") != NULL);
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            0u,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE router request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(top_weights_optional);
    free(top_ids_optional);
    free(top_weights);
    free(top_ids);
    free(probs);
    free(logits);
    free(expected_top_weights);
    free(expected_top_ids);
    free(expected_probs);
    free(expected_logits);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_moe_selected_experts_decode_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_EXPERT_INTERMEDIATE, TOP_K = UOCR_MOE_TOP_K };
    const uint32_t top_ids[TOP_K] = {7u, 2u, 63u, 0u, 5u, 11u};
    const float top_weights[TOP_K] = {0.375f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f};
    const uint16_t input_values[] = {
        0xb000u, /* -0.125 */
        0xa800u, /* -0.03125 */
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x3000u  /* 0.125 */
    };
    const uint16_t weight_values[] = {
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0xa000u, /* -0.0078125 */
        0xa400u, /* -0.015625 */
        0xa800u  /* -0.03125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)TOP_K * INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)TOP_K * INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)TOP_K * HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOP_K * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && mid != NULL &&
          expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        input[col] = input_values[(col * 7u + col / 23u + 3u) % input_value_count];
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            uint32_t g0 = (rank * 127u + row * 17u) % (uint32_t)HIDDEN;
            uint32_t g1 = (rank * 197u + row * 31u + 23u) % (uint32_t)HIDDEN;
            uint32_t u0 = (rank * 59u + row * 13u + 5u) % (uint32_t)HIDDEN;
            uint32_t u1 = (rank * 89u + row * 19u + 17u) % (uint32_t)HIDDEN;
            if (g1 == g0) g1 = (g1 + 1u) % (uint32_t)HIDDEN;
            if (u1 == u0) u1 = (u1 + 1u) % (uint32_t)HIDDEN;
            const size_t base = ((size_t)rank * INTERMEDIATE + row) * HIDDEN;
            gate_weight[base + g0] = weight_values[(rank + row + 1u) % weight_value_count];
            gate_weight[base + g1] = weight_values[(rank * 3u + row * 5u + 2u) % weight_value_count];
            up_weight[base + u0] = weight_values[(rank * 5u + row * 7u + 3u) % weight_value_count];
            up_weight[base + u1] = weight_values[(rank * 11u + row * 13u + 4u) % weight_value_count];
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint32_t d0 = (rank * 173u + col * 7u) % (uint32_t)INTERMEDIATE;
            uint32_t d1 = (rank * 181u + col * 11u + 37u) % (uint32_t)INTERMEDIATE;
            uint32_t d2 = (rank * 191u + col * 13u + 101u) % (uint32_t)INTERMEDIATE;
            if (d1 == d0) d1 = (d1 + 1u) % (uint32_t)INTERMEDIATE;
            if (d2 == d0 || d2 == d1) d2 = (d2 + 2u) % (uint32_t)INTERMEDIATE;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            down_weight[base + d0] = weight_values[(rank + col + 2u) % weight_value_count];
            down_weight[base + d1] = weight_values[(rank * 3u + col * 5u + 1u) % weight_value_count];
            down_weight[base + d2] = weight_values[(rank * 7u + col * 11u + 5u) % weight_value_count];
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            float gate = 0.0f;
            float up = 0.0f;
            const size_t base = ((size_t)rank * INTERMEDIATE + row) * HIDDEN;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[col]);
                gate += x * f16_bits_to_f32(gate_weight[base + col]);
                up += x * f16_bits_to_f32(up_weight[base + col]);
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid[(size_t)rank * INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        float routed_sum = 0.0f;
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            float expert_sum = 0.0f;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                expert_sum += mid[(size_t)rank * INTERMEDIATE + row] * f16_bits_to_f32(down_weight[base + row]);
            }
            routed_sum += expert_sum * top_weights[rank];
        }
        expected[col] = routed_sum;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(out_f32[col] - expected[col]) < 3.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             out_f16,
                                                             error,
                                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[col]) - expected[col]) < 3.0e-4f);
    }

    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             (uocr_metal_dense_output_type)99,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE selected-expert output type") != NULL);

    uint32_t bad_ids[TOP_K] = {7u, 2u, UOCR_ROUTED_EXPERTS, 0u, 5u, 11u};
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             bad_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected expert id") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(mid);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_combine_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = UOCR_HIDDEN_SIZE };
    const uint16_t routed_values[] = {
        0xb800u, /* -0.5 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3800u  /* 0.5 */
    };
    const uint16_t shared_values[] = {
        0xb400u, /* -0.25 */
        0xa800u, /* -0.03125 */
        0x2800u, /* 0.03125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t residual_values[] = {
        0xbc00u, /* -1.0 */
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t routed_value_count = (uint32_t)(sizeof(routed_values) / sizeof(routed_values[0]));
    const uint32_t shared_value_count = (uint32_t)(sizeof(shared_values) / sizeof(shared_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *routed = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *shared = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_no_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(routed != NULL && shared != NULL && residual != NULL && expected_residual != NULL &&
          expected_no_residual != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        routed[i] = routed_values[(i * 5u + i / 17u + 1u) % routed_value_count];
        shared[i] = shared_values[(i * 7u + i / 23u + 2u) % shared_value_count];
        residual[i] = residual_values[(i * 11u + i / 31u + 3u) % residual_value_count];
        expected_no_residual[i] = f16_bits_to_f32(routed[i]) + f16_bits_to_f32(shared[i]);
        expected_residual[i] = expected_no_residual[i] + f16_bits_to_f32(residual[i]);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             TOKENS,
                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_residual[i]) < 1.0e-7f);
    }

    memset(out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             NULL,
                                             TOKENS,
                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                             out_f16,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_no_residual[i]) < 8.0e-4f);
    }

    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             TOKENS,
                                             (uocr_metal_dense_output_type)99,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE combine output type") != NULL);
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             0u,
                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE combine request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected_no_residual);
    free(expected_residual);
    free(residual);
    free(shared);
    free(routed);
    return 0;
}

static void compute_rope_expected(const uint16_t *src,
                                  uint32_t n_tokens,
                                  uint32_t position_start,
                                  float *out) {
    enum { HEADS = 10, HEAD_DIM = 128, HALF_DIM = 64 };
    const float freq_scale = -2.0f * log2f(10000.0f) / (float)HEAD_DIM;
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        const float position = (float)(position_start + token);
        for (uint32_t head = 0u; head < (uint32_t)HEADS; ++head) {
            const uint32_t base = (token * (uint32_t)HEADS + head) * (uint32_t)HEAD_DIM;
            for (uint32_t pair = 0u; pair < (uint32_t)HALF_DIM; ++pair) {
                const uint32_t a = pair;
                const uint32_t b = pair + (uint32_t)HALF_DIM;
                const float angle = position * exp2f((float)pair * freq_scale);
                const float c = cosf(angle);
                const float s = sinf(angle);
                const float x0 = f16_bits_to_f32(src[base + a]);
                const float x1 = f16_bits_to_f32(src[base + b]);
                out[base + a] = x0 * c - x1 * s;
                out[base + b] = x0 * s + x1 * c;
            }
        }
    }
}

static int test_metal_rope_qk_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 4, HIDDEN = 1280 };
    const uint16_t values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t value_count = (uint32_t)(sizeof(values) / sizeof(values[0]));
    uint16_t *q = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected_q = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_k = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *q_out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *k_out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *q_out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k_out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && expected_q != NULL && expected_k != NULL &&
          q_out_f32 != NULL && k_out_f32 != NULL && q_out_f16 != NULL && k_out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        q[i] = values[(i * 17u + i / 31u + 1u) % value_count];
        k[i] = values[(i * 23u + i / 19u + 5u) % value_count];
    }

    const uint32_t position_start = 7u;
    compute_rope_expected(q, TOKENS, position_start, expected_q);
    compute_rope_expected(k, TOKENS, position_start, expected_k);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(q_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    memset(k_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(q_out_f32[i] - expected_q[i]) < 2.0e-4f);
        CHECK(fabsf(k_out_f32[i] - expected_k[i]) < 2.0e-4f);
    }

    memset(q_out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    memset(k_out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                         q_out_f16,
                                         k_out_f16,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(q_out_f16[i]) - expected_q[i]) < 4.0e-3f);
        CHECK(fabsf(f16_bits_to_f32(k_out_f16[i]) - expected_k[i]) < 4.0e-3f);
    }

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         0u,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal RoPE request") != NULL);

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         2u,
                                         32767u,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max positions") != NULL);

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         (uocr_metal_dense_output_type)99,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal RoPE output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(k_out_f16);
    free(q_out_f16);
    free(k_out_f32);
    free(q_out_f32);
    free(expected_k);
    free(expected_q);
    free(k);
    free(q);
    return 0;
}

static int compute_prefill_attention_expected(const uint16_t *q,
                                              const uint16_t *k,
                                              const uint16_t *v,
                                              uint32_t n_tokens,
                                              float *out) {
    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    float scores[16];
    if (n_tokens > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
        return 0;
    }
    for (uint32_t query = 0u; query < n_tokens; ++query) {
        for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
            float max_score = -3.4028234663852886e38f;
            for (uint32_t key = 0u; key <= query; ++key) {
                float score = 0.0f;
                const uint32_t q_base = (query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                const uint32_t k_base = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                    score += f16_bits_to_f32(q[q_base + dim]) * f16_bits_to_f32(k[k_base + dim]);
                }
                score *= scale;
                scores[key] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float denominator = 0.0f;
            for (uint32_t key = 0u; key <= query; ++key) {
                scores[key] = expf(scores[key] - max_score);
                denominator += scores[key];
            }
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                float value = 0.0f;
                for (uint32_t key = 0u; key <= query; ++key) {
                    const uint32_t v_index = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim;
                    value += (scores[key] / denominator) * f16_bits_to_f32(v[v_index]);
                }
                out[(query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim] = value;
            }
        }
    }
    return 1;
}

static int compute_prefill_attention_varlen_expected(const uint16_t *q,
                                                     const uint16_t *k,
                                                     const uint16_t *v,
                                                     const uint32_t *cu,
                                                     uint32_t batch,
                                                     uint32_t total_tokens,
                                                     float *out) {
    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    float scores[16];
    for (uint32_t b = 0u; b < batch; ++b) {
        const uint32_t start = cu[b];
        const uint32_t end = cu[b + 1u];
        if (end <= start || end > total_tokens || end - start > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
            return 0;
        }
        for (uint32_t query = start; query < end; ++query) {
            for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
                float max_score = -3.4028234663852886e38f;
                for (uint32_t key = start; key <= query; ++key) {
                    float score = 0.0f;
                    const uint32_t q_base = (query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                    const uint32_t k_base = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                        score += f16_bits_to_f32(q[q_base + dim]) * f16_bits_to_f32(k[k_base + dim]);
                    }
                    score *= scale;
                    scores[key - start] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                float denominator = 0.0f;
                for (uint32_t key = start; key <= query; ++key) {
                    scores[key - start] = expf(scores[key - start] - max_score);
                    denominator += scores[key - start];
                }
                for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                    float value = 0.0f;
                    for (uint32_t key = start; key <= query; ++key) {
                        const uint32_t v_index = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim;
                        value += (scores[key - start] / denominator) * f16_bits_to_f32(v[v_index]);
                    }
                    out[(query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim] = value;
                }
            }
        }
    }
    return 1;
}

static int test_metal_prefill_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 5, HIDDEN = UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM };
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *v = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && v != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        q[i] = q_values[(i * 7u + i / 19u + 1u) % q_count];
        k[i] = k_values[(i * 11u + i / 23u + 3u) % k_count];
        v[i] = v_values[(i * 13u + i / 29u + 5u) % v_count];
    }
    /* Make future tokens conspicuous so query-token zero verifies the causal
     * mask cannot leak later values.
     */
    for (uint32_t token = 1u; token < (uint32_t)TOKENS; ++token) {
        v[token * (uint32_t)HIDDEN] = 0x5000u; /* 32.0 */
    }

    CHECK(compute_prefill_attention_expected(q, k, v, TOKENS, expected) == 1);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 8.0e-4f);
    }
    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
        CHECK(out_f32[dim] == f16_bits_to_f32(v[dim]));
    }

    memset(out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   out_f16,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   0u,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal prefill attention request") != NULL);
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   (uocr_metal_dense_output_type)99,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal prefill attention output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_metal_prefill_attention_varlen_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { BATCH = 2, TOTAL_TOKENS = 7, MAX_SEQLEN = 4, HIDDEN = UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM };
    const uint32_t cu[BATCH + 1] = {0u, 3u, 7u};
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *v = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && v != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        q[i] = q_values[(i * 7u + i / 19u + 2u) % q_count];
        k[i] = k_values[(i * 11u + i / 23u + 4u) % k_count];
        v[i] = v_values[(i * 13u + i / 29u + 1u) % v_count];
    }
    /* Make cross-sequence leakage obvious: first token of sequence 1 must only
     * see itself, never the large values placed at the end of sequence 0.
     */
    v[2u * (uint32_t)HIDDEN] = 0x5000u; /* 32.0 */
    v[3u * (uint32_t)HIDDEN] = 0xb800u; /* -0.5 */
    CHECK(compute_prefill_attention_varlen_expected(q, k, v, cu, BATCH, TOTAL_TOKENS, expected) == 1);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-3f);
    }
    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
        CHECK(out_f32[dim] == f16_bits_to_f32(v[dim]));
        CHECK(out_f32[3u * (uint32_t)HIDDEN + dim] == f16_bits_to_f32(v[3u * (uint32_t)HIDDEN + dim]));
    }

    memset(out_f16, 0, (size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F16,
                                                          out_f16,
                                                          error,
                                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    const uint32_t bad_cu_order[BATCH + 1] = {0u, 5u, 4u};
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          bad_cu_order,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 0);
    CHECK(strstr(error, "sequence metadata") != NULL || strstr(error, "cu_seqlens") != NULL);

    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          2u,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max_seqlen") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v);
    free(k);
    free(q);
    return 0;
}

static uint16_t kv_cache_sentinel(uint64_t index, uint32_t stream) {
    return (uint16_t)(0x1800u + ((index * 17u + (uint64_t)stream * 97u) & 0x07ffu));
}

static uint16_t kv_source_value(uint64_t index, uint32_t stream) {
    return (uint16_t)(0x3000u + ((index * 23u + (uint64_t)stream * 211u) & 0x0fffu));
}

static uint32_t kv_cache_token_for_position(uint32_t position, uint32_t prompt_length) {
    if (position < prompt_length) {
        return position;
    }
    return prompt_length + ((position - prompt_length) % UOCR_GENERATED_RING_WINDOW);
}

static uint64_t kv_cache_flat_index(uint32_t batch_slots,
                                    uint32_t cache_token_capacity,
                                    uint32_t layer,
                                    uint32_t slot,
                                    uint32_t cache_token,
                                    uint32_t head,
                                    uint32_t dim) {
    return (((uint64_t)layer * (uint64_t)batch_slots + (uint64_t)slot) *
                (uint64_t)cache_token_capacity + (uint64_t)cache_token) *
               (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM +
           (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim;
}

static int compute_decode_attention_expected(const uint16_t *q,
                                             const uint16_t *k_cache,
                                             const uint16_t *v_cache,
                                             uint32_t batch_slots,
                                             uint32_t prompt_token_capacity,
                                             uint32_t layer,
                                             uint32_t slot,
                                             uint32_t prompt_length,
                                             uint32_t generated_count,
                                             float *out) {
    uint32_t attention_length = 0u;
    if (!uocr_metal_kv_cache_attention_length(prompt_length, generated_count, &attention_length)) {
        return 0;
    }
    const uint32_t cache_token_capacity = prompt_token_capacity + UOCR_GENERATED_RING_WINDOW;
    float *scores = (float *)malloc((size_t)attention_length * sizeof(float));
    if (scores == NULL) {
        return 0;
    }

    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
        float max_score = -3.4028234663852886e38f;
        for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
            uint32_t cache_token = UINT32_MAX;
            if (!uocr_metal_kv_cache_token_for_attention_index(prompt_length,
                                                               prompt_token_capacity,
                                                               generated_count,
                                                               attention_index,
                                                               &cache_token)) {
                free(scores);
                return 0;
            }
            float score = 0.0f;
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t q_index = (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim;
                const uint64_t k_index = kv_cache_flat_index(batch_slots,
                                                             cache_token_capacity,
                                                             layer,
                                                             slot,
                                                             cache_token,
                                                             head,
                                                             dim);
                score += f16_bits_to_f32(q[q_index]) * f16_bits_to_f32(k_cache[k_index]);
            }
            score *= scale;
            scores[attention_index] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        float denominator = 0.0f;
        for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
            scores[attention_index] = expf(scores[attention_index] - max_score);
            denominator += scores[attention_index];
        }
        for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
            float value = 0.0f;
            for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
                uint32_t cache_token = UINT32_MAX;
                if (!uocr_metal_kv_cache_token_for_attention_index(prompt_length,
                                                                   prompt_token_capacity,
                                                                   generated_count,
                                                                   attention_index,
                                                                   &cache_token)) {
                    free(scores);
                    return 0;
                }
                const uint64_t v_index = kv_cache_flat_index(batch_slots,
                                                             cache_token_capacity,
                                                             layer,
                                                             slot,
                                                             cache_token,
                                                             head,
                                                             dim);
                value += (scores[attention_index] / denominator) * f16_bits_to_f32(v_cache[v_index]);
            }
            out[(uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim] = value;
        }
    }

    free(scores);
    return 1;
}

static int test_metal_kv_cache_layout_helpers(void) {
    uint32_t cache_token = UINT32_MAX;
    uint32_t attention_length = 0u;
    uint64_t offset = 0u;
    uocr_metal_decode_attention_plan decode_plan;

    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 6u, &cache_token) == 1);
    CHECK(cache_token == 6u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u + UOCR_GENERATED_RING_WINDOW - 1u, &cache_token) == 1);
    CHECK(cache_token == 7u + UOCR_GENERATED_RING_WINDOW - 1u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u + UOCR_GENERATED_RING_WINDOW, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_position(0u, 16u, 0u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_position(17u, 16u, 0u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, UOCR_MAX_POSITIONS, &cache_token) == 0);

    CHECK(uocr_metal_kv_cache_attention_length(7u, 0u, &attention_length) == 1);
    CHECK(attention_length == 7u);
    CHECK(uocr_metal_kv_cache_attention_length(7u, 5u, &attention_length) == 1);
    CHECK(attention_length == 12u);
    CHECK(uocr_metal_kv_cache_attention_length(7u, UOCR_GENERATED_RING_WINDOW, &attention_length) == 1);
    CHECK(attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_attention_length(7u, UOCR_GENERATED_RING_WINDOW + 9u, &attention_length) == 1);
    CHECK(attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS, 0u, &attention_length) == 1);
    CHECK(attention_length == UOCR_MAX_POSITIONS);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS, 1u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS - 3u, 3u, &attention_length) == 1);
    CHECK(attention_length == UOCR_MAX_POSITIONS);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS - 3u, 4u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(UINT32_MAX, 0u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(0u, 1u, &attention_length) == 0);

    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 0u, 6u, &cache_token) == 1);
    CHECK(cache_token == 6u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 7u, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 11u, &cache_token) == 1);
    CHECK(cache_token == 11u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_GENERATED_RING_WINDOW + 9u,
                                                        7u,
                                                        &cache_token) == 1);
    CHECK(cache_token == 16u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_GENERATED_RING_WINDOW + 9u,
                                                        7u + UOCR_GENERATED_RING_WINDOW - 1u,
                                                        &cache_token) == 1);
    CHECK(cache_token == 15u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 12u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_MAX_POSITIONS,
                                                        7u,
                                                        &cache_token) == 0);

    memset(&decode_plan, 0, sizeof(decode_plan));
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u,
                                                    16u,
                                                    UOCR_GENERATED_RING_WINDOW + 9u,
                                                    &decode_plan) == 1);
    CHECK(decode_plan.prompt_length == 7u);
    CHECK(decode_plan.prompt_token_capacity == 16u);
    CHECK(decode_plan.cache_token_capacity == 16u + UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.generated_count == UOCR_GENERATED_RING_WINDOW + 9u);
    CHECK(decode_plan.live_generated == UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.first_generated_index == 9u);
    CHECK(decode_plan.first_generated_position == 16u);
    CHECK(decode_plan.query_position == 7u + UOCR_GENERATED_RING_WINDOW + 8u);
    CHECK(decode_plan.attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.generated_ring_window == UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 0u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 6u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 15u) == 0);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 16u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u + UOCR_GENERATED_RING_WINDOW + 8u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u + UOCR_GENERATED_RING_WINDOW + 9u) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 7u, &cache_token) == 1);
    CHECK(cache_token == 16u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan,
                                                              7u + UOCR_GENERATED_RING_WINDOW - 1u,
                                                              &cache_token) == 1);
    CHECK(cache_token == 15u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan,
                                                              7u + UOCR_GENERATED_RING_WINDOW,
                                                              &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u, 6u, 0u, &decode_plan) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_plan(0u, 16u, 0u, &decode_plan) == 0);

    memset(&decode_plan, 0, sizeof(decode_plan));
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u, 16u, 0u, &decode_plan) == 1);
    CHECK(decode_plan.live_generated == 0u);
    CHECK(decode_plan.attention_length == 7u);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 6u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u) == 0);

    uocr_metal_kv_cache_layout empty_layout;
    memset(&empty_layout, 0, sizeof(empty_layout));
    CHECK(uocr_metal_kv_cache_offset(&empty_layout, 0, 0u, 0u, 0u, 0u, 0u, &offset) == 0);

    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uocr_metal_kv_cache_layout layout;
    memset(&layout, 0, sizeof(layout));
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 0);
    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 2u, 16u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 1);
    CHECK(layout.decoder_layers == UOCR_DECODER_LAYERS);
    CHECK(layout.batch_slots == 2u);
    CHECK(layout.prompt_token_capacity == 16u);
    CHECK(layout.cache_token_capacity == 16u + UOCR_GENERATED_RING_WINDOW);
    CHECK(layout.kv_heads == UOCR_KV_HEADS);
    CHECK(layout.head_dim == UOCR_HEAD_DIM);
    CHECK(layout.generated_ring_window == UOCR_GENERATED_RING_WINDOW);
    CHECK(layout.token_stride_bytes == (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM * 2u);
    CHECK(layout.slot_stride_bytes == (uint64_t)layout.cache_token_capacity * layout.token_stride_bytes);
    CHECK(layout.layer_stride_bytes == (uint64_t)layout.batch_slots * layout.slot_stride_bytes);
    CHECK(layout.tensor_bytes == (uint64_t)UOCR_DECODER_LAYERS * layout.layer_stride_bytes);
    CHECK(layout.k_offset_bytes == 0u);
    CHECK(layout.v_offset_bytes == layout.tensor_bytes);
    CHECK(layout.total_bytes == layout.tensor_bytes * 2u);
    CHECK(layout.total_bytes == uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_KV_CACHE));

    const uint32_t layer = 2u;
    const uint32_t slot = 1u;
    const uint32_t token = 9u;
    const uint32_t head = 4u;
    const uint32_t dim = 5u;
    const uint64_t expected_inner = (uint64_t)layer * layout.layer_stride_bytes +
                                    (uint64_t)slot * layout.slot_stride_bytes +
                                    (uint64_t)token * layout.token_stride_bytes +
                                    (uint64_t)head * (uint64_t)UOCR_HEAD_DIM * 2u +
                                    (uint64_t)dim * 2u;
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, head, dim, &offset) == 1);
    CHECK(offset == layout.k_offset_bytes + expected_inner);
    CHECK(uocr_metal_kv_cache_offset(&layout, 1, layer, slot, token, head, dim, &offset) == 1);
    CHECK(offset == layout.v_offset_bytes + expected_inner);
    CHECK(uocr_metal_kv_cache_offset(&layout,
                                     1,
                                     layout.decoder_layers - 1u,
                                     layout.batch_slots - 1u,
                                     layout.cache_token_capacity - 1u,
                                     layout.kv_heads - 1u,
                                     layout.head_dim - 1u,
                                     &offset) == 1);
    CHECK(offset == layout.total_bytes - 2u);
    CHECK(uocr_metal_kv_cache_offset(&layout, 2, layer, slot, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, UOCR_DECODER_LAYERS, slot, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, 2u, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, layout.cache_token_capacity, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, UOCR_KV_HEADS, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, head, UOCR_HEAD_DIM, &offset) == 0);

    uocr_metal_context_release_runtime_arenas(ctx);
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 0);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_kv_cache_write_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        BATCH_SLOTS = 2,
        PROMPT_TOKEN_CAPACITY = 8,
        PROMPT_LENGTH = 5,
        CACHE_TOKEN_CAPACITY = PROMPT_TOKEN_CAPACITY + UOCR_GENERATED_RING_WINDOW,
        LAYER = 3,
        SLOT = 1,
        N_TOKENS = PROMPT_LENGTH + UOCR_GENERATED_RING_WINDOW + 3,
        HEADS = UOCR_KV_HEADS,
        HEAD_DIM = UOCR_HEAD_DIM,
        HEAD_AREA = HEADS * HEAD_DIM
    };
    const uint64_t cache_values = (uint64_t)UOCR_DECODER_LAYERS * (uint64_t)BATCH_SLOTS *
                                  (uint64_t)CACHE_TOKEN_CAPACITY * (uint64_t)HEAD_AREA;
    const uint64_t src_values = (uint64_t)N_TOKENS * (uint64_t)HEAD_AREA;

    uint16_t *k_src = (uint16_t *)malloc((size_t)src_values * sizeof(uint16_t));
    uint16_t *v_src = (uint16_t *)malloc((size_t)src_values * sizeof(uint16_t));
    uint16_t *initial_k = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *initial_v = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *k_out = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *v_out = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint32_t *last_token_for_cache_token = (uint32_t *)malloc((size_t)CACHE_TOKEN_CAPACITY * sizeof(uint32_t));
    CHECK(k_src != NULL && v_src != NULL && initial_k != NULL && initial_v != NULL &&
          k_out != NULL && v_out != NULL && last_token_for_cache_token != NULL);

    for (uint64_t i = 0u; i < src_values; ++i) {
        k_src[i] = kv_source_value(i, 0u);
        v_src[i] = kv_source_value(i, 1u);
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        initial_k[i] = kv_cache_sentinel(i, 0u);
        initial_v[i] = kv_cache_sentinel(i, 1u);
        k_out[i] = 0u;
        v_out[i] = 0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)CACHE_TOKEN_CAPACITY; ++i) {
        last_token_for_cache_token[i] = UINT32_MAX;
    }
    for (uint32_t token = 0u; token < (uint32_t)N_TOKENS; ++token) {
        const uint32_t cache_token = kv_cache_token_for_position(token, PROMPT_LENGTH);
        CHECK(cache_token < (uint32_t)CACHE_TOKEN_CAPACITY);
        last_token_for_cache_token[cache_token] = token;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                N_TOKENS,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint64_t index = 0u; index < cache_values; ++index) {
        uint64_t rem = index;
        const uint32_t dim = (uint32_t)(rem % (uint64_t)HEAD_DIM);
        rem /= (uint64_t)HEAD_DIM;
        const uint32_t head = (uint32_t)(rem % (uint64_t)HEADS);
        rem /= (uint64_t)HEADS;
        const uint32_t cache_token = (uint32_t)(rem % (uint64_t)CACHE_TOKEN_CAPACITY);
        rem /= (uint64_t)CACHE_TOKEN_CAPACITY;
        const uint32_t slot = (uint32_t)(rem % (uint64_t)BATCH_SLOTS);
        const uint32_t layer = (uint32_t)(rem / (uint64_t)BATCH_SLOTS);

        uint16_t expected_k = initial_k[index];
        uint16_t expected_v = initial_v[index];
        if (layer == (uint32_t)LAYER && slot == (uint32_t)SLOT &&
            last_token_for_cache_token[cache_token] != UINT32_MAX) {
            const uint64_t src_index = ((uint64_t)last_token_for_cache_token[cache_token] * (uint64_t)HEADS +
                                        (uint64_t)head) * (uint64_t)HEAD_DIM + (uint64_t)dim;
            expected_k = k_src[src_index];
            expected_v = v_src[src_index];
        }
        CHECK(k_out[index] == expected_k);
        CHECK(v_out[index] == expected_v);
    }

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                NULL,
                                                NULL,
                                                1u,
                                                1u,
                                                PROMPT_TOKEN_CAPACITY,
                                                0u,
                                                0u,
                                                PROMPT_LENGTH,
                                                2u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t index = 0u; index < cache_values / (uint64_t)BATCH_SLOTS; ++index) {
        uint64_t rem = index;
        const uint32_t dim = (uint32_t)(rem % (uint64_t)HEAD_DIM);
        rem /= (uint64_t)HEAD_DIM;
        const uint32_t head = (uint32_t)(rem % (uint64_t)HEADS);
        rem /= (uint64_t)HEADS;
        const uint32_t cache_token = (uint32_t)(rem % (uint64_t)CACHE_TOKEN_CAPACITY);
        rem /= (uint64_t)CACHE_TOKEN_CAPACITY;
        const uint32_t layer = (uint32_t)rem;
        const uint64_t src_index = ((uint64_t)head * (uint64_t)HEAD_DIM) + (uint64_t)dim;
        const int should_have_value = layer == 0u && cache_token == 2u;
        CHECK(k_out[index] == (should_have_value ? k_src[src_index] : 0u));
        CHECK(v_out[index] == (should_have_value ? v_src[src_index] : 0u));
    }

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                0u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal KV cache write request") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                1u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                UOCR_DECODER_LAYERS,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "layer/slot out of range") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                1u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_TOKEN_CAPACITY + 1u,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal KV cache layout") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                2u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                UOCR_MAX_POSITIONS - 1u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max positions") != NULL);

    uocr_metal_context_destroy(ctx);
    free(last_token_for_cache_token);
    free(v_out);
    free(k_out);
    free(initial_v);
    free(initial_k);
    free(v_src);
    free(k_src);
    return 0;
}

static int test_metal_decode_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        BATCH_SLOTS = 2,
        PROMPT_TOKEN_CAPACITY = 8,
        PROMPT_LENGTH = 5,
        CACHE_TOKEN_CAPACITY = PROMPT_TOKEN_CAPACITY + UOCR_GENERATED_RING_WINDOW,
        LAYER = 4,
        SLOT = 1,
        HEAD_AREA = UOCR_KV_HEADS * UOCR_HEAD_DIM
    };
    const uint64_t cache_values = (uint64_t)UOCR_DECODER_LAYERS * (uint64_t)BATCH_SLOTS *
                                  (uint64_t)CACHE_TOKEN_CAPACITY * (uint64_t)HEAD_AREA;
    const uint64_t out_values = (uint64_t)HEAD_AREA;
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0xac00u  /* -0.0625 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)out_values * sizeof(uint16_t));
    uint16_t *k_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *v_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)out_values * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)out_values * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)out_values * sizeof(uint16_t));
    CHECK(q != NULL && k_cache != NULL && v_cache != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t no_wrap_generated = 3u;
    uint32_t attention_length = 0u;
    CHECK(uocr_metal_kv_cache_attention_length(PROMPT_LENGTH, no_wrap_generated, &attention_length) == 1);
    for (uint64_t i = 0u; i < out_values; ++i) {
        q[i] = q_values[(i * 7u + i / 11u + 3u) % q_count];
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        k_cache[i] = 0x0000u;
        v_cache[i] = 0x5000u; /* 32.0; catches accidental attention to unused ring slots. */
    }
    for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
        uint32_t cache_token = UINT32_MAX;
        CHECK(uocr_metal_kv_cache_token_for_attention_index(PROMPT_LENGTH,
                                                            PROMPT_TOKEN_CAPACITY,
                                                            no_wrap_generated,
                                                            attention_index,
                                                            &cache_token) == 1);
        for (uint32_t head = 0u; head < UOCR_KV_HEADS; ++head) {
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t serial = ((uint64_t)attention_index * (uint64_t)HEAD_AREA +
                                         (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim);
                const uint64_t index = kv_cache_flat_index(BATCH_SLOTS,
                                                           CACHE_TOKEN_CAPACITY,
                                                           LAYER,
                                                           SLOT,
                                                           cache_token,
                                                           head,
                                                           dim);
                k_cache[index] = k_values[(serial * 11u + serial / 17u + 1u) % k_count];
                v_cache[index] = v_values[(serial * 13u + serial / 19u + 4u) % v_count];
            }
        }
    }
    CHECK(compute_decode_attention_expected(q,
                                            k_cache,
                                            v_cache,
                                            BATCH_SLOTS,
                                            PROMPT_TOKEN_CAPACITY,
                                            LAYER,
                                            SLOT,
                                            PROMPT_LENGTH,
                                            no_wrap_generated,
                                            expected) == 1);

    memset(out_f32, 0, (size_t)out_values * sizeof(float));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-3f);
    }

    memset(out_f16, 0, (size_t)out_values * sizeof(uint16_t));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  out_f16,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    const uint32_t wrapped_generated = UOCR_GENERATED_RING_WINDOW + 3u;
    CHECK(uocr_metal_kv_cache_attention_length(PROMPT_LENGTH, wrapped_generated, &attention_length) == 1);
    for (uint64_t i = 0u; i < out_values; ++i) {
        q[i] = 0x0000u;
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        k_cache[i] = 0x0000u;
        v_cache[i] = 0x5000u; /* Must not be read from inactive layers/slots/tokens. */
    }
    for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
        uint32_t cache_token = UINT32_MAX;
        CHECK(uocr_metal_kv_cache_token_for_attention_index(PROMPT_LENGTH,
                                                            PROMPT_TOKEN_CAPACITY,
                                                            wrapped_generated,
                                                            attention_index,
                                                            &cache_token) == 1);
        const uint16_t value = attention_index < PROMPT_LENGTH ? 0x3c00u : 0xbc00u; /* +1 prompt, -1 generated. */
        for (uint32_t head = 0u; head < UOCR_KV_HEADS; ++head) {
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t index = kv_cache_flat_index(BATCH_SLOTS,
                                                           CACHE_TOKEN_CAPACITY,
                                                           LAYER,
                                                           SLOT,
                                                           cache_token,
                                                           head,
                                                           dim);
                k_cache[index] = 0x0000u;
                v_cache[index] = value;
            }
        }
    }
    CHECK(compute_decode_attention_expected(q,
                                            k_cache,
                                            v_cache,
                                            BATCH_SLOTS,
                                            PROMPT_TOKEN_CAPACITY,
                                            LAYER,
                                            SLOT,
                                            PROMPT_LENGTH,
                                            wrapped_generated,
                                            expected) == 1);
    memset(out_f32, 0, (size_t)out_values * sizeof(float));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  wrapped_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-5f);
    }
    CHECK(out_f32[0] > -0.98f && out_f32[0] < -0.90f);

    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  UOCR_MAX_POSITIONS,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "generated count") != NULL);

    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  (uocr_metal_dense_output_type)99,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal decode attention output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v_cache);
    free(k_cache);
    free(q);
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
    if (test_metal_final_rmsnorm_f16() != 0) return 1;
    if (test_metal_lm_head_f16() != 0) return 1;
    if (test_metal_argmax_f32() != 0) return 1;
    if (test_metal_select_greedy_with_no_repeat_f32() != 0) return 1;
    if (test_metal_dense_f16() != 0) return 1;
    if (test_metal_attention_qkvo_f16() != 0) return 1;
    if (test_metal_attention_output_residual_f16() != 0) return 1;
    if (test_metal_dense_swiglu_f16() != 0) return 1;
    if (test_metal_moe_shared_experts_f16() != 0) return 1;
    if (test_metal_moe_router_f16() != 0) return 1;
    if (test_metal_moe_selected_experts_decode_f16() != 0) return 1;
    if (test_metal_moe_combine_f16() != 0) return 1;
    if (test_metal_rope_qk_f16() != 0) return 1;
    if (test_metal_prefill_attention_f16() != 0) return 1;
    if (test_metal_prefill_attention_varlen_f16() != 0) return 1;
    if (test_metal_kv_cache_layout_helpers() != 0) return 1;
    if (test_metal_kv_cache_write_f16() != 0) return 1;
    if (test_metal_decode_attention_f16() != 0) return 1;
    if (test_metal_recent_decoder_primitives_stress() != 0) return 1;
    if (test_metal_runtime_arenas() != 0) return 1;
    if (test_metal_model_mapping() != 0) return 1;
    if (test_public_engine_open_initializes_metal() != 0) return 1;
    return 0;
}
