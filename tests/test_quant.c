#include "quant/uocr_quant.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

static int make_quant_tensor(uocr_tensor_entry *tensor,
                             uint32_t qtype,
                             uint32_t rows,
                             uint32_t cols,
                             uint64_t payload_offset) {
    memset(tensor, 0, sizeof(*tensor));
    tensor->id = 1u;
    tensor->family = UOCR_TENSOR_FAMILY_LAYER_ATTN;
    tensor->layer = 0;
    tensor->expert = -1;
    tensor->projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensor->usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensor->qtype = qtype;
    tensor->rank = 2u;
    tensor->logical_shape[0] = rows;
    tensor->logical_shape[1] = cols;
    tensor->physical_shape[0] = rows;
    tensor->physical_shape[1] = cols;
    tensor->payload_offset = payload_offset;

    uocr_quant_type_info info;
    uint64_t row_size = 0u;
    CHECK(uocr_quant_get_type_info(qtype, &info) == 1);
    CHECK(uocr_quant_row_size(qtype, cols, &row_size) == 1);
    tensor->block_size = info.block_size;
    tensor->row_size = (uint32_t)row_size;
    tensor->payload_size = row_size * (uint64_t)rows;
    return 0;
}

static int test_quant_type_traits(void) {
    uocr_quant_type_info info;
    CHECK(uocr_quant_get_type_info(UOCR_TENSOR_Q8_0, &info) == 1);
    CHECK(info.is_enabled == 1);
    CHECK(info.is_quantized == 1);
    CHECK(info.block_size == UOCR_Q8_0_BLOCK_SIZE);
    CHECK(info.type_size == UOCR_Q8_0_TYPE_SIZE);
    CHECK(strcmp(info.name, "q8_0") == 0);

    CHECK(uocr_quant_get_type_info(UOCR_TENSOR_Q4_K, &info) == 1);
    CHECK(info.is_enabled == 1);
    CHECK(info.block_size == UOCR_Q4_K_BLOCK_SIZE);
    CHECK(info.type_size == UOCR_Q4_K_TYPE_SIZE);
    CHECK(strcmp(info.name, "q4_K") == 0);

    CHECK(uocr_quant_get_type_info(UOCR_TENSOR_PADDED_Q4_K, &info) == 1);
    CHECK(info.is_quantized == 1);
    CHECK(info.is_enabled == 0);
    CHECK(info.block_size == UOCR_Q4_K_BLOCK_SIZE);
    CHECK(info.type_size == UOCR_Q4_K_TYPE_SIZE);
    uint64_t disabled_row_size = 0u;
    CHECK(uocr_quant_row_size(UOCR_TENSOR_PADDED_Q4_K, 1024u, &disabled_row_size) == 0);

    CHECK(uocr_quant_get_type_info(UOCR_TENSOR_Q2_K, &info) == 1);
    CHECK(info.is_quantized == 1);
    CHECK(info.is_enabled == 0);
    CHECK(uocr_quant_is_enabled(UOCR_TENSOR_IQ2_XXS) == 0);
    CHECK(uocr_quant_get_type_info(9999u, &info) == 0);
    return 0;
}

static int test_quant_row_sizes(void) {
    uint64_t row_size = 0u;
    CHECK(uocr_quant_row_size(UOCR_TENSOR_Q8_0, 32u, &row_size) == 1);
    CHECK(row_size == 34u);
    CHECK(uocr_quant_row_size(UOCR_TENSOR_Q8_0, 64u, &row_size) == 1);
    CHECK(row_size == 68u);
    CHECK(uocr_quant_row_size(UOCR_TENSOR_Q8_0, 33u, &row_size) == 0);

    CHECK(uocr_quant_row_size(UOCR_TENSOR_Q4_K, 256u, &row_size) == 1);
    CHECK(row_size == 144u);
    CHECK(uocr_quant_row_size(UOCR_TENSOR_Q4_K, 512u, &row_size) == 1);
    CHECK(row_size == 288u);
    CHECK(uocr_quant_row_size(UOCR_TENSOR_Q4_K, 128u, &row_size) == 0);

    uint32_t shape[UOCR_TENSOR_MAX_DIMS] = {3u, 512u, 0u, 0u};
    uint64_t payload_size = 0u;
    CHECK(uocr_quant_tensor_payload_size(UOCR_TENSOR_Q4_K, shape, 2u, &payload_size) == 1);
    CHECK(payload_size == 3u * 288u);
    shape[1] = 384u;
    CHECK(uocr_quant_tensor_payload_size(UOCR_TENSOR_Q4_K, shape, 2u, &payload_size) == 0);
    return 0;
}

static int nearly_equal(float a, float b, float tolerance) {
    const float diff = a >= b ? a - b : b - a;
    return diff <= tolerance;
}

static void write_f16_le(uint8_t *dst, uint16_t bits) {
    dst[0] = (uint8_t)(bits & 0xffu);
    dst[1] = (uint8_t)(bits >> 8u);
}

static int test_q8_cpu_dequant_and_dot(void) {
    uint8_t row[UOCR_Q8_0_TYPE_SIZE * 2u];
    memset(row, 0, sizeof(row));
    write_f16_le(row, 0x3800u); /* 0.5 */
    int8_t *q0 = (int8_t *)(void *)(row + 2u);
    for (uint32_t i = 0u; i < UOCR_Q8_0_BLOCK_SIZE; ++i) {
        q0[i] = (int8_t)((int)i - 16);
    }
    write_f16_le(row + UOCR_Q8_0_TYPE_SIZE, 0x3400u); /* 0.25 */
    int8_t *q1 = (int8_t *)(void *)(row + UOCR_Q8_0_TYPE_SIZE + 2u);
    q1[0] = 12;
    q1[1] = -8; /* padded/logically ignored in this test */

    float dequant[33];
    CHECK(uocr_quant_q8_0_dequantize_row_f32(row, 33u, 64u, dequant) == 1);
    for (uint32_t i = 0u; i < 32u; ++i) {
        CHECK(nearly_equal(dequant[i], 0.5f * (float)((int)i - 16), 1.0e-6f));
    }
    CHECK(nearly_equal(dequant[32], 3.0f, 1.0e-6f));

    float values[33];
    float expected = 0.0f;
    for (uint32_t i = 0u; i < 33u; ++i) {
        values[i] = (float)((int)(i % 7u) - 3) * 0.125f;
        expected += dequant[i] * values[i];
    }
    float actual = 0.0f;
    CHECK(uocr_quant_q8_0_dot_row_f32(row, values, 33u, 64u, &actual) == 1);
    CHECK(nearly_equal(actual, expected, 1.0e-5f));

    CHECK(uocr_quant_q8_0_dequantize_row_f32(row, 65u, 64u, dequant) == 0);
    CHECK(uocr_quant_q8_0_dot_row_f32(row, values, 33u, 33u, &actual) == 0);
    CHECK(uocr_quant_q8_0_dot_row_f32(NULL, values, 33u, 64u, &actual) == 0);
    return 0;
}

static int test_quant_tensor_input_widths(void) {
    uint32_t logical = 0u;
    uint32_t physical = 0u;
    uocr_tensor_entry tensor;

    CHECK(make_quant_tensor(&tensor, UOCR_TENSOR_Q8_0, 2u, 64u, 4096u) == 0);
    CHECK(uocr_quant_tensor_input_widths(&tensor, &logical, &physical) == 1);
    CHECK(logical == 64u);
    CHECK(physical == 64u);

    tensor.logical_shape[1] = 33u;
    tensor.physical_shape[1] = 64u;
    CHECK(uocr_quant_tensor_input_widths(&tensor, &logical, &physical) == 1);
    CHECK(logical == 33u);
    CHECK(physical == 64u);

    CHECK(make_quant_tensor(&tensor, UOCR_TENSOR_Q4_K, 3u, 256u, 4096u) == 0);
    CHECK(uocr_quant_tensor_input_widths(&tensor, &logical, &physical) == 1);
    CHECK(logical == 256u);
    CHECK(physical == 256u);
    tensor.logical_shape[1] = 384u;
    tensor.physical_shape[1] = 512u;
    CHECK(uocr_quant_tensor_input_widths(&tensor, &logical, &physical) == 0);

    tensor.qtype = UOCR_TENSOR_F16;
    CHECK(uocr_quant_tensor_input_widths(&tensor, &logical, &physical) == 0);
    CHECK(uocr_quant_tensor_input_widths(NULL, &logical, &physical) == 0);
    CHECK(uocr_quant_tensor_input_widths(&tensor, NULL, &physical) == 0);
    CHECK(uocr_quant_tensor_input_widths(&tensor, &logical, NULL) == 0);
    return 0;
}

static int test_quant_tensor_validation(void) {
    char error[256];
    uocr_tensor_entry tensor;
    CHECK(make_quant_tensor(&tensor, UOCR_TENSOR_Q8_0, 2u, 64u, 4096u) == 0);
    CHECK(uocr_quant_validate_tensor_entry(&tensor, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    memset(&tensor, 0, sizeof(tensor));
    tensor.id = 1u;
    tensor.family = UOCR_TENSOR_FAMILY_LAYER_ATTN;
    tensor.layer = 0;
    tensor.expert = -1;
    tensor.projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensor.usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensor.qtype = UOCR_TENSOR_Q8_0;
    tensor.rank = 2u;
    tensor.logical_shape[0] = 1u;
    tensor.logical_shape[1] = 33u;
    tensor.physical_shape[0] = 1u;
    tensor.physical_shape[1] = 64u;
    tensor.payload_offset = 4096u;
    tensor.block_size = UOCR_Q8_0_BLOCK_SIZE;
    tensor.row_size = UOCR_Q8_0_TYPE_SIZE * 2u;
    tensor.payload_size = tensor.row_size;
    CHECK(uocr_quant_validate_tensor_entry(&tensor, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    CHECK(make_quant_tensor(&tensor, UOCR_TENSOR_Q8_0, 2u, 64u, 4096u) == 0);
    tensor.row_size = 67u;
    CHECK(uocr_quant_validate_tensor_entry(&tensor, error, sizeof(error)) == 0);
    CHECK(strstr(error, "row size mismatch") != NULL);

    CHECK(make_quant_tensor(&tensor, UOCR_TENSOR_Q4_K, 3u, 256u, 4096u) == 0);
    CHECK(uocr_quant_validate_tensor_entry(&tensor, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    tensor.physical_shape[1] = 512u;
    CHECK(uocr_quant_validate_tensor_entry(&tensor, error, sizeof(error)) == 0);
    CHECK(strstr(error, "padded inner dimension") != NULL);

    CHECK(make_quant_tensor(&tensor, UOCR_TENSOR_Q8_0, 1u, 32u, 4096u) == 0);
    tensor.qtype = UOCR_TENSOR_Q2_K;
    CHECK(uocr_quant_validate_tensor_entry(&tensor, error, sizeof(error)) == 0);
    CHECK(strstr(error, "disabled") != NULL);
    return 0;
}

int main(void) {
    if (test_quant_type_traits() != 0) return 1;
    if (test_quant_row_sizes() != 0) return 1;
    if (test_q8_cpu_dequant_and_dot() != 0) return 1;
    if (test_quant_tensor_input_widths() != 0) return 1;
    if (test_quant_tensor_validation() != 0) return 1;
    return 0;
}
