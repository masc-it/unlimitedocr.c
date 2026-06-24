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

static int test_quant_tensor_validation(void) {
    char error[256];
    uocr_tensor_entry tensor;
    CHECK(make_quant_tensor(&tensor, UOCR_TENSOR_Q8_0, 2u, 64u, 4096u) == 0);
    CHECK(uocr_quant_validate_tensor_entry(&tensor, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

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
    if (test_quant_tensor_validation() != 0) return 1;
    return 0;
}
