#include "quant/uocr_quant.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int quant_fail(char *error, size_t error_size, const char *fmt, ...) {
    if (error != NULL && error_size > 0u) {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(error, error_size, fmt, ap);
        va_end(ap);
        error[error_size - 1u] = '\0';
    }
    return 0;
}

static void quant_clear_error(char *error, size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL) {
        return 0;
    }
    if (a != 0u && b > UINT64_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

int uocr_quant_get_type_info(uint32_t qtype, uocr_quant_type_info *out_info) {
    if (out_info == NULL) {
        return 0;
    }
    uocr_quant_type_info info;
    memset(&info, 0, sizeof(info));
    info.qtype = qtype;

    switch (qtype) {
        case UOCR_TENSOR_F16:
            info.name = "f16";
            info.type_size = 2u;
            info.is_enabled = 1;
            break;
        case UOCR_TENSOR_F32:
            info.name = "f32";
            info.type_size = 4u;
            info.is_enabled = 1;
            break;
        case UOCR_TENSOR_Q8_0:
            info.name = "q8_0";
            info.block_size = UOCR_Q8_0_BLOCK_SIZE;
            info.type_size = UOCR_Q8_0_TYPE_SIZE;
            info.is_quantized = 1;
            info.is_enabled = 1;
            break;
        case UOCR_TENSOR_Q4_K:
            info.name = "q4_K";
            info.block_size = UOCR_Q4_K_BLOCK_SIZE;
            info.type_size = UOCR_Q4_K_TYPE_SIZE;
            info.is_quantized = 1;
            info.is_enabled = 1;
            break;
        case UOCR_TENSOR_PADDED_Q4_K:
            info.name = "padded_q4_K";
            info.block_size = UOCR_Q4_K_BLOCK_SIZE;
            info.type_size = UOCR_Q4_K_TYPE_SIZE;
            info.is_quantized = 1;
            info.is_enabled = 0;
            break;
        case UOCR_TENSOR_Q2_K:
            info.name = "q2_K";
            info.block_size = 256u;
            info.type_size = 84u;
            info.is_quantized = 1;
            info.is_enabled = 0;
            break;
        case UOCR_TENSOR_IQ2_XXS:
            info.name = "iq2_xxs";
            info.block_size = 256u;
            info.type_size = 66u;
            info.is_quantized = 1;
            info.is_enabled = 0;
            break;
        default:
            return 0;
    }

    *out_info = info;
    return 1;
}

int uocr_quant_is_quantized(uint32_t qtype) {
    uocr_quant_type_info info;
    return uocr_quant_get_type_info(qtype, &info) != 0 && info.is_quantized != 0;
}

int uocr_quant_is_enabled(uint32_t qtype) {
    uocr_quant_type_info info;
    return uocr_quant_get_type_info(qtype, &info) != 0 && info.is_enabled != 0;
}

int uocr_quant_row_size(uint32_t qtype, uint64_t physical_cols, uint64_t *out_row_size) {
    if (out_row_size == NULL) {
        return 0;
    }
    *out_row_size = 0u;

    uocr_quant_type_info info;
    if (!uocr_quant_get_type_info(qtype, &info) || info.is_quantized == 0 || info.is_enabled == 0 ||
        info.block_size == 0u || info.type_size == 0u || physical_cols == 0u) {
        return 0;
    }
    if ((physical_cols % (uint64_t)info.block_size) != 0u) {
        return 0;
    }

    const uint64_t blocks = physical_cols / (uint64_t)info.block_size;
    return checked_mul_u64(blocks, (uint64_t)info.type_size, out_row_size);
}

static uint64_t shape_rows_except_last(const uint32_t *shape, uint32_t rank) {
    uint64_t rows = 1u;
    if (shape == NULL || rank < 2u) {
        return 0u;
    }
    for (uint32_t i = 0u; i + 1u < rank; ++i) {
        if (shape[i] == 0u || !checked_mul_u64(rows, (uint64_t)shape[i], &rows)) {
            return 0u;
        }
    }
    return rows;
}

int uocr_quant_tensor_payload_size(uint32_t qtype,
                                   const uint32_t *physical_shape,
                                   uint32_t rank,
                                   uint64_t *out_payload_size) {
    if (out_payload_size == NULL) {
        return 0;
    }
    *out_payload_size = 0u;
    if (physical_shape == NULL || rank < 2u || rank > UOCR_TENSOR_MAX_DIMS) {
        return 0;
    }

    uint64_t row_size = 0u;
    if (!uocr_quant_row_size(qtype, (uint64_t)physical_shape[rank - 1u], &row_size)) {
        return 0;
    }
    const uint64_t rows = shape_rows_except_last(physical_shape, rank);
    if (rows == 0u) {
        return 0;
    }
    return checked_mul_u64(rows, row_size, out_payload_size);
}

int uocr_quant_tensor_input_widths(const uocr_tensor_entry *tensor,
                                   uint32_t *out_logical_input_width,
                                   uint32_t *out_physical_input_width) {
    if (out_logical_input_width == NULL || out_physical_input_width == NULL) {
        return 0;
    }
    *out_logical_input_width = 0u;
    *out_physical_input_width = 0u;
    if (tensor == NULL || tensor->rank < 2u || tensor->rank > UOCR_TENSOR_MAX_DIMS) {
        return 0;
    }

    uocr_quant_type_info info;
    if (!uocr_quant_get_type_info(tensor->qtype, &info) || info.is_quantized == 0) {
        return 0;
    }
    for (uint32_t i = 0u; i < tensor->rank; ++i) {
        if (tensor->logical_shape[i] == 0u || tensor->physical_shape[i] == 0u) {
            return 0;
        }
    }
    for (uint32_t i = 0u; i + 1u < tensor->rank; ++i) {
        if (tensor->logical_shape[i] != tensor->physical_shape[i]) {
            return 0;
        }
    }

    const uint32_t logical = tensor->logical_shape[tensor->rank - 1u];
    const uint32_t physical = tensor->physical_shape[tensor->rank - 1u];
    if (logical == 0u || physical < logical || info.block_size == 0u || (physical % info.block_size) != 0u) {
        return 0;
    }
    if (physical != logical && tensor->qtype != UOCR_TENSOR_Q8_0 && tensor->qtype != UOCR_TENSOR_PADDED_Q4_K) {
        return 0;
    }
    *out_logical_input_width = logical;
    *out_physical_input_width = physical;
    return 1;
}

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16u;
    uint32_t exp = ((uint32_t)h >> 10u) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1u;
                --e;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(e + 127) << 23u) | (mant << 13u);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13u);
    } else {
        exp = exp + (127u - 15u);
        bits = sign | (exp << 23u) | (mant << 13u);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static int validate_q8_0_row_request(const void *row,
                                     uint32_t logical_cols,
                                     uint32_t physical_cols,
                                     const void *out_or_values) {
    if (row == NULL || out_or_values == NULL || logical_cols == 0u || physical_cols == 0u) {
        return 0;
    }
    if (logical_cols > physical_cols) {
        return 0;
    }
    if ((physical_cols % UOCR_Q8_0_BLOCK_SIZE) != 0u) {
        return 0;
    }
    return 1;
}

int uocr_quant_q8_0_dequantize_row_f32(const void *row,
                                       uint32_t logical_cols,
                                       uint32_t physical_cols,
                                       float *out_values) {
    if (!validate_q8_0_row_request(row, logical_cols, physical_cols, out_values)) {
        return 0;
    }
    const uint8_t *bytes = (const uint8_t *)row;
    for (uint32_t col = 0u; col < logical_cols; ++col) {
        const uint32_t block = col / UOCR_Q8_0_BLOCK_SIZE;
        const uint32_t in_block = col % UOCR_Q8_0_BLOCK_SIZE;
        const uint8_t *packed = bytes + (uint64_t)block * UOCR_Q8_0_TYPE_SIZE;
        const uint16_t scale_bits = (uint16_t)((uint16_t)packed[0] | ((uint16_t)packed[1] << 8u));
        const float scale = f16_bits_to_f32(scale_bits);
        const int8_t q = (int8_t)packed[2u + in_block];
        out_values[col] = scale * (float)q;
    }
    return 1;
}

int uocr_quant_q8_0_dot_row_f32(const void *row,
                                const float *values,
                                uint32_t logical_cols,
                                uint32_t physical_cols,
                                float *out_dot) {
    if (!validate_q8_0_row_request(row, logical_cols, physical_cols, values) || out_dot == NULL) {
        return 0;
    }
    const uint8_t *bytes = (const uint8_t *)row;
    float sum = 0.0f;
    for (uint32_t col = 0u; col < logical_cols; ++col) {
        const uint32_t block = col / UOCR_Q8_0_BLOCK_SIZE;
        const uint32_t in_block = col % UOCR_Q8_0_BLOCK_SIZE;
        const uint8_t *packed = bytes + (uint64_t)block * UOCR_Q8_0_TYPE_SIZE;
        const uint16_t scale_bits = (uint16_t)((uint16_t)packed[0] | ((uint16_t)packed[1] << 8u));
        const float scale = f16_bits_to_f32(scale_bits);
        const int8_t q = (int8_t)packed[2u + in_block];
        sum += scale * (float)q * values[col];
    }
    *out_dot = sum;
    return 1;
}

static int tensor_shape_elements(const uint32_t *shape, uint32_t rank, uint64_t *out_elements) {
    if (shape == NULL || out_elements == NULL || rank == 0u || rank > UOCR_TENSOR_MAX_DIMS) {
        return 0;
    }
    uint64_t elements = 1u;
    for (uint32_t i = 0u; i < rank; ++i) {
        if (shape[i] == 0u || !checked_mul_u64(elements, (uint64_t)shape[i], &elements)) {
            return 0;
        }
    }
    *out_elements = elements;
    return 1;
}

static int validate_plain_tensor_entry(const uocr_tensor_entry *tensor,
                                       const uocr_quant_type_info *info,
                                       char *error,
                                       size_t error_size) {
    uint64_t elements = 0u;
    uint64_t expected_payload_size = 0u;
    if (!tensor_shape_elements(tensor->physical_shape, tensor->rank, &elements) ||
        !checked_mul_u64(elements, (uint64_t)info->type_size, &expected_payload_size)) {
        return quant_fail(error, error_size, "tensor entry has invalid plain physical shape");
    }
    if (tensor->payload_size != expected_payload_size) {
        return quant_fail(error,
                          error_size,
                          "tensor entry payload size mismatch: got %llu expected %llu",
                          (unsigned long long)tensor->payload_size,
                          (unsigned long long)expected_payload_size);
    }
    if (tensor->block_size != 0u || tensor->row_size != 0u || tensor->scale_offset != 0u || tensor->scale_size != 0u ||
        tensor->min_offset != 0u || tensor->min_size != 0u) {
        return quant_fail(error, error_size, "plain tensor entry has quantization metadata");
    }
    return 1;
}

static int validate_quant_shape(const uocr_tensor_entry *tensor,
                                const uocr_quant_type_info *info,
                                char *error,
                                size_t error_size) {
    if (tensor->rank < 2u || tensor->rank > UOCR_TENSOR_MAX_DIMS) {
        return quant_fail(error, error_size, "quantized tensor entry rank %u is invalid", tensor->rank);
    }
    for (uint32_t i = 0u; i < tensor->rank; ++i) {
        if (tensor->logical_shape[i] == 0u || tensor->physical_shape[i] == 0u) {
            return quant_fail(error, error_size, "quantized tensor entry has zero shape dimension");
        }
    }
    for (uint32_t i = 0u; i + 1u < tensor->rank; ++i) {
        if (tensor->physical_shape[i] != tensor->logical_shape[i]) {
            return quant_fail(error,
                              error_size,
                              "quantized tensor entry physical dim %u=%u differs from logical dim %u",
                              i,
                              tensor->physical_shape[i],
                              tensor->logical_shape[i]);
        }
    }
    const uint32_t logical_inner = tensor->logical_shape[tensor->rank - 1u];
    const uint32_t physical_inner = tensor->physical_shape[tensor->rank - 1u];
    if (physical_inner < logical_inner) {
        return quant_fail(error,
                          error_size,
                          "%s tensor entry physical inner dimension %u is smaller than logical %u",
                          info->name,
                          physical_inner,
                          logical_inner);
    }
    if (physical_inner != logical_inner && tensor->qtype != UOCR_TENSOR_Q8_0) {
        return quant_fail(error, error_size, "%s tensor entry uses unsupported padded inner dimension", info->name);
    }
    return 1;
}

int uocr_quant_validate_tensor_entry(const uocr_tensor_entry *tensor, char *error, size_t error_size) {
    quant_clear_error(error, error_size);
    if (tensor == NULL) {
        return quant_fail(error, error_size, "tensor entry is null");
    }

    uocr_quant_type_info info;
    if (!uocr_quant_get_type_info(tensor->qtype, &info)) {
        return quant_fail(error, error_size, "tensor entry has unknown qtype %u", tensor->qtype);
    }
    if (info.is_enabled == 0) {
        return quant_fail(error, error_size, "tensor qtype %s is disabled in this build", info.name);
    }
    if (info.is_quantized == 0) {
        return validate_plain_tensor_entry(tensor, &info, error, error_size);
    }
    if (!validate_quant_shape(tensor, &info, error, error_size)) {
        return 0;
    }

    uint64_t expected_row_size = 0u;
    if (!uocr_quant_row_size(tensor->qtype, (uint64_t)tensor->physical_shape[tensor->rank - 1u], &expected_row_size)) {
        return quant_fail(error,
                          error_size,
                          "tensor qtype %s inner dimension %u is not aligned to block size %u",
                          info.name,
                          tensor->physical_shape[tensor->rank - 1u],
                          info.block_size);
    }
    if (tensor->block_size != info.block_size) {
        return quant_fail(error,
                          error_size,
                          "tensor qtype %s block size mismatch: got %u expected %u",
                          info.name,
                          tensor->block_size,
                          info.block_size);
    }
    if (tensor->row_size != (uint32_t)expected_row_size) {
        return quant_fail(error,
                          error_size,
                          "tensor qtype %s row size mismatch: got %u expected %llu",
                          info.name,
                          tensor->row_size,
                          (unsigned long long)expected_row_size);
    }

    uint64_t expected_payload_size = 0u;
    if (!uocr_quant_tensor_payload_size(tensor->qtype, tensor->physical_shape, tensor->rank, &expected_payload_size)) {
        return quant_fail(error, error_size, "tensor qtype %s payload-size overflow", info.name);
    }
    if (tensor->payload_size != expected_payload_size) {
        return quant_fail(error,
                          error_size,
                          "tensor qtype %s payload size mismatch: got %llu expected %llu",
                          info.name,
                          (unsigned long long)tensor->payload_size,
                          (unsigned long long)expected_payload_size);
    }
    if (tensor->scale_offset != 0u || tensor->scale_size != 0u || tensor->min_offset != 0u || tensor->min_size != 0u) {
        return quant_fail(error, error_size, "tensor qtype %s uses unsupported side scale/min buffers", info.name);
    }

    quant_clear_error(error, error_size);
    return 1;
}
