#ifndef UOCR_QUANT_H
#define UOCR_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "model/uocr_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Quantization format helpers for Unlimited-OCR model files.
 *
 * The initial block sizes and packed type sizes are adapted from the
 * MIT-licensed DS4 GGUF tools / GGML quantization metadata:
 *   - Q8_0: block size 32, packed block size 34 bytes
 *   - Q4_K: block size 256, packed block size 144 bytes
 *
 * The actual quantization kernels are introduced in later milestones; these
 * helpers establish the stable row-size and tensor-entry validation contract.
 */
#define UOCR_Q8_0_BLOCK_SIZE 32u
#define UOCR_Q8_0_TYPE_SIZE 34u
#define UOCR_Q4_K_BLOCK_SIZE 256u
#define UOCR_Q4_K_TYPE_SIZE 144u

typedef struct uocr_quant_type_info {
    uint32_t qtype;
    const char *name;
    uint32_t block_size;
    uint32_t type_size;
    int is_quantized;
    int is_enabled;
} uocr_quant_type_info;

int uocr_quant_get_type_info(uint32_t qtype, uocr_quant_type_info *out_info);
int uocr_quant_is_quantized(uint32_t qtype);
int uocr_quant_is_enabled(uint32_t qtype);
int uocr_quant_row_size(uint32_t qtype, uint64_t physical_cols, uint64_t *out_row_size);
int uocr_quant_tensor_payload_size(uint32_t qtype,
                                   const uint32_t *physical_shape,
                                   uint32_t rank,
                                   uint64_t *out_payload_size);
int uocr_quant_validate_tensor_entry(const uocr_tensor_entry *tensor, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_QUANT_H */
