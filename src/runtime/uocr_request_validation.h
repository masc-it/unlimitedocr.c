#ifndef UOCR_REQUEST_VALIDATION_H
#define UOCR_REQUEST_VALIDATION_H

#include <stddef.h>
#include <stdint.h>

#include "unlimitedocr.h"

typedef struct uocr_request_limits {
    uint32_t max_prompt_tokens;
    uint32_t max_gen_tokens;
} uocr_request_limits;

int uocr_validate_prepared_request(const uocr_prepared_request *request,
                                   const uocr_request_limits *limits,
                                   char *error,
                                   size_t error_size);

uint32_t uocr_count_image_placeholders(const uocr_prepared_request *request);

#endif /* UOCR_REQUEST_VALIDATION_H */
