#ifndef UOCR_SEQUENCE_H
#define UOCR_SEQUENCE_H

#include <stddef.h>
#include <stdint.h>

#include "unlimitedocr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define UOCR_SEQUENCE_NO_IMAGE_SPAN UINT32_MAX

typedef struct uocr_sequence_state {
    uint32_t prompt_token_count;
    uint32_t image_span_start;
    uint32_t image_span_length;
    uint32_t text_prefix_length;
    uint32_t text_suffix_start;
    uint32_t text_suffix_length;
    uint32_t max_new_tokens;
    uint32_t no_repeat_ngram_size;
    uint32_t no_repeat_window;
    uint32_t generated_count;
    uint32_t position;
    int eos;
} uocr_sequence_state;

int uocr_build_sequence_state(const uocr_prepared_request *request,
                              uocr_sequence_state *out_state,
                              char *error,
                              size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_SEQUENCE_H */
