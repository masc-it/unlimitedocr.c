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

/* True when the decode loop must stop for this sequence.  Generation stops
 * after EOS token id 1 has been emitted or after max_new_tokens have been
 * accepted.  A NULL state is treated as stopped for defensive callers.
 */
int uocr_sequence_generation_done(const uocr_sequence_state *state);

/* Append one selected token to caller-owned generated_tokens storage, then
 * update generated_count, absolute position, and EOS state.  The helper does
 * no allocation and is intended to sit immediately after backend greedy
 * selection in the decode loop.
 */
int uocr_sequence_accept_generated_token(uocr_sequence_state *state,
                                         int32_t token_id,
                                         int32_t *generated_tokens,
                                         uint32_t generated_capacity);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_SEQUENCE_H */
