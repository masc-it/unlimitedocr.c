#include "runtime/uocr_sequence.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int fail(char *error, size_t error_size, const char *fmt, ...) {
    if (error != NULL && error_size > 0u) {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(error, error_size, fmt, ap);
        va_end(ap);
        error[error_size - 1u] = '\0';
    }
    return UOCR_ERROR_INVALID_ARGUMENT;
}

int uocr_build_sequence_state(const uocr_prepared_request *request,
                              uocr_sequence_state *out_state,
                              char *error,
                              size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
    if (request == NULL || out_state == NULL) {
        return fail(error, error_size, "request and out_state must be non-null");
    }
    if (request->image_mask == NULL) {
        return fail(error, error_size, "image_mask pointer is null");
    }

    uint32_t first_image = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_count = 0u;
    uint32_t last_image = 0u;
    for (uint32_t i = 0u; i < request->n_tokens; ++i) {
        if (request->image_mask[i] == 0u) {
            continue;
        }
        if (request->image_mask[i] != 1u) {
            return fail(error, error_size, "image_mask[%u] must be 0 or 1", i);
        }
        if (first_image == UOCR_SEQUENCE_NO_IMAGE_SPAN) {
            first_image = i;
        }
        last_image = i;
        ++image_count;
    }

    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    state.prompt_token_count = request->n_tokens;
    state.max_new_tokens = request->max_new_tokens;
    state.no_repeat_ngram_size = request->no_repeat_ngram_size;
    state.no_repeat_window = request->no_repeat_window;
    state.position = request->n_tokens;
    state.eos = 0;

    if (image_count == 0u) {
        state.image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
        state.image_span_length = 0u;
        state.text_prefix_length = request->n_tokens;
        state.text_suffix_start = request->n_tokens;
        state.text_suffix_length = 0u;
        *out_state = state;
        return UOCR_OK;
    }

    const uint32_t image_span_length = last_image - first_image + 1u;
    if (image_span_length != image_count) {
        return fail(error,
                    error_size,
                    "image placeholders must be one contiguous span; first=%u last=%u count=%u",
                    first_image,
                    last_image,
                    image_count);
    }

    state.image_span_start = first_image;
    state.image_span_length = image_span_length;
    state.text_prefix_length = first_image;
    state.text_suffix_start = last_image + 1u;
    state.text_suffix_length = request->n_tokens - state.text_suffix_start;
    *out_state = state;
    return UOCR_OK;
}
