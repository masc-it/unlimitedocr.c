#include "runtime/uocr_sequence.h"

#include "model/uocr_constants.h"

#include <limits.h>
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

static int sequence_token_id_valid(int32_t token_id) {
    return token_id >= 0 && (uint32_t)token_id < UOCR_VOCAB_SIZE;
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
    state.prompt_tokens = request->input_ids;
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

int uocr_sequence_attach_generation_buffers(uocr_sequence_state *state,
                                            int32_t *generated_tokens,
                                            uint32_t generated_capacity,
                                            int32_t *token_history,
                                            uint32_t token_history_capacity) {
    if (state == NULL || state->prompt_tokens == NULL || token_history == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (state->generated_count != 0u || state->position != state->prompt_token_count) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (state->max_new_tokens != 0u && generated_tokens == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (generated_capacity < state->max_new_tokens || token_history_capacity < state->prompt_token_count ||
        token_history_capacity - state->prompt_token_count < state->max_new_tokens) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    memcpy(token_history, state->prompt_tokens, (size_t)state->prompt_token_count * sizeof(int32_t));
    state->generated_tokens = generated_tokens;
    state->generated_capacity = generated_capacity;
    state->token_history = token_history;
    state->token_history_capacity = token_history_capacity;
    state->token_history_count = state->prompt_token_count;
    return UOCR_OK;
}

int uocr_sequence_no_repeat_config(const uocr_sequence_state *state,
                                   uocr_no_repeat_ngram_config *out_config) {
    if (out_config == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    memset(out_config, 0, sizeof(*out_config));
    if (state == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (state->no_repeat_ngram_size == 0u) {
        return UOCR_OK;
    }
    if (state->token_history != NULL) {
        out_config->sequence = state->token_history;
        out_config->sequence_len = state->token_history_count;
    } else if (state->generated_count == 0u && state->prompt_tokens != NULL) {
        out_config->sequence = state->prompt_tokens;
        out_config->sequence_len = state->prompt_token_count;
    } else {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    out_config->ngram_size = state->no_repeat_ngram_size;
    out_config->window = state->no_repeat_window;
    return UOCR_OK;
}

int uocr_sequence_generation_done(const uocr_sequence_state *state) {
    if (state == NULL) {
        return 1;
    }
    return state->eos || state->generated_count >= state->max_new_tokens;
}

int uocr_sequence_accept_generated_token(uocr_sequence_state *state,
                                         int32_t token_id,
                                         int32_t *generated_tokens,
                                         uint32_t generated_capacity) {
    if (state == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    int32_t *target_generated = generated_tokens != NULL ? generated_tokens : state->generated_tokens;
    const uint32_t target_capacity = generated_tokens != NULL ? generated_capacity : state->generated_capacity;
    if (target_generated == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (!sequence_token_id_valid(token_id)) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (uocr_sequence_generation_done(state)) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (state->generated_count >= target_capacity) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    if (state->token_history != NULL && state->token_history_count >= state->token_history_capacity) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    if (state->position == UINT32_MAX) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    target_generated[state->generated_count] = token_id;
    if (state->token_history != NULL) {
        state->token_history[state->token_history_count] = token_id;
        ++state->token_history_count;
    }
    ++state->generated_count;
    ++state->position;
    if (token_id == UOCR_TOKEN_EOS) {
        state->eos = 1;
    }
    return UOCR_OK;
}
