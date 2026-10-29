#include "runtime/uocr_page_state.h"

#include "model/uocr_constants.h"
#include "unlimitedocr.h"

#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(uocr_page_position) == 16u, "CPU/Metal page-position layout mismatch");

static int fail(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size != 0u) {
        (void)snprintf(error, error_size, "%s", message);
    }
    return UOCR_ERROR_INVALID_ARGUMENT;
}

int uocr_page_state_init(uocr_page_state *state, uint32_t *token_counts,
                         uint32_t page_capacity, uint64_t document_page_base,
                         char *error, size_t error_size) {
    if (error != NULL && error_size != 0u) error[0] = '\0';
    if (state == NULL || token_counts == NULL || page_capacity < 2u ||
        page_capacity > UOCR_V2_MAX_POSITIONS + 1u) {
        return fail(error, error_size, "page state requires prefix/page counter storage");
    }
    memset(token_counts, 0, (size_t)page_capacity * sizeof(*token_counts));
    *state = (uocr_page_state){
        .page_capacity = page_capacity,
        .page_base_parity = (uint32_t)(document_page_base & 1u),
        .token_counts = token_counts,
    };
    return UOCR_OK;
}

int uocr_page_state_step(uocr_page_state *state, int32_t token_id, int generated,
                         uocr_page_position *out, char *error, size_t error_size) {
    if (error != NULL && error_size != 0u) error[0] = '\0';
    if (state == NULL || state->token_counts == NULL || state->page_capacity < 2u || out == NULL ||
        token_id < 0 || (uint32_t)token_id >= UOCR_VOCAB_SIZE || (generated != 0 && generated != 1)) {
        return fail(error, error_size, "invalid page-state token request");
    }
    if ((!generated && state->generated_started) || (generated && state->prompt_pages == 0u)) {
        return fail(error, error_size, "invalid prefill/decode page-state transition");
    }
    const uint32_t marker = state->marker_count + (token_id == UOCR_TOKEN_PAGE ? 1u : 0u);
    if (marker < state->marker_count || marker > INT32_MAX) {
        return fail(error, error_size, "page marker count overflow");
    }
    const int image = token_id == UOCR_TOKEN_IMAGE;
    if (image && marker == 0u) {
        return fail(error, error_size, "v2 image tokens require a preceding page marker");
    }
    const uint32_t prompt_pages = image && marker > state->prompt_pages ? marker : state->prompt_pages;
    const int in_prefill = !generated && (image || marker > prompt_pages);
    uint32_t page = marker > prompt_pages ? marker - prompt_pages : 0u;
    if (page > prompt_pages) page = prompt_pages; /* Extra generated markers stay on the last RoPE page. */
    if (in_prefill) page = marker;
    if (prompt_pages >= state->page_capacity || page >= state->page_capacity ||
        state->token_counts[page] >= UOCR_V2_MAX_POSITIONS) {
        return fail(error, error_size, "page position exceeds active-window capacity");
    }
    const int32_t image_page = image ? (int32_t)marker - 1 : -1;
    int32_t query_page = -1;
    if (image) {
        query_page = image_page;
    } else if (marker > prompt_pages) {
        query_page = generated ? (int32_t)(marker - prompt_pages) - 1 : (int32_t)marker - 1;
    }
    const uocr_page_position position = {
        .token_position = state->token_counts[page],
        /* Shift only real pages; the prefix/instruction bucket has no document page. */
        .page_position = page + (page != 0u ? state->page_base_parity : 0u),
        .image_page = image_page,
        .query_page = query_page,
    };
    ++state->token_counts[page];
    state->marker_count = marker;
    state->prompt_pages = prompt_pages;
    state->generated_started = (uint32_t)generated;
    *out = position;
    return UOCR_OK;
}
