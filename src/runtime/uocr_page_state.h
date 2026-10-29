#ifndef UOCR_PAGE_STATE_H
#define UOCR_PAGE_STATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Shared CPU/Metal layout. RoPE uses token_position and page_position % 2.
 * Routing uses window-local zero-based image/query pages; -1 means no page.
 * Page 0 in the RoPE domain is reserved for prefix/instruction tokens. */
typedef struct uocr_page_position {
    uint32_t token_position;
    uint32_t page_position;
    int32_t image_page;
    int32_t query_page;
} uocr_page_position;

/* Request-owned state, independent of decoder layer. token_counts is borrowed
 * storage for page_capacity buckets (prefix bucket + active-window pages).
 * Advance once per input token, then share its position across all layers.
 * Retain across prefill chunks and decode; reinitialize for a new window before
 * restoring the scheduler's continuation state. No allocation or global state. */
typedef struct uocr_page_state {
    uint32_t marker_count;
    uint32_t prompt_pages;
    uint32_t page_capacity;
    uint32_t page_base_parity;
    uint32_t generated_started;
    uint32_t *token_counts;
} uocr_page_state;

/* Generated-output state only; prompt markers are not counted here.
 * Selection filter: 0 = unrestricted, token+1 = exclude, high bit = force.
 * max_page_tokens=0 disables the cap. Mirrors the pinned HF page-count guard. */
typedef struct uocr_page_guard {
    uint32_t expected_pages, max_page_tokens, markers, page_length;
} uocr_page_guard;
uint32_t uocr_page_guard_filter(const uocr_page_guard *guard);
void uocr_page_guard_accept(uocr_page_guard *guard, int32_t token_id);

int uocr_page_state_init(uocr_page_state *state, uint32_t *token_counts,
                         uint32_t page_capacity, uint64_t document_page_base,
                         char *error, size_t error_size);

/* Matches SGLang PageCoordState/PageState for raw prepared token IDs. Call with
 * generated=0 during prefill and generated=1 for generated tokens fed to decode.
 * The returned coordinate is for this input token, not the token it predicts.
 * Errors leave both state and output unchanged. Buffers must not alias state. */
int uocr_page_state_step(uocr_page_state *state, int32_t token_id, int generated,
                         uocr_page_position *out, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_PAGE_STATE_H */
