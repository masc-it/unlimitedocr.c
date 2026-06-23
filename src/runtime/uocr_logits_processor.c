#include "runtime/uocr_logits_processor.h"

#include <math.h>
#include <stddef.h>

static int valid_token_id(int32_t token_id, uint32_t vocab_size) {
    return token_id >= 0 && (uint32_t)token_id < vocab_size;
}

static int is_whitelisted(int32_t token_id, const int32_t *whitelist_token_ids, uint32_t whitelist_count) {
    for (uint32_t i = 0u; i < whitelist_count; ++i) {
        if (whitelist_token_ids[i] == token_id) {
            return 1;
        }
    }
    return 0;
}

static int prefix_matches(const int32_t *sequence, uint32_t idx, uint32_t current_prefix_start, uint32_t ngram_size) {
    if (ngram_size <= 1u) {
        return 1;
    }
    const uint32_t prefix_len = ngram_size - 1u;
    for (uint32_t j = 0u; j < prefix_len; ++j) {
        if (sequence[idx + j] != sequence[current_prefix_start + j]) {
            return 0;
        }
    }
    return 1;
}

static int token_was_banned_earlier(const int32_t *sequence,
                                    uint32_t search_start,
                                    uint32_t idx,
                                    uint32_t current_prefix_start,
                                    uint32_t ngram_size,
                                    int32_t token_id) {
    for (uint32_t prev = search_start; prev < idx; ++prev) {
        if (sequence[prev + ngram_size - 1u] == token_id &&
            prefix_matches(sequence, prev, current_prefix_start, ngram_size)) {
            return 1;
        }
    }
    return 0;
}

static int validate_whitelist(const int32_t *whitelist_token_ids, uint32_t whitelist_count, uint32_t vocab_size) {
    if (whitelist_count != 0u && whitelist_token_ids == NULL) {
        return 0;
    }
    for (uint32_t i = 0u; i < whitelist_count; ++i) {
        if (!valid_token_id(whitelist_token_ids[i], vocab_size)) {
            return 0;
        }
    }
    return 1;
}

static int no_repeat_bounds(uint32_t sequence_len,
                            uint32_t ngram_size,
                            uint32_t window,
                            uint32_t *out_search_start,
                            uint32_t *out_search_end,
                            uint32_t *out_current_prefix_start) {
    if (ngram_size == 0u || sequence_len < ngram_size) {
        return 0;
    }

    const uint32_t effective_window = window == 0u || window > sequence_len ? sequence_len : window;
    const uint32_t search_start = sequence_len - effective_window;
    const uint32_t search_end = sequence_len - ngram_size + 1u;
    if (search_end <= search_start) {
        return 0;
    }

    *out_search_start = search_start;
    *out_search_end = search_end;
    *out_current_prefix_start = ngram_size > 1u ? sequence_len - (ngram_size - 1u) : sequence_len;
    return 1;
}

int uocr_no_repeat_ngram_collect_banned(const int32_t *sequence,
                                        uint32_t sequence_len,
                                        uint32_t vocab_size,
                                        uint32_t ngram_size,
                                        uint32_t window,
                                        const int32_t *whitelist_token_ids,
                                        uint32_t whitelist_count,
                                        int32_t *out_banned,
                                        uint32_t out_capacity,
                                        uint32_t *out_count) {
    if (out_count == NULL || vocab_size == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    *out_count = 0u;

    if (ngram_size == 0u) {
        return UOCR_OK;
    }
    if (sequence == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (out_capacity != 0u && out_banned == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (!validate_whitelist(whitelist_token_ids, whitelist_count, vocab_size)) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint32_t search_start = 0u;
    uint32_t search_end = 0u;
    uint32_t current_prefix_start = 0u;
    if (!no_repeat_bounds(sequence_len, ngram_size, window, &search_start, &search_end, &current_prefix_start)) {
        return UOCR_OK;
    }

    uint32_t required = 0u;
    int overflowed = 0;
    for (uint32_t idx = search_start; idx < search_end; ++idx) {
        if (!prefix_matches(sequence, idx, current_prefix_start, ngram_size)) {
            continue;
        }
        const int32_t token_id = sequence[idx + ngram_size - 1u];
        if (!valid_token_id(token_id, vocab_size)) {
            return UOCR_ERROR_INVALID_ARGUMENT;
        }
        if (is_whitelisted(token_id, whitelist_token_ids, whitelist_count)) {
            continue;
        }
        if (token_was_banned_earlier(sequence, search_start, idx, current_prefix_start, ngram_size, token_id)) {
            continue;
        }
        if (required < out_capacity) {
            out_banned[required] = token_id;
        } else {
            overflowed = 1;
        }
        ++required;
    }

    *out_count = required;
    return overflowed ? UOCR_ERROR_OUT_OF_MEMORY : UOCR_OK;
}

int uocr_no_repeat_ngram_apply(float *logits,
                               uint32_t vocab_size,
                               const int32_t *sequence,
                               uint32_t sequence_len,
                               uint32_t ngram_size,
                               uint32_t window,
                               const int32_t *whitelist_token_ids,
                               uint32_t whitelist_count) {
    if (logits == NULL || vocab_size == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (ngram_size == 0u) {
        return UOCR_OK;
    }
    if (sequence == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (!validate_whitelist(whitelist_token_ids, whitelist_count, vocab_size)) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint32_t search_start = 0u;
    uint32_t search_end = 0u;
    uint32_t current_prefix_start = 0u;
    if (!no_repeat_bounds(sequence_len, ngram_size, window, &search_start, &search_end, &current_prefix_start)) {
        return UOCR_OK;
    }

    for (uint32_t idx = search_start; idx < search_end; ++idx) {
        if (!prefix_matches(sequence, idx, current_prefix_start, ngram_size)) {
            continue;
        }
        const int32_t token_id = sequence[idx + ngram_size - 1u];
        if (!valid_token_id(token_id, vocab_size)) {
            return UOCR_ERROR_INVALID_ARGUMENT;
        }
        if (is_whitelisted(token_id, whitelist_token_ids, whitelist_count)) {
            continue;
        }
        logits[token_id] = -INFINITY;
    }
    return UOCR_OK;
}
