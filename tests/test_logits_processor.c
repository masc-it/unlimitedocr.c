#include "runtime/uocr_logits_processor.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(expr)                    \
    do {                               \
        if (!(expr)) {                 \
            return __LINE__;           \
        }                              \
    } while (0)

static int contains_token(const int32_t *tokens, uint32_t count, int32_t token) {
    for (uint32_t i = 0u; i < count; ++i) {
        if (tokens[i] == token) {
            return 1;
        }
    }
    return 0;
}

static int test_collect_matches_upstream_prefix_rule(void) {
    const int32_t sequence[] = {1, 2, 3, 1, 2};
    int32_t banned[4] = {0};
    uint32_t count = 0u;
    CHECK(uocr_no_repeat_ngram_collect_banned(sequence,
                                              5u,
                                              16u,
                                              3u,
                                              0u,
                                              NULL,
                                              0u,
                                              banned,
                                              4u,
                                              &count) == UOCR_OK);
    CHECK(count == 1u);
    CHECK(banned[0] == 3);
    return 0;
}

static int test_window_limits_search_start(void) {
    const int32_t sequence[] = {1, 2, 3, 9, 1, 2};
    int32_t banned[4] = {0};
    uint32_t count = 99u;

    CHECK(uocr_no_repeat_ngram_collect_banned(sequence,
                                              6u,
                                              16u,
                                              3u,
                                              3u,
                                              NULL,
                                              0u,
                                              banned,
                                              4u,
                                              &count) == UOCR_OK);
    CHECK(count == 0u);

    CHECK(uocr_no_repeat_ngram_collect_banned(sequence,
                                              6u,
                                              16u,
                                              3u,
                                              6u,
                                              NULL,
                                              0u,
                                              banned,
                                              4u,
                                              &count) == UOCR_OK);
    CHECK(count == 1u);
    CHECK(banned[0] == 3);
    return 0;
}

static int test_unigram_and_whitelist(void) {
    const int32_t sequence[] = {4, 5, 4};
    const int32_t whitelist[] = {4};
    int32_t banned[4] = {0};
    uint32_t count = 0u;

    CHECK(uocr_no_repeat_ngram_collect_banned(sequence,
                                              3u,
                                              16u,
                                              1u,
                                              2u,
                                              whitelist,
                                              1u,
                                              banned,
                                              4u,
                                              &count) == UOCR_OK);
    CHECK(count == 1u);
    CHECK(banned[0] == 5);
    return 0;
}

static int test_collect_deduplicates_and_reports_capacity(void) {
    const int32_t sequence[] = {7, 8, 9, 7, 8, 9, 7, 8};
    int32_t banned[1] = {0};
    uint32_t count = 0u;

    CHECK(uocr_no_repeat_ngram_collect_banned(sequence,
                                              8u,
                                              16u,
                                              3u,
                                              0u,
                                              NULL,
                                              0u,
                                              banned,
                                              1u,
                                              &count) == UOCR_OK);
    CHECK(count == 1u);
    CHECK(banned[0] == 9);

    const int32_t sequence_two_bans[] = {1, 2, 3, 1, 2, 4, 1, 2};
    count = 0u;
    banned[0] = 0;
    CHECK(uocr_no_repeat_ngram_collect_banned(sequence_two_bans,
                                              8u,
                                              16u,
                                              3u,
                                              0u,
                                              NULL,
                                              0u,
                                              banned,
                                              1u,
                                              &count) == UOCR_ERROR_OUT_OF_MEMORY);
    CHECK(count == 2u);
    CHECK(contains_token(banned, 1u, 3) || contains_token(banned, 1u, 4));
    return 0;
}

static int test_apply_sets_negative_infinity(void) {
    const int32_t sequence[] = {1, 2, 3, 1, 2};
    float logits[8];
    for (uint32_t i = 0u; i < 8u; ++i) {
        logits[i] = (float)i;
    }

    CHECK(uocr_no_repeat_ngram_apply(logits, 8u, sequence, 5u, 3u, 0u, NULL, 0u) == UOCR_OK);
    CHECK(isinf(logits[3]) && logits[3] < 0.0f);
    CHECK(logits[2] == 2.0f);
    CHECK(logits[4] == 4.0f);
    return 0;
}

static int test_disabled_and_invalid_inputs(void) {
    const int32_t sequence[] = {1, 2, 3};
    int32_t banned[2] = {123, 456};
    uint32_t count = 999u;

    CHECK(uocr_no_repeat_ngram_collect_banned(sequence, 3u, 8u, 0u, 0u, NULL, 0u, banned, 2u, &count) == UOCR_OK);
    CHECK(count == 0u);
    CHECK(banned[0] == 123);
    CHECK(uocr_no_repeat_ngram_collect_banned(NULL, 3u, 8u, 2u, 0u, NULL, 0u, banned, 2u, &count) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(uocr_no_repeat_ngram_collect_banned(sequence, 3u, 8u, 2u, 0u, NULL, 0u, NULL, 2u, &count) == UOCR_ERROR_INVALID_ARGUMENT);

    const int32_t bad_sequence[] = {1, 2, 99, 1, 2};
    CHECK(uocr_no_repeat_ngram_collect_banned(bad_sequence, 5u, 8u, 3u, 0u, NULL, 0u, banned, 2u, &count) == UOCR_ERROR_INVALID_ARGUMENT);
    return 0;
}

int main(void) {
    int status = 0;
    if ((status = test_collect_matches_upstream_prefix_rule()) != 0) return status;
    if ((status = test_window_limits_search_start()) != 0) return status;
    if ((status = test_unigram_and_whitelist()) != 0) return status;
    if ((status = test_collect_deduplicates_and_reports_capacity()) != 0) return status;
    if ((status = test_apply_sets_negative_infinity()) != 0) return status;
    if ((status = test_disabled_and_invalid_inputs()) != 0) return status;
    return 0;
}
