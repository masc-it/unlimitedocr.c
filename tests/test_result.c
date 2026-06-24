#include "core/uocr_result.h"
#include "unlimitedocr.h"

#include <stdint.h>

#define CHECK(expr)                    \
    do {                               \
        if (!(expr)) {                 \
            return __LINE__;           \
        }                              \
    } while (0)

static int test_empty_result_shape(void) {
    uocr_result *result = NULL;
    CHECK(uocr_result_create_empty(2u, &result) == UOCR_OK);
    CHECK(result != NULL);
    CHECK(uocr_result_count(result) == 2u);

    for (uint32_t i = 0u; i < 2u; ++i) {
        uint32_t n_tokens = 123u;
        const int32_t *tokens = uocr_result_tokens(result, i, &n_tokens);
        CHECK(tokens == NULL);
        CHECK(n_tokens == 0u);
    }

    uint32_t n_tokens = 123u;
    CHECK(uocr_result_tokens(result, 2u, &n_tokens) == NULL);
    CHECK(n_tokens == 0u);
    uocr_result_free(result);
    return 0;
}

static int test_generated_result_copies_tokens(void) {
    int32_t seq0[] = {17, 42, 1};
    int32_t seq1[] = {128815, 2};
    const int32_t *tokens[] = {seq0, seq1, NULL};
    const uint32_t counts[] = {3u, 2u, 0u};

    uocr_result *result = NULL;
    CHECK(uocr_result_create_from_generated(3u, tokens, counts, &result) == UOCR_OK);
    CHECK(result != NULL);
    CHECK(uocr_result_count(result) == 3u);

    seq0[0] = 999;
    seq1[1] = 999;

    uint32_t n_tokens = 0u;
    const int32_t *out0 = uocr_result_tokens(result, 0u, &n_tokens);
    CHECK(out0 != NULL);
    CHECK(n_tokens == 3u);
    CHECK(out0[0] == 17);
    CHECK(out0[1] == 42);
    CHECK(out0[2] == 1);

    const int32_t *out1 = uocr_result_tokens(result, 1u, &n_tokens);
    CHECK(out1 != NULL);
    CHECK(n_tokens == 2u);
    CHECK(out1[0] == 128815);
    CHECK(out1[1] == 2);

    const int32_t *out2 = uocr_result_tokens(result, 2u, &n_tokens);
    CHECK(out2 == NULL);
    CHECK(n_tokens == 0u);

    uocr_result_free(result);
    return 0;
}

static int test_generated_result_rejects_invalid_inputs(void) {
    uocr_result *result = (uocr_result *)(uintptr_t)1u;
    const uint32_t counts_one[] = {1u};
    const uint32_t counts_zero[] = {0u};
    const int32_t *tokens_null[] = {NULL};

    CHECK(uocr_result_create_empty(0u, &result) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == (uocr_result *)(uintptr_t)1u);
    CHECK(uocr_result_create_empty(1u, NULL) == UOCR_ERROR_INVALID_ARGUMENT);

    result = (uocr_result *)(uintptr_t)1u;
    CHECK(uocr_result_create_from_generated(0u, NULL, counts_zero, &result) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == (uocr_result *)(uintptr_t)1u);
    CHECK(uocr_result_create_from_generated(1u, NULL, NULL, &result) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == (uocr_result *)(uintptr_t)1u);
    CHECK(uocr_result_create_from_generated(1u, NULL, counts_one, &result) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == NULL);
    result = (uocr_result *)(uintptr_t)1u;
    CHECK(uocr_result_create_from_generated(1u, tokens_null, counts_one, &result) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == NULL);

    CHECK(uocr_result_count(NULL) == 0u);
    uint32_t n_tokens = 123u;
    CHECK(uocr_result_tokens(NULL, 0u, &n_tokens) == NULL);
    CHECK(n_tokens == 0u);
    return 0;
}

int main(void) {
    int status = 0;
    if ((status = test_empty_result_shape()) != 0) return status;
    if ((status = test_generated_result_copies_tokens()) != 0) return status;
    if ((status = test_generated_result_rejects_invalid_inputs()) != 0) return status;
    return 0;
}
