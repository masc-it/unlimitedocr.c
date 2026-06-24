#include "core/uocr_result.h"

#include "core/uocr_alloc.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct uocr_result {
    uint32_t n_sequences;
    int32_t **tokens;
    uint32_t *n_tokens;
};

static int checked_token_bytes(uint32_t n_tokens, size_t *out_bytes) {
    if (out_bytes == NULL) {
        return 0;
    }
    if ((size_t)n_tokens > (size_t)-1 / sizeof(int32_t)) {
        return 0;
    }
    *out_bytes = (size_t)n_tokens * sizeof(int32_t);
    return 1;
}

int uocr_result_create_empty(uint32_t n_sequences, uocr_result **out_result) {
    if (out_result == NULL || n_sequences == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    *out_result = NULL;

    uocr_result *result = (uocr_result *)uocr_calloc(1u, sizeof(*result));
    if (result == NULL) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    result->n_sequences = n_sequences;
    result->tokens = (int32_t **)uocr_calloc(n_sequences, sizeof(result->tokens[0]));
    result->n_tokens = (uint32_t *)uocr_calloc(n_sequences, sizeof(result->n_tokens[0]));
    if (result->tokens == NULL || result->n_tokens == NULL) {
        uocr_result_free(result);
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    *out_result = result;
    return UOCR_OK;
}

int uocr_result_create_from_generated(uint32_t n_sequences,
                                      const int32_t *const *generated_tokens,
                                      const uint32_t *generated_counts,
                                      uocr_result **out_result) {
    if (out_result == NULL || generated_counts == NULL || n_sequences == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    *out_result = NULL;

    for (uint32_t i = 0u; i < n_sequences; ++i) {
        if (generated_counts[i] != 0u && (generated_tokens == NULL || generated_tokens[i] == NULL)) {
            return UOCR_ERROR_INVALID_ARGUMENT;
        }
    }

    uocr_result *result = NULL;
    int status = uocr_result_create_empty(n_sequences, &result);
    if (status != UOCR_OK) {
        return status;
    }

    for (uint32_t i = 0u; i < n_sequences; ++i) {
        const uint32_t count = generated_counts[i];
        result->n_tokens[i] = count;
        if (count == 0u) {
            continue;
        }

        size_t bytes = 0u;
        if (!checked_token_bytes(count, &bytes)) {
            uocr_result_free(result);
            return UOCR_ERROR_OUT_OF_MEMORY;
        }
        result->tokens[i] = (int32_t *)uocr_malloc(bytes);
        if (result->tokens[i] == NULL) {
            uocr_result_free(result);
            return UOCR_ERROR_OUT_OF_MEMORY;
        }
        memcpy(result->tokens[i], generated_tokens[i], bytes);
    }

    *out_result = result;
    return UOCR_OK;
}

uint32_t uocr_result_count(const uocr_result *result) {
    return result != NULL ? result->n_sequences : 0u;
}

const int32_t *uocr_result_tokens(const uocr_result *result, uint32_t index, uint32_t *n_tokens) {
    if (n_tokens != NULL) {
        *n_tokens = 0u;
    }
    if (result == NULL || index >= result->n_sequences) {
        return NULL;
    }
    if (n_tokens != NULL) {
        *n_tokens = result->n_tokens[index];
    }
    return result->tokens[index];
}

void uocr_result_free(uocr_result *result) {
    if (result == NULL) {
        return;
    }
    if (result->tokens != NULL) {
        for (uint32_t i = 0u; i < result->n_sequences; ++i) {
            uocr_free(result->tokens[i]);
        }
    }
    uocr_free(result->tokens);
    uocr_free(result->n_tokens);
    uocr_free(result);
}
