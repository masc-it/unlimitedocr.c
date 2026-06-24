#include "runtime/uocr_sequence.h"

#include <stdint.h>
#include <string.h>

#define CHECK(expr)                    \
    do {                               \
        if (!(expr)) {                 \
            return __LINE__;           \
        }                              \
    } while (0)

static int test_text_only_sequence_state(void) {
    const int32_t input_ids[] = {0, 42, 43};
    const uint8_t image_mask[] = {0, 0, 0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 3u;
    request.max_new_tokens = 5u;
    request.no_repeat_ngram_size = 2u;
    request.no_repeat_window = 16u;

    char error[128];
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(state.prompt_token_count == 3u);
    CHECK(state.image_span_start == UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(state.image_span_length == 0u);
    CHECK(state.text_prefix_length == 3u);
    CHECK(state.text_suffix_start == 3u);
    CHECK(state.text_suffix_length == 0u);
    CHECK(state.max_new_tokens == 5u);
    CHECK(state.no_repeat_ngram_size == 2u);
    CHECK(state.no_repeat_window == 16u);
    CHECK(state.generated_count == 0u);
    CHECK(state.position == 3u);
    CHECK(state.eos == 0);
    return 0;
}

static int test_single_image_span_sequence_state(void) {
    const int32_t input_ids[] = {0, 11, 128815, 128815, 12, 13};
    const uint8_t image_mask[] = {0, 0, 1, 1, 0, 0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 6u;
    request.max_new_tokens = 7u;

    char error[128];
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    CHECK(state.prompt_token_count == 6u);
    CHECK(state.image_span_start == 2u);
    CHECK(state.image_span_length == 2u);
    CHECK(state.text_prefix_length == 2u);
    CHECK(state.text_suffix_start == 4u);
    CHECK(state.text_suffix_length == 2u);
    CHECK(state.position == 6u);
    return 0;
}

static int test_rejects_discontiguous_image_placeholders(void) {
    const int32_t input_ids[] = {0, 128815, 7, 128815};
    const uint8_t image_mask[] = {0, 1, 0, 1};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 4u;

    char error[128];
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(strstr(error, "contiguous") != NULL);
    return 0;
}

static int test_generation_stops_on_eos(void) {
    const int32_t input_ids[] = {0, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;
    request.max_new_tokens = 4u;

    uocr_sequence_state state;
    char error[128];
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    CHECK(uocr_sequence_generation_done(&state) == 0);

    int32_t generated[4] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, 77, generated, 4u) == UOCR_OK);
    CHECK(generated[0] == 77);
    CHECK(state.generated_count == 1u);
    CHECK(state.position == 3u);
    CHECK(state.eos == 0);
    CHECK(uocr_sequence_generation_done(&state) == 0);

    CHECK(uocr_sequence_accept_generated_token(&state, 1, generated, 4u) == UOCR_OK);
    CHECK(generated[1] == 1);
    CHECK(state.generated_count == 2u);
    CHECK(state.position == 4u);
    CHECK(state.eos == 1);
    CHECK(uocr_sequence_generation_done(&state) == 1);
    CHECK(uocr_sequence_accept_generated_token(&state, 88, generated, 4u) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(state.generated_count == 2u);
    return 0;
}

static int test_generation_stops_on_max_new_tokens(void) {
    const int32_t input_ids[] = {0, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;
    request.max_new_tokens = 2u;

    uocr_sequence_state state;
    char error[128];
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);

    int32_t generated[2] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, 5, generated, 2u) == UOCR_OK);
    CHECK(uocr_sequence_generation_done(&state) == 0);
    CHECK(uocr_sequence_accept_generated_token(&state, 6, generated, 2u) == UOCR_OK);
    CHECK(generated[0] == 5);
    CHECK(generated[1] == 6);
    CHECK(state.generated_count == 2u);
    CHECK(state.position == 4u);
    CHECK(state.eos == 0);
    CHECK(uocr_sequence_generation_done(&state) == 1);
    CHECK(uocr_sequence_accept_generated_token(&state, 7, generated, 2u) == UOCR_ERROR_INVALID_ARGUMENT);
    return 0;
}

static int test_generation_zero_budget_is_done(void) {
    const int32_t input_ids[] = {0};
    const uint8_t image_mask[] = {0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 1u;
    request.max_new_tokens = 0u;

    uocr_sequence_state state;
    char error[128];
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    CHECK(uocr_sequence_generation_done(&state) == 1);
    int32_t generated[1] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, 5, generated, 1u) == UOCR_ERROR_INVALID_ARGUMENT);
    return 0;
}

static int test_generation_rejects_bad_token_and_short_buffer(void) {
    const int32_t input_ids[] = {0, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;
    request.max_new_tokens = 2u;

    uocr_sequence_state state;
    char error[128];
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);

    int32_t generated[1] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, -1, generated, 1u) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(state.generated_count == 0u);
    CHECK(uocr_sequence_accept_generated_token(&state, 129280, generated, 1u) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(state.generated_count == 0u);
    CHECK(uocr_sequence_accept_generated_token(&state, 7, NULL, 1u) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(state.generated_count == 0u);

    CHECK(uocr_sequence_accept_generated_token(&state, 7, generated, 1u) == UOCR_OK);
    CHECK(state.generated_count == 1u);
    CHECK(uocr_sequence_accept_generated_token(&state, 8, generated, 1u) == UOCR_ERROR_OUT_OF_MEMORY);
    CHECK(state.generated_count == 1u);
    CHECK(state.position == 3u);
    return 0;
}

static int test_rejects_bad_mask_value(void) {
    const int32_t input_ids[] = {0, 128815};
    const uint8_t image_mask[] = {0, 2};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;

    char error[128];
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(strstr(error, "image_mask") != NULL);
    return 0;
}

int main(void) {
    int status = 0;
    if ((status = test_text_only_sequence_state()) != 0) return status;
    if ((status = test_single_image_span_sequence_state()) != 0) return status;
    if ((status = test_rejects_discontiguous_image_placeholders()) != 0) return status;
    if ((status = test_generation_stops_on_eos()) != 0) return status;
    if ((status = test_generation_stops_on_max_new_tokens()) != 0) return status;
    if ((status = test_generation_zero_budget_is_done()) != 0) return status;
    if ((status = test_generation_rejects_bad_token_and_short_buffer()) != 0) return status;
    if ((status = test_rejects_bad_mask_value()) != 0) return status;
    return 0;
}
