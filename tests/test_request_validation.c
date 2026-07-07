#include "runtime/uocr_request_validation.h"

#include "model/uocr_constants.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

typedef struct owned_request {
    uocr_prepared_request request;
    int32_t *input_ids;
    uint8_t *image_mask;
    uocr_image_view *views;
} owned_request;

static const float pixel_sentinel = 0.0f;

static void free_owned_request(owned_request *owned) {
    if (owned == NULL) {
        return;
    }
    free(owned->views);
    free(owned->image_mask);
    free(owned->input_ids);
    memset(owned, 0, sizeof(*owned));
}

static int make_text_request(uint32_t n_tokens, owned_request *out) {
    if (out == NULL || n_tokens == 0u) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    out->input_ids = (int32_t *)calloc((size_t)n_tokens, sizeof(*out->input_ids));
    out->image_mask = (uint8_t *)calloc((size_t)n_tokens, sizeof(*out->image_mask));
    if (out->input_ids == NULL || out->image_mask == NULL) {
        free_owned_request(out);
        return 0;
    }
    out->input_ids[0] = UOCR_TOKEN_BOS;
    for (uint32_t i = 1u; i < n_tokens; ++i) {
        out->input_ids[i] = 42;
    }
    out->request.input_ids = out->input_ids;
    out->request.image_mask = out->image_mask;
    out->request.n_tokens = n_tokens;
    out->request.crop_grid_w = 1u;
    out->request.crop_grid_h = 1u;
    return 1;
}

static int make_image_request(uint32_t visual_tokens,
                              uint32_t n_views,
                              uint32_t crop_grid_w,
                              uint32_t crop_grid_h,
                              owned_request *out) {
    if (!make_text_request(1u + visual_tokens, out)) {
        return 0;
    }
    out->views = n_views != 0u ? (uocr_image_view *)calloc((size_t)n_views, sizeof(*out->views)) : NULL;
    if (n_views != 0u && out->views == NULL) {
        free_owned_request(out);
        return 0;
    }
    for (uint32_t i = 0u; i < visual_tokens; ++i) {
        out->input_ids[1u + i] = UOCR_TOKEN_IMAGE;
        out->image_mask[1u + i] = 1u;
    }
    out->request.views = out->views;
    out->request.n_views = n_views;
    out->request.crop_grid_w = crop_grid_w;
    out->request.crop_grid_h = crop_grid_h;
    return 1;
}

static void fill_global_view(uocr_image_view *view) {
    memset(view, 0, sizeof(*view));
    view->pixels = &pixel_sentinel;
    view->width = UOCR_GLOBAL_VIEW_SIZE;
    view->height = UOCR_GLOBAL_VIEW_SIZE;
    view->format = UOCR_PIXEL_F32_NCHW;
    view->kind = UOCR_VIEW_GLOBAL;
}

static void fill_local_view(uocr_image_view *view) {
    memset(view, 0, sizeof(*view));
    view->pixels = &pixel_sentinel;
    view->width = UOCR_LOCAL_VIEW_SIZE;
    view->height = UOCR_LOCAL_VIEW_SIZE;
    view->format = UOCR_PIXEL_F16_NCHW;
    view->kind = UOCR_VIEW_LOCAL;
}

static uocr_request_limits default_limits(void) {
    uocr_request_limits limits;
    memset(&limits, 0, sizeof(limits));
    limits.max_prompt_tokens = UOCR_DEFAULT_MAX_PROMPT_TOKENS;
    limits.max_gen_tokens = UOCR_DEFAULT_MAX_GEN_TOKENS;
    limits.max_position_tokens = UOCR_MAX_POSITIONS;
    limits.generated_ring_window = UOCR_GENERATED_RING_WINDOW;
    return limits;
}

static int expect_validation_ok(const uocr_prepared_request *request) {
    char error[512];
    const uocr_request_limits limits = default_limits();
    const int status = uocr_validate_prepared_request(request, &limits, error, sizeof(error));
    if (status != UOCR_OK) {
        fprintf(stderr, "expected request to validate, got status %d: %s\n", status, error);
        return 1;
    }
    return 0;
}

static int expect_validation_error(const uocr_prepared_request *request,
                                   const uocr_request_limits *limits,
                                   const char *needle) {
    char error[512];
    const int status = uocr_validate_prepared_request(request, limits, error, sizeof(error));
    if (status != UOCR_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "expected invalid argument, got status %d with error: %s\n", status, error);
        return 1;
    }
    if (needle != NULL && strstr(error, needle) == NULL) {
        fprintf(stderr, "expected error containing %s, got: %s\n", needle, error);
        return 1;
    }
    return 0;
}

static int test_text_request_validation_contract(void) {
    owned_request owned;
    CHECK(make_text_request(3u, &owned));
    owned.request.max_new_tokens = 4u;

    CHECK(expect_validation_ok(&owned.request) == 0);
    CHECK(uocr_count_image_placeholders(&owned.request) == 0u);

    free_owned_request(&owned);
    return 0;
}

static int test_rejects_null_and_empty_requests(void) {
    const uocr_request_limits limits = default_limits();
    CHECK(expect_validation_error(NULL, &limits, "prepared request is null") == 0);

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    CHECK(expect_validation_error(&request, &limits, "at least BOS") == 0);

    request.n_tokens = 1u;
    CHECK(expect_validation_error(&request, &limits, "input_ids pointer is null") == 0);

    const int32_t input_ids[] = {UOCR_TOKEN_BOS};
    request.input_ids = input_ids;
    CHECK(expect_validation_error(&request, &limits, "image_mask pointer is null") == 0);
    CHECK(expect_validation_error(&request, NULL, "request limits are null") == 0);
    return 0;
}

static int test_rejects_token_and_mask_contract_violations(void) {
    owned_request owned;
    const uocr_request_limits limits = default_limits();
    CHECK(make_text_request(3u, &owned));

    owned.input_ids[1] = -1;
    CHECK(expect_validation_error(&owned.request, &limits, "out of vocabulary") == 0);

    owned.input_ids[1] = (int32_t)UOCR_VOCAB_SIZE;
    CHECK(expect_validation_error(&owned.request, &limits, "out of vocabulary") == 0);

    owned.input_ids[1] = UOCR_TOKEN_IMAGE;
    owned.image_mask[1] = 0u;
    CHECK(expect_validation_error(&owned.request, &limits, "image token") == 0);

    owned.image_mask[1] = 2u;
    CHECK(uocr_count_image_placeholders(&owned.request) == 0u);
    CHECK(expect_validation_error(&owned.request, &limits, "must be 0 or 1") == 0);

    owned.input_ids[1] = 42;
    owned.image_mask[1] = 1u;
    CHECK(uocr_count_image_placeholders(&owned.request) == 1u);
    CHECK(expect_validation_error(&owned.request, &limits, "expected image token") == 0);

    free_owned_request(&owned);
    return 0;
}

static int test_rejects_discontiguous_image_span(void) {
    owned_request owned;
    const uocr_request_limits limits = default_limits();
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS + 1u, 1u, 1u, 1u, &owned));
    fill_global_view(&owned.views[0]);

    owned.input_ids[UOCR_GLOBAL_VISUAL_TOKENS] = 42;
    owned.image_mask[UOCR_GLOBAL_VISUAL_TOKENS] = 0u;
    CHECK(uocr_count_image_placeholders(&owned.request) == UOCR_GLOBAL_VISUAL_TOKENS);
    CHECK(expect_validation_error(&owned.request, &limits, "one contiguous span") == 0);

    free_owned_request(&owned);
    return 0;
}

static int test_global_and_multipage_visual_contracts(void) {
    owned_request base;
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS, 1u, 1u, 1u, &base));
    fill_global_view(&base.views[0]);
    CHECK(expect_validation_ok(&base.request) == 0);
    CHECK(uocr_count_image_placeholders(&base.request) == UOCR_GLOBAL_VISUAL_TOKENS);
    free_owned_request(&base);

    owned_request pages;
    CHECK(make_image_request(2u * UOCR_GLOBAL_VISUAL_TOKENS, 2u, 1u, 1u, &pages));
    fill_global_view(&pages.views[0]);
    fill_global_view(&pages.views[1]);
    CHECK(expect_validation_ok(&pages.request) == 0);
    free_owned_request(&pages);

    const uocr_request_limits limits = default_limits();
    owned_request bad_count;
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS - 1u, 1u, 1u, 1u, &bad_count));
    fill_global_view(&bad_count.views[0]);
    CHECK(expect_validation_error(&bad_count.request, &limits, "placeholder count mismatch") == 0);
    free_owned_request(&bad_count);

    owned_request bad_grid;
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS, 1u, 2u, 1u, &bad_grid));
    fill_global_view(&bad_grid.views[0]);
    CHECK(expect_validation_error(&bad_grid.request, &limits, "global-only requests") == 0);
    free_owned_request(&bad_grid);

    owned_request bad_shape;
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS, 1u, 1u, 1u, &bad_shape));
    fill_global_view(&bad_shape.views[0]);
    bad_shape.views[0].width = UOCR_LOCAL_VIEW_SIZE;
    CHECK(expect_validation_error(&bad_shape.request, &limits, "global view 0") == 0);
    free_owned_request(&bad_shape);

    owned_request bad_format;
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS, 1u, 1u, 1u, &bad_format));
    fill_global_view(&bad_format.views[0]);
    bad_format.views[0].format = (uocr_pixel_format)99;
    CHECK(expect_validation_error(&bad_format.request, &limits, "unsupported pixel format") == 0);
    free_owned_request(&bad_format);

    owned_request null_pixels;
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS, 1u, 1u, 1u, &null_pixels));
    fill_global_view(&null_pixels.views[0]);
    null_pixels.views[0].pixels = NULL;
    CHECK(expect_validation_error(&null_pixels.request, &limits, "null pixel pointer") == 0);
    free_owned_request(&null_pixels);
    return 0;
}

static int test_crop_mode_visual_contracts(void) {
    const uint32_t crop_visual_tokens = uocr_local_visual_token_count(2u, 1u) + UOCR_GLOBAL_VISUAL_TOKENS;
    owned_request crop;
    CHECK(make_image_request(crop_visual_tokens, 3u, 2u, 1u, &crop));
    fill_local_view(&crop.views[0]);
    fill_local_view(&crop.views[1]);
    fill_global_view(&crop.views[2]);
    CHECK(expect_validation_ok(&crop.request) == 0);
    free_owned_request(&crop);

    const uocr_request_limits limits = default_limits();
    owned_request bad_order;
    CHECK(make_image_request(crop_visual_tokens, 3u, 2u, 1u, &bad_order));
    fill_global_view(&bad_order.views[0]);
    fill_local_view(&bad_order.views[1]);
    fill_local_view(&bad_order.views[2]);
    CHECK(expect_validation_error(&bad_order.request, &limits, "local views must come first") == 0);
    free_owned_request(&bad_order);

    owned_request bad_local_count;
    CHECK(make_image_request(crop_visual_tokens, 2u, 2u, 1u, &bad_local_count));
    fill_local_view(&bad_local_count.views[0]);
    fill_global_view(&bad_local_count.views[1]);
    CHECK(expect_validation_error(&bad_local_count.request, &limits, "expects 2 local views") == 0);
    free_owned_request(&bad_local_count);

    owned_request bad_single_crop;
    CHECK(make_image_request(UOCR_GLOBAL_VISUAL_TOKENS + uocr_local_visual_token_count(1u, 1u), 2u, 1u, 1u, &bad_single_crop));
    fill_local_view(&bad_single_crop.views[0]);
    fill_global_view(&bad_single_crop.views[1]);
    CHECK(expect_validation_error(&bad_single_crop.request, &limits, "should be represented as global-only") == 0);
    free_owned_request(&bad_single_crop);

    owned_request bad_global_count;
    CHECK(make_image_request(crop_visual_tokens, 3u, 2u, 1u, &bad_global_count));
    fill_local_view(&bad_global_count.views[0]);
    fill_global_view(&bad_global_count.views[1]);
    fill_global_view(&bad_global_count.views[2]);
    CHECK(expect_validation_error(&bad_global_count.request, &limits, "exactly one global") == 0);
    free_owned_request(&bad_global_count);

    owned_request bad_local_shape;
    CHECK(make_image_request(crop_visual_tokens, 3u, 2u, 1u, &bad_local_shape));
    fill_local_view(&bad_local_shape.views[0]);
    fill_local_view(&bad_local_shape.views[1]);
    fill_global_view(&bad_local_shape.views[2]);
    bad_local_shape.views[0].height = UOCR_GLOBAL_VIEW_SIZE;
    CHECK(expect_validation_error(&bad_local_shape.request, &limits, "local view 0") == 0);
    free_owned_request(&bad_local_shape);
    return 0;
}

static int test_budget_validation(void) {
    owned_request owned;
    CHECK(make_text_request(4u, &owned));
    owned.request.max_new_tokens = 4u;

    uocr_request_limits limits = default_limits();
    limits.max_prompt_tokens = 3u;
    CHECK(expect_validation_error(&owned.request, &limits, "engine limit") == 0);

    limits = default_limits();
    limits.max_gen_tokens = 3u;
    CHECK(expect_validation_error(&owned.request, &limits, "new tokens") == 0);

    limits = default_limits();
    limits.max_position_tokens = 7u;
    CHECK(expect_validation_error(&owned.request, &limits, "position budget") == 0);


    free_owned_request(&owned);
    return 0;
}

int main(void) {
    if (test_text_request_validation_contract() != 0) return 1;
    if (test_rejects_null_and_empty_requests() != 0) return 1;
    if (test_rejects_token_and_mask_contract_violations() != 0) return 1;
    if (test_rejects_discontiguous_image_span() != 0) return 1;
    if (test_global_and_multipage_visual_contracts() != 0) return 1;
    if (test_crop_mode_visual_contracts() != 0) return 1;
    if (test_budget_validation() != 0) return 1;
    return 0;
}
