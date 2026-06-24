#include "runtime/uocr_vision.h"

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

static int make_request(uint32_t visual_tokens,
                        uint32_t n_views,
                        uint32_t crop_grid_w,
                        uint32_t crop_grid_h,
                        owned_request *out) {
    if (out == NULL) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    const uint32_t n_tokens = 1u + visual_tokens;
    out->input_ids = (int32_t *)calloc((size_t)n_tokens, sizeof(int32_t));
    out->image_mask = (uint8_t *)calloc((size_t)n_tokens, sizeof(uint8_t));
    out->views = n_views != 0u ? (uocr_image_view *)calloc((size_t)n_views, sizeof(uocr_image_view)) : NULL;
    if (out->input_ids == NULL || out->image_mask == NULL || (n_views != 0u && out->views == NULL)) {
        free_owned_request(out);
        return 0;
    }
    out->input_ids[0] = UOCR_TOKEN_BOS;
    for (uint32_t i = 1u; i < n_tokens; ++i) {
        out->input_ids[i] = UOCR_TOKEN_IMAGE;
        out->image_mask[i] = 1u;
    }
    out->request.input_ids = out->input_ids;
    out->request.image_mask = out->image_mask;
    out->request.n_tokens = n_tokens;
    out->request.views = out->views;
    out->request.n_views = n_views;
    out->request.crop_grid_w = crop_grid_w;
    out->request.crop_grid_h = crop_grid_h;
    return 1;
}

static void fill_global_view(uocr_image_view *view) {
    view->pixels = &pixel_sentinel;
    view->width = UOCR_GLOBAL_VIEW_SIZE;
    view->height = UOCR_GLOBAL_VIEW_SIZE;
    view->format = UOCR_PIXEL_F32_NCHW;
    view->kind = UOCR_VIEW_GLOBAL;
}

static void fill_local_view(uocr_image_view *view) {
    view->pixels = &pixel_sentinel;
    view->width = UOCR_LOCAL_VIEW_SIZE;
    view->height = UOCR_LOCAL_VIEW_SIZE;
    view->format = UOCR_PIXEL_F16_NCHW;
    view->kind = UOCR_VIEW_LOCAL;
}

static int test_global_views_are_chunked_in_page_order(void) {
    owned_request owned;
    CHECK(make_request(3u * UOCR_GLOBAL_VISUAL_TOKENS, 3u, 1u, 1u, &owned));
    for (uint32_t i = 0u; i < 3u; ++i) {
        fill_global_view(&owned.views[i]);
    }

    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    uocr_vision_chunk chunks[2];
    memset(chunks, 0, sizeof(chunks));
    CHECK(uocr_plan_vision_schedule(&owned.request, 2u, chunks, 2u, &schedule, error, sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(schedule.chunk_count == 2u);
    CHECK(schedule.local_view_count == 0u);
    CHECK(schedule.global_view_count == 3u);
    CHECK(schedule.max_views_per_chunk == 2u);
    CHECK(schedule.max_chunk_views == 2u);
    CHECK(schedule.final_visual_tokens == 3u * UOCR_GLOBAL_VISUAL_TOKENS);
    CHECK(schedule.projected_tokens_total == 3u * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
    CHECK(chunks[0].kind == UOCR_VISION_CHUNK_GLOBAL);
    CHECK(chunks[0].first_view == 0u);
    CHECK(chunks[0].view_count == 2u);
    CHECK(chunks[0].projected_grid_w == UOCR_GLOBAL_GRID_QUERIES);
    CHECK(chunks[0].projected_token_start == 0u);
    CHECK(chunks[0].projected_token_count == 2u * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
    CHECK(chunks[1].kind == UOCR_VISION_CHUNK_GLOBAL);
    CHECK(chunks[1].first_view == 2u);
    CHECK(chunks[1].view_count == 1u);
    CHECK(chunks[1].projected_token_start == chunks[0].projected_token_count);

    free_owned_request(&owned);
    return 0;
}

static int test_crop_local_views_are_chunked_before_global(void) {
    enum { GRID_W = 3u, GRID_H = 2u, LOCAL_VIEWS = GRID_W * GRID_H };
    const uint32_t visual_tokens = uocr_local_visual_token_count(GRID_W, GRID_H) + UOCR_GLOBAL_VISUAL_TOKENS;
    owned_request owned;
    CHECK(make_request(visual_tokens, LOCAL_VIEWS + 1u, GRID_W, GRID_H, &owned));
    for (uint32_t i = 0u; i < LOCAL_VIEWS; ++i) {
        fill_local_view(&owned.views[i]);
    }
    fill_global_view(&owned.views[LOCAL_VIEWS]);

    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    uocr_vision_chunk chunks[4];
    memset(chunks, 0, sizeof(chunks));
    CHECK(uocr_plan_vision_schedule(&owned.request, 2u, chunks, 4u, &schedule, error, sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(schedule.chunk_count == 4u);
    CHECK(schedule.local_view_count == LOCAL_VIEWS);
    CHECK(schedule.global_view_count == 1u);
    CHECK(schedule.max_chunk_views == 2u);
    CHECK(schedule.final_visual_tokens == visual_tokens);
    CHECK(schedule.projected_tokens_total == LOCAL_VIEWS * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES +
                                               UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);

    for (uint32_t i = 0u; i < 3u; ++i) {
        CHECK(chunks[i].kind == UOCR_VISION_CHUNK_LOCAL);
        CHECK(chunks[i].first_view == i * 2u);
        CHECK(chunks[i].view_count == 2u);
        CHECK(chunks[i].projected_grid_w == UOCR_LOCAL_GRID_QUERIES);
        CHECK(chunks[i].projected_grid_h == UOCR_LOCAL_GRID_QUERIES);
        CHECK(chunks[i].projected_tokens_per_view == UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES);
        CHECK(chunks[i].projected_token_start == i * 2u * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES);
    }
    CHECK(chunks[3].kind == UOCR_VISION_CHUNK_GLOBAL);
    CHECK(chunks[3].first_view == LOCAL_VIEWS);
    CHECK(chunks[3].view_count == 1u);
    CHECK(chunks[3].projected_token_start == LOCAL_VIEWS * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES);
    CHECK(chunks[3].projected_token_count == UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);

    free_owned_request(&owned);
    return 0;
}

static int test_schedule_query_and_capacity_failure(void) {
    enum { GRID_W = 2u, GRID_H = 2u, LOCAL_VIEWS = GRID_W * GRID_H };
    const uint32_t visual_tokens = uocr_local_visual_token_count(GRID_W, GRID_H) + UOCR_GLOBAL_VISUAL_TOKENS;
    owned_request owned;
    CHECK(make_request(visual_tokens, LOCAL_VIEWS + 1u, GRID_W, GRID_H, &owned));
    for (uint32_t i = 0u; i < LOCAL_VIEWS; ++i) {
        fill_local_view(&owned.views[i]);
    }
    fill_global_view(&owned.views[LOCAL_VIEWS]);

    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_plan_vision_schedule(&owned.request, 1u, NULL, 0u, &schedule, error, sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(schedule.chunk_count == LOCAL_VIEWS + 1u);
    CHECK(schedule.max_chunk_views == 1u);

    uocr_vision_chunk too_small[2];
    memset(too_small, 0, sizeof(too_small));
    CHECK(uocr_plan_vision_schedule(&owned.request, 1u, too_small, 2u, &schedule, error, sizeof(error)) ==
          UOCR_ERROR_OUT_OF_MEMORY);
    CHECK(schedule.chunk_count == LOCAL_VIEWS + 1u);
    CHECK(strstr(error, "vision schedule needs") != NULL);

    free_owned_request(&owned);
    return 0;
}

static int test_schedule_rejects_invalid_view_order(void) {
    enum { GRID_W = 2u, GRID_H = 1u, LOCAL_VIEWS = GRID_W * GRID_H };
    const uint32_t visual_tokens = uocr_local_visual_token_count(GRID_W, GRID_H) + UOCR_GLOBAL_VISUAL_TOKENS;
    owned_request owned;
    CHECK(make_request(visual_tokens, LOCAL_VIEWS + 1u, GRID_W, GRID_H, &owned));
    fill_local_view(&owned.views[0]);
    fill_global_view(&owned.views[1]);
    fill_local_view(&owned.views[2]);

    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_plan_vision_schedule(&owned.request, 1u, NULL, 0u, &schedule, error, sizeof(error)) ==
          UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(strstr(error, "local views must come first") != NULL || strstr(error, "final view") != NULL);

    free_owned_request(&owned);
    return 0;
}

int main(void) {
    if (test_global_views_are_chunked_in_page_order() != 0) return 1;
    if (test_crop_local_views_are_chunked_before_global() != 0) return 1;
    if (test_schedule_query_and_capacity_failure() != 0) return 1;
    if (test_schedule_rejects_invalid_view_order() != 0) return 1;
    return 0;
}
