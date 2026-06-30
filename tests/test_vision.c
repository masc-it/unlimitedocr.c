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

typedef struct synthetic_projector {
    uint32_t calls;
    uint32_t scratch_reused;
    uint32_t min_scratch_rows_seen;
    uint16_t *first_scratch;
    uocr_vision_chunk chunks[16];
} synthetic_projector;

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

static uint16_t *alloc_f16_rows(uint32_t rows) {
    return (uint16_t *)calloc((size_t)rows * (size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
}

static const uint16_t *const_row(const uint16_t *rows, uint32_t row) {
    return &rows[(size_t)row * (size_t)UOCR_HIDDEN_SIZE];
}

static uint16_t *row_mut(uint16_t *rows, uint32_t row) {
    return &rows[(size_t)row * (size_t)UOCR_HIDDEN_SIZE];
}

static uint16_t projected_value(uocr_vision_chunk_kind kind, uint32_t view, uint32_t row, uint32_t col, uint32_t channel) {
    const uint32_t base = kind == UOCR_VISION_CHUNK_LOCAL ? 0x4000u : 0x1000u;
    return (uint16_t)(base + (view & 0x3fu) * 0x100u + (row & 0x1fu) * 0x10u + (col & 0x0fu) + (channel & 0x03u));
}

static uint16_t newline_value(uint32_t channel) {
    return (uint16_t)(0x7000u + (channel & 0x7fu));
}

static uint16_t separator_value(uint32_t channel) {
    return (uint16_t)(0x7800u + (channel & 0x7fu));
}

static void fill_special_rows(uint16_t *newline, uint16_t *separator) {
    for (uint32_t col = 0u; col < UOCR_HIDDEN_SIZE; ++col) {
        newline[col] = newline_value(col);
        separator[col] = separator_value(col);
    }
}

static int synthetic_project_chunk(const uocr_vision_chunk *chunk,
                                   uint16_t *projected_scratch_f16,
                                   uint32_t projected_scratch_rows,
                                   void *user_data,
                                   char *error,
                                   size_t error_size) {
    (void)error;
    (void)error_size;
    synthetic_projector *projector = (synthetic_projector *)user_data;
    if (chunk == NULL || projected_scratch_f16 == NULL || projector == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (projected_scratch_rows < chunk->projected_token_count) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    if (projector->calls < (uint32_t)(sizeof(projector->chunks) / sizeof(projector->chunks[0]))) {
        projector->chunks[projector->calls] = *chunk;
    }
    if (projector->calls == 0u) {
        projector->first_scratch = projected_scratch_f16;
        projector->scratch_reused = 1u;
        projector->min_scratch_rows_seen = projected_scratch_rows;
    } else if (projector->first_scratch != projected_scratch_f16) {
        projector->scratch_reused = 0u;
    }
    if (projected_scratch_rows < projector->min_scratch_rows_seen) {
        projector->min_scratch_rows_seen = projected_scratch_rows;
    }
    ++projector->calls;

    for (uint32_t view = 0u; view < chunk->view_count; ++view) {
        const uint32_t absolute_view = chunk->first_view + view;
        const uint32_t view_base = view * chunk->projected_tokens_per_view;
        for (uint32_t row = 0u; row < chunk->projected_grid_h; ++row) {
            for (uint32_t col = 0u; col < chunk->projected_grid_w; ++col) {
                uint16_t *dst = row_mut(projected_scratch_f16, view_base + row * chunk->projected_grid_w + col);
                for (uint32_t channel = 0u; channel < UOCR_HIDDEN_SIZE; ++channel) {
                    dst[channel] = projected_value(chunk->kind, absolute_view, row, col, channel);
                }
            }
        }
    }
    return UOCR_OK;
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
    CHECK(schedule.max_chunk_projected_tokens == 2u * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
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
    CHECK(schedule.max_chunk_projected_tokens == UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
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

static int test_default_local_chunk_limit_is_memory_aware(void) {
    enum { GRID_W = 4u, GRID_H = 3u, LOCAL_VIEWS = GRID_W * GRID_H };
    const uint32_t local_visual_tokens = uocr_local_visual_token_count(GRID_W, GRID_H);
    const uint32_t visual_tokens = local_visual_tokens + UOCR_GLOBAL_VISUAL_TOKENS;
    owned_request owned;
    CHECK(make_request(visual_tokens, LOCAL_VIEWS + 1u, GRID_W, GRID_H, &owned));
    for (uint32_t i = 0u; i < LOCAL_VIEWS; ++i) {
        fill_local_view(&owned.views[i]);
    }
    fill_global_view(&owned.views[LOCAL_VIEWS]);

    const uint32_t default_limit = uocr_default_vision_max_views_per_chunk(&owned.request);
    CHECK(default_limit == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK);

    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    uocr_vision_chunk chunks[3];
    memset(chunks, 0, sizeof(chunks));
    CHECK(uocr_plan_vision_schedule(&owned.request, default_limit, chunks, 3u, &schedule, error, sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(schedule.chunk_count == 3u);
    CHECK(schedule.local_view_count == LOCAL_VIEWS);
    CHECK(schedule.global_view_count == 1u);
    CHECK(schedule.max_views_per_chunk == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK);
    CHECK(schedule.max_chunk_views == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK);
    CHECK(schedule.max_chunk_projected_tokens == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES);
    CHECK(schedule.max_local_chunk_views == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK);
    CHECK(schedule.max_local_chunk_projected_tokens == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES);
    CHECK(schedule.max_global_chunk_views == 1u);
    CHECK(schedule.max_global_chunk_projected_tokens == UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
    CHECK(schedule.final_visual_tokens == visual_tokens);
    CHECK(chunks[0].kind == UOCR_VISION_CHUNK_LOCAL);
    CHECK(chunks[0].first_view == 0u && chunks[0].view_count == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK);
    CHECK(chunks[0].projected_token_count == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES);
    CHECK(chunks[1].kind == UOCR_VISION_CHUNK_LOCAL);
    CHECK(chunks[1].first_view == UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK);
    CHECK(chunks[1].view_count == LOCAL_VIEWS - UOCR_DEFAULT_LOCAL_VIEWS_PER_CHUNK);
    CHECK(chunks[2].kind == UOCR_VISION_CHUNK_GLOBAL);
    CHECK(chunks[2].first_view == LOCAL_VIEWS);
    CHECK(chunks[2].final_token_start == local_visual_tokens);

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
    CHECK(schedule.max_chunk_projected_tokens == UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);

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

static int test_schedule_rejects_visual_placeholder_count_mismatch(void) {
    owned_request owned;
    CHECK(make_request(UOCR_GLOBAL_VISUAL_TOKENS - 1u, 1u, 1u, 1u, &owned));
    fill_global_view(&owned.views[0]);

    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_plan_vision_schedule(&owned.request, 1u, NULL, 0u, &schedule, error, sizeof(error)) ==
          UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(strstr(error, "image placeholder count mismatch") != NULL);

    free_owned_request(&owned);
    return 0;
}

static int test_text_only_processing_is_a_noop(void) {
    owned_request owned;
    CHECK(make_request(0u, 0u, 1u, 1u, &owned));

    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_process_vision_chunks_f16(&owned.request,
                                         1u,
                                         NULL,
                                         NULL,
                                         NULL,
                                         0u,
                                         NULL,
                                         NULL,
                                         NULL,
                                         0u,
                                         &schedule,
                                         error,
                                         sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(schedule.final_visual_tokens == 0u);
    CHECK(schedule.chunk_count == 0u);

    CHECK(uocr_process_vision_chunks_f16(&owned.request,
                                         1u,
                                         NULL,
                                         NULL,
                                         NULL,
                                         0u,
                                         NULL,
                                         NULL,
                                         NULL,
                                         1u,
                                         &schedule,
                                         error,
                                         sizeof(error)) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(strstr(error, "expects zero output rows") != NULL);

    free_owned_request(&owned);
    return 0;
}

static int test_process_single_global_view_supports_zero_locals(void) {
    owned_request owned;
    CHECK(make_request(UOCR_GLOBAL_VISUAL_TOKENS, 1u, 1u, 1u, &owned));
    fill_global_view(&owned.views[0]);

    uint16_t *scratch = alloc_f16_rows(UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
    uint16_t *out = alloc_f16_rows(UOCR_GLOBAL_VISUAL_TOKENS);
    uint16_t newline[UOCR_HIDDEN_SIZE];
    uint16_t separator[UOCR_HIDDEN_SIZE];
    CHECK(scratch != NULL);
    CHECK(out != NULL);
    fill_special_rows(newline, separator);

    synthetic_projector projector;
    memset(&projector, 0, sizeof(projector));
    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_process_vision_chunks_f16(&owned.request,
                                         0u,
                                         synthetic_project_chunk,
                                         &projector,
                                         scratch,
                                         UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
                                         newline,
                                         separator,
                                         out,
                                         UOCR_GLOBAL_VISUAL_TOKENS,
                                         &schedule,
                                         error,
                                         sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(projector.calls == 1u);
    CHECK(projector.chunks[0].kind == UOCR_VISION_CHUNK_GLOBAL);
    CHECK(projector.chunks[0].first_view == 0u);
    CHECK(projector.chunks[0].view_count == 1u);
    CHECK(schedule.local_view_count == 0u);
    CHECK(schedule.global_view_count == 1u);
    CHECK(schedule.final_visual_tokens == UOCR_GLOBAL_VISUAL_TOKENS);
    CHECK(const_row(out, 0u)[0] == projected_value(UOCR_VISION_CHUNK_GLOBAL, 0u, 0u, 0u, 0u));
    CHECK(const_row(out, UOCR_GLOBAL_GRID_QUERIES)[0] == newline_value(0u));
    CHECK(const_row(out, UOCR_GLOBAL_ROW_NEWLINE_TOKENS)[0] == separator_value(0u));

    free(out);
    free(scratch);
    free_owned_request(&owned);
    return 0;
}

static int test_process_global_views_formats_rows_and_reuses_scratch(void) {
    owned_request owned;
    CHECK(make_request(3u * UOCR_GLOBAL_VISUAL_TOKENS, 3u, 1u, 1u, &owned));
    for (uint32_t i = 0u; i < 3u; ++i) {
        fill_global_view(&owned.views[i]);
    }

    uint16_t *scratch = alloc_f16_rows(2u * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
    uint16_t *out = alloc_f16_rows(3u * UOCR_GLOBAL_VISUAL_TOKENS);
    uint16_t newline[UOCR_HIDDEN_SIZE];
    uint16_t separator[UOCR_HIDDEN_SIZE];
    CHECK(scratch != NULL);
    CHECK(out != NULL);
    fill_special_rows(newline, separator);

    synthetic_projector projector;
    memset(&projector, 0, sizeof(projector));
    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_process_vision_chunks_f16(&owned.request,
                                         2u,
                                         synthetic_project_chunk,
                                         &projector,
                                         scratch,
                                         2u * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
                                         newline,
                                         separator,
                                         out,
                                         3u * UOCR_GLOBAL_VISUAL_TOKENS,
                                         &schedule,
                                         error,
                                         sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(projector.calls == 2u);
    CHECK(projector.scratch_reused == 1u);
    CHECK(projector.chunks[0].first_view == 0u && projector.chunks[0].view_count == 2u);
    CHECK(projector.chunks[1].first_view == 2u && projector.chunks[1].view_count == 1u);
    CHECK(schedule.max_chunk_projected_tokens == 2u * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);

    const uint32_t channels[] = {0u, 1u, 1279u};
    for (uint32_t view = 0u; view < 3u; ++view) {
        const uint32_t base = view * UOCR_GLOBAL_VISUAL_TOKENS;
        for (uint32_t row = 0u; row < UOCR_GLOBAL_GRID_QUERIES; ++row) {
            const uint32_t dst_row_base = base + row * (UOCR_GLOBAL_GRID_QUERIES + 1u);
            for (uint32_t col = 0u; col < UOCR_GLOBAL_GRID_QUERIES; ++col) {
                const uint16_t *actual = const_row(out, dst_row_base + col);
                for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
                    CHECK(actual[channels[c]] == projected_value(UOCR_VISION_CHUNK_GLOBAL, view, row, col, channels[c]));
                }
            }
            const uint16_t *newline_row = const_row(out, dst_row_base + UOCR_GLOBAL_GRID_QUERIES);
            for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
                CHECK(newline_row[channels[c]] == newline_value(channels[c]));
            }
        }
        const uint16_t *separator_row = const_row(out, base + UOCR_GLOBAL_ROW_NEWLINE_TOKENS);
        for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
            CHECK(separator_row[channels[c]] == separator_value(channels[c]));
        }
    }

    free(out);
    free(scratch);
    free_owned_request(&owned);
    return 0;
}

static int test_process_crop_views_scatters_partial_chunks_to_final_order(void) {
    enum { GRID_W = 3u, GRID_H = 2u, LOCAL_VIEWS = GRID_W * GRID_H };
    const uint32_t local_visual_tokens = uocr_local_visual_token_count(GRID_W, GRID_H);
    const uint32_t visual_tokens = local_visual_tokens + UOCR_GLOBAL_VISUAL_TOKENS;
    owned_request owned;
    CHECK(make_request(visual_tokens, LOCAL_VIEWS + 1u, GRID_W, GRID_H, &owned));
    for (uint32_t i = 0u; i < LOCAL_VIEWS; ++i) {
        fill_local_view(&owned.views[i]);
    }
    fill_global_view(&owned.views[LOCAL_VIEWS]);

    uint16_t *scratch = alloc_f16_rows(UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);
    uint16_t *out = alloc_f16_rows(visual_tokens);
    uint16_t newline[UOCR_HIDDEN_SIZE];
    uint16_t separator[UOCR_HIDDEN_SIZE];
    CHECK(scratch != NULL);
    CHECK(out != NULL);
    fill_special_rows(newline, separator);

    synthetic_projector projector;
    memset(&projector, 0, sizeof(projector));
    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_process_vision_chunks_f16(&owned.request,
                                         2u,
                                         synthetic_project_chunk,
                                         &projector,
                                         scratch,
                                         UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
                                         newline,
                                         separator,
                                         out,
                                         visual_tokens,
                                         &schedule,
                                         error,
                                         sizeof(error)) == UOCR_OK);
    CHECK(error[0] == '\0');
    CHECK(projector.calls == 4u);
    CHECK(projector.scratch_reused == 1u);
    CHECK(projector.chunks[0].kind == UOCR_VISION_CHUNK_LOCAL && projector.chunks[0].first_view == 0u);
    CHECK(projector.chunks[1].kind == UOCR_VISION_CHUNK_LOCAL && projector.chunks[1].first_view == 2u);
    CHECK(projector.chunks[2].kind == UOCR_VISION_CHUNK_LOCAL && projector.chunks[2].first_view == 4u);
    CHECK(projector.chunks[3].kind == UOCR_VISION_CHUNK_GLOBAL && projector.chunks[3].first_view == LOCAL_VIEWS);
    CHECK(schedule.max_chunk_projected_tokens == UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES);

    const uint32_t channels[] = {0u, 5u, 1279u};
    const uint32_t local_row_stride = GRID_W * UOCR_LOCAL_GRID_QUERIES + 1u;
    for (uint32_t view = 0u; view < LOCAL_VIEWS; ++view) {
        const uint32_t crop_y = view / GRID_W;
        const uint32_t crop_x = view - crop_y * GRID_W;
        for (uint32_t row = 0u; row < UOCR_LOCAL_GRID_QUERIES; ++row) {
            const uint32_t dst_row_base = (crop_y * UOCR_LOCAL_GRID_QUERIES + row) * local_row_stride +
                                          crop_x * UOCR_LOCAL_GRID_QUERIES;
            for (uint32_t col = 0u; col < UOCR_LOCAL_GRID_QUERIES; ++col) {
                const uint16_t *actual = const_row(out, dst_row_base + col);
                for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
                    CHECK(actual[channels[c]] == projected_value(UOCR_VISION_CHUNK_LOCAL, view, row, col, channels[c]));
                }
            }
        }
    }

    for (uint32_t row = 0u; row < GRID_H * UOCR_LOCAL_GRID_QUERIES; ++row) {
        const uint16_t *newline_row = const_row(out, row * local_row_stride + GRID_W * UOCR_LOCAL_GRID_QUERIES);
        for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
            CHECK(newline_row[channels[c]] == newline_value(channels[c]));
        }
    }

    for (uint32_t row = 0u; row < UOCR_GLOBAL_GRID_QUERIES; ++row) {
        const uint32_t dst_row_base = local_visual_tokens + row * (UOCR_GLOBAL_GRID_QUERIES + 1u);
        for (uint32_t col = 0u; col < UOCR_GLOBAL_GRID_QUERIES; ++col) {
            const uint16_t *actual = const_row(out, dst_row_base + col);
            for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
                CHECK(actual[channels[c]] == projected_value(UOCR_VISION_CHUNK_GLOBAL, LOCAL_VIEWS, row, col, channels[c]));
            }
        }
        const uint16_t *newline_row = const_row(out, dst_row_base + UOCR_GLOBAL_GRID_QUERIES);
        for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
            CHECK(newline_row[channels[c]] == newline_value(channels[c]));
        }
    }
    const uint16_t *separator_row = const_row(out, local_visual_tokens + UOCR_GLOBAL_ROW_NEWLINE_TOKENS);
    for (uint32_t c = 0u; c < (uint32_t)(sizeof(channels) / sizeof(channels[0])); ++c) {
        CHECK(separator_row[channels[c]] == separator_value(channels[c]));
    }

    free(out);
    free(scratch);
    free_owned_request(&owned);
    return 0;
}

static int test_process_rejects_too_small_projected_scratch(void) {
    enum { GRID_W = 3u, GRID_H = 2u, LOCAL_VIEWS = GRID_W * GRID_H };
    const uint32_t visual_tokens = uocr_local_visual_token_count(GRID_W, GRID_H) + UOCR_GLOBAL_VISUAL_TOKENS;
    owned_request owned;
    CHECK(make_request(visual_tokens, LOCAL_VIEWS + 1u, GRID_W, GRID_H, &owned));
    for (uint32_t i = 0u; i < LOCAL_VIEWS; ++i) {
        fill_local_view(&owned.views[i]);
    }
    fill_global_view(&owned.views[LOCAL_VIEWS]);

    uint16_t *scratch = alloc_f16_rows(UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES - 1u);
    uint16_t *out = alloc_f16_rows(visual_tokens);
    uint16_t newline[UOCR_HIDDEN_SIZE];
    uint16_t separator[UOCR_HIDDEN_SIZE];
    CHECK(scratch != NULL);
    CHECK(out != NULL);
    fill_special_rows(newline, separator);

    synthetic_projector projector;
    memset(&projector, 0, sizeof(projector));
    char error[256];
    memset(error, 0, sizeof(error));
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    CHECK(uocr_process_vision_chunks_f16(&owned.request,
                                         2u,
                                         synthetic_project_chunk,
                                         &projector,
                                         scratch,
                                         UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES - 1u,
                                         newline,
                                         separator,
                                         out,
                                         visual_tokens,
                                         &schedule,
                                         error,
                                         sizeof(error)) == UOCR_ERROR_OUT_OF_MEMORY);
    CHECK(projector.calls == 0u);
    CHECK(strstr(error, "projected scratch") != NULL);

    free(out);
    free(scratch);
    free_owned_request(&owned);
    return 0;
}

int main(void) {
    if (test_global_views_are_chunked_in_page_order() != 0) return 1;
    if (test_crop_local_views_are_chunked_before_global() != 0) return 1;
    if (test_default_local_chunk_limit_is_memory_aware() != 0) return 1;
    if (test_schedule_query_and_capacity_failure() != 0) return 1;
    if (test_schedule_rejects_invalid_view_order() != 0) return 1;
    if (test_schedule_rejects_visual_placeholder_count_mismatch() != 0) return 1;
    if (test_text_only_processing_is_a_noop() != 0) return 1;
    if (test_process_single_global_view_supports_zero_locals() != 0) return 1;
    if (test_process_global_views_formats_rows_and_reuses_scratch() != 0) return 1;
    if (test_process_crop_views_scatters_partial_chunks_to_final_order() != 0) return 1;
    if (test_process_rejects_too_small_projected_scratch() != 0) return 1;
    return 0;
}
