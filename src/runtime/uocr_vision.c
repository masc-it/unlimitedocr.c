#include "runtime/uocr_vision.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "model/uocr_constants.h"
#include "runtime/uocr_request_validation.h"

static int vision_fail(char *error, size_t error_size, int status, const char *fmt, ...) {
    if (error != NULL && error_size > 0u) {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(error, error_size, fmt, ap);
        va_end(ap);
        error[error_size - 1u] = '\0';
    }
    return status;
}

static void vision_clear_error(char *error, size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
}

static uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static int add_chunk(uocr_vision_chunk *chunks,
                     uint32_t chunk_capacity,
                     uint32_t *chunk_index,
                     uocr_vision_schedule *summary,
                     uocr_vision_chunk_kind kind,
                     uint32_t first_view,
                     uint32_t view_count,
                     uint32_t projected_token_start) {
    const uint32_t tokens_per_view = kind == UOCR_VISION_CHUNK_LOCAL ?
                                         UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES :
                                         UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES;
    const uint32_t projected_grid = kind == UOCR_VISION_CHUNK_LOCAL ? UOCR_LOCAL_GRID_QUERIES : UOCR_GLOBAL_GRID_QUERIES;

    if (view_count > summary->max_chunk_views) {
        summary->max_chunk_views = view_count;
    }

    const uint32_t index = *chunk_index;
    if (chunks != NULL && index < chunk_capacity) {
        chunks[index].kind = kind;
        chunks[index].first_view = first_view;
        chunks[index].view_count = view_count;
        chunks[index].projected_grid_w = projected_grid;
        chunks[index].projected_grid_h = projected_grid;
        chunks[index].projected_tokens_per_view = tokens_per_view;
        chunks[index].projected_token_start = projected_token_start;
        chunks[index].projected_token_count = view_count * tokens_per_view;
    }
    *chunk_index = index + 1u;
    return 1;
}

int uocr_plan_vision_schedule(const uocr_prepared_request *request,
                              uint32_t max_views_per_chunk,
                              uocr_vision_chunk *chunks,
                              uint32_t chunk_capacity,
                              uocr_vision_schedule *out_schedule,
                              char *error,
                              size_t error_size) {
    vision_clear_error(error, error_size);
    if (out_schedule == NULL) {
        return vision_fail(error, error_size, UOCR_ERROR_INVALID_ARGUMENT, "vision schedule output is null");
    }
    memset(out_schedule, 0, sizeof(*out_schedule));
    if (request == NULL) {
        return vision_fail(error, error_size, UOCR_ERROR_INVALID_ARGUMENT, "prepared request is null");
    }

    const uint32_t chunk_limit = max_views_per_chunk == 0u ? 1u : max_views_per_chunk;
    out_schedule->max_views_per_chunk = chunk_limit;

    uocr_request_limits limits;
    memset(&limits, 0, sizeof(limits));
    const int validation_status = uocr_validate_prepared_request(request, &limits, error, error_size);
    if (validation_status != UOCR_OK) {
        return validation_status;
    }

    if (request->n_views == 0u) {
        if (uocr_count_image_placeholders(request) != 0u) {
            return vision_fail(error, error_size, UOCR_ERROR_INVALID_ARGUMENT, "image placeholders require image views");
        }
        return UOCR_OK;
    }

    uint32_t local_views = 0u;
    uint32_t global_views = 0u;
    for (uint32_t i = 0u; i < request->n_views; ++i) {
        if (request->views[i].kind == UOCR_VIEW_LOCAL) {
            ++local_views;
        } else if (request->views[i].kind == UOCR_VIEW_GLOBAL) {
            ++global_views;
        }
    }

    out_schedule->local_view_count = local_views;
    out_schedule->global_view_count = global_views;

    if (local_views == 0u) {
        out_schedule->final_visual_tokens = global_views * UOCR_GLOBAL_VISUAL_TOKENS;
        out_schedule->projected_tokens_total = global_views * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES;
    } else {
        const uint32_t local_visual = uocr_local_visual_token_count(request->crop_grid_w, request->crop_grid_h);
        out_schedule->final_visual_tokens = local_visual + UOCR_GLOBAL_VISUAL_TOKENS;
        out_schedule->projected_tokens_total =
            local_views * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES +
            UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES;
    }

    uint32_t chunk_index = 0u;
    uint32_t projected_start = 0u;
    if (local_views == 0u) {
        for (uint32_t first = 0u; first < global_views; first += chunk_limit) {
            const uint32_t count = min_u32(chunk_limit, global_views - first);
            (void)add_chunk(chunks,
                            chunk_capacity,
                            &chunk_index,
                            out_schedule,
                            UOCR_VISION_CHUNK_GLOBAL,
                            first,
                            count,
                            projected_start);
            projected_start += count * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES;
        }
    } else {
        for (uint32_t first = 0u; first < local_views; first += chunk_limit) {
            const uint32_t count = min_u32(chunk_limit, local_views - first);
            (void)add_chunk(chunks,
                            chunk_capacity,
                            &chunk_index,
                            out_schedule,
                            UOCR_VISION_CHUNK_LOCAL,
                            first,
                            count,
                            projected_start);
            projected_start += count * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES;
        }
        (void)add_chunk(chunks,
                        chunk_capacity,
                        &chunk_index,
                        out_schedule,
                        UOCR_VISION_CHUNK_GLOBAL,
                        local_views,
                        1u,
                        projected_start);
    }

    out_schedule->chunk_count = chunk_index;
    if (chunks != NULL && chunk_capacity != 0u && chunk_capacity < chunk_index) {
        return vision_fail(error,
                           error_size,
                           UOCR_ERROR_OUT_OF_MEMORY,
                           "vision schedule needs %u chunks, capacity is %u",
                           chunk_index,
                           chunk_capacity);
    }
    return UOCR_OK;
}
