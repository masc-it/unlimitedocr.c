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

static void make_chunk(uocr_vision_chunk *chunk,
                       uocr_vision_chunk_kind kind,
                       uint32_t first_view,
                       uint32_t view_count,
                       uint32_t projected_token_start) {
    const uint32_t tokens_per_view = kind == UOCR_VISION_CHUNK_LOCAL ?
                                         UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES :
                                         UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES;
    const uint32_t projected_grid = kind == UOCR_VISION_CHUNK_LOCAL ? UOCR_LOCAL_GRID_QUERIES : UOCR_GLOBAL_GRID_QUERIES;

    chunk->kind = kind;
    chunk->first_view = first_view;
    chunk->view_count = view_count;
    chunk->projected_grid_w = projected_grid;
    chunk->projected_grid_h = projected_grid;
    chunk->projected_tokens_per_view = tokens_per_view;
    chunk->projected_token_start = projected_token_start;
    chunk->projected_token_count = view_count * tokens_per_view;
}

static int add_chunk(uocr_vision_chunk *chunks,
                     uint32_t chunk_capacity,
                     uint32_t *chunk_index,
                     uocr_vision_schedule *summary,
                     uocr_vision_chunk_kind kind,
                     uint32_t first_view,
                     uint32_t view_count,
                     uint32_t projected_token_start) {
    uocr_vision_chunk chunk;
    make_chunk(&chunk, kind, first_view, view_count, projected_token_start);

    if (view_count > summary->max_chunk_views) {
        summary->max_chunk_views = view_count;
    }
    if (chunk.projected_token_count > summary->max_chunk_projected_tokens) {
        summary->max_chunk_projected_tokens = chunk.projected_token_count;
    }

    const uint32_t index = *chunk_index;
    if (chunks != NULL && index < chunk_capacity) {
        chunks[index] = chunk;
    }
    *chunk_index = index + 1u;
    return 1;
}

uint32_t uocr_default_vision_max_views_per_chunk(const uocr_prepared_request *request) {
    if (request == NULL || request->n_views == 0u || request->views == NULL) {
        return 1u;
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

    if (local_views != 0u) {
        return local_views;
    }
    if (global_views != 0u) {
        return global_views;
    }
    return 1u;
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

int uocr_plan_vision_schedule_same_shape(const uocr_prepared_request *request,
                                         uocr_vision_chunk *chunks,
                                         uint32_t chunk_capacity,
                                         uocr_vision_schedule *out_schedule,
                                         char *error,
                                         size_t error_size) {
    const uint32_t chunk_limit = uocr_default_vision_max_views_per_chunk(request);
    const int status = uocr_plan_vision_schedule(request,
                                                 chunk_limit,
                                                 chunks,
                                                 chunk_capacity,
                                                 out_schedule,
                                                 error,
                                                 error_size);
    if (status != UOCR_OK || out_schedule == NULL) {
        return status;
    }

    uint32_t expected_chunks = 0u;
    if (out_schedule->final_visual_tokens != 0u) {
        expected_chunks = out_schedule->local_view_count == 0u ? 1u : 2u;
    }
    if (out_schedule->chunk_count != expected_chunks) {
        return vision_fail(error,
                           error_size,
                           UOCR_ERROR_INTERNAL,
                           "same-shape schedule produced %u chunks, expected %u",
                           out_schedule->chunk_count,
                           expected_chunks);
    }
    return UOCR_OK;
}

static void copy_hidden_row(uint16_t *dst_rows, uint32_t dst_row, const uint16_t *src_row) {
    memcpy(&dst_rows[(size_t)dst_row * (size_t)UOCR_HIDDEN_SIZE],
           src_row,
           (size_t)UOCR_HIDDEN_SIZE * sizeof(uint16_t));
}

static const uint16_t *scratch_row_const(const uint16_t *scratch_rows, uint32_t row) {
    return &scratch_rows[(size_t)row * (size_t)UOCR_HIDDEN_SIZE];
}

static void format_global_projected_view(const uint16_t *projected_grid_f16,
                                         const uint16_t *image_newline_f16,
                                         const uint16_t *view_separator_f16,
                                         uint16_t *out_visual_features_f16,
                                         uint32_t dst_token_base) {
    for (uint32_t row = 0u; row < UOCR_GLOBAL_GRID_QUERIES; ++row) {
        const uint32_t dst_row_base = dst_token_base + row * (UOCR_GLOBAL_GRID_QUERIES + 1u);
        const uint32_t src_row_base = row * UOCR_GLOBAL_GRID_QUERIES;
        for (uint32_t col = 0u; col < UOCR_GLOBAL_GRID_QUERIES; ++col) {
            copy_hidden_row(out_visual_features_f16, dst_row_base + col, scratch_row_const(projected_grid_f16, src_row_base + col));
        }
        copy_hidden_row(out_visual_features_f16, dst_row_base + UOCR_GLOBAL_GRID_QUERIES, image_newline_f16);
    }
    copy_hidden_row(out_visual_features_f16, dst_token_base + UOCR_GLOBAL_ROW_NEWLINE_TOKENS, view_separator_f16);
}

static void format_local_projected_view(const uocr_vision_chunk *chunk,
                                        uint32_t chunk_view_index,
                                        uint32_t crop_grid_w,
                                        uint16_t *out_visual_features_f16,
                                        const uint16_t *projected_scratch_f16) {
    const uint32_t local_view = chunk->first_view + chunk_view_index;
    const uint32_t crop_y = local_view / crop_grid_w;
    const uint32_t crop_x = local_view - crop_y * crop_grid_w;
    const uint32_t stitched_row_stride = crop_grid_w * UOCR_LOCAL_GRID_QUERIES + 1u;
    const uint32_t scratch_view_base = chunk_view_index * chunk->projected_tokens_per_view;

    for (uint32_t row = 0u; row < UOCR_LOCAL_GRID_QUERIES; ++row) {
        const uint32_t dst_row_base = (crop_y * UOCR_LOCAL_GRID_QUERIES + row) * stitched_row_stride +
                                      crop_x * UOCR_LOCAL_GRID_QUERIES;
        const uint32_t src_row_base = scratch_view_base + row * UOCR_LOCAL_GRID_QUERIES;
        for (uint32_t col = 0u; col < UOCR_LOCAL_GRID_QUERIES; ++col) {
            copy_hidden_row(out_visual_features_f16, dst_row_base + col, scratch_row_const(projected_scratch_f16, src_row_base + col));
        }
    }
}

static int run_projector_for_chunk(const uocr_vision_chunk *chunk,
                                   uocr_vision_project_chunk_f16_fn project_chunk,
                                   void *project_user_data,
                                   uint16_t *projected_scratch_f16,
                                   uint32_t projected_scratch_rows,
                                   char *error,
                                   size_t error_size) {
    if (chunk->projected_token_count > projected_scratch_rows) {
        return vision_fail(error,
                           error_size,
                           UOCR_ERROR_OUT_OF_MEMORY,
                           "vision chunk scratch holds %u projected rows, need %u",
                           projected_scratch_rows,
                           chunk->projected_token_count);
    }
    return project_chunk(chunk, projected_scratch_f16, projected_scratch_rows, project_user_data, error, error_size);
}

int uocr_process_vision_chunks_f16(const uocr_prepared_request *request,
                                   uint32_t max_views_per_chunk,
                                   uocr_vision_project_chunk_f16_fn project_chunk,
                                   void *project_user_data,
                                   uint16_t *projected_scratch_f16,
                                   uint32_t projected_scratch_rows,
                                   const uint16_t *image_newline_f16,
                                   const uint16_t *view_separator_f16,
                                   uint16_t *out_visual_features_f16,
                                   uint32_t out_visual_rows,
                                   uocr_vision_schedule *out_schedule,
                                   char *error,
                                   size_t error_size) {
    vision_clear_error(error, error_size);
    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    const int schedule_status = uocr_plan_vision_schedule(request,
                                                          max_views_per_chunk,
                                                          NULL,
                                                          0u,
                                                          &schedule,
                                                          error,
                                                          error_size);
    if (schedule_status != UOCR_OK) {
        return schedule_status;
    }
    if (out_schedule != NULL) {
        *out_schedule = schedule;
    }

    if (schedule.final_visual_tokens == 0u) {
        if (out_visual_rows != 0u) {
            return vision_fail(error,
                               error_size,
                               UOCR_ERROR_INVALID_ARGUMENT,
                               "text-only vision processing expects zero output rows, got %u",
                               out_visual_rows);
        }
        return UOCR_OK;
    }

    if (project_chunk == NULL || projected_scratch_f16 == NULL || image_newline_f16 == NULL || view_separator_f16 == NULL ||
        out_visual_features_f16 == NULL) {
        return vision_fail(error,
                           error_size,
                           UOCR_ERROR_INVALID_ARGUMENT,
                           "vision chunk processing requires projector, scratch, newline, separator, and output buffers");
    }
    if (out_visual_rows != schedule.final_visual_tokens) {
        return vision_fail(error,
                           error_size,
                           UOCR_ERROR_INVALID_ARGUMENT,
                           "vision output row count mismatch: got %u, need %u",
                           out_visual_rows,
                           schedule.final_visual_tokens);
    }
    if (projected_scratch_rows < schedule.max_chunk_projected_tokens) {
        return vision_fail(error,
                           error_size,
                           UOCR_ERROR_OUT_OF_MEMORY,
                           "vision projected scratch has %u rows, need %u for largest chunk",
                           projected_scratch_rows,
                           schedule.max_chunk_projected_tokens);
    }

    const uint32_t chunk_limit = max_views_per_chunk == 0u ? 1u : max_views_per_chunk;
    uint32_t projected_start = 0u;
    if (schedule.local_view_count == 0u) {
        for (uint32_t first = 0u; first < schedule.global_view_count; first += chunk_limit) {
            const uint32_t count = min_u32(chunk_limit, schedule.global_view_count - first);
            uocr_vision_chunk chunk;
            make_chunk(&chunk, UOCR_VISION_CHUNK_GLOBAL, first, count, projected_start);
            const int status = run_projector_for_chunk(&chunk,
                                                       project_chunk,
                                                       project_user_data,
                                                       projected_scratch_f16,
                                                       projected_scratch_rows,
                                                       error,
                                                       error_size);
            if (status != UOCR_OK) {
                return status;
            }
            for (uint32_t view = 0u; view < count; ++view) {
                const uint32_t scratch_base = view * chunk.projected_tokens_per_view;
                const uint32_t dst_base = (first + view) * UOCR_GLOBAL_VISUAL_TOKENS;
                format_global_projected_view(scratch_row_const(projected_scratch_f16, scratch_base),
                                             image_newline_f16,
                                             view_separator_f16,
                                             out_visual_features_f16,
                                             dst_base);
            }
            projected_start += count * UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES;
        }
        return UOCR_OK;
    }

    const uint32_t local_visual_tokens = uocr_local_visual_token_count(request->crop_grid_w, request->crop_grid_h);
    const uint32_t local_row_stride = request->crop_grid_w * UOCR_LOCAL_GRID_QUERIES + 1u;
    const uint32_t local_rows = request->crop_grid_h * UOCR_LOCAL_GRID_QUERIES;
    for (uint32_t row = 0u; row < local_rows; ++row) {
        copy_hidden_row(out_visual_features_f16, row * local_row_stride + request->crop_grid_w * UOCR_LOCAL_GRID_QUERIES, image_newline_f16);
    }

    for (uint32_t first = 0u; first < schedule.local_view_count; first += chunk_limit) {
        const uint32_t count = min_u32(chunk_limit, schedule.local_view_count - first);
        uocr_vision_chunk chunk;
        make_chunk(&chunk, UOCR_VISION_CHUNK_LOCAL, first, count, projected_start);
        const int status = run_projector_for_chunk(&chunk,
                                                   project_chunk,
                                                   project_user_data,
                                                   projected_scratch_f16,
                                                   projected_scratch_rows,
                                                   error,
                                                   error_size);
        if (status != UOCR_OK) {
            return status;
        }
        for (uint32_t view = 0u; view < count; ++view) {
            format_local_projected_view(&chunk, view, request->crop_grid_w, out_visual_features_f16, projected_scratch_f16);
        }
        projected_start += count * UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES;
    }

    uocr_vision_chunk global_chunk;
    make_chunk(&global_chunk, UOCR_VISION_CHUNK_GLOBAL, schedule.local_view_count, 1u, projected_start);
    const int status = run_projector_for_chunk(&global_chunk,
                                               project_chunk,
                                               project_user_data,
                                               projected_scratch_f16,
                                               projected_scratch_rows,
                                               error,
                                               error_size);
    if (status != UOCR_OK) {
        return status;
    }
    format_global_projected_view(projected_scratch_f16,
                                 image_newline_f16,
                                 view_separator_f16,
                                 out_visual_features_f16,
                                 local_visual_tokens);
    return UOCR_OK;
}
