#ifndef UOCR_VISION_H
#define UOCR_VISION_H

#include <stddef.h>
#include <stdint.h>

#include "unlimitedocr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum uocr_vision_chunk_kind {
    UOCR_VISION_CHUNK_LOCAL = 0,
    UOCR_VISION_CHUNK_GLOBAL = 1
} uocr_vision_chunk_kind;

typedef struct uocr_vision_chunk {
    uocr_vision_chunk_kind kind;
    uint32_t first_view;
    uint32_t view_count;
    uint32_t projected_grid_w;
    uint32_t projected_grid_h;
    uint32_t projected_tokens_per_view;
    uint32_t projected_token_start;
    uint32_t projected_token_count;
} uocr_vision_chunk;

typedef struct uocr_vision_schedule {
    uint32_t chunk_count;
    uint32_t local_view_count;
    uint32_t global_view_count;
    uint32_t max_views_per_chunk;
    uint32_t max_chunk_views;
    uint32_t final_visual_tokens;
    uint32_t projected_tokens_total;
    uint32_t reserved0;
} uocr_vision_schedule;

/*
 * Build the view-processing schedule used by the Metal vision bring-up path.
 *
 * The prepared request is validated with unbounded token/generation limits before
 * planning.  If ``chunks`` is NULL or ``chunk_capacity`` is zero, only the summary
 * is written and UOCR_OK is returned.  If a non-empty chunk array is supplied but
 * it is too small, the summary still reports the required ``chunk_count`` and the
 * function returns UOCR_ERROR_OUT_OF_MEMORY.
 *
 * ``max_views_per_chunk == 0`` means one view per chunk.  Crop-mode local views
 * are chunked before the global view, preserving the Python frontend's source
 * feature order: locals in row-major order, then global.  Global-only/multi-page
 * requests are chunked in page order.
 */
int uocr_plan_vision_schedule(const uocr_prepared_request *request,
                              uint32_t max_views_per_chunk,
                              uocr_vision_chunk *chunks,
                              uint32_t chunk_capacity,
                              uocr_vision_schedule *out_schedule,
                              char *error,
                              size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_VISION_H */
