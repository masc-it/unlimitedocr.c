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
    uint32_t max_chunk_projected_tokens;
    uint32_t final_visual_tokens;
    uint32_t projected_tokens_total;
} uocr_vision_schedule;

typedef int (*uocr_vision_project_chunk_f16_fn)(const uocr_vision_chunk *chunk,
                                                uint16_t *projected_scratch_f16,
                                                uint32_t projected_scratch_rows,
                                                void *user_data,
                                                char *error,
                                                size_t error_size);

/*
 * Return the production chunk limit for the fixed same-shape batching policy:
 * all local 640x640 views in one chunk, and all global 1024x1024 views in one
 * chunk for global-only requests. Invalid requests still return a safe non-zero
 * default; validation errors are reported by uocr_plan_vision_schedule().
 */
uint32_t uocr_default_vision_max_views_per_chunk(const uocr_prepared_request *request);

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

/*
 * Execute the schedule with a caller-supplied per-chunk projector and format the
 * projected grids into the final visual-feature order expected by the decoder.
 *
 * The projector receives the same reusable scratch buffer for every chunk and
 * must write ``chunk->projected_token_count`` rows of fp16 hidden-size features
 * into it.  This lets the Metal vision path keep SAM/CLIP/projector temporaries
 * bounded by one chunk, then immediately scatter/append rows into the final
 * `[visual_tokens,1280]` buffer.
 */
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
                                   size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_VISION_H */
