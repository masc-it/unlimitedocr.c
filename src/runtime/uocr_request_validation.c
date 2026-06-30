#include "runtime/uocr_request_validation.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "model/uocr_constants.h"

static int fail(char *error, size_t error_size, const char *fmt, ...) {
    if (error != NULL && error_size > 0u) {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(error, error_size, fmt, ap);
        va_end(ap);
        error[error_size - 1u] = '\0';
    }
    return UOCR_ERROR_INVALID_ARGUMENT;
}

static int is_valid_format(uocr_pixel_format format) {
    return format == UOCR_PIXEL_F16_NCHW || format == UOCR_PIXEL_F32_NCHW;
}

static int validate_view_shape(const uocr_image_view *view, uint32_t index, char *error, size_t error_size) {
    if (view->pixels == NULL) {
        return fail(error, error_size, "view %u has null pixel pointer", index);
    }
    if (!is_valid_format(view->format)) {
        return fail(error, error_size, "view %u has unsupported pixel format %d", index, (int)view->format);
    }
    switch (view->kind) {
        case UOCR_VIEW_GLOBAL:
            if (view->width != UOCR_GLOBAL_VIEW_SIZE || view->height != UOCR_GLOBAL_VIEW_SIZE) {
                return fail(error,
                            error_size,
                            "global view %u must be %ux%u, got %ux%u",
                            index,
                            UOCR_GLOBAL_VIEW_SIZE,
                            UOCR_GLOBAL_VIEW_SIZE,
                            view->width,
                            view->height);
            }
            return UOCR_OK;
        case UOCR_VIEW_LOCAL:
            if (view->width != UOCR_LOCAL_VIEW_SIZE || view->height != UOCR_LOCAL_VIEW_SIZE) {
                return fail(error,
                            error_size,
                            "local view %u must be %ux%u, got %ux%u",
                            index,
                            UOCR_LOCAL_VIEW_SIZE,
                            UOCR_LOCAL_VIEW_SIZE,
                            view->width,
                            view->height);
            }
            return UOCR_OK;
        default:
            return fail(error, error_size, "view %u has unsupported kind %d", index, (int)view->kind);
    }
}

uint32_t uocr_count_image_placeholders(const uocr_prepared_request *request) {
    uint32_t count = 0u;
    if (request == NULL || request->image_mask == NULL) {
        return 0u;
    }
    for (uint32_t i = 0u; i < request->n_tokens; ++i) {
        count += request->image_mask[i] == 1u ? 1u : 0u;
    }
    return count;
}

static int validate_kv_and_position_budget(const uocr_prepared_request *request,
                                           const uocr_request_limits *limits,
                                           char *error,
                                           size_t error_size) {
    if (limits->max_position_tokens != 0u) {
        if (request->n_tokens > limits->max_position_tokens ||
            request->max_new_tokens > limits->max_position_tokens - request->n_tokens) {
            return fail(error,
                        error_size,
                        "sequence length exceeds model position budget: prompt %u + max_new_tokens %u > %u",
                        request->n_tokens,
                        request->max_new_tokens,
                        limits->max_position_tokens);
        }
    }

    if (limits->max_prompt_tokens != 0u && limits->generated_ring_window != 0u) {
        const uint32_t live_generated = request->max_new_tokens < limits->generated_ring_window ?
                                            request->max_new_tokens :
                                            limits->generated_ring_window;
        const uint64_t required_cached_tokens = (uint64_t)request->n_tokens + (uint64_t)live_generated;
        const uint64_t capacity_cached_tokens = (uint64_t)limits->max_prompt_tokens +
                                                (uint64_t)limits->generated_ring_window;
        if (required_cached_tokens > capacity_cached_tokens) {
            return fail(error,
                        error_size,
                        "KV cache budget exceeded: request needs %llu cached tokens (prompt %u + live generated %u), capacity is %llu (prompt capacity %u + ring %u)",
                        (unsigned long long)required_cached_tokens,
                        request->n_tokens,
                        live_generated,
                        (unsigned long long)capacity_cached_tokens,
                        limits->max_prompt_tokens,
                        limits->generated_ring_window);
        }
    }

    return UOCR_OK;
}

static int expected_visual_tokens(const uocr_prepared_request *request,
                                  uint32_t *out_expected,
                                  char *error,
                                  size_t error_size) {
    uint32_t globals = 0u;
    uint32_t locals = 0u;

    if (request->n_views == 0u) {
        *out_expected = 0u;
        return UOCR_OK;
    }
    if (request->views == NULL) {
        return fail(error, error_size, "n_views is %u but views pointer is null", request->n_views);
    }

    for (uint32_t i = 0u; i < request->n_views; ++i) {
        int status = validate_view_shape(&request->views[i], i, error, error_size);
        if (status != UOCR_OK) {
            return status;
        }
        if (request->views[i].kind == UOCR_VIEW_GLOBAL) {
            ++globals;
        } else if (request->views[i].kind == UOCR_VIEW_LOCAL) {
            ++locals;
        }
    }

    if (locals == 0u) {
        if (globals != request->n_views) {
            return fail(error, error_size, "internal view-count mismatch");
        }
        if (!((request->crop_grid_w == 0u || request->crop_grid_w == 1u) &&
              (request->crop_grid_h == 0u || request->crop_grid_h == 1u))) {
            return fail(error,
                        error_size,
                        "global-only requests must use crop grid 1x1 (or 0x0 unset), got %ux%u",
                        request->crop_grid_w,
                        request->crop_grid_h);
        }
        *out_expected = globals * UOCR_GLOBAL_VISUAL_TOKENS;
        return UOCR_OK;
    }

    if (request->crop_grid_w == 0u || request->crop_grid_h == 0u) {
        return fail(error, error_size, "local crop views require a non-zero crop grid");
    }
    if (globals != 1u) {
        return fail(error, error_size, "crop-mode requests require exactly one global view, got %u", globals);
    }

    const uint64_t expected_locals = (uint64_t)request->crop_grid_w * (uint64_t)request->crop_grid_h;
    if (expected_locals <= 1u) {
        return fail(error,
                    error_size,
                    "crop grid %ux%u should be represented as global-only; no local [1,1] view is expected",
                    request->crop_grid_w,
                    request->crop_grid_h);
    }
    if (expected_locals != (uint64_t)locals) {
        return fail(error,
                    error_size,
                    "crop grid %ux%u expects %llu local views, got %u",
                    request->crop_grid_w,
                    request->crop_grid_h,
                    (unsigned long long)expected_locals,
                    locals);
    }

    for (uint32_t i = 0u; i < locals; ++i) {
        if (request->views[i].kind != UOCR_VIEW_LOCAL) {
            return fail(error, error_size, "crop-mode local views must come first; view %u is not local", i);
        }
    }
    if (request->views[locals].kind != UOCR_VIEW_GLOBAL) {
        return fail(error, error_size, "crop-mode final view must be the global view");
    }

    const uint64_t local_tokens = (uint64_t)uocr_local_visual_token_count(request->crop_grid_w, request->crop_grid_h);
    const uint64_t total = local_tokens + UOCR_GLOBAL_VISUAL_TOKENS;
    if (total > UINT32_MAX) {
        return fail(error, error_size, "visual token count overflow for crop grid %ux%u", request->crop_grid_w, request->crop_grid_h);
    }
    *out_expected = (uint32_t)total;
    return UOCR_OK;
}

int uocr_validate_prepared_request(const uocr_prepared_request *request,
                                   const uocr_request_limits *limits,
                                   char *error,
                                   size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
    if (request == NULL) {
        return fail(error, error_size, "prepared request is null");
    }
    if (limits == NULL) {
        return fail(error, error_size, "request limits are null");
    }
    if (request->n_tokens == 0u) {
        return fail(error, error_size, "prepared request must contain at least BOS");
    }
    if (request->input_ids == NULL) {
        return fail(error, error_size, "input_ids pointer is null");
    }
    if (request->image_mask == NULL) {
        return fail(error, error_size, "image_mask pointer is null");
    }
    if (limits->max_prompt_tokens > 0u && request->n_tokens > limits->max_prompt_tokens) {
        return fail(error,
                    error_size,
                    "prompt has %u tokens, engine limit is %u",
                    request->n_tokens,
                    limits->max_prompt_tokens);
    }
    if (limits->max_gen_tokens > 0u && request->max_new_tokens > limits->max_gen_tokens) {
        return fail(error,
                    error_size,
                    "request asks for %u new tokens, engine limit is %u",
                    request->max_new_tokens,
                    limits->max_gen_tokens);
    }
    const int budget_status = validate_kv_and_position_budget(request, limits, error, error_size);
    if (budget_status != UOCR_OK) {
        return budget_status;
    }
    if (request->input_ids[0] != UOCR_TOKEN_BOS) {
        return fail(error, error_size, "first token must be BOS id %d, got %d", UOCR_TOKEN_BOS, request->input_ids[0]);
    }
    if (request->no_repeat_ngram_size == 0u && request->no_repeat_window != 0u) {
        return fail(error, error_size, "no_repeat_window is set but no_repeat_ngram_size is zero");
    }
    if (request->no_repeat_ngram_size > UOCR_MAX_POSITIONS) {
        return fail(error,
                    error_size,
                    "no_repeat_ngram_size %u exceeds max positions %u",
                    request->no_repeat_ngram_size,
                    UOCR_MAX_POSITIONS);
    }
    if (request->no_repeat_window > UOCR_MAX_POSITIONS) {
        return fail(error,
                    error_size,
                    "no_repeat_window %u exceeds max positions %u",
                    request->no_repeat_window,
                    UOCR_MAX_POSITIONS);
    }

    uint32_t image_placeholders = 0u;
    uint32_t first_image = request->n_tokens;
    uint32_t last_image = 0u;
    for (uint32_t i = 0u; i < request->n_tokens; ++i) {
        const int32_t token = request->input_ids[i];
        if (token < 0 || token >= (int32_t)UOCR_VOCAB_SIZE) {
            return fail(error, error_size, "token %u is out of vocabulary: %d", i, token);
        }
        if (request->image_mask[i] > 1u) {
            return fail(error, error_size, "image_mask[%u] must be 0 or 1, got %u", i, (unsigned)request->image_mask[i]);
        }
        if (request->image_mask[i] == 1u) {
            if (token != UOCR_TOKEN_IMAGE) {
                return fail(error,
                            error_size,
                            "image_mask[%u] is set but token id is %d, expected image token %d",
                            i,
                            token,
                            UOCR_TOKEN_IMAGE);
            }
            if (first_image == request->n_tokens) {
                first_image = i;
            }
            last_image = i;
            ++image_placeholders;
        } else if (token == UOCR_TOKEN_IMAGE) {
            return fail(error,
                        error_size,
                        "token %u is image token %d but image_mask is 0",
                        i,
                        UOCR_TOKEN_IMAGE);
        }
    }
    if (image_placeholders != 0u && last_image - first_image + 1u != image_placeholders) {
        return fail(error,
                    error_size,
                    "image placeholders must be one contiguous span; first=%u last=%u count=%u",
                    first_image,
                    last_image,
                    image_placeholders);
    }

    uint32_t expected_visual = 0u;
    int status = expected_visual_tokens(request, &expected_visual, error, error_size);
    if (status != UOCR_OK) {
        return status;
    }
    if (image_placeholders != expected_visual) {
        return fail(error,
                    error_size,
                    "image placeholder count mismatch: prompt has %u, views imply %u",
                    image_placeholders,
                    expected_visual);
    }
    if (image_placeholders == 0u && request->n_views != 0u) {
        return fail(error, error_size, "views are present but prompt has no image placeholders");
    }

    return UOCR_OK;
}
