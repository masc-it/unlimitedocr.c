#ifndef UNLIMITEDOCR_H
#define UNLIMITEDOCR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(UOCR_BUILDING_SHARED)
#    define UOCR_API __declspec(dllexport)
#  else
#    define UOCR_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define UOCR_API __attribute__((visibility("default")))
#else
#  define UOCR_API
#endif

#define UOCR_VERSION_MAJOR 0u
#define UOCR_VERSION_MINOR 1u
#define UOCR_VERSION_PATCH 0u
#define UOCR_ABI_VERSION ((UOCR_VERSION_MAJOR << 16) | (UOCR_VERSION_MINOR << 8) | UOCR_VERSION_PATCH)

typedef struct uocr_engine uocr_engine;
typedef struct uocr_result uocr_result;

typedef enum uocr_status {
    UOCR_OK = 0,
    UOCR_ERROR_INVALID_ARGUMENT = -1,
    UOCR_ERROR_UNSUPPORTED = -2,
    UOCR_ERROR_OUT_OF_MEMORY = -3,
    UOCR_ERROR_NOT_IMPLEMENTED = -4,
    UOCR_ERROR_INTERNAL = -5
} uocr_status;

typedef enum uocr_pixel_format {
    UOCR_PIXEL_F16_NCHW = 0,
    UOCR_PIXEL_F32_NCHW = 1
} uocr_pixel_format;

typedef enum uocr_view_kind {
    UOCR_VIEW_GLOBAL = 0,
    UOCR_VIEW_LOCAL = 1
} uocr_view_kind;

typedef enum uocr_memory_category {
    UOCR_MEMORY_MODEL_VIEWS = 0,
    UOCR_MEMORY_KV_CACHE = 1,
    UOCR_MEMORY_PROMPT_EMBEDDINGS = 2,
    UOCR_MEMORY_VISION_SCRATCH = 3,
    UOCR_MEMORY_DECODER_SCRATCH = 4,
    UOCR_MEMORY_MOE_SCRATCH = 5,
    UOCR_MEMORY_LOGITS_READBACK = 6,
    UOCR_MEMORY_TRANSIENT_BUFFERS = 7,
    UOCR_MEMORY_CATEGORY_COUNT = 8
} uocr_memory_category;

typedef struct uocr_memory_report {
    uint64_t category_live_bytes[UOCR_MEMORY_CATEGORY_COUNT];
    uint64_t category_peak_bytes[UOCR_MEMORY_CATEGORY_COUNT];
    uint64_t total_live_bytes;
    uint64_t total_peak_bytes;
    uint64_t estimated_model_views_bytes;
    uint64_t estimated_kv_cache_bytes;
    uint64_t estimated_prompt_embeddings_bytes;
    uint64_t estimated_vision_scratch_bytes;
    uint64_t estimated_decoder_scratch_bytes;
    uint64_t estimated_moe_scratch_bytes;
    uint64_t estimated_logits_readback_bytes;
    uint64_t estimated_transient_bytes;
    uint64_t estimated_safety_margin_bytes;
    uint64_t estimated_total_bytes;
    uint64_t memory_budget_bytes;
    uint64_t recommended_working_set_bytes;
} uocr_memory_report;

typedef struct uocr_image_view {
    const void *pixels;          /* contiguous [3,H,W], normalized to [-1,1] */
    uint32_t width;
    uint32_t height;
    uocr_pixel_format format;
    uocr_view_kind kind;
} uocr_image_view;

typedef struct uocr_prepared_request {
    const int32_t *input_ids;    /* includes BOS and image placeholder ids */
    const uint8_t *image_mask;   /* 1 where input_ids is an image placeholder */
    uint32_t n_tokens;

    const uocr_image_view *views;
    uint32_t n_views;
    uint32_t crop_grid_w;        /* 1 for non-crop/global-only */
    uint32_t crop_grid_h;

    uint32_t max_new_tokens;
    uint32_t no_repeat_ngram_size;
    uint32_t no_repeat_window;
} uocr_prepared_request;

typedef struct uocr_engine_opts {
    const char *model_path;      /* .uocr path; optional until the loader lands */
    const char *backend;         /* "auto", "metal", or "cpu-ref" */
    const char *resource_path;   /* Metal source/metallib directory; optional for now */

    uint32_t max_batch;          /* 0 = default */
    uint32_t max_prompt_tokens;  /* 0 = default */
    uint32_t max_gen_tokens;     /* 0 = default */
    uint64_t memory_budget_bytes;/* 0 = backend default */
} uocr_engine_opts;

UOCR_API uint32_t uocr_abi_version(void);
UOCR_API const char *uocr_status_string(int status);

UOCR_API uocr_engine *uocr_engine_open(const uocr_engine_opts *opts);
UOCR_API void uocr_engine_close(uocr_engine *engine);
UOCR_API const char *uocr_last_error(const uocr_engine *engine);
UOCR_API const char *uocr_engine_backend(const uocr_engine *engine);
UOCR_API const char *uocr_memory_category_name(uocr_memory_category category);
UOCR_API int uocr_engine_memory_report(const uocr_engine *engine, uocr_memory_report *out_report);

UOCR_API int uocr_generate_prepared(uocr_engine *engine,
                                    const uocr_prepared_request *requests,
                                    uint32_t n_requests,
                                    uocr_result **out_result);

UOCR_API uint32_t uocr_result_count(const uocr_result *result);
UOCR_API const int32_t *uocr_result_tokens(const uocr_result *result,
                                           uint32_t index,
                                           uint32_t *n_tokens);
UOCR_API void uocr_result_free(uocr_result *result);

#ifdef __cplusplus
}
#endif

#endif /* UNLIMITEDOCR_H */
