#include "unlimitedocr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static int test_engine_open_close(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);
    CHECK(strcmp(uocr_engine_backend(engine), "cpu-ref") == 0);
    uocr_engine_close(engine);
    return 0;
}

static int test_empty_generation_smoke(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    const int32_t input_ids[] = {0, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request req;
    memset(&req, 0, sizeof(req));
    req.input_ids = input_ids;
    req.image_mask = image_mask;
    req.n_tokens = 2;
    req.crop_grid_w = 1;
    req.crop_grid_h = 1;
    req.max_new_tokens = 0;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_OK);
    CHECK(result != NULL);
    CHECK(uocr_result_count(result) == 1);
    uint32_t n_tokens = 123;
    const int32_t *tokens = uocr_result_tokens(result, 0, &n_tokens);
    CHECK(tokens == NULL);
    CHECK(n_tokens == 0);
    uocr_result_free(result);
    uocr_engine_close(engine);
    return 0;
}

static int test_generation_not_implemented(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    const int32_t input_ids[] = {0, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request req;
    memset(&req, 0, sizeof(req));
    req.input_ids = input_ids;
    req.image_mask = image_mask;
    req.n_tokens = 2;
    req.crop_grid_w = 1;
    req.crop_grid_h = 1;
    req.max_new_tokens = 1;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_ERROR_NOT_IMPLEMENTED);
    CHECK(result == NULL);
    CHECK(strstr(uocr_last_error(engine), "inference kernels") != NULL);

    uocr_engine_close(engine);
    return 0;
}

static int test_global_image_request_validation(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 512;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    enum { N_TOKENS = 1 + 273 };
    int32_t *input_ids = (int32_t *)calloc(N_TOKENS, sizeof(*input_ids));
    uint8_t *image_mask = (uint8_t *)calloc(N_TOKENS, sizeof(*image_mask));
    CHECK(input_ids != NULL);
    CHECK(image_mask != NULL);
    input_ids[0] = 0;
    for (uint32_t i = 1; i < N_TOKENS; ++i) {
        input_ids[i] = 128815;
        image_mask[i] = 1;
    }

    static const float pixel_sentinel = 0.0f;
    uocr_image_view view;
    memset(&view, 0, sizeof(view));
    view.pixels = &pixel_sentinel;
    view.width = 1024;
    view.height = 1024;
    view.format = UOCR_PIXEL_F32_NCHW;
    view.kind = UOCR_VIEW_GLOBAL;

    uocr_prepared_request req;
    memset(&req, 0, sizeof(req));
    req.input_ids = input_ids;
    req.image_mask = image_mask;
    req.n_tokens = N_TOKENS;
    req.views = &view;
    req.n_views = 1;
    req.crop_grid_w = 1;
    req.crop_grid_h = 1;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_OK);
    CHECK(result != NULL);
    uocr_result_free(result);
    free(input_ids);
    free(image_mask);
    uocr_engine_close(engine);
    return 0;
}

static int test_crop_image_request_validation(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 1024;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    enum { VISUAL_TOKENS = 483, N_TOKENS = 1 + VISUAL_TOKENS };
    int32_t *input_ids = (int32_t *)calloc(N_TOKENS, sizeof(*input_ids));
    uint8_t *image_mask = (uint8_t *)calloc(N_TOKENS, sizeof(*image_mask));
    CHECK(input_ids != NULL);
    CHECK(image_mask != NULL);
    input_ids[0] = 0;
    for (uint32_t i = 1; i < N_TOKENS; ++i) {
        input_ids[i] = 128815;
        image_mask[i] = 1;
    }

    static const float pixel_sentinel = 0.0f;
    uocr_image_view views[3];
    memset(views, 0, sizeof(views));
    for (uint32_t i = 0; i < 2; ++i) {
        views[i].pixels = &pixel_sentinel;
        views[i].width = 640;
        views[i].height = 640;
        views[i].format = UOCR_PIXEL_F32_NCHW;
        views[i].kind = UOCR_VIEW_LOCAL;
    }
    views[2].pixels = &pixel_sentinel;
    views[2].width = 1024;
    views[2].height = 1024;
    views[2].format = UOCR_PIXEL_F32_NCHW;
    views[2].kind = UOCR_VIEW_GLOBAL;

    uocr_prepared_request req;
    memset(&req, 0, sizeof(req));
    req.input_ids = input_ids;
    req.image_mask = image_mask;
    req.n_tokens = N_TOKENS;
    req.views = views;
    req.n_views = 3;
    req.crop_grid_w = 2;
    req.crop_grid_h = 1;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_OK);
    CHECK(result != NULL);
    uocr_result_free(result);
    free(input_ids);
    free(image_mask);
    uocr_engine_close(engine);
    return 0;
}

static int test_request_validation_rejects_bad_visual_count(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 512;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    enum { N_TOKENS = 1 + 272 };
    int32_t *input_ids = (int32_t *)calloc(N_TOKENS, sizeof(*input_ids));
    uint8_t *image_mask = (uint8_t *)calloc(N_TOKENS, sizeof(*image_mask));
    CHECK(input_ids != NULL);
    CHECK(image_mask != NULL);
    input_ids[0] = 0;
    for (uint32_t i = 1; i < N_TOKENS; ++i) {
        input_ids[i] = 128815;
        image_mask[i] = 1;
    }

    static const float pixel_sentinel = 0.0f;
    uocr_image_view view;
    memset(&view, 0, sizeof(view));
    view.pixels = &pixel_sentinel;
    view.width = 1024;
    view.height = 1024;
    view.format = UOCR_PIXEL_F32_NCHW;
    view.kind = UOCR_VIEW_GLOBAL;

    uocr_prepared_request req;
    memset(&req, 0, sizeof(req));
    req.input_ids = input_ids;
    req.image_mask = image_mask;
    req.n_tokens = N_TOKENS;
    req.views = &view;
    req.n_views = 1;
    req.crop_grid_w = 1;
    req.crop_grid_h = 1;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(strstr(uocr_last_error(engine), "placeholder count mismatch") != NULL);

    free(input_ids);
    free(image_mask);
    uocr_engine_close(engine);
    return 0;
}

static int test_request_validation_rejects_bad_bos(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    const int32_t input_ids[] = {2, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request req;
    memset(&req, 0, sizeof(req));
    req.input_ids = input_ids;
    req.image_mask = image_mask;
    req.n_tokens = 2;
    req.crop_grid_w = 1;
    req.crop_grid_h = 1;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(strstr(uocr_last_error(engine), "BOS") != NULL);

    uocr_engine_close(engine);
    return 0;
}

int main(void) {
    CHECK(uocr_abi_version() == UOCR_ABI_VERSION);
    CHECK(strcmp(uocr_status_string(UOCR_OK), "OK") == 0);

    if (test_engine_open_close() != 0) return 1;
    if (test_empty_generation_smoke() != 0) return 1;
    if (test_generation_not_implemented() != 0) return 1;
    if (test_global_image_request_validation() != 0) return 1;
    if (test_crop_image_request_validation() != 0) return 1;
    if (test_request_validation_rejects_bad_visual_count() != 0) return 1;
    if (test_request_validation_rejects_bad_bos() != 0) return 1;
    return 0;
}
