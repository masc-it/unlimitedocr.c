#include "unlimitedocr.h"

#include "model/uocr_constants.h"
#include "model/uocr_format.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_vision.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static uint64_t align_up_u64(uint64_t value, uint64_t align) {
    const uint64_t rem = value % align;
    return rem == 0u ? value : value + (align - rem);
}

static void fill_hash(uint8_t hash[32], uint8_t seed) {
    for (uint32_t i = 0u; i < 32u; ++i) {
        hash[i] = (uint8_t)(seed + i);
    }
}

static int make_temp_path(char *path, size_t path_size) {
    const char *template_path = "/tmp/uocr-api-model-XXXXXX";
    if (strlen(template_path) + 1u > path_size) {
        return 1;
    }
    strcpy(path, template_path);
    int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    close(fd);
    return 0;
}

static int write_tiny_uocr_model(const char *path) {
    const uint64_t section_dir_offset = sizeof(uocr_file_header);
    const uint32_t section_count = 5u;
    const uint64_t config_offset = section_dir_offset + section_count * sizeof(uocr_section_entry);
    const uint64_t tokenizer_offset = align_up_u64(config_offset + sizeof(uocr_config_record), 8u);
    const uint64_t provenance_offset = align_up_u64(tokenizer_offset + sizeof(uocr_tokenizer_metadata_record), 8u);
    const uint64_t tensor_dir_offset = align_up_u64(provenance_offset + sizeof(uocr_provenance_record), 8u);
    const uint64_t tensor_dir_size = sizeof(uocr_tensor_directory_header) + sizeof(uocr_tensor_entry);
    const uint64_t tensor_data_offset = align_up_u64(tensor_dir_offset + tensor_dir_size, UOCR_TENSOR_DATA_ALIGNMENT);
    const uint64_t tensor_data_size = 16u;
    const uint64_t file_size = tensor_data_offset + tensor_data_size;

    uint8_t *buffer = (uint8_t *)calloc(1u, (size_t)file_size);
    if (buffer == NULL) {
        return 1;
    }

    uocr_file_header *header = (uocr_file_header *)(void *)buffer;
    memcpy(header->magic, UOCR_FILE_MAGIC, 4u);
    header->version = UOCR_FORMAT_VERSION;
    header->header_size = (uint32_t)sizeof(uocr_file_header);
    header->endian_marker = UOCR_ENDIAN_MARKER;
    header->required_alignment = UOCR_TENSOR_DATA_ALIGNMENT;
    header->qprofile = UOCR_QPROFILE_FP16;
    header->section_count = section_count;
    header->file_size = file_size;
    header->section_dir_offset = section_dir_offset;
    fill_hash(header->model_id_hash, 0x80u);

    uocr_section_entry *sections = (uocr_section_entry *)(void *)(buffer + section_dir_offset);
    sections[0].type = UOCR_SECTION_CONFIG;
    sections[0].offset = config_offset;
    sections[0].size = sizeof(uocr_config_record);
    sections[0].alignment = 8u;
    sections[1].type = UOCR_SECTION_TOKENIZER_METADATA;
    sections[1].offset = tokenizer_offset;
    sections[1].size = sizeof(uocr_tokenizer_metadata_record);
    sections[1].alignment = 8u;
    sections[2].type = UOCR_SECTION_PROVENANCE;
    sections[2].offset = provenance_offset;
    sections[2].size = sizeof(uocr_provenance_record);
    sections[2].alignment = 8u;
    sections[3].type = UOCR_SECTION_TENSOR_DIRECTORY;
    sections[3].offset = tensor_dir_offset;
    sections[3].size = tensor_dir_size;
    sections[3].alignment = 8u;
    sections[4].type = UOCR_SECTION_TENSOR_DATA;
    sections[4].offset = tensor_data_offset;
    sections[4].size = tensor_data_size;
    sections[4].alignment = UOCR_TENSOR_DATA_ALIGNMENT;

    uocr_config_record config = uocr_default_config_record();
    memcpy(buffer + config_offset, &config, sizeof(config));

    uocr_tokenizer_metadata_record *tokenizer = (uocr_tokenizer_metadata_record *)(void *)(buffer + tokenizer_offset);
    tokenizer->magic = UOCR_TOKENIZER_METADATA_MAGIC;
    tokenizer->version = UOCR_FORMAT_VERSION;
    tokenizer->record_size = (uint32_t)sizeof(uocr_tokenizer_metadata_record);
    tokenizer->flags = UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED;
    fill_hash(tokenizer->tokenizer_hash, 0x10u);
    tokenizer->vocab_size = UOCR_VOCAB_SIZE;
    tokenizer->bpe_vocab_size = UOCR_BPE_VOCAB_SIZE;
    tokenizer->added_token_count = UOCR_ADDED_TOKEN_COUNT;
    tokenizer->bos_token_id = (uint32_t)UOCR_TOKEN_BOS;
    tokenizer->eos_token_id = (uint32_t)UOCR_TOKEN_EOS;
    tokenizer->pad_token_id = (uint32_t)UOCR_TOKEN_PAD;
    tokenizer->image_token_id = (uint32_t)UOCR_TOKEN_IMAGE;

    uocr_provenance_record *provenance = (uocr_provenance_record *)(void *)(buffer + provenance_offset);
    provenance->magic = UOCR_PROVENANCE_MAGIC;
    provenance->version = UOCR_FORMAT_VERSION;
    provenance->record_size = (uint32_t)sizeof(uocr_provenance_record);
    provenance->source_tensor_count = 1u;
    provenance->runtime_tensor_count = 1u;
    provenance->preserved_unused_tensor_count = 0u;
    provenance->omitted_tensor_count = 0u;
    provenance->safetensors_file_count = 1u;
    provenance->qprofile = UOCR_QPROFILE_FP16;
    fill_hash(provenance->config_hash, 0x30u);
    fill_hash(provenance->tokenizer_hash, 0x10u);
    fill_hash(provenance->safetensors_index_hash, 0x50u);

    uocr_tensor_directory_header *dir = (uocr_tensor_directory_header *)(void *)(buffer + tensor_dir_offset);
    dir->magic = UOCR_TENSOR_DIR_MAGIC;
    dir->version = UOCR_FORMAT_VERSION;
    dir->entry_size = (uint32_t)sizeof(uocr_tensor_entry);
    dir->tensor_count = 1u;

    uocr_tensor_entry *tensor = (uocr_tensor_entry *)(void *)(buffer + tensor_dir_offset + sizeof(*dir));
    tensor->id = 1u;
    tensor->family = UOCR_TENSOR_FAMILY_TOK_EMBED;
    tensor->layer = -1;
    tensor->expert = -1;
    tensor->projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensor->usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensor->qtype = UOCR_TENSOR_F16;
    tensor->rank = 1u;
    tensor->logical_shape[0] = 8u;
    tensor->physical_shape[0] = 8u;
    tensor->payload_offset = tensor_data_offset;
    tensor->payload_size = tensor_data_size;

    for (uint32_t i = 0u; i < tensor_data_size; ++i) {
        buffer[tensor_data_offset + i] = (uint8_t)i;
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(buffer);
        perror("fopen");
        return 1;
    }
    int failed = fwrite(buffer, 1u, (size_t)file_size, f) != (size_t)file_size;
    failed |= fclose(f) != 0;
    free(buffer);
    return failed ? 1 : 0;
}

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

static int test_engine_memory_report(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.memory_budget_bytes == 0u);
    CHECK(report.recommended_working_set_bytes == 0u);
    CHECK(report.total_live_bytes == 0u);
    CHECK(report.estimated_kv_cache_bytes > 0u);
    CHECK(report.estimated_prompt_embeddings_bytes == 16u * 1280u * 2u);
    CHECK(report.estimated_vision_scratch_bytes == 0u);
    CHECK(report.estimated_decoder_scratch_bytes > 0u);
    CHECK(report.estimated_moe_scratch_bytes > 0u);
    CHECK(report.estimated_logits_readback_bytes == 129280u * 4u + 4u);
    CHECK(report.estimated_safety_margin_bytes > 0u);
    CHECK(report.estimated_total_bytes == report.estimated_kv_cache_bytes +
                                            report.estimated_prompt_embeddings_bytes +
                                            report.estimated_vision_scratch_bytes +
                                            report.estimated_decoder_scratch_bytes +
                                            report.estimated_moe_scratch_bytes +
                                            report.estimated_logits_readback_bytes +
                                            report.estimated_safety_margin_bytes);
    CHECK(strcmp(uocr_memory_category_name(UOCR_MEMORY_KV_CACHE), "kv-cache") == 0);

    uocr_engine_close(engine);
    return 0;
}

static int test_engine_loads_model_memory_report(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_tiny_uocr_model(path) == 0);

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = path;
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);

    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.category_live_bytes[UOCR_MEMORY_MODEL_VIEWS] == 16u);
    CHECK(report.category_peak_bytes[UOCR_MEMORY_MODEL_VIEWS] == 16u);
    CHECK(report.total_live_bytes == 16u);
    CHECK(report.estimated_model_views_bytes == 16u);
    CHECK(report.estimated_total_bytes == report.estimated_model_views_bytes +
                                            report.estimated_kv_cache_bytes +
                                            report.estimated_prompt_embeddings_bytes +
                                            report.estimated_vision_scratch_bytes +
                                            report.estimated_decoder_scratch_bytes +
                                            report.estimated_moe_scratch_bytes +
                                            report.estimated_logits_readback_bytes +
                                            report.estimated_safety_margin_bytes);

    uocr_engine_close(engine);
    unlink(path);
    return 0;
}

static int test_engine_rejects_model_over_budget(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_tiny_uocr_model(path) == 0);

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = path;
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;
    opts.memory_budget_bytes = 8u;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine == NULL);
    const char *error = uocr_last_error(NULL);
    CHECK(strstr(error, "engine admission rejected") != NULL);
    CHECK(strstr(error, "exceeds budget") != NULL);
    CHECK(strstr(error, "model=") != NULL);
    CHECK(strstr(error, "kv=") != NULL);
    CHECK(strstr(error, "prompt=") != NULL);
    CHECK(strstr(error, "vision=") != NULL);
    CHECK(strstr(error, "decoder=") != NULL);
    CHECK(strstr(error, "moe=") != NULL);
    CHECK(strstr(error, "logits=") != NULL);
    CHECK(strstr(error, "transient=") != NULL);
    CHECK(strstr(error, "safety=") != NULL);
    unlink(path);
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

static int test_memory_budget_rejects_request(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;
    opts.memory_budget_bytes = 1u;

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
    const int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_ERROR_OUT_OF_MEMORY);
    CHECK(result == NULL);
    const char *error = uocr_last_error(engine);
    CHECK(strstr(error, "request admission rejected") != NULL);
    CHECK(strstr(error, "exceeds budget") != NULL);
    CHECK(strstr(error, "model=") != NULL);
    CHECK(strstr(error, "kv=") != NULL);
    CHECK(strstr(error, "prompt=") != NULL);
    CHECK(strstr(error, "vision=") != NULL);
    CHECK(strstr(error, "decoder=") != NULL);
    CHECK(strstr(error, "moe=") != NULL);
    CHECK(strstr(error, "logits=") != NULL);
    CHECK(strstr(error, "safety=") != NULL);

    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.memory_budget_bytes == 1u);
    CHECK(report.estimated_total_bytes > report.memory_budget_bytes);

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

    uocr_vision_schedule schedule;
    char schedule_error[256];
    memset(&schedule, 0, sizeof(schedule));
    memset(schedule_error, 0, sizeof(schedule_error));
    CHECK(uocr_plan_vision_schedule(&req,
                                    1u,
                                    NULL,
                                    0u,
                                    &schedule,
                                    schedule_error,
                                    sizeof(schedule_error)) == UOCR_OK);
    uint64_t expected_vision = 0u;
    CHECK(uocr_estimate_vision_scratch_bytes_for_rows(schedule.final_visual_tokens,
                                                       schedule.max_chunk_projected_tokens,
                                                       &expected_vision) == UOCR_OK);
    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.estimated_vision_scratch_bytes == expected_vision);
    CHECK(report.estimated_vision_scratch_bytes > 0u);
    uint64_t global_vision = 0u;
    CHECK(uocr_estimate_vision_scratch_bytes(&global_vision) == UOCR_OK);
    CHECK(report.estimated_vision_scratch_bytes > global_vision);

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

static int test_request_validation_accepts_upstream_no_repeat_defaults(void) {
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
    req.no_repeat_ngram_size = 35;
    req.no_repeat_window = 128;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_OK);
    CHECK(result != NULL);
    uocr_result_free(result);
    uocr_engine_close(engine);
    return 0;
}

static int test_request_validation_rejects_bad_no_repeat_config(void) {
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
    req.no_repeat_ngram_size = 0;
    req.no_repeat_window = 128;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(strstr(uocr_last_error(engine), "no_repeat_window") != NULL);

    uocr_engine_close(engine);
    return 0;
}

static int test_request_validation_rejects_sequence_position_overflow(void) {
    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "cpu-ref";
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = UOCR_MAX_POSITIONS;

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
    req.max_new_tokens = UOCR_MAX_POSITIONS - 1u;

    uocr_result *result = NULL;
    int status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(strstr(uocr_last_error(engine), "position budget") != NULL);

    req.max_new_tokens = UOCR_MAX_POSITIONS - 2u;
    status = uocr_generate_prepared(engine, &req, 1, &result);
    CHECK(status == UOCR_ERROR_NOT_IMPLEMENTED);
    CHECK(result == NULL);
    CHECK(strstr(uocr_last_error(engine), "inference kernels") != NULL);

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
    if (test_engine_memory_report() != 0) return 1;
    if (test_engine_loads_model_memory_report() != 0) return 1;
    if (test_engine_rejects_model_over_budget() != 0) return 1;
    if (test_empty_generation_smoke() != 0) return 1;
    if (test_memory_budget_rejects_request() != 0) return 1;
    if (test_generation_not_implemented() != 0) return 1;
    if (test_global_image_request_validation() != 0) return 1;
    if (test_crop_image_request_validation() != 0) return 1;
    if (test_request_validation_rejects_bad_visual_count() != 0) return 1;
    if (test_request_validation_accepts_upstream_no_repeat_defaults() != 0) return 1;
    if (test_request_validation_rejects_bad_no_repeat_config() != 0) return 1;
    if (test_request_validation_rejects_sequence_position_overflow() != 0) return 1;
    if (test_request_validation_rejects_bad_bos() != 0) return 1;
    return 0;
}
