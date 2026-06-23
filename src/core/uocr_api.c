#include "unlimitedocr.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "core/uocr_alloc.h"
#include "model/uocr_constants.h"
#include "model/uocr_model_file.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_request_validation.h"
#include "runtime/uocr_sequence.h"

#if UOCR_HAVE_METAL
#include "backend/metal/uocr_metal.h"
#endif

#if defined(_MSC_VER)
#define UOCR_THREAD_LOCAL __declspec(thread)
#else
#define UOCR_THREAD_LOCAL _Thread_local
#endif

struct uocr_engine {
    char *model_path;
    char *backend;
    char *resource_path;
    uint32_t max_batch;
    uint32_t max_prompt_tokens;
    uint32_t max_gen_tokens;
    uint64_t memory_budget_bytes;
    uint64_t recommended_working_set_bytes;
    uint64_t model_view_bytes;
    int has_model_file;
    uocr_model_file model_file;
    uocr_memory_tracker memory_tracker;
    uocr_runtime_memory_estimate capacity_estimate;
    uocr_runtime_memory_estimate last_estimate;
#if UOCR_HAVE_METAL
    uocr_metal_context *metal;
#endif
    char last_error[512];
};

struct uocr_result {
    uint32_t n_sequences;
    int32_t **tokens;
    uint32_t *n_tokens;
};

static UOCR_THREAD_LOCAL char g_last_error[512] = "OK";

static char *uocr_strdup_or_null(const char *s) {
    if (s == NULL) {
        return NULL;
    }
    const size_t n = strlen(s) + 1u;
    char *copy = (char *)uocr_malloc(n);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, s, n);
    return copy;
}

static void set_global_error_text(const char *text) {
    if (text == NULL || text[0] == '\0') {
        text = "OK";
    }
    (void)snprintf(g_last_error, sizeof(g_last_error), "%s", text);
}

static int set_engine_errorf(uocr_engine *engine, int status, const char *fmt, ...) {
    char buffer[512];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(buffer, sizeof(buffer), fmt, ap);
    va_end(ap);
    buffer[sizeof(buffer) - 1u] = '\0';

    if (engine != NULL) {
        (void)snprintf(engine->last_error, sizeof(engine->last_error), "%s", buffer);
    }
    set_global_error_text(buffer);
    return status;
}

static void clear_engine_error(uocr_engine *engine) {
    if (engine != NULL) {
        (void)snprintf(engine->last_error, sizeof(engine->last_error), "%s", "OK");
    }
    set_global_error_text("OK");
}

static void copy_estimate_to_report(const uocr_runtime_memory_estimate *estimate, uocr_memory_report *report) {
    report->estimated_model_views_bytes = estimate->model_views_bytes;
    report->estimated_kv_cache_bytes = estimate->kv_cache_bytes;
    report->estimated_prompt_embeddings_bytes = estimate->prompt_embeddings_bytes;
    report->estimated_vision_scratch_bytes = estimate->vision_scratch_bytes;
    report->estimated_decoder_scratch_bytes = estimate->decoder_scratch_bytes;
    report->estimated_moe_scratch_bytes = estimate->moe_scratch_bytes;
    report->estimated_logits_readback_bytes = estimate->logits_readback_bytes;
    report->estimated_transient_bytes = estimate->transient_bytes;
    report->estimated_safety_margin_bytes = estimate->safety_margin_bytes;
    report->estimated_total_bytes = estimate->total_bytes;
}

static uint64_t model_tensor_data_bytes(const uocr_model_file *model) {
    const uocr_section_entry *section = uocr_model_file_find_section(model, UOCR_SECTION_TENSOR_DATA);
    return section != NULL ? section->size : 0u;
}

#if UOCR_HAVE_METAL
static int reserve_runtime_arena_accounting(uocr_engine *engine, const uocr_runtime_memory_estimate *estimate) {
    int status = uocr_memory_reserve(&engine->memory_tracker, UOCR_MEMORY_KV_CACHE, estimate->kv_cache_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_memory_reserve(&engine->memory_tracker, UOCR_MEMORY_PROMPT_EMBEDDINGS, estimate->prompt_embeddings_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_memory_reserve(&engine->memory_tracker, UOCR_MEMORY_VISION_SCRATCH, estimate->vision_scratch_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_memory_reserve(&engine->memory_tracker, UOCR_MEMORY_DECODER_SCRATCH, estimate->decoder_scratch_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_memory_reserve(&engine->memory_tracker, UOCR_MEMORY_MOE_SCRATCH, estimate->moe_scratch_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    return uocr_memory_reserve(&engine->memory_tracker, UOCR_MEMORY_LOGITS_READBACK, estimate->logits_readback_bytes);
}
#endif

static void fill_memory_report(const uocr_engine *engine, uocr_memory_report *report) {
    memset(report, 0, sizeof(*report));
    for (uint32_t i = 0u; i < (uint32_t)UOCR_MEMORY_CATEGORY_COUNT; ++i) {
        report->category_live_bytes[i] = engine->memory_tracker.categories[i].live_bytes;
        report->category_peak_bytes[i] = engine->memory_tracker.categories[i].peak_bytes;
    }
    report->total_live_bytes = engine->memory_tracker.total_live_bytes;
    report->total_peak_bytes = engine->memory_tracker.total_peak_bytes;
    copy_estimate_to_report(&engine->last_estimate, report);
    report->memory_budget_bytes = engine->memory_budget_bytes;
    report->recommended_working_set_bytes = engine->recommended_working_set_bytes;
}

uint32_t uocr_abi_version(void) {
    return UOCR_ABI_VERSION;
}

const char *uocr_status_string(int status) {
    switch (status) {
        case UOCR_OK:
            return "OK";
        case UOCR_ERROR_INVALID_ARGUMENT:
            return "invalid argument";
        case UOCR_ERROR_UNSUPPORTED:
            return "unsupported";
        case UOCR_ERROR_OUT_OF_MEMORY:
            return "out of memory";
        case UOCR_ERROR_NOT_IMPLEMENTED:
            return "not implemented";
        case UOCR_ERROR_INTERNAL:
            return "internal error";
        default:
            return "unknown status";
    }
}

const char *uocr_memory_category_name(uocr_memory_category category) {
    switch (category) {
        case UOCR_MEMORY_MODEL_VIEWS:
            return "model-views";
        case UOCR_MEMORY_KV_CACHE:
            return "kv-cache";
        case UOCR_MEMORY_PROMPT_EMBEDDINGS:
            return "prompt-embeddings";
        case UOCR_MEMORY_VISION_SCRATCH:
            return "vision-scratch";
        case UOCR_MEMORY_DECODER_SCRATCH:
            return "decoder-scratch";
        case UOCR_MEMORY_MOE_SCRATCH:
            return "moe-scratch";
        case UOCR_MEMORY_LOGITS_READBACK:
            return "logits-readback";
        case UOCR_MEMORY_TRANSIENT_BUFFERS:
            return "transient-buffers";
        default:
            return "unknown";
    }
}

static const char *default_backend(void) {
#if UOCR_HAVE_METAL
    if (uocr_metal_is_available()) {
        return "metal";
    }
#endif
#if UOCR_HAVE_CPU_REF
    return "cpu-ref";
#else
    return NULL;
#endif
}

static int backend_supported(const char *backend, char *why, size_t why_size) {
    if (strcmp(backend, "cpu-ref") == 0) {
#if UOCR_HAVE_CPU_REF
        return 1;
#else
        (void)snprintf(why, why_size, "cpu-ref backend was disabled at build time");
        return 0;
#endif
    }
    if (strcmp(backend, "metal") == 0) {
#if UOCR_HAVE_METAL
        if (uocr_metal_is_available()) {
            return 1;
        }
        (void)snprintf(why, why_size, "Metal backend is compiled but no Metal device is available");
        return 0;
#else
        (void)snprintf(why, why_size, "Metal backend was not compiled for this build");
        return 0;
#endif
    }
    (void)snprintf(why, why_size, "unsupported backend '%s'", backend);
    return 0;
}

uocr_engine *uocr_engine_open(const uocr_engine_opts *opts) {
    const char *requested_backend = NULL;
    if (opts != NULL) {
        requested_backend = opts->backend;
    }
    if (requested_backend == NULL || requested_backend[0] == '\0' || strcmp(requested_backend, "auto") == 0) {
        requested_backend = default_backend();
        if (requested_backend == NULL) {
            set_engine_errorf(NULL, UOCR_ERROR_UNSUPPORTED, "no backend is available in this build");
            return NULL;
        }
    }

    char why[256] = {0};
    if (!backend_supported(requested_backend, why, sizeof(why))) {
        set_engine_errorf(NULL, UOCR_ERROR_UNSUPPORTED, "%s", why);
        return NULL;
    }

    uocr_engine *engine = (uocr_engine *)uocr_calloc(1u, sizeof(*engine));
    if (engine == NULL) {
        set_engine_errorf(NULL, UOCR_ERROR_OUT_OF_MEMORY, "failed to allocate engine");
        return NULL;
    }

    engine->max_batch = (opts != NULL && opts->max_batch != 0u) ? opts->max_batch : UOCR_DEFAULT_MAX_BATCH;
    engine->max_prompt_tokens = (opts != NULL && opts->max_prompt_tokens != 0u) ? opts->max_prompt_tokens : UOCR_DEFAULT_MAX_PROMPT_TOKENS;
    engine->max_gen_tokens = (opts != NULL && opts->max_gen_tokens != 0u) ? opts->max_gen_tokens : UOCR_DEFAULT_MAX_GEN_TOKENS;
    engine->memory_budget_bytes = opts != NULL ? opts->memory_budget_bytes : 0u;
    engine->recommended_working_set_bytes = 0u;
#if UOCR_HAVE_METAL
    if (strcmp(requested_backend, "metal") == 0) {
        engine->recommended_working_set_bytes = uocr_metal_recommended_working_set_size();
        if (engine->memory_budget_bytes == 0u && engine->recommended_working_set_bytes != 0u) {
            engine->memory_budget_bytes = uocr_metal_default_memory_budget_bytes(engine->recommended_working_set_bytes);
        }
    }
#endif
    engine->model_view_bytes = 0u;
    engine->has_model_file = 0;
    uocr_memory_tracker_init(&engine->memory_tracker);

    if (engine->max_batch == 0u || engine->max_prompt_tokens == 0u || engine->max_gen_tokens == 0u) {
        set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "engine limits must be non-zero");
        uocr_engine_close(engine);
        return NULL;
    }

    const char *model_path = opts != NULL ? opts->model_path : NULL;
    if (model_path != NULL && model_path[0] != '\0') {
        char model_error[512];
        if (uocr_model_file_open(model_path, &engine->model_file, model_error, sizeof(model_error)) != 0) {
            set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "failed to open model file '%s': %s", model_path, model_error);
            uocr_engine_close(engine);
            return NULL;
        }
        engine->has_model_file = 1;
        engine->model_view_bytes = model_tensor_data_bytes(&engine->model_file);
        if (engine->memory_budget_bytes != 0u && engine->model_view_bytes > engine->memory_budget_bytes) {
            set_engine_errorf(engine,
                              UOCR_ERROR_OUT_OF_MEMORY,
                              "model tensor-data view estimate %llu bytes exceeds budget %llu bytes",
                              (unsigned long long)engine->model_view_bytes,
                              (unsigned long long)engine->memory_budget_bytes);
            uocr_engine_close(engine);
            return NULL;
        }
        const int reserve_status = uocr_memory_reserve(&engine->memory_tracker, UOCR_MEMORY_MODEL_VIEWS, engine->model_view_bytes);
        if (reserve_status != UOCR_OK) {
            set_engine_errorf(engine, reserve_status, "failed to account model tensor-data view bytes");
            uocr_engine_close(engine);
            return NULL;
        }
    }

    const int estimate_status = uocr_estimate_minimal_runtime_memory(engine->max_batch,
                                                                     engine->max_prompt_tokens,
                                                                     engine->model_view_bytes,
                                                                     &engine->capacity_estimate);
    if (estimate_status != UOCR_OK) {
        set_engine_errorf(engine, estimate_status, "failed to estimate engine memory requirements");
        uocr_engine_close(engine);
        return NULL;
    }
    engine->last_estimate = engine->capacity_estimate;

    engine->backend = uocr_strdup_or_null(requested_backend);
    engine->model_path = uocr_strdup_or_null(model_path);
    engine->resource_path = uocr_strdup_or_null(opts != NULL ? opts->resource_path : NULL);
    if (engine->backend == NULL || (opts != NULL && opts->model_path != NULL && engine->model_path == NULL) ||
        (opts != NULL && opts->resource_path != NULL && engine->resource_path == NULL)) {
        set_engine_errorf(engine, UOCR_ERROR_OUT_OF_MEMORY, "failed to copy engine option strings");
        uocr_engine_close(engine);
        return NULL;
    }

#if UOCR_HAVE_METAL
    if (strcmp(engine->backend, "metal") == 0) {
        if (engine->memory_budget_bytes != 0u && engine->capacity_estimate.total_bytes > engine->memory_budget_bytes) {
            set_engine_errorf(engine,
                              UOCR_ERROR_OUT_OF_MEMORY,
                              "engine runtime memory estimate %llu bytes exceeds Metal budget %llu bytes",
                              (unsigned long long)engine->capacity_estimate.total_bytes,
                              (unsigned long long)engine->memory_budget_bytes);
            uocr_engine_close(engine);
            return NULL;
        }

        char metal_error[512];
        engine->metal = uocr_metal_context_create(engine->resource_path, metal_error, sizeof(metal_error));
        if (engine->metal == NULL) {
            set_engine_errorf(engine, UOCR_ERROR_INTERNAL, "failed to initialize Metal backend: %s", metal_error);
            uocr_engine_close(engine);
            return NULL;
        }
        if (engine->has_model_file && !uocr_metal_context_map_model(engine->metal, &engine->model_file, metal_error, sizeof(metal_error))) {
            set_engine_errorf(engine, UOCR_ERROR_INTERNAL, "failed to map model into Metal no-copy views: %s", metal_error);
            uocr_engine_close(engine);
            return NULL;
        }
        if (!uocr_metal_context_allocate_runtime_arenas(engine->metal,
                                                        engine->max_batch,
                                                        engine->max_prompt_tokens,
                                                        metal_error,
                                                        sizeof(metal_error))) {
            set_engine_errorf(engine, UOCR_ERROR_OUT_OF_MEMORY, "failed to allocate Metal runtime arenas: %s", metal_error);
            uocr_engine_close(engine);
            return NULL;
        }
        const int arena_reserve_status = reserve_runtime_arena_accounting(engine, &engine->capacity_estimate);
        if (arena_reserve_status != UOCR_OK) {
            set_engine_errorf(engine, arena_reserve_status, "failed to account Metal runtime arena bytes");
            uocr_engine_close(engine);
            return NULL;
        }
    }
#endif

    clear_engine_error(engine);
    return engine;
}

void uocr_engine_close(uocr_engine *engine) {
    if (engine == NULL) {
        return;
    }
#if UOCR_HAVE_METAL
    uocr_metal_context_destroy(engine->metal);
#endif
    if (engine->has_model_file) {
        uocr_model_file_close(&engine->model_file);
    }
    uocr_free(engine->model_path);
    uocr_free(engine->backend);
    uocr_free(engine->resource_path);
    uocr_free(engine);
}

const char *uocr_last_error(const uocr_engine *engine) {
    if (engine != NULL) {
        return engine->last_error[0] != '\0' ? engine->last_error : "OK";
    }
    return g_last_error[0] != '\0' ? g_last_error : "OK";
}

const char *uocr_engine_backend(const uocr_engine *engine) {
    if (engine == NULL) {
        set_engine_errorf(NULL, UOCR_ERROR_INVALID_ARGUMENT, "engine is null");
        return NULL;
    }
    return engine->backend;
}

int uocr_engine_memory_report(const uocr_engine *engine, uocr_memory_report *out_report) {
    if (engine == NULL || out_report == NULL) {
        return set_engine_errorf(NULL, UOCR_ERROR_INVALID_ARGUMENT, "engine and out_report must be non-null");
    }
    fill_memory_report(engine, out_report);
    return UOCR_OK;
}

static int allocate_empty_result(uint32_t n_requests, uocr_result **out_result) {
    uocr_result *result = (uocr_result *)uocr_calloc(1u, sizeof(*result));
    if (result == NULL) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    result->n_sequences = n_requests;
    result->tokens = (int32_t **)uocr_calloc(n_requests, sizeof(result->tokens[0]));
    result->n_tokens = (uint32_t *)uocr_calloc(n_requests, sizeof(result->n_tokens[0]));
    if (result->tokens == NULL || result->n_tokens == NULL) {
        uocr_result_free(result);
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    *out_result = result;
    return UOCR_OK;
}

int uocr_generate_prepared(uocr_engine *engine,
                           const uocr_prepared_request *requests,
                           uint32_t n_requests,
                           uocr_result **out_result) {
    if (out_result != NULL) {
        *out_result = NULL;
    }
    if (engine == NULL) {
        return set_engine_errorf(NULL, UOCR_ERROR_INVALID_ARGUMENT, "engine is null");
    }
    if (out_result == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "out_result pointer is null");
    }
    if (requests == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "requests pointer is null");
    }
    if (n_requests == 0u) {
        return set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "n_requests must be non-zero");
    }
    if (n_requests > engine->max_batch) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INVALID_ARGUMENT,
                                 "batch size %u exceeds engine limit %u",
                                 n_requests,
                                 engine->max_batch);
    }

    const uocr_request_limits limits = {engine->max_prompt_tokens, engine->max_gen_tokens};
    uint32_t max_prompt_tokens_in_batch = 0u;
    int any_generation_requested = 0;
    for (uint32_t i = 0u; i < n_requests; ++i) {
        char validation_error[512];
        const int status = uocr_validate_prepared_request(&requests[i], &limits, validation_error, sizeof(validation_error));
        if (status != UOCR_OK) {
            return set_engine_errorf(engine, status, "request %u invalid: %s", i, validation_error);
        }
        uocr_sequence_state sequence_state;
        const int state_status = uocr_build_sequence_state(&requests[i], &sequence_state, validation_error, sizeof(validation_error));
        if (state_status != UOCR_OK) {
            return set_engine_errorf(engine, state_status, "request %u state build failed: %s", i, validation_error);
        }
        (void)sequence_state; /* retained here to exercise state construction before inference kernels land */
        if (requests[i].n_tokens > max_prompt_tokens_in_batch) {
            max_prompt_tokens_in_batch = requests[i].n_tokens;
        }
        if (requests[i].max_new_tokens > 0u) {
            any_generation_requested = 1;
        }
    }

    uocr_runtime_memory_estimate request_estimate;
    const int estimate_status = uocr_estimate_minimal_runtime_memory(n_requests,
                                                                     max_prompt_tokens_in_batch,
                                                                     engine->model_view_bytes,
                                                                     &request_estimate);
    if (estimate_status != UOCR_OK) {
        return set_engine_errorf(engine, estimate_status, "failed to estimate request memory requirements");
    }
    engine->last_estimate = request_estimate;
    if (engine->memory_budget_bytes != 0u && request_estimate.total_bytes > engine->memory_budget_bytes) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_OUT_OF_MEMORY,
                                 "request memory estimate %llu bytes exceeds budget %llu bytes",
                                 (unsigned long long)request_estimate.total_bytes,
                                 (unsigned long long)engine->memory_budget_bytes);
    }

    if (any_generation_requested) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "inference kernels are not implemented yet; use max_new_tokens=0 for ABI smoke tests");
    }

    const int status = allocate_empty_result(n_requests, out_result);
    if (status != UOCR_OK) {
        return set_engine_errorf(engine, status, "failed to allocate result");
    }
    clear_engine_error(engine);
    return UOCR_OK;
}

uint32_t uocr_result_count(const uocr_result *result) {
    return result != NULL ? result->n_sequences : 0u;
}

const int32_t *uocr_result_tokens(const uocr_result *result, uint32_t index, uint32_t *n_tokens) {
    if (n_tokens != NULL) {
        *n_tokens = 0u;
    }
    if (result == NULL || index >= result->n_sequences) {
        return NULL;
    }
    if (n_tokens != NULL) {
        *n_tokens = result->n_tokens[index];
    }
    return result->tokens[index];
}

void uocr_result_free(uocr_result *result) {
    if (result == NULL) {
        return;
    }
    if (result->tokens != NULL) {
        for (uint32_t i = 0u; i < result->n_sequences; ++i) {
            uocr_free(result->tokens[i]);
        }
    }
    uocr_free(result->tokens);
    uocr_free(result->n_tokens);
    uocr_free(result);
}
