#include "unlimitedocr.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "core/uocr_alloc.h"
#include "core/uocr_result.h"
#include "model/uocr_constants.h"
#include "model/uocr_model_file.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_profile.h"
#include "runtime/uocr_request_validation.h"
#include "runtime/uocr_sequence.h"
#include "runtime/uocr_vision.h"

#if UOCR_HAVE_METAL
#include "backend/metal/uocr_metal.h"
#endif

#if defined(_MSC_VER)
#define UOCR_THREAD_LOCAL __declspec(thread)
#else
#define UOCR_THREAD_LOCAL _Thread_local
#endif

#define UOCR_ERROR_TEXT_SIZE 1024u

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
    uocr_runtime_memory_estimate metal_runtime_arena_estimate;
    int has_metal_runtime_arena_accounting;
#endif
    uocr_profile_state profile;
#if UOCR_HAVE_METAL
    uocr_metal_context *metal;
#endif
    char last_error[UOCR_ERROR_TEXT_SIZE];
};

static UOCR_THREAD_LOCAL char g_last_error[UOCR_ERROR_TEXT_SIZE] = "OK";

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
    char buffer[UOCR_ERROR_TEXT_SIZE];
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

typedef struct uocr_admission_request_shape {
    uint32_t request_count;
    uint64_t requested_views;
    uint32_t prompt_tokens;
    uint32_t max_new_tokens;
    uint32_t visual_rows;
    uint32_t max_chunk_projected_rows;
    uint32_t max_view_size;
} uocr_admission_request_shape;

static uint64_t saturated_add_u64(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static void copy_estimate_to_report(const uocr_runtime_memory_estimate *estimate, uocr_memory_report *report) {
    report->estimated_model_views_bytes = estimate->model_views_bytes;
    report->estimated_kv_cache_bytes = estimate->kv_cache_bytes;
    report->estimated_prompt_embeddings_bytes = estimate->prompt_embeddings_bytes;
    report->estimated_vision_scratch_bytes = estimate->vision_scratch_bytes;
    report->estimated_vision_gpu_workspace_bytes = estimate->vision_gpu_workspace_bytes;
    report->estimated_vision_final_features_bytes = estimate->vision_final_features_bytes;
    report->estimated_vision_host_staging_bytes = estimate->vision_host_staging_bytes;
    report->estimated_decoder_scratch_bytes = estimate->decoder_scratch_bytes;
    report->estimated_moe_scratch_bytes = estimate->moe_scratch_bytes;
    report->estimated_logits_readback_bytes = estimate->logits_readback_bytes;
    report->estimated_transient_bytes = estimate->transient_bytes;
    report->estimated_safety_margin_bytes = estimate->safety_margin_bytes;
    report->estimated_total_bytes = estimate->total_bytes;
}

static uint32_t current_metal_arena_batch_slots(const uocr_engine *engine) {
#if UOCR_HAVE_METAL
    return (engine != NULL && engine->metal != NULL) ?
               uocr_metal_context_runtime_arena_batch_slots(engine->metal) :
               0u;
#else
    (void)engine;
    return 0u;
#endif
}

static uint32_t current_metal_arena_prompt_token_capacity(const uocr_engine *engine) {
#if UOCR_HAVE_METAL
    return (engine != NULL && engine->metal != NULL) ?
               uocr_metal_context_runtime_arena_prompt_token_capacity(engine->metal) :
               0u;
#else
    (void)engine;
    return 0u;
#endif
}

static uint64_t current_metal_arena_bytes(const uocr_engine *engine) {
#if UOCR_HAVE_METAL
    return (engine != NULL && engine->metal != NULL) ?
               uocr_metal_context_total_runtime_arena_capacity(engine->metal) :
               0u;
#else
    (void)engine;
    return 0u;
#endif
}

static int set_admission_error_with_shape(uocr_engine *engine,
                                           const char *scope,
                                           const uocr_runtime_memory_estimate *estimate,
                                           uint64_t budget_bytes,
                                           const uocr_admission_request_shape *shape) {
    if (scope == NULL || scope[0] == '\0') {
        scope = "request";
    }
    const uint32_t configured_max_batch = engine != NULL ? engine->max_batch : 0u;
    const uint32_t configured_max_prompt_tokens = engine != NULL ? engine->max_prompt_tokens : 0u;
    const uint32_t configured_max_gen_tokens = engine != NULL ? engine->max_gen_tokens : 0u;
    const uint32_t arena_batch = current_metal_arena_batch_slots(engine);
    const uint32_t arena_prompt = current_metal_arena_prompt_token_capacity(engine);
    const uint64_t arena_bytes = current_metal_arena_bytes(engine);
    if (estimate == NULL) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_OUT_OF_MEMORY,
                                 "%s admission rejected: memory estimate exceeds budget %llu bytes "
                                 "(configured_max_batch=%u configured_max_prompt_tokens=%u configured_max_gen_tokens=%u "
                                 "current_metal_arena_batch_slots=%u current_metal_arena_prompt_token_capacity=%u "
                                 "current_metal_arena_bytes=%llu)",
                                 scope,
                                 (unsigned long long)budget_bytes,
                                 configured_max_batch,
                                 configured_max_prompt_tokens,
                                 configured_max_gen_tokens,
                                 arena_batch,
                                 arena_prompt,
                                 (unsigned long long)arena_bytes);
    }
    const uint64_t vision_workspace_bytes = saturated_add_u64(estimate->vision_gpu_workspace_bytes,
                                                              estimate->vision_final_features_bytes);
    if (shape != NULL) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_OUT_OF_MEMORY,
                                 "%s admission rejected: memory estimate %llu bytes exceeds budget %llu bytes "
                                 "(configured_budget=%llu requests=%u requested_views=%llu prompt_tokens=%u max_new_tokens=%u "
                                 "visual_rows=%u max_view_size=%u max_chunk_projected_rows=%u configured_max_batch=%u "
                                 "configured_max_prompt_tokens=%u configured_max_gen_tokens=%u current_metal_arena_batch_slots=%u "
                                 "current_metal_arena_prompt_token_capacity=%u current_metal_arena_bytes=%llu vision_workspace=%llu "
                                 "vision=%llu vision_gpu_workspace=%llu vision_final_features=%llu vision_host_staging=%llu "
                                 "model=%llu kv=%llu prompt=%llu decoder=%llu moe=%llu logits=%llu transient=%llu safety=%llu)",
                                 scope,
                                 (unsigned long long)estimate->total_bytes,
                                 (unsigned long long)budget_bytes,
                                 (unsigned long long)budget_bytes,
                                 shape->request_count,
                                 (unsigned long long)shape->requested_views,
                                 shape->prompt_tokens,
                                 shape->max_new_tokens,
                                 shape->visual_rows,
                                 shape->max_view_size,
                                 shape->max_chunk_projected_rows,
                                 configured_max_batch,
                                 configured_max_prompt_tokens,
                                 configured_max_gen_tokens,
                                 arena_batch,
                                 arena_prompt,
                                 (unsigned long long)arena_bytes,
                                 (unsigned long long)vision_workspace_bytes,
                                 (unsigned long long)estimate->vision_scratch_bytes,
                                 (unsigned long long)estimate->vision_gpu_workspace_bytes,
                                 (unsigned long long)estimate->vision_final_features_bytes,
                                 (unsigned long long)estimate->vision_host_staging_bytes,
                                 (unsigned long long)estimate->model_views_bytes,
                                 (unsigned long long)estimate->kv_cache_bytes,
                                 (unsigned long long)estimate->prompt_embeddings_bytes,
                                 (unsigned long long)estimate->decoder_scratch_bytes,
                                 (unsigned long long)estimate->moe_scratch_bytes,
                                 (unsigned long long)estimate->logits_readback_bytes,
                                 (unsigned long long)estimate->transient_bytes,
                                 (unsigned long long)estimate->safety_margin_bytes);
    }
    return set_engine_errorf(engine,
                             UOCR_ERROR_OUT_OF_MEMORY,
                             "%s admission rejected: memory estimate %llu bytes exceeds budget %llu bytes "
                             "(configured_budget=%llu configured_max_batch=%u configured_max_prompt_tokens=%u configured_max_gen_tokens=%u "
                             "current_metal_arena_batch_slots=%u current_metal_arena_prompt_token_capacity=%u "
                             "current_metal_arena_bytes=%llu vision_workspace=%llu model=%llu kv=%llu prompt=%llu vision=%llu "
                             "vision_gpu_workspace=%llu vision_final_features=%llu vision_host_staging=%llu decoder=%llu moe=%llu logits=%llu transient=%llu safety=%llu)",
                             scope,
                             (unsigned long long)estimate->total_bytes,
                             (unsigned long long)budget_bytes,
                             (unsigned long long)budget_bytes,
                             configured_max_batch,
                             configured_max_prompt_tokens,
                             configured_max_gen_tokens,
                             arena_batch,
                             arena_prompt,
                             (unsigned long long)arena_bytes,
                             (unsigned long long)vision_workspace_bytes,
                             (unsigned long long)estimate->model_views_bytes,
                             (unsigned long long)estimate->kv_cache_bytes,
                             (unsigned long long)estimate->prompt_embeddings_bytes,
                             (unsigned long long)estimate->vision_scratch_bytes,
                             (unsigned long long)estimate->vision_gpu_workspace_bytes,
                             (unsigned long long)estimate->vision_final_features_bytes,
                             (unsigned long long)estimate->vision_host_staging_bytes,
                             (unsigned long long)estimate->decoder_scratch_bytes,
                             (unsigned long long)estimate->moe_scratch_bytes,
                             (unsigned long long)estimate->logits_readback_bytes,
                             (unsigned long long)estimate->transient_bytes,
                             (unsigned long long)estimate->safety_margin_bytes);
}

static int set_admission_error(uocr_engine *engine,
                               const char *scope,
                               const uocr_runtime_memory_estimate *estimate,
                               uint64_t budget_bytes) {
    return set_admission_error_with_shape(engine, scope, estimate, budget_bytes, NULL);
}

static uint64_t model_tensor_data_bytes(const uocr_model_file *model) {
    const uocr_section_entry *section = uocr_model_file_find_section(model, UOCR_SECTION_TENSOR_DATA);
    return section != NULL ? section->size : 0u;
}

#if UOCR_HAVE_METAL
static int track_engine_memory(uocr_engine *engine,
                               uocr_memory_category category,
                               uint64_t bytes,
                               const char *label) {
    if (engine == NULL || bytes == 0u) {
        return UOCR_OK;
    }
    const int status = uocr_memory_reserve(&engine->memory_tracker, category, bytes);
    if (status != UOCR_OK) {
        return set_engine_errorf(engine,
                                 status,
                                 "failed to account %s memory (%llu bytes)",
                                 label != NULL ? label : uocr_memory_category_name(category),
                                 (unsigned long long)bytes);
    }
    return UOCR_OK;
}

static void untrack_engine_memory(uocr_engine *engine, uocr_memory_category category, uint64_t bytes) {
    if (engine != NULL && bytes != 0u) {
        (void)uocr_memory_release(&engine->memory_tracker, category, bytes);
    }
}

static void release_runtime_arena_accounting_estimate(uocr_engine *engine,
                                                      const uocr_runtime_memory_estimate *estimate) {
    if (engine == NULL || estimate == NULL) {
        return;
    }
    (void)uocr_memory_release(&engine->memory_tracker, UOCR_MEMORY_KV_CACHE, estimate->kv_cache_bytes);
    (void)uocr_memory_release(&engine->memory_tracker, UOCR_MEMORY_PROMPT_EMBEDDINGS, estimate->prompt_embeddings_bytes);
    (void)uocr_memory_release(&engine->memory_tracker, UOCR_MEMORY_DECODER_SCRATCH, estimate->decoder_scratch_bytes);
    (void)uocr_memory_release(&engine->memory_tracker, UOCR_MEMORY_MOE_SCRATCH, estimate->moe_scratch_bytes);
    (void)uocr_memory_release(&engine->memory_tracker, UOCR_MEMORY_LOGITS_READBACK, estimate->logits_readback_bytes);
}

static void release_metal_runtime_arena_accounting(uocr_engine *engine) {
    if (engine == NULL || !engine->has_metal_runtime_arena_accounting) {
        return;
    }
    release_runtime_arena_accounting_estimate(engine, &engine->metal_runtime_arena_estimate);
    memset(&engine->metal_runtime_arena_estimate, 0, sizeof(engine->metal_runtime_arena_estimate));
    engine->has_metal_runtime_arena_accounting = 0;
}

static int reserve_runtime_arena_accounting(uocr_engine *engine, const uocr_runtime_memory_estimate *estimate) {
    if (engine == NULL || estimate == NULL || engine->has_metal_runtime_arena_accounting) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uocr_runtime_memory_estimate reserved;
    memset(&reserved, 0, sizeof(reserved));
#define RESERVE_ARENA_CATEGORY(category__, field__)                                      \
    do {                                                                                \
        const uint64_t bytes__ = estimate->field__;                                     \
        const int status__ = uocr_memory_reserve(&engine->memory_tracker,               \
                                                 (category__),                          \
                                                 bytes__);                              \
        if (status__ != UOCR_OK) {                                                      \
            release_runtime_arena_accounting_estimate(engine, &reserved);               \
            return status__;                                                            \
        }                                                                               \
        reserved.field__ = bytes__;                                                     \
    } while (0)

    RESERVE_ARENA_CATEGORY(UOCR_MEMORY_KV_CACHE, kv_cache_bytes);
    RESERVE_ARENA_CATEGORY(UOCR_MEMORY_PROMPT_EMBEDDINGS, prompt_embeddings_bytes);
    RESERVE_ARENA_CATEGORY(UOCR_MEMORY_DECODER_SCRATCH, decoder_scratch_bytes);
    RESERVE_ARENA_CATEGORY(UOCR_MEMORY_MOE_SCRATCH, moe_scratch_bytes);
    RESERVE_ARENA_CATEGORY(UOCR_MEMORY_LOGITS_READBACK, logits_readback_bytes);

#undef RESERVE_ARENA_CATEGORY

    engine->metal_runtime_arena_estimate = reserved;
    engine->has_metal_runtime_arena_accounting = 1;
    return UOCR_OK;
}

static int ensure_metal_runtime_arenas_for_request(uocr_engine *engine,
                                                   uint32_t batch_slots,
                                                   uint32_t prompt_token_capacity) {
    if (engine == NULL || engine->metal == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "invalid Metal runtime arena request shape");
    }
    if (batch_slots > engine->max_batch || prompt_token_capacity > engine->max_prompt_tokens) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INVALID_ARGUMENT,
                                 "Metal runtime arena request %u slots/%u prompt tokens exceeds engine caps %u/%u "
                                 "(requested_batch=%u requested_prompt_tokens=%u configured_max_batch=%u "
                                 "configured_max_prompt_tokens=%u configured_max_gen_tokens=%u current_metal_arena_batch_slots=%u "
                                 "current_metal_arena_prompt_token_capacity=%u current_metal_arena_bytes=%llu)",
                                 batch_slots,
                                 prompt_token_capacity,
                                 engine->max_batch,
                                 engine->max_prompt_tokens,
                                 batch_slots,
                                 prompt_token_capacity,
                                 engine->max_batch,
                                 engine->max_prompt_tokens,
                                 engine->max_gen_tokens,
                                 current_metal_arena_batch_slots(engine),
                                 current_metal_arena_prompt_token_capacity(engine),
                                 (unsigned long long)current_metal_arena_bytes(engine));
    }

    const uint32_t current_batch = uocr_metal_context_runtime_arena_batch_slots(engine->metal);
    const uint32_t current_prompt = uocr_metal_context_runtime_arena_prompt_token_capacity(engine->metal);
    const uint64_t current_bytes = uocr_metal_context_total_runtime_arena_capacity(engine->metal);
    if (current_batch >= batch_slots && current_prompt >= prompt_token_capacity) {
        return UOCR_OK;
    }

    const uint32_t target_batch = current_batch > batch_slots ? current_batch : batch_slots;
    const uint32_t target_prompt = current_prompt > prompt_token_capacity ? current_prompt : prompt_token_capacity;
    if (target_batch > engine->max_batch || target_prompt > engine->max_prompt_tokens) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INVALID_ARGUMENT,
                                 "Metal runtime arena growth target %u slots/%u prompt tokens exceeds engine caps %u/%u "
                                 "(requested_batch=%u requested_prompt_tokens=%u configured_max_batch=%u "
                                 "configured_max_prompt_tokens=%u configured_max_gen_tokens=%u current_metal_arena_batch_slots=%u "
                                 "current_metal_arena_prompt_token_capacity=%u current_metal_arena_bytes=%llu)",
                                 target_batch,
                                 target_prompt,
                                 engine->max_batch,
                                 engine->max_prompt_tokens,
                                 batch_slots,
                                 prompt_token_capacity,
                                 engine->max_batch,
                                 engine->max_prompt_tokens,
                                 engine->max_gen_tokens,
                                 current_batch,
                                 current_prompt,
                                 (unsigned long long)current_bytes);
    }

    uocr_runtime_memory_estimate arena_estimate;
    memset(&arena_estimate, 0, sizeof(arena_estimate));
    const int estimate_status = uocr_estimate_minimal_runtime_memory(target_batch,
                                                                     target_prompt,
                                                                     0u,
                                                                     &arena_estimate);
    if (estimate_status != UOCR_OK) {
        return set_engine_errorf(engine,
                                 estimate_status,
                                 "failed to estimate Metal runtime arena growth to %u slots/%u prompt tokens",
                                 target_batch,
                                 target_prompt);
    }

    char metal_error[512];
    memset(metal_error, 0, sizeof(metal_error));
    const uint64_t grow_start_ns = uocr_profile_now_ns();
    release_metal_runtime_arena_accounting(engine);
    if (!uocr_metal_context_allocate_runtime_arenas(engine->metal,
                                                    target_batch,
                                                    target_prompt,
                                                    metal_error,
                                                    sizeof(metal_error))) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_OUT_OF_MEMORY,
                                 "failed to grow Metal runtime arenas to %u slots/%u prompt tokens "
                                 "(requested_batch=%u requested_prompt_tokens=%u configured_max_batch=%u "
                                 "configured_max_prompt_tokens=%u configured_max_gen_tokens=%u previous_arena_batch_slots=%u "
                                 "previous_arena_prompt_token_capacity=%u previous_arena_bytes=%llu): %s",
                                 target_batch,
                                 target_prompt,
                                 batch_slots,
                                 prompt_token_capacity,
                                 engine->max_batch,
                                 engine->max_prompt_tokens,
                                 engine->max_gen_tokens,
                                 current_batch,
                                 current_prompt,
                                 (unsigned long long)current_bytes,
                                 metal_error[0] != '\0' ? metal_error : "unknown error");
    }
    const int reserve_status = reserve_runtime_arena_accounting(engine, &arena_estimate);
    if (reserve_status != UOCR_OK) {
        uocr_metal_context_release_runtime_arenas(engine->metal);
        return set_engine_errorf(engine,
                                 reserve_status,
                                 "failed to account grown Metal runtime arenas for %u slots/%u prompt tokens",
                                 target_batch,
                                 target_prompt);
    }
    uocr_profile_add_event_now(&engine->profile, "metal.runtime_arena_grow", grow_start_ns);
    return UOCR_OK;
}

static int generate_metal_text_fp16(uocr_engine *engine,
                                    const uocr_prepared_request *request,
                                    uocr_result **out_result) {
    if (engine == NULL || request == NULL || out_result == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "invalid Metal text generation request");
    }
    if (engine->metal == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_INTERNAL, "Metal backend context is not initialized");
    }
    if (!engine->has_model_file || engine->model_file.header == NULL) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "Metal fp16 text generation requires a mapped fp16 .uocr model");
    }
    if (engine->model_file.header->qprofile != UOCR_QPROFILE_FP16) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "Metal text generation currently supports only fp16 .uocr models, got %s",
                                 uocr_qprofile_name(engine->model_file.header->qprofile));
    }
    if (!uocr_metal_context_decoder_bindings_ready(engine->metal)) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "Metal fp16 text generation requires complete validated decoder tensor bindings");
    }

    int status = ensure_metal_runtime_arenas_for_request(engine, 1u, request->n_tokens);
    if (status != UOCR_OK) {
        return status;
    }
    uint64_t generated_bytes = 0u;
    if ((uint64_t)request->max_new_tokens > (uint64_t)SIZE_MAX / sizeof(int32_t)) {
        return set_engine_errorf(engine, UOCR_ERROR_OUT_OF_MEMORY, "generated-token buffer size overflow");
    }
    generated_bytes = (uint64_t)request->max_new_tokens * (uint64_t)sizeof(int32_t);
    int32_t *generated = (int32_t *)uocr_malloc((size_t)generated_bytes);
    if (generated == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_OUT_OF_MEMORY, "failed to allocate generated-token buffer");
    }
    status = track_engine_memory(engine, UOCR_MEMORY_TRANSIENT_BUFFERS, generated_bytes, "generated-token transient buffer");
    if (status != UOCR_OK) {
        uocr_free(generated);
        return status;
    }

    uocr_metal_decoder_request_f16 metal_request;
    memset(&metal_request, 0, sizeof(metal_request));
    metal_request.input_ids = request->input_ids;
    metal_request.image_mask = request->image_mask;
    metal_request.n_tokens = request->n_tokens;
    metal_request.max_new_tokens = request->max_new_tokens;
    metal_request.slot = 0u;
    metal_request.image_span_start = UINT32_MAX;
    metal_request.image_span_length = 0u;
    metal_request.no_repeat_ngram_size = request->no_repeat_ngram_size;
    metal_request.no_repeat_window = request->no_repeat_window;

    uocr_metal_decoder_result_f16 metal_result;
    memset(&metal_result, 0, sizeof(metal_result));
    metal_result.generated_ids = generated;
    metal_result.generated_capacity = request->max_new_tokens;

    char metal_error[1024];
    memset(metal_error, 0, sizeof(metal_error));
    const uint64_t generate_start_ns = uocr_profile_now_ns();
    if (!uocr_metal_context_generate_f16(engine->metal,
                                         &metal_request,
                                         &metal_result,
                                         metal_error,
                                         sizeof(metal_error))) {
        uocr_free(generated);
        untrack_engine_memory(engine, UOCR_MEMORY_TRANSIENT_BUFFERS, generated_bytes);
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INTERNAL,
                                 "Metal fp16 text generation failed: %s",
                                 metal_error[0] != '\0' ? metal_error : "unknown error");
    }
    uocr_profile_add_event_now(&engine->profile, "metal.text_generate", generate_start_ns);

    const int32_t *generated_lists[1] = {generated};
    const uint32_t generated_counts[1] = {metal_result.generated_count};
    status = uocr_result_create_from_generated(1u, generated_lists, generated_counts, out_result);
    uocr_free(generated);
    untrack_engine_memory(engine, UOCR_MEMORY_TRANSIENT_BUFFERS, generated_bytes);
    if (status != UOCR_OK) {
        return set_engine_errorf(engine, status, "failed to allocate generated-token result");
    }

    clear_engine_error(engine);
    return UOCR_OK;
}

static int generate_metal_image_fp16(uocr_engine *engine,
                                     const uocr_prepared_request *request,
                                     uocr_result **out_result) {
    if (engine == NULL || request == NULL || out_result == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_INVALID_ARGUMENT, "invalid Metal image generation request");
    }
    if (engine->metal == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_INTERNAL, "Metal backend context is not initialized");
    }
    if (!engine->has_model_file || engine->model_file.header == NULL) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "Metal fp16 image generation requires a mapped fp16 .uocr model");
    }
    if (engine->model_file.header->qprofile != UOCR_QPROFILE_FP16) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "Metal image generation currently supports only fp16 .uocr models, got %s",
                                 uocr_qprofile_name(engine->model_file.header->qprofile));
    }
    if (!uocr_metal_context_decoder_bindings_ready(engine->metal)) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "Metal fp16 image generation requires complete validated decoder tensor bindings");
    }
    if (!uocr_metal_context_vision_bindings_ready(engine->metal)) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "Metal fp16 image generation requires complete validated vision tensor bindings: %s",
                                 uocr_metal_context_vision_binding_error(engine->metal));
    }

    char validation_error[512];
    memset(validation_error, 0, sizeof(validation_error));
    uocr_sequence_state sequence_state;
    memset(&sequence_state, 0, sizeof(sequence_state));
    int status = uocr_build_sequence_state(request, &sequence_state, validation_error, sizeof(validation_error));
    if (status != UOCR_OK) {
        return set_engine_errorf(engine,
                                 status,
                                 "Metal fp16 image generation requires one contiguous image span: %s",
                                 validation_error[0] != '\0' ? validation_error : "unknown error");
    }
    if (sequence_state.image_span_start == UOCR_SEQUENCE_NO_IMAGE_SPAN || sequence_state.image_span_length == 0u) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INVALID_ARGUMENT,
                                 "Metal fp16 image generation requires at least one image placeholder");
    }

    uocr_vision_schedule vision_schedule;
    memset(&vision_schedule, 0, sizeof(vision_schedule));
    status = uocr_plan_vision_schedule_same_shape(request,
                                                  NULL,
                                                  0u,
                                                  &vision_schedule,
                                                  validation_error,
                                                  sizeof(validation_error));
    if (status != UOCR_OK) {
        return set_engine_errorf(engine,
                                 status,
                                 "Metal fp16 image generation vision schedule failed: %s",
                                 validation_error[0] != '\0' ? validation_error : "unknown error");
    }
    if (vision_schedule.final_visual_tokens == 0u) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INVALID_ARGUMENT,
                                 "Metal fp16 image generation requires image views that produce visual features");
    }
    if (sequence_state.image_span_length != vision_schedule.final_visual_tokens) {
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INVALID_ARGUMENT,
                                 "Metal fp16 image generation image span length %u does not match formatted visual rows %u",
                                 sequence_state.image_span_length,
                                 vision_schedule.final_visual_tokens);
    }

    status = ensure_metal_runtime_arenas_for_request(engine, 1u, request->n_tokens);
    if (status != UOCR_OK) {
        return status;
    }

    if ((uint64_t)request->max_new_tokens > (uint64_t)SIZE_MAX / sizeof(int32_t)) {
        return set_engine_errorf(engine, UOCR_ERROR_OUT_OF_MEMORY, "generated-token buffer size overflow");
    }
    const uint64_t generated_bytes = (uint64_t)request->max_new_tokens * (uint64_t)sizeof(int32_t);
    int32_t *generated = (int32_t *)uocr_malloc((size_t)generated_bytes);
    if (generated == NULL) {
        return set_engine_errorf(engine, UOCR_ERROR_OUT_OF_MEMORY, "failed to allocate generated-token buffer");
    }
    status = track_engine_memory(engine, UOCR_MEMORY_TRANSIENT_BUFFERS, generated_bytes, "generated-token transient buffer");
    if (status != UOCR_OK) {
        uocr_free(generated);
        return status;
    }

    uocr_metal_decoder_result_f16 metal_result;
    memset(&metal_result, 0, sizeof(metal_result));
    metal_result.generated_ids = generated;
    metal_result.generated_capacity = request->max_new_tokens;

    char metal_error[1024];
    memset(metal_error, 0, sizeof(metal_error));
    if (!uocr_metal_context_generate_image_f16(engine->metal,
                                               request,
                                               vision_schedule.max_views_per_chunk,
                                               0u,
                                               &metal_result,
                                               metal_error,
                                               sizeof(metal_error))) {
        uocr_free(generated);
        untrack_engine_memory(engine, UOCR_MEMORY_TRANSIENT_BUFFERS, generated_bytes);
        return set_engine_errorf(engine,
                                 UOCR_ERROR_INTERNAL,
                                 "Metal fp16 image generation failed: %s",
                                 metal_error[0] != '\0' ? metal_error : "unknown error");
    }

    const int32_t *generated_lists[1] = {generated};
    const uint32_t generated_counts[1] = {metal_result.generated_count};
    status = uocr_result_create_from_generated(1u, generated_lists, generated_counts, out_result);
    uocr_free(generated);
    untrack_engine_memory(engine, UOCR_MEMORY_TRANSIENT_BUFFERS, generated_bytes);
    if (status != UOCR_OK) {
        return set_engine_errorf(engine, status, "failed to allocate generated-token result");
    }

    clear_engine_error(engine);
    return UOCR_OK;
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
#if UOCR_HAVE_METAL
    if (engine->metal != NULL) {
        report->vision_workspace_capacity_bytes =
            uocr_metal_context_scratch_capacity(engine->metal, UOCR_METAL_SCRATCH_VISION);
        report->vision_workspace_high_watermark_bytes =
            uocr_metal_context_scratch_high_watermark(engine->metal, UOCR_METAL_SCRATCH_VISION);
    }
#endif
}

static void fill_profile_report(const uocr_engine *engine, uocr_profile_report *report) {
    *report = engine->profile.report;
    fill_memory_report(engine, &report->memory);
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
        case UOCR_MEMORY_VISION_GPU_WORKSPACE:
            return "vision-gpu-workspace";
        case UOCR_MEMORY_VISION_FINAL_FEATURES:
            return "vision-final-features";
        case UOCR_MEMORY_VISION_HOST_STAGING:
            return "vision-host-staging";
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
    uocr_profile_state_init(&engine->profile, (opts != NULL && opts->profile != 0u) || uocr_profile_env_enabled());
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
            uocr_runtime_memory_estimate reject_estimate;
            memset(&reject_estimate, 0, sizeof(reject_estimate));
            const int reject_estimate_status = uocr_estimate_minimal_runtime_memory(engine->max_batch,
                                                                                    engine->max_prompt_tokens,
                                                                                    engine->model_view_bytes,
                                                                                    &reject_estimate);
            if (reject_estimate_status == UOCR_OK) {
                engine->capacity_estimate = reject_estimate;
                engine->last_estimate = reject_estimate;
                (void)set_admission_error(engine, "engine", &reject_estimate, engine->memory_budget_bytes);
            } else {
                set_engine_errorf(engine,
                                  UOCR_ERROR_OUT_OF_MEMORY,
                                  "engine admission rejected: model tensor-data view estimate %llu bytes exceeds budget %llu bytes",
                                  (unsigned long long)engine->model_view_bytes,
                                  (unsigned long long)engine->memory_budget_bytes);
            }
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
        char metal_error[512];
        engine->metal = uocr_metal_context_create(engine->resource_path, metal_error, sizeof(metal_error));
        if (engine->metal == NULL) {
            set_engine_errorf(engine, UOCR_ERROR_INTERNAL, "failed to initialize Metal backend: %s", metal_error);
            uocr_engine_close(engine);
            return NULL;
        }
        uocr_metal_context_set_profile(engine->metal, &engine->profile);
        uocr_metal_context_set_memory_tracker(engine->metal, &engine->memory_tracker);
        if (engine->has_model_file && !uocr_metal_context_map_model(engine->metal, &engine->model_file, metal_error, sizeof(metal_error))) {
            set_engine_errorf(engine, UOCR_ERROR_INTERNAL, "failed to map model into Metal no-copy views: %s", metal_error);
            uocr_engine_close(engine);
            return NULL;
        }
        if (engine->has_model_file && engine->model_file.header->qprofile == UOCR_QPROFILE_FP16 &&
            !uocr_metal_context_vision_bindings_ready(engine->metal)) {
            set_engine_errorf(engine,
                              UOCR_ERROR_NOT_IMPLEMENTED,
                              "Metal fp16 image generation requires complete validated vision tensor bindings: %s",
                              uocr_metal_context_vision_binding_error(engine->metal));
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

int uocr_engine_profile_report(const uocr_engine *engine, uocr_profile_report *out_report) {
    if (engine == NULL || out_report == NULL) {
        return set_engine_errorf(NULL, UOCR_ERROR_INVALID_ARGUMENT, "engine and out_report must be non-null");
    }
    fill_profile_report(engine, out_report);
    return UOCR_OK;
}

int uocr_engine_profile_reset(uocr_engine *engine) {
    if (engine == NULL) {
        return set_engine_errorf(NULL, UOCR_ERROR_INVALID_ARGUMENT, "engine is null");
    }
    uocr_profile_reset(&engine->profile);
    uocr_memory_tracker_reset_peaks(&engine->memory_tracker);
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
    clear_engine_error(engine);
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
                                 "batch size %u exceeds engine limit %u "
                                 "(requested_batch=%u configured_max_batch=%u configured_max_prompt_tokens=%u "
                                 "configured_max_gen_tokens=%u current_metal_arena_batch_slots=%u "
                                 "current_metal_arena_prompt_token_capacity=%u current_metal_arena_bytes=%llu)",
                                 n_requests,
                                 engine->max_batch,
                                 n_requests,
                                 engine->max_batch,
                                 engine->max_prompt_tokens,
                                 engine->max_gen_tokens,
                                 current_metal_arena_batch_slots(engine),
                                 current_metal_arena_prompt_token_capacity(engine),
                                 (unsigned long long)current_metal_arena_bytes(engine));
    }

    uocr_profile_begin_request(&engine->profile);
    if (uocr_profile_is_enabled(&engine->profile)) {
        uocr_memory_tracker_reset_peaks(&engine->memory_tracker);
    }

    uocr_request_limits limits;
    memset(&limits, 0, sizeof(limits));
    limits.max_prompt_tokens = engine->max_prompt_tokens;
    limits.max_gen_tokens = engine->max_gen_tokens;
    limits.max_position_tokens = UOCR_MAX_POSITIONS;
    limits.generated_ring_window = UOCR_GENERATED_RING_WINDOW;
    uint32_t max_prompt_tokens_in_batch = 0u;
    uint32_t max_new_tokens_in_batch = 0u;
    uint32_t max_visual_tokens_in_batch = 0u;
    uint32_t max_vision_chunk_projected_rows = 0u;
    uint32_t max_vision_view_size = 0u;
    uint64_t requested_views_in_batch = 0u;
    int any_generation_requested = 0;
    const uint64_t validation_start_ns = uocr_profile_now_ns();
    for (uint32_t i = 0u; i < n_requests; ++i) {
        char validation_error[512];
        const int status = uocr_validate_prepared_request(&requests[i], &limits, validation_error, sizeof(validation_error));
        if (status != UOCR_OK) {
            return set_engine_errorf(engine,
                                     status,
                                     "request %u invalid: %s (requested_batch=%u requested_prompt_tokens=%u "
                                     "requested_max_new_tokens=%u configured_max_batch=%u configured_max_prompt_tokens=%u "
                                     "configured_max_gen_tokens=%u current_metal_arena_batch_slots=%u "
                                     "current_metal_arena_prompt_token_capacity=%u current_metal_arena_bytes=%llu)",
                                     i,
                                     validation_error,
                                     n_requests,
                                     requests[i].n_tokens,
                                     requests[i].max_new_tokens,
                                     engine->max_batch,
                                     engine->max_prompt_tokens,
                                     engine->max_gen_tokens,
                                     current_metal_arena_batch_slots(engine),
                                     current_metal_arena_prompt_token_capacity(engine),
                                     (unsigned long long)current_metal_arena_bytes(engine));
        }
        uocr_sequence_state sequence_state;
        const int state_status = uocr_build_sequence_state(&requests[i], &sequence_state, validation_error, sizeof(validation_error));
        if (state_status != UOCR_OK) {
            return set_engine_errorf(engine, state_status, "request %u state build failed: %s", i, validation_error);
        }
        (void)sequence_state; /* retained here to exercise state construction before inference kernels land */
        uocr_vision_schedule vision_schedule;
        memset(&vision_schedule, 0, sizeof(vision_schedule));
        const int vision_status = uocr_plan_vision_schedule_same_shape(&requests[i],
                                                                       NULL,
                                                                       0u,
                                                                       &vision_schedule,
                                                                       validation_error,
                                                                       sizeof(validation_error));
        if (vision_status != UOCR_OK) {
            return set_engine_errorf(engine, vision_status, "request %u vision scheduling failed: %s", i, validation_error);
        }
        requested_views_in_batch += (uint64_t)requests[i].n_views;
        if (vision_schedule.final_visual_tokens > max_visual_tokens_in_batch) {
            max_visual_tokens_in_batch = vision_schedule.final_visual_tokens;
        }
        if (vision_schedule.max_chunk_projected_tokens > max_vision_chunk_projected_rows) {
            max_vision_chunk_projected_rows = vision_schedule.max_chunk_projected_tokens;
        }
        for (uint32_t view_index = 0u; view_index < requests[i].n_views; ++view_index) {
            const uint32_t view_size = requests[i].views[view_index].width > requests[i].views[view_index].height ?
                                           requests[i].views[view_index].width :
                                           requests[i].views[view_index].height;
            if (view_size > max_vision_view_size) {
                max_vision_view_size = view_size;
            }
        }
        if (requests[i].n_tokens > max_prompt_tokens_in_batch) {
            max_prompt_tokens_in_batch = requests[i].n_tokens;
        }
        if (requests[i].max_new_tokens > max_new_tokens_in_batch) {
            max_new_tokens_in_batch = requests[i].max_new_tokens;
        }
        if (requests[i].max_new_tokens > 0u) {
            any_generation_requested = 1;
        }
    }

    uocr_profile_add_event_now(&engine->profile, "request.validation", validation_start_ns);

    uocr_runtime_memory_estimate request_estimate;
    const uint64_t estimate_start_ns = uocr_profile_now_ns();
    const int estimate_status = uocr_estimate_runtime_memory_with_vision_shape(n_requests,
                                                                               max_prompt_tokens_in_batch,
                                                                               engine->model_view_bytes,
                                                                               max_visual_tokens_in_batch,
                                                                               max_vision_chunk_projected_rows,
                                                                               max_vision_view_size,
                                                                               &request_estimate);
    if (estimate_status != UOCR_OK) {
        return set_engine_errorf(engine, estimate_status, "failed to estimate request memory requirements");
    }
    engine->last_estimate = request_estimate;
    uocr_profile_add_event_now(&engine->profile, "request.memory_estimate", estimate_start_ns);
    if (engine->memory_budget_bytes != 0u && request_estimate.total_bytes > engine->memory_budget_bytes) {
        uocr_admission_request_shape request_shape;
        memset(&request_shape, 0, sizeof(request_shape));
        request_shape.request_count = n_requests;
        request_shape.requested_views = requested_views_in_batch;
        request_shape.prompt_tokens = max_prompt_tokens_in_batch;
        request_shape.max_new_tokens = max_new_tokens_in_batch;
        request_shape.visual_rows = max_visual_tokens_in_batch;
        request_shape.max_chunk_projected_rows = max_vision_chunk_projected_rows;
        request_shape.max_view_size = max_vision_view_size;
        return set_admission_error_with_shape(engine,
                                              "request",
                                              &request_estimate,
                                              engine->memory_budget_bytes,
                                              &request_shape);
    }

    if (any_generation_requested) {
#if UOCR_HAVE_METAL
        if (strcmp(engine->backend, "metal") == 0) {
            if (n_requests != 1u) {
                return set_engine_errorf(engine,
                                         UOCR_ERROR_NOT_IMPLEMENTED,
                                         "Metal generation currently supports exactly one request");
            }
            const uocr_prepared_request *request = &requests[0];
            if (request->max_new_tokens == 0u) {
                return set_engine_errorf(engine,
                                         UOCR_ERROR_INTERNAL,
                                         "internal generation dispatch error: no generated tokens requested");
            }
            const uint32_t image_placeholders = uocr_count_image_placeholders(request);
            if (request->n_views != 0u || image_placeholders != 0u) {
                return generate_metal_image_fp16(engine, request, out_result);
            }
            return generate_metal_text_fp16(engine, request, out_result);
        }
#endif
        return set_engine_errorf(engine,
                                 UOCR_ERROR_NOT_IMPLEMENTED,
                                 "inference kernels are not implemented yet; use max_new_tokens=0 for ABI smoke tests");
    }

    const int status = uocr_result_create_empty(n_requests, out_result);
    if (status != UOCR_OK) {
        return set_engine_errorf(engine, status, "failed to allocate result");
    }
    clear_engine_error(engine);
    return UOCR_OK;
}
