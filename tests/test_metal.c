#include "backend/metal/uocr_metal.h"
#include "model/uocr_model_file.h"
#include "runtime/uocr_memory.h"
#include "unlimitedocr.h"

#include "uocr_test_model_file.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifndef UOCR_TEST_METAL_RESOURCE_PATH
#define UOCR_TEST_METAL_RESOURCE_PATH NULL
#endif

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static int test_metal_smoke(void) {
    if (!uocr_metal_is_available()) {
        printf("Metal device not available; skipping Metal smoke test\n");
        return 0;
    }

    const uint64_t recommended = uocr_metal_recommended_working_set_size();
    CHECK(recommended > 0u);
    const uint64_t default_budget = uocr_metal_default_memory_budget_bytes(recommended);
    CHECK(default_budget > 0u);
    CHECK(default_budget <= recommended);

    char error[1024];
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_smoke_test(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    return 0;
}

static int test_metal_named_scratch_buffers(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) == 0u);
    CHECK(uocr_metal_context_total_scratch_high_watermark(ctx) == 0u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 1024u, 0, error, sizeof(error)) == 1);
    const uint64_t first_capacity = uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(first_capacity >= 1024u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 512u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, first_capacity + 1u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) >= first_capacity + 1u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_LOGITS, 4096u, 1, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS) >= 4096u);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) >=
          uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) +
              uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS));

    uocr_metal_context_release_scratch(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == 0u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_runtime_arenas(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 16u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    uint64_t expected_kv = 0u;
    uint64_t expected_prompt = 0u;
    uint64_t expected_decoder = 0u;
    uint64_t expected_router_topk = 0u;
    uint64_t expected_moe_intermediate = 0u;
    uint64_t expected_vision = 0u;
    uint64_t expected_logits = 0u;
    CHECK(uocr_estimate_kv_cache_bytes(1u, 16u, &expected_kv) == UOCR_OK);
    CHECK(uocr_estimate_prompt_embedding_bytes(1u, 16u, &expected_prompt) == UOCR_OK);
    CHECK(uocr_estimate_decoder_scratch_bytes(1u, 16u, &expected_decoder) == UOCR_OK);
    CHECK(uocr_estimate_moe_router_topk_bytes(1u, 16u, &expected_router_topk) == UOCR_OK);
    CHECK(uocr_estimate_moe_intermediate_bytes(1u, 16u, &expected_moe_intermediate) == UOCR_OK);
    CHECK(uocr_estimate_vision_scratch_bytes(&expected_vision) == UOCR_OK);
    CHECK(uocr_estimate_logits_readback_bytes(1u, &expected_logits) == UOCR_OK);

    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_KV_CACHE) == expected_kv);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_PROMPT_EMBEDDINGS) == expected_prompt);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_HIDDEN_PINGPONG) == expected_decoder);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_ROUTER_TOPK) == expected_router_topk);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_MOE_INTERMEDIATE) == expected_moe_intermediate);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_VISION_SCRATCH) == expected_vision);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_LOGITS_READBACK) == expected_logits);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == expected_kv + expected_prompt + expected_decoder +
                                                               expected_router_topk + expected_moe_intermediate +
                                                               expected_vision + expected_logits);

    uocr_metal_context_release_runtime_arenas(ctx);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_model_mapping(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_two_tensor_uocr_model(path) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_model_view_count(ctx) == 1u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 2u);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_model_view_info view_info;
    memset(&view_info, 0, sizeof(view_info));
    CHECK(uocr_metal_context_get_model_view_info(ctx, 0u, &view_info) == 1);
    CHECK(view_info.file_offset <= model.tensors[0].payload_offset);
    CHECK(view_info.length >= UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_tensor_binding binding;
    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 1u, &binding) == 1);
    CHECK(binding.tensor_id == 1u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[0].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 16u);

    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 2u, &binding) == 1);
    CHECK(binding.tensor_id == 2u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[1].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 32u);
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 999u, &binding) == 0);

    CHECK(uocr_metal_context_warmup_model_views(ctx, 128u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_last_warmup_bytes(ctx) == 128u);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_TRANSIENT) >= sizeof(uint32_t));

    uocr_metal_context_unmap_model(ctx);
    CHECK(uocr_metal_context_model_view_count(ctx) == 0u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == 0u);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_public_engine_open_initializes_metal(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);

    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.recommended_working_set_bytes == uocr_metal_recommended_working_set_size());
    CHECK(report.memory_budget_bytes == uocr_metal_default_memory_budget_bytes(report.recommended_working_set_bytes));
    CHECK(report.memory_budget_bytes > 0u);
    CHECK(report.category_live_bytes[UOCR_MEMORY_KV_CACHE] == report.estimated_kv_cache_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_PROMPT_EMBEDDINGS] == report.estimated_prompt_embeddings_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_VISION_SCRATCH] == report.estimated_vision_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_DECODER_SCRATCH] == report.estimated_decoder_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_MOE_SCRATCH] == report.estimated_moe_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_LOGITS_READBACK] == report.estimated_logits_readback_bytes);
    CHECK(report.total_live_bytes == report.estimated_kv_cache_bytes +
                                     report.estimated_prompt_embeddings_bytes +
                                     report.estimated_vision_scratch_bytes +
                                     report.estimated_decoder_scratch_bytes +
                                     report.estimated_moe_scratch_bytes +
                                     report.estimated_logits_readback_bytes);

    uocr_engine_close(engine);
    return 0;
}

int main(void) {
    CHECK(strcmp(uocr_metal_backend_name(), "metal") == 0);
    if (test_metal_smoke() != 0) return 1;
    if (test_metal_named_scratch_buffers() != 0) return 1;
    if (test_metal_runtime_arenas() != 0) return 1;
    if (test_metal_model_mapping() != 0) return 1;
    if (test_public_engine_open_initializes_metal() != 0) return 1;
    return 0;
}
