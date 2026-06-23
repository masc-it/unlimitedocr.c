#include "runtime/uocr_memory.h"

#include <stdint.h>

#include "model/uocr_constants.h"

#define CHECK(expr)                    \
    do {                               \
        if (!(expr)) {                 \
            return __LINE__;           \
        }                              \
    } while (0)

static int test_tracker_live_and_peak_counters(void) {
    uocr_memory_tracker tracker;
    uocr_memory_tracker_init(&tracker);

    CHECK(uocr_memory_reserve(&tracker, UOCR_MEMORY_KV_CACHE, 100u) == UOCR_OK);
    CHECK(uocr_memory_reserve(&tracker, UOCR_MEMORY_PROMPT_EMBEDDINGS, 25u) == UOCR_OK);
    CHECK(tracker.categories[UOCR_MEMORY_KV_CACHE].live_bytes == 100u);
    CHECK(tracker.categories[UOCR_MEMORY_KV_CACHE].peak_bytes == 100u);
    CHECK(tracker.categories[UOCR_MEMORY_PROMPT_EMBEDDINGS].live_bytes == 25u);
    CHECK(tracker.total_live_bytes == 125u);
    CHECK(tracker.total_peak_bytes == 125u);

    CHECK(uocr_memory_release(&tracker, UOCR_MEMORY_KV_CACHE, 40u) == UOCR_OK);
    CHECK(tracker.categories[UOCR_MEMORY_KV_CACHE].live_bytes == 60u);
    CHECK(tracker.categories[UOCR_MEMORY_KV_CACHE].peak_bytes == 100u);
    CHECK(tracker.total_live_bytes == 85u);
    CHECK(tracker.total_peak_bytes == 125u);

    CHECK(uocr_memory_reserve(&tracker, UOCR_MEMORY_DECODER_SCRATCH, 200u) == UOCR_OK);
    CHECK(tracker.total_live_bytes == 285u);
    CHECK(tracker.total_peak_bytes == 285u);

    CHECK(uocr_memory_release(&tracker, UOCR_MEMORY_KV_CACHE, 61u) == UOCR_ERROR_INVALID_ARGUMENT);
    CHECK(tracker.categories[UOCR_MEMORY_KV_CACHE].live_bytes == 60u);
    CHECK(tracker.total_live_bytes == 285u);

    uocr_memory_tracker_reset_peaks(&tracker);
    CHECK(tracker.categories[UOCR_MEMORY_KV_CACHE].peak_bytes == 60u);
    CHECK(tracker.categories[UOCR_MEMORY_PROMPT_EMBEDDINGS].peak_bytes == 25u);
    CHECK(tracker.categories[UOCR_MEMORY_DECODER_SCRATCH].peak_bytes == 200u);
    CHECK(tracker.total_peak_bytes == tracker.total_live_bytes);
    return 0;
}

static int test_kv_formula(void) {
    const uint64_t per_token = uocr_kv_cache_bytes_per_token();
    CHECK(per_token == 61440u);
    CHECK(per_token == 2ull * UOCR_DECODER_LAYERS * UOCR_KV_HEADS * UOCR_HEAD_DIM * 2ull);

    uint64_t single_image_prompt_kv = 0u;
    CHECK(uocr_estimate_kv_cache_bytes(1u, UOCR_GLOBAL_VISUAL_TOKENS, &single_image_prompt_kv) == UOCR_OK);
    CHECK(single_image_prompt_kv == (uint64_t)(UOCR_GLOBAL_VISUAL_TOKENS + UOCR_GENERATED_RING_WINDOW) * per_token);

    const uint64_t prefill_only_visual_kv = (uint64_t)UOCR_GLOBAL_VISUAL_TOKENS * per_token;
    CHECK(prefill_only_visual_kv == 16773120u);
    return 0;
}

static int test_minimal_runtime_estimate(void) {
    const uint32_t batch = 2u;
    const uint32_t prompt_tokens = 4096u;
    const uint64_t model_bytes = 6672212480ull;

    uocr_runtime_memory_estimate estimate;
    CHECK(uocr_estimate_minimal_runtime_memory(batch, prompt_tokens, model_bytes, &estimate) == UOCR_OK);

    uint64_t expected_kv = 0u;
    uint64_t expected_prompt = 0u;
    uint64_t expected_vision = 0u;
    uint64_t expected_decoder = 0u;
    uint64_t expected_moe = 0u;
    uint64_t expected_logits = 0u;
    CHECK(uocr_estimate_kv_cache_bytes(batch, prompt_tokens, &expected_kv) == UOCR_OK);
    CHECK(uocr_estimate_prompt_embedding_bytes(batch, prompt_tokens, &expected_prompt) == UOCR_OK);
    CHECK(uocr_estimate_vision_scratch_bytes(&expected_vision) == UOCR_OK);
    CHECK(uocr_estimate_decoder_scratch_bytes(batch, prompt_tokens, &expected_decoder) == UOCR_OK);
    CHECK(uocr_estimate_moe_scratch_bytes(batch, prompt_tokens, &expected_moe) == UOCR_OK);
    uint64_t expected_moe_router_topk = 0u;
    uint64_t expected_moe_intermediate = 0u;
    CHECK(uocr_estimate_moe_router_topk_bytes(batch, prompt_tokens, &expected_moe_router_topk) == UOCR_OK);
    CHECK(uocr_estimate_moe_intermediate_bytes(batch, prompt_tokens, &expected_moe_intermediate) == UOCR_OK);
    CHECK(expected_moe == expected_moe_router_topk + expected_moe_intermediate);
    CHECK(uocr_estimate_logits_readback_bytes(batch, &expected_logits) == UOCR_OK);

    const uint64_t subtotal = model_bytes + expected_kv + expected_prompt + expected_vision + expected_decoder + expected_moe + expected_logits;
    uint64_t expected_safety = 0u;
    CHECK(uocr_estimate_safety_margin_bytes(subtotal, &expected_safety) == UOCR_OK);

    CHECK(estimate.model_views_bytes == model_bytes);
    CHECK(estimate.kv_cache_bytes == expected_kv);
    CHECK(estimate.prompt_embeddings_bytes == expected_prompt);
    CHECK(estimate.vision_scratch_bytes == expected_vision);
    CHECK(estimate.decoder_scratch_bytes == expected_decoder);
    CHECK(estimate.moe_scratch_bytes == expected_moe);
    CHECK(estimate.logits_readback_bytes == expected_logits);
    CHECK(estimate.safety_margin_bytes == expected_safety);
    CHECK(estimate.total_bytes == subtotal + expected_safety);
    return 0;
}

int main(void) {
    int status = 0;
    if ((status = test_tracker_live_and_peak_counters()) != 0) return status;
    if ((status = test_kv_formula()) != 0) return status;
    if ((status = test_minimal_runtime_estimate()) != 0) return status;
    return 0;
}
