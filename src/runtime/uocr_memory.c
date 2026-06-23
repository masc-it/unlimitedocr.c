#include "runtime/uocr_memory.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#include "model/uocr_constants.h"

static int valid_category(uocr_memory_category category) {
    return category >= 0 && category < UOCR_MEMORY_CATEGORY_COUNT;
}

static int checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (UINT64_MAX - a < b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > UINT64_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int checked_add_to_total(uint64_t *total, uint64_t value) {
    uint64_t next = 0u;
    if (!checked_add_u64(*total, value, &next)) {
        return 0;
    }
    *total = next;
    return 1;
}

void uocr_memory_tracker_init(uocr_memory_tracker *tracker) {
    if (tracker != NULL) {
        memset(tracker, 0, sizeof(*tracker));
    }
}

int uocr_memory_reserve(uocr_memory_tracker *tracker, uocr_memory_category category, uint64_t bytes) {
    if (tracker == NULL || !valid_category(category)) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (bytes == 0u) {
        return UOCR_OK;
    }

    uocr_memory_counter *counter = &tracker->categories[category];
    uint64_t category_live = 0u;
    uint64_t total_live = 0u;
    if (!checked_add_u64(counter->live_bytes, bytes, &category_live) ||
        !checked_add_u64(tracker->total_live_bytes, bytes, &total_live)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    counter->live_bytes = category_live;
    if (counter->peak_bytes < category_live) {
        counter->peak_bytes = category_live;
    }
    tracker->total_live_bytes = total_live;
    if (tracker->total_peak_bytes < total_live) {
        tracker->total_peak_bytes = total_live;
    }
    return UOCR_OK;
}

int uocr_memory_release(uocr_memory_tracker *tracker, uocr_memory_category category, uint64_t bytes) {
    if (tracker == NULL || !valid_category(category)) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (bytes == 0u) {
        return UOCR_OK;
    }

    uocr_memory_counter *counter = &tracker->categories[category];
    if (counter->live_bytes < bytes || tracker->total_live_bytes < bytes) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    counter->live_bytes -= bytes;
    tracker->total_live_bytes -= bytes;
    return UOCR_OK;
}

void uocr_memory_tracker_reset_peaks(uocr_memory_tracker *tracker) {
    if (tracker == NULL) {
        return;
    }
    tracker->total_peak_bytes = tracker->total_live_bytes;
    for (uint32_t i = 0u; i < (uint32_t)UOCR_MEMORY_CATEGORY_COUNT; ++i) {
        tracker->categories[i].peak_bytes = tracker->categories[i].live_bytes;
    }
}

uint64_t uocr_kv_cache_bytes_per_token(void) {
    /*
     * Unlimited-OCR uses standard Llama-style MHA KV cache:
     *   2 K/V tensors * 12 layers * 10 KV heads * 128 head dim * 2 fp16 bytes
     * = 61,440 bytes per cached token per sequence.
     *
     * Prefill tokens stay fully attendable.  Generated tokens append to, then
     * overwrite, a fixed 128-token ring, so capacity is prompt + 128 per slot.
     */
    return 2ull *
           (uint64_t)UOCR_DECODER_LAYERS *
           (uint64_t)UOCR_KV_HEADS *
           (uint64_t)UOCR_HEAD_DIM *
           2ull;
}

int uocr_estimate_kv_cache_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes) {
    if (out_bytes == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint64_t tokens_per_slot = 0u;
    uint64_t total_tokens = 0u;
    uint64_t total_bytes = 0u;
    if (!checked_add_u64((uint64_t)prompt_token_capacity, (uint64_t)UOCR_GENERATED_RING_WINDOW, &tokens_per_slot) ||
        !checked_mul_u64((uint64_t)batch_slots, tokens_per_slot, &total_tokens) ||
        !checked_mul_u64(total_tokens, uocr_kv_cache_bytes_per_token(), &total_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    *out_bytes = total_bytes;
    return UOCR_OK;
}

int uocr_estimate_prompt_embedding_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes) {
    if (out_bytes == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint64_t tokens = 0u;
    uint64_t hidden_values = 0u;
    uint64_t total_bytes = 0u;
    if (!checked_mul_u64((uint64_t)batch_slots, (uint64_t)prompt_token_capacity, &tokens) ||
        !checked_mul_u64(tokens, (uint64_t)UOCR_HIDDEN_SIZE, &hidden_values) ||
        !checked_mul_u64(hidden_values, 2u, &total_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    *out_bytes = total_bytes;
    return UOCR_OK;
}

int uocr_estimate_vision_scratch_bytes(uint64_t *out_bytes) {
    if (out_bytes == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    /*
     * One-view chunk estimate for the fp16 vision bring-up path.  Crop mode can
     * process local views in chunks, so capacity is based on the largest global
     * 1024x1024 view plus the major SAM/CLIP/projector activation buffers that
     * must coexist during a view pass.  Attention internals should use tiled
     * kernels and named scratch arenas rather than materializing full matrices.
     */
    const uint64_t global_patches =
        ((uint64_t)UOCR_GLOBAL_VIEW_SIZE / (uint64_t)UOCR_VISION_PATCH_SIZE) *
        ((uint64_t)UOCR_GLOBAL_VIEW_SIZE / (uint64_t)UOCR_VISION_PATCH_SIZE);
    const uint64_t global_queries = (uint64_t)UOCR_GLOBAL_GRID_QUERIES * (uint64_t)UOCR_GLOBAL_GRID_QUERIES;
    const uint64_t global_clip_tokens = global_queries + 1u; /* CLS */

    uint64_t input_bytes = 0u;
    uint64_t sam_patch_bytes = 0u;
    uint64_t sam_feature_bytes = 0u;
    uint64_t clip_hidden_bytes = 0u;
    uint64_t concat_bytes = 0u;
    uint64_t projected_bytes = 0u;
    uint64_t total = 0u;
    if (!checked_mul_u64(3ull * (uint64_t)UOCR_GLOBAL_VIEW_SIZE * (uint64_t)UOCR_GLOBAL_VIEW_SIZE, 2u, &input_bytes) ||
        !checked_mul_u64(global_patches * (uint64_t)UOCR_SAM_HIDDEN_SIZE, 2u, &sam_patch_bytes) ||
        !checked_mul_u64(global_queries * (uint64_t)UOCR_SAM_FEATURE_CHANNELS, 2u, &sam_feature_bytes) ||
        !checked_mul_u64(global_clip_tokens * (uint64_t)UOCR_CLIP_HIDDEN_SIZE, 2u, &clip_hidden_bytes) ||
        !checked_mul_u64(global_queries * (uint64_t)UOCR_PROJECTOR_IN_SIZE, 2u, &concat_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_GLOBAL_VISUAL_TOKENS * (uint64_t)UOCR_HIDDEN_SIZE, 2u, &projected_bytes) ||
        !checked_add_to_total(&total, input_bytes) ||
        !checked_add_to_total(&total, sam_patch_bytes) ||
        !checked_add_to_total(&total, sam_feature_bytes) ||
        !checked_add_to_total(&total, clip_hidden_bytes) ||
        !checked_add_to_total(&total, concat_bytes) ||
        !checked_add_to_total(&total, projected_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    *out_bytes = total;
    return UOCR_OK;
}

int uocr_estimate_decoder_scratch_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes) {
    if (out_bytes == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint64_t tokens = 0u;
    uint64_t hidden_values = 0u;
    uint64_t hidden_bytes = 0u;
    if (!checked_mul_u64((uint64_t)batch_slots, (uint64_t)prompt_token_capacity, &tokens) ||
        !checked_mul_u64(tokens, (uint64_t)UOCR_HIDDEN_SIZE, &hidden_values) ||
        !checked_mul_u64(hidden_values, 2u, &hidden_bytes) ||
        !checked_mul_u64(hidden_bytes, 4u, out_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    return UOCR_OK;
}

int uocr_estimate_moe_router_topk_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes) {
    if (out_bytes == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint64_t tokens = 0u;
    uint64_t router_values = 0u;
    uint64_t selected_values = 0u;
    uint64_t router_scores = 0u;
    uint64_t selected_ids_and_weights = 0u;
    if (!checked_mul_u64((uint64_t)batch_slots, (uint64_t)prompt_token_capacity, &tokens) ||
        !checked_mul_u64(tokens, (uint64_t)UOCR_ROUTED_EXPERTS, &router_values) ||
        !checked_mul_u64(tokens, (uint64_t)UOCR_MOE_TOP_K, &selected_values) ||
        !checked_mul_u64(router_values, (uint64_t)sizeof(float), &router_scores) ||
        !checked_mul_u64(selected_values, 2ull * (uint64_t)sizeof(uint32_t), &selected_ids_and_weights) ||
        !checked_add_u64(router_scores, selected_ids_and_weights, out_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    return UOCR_OK;
}

int uocr_estimate_moe_intermediate_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes) {
    if (out_bytes == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint64_t tokens = 0u;
    uint64_t selected_values = 0u;
    uint64_t routed_values = 0u;
    uint64_t shared_values = 0u;
    uint64_t routed_intermediate = 0u;
    uint64_t shared_intermediate = 0u;
    if (!checked_mul_u64((uint64_t)batch_slots, (uint64_t)prompt_token_capacity, &tokens) ||
        !checked_mul_u64(tokens, (uint64_t)UOCR_MOE_TOP_K, &selected_values) ||
        !checked_mul_u64(selected_values, (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE, &routed_values) ||
        !checked_mul_u64(tokens, (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE, &shared_values) ||
        !checked_mul_u64(routed_values, 2u, &routed_intermediate) ||
        !checked_mul_u64(shared_values, 2u, &shared_intermediate) ||
        !checked_add_u64(routed_intermediate, shared_intermediate, out_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    return UOCR_OK;
}

int uocr_estimate_moe_scratch_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes) {
    if (out_bytes == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint64_t router_topk = 0u;
    uint64_t intermediate = 0u;
    int status = uocr_estimate_moe_router_topk_bytes(batch_slots, prompt_token_capacity, &router_topk);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_moe_intermediate_bytes(batch_slots, prompt_token_capacity, &intermediate);
    if (status != UOCR_OK) {
        return status;
    }
    if (!checked_add_u64(router_topk, intermediate, out_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    return UOCR_OK;
}

int uocr_estimate_logits_readback_bytes(uint32_t batch_slots, uint64_t *out_bytes) {
    if (out_bytes == NULL || batch_slots == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint64_t logits_values = 0u;
    uint64_t logits_bytes = 0u;
    uint64_t next_token_bytes = 0u;
    if (!checked_mul_u64((uint64_t)batch_slots, (uint64_t)UOCR_VOCAB_SIZE, &logits_values) ||
        !checked_mul_u64(logits_values, (uint64_t)sizeof(float), &logits_bytes) ||
        !checked_mul_u64((uint64_t)batch_slots, (uint64_t)sizeof(int32_t), &next_token_bytes) ||
        !checked_add_u64(logits_bytes, next_token_bytes, out_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    return UOCR_OK;
}

int uocr_estimate_safety_margin_bytes(uint64_t subtotal_bytes, uint64_t *out_bytes) {
    if (out_bytes == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    const uint64_t min_margin = 64ull * 1024ull * 1024ull;
    const uint64_t fractional_margin = subtotal_bytes / 20ull; /* 5% */
    *out_bytes = fractional_margin > min_margin ? fractional_margin : min_margin;
    return UOCR_OK;
}

int uocr_estimate_minimal_runtime_memory(uint32_t batch_slots,
                                         uint32_t prompt_token_capacity,
                                         uint64_t model_view_bytes,
                                         uocr_runtime_memory_estimate *out_estimate) {
    if (out_estimate == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uocr_runtime_memory_estimate estimate;
    memset(&estimate, 0, sizeof(estimate));
    estimate.model_views_bytes = model_view_bytes;

    int status = uocr_estimate_kv_cache_bytes(batch_slots, prompt_token_capacity, &estimate.kv_cache_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_prompt_embedding_bytes(batch_slots, prompt_token_capacity, &estimate.prompt_embeddings_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_vision_scratch_bytes(&estimate.vision_scratch_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_decoder_scratch_bytes(batch_slots, prompt_token_capacity, &estimate.decoder_scratch_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_moe_scratch_bytes(batch_slots, prompt_token_capacity, &estimate.moe_scratch_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_logits_readback_bytes(batch_slots, &estimate.logits_readback_bytes);
    if (status != UOCR_OK) {
        return status;
    }

    uint64_t total = 0u;
    if (!checked_add_to_total(&total, estimate.model_views_bytes) ||
        !checked_add_to_total(&total, estimate.kv_cache_bytes) ||
        !checked_add_to_total(&total, estimate.prompt_embeddings_bytes) ||
        !checked_add_to_total(&total, estimate.vision_scratch_bytes) ||
        !checked_add_to_total(&total, estimate.decoder_scratch_bytes) ||
        !checked_add_to_total(&total, estimate.moe_scratch_bytes) ||
        !checked_add_to_total(&total, estimate.logits_readback_bytes) ||
        !checked_add_to_total(&total, estimate.transient_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    status = uocr_estimate_safety_margin_bytes(total, &estimate.safety_margin_bytes);
    if (status != UOCR_OK) {
        return status;
    }
    if (!checked_add_to_total(&total, estimate.safety_margin_bytes)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    estimate.total_bytes = total;
    *out_estimate = estimate;
    return UOCR_OK;
}
