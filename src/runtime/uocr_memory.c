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

static int checked_align_up_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (out == NULL || alignment == 0u) {
        return 0;
    }
    const uint64_t rem = value % alignment;
    if (rem == 0u) {
        *out = value;
        return 1;
    }
    return checked_add_u64(value, alignment - rem, out);
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

static int vision_shape_for_max_view_size(uint32_t max_view_size,
                                          uint32_t *out_patch_grid,
                                          uint32_t *out_projected_grid) {
    if (out_patch_grid == NULL || out_projected_grid == NULL) {
        return 0;
    }
    if (max_view_size == UOCR_GLOBAL_VIEW_SIZE) {
        *out_patch_grid = UOCR_GLOBAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE;
        *out_projected_grid = UOCR_GLOBAL_GRID_QUERIES;
        return 1;
    }
    if (max_view_size == UOCR_LOCAL_VIEW_SIZE) {
        *out_patch_grid = UOCR_LOCAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE;
        *out_projected_grid = UOCR_LOCAL_GRID_QUERIES;
        return 1;
    }
    return 0;
}

static int f16_slice_bytes(uint64_t value_count, uint64_t *out_bytes) {
    if (out_bytes == NULL || value_count == 0u) {
        return 0;
    }
    return checked_mul_u64(value_count, 2u, out_bytes);
}

static int add_aligned_f16_slice_bytes(uint64_t *total, uint64_t value_count) {
    uint64_t bytes = 0u;
    if (total == NULL || !f16_slice_bytes(value_count, &bytes) ||
        !checked_align_up_u64(*total, 256u, total) || !checked_add_to_total(total, bytes)) {
        return 0;
    }
    return 1;
}

int uocr_estimate_vision_memory_for_shape(uint32_t max_view_size,
                                          uint32_t final_visual_rows,
                                          uint32_t max_chunk_projected_rows,
                                          uocr_vision_memory_estimate *out_estimate) {
    if (out_estimate == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    memset(out_estimate, 0, sizeof(*out_estimate));
    if (final_visual_rows == 0u && max_chunk_projected_rows == 0u) {
        return UOCR_OK;
    }
    if (final_visual_rows == 0u || max_chunk_projected_rows == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    uint32_t patch_grid = 0u;
    uint32_t projected_grid = 0u;
    if (!vision_shape_for_max_view_size(max_view_size, &patch_grid, &projected_grid)) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    /* Exact reusable Metal vision workspace high-water estimate.  This mirrors
     * uocr_metal_vision_workspace: aligned fp16 slices for SAM patch/position/
     * transformer state, SAM/CLIP transformer block scratch, SAM neck/net
     * state, CLIP ping-pong/final tokens, one-view concat, chunk projected
     * rows, and final formatted rows. Production vision transformer scratch is
     * GPU/workspace-resident; diagnostic host staging is not part of the public
     * request-shaped workspace estimate.
     */
    const uint64_t patch_tokens = (uint64_t)patch_grid * (uint64_t)patch_grid;
    const uint64_t projected_tokens_per_view = (uint64_t)projected_grid * (uint64_t)projected_grid;
    const uint64_t net2_grid = ((uint64_t)patch_grid + 1u) / 2u;
    const uint64_t net3_grid = (net2_grid + 1u) / 2u;
    const uint64_t clip_tokens = projected_tokens_per_view + (uint64_t)UOCR_CLIP_CLASS_TOKENS;

    uint64_t sam_bhwc_values = 0u;
    uint64_t sam_neck_values = 0u;
    uint64_t sam_net2_values = 0u;
    uint64_t sam_net3_values = 0u;
    uint64_t sam_window_count = 0u;
    uint64_t sam_window_attention_tokens = 0u;
    uint64_t sam_attention_tokens = patch_tokens;
    uint64_t sam_attention_values = 0u;
    uint64_t clip_values = 0u;
    uint64_t clip_mlp_values = 0u;
    uint64_t concat_values = 0u;
    uint64_t projected_values = 0u;
    uint64_t final_visual_values = 0u;
    uint64_t total = 0u;
    if (!checked_mul_u64(patch_tokens, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &sam_bhwc_values) ||
        !checked_mul_u64(patch_tokens, (uint64_t)UOCR_SAM_NECK_CHANNELS, &sam_neck_values) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_NET2_CHANNELS, net2_grid, &sam_net2_values) ||
        !checked_mul_u64(sam_net2_values, net2_grid, &sam_net2_values) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_NET3_CHANNELS, net3_grid, &sam_net3_values) ||
        !checked_mul_u64(sam_net3_values, net3_grid, &sam_net3_values) ||
        !checked_mul_u64(((uint64_t)patch_grid + (uint64_t)UOCR_SAM_WINDOW_SIZE - 1u) /
                             (uint64_t)UOCR_SAM_WINDOW_SIZE,
                         ((uint64_t)patch_grid + (uint64_t)UOCR_SAM_WINDOW_SIZE - 1u) /
                             (uint64_t)UOCR_SAM_WINDOW_SIZE,
                         &sam_window_count) ||
        !checked_mul_u64(sam_window_count, (uint64_t)UOCR_SAM_WINDOW_TOKENS, &sam_window_attention_tokens) ||
        !checked_mul_u64(sam_attention_tokens > sam_window_attention_tokens ? sam_attention_tokens : sam_window_attention_tokens,
                         (uint64_t)UOCR_SAM_HIDDEN_SIZE,
                         &sam_attention_values) ||
        !checked_mul_u64(clip_tokens, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &clip_values) ||
        !checked_mul_u64(clip_tokens, (uint64_t)UOCR_CLIP_MLP_INTERMEDIATE, &clip_mlp_values) ||
        !checked_mul_u64(projected_tokens_per_view, (uint64_t)UOCR_PROJECTOR_IN_SIZE, &concat_values) ||
        !checked_mul_u64((uint64_t)max_chunk_projected_rows, (uint64_t)UOCR_HIDDEN_SIZE, &projected_values) ||
        !checked_mul_u64((uint64_t)final_visual_rows, (uint64_t)UOCR_HIDDEN_SIZE, &final_visual_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_attention_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_attention_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_attention_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_attention_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_attention_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_attention_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_bhwc_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_neck_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_neck_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_net2_values) ||
        !add_aligned_f16_slice_bytes(&total, sam_net3_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_mlp_values) ||
        !add_aligned_f16_slice_bytes(&total, clip_mlp_values) ||
        !add_aligned_f16_slice_bytes(&total, concat_values) ||
        !add_aligned_f16_slice_bytes(&total, projected_values) ||
        !add_aligned_f16_slice_bytes(&total, final_visual_values)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    uint64_t final_visual_bytes = 0u;
    uint64_t host_staging_bytes = 0u;
    uint64_t vision_total = 0u;
    if (!f16_slice_bytes(final_visual_values, &final_visual_bytes) || final_visual_bytes > total ||
        !checked_add_u64(total, host_staging_bytes, &vision_total)) {
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    out_estimate->gpu_workspace_bytes = total - final_visual_bytes;
    out_estimate->final_feature_bytes = final_visual_bytes;
    out_estimate->host_staging_bytes = host_staging_bytes;
    out_estimate->total_bytes = vision_total;
    return UOCR_OK;
}

int uocr_estimate_vision_scratch_bytes_for_shape(uint32_t max_view_size,
                                                 uint32_t final_visual_rows,
                                                 uint32_t max_chunk_projected_rows,
                                                 uint64_t *out_bytes) {
    if (out_bytes == NULL) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    uocr_vision_memory_estimate estimate;
    const int status = uocr_estimate_vision_memory_for_shape(max_view_size,
                                                             final_visual_rows,
                                                             max_chunk_projected_rows,
                                                             &estimate);
    if (status != UOCR_OK) {
        return status;
    }
    *out_bytes = estimate.total_bytes;
    return UOCR_OK;
}

int uocr_estimate_vision_scratch_bytes_for_rows(uint32_t final_visual_rows,
                                                uint32_t max_chunk_projected_rows,
                                                uint64_t *out_bytes) {
    return uocr_estimate_vision_scratch_bytes_for_shape(final_visual_rows != 0u ? UOCR_GLOBAL_VIEW_SIZE : 0u,
                                                        final_visual_rows,
                                                        max_chunk_projected_rows,
                                                        out_bytes);
}

int uocr_estimate_vision_scratch_bytes(uint64_t *out_bytes) {
    return uocr_estimate_vision_scratch_bytes_for_rows(UOCR_GLOBAL_VISUAL_TOKENS,
                                                       UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
                                                       out_bytes);
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
        /* Integrated prefill keeps the prompt arena immutable and uses five
         * hidden-sized scratch rows per slot: norm/context, q/attention
         * residual, k/output, v/shared, and one spare/current ping-pong lane.
         */
        !checked_mul_u64(hidden_bytes, 5u, out_bytes)) {
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

    uint64_t next_token_bytes = 0u;
    uint64_t score_bytes = 0u;
    if (!checked_mul_u64((uint64_t)batch_slots, (uint64_t)sizeof(int32_t), &next_token_bytes) ||
        !checked_mul_u64((uint64_t)batch_slots, (uint64_t)sizeof(float), &score_bytes) ||
        !checked_add_u64(next_token_bytes, score_bytes, out_bytes)) {
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

int uocr_estimate_runtime_memory_with_vision_shape(uint32_t batch_slots,
                                                   uint32_t prompt_token_capacity,
                                                   uint64_t model_view_bytes,
                                                   uint32_t final_visual_token_capacity,
                                                   uint32_t max_chunk_projected_rows,
                                                   uint32_t max_view_size,
                                                   uocr_runtime_memory_estimate *out_estimate) {
    if (out_estimate == NULL || batch_slots == 0u || prompt_token_capacity == 0u ||
        final_visual_token_capacity > prompt_token_capacity) {
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
    uocr_vision_memory_estimate vision_estimate;
    status = uocr_estimate_vision_memory_for_shape(max_view_size,
                                                   final_visual_token_capacity,
                                                   max_chunk_projected_rows,
                                                   &vision_estimate);
    if (status != UOCR_OK) {
        return status;
    }
    estimate.vision_scratch_bytes = vision_estimate.total_bytes;
    estimate.vision_gpu_workspace_bytes = vision_estimate.gpu_workspace_bytes;
    estimate.vision_final_features_bytes = vision_estimate.final_feature_bytes;
    estimate.vision_host_staging_bytes = vision_estimate.host_staging_bytes;
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

int uocr_estimate_runtime_memory_with_vision(uint32_t batch_slots,
                                             uint32_t prompt_token_capacity,
                                             uint64_t model_view_bytes,
                                             uint32_t final_visual_token_capacity,
                                             uint32_t max_chunk_projected_rows,
                                             uocr_runtime_memory_estimate *out_estimate) {
    return uocr_estimate_runtime_memory_with_vision_shape(batch_slots,
                                                         prompt_token_capacity,
                                                         model_view_bytes,
                                                         final_visual_token_capacity,
                                                         max_chunk_projected_rows,
                                                         final_visual_token_capacity != 0u ? UOCR_GLOBAL_VIEW_SIZE : 0u,
                                                         out_estimate);
}

int uocr_estimate_minimal_runtime_memory(uint32_t batch_slots,
                                         uint32_t prompt_token_capacity,
                                         uint64_t model_view_bytes,
                                         uocr_runtime_memory_estimate *out_estimate) {
    /* Engine-open estimates cover only always-live runtime arenas. Vision
     * workspace is request-shaped and allocated by the Metal scratch allocator
     * when an image request is accepted, so no prompt-sized placeholder is
     * reserved here.
     */
    return uocr_estimate_runtime_memory_with_vision(batch_slots,
                                                   prompt_token_capacity,
                                                   model_view_bytes,
                                                   0u,
                                                   0u,
                                                   out_estimate);
}
