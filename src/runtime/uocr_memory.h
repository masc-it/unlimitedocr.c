#ifndef UOCR_RUNTIME_MEMORY_H
#define UOCR_RUNTIME_MEMORY_H

#include <stdint.h>

#include "unlimitedocr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uocr_memory_counter {
    uint64_t live_bytes;
    uint64_t peak_bytes;
} uocr_memory_counter;

typedef struct uocr_memory_tracker {
    uocr_memory_counter categories[UOCR_MEMORY_CATEGORY_COUNT];
    uint64_t total_live_bytes;
    uint64_t total_peak_bytes;
} uocr_memory_tracker;

typedef struct uocr_runtime_memory_estimate {
    uint64_t model_views_bytes;
    uint64_t kv_cache_bytes;
    uint64_t prompt_embeddings_bytes;
    uint64_t vision_scratch_bytes;
    uint64_t decoder_scratch_bytes;
    uint64_t moe_scratch_bytes;
    uint64_t logits_readback_bytes;
    uint64_t transient_bytes;
    uint64_t safety_margin_bytes;
    uint64_t total_bytes;
} uocr_runtime_memory_estimate;

void uocr_memory_tracker_init(uocr_memory_tracker *tracker);
int uocr_memory_reserve(uocr_memory_tracker *tracker, uocr_memory_category category, uint64_t bytes);
int uocr_memory_release(uocr_memory_tracker *tracker, uocr_memory_category category, uint64_t bytes);
void uocr_memory_tracker_reset_peaks(uocr_memory_tracker *tracker);

uint64_t uocr_kv_cache_bytes_per_token(void);
int uocr_estimate_kv_cache_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes);
int uocr_estimate_prompt_embedding_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes);
int uocr_estimate_vision_scratch_bytes(uint64_t *out_bytes);
int uocr_estimate_decoder_scratch_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes);
int uocr_estimate_moe_scratch_bytes(uint32_t batch_slots, uint32_t prompt_token_capacity, uint64_t *out_bytes);
int uocr_estimate_logits_readback_bytes(uint32_t batch_slots, uint64_t *out_bytes);
int uocr_estimate_safety_margin_bytes(uint64_t subtotal_bytes, uint64_t *out_bytes);
int uocr_estimate_minimal_runtime_memory(uint32_t batch_slots,
                                         uint32_t prompt_token_capacity,
                                         uint64_t model_view_bytes,
                                         uocr_runtime_memory_estimate *out_estimate);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_RUNTIME_MEMORY_H */
