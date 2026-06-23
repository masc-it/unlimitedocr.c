#ifndef UOCR_METAL_H
#define UOCR_METAL_H

#include <stddef.h>
#include <stdint.h>

#include "model/uocr_model_file.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uocr_metal_context uocr_metal_context;

typedef struct uocr_metal_tensor_binding {
    uint32_t tensor_id;
    uint32_t view_index;
    uint64_t inner_offset;
    uint64_t payload_size;
} uocr_metal_tensor_binding;

typedef struct uocr_metal_model_view_info {
    uint64_t file_offset;
    uint64_t length;
} uocr_metal_model_view_info;

typedef enum uocr_metal_scratch_slot {
    UOCR_METAL_SCRATCH_VISION = 0,
    UOCR_METAL_SCRATCH_DECODER = 1,
    UOCR_METAL_SCRATCH_MOE = 2,
    UOCR_METAL_SCRATCH_LOGITS = 3,
    UOCR_METAL_SCRATCH_TRANSIENT = 4,
    UOCR_METAL_SCRATCH_COUNT = 5
} uocr_metal_scratch_slot;

typedef enum uocr_metal_runtime_arena_slot {
    UOCR_METAL_ARENA_KV_CACHE = 0,
    UOCR_METAL_ARENA_PROMPT_EMBEDDINGS = 1,
    UOCR_METAL_ARENA_HIDDEN_PINGPONG = 2,
    UOCR_METAL_ARENA_ROUTER_TOPK = 3,
    UOCR_METAL_ARENA_MOE_INTERMEDIATE = 4,
    UOCR_METAL_ARENA_VISION_SCRATCH = 5,
    UOCR_METAL_ARENA_LOGITS_READBACK = 6,
    UOCR_METAL_ARENA_COUNT = 7
} uocr_metal_runtime_arena_slot;

int uocr_metal_is_available(void);
const char *uocr_metal_backend_name(void);
uint64_t uocr_metal_recommended_working_set_size(void);
uint64_t uocr_metal_default_memory_budget_bytes(uint64_t recommended_working_set_bytes);

uocr_metal_context *uocr_metal_context_create(const char *resource_path, char *error, size_t error_size);
void uocr_metal_context_destroy(uocr_metal_context *ctx);

int uocr_metal_context_map_model(uocr_metal_context *ctx, const uocr_model_file *model, char *error, size_t error_size);
void uocr_metal_context_unmap_model(uocr_metal_context *ctx);
uint32_t uocr_metal_context_model_view_count(const uocr_metal_context *ctx);
uint32_t uocr_metal_context_tensor_binding_count(const uocr_metal_context *ctx);
uint64_t uocr_metal_context_model_view_bytes(const uocr_metal_context *ctx);
int uocr_metal_context_get_model_view_info(const uocr_metal_context *ctx,
                                           uint32_t view_index,
                                           uocr_metal_model_view_info *out_info);
int uocr_metal_context_get_tensor_binding(const uocr_metal_context *ctx,
                                          uint32_t tensor_id,
                                          uocr_metal_tensor_binding *out_binding);

int uocr_metal_context_ensure_scratch(uocr_metal_context *ctx,
                                      uocr_metal_scratch_slot slot,
                                      uint64_t min_length,
                                      int cpu_visible,
                                      char *error,
                                      size_t error_size);
void uocr_metal_context_release_scratch(uocr_metal_context *ctx, uocr_metal_scratch_slot slot);
uint64_t uocr_metal_context_scratch_capacity(const uocr_metal_context *ctx, uocr_metal_scratch_slot slot);
uint64_t uocr_metal_context_scratch_high_watermark(const uocr_metal_context *ctx, uocr_metal_scratch_slot slot);
uint64_t uocr_metal_context_total_scratch_capacity(const uocr_metal_context *ctx);
uint64_t uocr_metal_context_total_scratch_high_watermark(const uocr_metal_context *ctx);

int uocr_metal_context_warmup_model_views(uocr_metal_context *ctx,
                                          uint64_t max_bytes_per_view,
                                          char *error,
                                          size_t error_size);
uint64_t uocr_metal_context_last_warmup_bytes(const uocr_metal_context *ctx);

int uocr_metal_context_allocate_runtime_arenas(uocr_metal_context *ctx,
                                               uint32_t batch_slots,
                                               uint32_t prompt_token_capacity,
                                               char *error,
                                               size_t error_size);
void uocr_metal_context_release_runtime_arenas(uocr_metal_context *ctx);
uint64_t uocr_metal_context_runtime_arena_capacity(const uocr_metal_context *ctx,
                                                   uocr_metal_runtime_arena_slot slot);
uint64_t uocr_metal_context_total_runtime_arena_capacity(const uocr_metal_context *ctx);

int uocr_metal_smoke_test(const char *resource_path, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_METAL_H */
