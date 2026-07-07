#ifndef UOCR_METAL_H
#define UOCR_METAL_H

#include <stddef.h>
#include <stdint.h>

#include "model/uocr_model_file.h"
#include "runtime/uocr_logits_processor.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_profile.h"
#include "unlimitedocr.h"


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
    UOCR_METAL_SCRATCH_VISION_FINAL = 5,
    UOCR_METAL_SCRATCH_LM_HEAD_LOGITS = 6,
    UOCR_METAL_SCRATCH_COUNT = 7
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

typedef enum uocr_metal_get_rows_output_type {
    UOCR_METAL_GET_ROWS_OUTPUT_F16 = 0,
    UOCR_METAL_GET_ROWS_OUTPUT_F32 = 1
} uocr_metal_get_rows_output_type;

typedef enum uocr_metal_rmsnorm_output_type {
    UOCR_METAL_RMSNORM_OUTPUT_F16 = 0,
    UOCR_METAL_RMSNORM_OUTPUT_F32 = 1
} uocr_metal_rmsnorm_output_type;

typedef enum uocr_metal_layernorm_output_type {
    UOCR_METAL_LAYERNORM_OUTPUT_F16 = 0,
    UOCR_METAL_LAYERNORM_OUTPUT_F32 = 1
} uocr_metal_layernorm_output_type;

typedef enum uocr_metal_dense_output_type {
    UOCR_METAL_DENSE_OUTPUT_F16 = 0,
    UOCR_METAL_DENSE_OUTPUT_F32 = 1
} uocr_metal_dense_output_type;

typedef struct uocr_metal_kv_cache_layout {
    uint32_t decoder_layers;
    uint32_t batch_slots;
    uint32_t prompt_token_capacity;
    uint32_t cache_token_capacity;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t generated_ring_window;
    uint32_t reserved;
    uint64_t token_stride_bytes;
    uint64_t slot_stride_bytes;
    uint64_t layer_stride_bytes;
    uint64_t tensor_bytes;
    uint64_t k_offset_bytes;
    uint64_t v_offset_bytes;
    uint64_t total_bytes;
} uocr_metal_kv_cache_layout;

typedef struct uocr_metal_decode_attention_plan {
    uint32_t prompt_length;
    uint32_t prompt_token_capacity;
    uint32_t cache_token_capacity;
    uint32_t generated_count;
    uint32_t live_generated;
    uint32_t first_generated_index;
    uint32_t first_generated_position;
    uint32_t query_position;
    uint32_t attention_length;
    uint32_t generated_ring_window;
    uint32_t reserved0;
    uint32_t reserved1;
} uocr_metal_decode_attention_plan;

typedef struct uocr_metal_decoder_request_f16 {
    const int32_t *input_ids;
    const uint8_t *image_mask;
    const uint16_t *image_features_f16; /* optional; image_span_start is UINT32_MAX when absent */
    uint32_t n_tokens;
    uint32_t max_new_tokens;
    uint32_t slot;
    uint32_t image_span_start;
    uint32_t image_span_length;
    uint32_t reserved0;
} uocr_metal_decoder_request_f16;

typedef struct uocr_metal_decoder_result_f16 {
    int32_t *generated_ids;
    float *generated_scores_f32_or_null;
    uint32_t generated_capacity;
    uint32_t generated_count;
    uint32_t stopped_on_eos;
    uint32_t last_token_id;
    float last_score_f32;
    uint32_t reserved0;
} uocr_metal_decoder_result_f16;

typedef int (*uocr_metal_runtime_arena_prepare_fn)(void *user,
                                                   uint32_t batch_slots,
                                                   uint32_t prompt_token_capacity,
                                                   char *error,
                                                   size_t error_size);

typedef struct uocr_metal_clip_transformer_block_f16 {
    const uint16_t *ln1_weight_f16;
    const uint16_t *ln1_bias_f16;
    const uint16_t *qkv_weight_f16;
    const uint16_t *qkv_bias_f16;
    const uint16_t *out_proj_weight_f16;
    const uint16_t *out_proj_bias_f16;
    const uint16_t *ln2_weight_f16;
    const uint16_t *ln2_bias_f16;
    const uint16_t *mlp_fc1_weight_f16;
    const uint16_t *mlp_fc1_bias_f16;
    const uint16_t *mlp_fc2_weight_f16;
    const uint16_t *mlp_fc2_bias_f16;
} uocr_metal_clip_transformer_block_f16;

typedef struct uocr_metal_sam_transformer_block_f16 {
    const uint16_t *norm1_weight_f16;
    const uint16_t *norm1_bias_f16;
    const uint16_t *qkv_weight_f16;
    const uint16_t *qkv_bias_f16;
    const uint16_t *proj_weight_f16;
    const uint16_t *proj_bias_f16;
    const uint16_t *rel_pos_h_f16;
    const uint16_t *rel_pos_w_f16;
    uint32_t rel_pos_h_length;
    uint32_t rel_pos_w_length;
    const uint16_t *norm2_weight_f16;
    const uint16_t *norm2_bias_f16;
    const uint16_t *mlp_lin1_weight_f16;
    const uint16_t *mlp_lin1_bias_f16;
    const uint16_t *mlp_lin2_weight_f16;
    const uint16_t *mlp_lin2_bias_f16;
} uocr_metal_sam_transformer_block_f16;

int uocr_metal_is_available(void);
const char *uocr_metal_backend_name(void);
uint64_t uocr_metal_recommended_working_set_size(void);
uint64_t uocr_metal_default_memory_budget_bytes(uint64_t recommended_working_set_bytes);

uocr_metal_context *uocr_metal_context_create(const char *resource_path, char *error, size_t error_size);
void uocr_metal_context_destroy(uocr_metal_context *ctx);
void uocr_metal_context_set_profile(uocr_metal_context *ctx, uocr_profile_state *profile);
void uocr_metal_context_set_memory_tracker(uocr_metal_context *ctx, uocr_memory_tracker *memory_tracker);

int uocr_metal_context_map_model(uocr_metal_context *ctx, const uocr_model_file *model, char *error, size_t error_size);
void uocr_metal_context_unmap_model(uocr_metal_context *ctx);
uint32_t uocr_metal_context_model_view_count(const uocr_metal_context *ctx);
uint32_t uocr_metal_context_tensor_binding_count(const uocr_metal_context *ctx);
uint32_t uocr_metal_context_library_function_count(const uocr_metal_context *ctx);
uint32_t uocr_metal_context_pipeline_cache_count(const uocr_metal_context *ctx);
int uocr_metal_context_compile_all_pipelines(uocr_metal_context *ctx,
                                             uint32_t *out_pipeline_count,
                                             char *error,
                                             size_t error_size);
uint32_t uocr_metal_context_decoder_binding_count(const uocr_metal_context *ctx);
int uocr_metal_context_decoder_bindings_ready(const uocr_metal_context *ctx);
uint32_t uocr_metal_context_vision_binding_count(const uocr_metal_context *ctx);
int uocr_metal_context_vision_bindings_ready(const uocr_metal_context *ctx);
const char *uocr_metal_context_vision_binding_error(const uocr_metal_context *ctx);
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
uint32_t uocr_metal_context_runtime_arena_batch_slots(const uocr_metal_context *ctx);
uint32_t uocr_metal_context_runtime_arena_prompt_token_capacity(const uocr_metal_context *ctx);
int uocr_metal_context_get_kv_cache_layout(const uocr_metal_context *ctx,
                                           uocr_metal_kv_cache_layout *out_layout);
int uocr_metal_kv_cache_offset(const uocr_metal_kv_cache_layout *layout,
                               int value_cache,
                               uint32_t layer,
                               uint32_t slot,
                               uint32_t cache_token,
                               uint32_t head,
                               uint32_t dim,
                               uint64_t *out_offset_bytes);
int uocr_metal_kv_cache_token_for_position(uint32_t prompt_length,
                                           uint32_t prompt_token_capacity,
                                           uint32_t position,
                                           uint32_t *out_cache_token);
int uocr_metal_kv_cache_attention_length(uint32_t prompt_length,
                                         uint32_t generated_count,
                                         uint32_t *out_attention_length);
int uocr_metal_kv_cache_decode_attention_plan(uint32_t prompt_length,
                                             uint32_t prompt_token_capacity,
                                             uint32_t generated_count,
                                             uocr_metal_decode_attention_plan *out_plan);
int uocr_metal_kv_cache_decode_position_allowed(const uocr_metal_decode_attention_plan *plan,
                                                uint32_t key_position);
int uocr_metal_kv_cache_decode_attention_index_to_token(const uocr_metal_decode_attention_plan *plan,
                                                        uint32_t attention_index,
                                                        uint32_t *out_cache_token);
int uocr_metal_kv_cache_token_for_attention_index(uint32_t prompt_length,
                                                  uint32_t prompt_token_capacity,
                                                  uint32_t generated_count,
                                                  uint32_t attention_index,
                                                  uint32_t *out_cache_token);

/* Integrated fp16 decoder orchestration boundary. The request/result structs
 * are internal to the Metal backend and intentionally narrower than the public
 * prepared-request ABI: frontend validation and sequence construction remain
 * in shared C, while this entry point owns prompt assembly, prefill, decode,
 * logits processing, and token handoff as those stages are wired. Existing
 * per-op functions below remain diagnostic/parity helpers.
 */
int uocr_metal_context_generate_f16(uocr_metal_context *ctx,
                                    const uocr_metal_decoder_request_f16 *request,
                                    uocr_metal_decoder_result_f16 *result,
                                    char *error,
                                    size_t error_size);

/* Production public-image fp16 orchestration. Encodes vision rows into a
 * GPU-resident final-feature buffer, then splices that Metal slice directly
 * into the prompt arena before decoder prefill/decode. This avoids the old
 * public image path's host visual-feature allocation and prompt re-upload.
 */
int uocr_metal_context_generate_image_f16(uocr_metal_context *ctx,
                                          const uocr_prepared_request *request,
                                          uint32_t max_views_per_chunk,
                                          uint32_t slot,
                                          uocr_metal_decoder_result_f16 *result,
                                          char *error,
                                          size_t error_size);
int uocr_metal_context_generate_image_f16_deferred_runtime(uocr_metal_context *ctx,
                                                           const uocr_prepared_request *request,
                                                           uint32_t max_views_per_chunk,
                                                           uint32_t slot,
                                                           uocr_metal_decoder_result_f16 *result,
                                                           uocr_metal_runtime_arena_prepare_fn prepare_runtime,
                                                           void *prepare_user,
                                                           char *error,
                                                           size_t error_size);
/* Runtime prompt assembly helper. Binds the mmap-backed TOK_EMBED tensor from
 * mapped .uocr model views, uses fixed Unlimited-OCR vocab/hidden sizes, and
 * supports the same direct image-feature span used by the diagnostic helper.
 */
int uocr_metal_context_assemble_prompt_from_model_f16(uocr_metal_context *ctx,
                                                      const int32_t *input_ids,
                                                      uint32_t n_tokens,
                                                      uint32_t image_span_start,
                                                      uint32_t image_span_length,
                                                      const uint16_t *image_features_f16,
                                                      uint16_t *out_prompt_f16,
                                                      char *error,
                                                      size_t error_size);
/* Runtime prompt assembly into the persistent Metal prompt-embedding arena.
 * The arena must have been allocated with uocr_metal_context_allocate_runtime_arenas().
 * slot selects the batch slot. Tests can inspect the private arena with the
 * readback helper below; production decoder stages should consume it in-place.
 */
int uocr_metal_context_assemble_prompt_from_model_to_arena_f16(uocr_metal_context *ctx,
                                                               const int32_t *input_ids,
                                                               uint32_t n_tokens,
                                                               uint32_t image_span_start,
                                                               uint32_t image_span_length,
                                                               const uint16_t *image_features_f16,
                                                               uint32_t slot,
                                                               char *error,
                                                               size_t error_size);
int uocr_metal_context_read_prompt_arena_f16(uocr_metal_context *ctx,
                                             uint32_t slot,
                                             uint32_t n_tokens,
                                             uint16_t *out_prompt_f16,
                                             char *error,
                                             size_t error_size);
int uocr_metal_context_read_decoder_final_hidden_f16(uocr_metal_context *ctx,
                                                     uint32_t slot,
                                                     uint32_t n_tokens,
                                                     uint16_t *out_hidden_f16,
                                                     char *error,
                                                     size_t error_size);
/* Final decoder RMSNorm helper. Binds the mmap-backed FINAL_NORM tensor from
 * mapped .uocr model views, applies the fixed Unlimited-OCR hidden size and
 * eps=1e-6, and returns either fp16 activations or fp32 diagnostics.
 */
int uocr_metal_context_final_rmsnorm_f16(uocr_metal_context *ctx,
                                         const uint16_t *input_f16,
                                         uint32_t n_rows,
                                         uocr_metal_rmsnorm_output_type output_type,
                                         void *out,
                                         char *error,
                                         size_t error_size);

/* LM-head helper for decode/generation. Binds the mmap-backed LM_HEAD tensor
 * [129280,1280] from mapped .uocr model views and computes fp32 logits for
 * one or more normalized hidden rows. The initial synchronous helper reads
 * logits back to host for correctness and CPU logits processors.
 */
int uocr_metal_context_lm_head_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   uint32_t n_rows,
                                   float *logits_out_f32,
                                   char *error,
                                   size_t error_size);

/* Greedy argmax over fp32 logits. Ties choose the lowest token id, matching
 * deterministic first-index argmax semantics. scores_out_f32_or_null is
 * optional and receives the selected logit for each row.
 */
int uocr_metal_context_argmax_f32(uocr_metal_context *ctx,
                                  const float *logits_f32,
                                  uint32_t n_rows,
                                  uint32_t vocab_size,
                                  uint32_t *token_ids_out,
                                  float *scores_out_f32_or_null,
                                  char *error,
                                  size_t error_size);


/* Decode-time greedy selection helper. Runs the Metal argmax kernel on a
 * row-major fp32 logits buffer.
 */
int uocr_metal_context_select_greedy_f32(uocr_metal_context *ctx,
                                         float *logits_f32,
                                         uint32_t n_rows,
                                         uint32_t vocab_size,
                                         uint32_t *token_ids_out,
                                         float *scores_out_f32_or_null,
                                         char *error,
                                         size_t error_size);

/* Decode-time final-token selection helper. This wires the mapped final
 * RMSNorm, mapped LM head, and Metal greedy argmax into one synchronous stage
 * for bring-up/parity tests. The caller provides fp16 normalized-hidden
 * scratch sized [n_rows,1280] and fp32 logits scratch sized [n_rows,129280].
 */
int uocr_metal_context_select_next_token_f16(uocr_metal_context *ctx,
                                             const uint16_t *hidden_f16,
                                             uint32_t n_rows,
                                             uint16_t *normed_scratch_f16,
                                             float *logits_scratch_f32,
                                             uint32_t *token_ids_out,
                                             float *scores_out_f32_or_null,
                                             char *error,
                                             size_t error_size);
int uocr_metal_smoke_test(const char *resource_path, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_METAL_H */
