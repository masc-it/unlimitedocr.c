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

typedef enum uocr_metal_get_rows_output_type {
    UOCR_METAL_GET_ROWS_OUTPUT_F16 = 0,
    UOCR_METAL_GET_ROWS_OUTPUT_F32 = 1
} uocr_metal_get_rows_output_type;

typedef enum uocr_metal_rmsnorm_output_type {
    UOCR_METAL_RMSNORM_OUTPUT_F16 = 0,
    UOCR_METAL_RMSNORM_OUTPUT_F32 = 1
} uocr_metal_rmsnorm_output_type;

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
int uocr_metal_kv_cache_token_for_attention_index(uint32_t prompt_length,
                                                  uint32_t prompt_token_capacity,
                                                  uint32_t generated_count,
                                                  uint32_t attention_index,
                                                  uint32_t *out_cache_token);

/* Diagnostic get-rows entry point used by synthetic tests. Runtime prompt
 * assembly should bind mmap-backed model buffers directly and reuse the same
 * kernels without uploading the embedding table through this helper.
 */
int uocr_metal_context_get_rows_f16(uocr_metal_context *ctx,
                                    const uint16_t *table_f16,
                                    uint32_t table_rows,
                                    uint32_t row_width,
                                    const int32_t *row_ids,
                                    uint32_t n_row_ids,
                                    uocr_metal_get_rows_output_type output_type,
                                    void *out,
                                    char *error,
                                    size_t error_size);

/* Diagnostic prompt assembly helper for synthetic tests. image_span_start uses
 * UINT32_MAX when there is no image span. Runtime inference should bind the
 * mapped token-embedding tensor directly and write into the prompt arena.
 */
int uocr_metal_context_assemble_prompt_f16(uocr_metal_context *ctx,
                                           const uint16_t *embedding_table_f16,
                                           uint32_t table_rows,
                                           uint32_t hidden_size,
                                           const int32_t *input_ids,
                                           uint32_t n_tokens,
                                           uint32_t image_span_start,
                                           uint32_t image_span_length,
                                           const uint16_t *image_features_f16,
                                           uint16_t *out_prompt_f16,
                                           char *error,
                                           size_t error_size);

/* Diagnostic RMSNorm helper for synthetic tests. The kernel accumulates the
 * row variance in fp32 and applies fp16 learned weights.
 */
int uocr_metal_context_rmsnorm_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   const uint16_t *weight_f16,
                                   uint32_t n_rows,
                                   uint32_t hidden_size,
                                   float eps,
                                   uocr_metal_rmsnorm_output_type output_type,
                                   void *out,
                                   char *error,
                                   size_t error_size);

/* Diagnostic fp16 dense helper for synthetic tests. Computes
 * out[row, out_col] = dot(input[row, :], weight[out_col, :]) + optional bias,
 * where weights are row-major [out_features, in_features]. Dot products are
 * accumulated in fp32.
 */
int uocr_metal_context_dense_f16(uocr_metal_context *ctx,
                                 const uint16_t *input_f16,
                                 const uint16_t *weight_f16,
                                 const uint16_t *bias_f16_or_null,
                                 uint32_t input_rows,
                                 uint32_t in_features,
                                 uint32_t out_features,
                                 uocr_metal_dense_output_type output_type,
                                 void *out,
                                 char *error,
                                 size_t error_size);

/* Diagnostic attention projection helper for synthetic tests. Computes the
 * decoder Q/K/V/O fp16 projections for hidden size 1280 with row-major
 * [1280,1280] weights and fp32 accumulation.
 */
int uocr_metal_context_attention_qkvo_f16(uocr_metal_context *ctx,
                                          const uint16_t *input_f16,
                                          const uint16_t *q_weight_f16,
                                          const uint16_t *k_weight_f16,
                                          const uint16_t *v_weight_f16,
                                          const uint16_t *o_weight_f16,
                                          uint32_t n_tokens,
                                          uocr_metal_dense_output_type output_type,
                                          void *q_out,
                                          void *k_out,
                                          void *v_out,
                                          void *o_out,
                                          char *error,
                                          size_t error_size);

/* Diagnostic RoPE helper for synthetic decoder tests. Applies Unlimited-OCR's
 * Llama-style split-half RoPE to projected Q and K tensors shaped
 * [n_tokens, 10 heads, 128 dim], using monotonically increasing positions
 * starting at position_start and theta=10000.
 */
int uocr_metal_context_rope_qk_f16(uocr_metal_context *ctx,
                                   const uint16_t *q_f16,
                                   const uint16_t *k_f16,
                                   uint32_t n_tokens,
                                   uint32_t position_start,
                                   uocr_metal_dense_output_type output_type,
                                   void *q_out,
                                   void *k_out,
                                   char *error,
                                   size_t error_size);

/* Diagnostic prompt-prefill SDPA helper for synthetic decoder tests. Computes
 * causal full-prompt attention for Q/K/V tensors shaped
 * [n_tokens, 10 heads, 128 dim]. The prompt remains fully attendable under a
 * standard lower-triangular causal mask, and softmax/dot products are fp32.
 */
int uocr_metal_context_prefill_attention_f16(uocr_metal_context *ctx,
                                             const uint16_t *q_f16,
                                             const uint16_t *k_f16,
                                             const uint16_t *v_f16,
                                             uint32_t n_tokens,
                                             uocr_metal_dense_output_type output_type,
                                             void *out,
                                             char *error,
                                             size_t error_size);

/* Diagnostic packed/variable-length prefill attention helper adapted from the
 * gradients.c sdpa_varlen layout. Q/K/V are packed as
 * [total_tokens, 10 heads, 128 dim], and cu_seqlens has batch + 1 entries.
 * Causal masking is applied independently within each sequence.
 */
int uocr_metal_context_prefill_attention_varlen_f16(uocr_metal_context *ctx,
                                                    const uint16_t *q_f16,
                                                    const uint16_t *k_f16,
                                                    const uint16_t *v_f16,
                                                    const uint32_t *cu_seqlens,
                                                    uint32_t batch,
                                                    uint32_t total_tokens,
                                                    uint32_t max_seqlen,
                                                    uocr_metal_dense_output_type output_type,
                                                    void *out,
                                                    char *error,
                                                    size_t error_size);

/* Diagnostic single-token decode SDPA helper for synthetic decoder tests. Q is
 * shaped [10 heads, 128 dim]. K/V caches are shaped
 * [12 layers, batch_slots, prompt_token_capacity + 128, 10 heads, 128 dim].
 * The attention set is all prompt tokens plus the live generated-token ring;
 * generated_count is the number of generated tokens already written to KV,
 * including the current decode token when called after the KV write.
 */
int uocr_metal_context_decode_attention_f16(uocr_metal_context *ctx,
                                            const uint16_t *q_f16,
                                            const uint16_t *k_cache_f16,
                                            const uint16_t *v_cache_f16,
                                            uint32_t batch_slots,
                                            uint32_t prompt_token_capacity,
                                            uint32_t layer,
                                            uint32_t slot,
                                            uint32_t prompt_length,
                                            uint32_t generated_count,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size);

/* Diagnostic KV-cache write helper for synthetic decoder tests. Writes K/V
 * tensors shaped [n_tokens, 10 heads, 128 dim] into separate caches laid out as
 * [12 layers, batch_slots, prompt_token_capacity + 128, 10 heads, 128 dim].
 * Prompt positions map directly; generated positions use a 128-token ring that
 * starts at the actual prompt_length, not at prompt_token_capacity.
 */
int uocr_metal_context_write_kv_cache_f16(uocr_metal_context *ctx,
                                          const uint16_t *k_f16,
                                          const uint16_t *v_f16,
                                          const uint16_t *initial_k_cache_f16_or_null,
                                          const uint16_t *initial_v_cache_f16_or_null,
                                          uint32_t n_tokens,
                                          uint32_t batch_slots,
                                          uint32_t prompt_token_capacity,
                                          uint32_t layer,
                                          uint32_t slot,
                                          uint32_t prompt_length,
                                          uint32_t position_start,
                                          uint16_t *k_cache_out_f16,
                                          uint16_t *v_cache_out_f16,
                                          char *error,
                                          size_t error_size);

int uocr_metal_smoke_test(const char *resource_path, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_METAL_H */
