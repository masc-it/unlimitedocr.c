#ifndef UOCR_METAL_H
#define UOCR_METAL_H

#include <stddef.h>
#include <stdint.h>

#include "model/uocr_model_file.h"
#include "runtime/uocr_logits_processor.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_profile.h"
#include "unlimitedocr.h"

/* Diagnostic CPU-pointer helpers allocate/copy/wait/read back for parity and
 * probes. Production translation units must not see those prototypes unless
 * they explicitly opt in, which keeps public OCR dispatch on integrated
 * GPU-resident entry points.
 */
#ifndef UOCR_METAL_ENABLE_DIAGNOSTIC_API
#define UOCR_METAL_ENABLE_DIAGNOSTIC_API 0
#endif

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
    UOCR_METAL_SCRATCH_COUNT = 6
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
    uint32_t no_repeat_ngram_size;
    uint32_t no_repeat_window;
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

#if UOCR_METAL_ENABLE_DIAGNOSTIC_API
/* Diagnostic final-visual parity boundary. Production public OCR must call
 * uocr_metal_context_generate_image_f16() so final visual rows stay in the
 * reusable Metal vision workspace and are spliced directly into the prompt
 * arena. This helper runs the same GPU-resident vision graph, then performs a
 * synchronous host copy of [out_visual_rows,1280] solely for opt-in parity and
 * probe code. max_views_per_chunk==0 means one view per chunk.
 */
int uocr_metal_context_diagnostic_encode_visual_features_f16(uocr_metal_context *ctx,
                                                  const uocr_prepared_request *request,
                                                  uint32_t max_views_per_chunk,
                                                  uint16_t *out_visual_features_f16,
                                                  uint32_t out_visual_rows,
                                                  char *error,
                                                  size_t error_size);

/* Diagnostic SAM-stage parity boundary. Runs the same SAM path used by
 * production vision for one public preprocessed view and writes the
 * upstream sam_model(view) output without batch dimension as fp16 NCHW
 * [1024,out_grid_h,out_grid_w]. This is intentionally internal/opt-in test
 * surface, not a stable public OCR API.
 */
int uocr_metal_context_diagnostic_encode_sam_features_f16(uocr_metal_context *ctx,
                                               const uocr_image_view *view,
                                               uint16_t *out_sam_features_f16,
                                               uint32_t out_grid_w,
                                               uint32_t out_grid_h,
                                               char *error,
                                               size_t error_size);

/* Diagnostic CLIP-stage parity boundary. Runs the same SAM+CLIP path used by
 * production vision for one public preprocessed view and writes the upstream
 * vision_model(view, sam_features) output without batch dimension as fp16
 * [out_token_count,1024], including the leading class token. This is
 * intentionally internal/opt-in test
 * surface, not a stable public OCR API.
 */
int uocr_metal_context_diagnostic_encode_clip_features_f16(uocr_metal_context *ctx,
                                                const uocr_image_view *view,
                                                uint16_t *out_clip_features_f16,
                                                uint32_t out_token_count,
                                                char *error,
                                                size_t error_size);

/* Diagnostic projector-stage parity boundary. Runs the same SAM+CLIP+
 * projector path used by production vision for one public preprocessed view
 * and writes the upstream per-view projector output without batch dimension as
 * fp16 [out_projected_rows,1280], before newline/view-separator formatting.
 * This is intentionally internal/opt-in test surface, not a stable public OCR API.
 */
int uocr_metal_context_diagnostic_encode_projected_features_f16(uocr_metal_context *ctx,
                                                     const uocr_image_view *view,
                                                     uint16_t *out_projected_features_f16,
                                                     uint32_t out_projected_rows,
                                                     char *error,
                                                     size_t error_size);
#endif /* UOCR_METAL_ENABLE_DIAGNOSTIC_API */

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

/* Production public-image fp16 orchestration. Encodes vision rows into the
 * reusable Metal vision workspace, then splices that Metal slice directly into
 * the prompt arena before decoder prefill/decode. This avoids the old public
 * image path's host visual-feature allocation and prompt re-upload.
 */
int uocr_metal_context_generate_image_f16(uocr_metal_context *ctx,
                                          const uocr_prepared_request *request,
                                          uint32_t max_views_per_chunk,
                                          uint32_t slot,
                                          uocr_metal_decoder_result_f16 *result,
                                          char *error,
                                          size_t error_size);

#if UOCR_METAL_ENABLE_DIAGNOSTIC_API
/* Diagnostic get-rows entry point used by synthetic tests. Runtime prompt
 * assembly should bind mmap-backed model buffers directly and reuse the same
 * kernels without uploading the embedding table through this helper.
 */
int uocr_metal_context_diagnostic_get_rows_f16(uocr_metal_context *ctx,
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
int uocr_metal_context_diagnostic_assemble_prompt_f16(uocr_metal_context *ctx,
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
#endif /* UOCR_METAL_ENABLE_DIAGNOSTIC_API */

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

#if UOCR_METAL_ENABLE_DIAGNOSTIC_API
/* Diagnostic SAM patch-embedding helper for the Metal vision bring-up path.
 * Computes Conv2d weight [768,3,16,16], optional bias [768], stride 16, and
 * writes BHWC fp16 output [height/16,width/16,768], matching upstream
 * PatchEmbed.forward()'s NCHW -> BHWC layout conversion.
 */
int uocr_metal_context_diagnostic_sam_patch_embed_f16(uocr_metal_context *ctx,
                                           const void *pixels,
                                           uocr_pixel_format pixel_format,
                                           uint32_t width,
                                           uint32_t height,
                                           const uint16_t *patch_weight_f16,
                                           const uint16_t *patch_bias_f16_or_null,
                                           uint16_t *out_bhwc_f16,
                                           uint32_t *out_grid_w,
                                           uint32_t *out_grid_h,
                                           char *error,
                                           size_t error_size);

/* Diagnostic SAM absolute-position helper. Adds pos_embed [1,64,64,768]
 * to patch embeddings shaped BHWC [grid_h,grid_w,768]. When the target grid is
 * not 64x64, the position table is resized with get_abs_pos_sam-style bicubic
 * interpolation, antialias=true, align_corners=false semantics.
 */
int uocr_metal_context_diagnostic_sam_add_abs_pos_f16(uocr_metal_context *ctx,
                                           const uint16_t *patch_bhwc_f16,
                                           const uint16_t *pos_embed_f16,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uint16_t *out_bhwc_f16,
                                           char *error,
                                           size_t error_size);

/* Diagnostic SAM transformer LayerNorm helper. Normalizes rows over the last
 * dimension using fp32 mean/variance, applies fp16 weight+bias, and uses the
 * upstream SAM transformer epsilon 1e-6 for hidden size 768.
 */
int uocr_metal_context_diagnostic_sam_layernorm_f16(uocr_metal_context *ctx,
                                         const uint16_t *input_f16,
                                         const uint16_t *weight_f16,
                                         const uint16_t *bias_f16,
                                         uint32_t n_rows,
                                         uocr_metal_layernorm_output_type output_type,
                                         void *out,
                                         char *error,
                                         size_t error_size);

/* Diagnostic SAM transformer QKV helper. Computes the biased qkv linear
 * projection with input [n_rows,768], weight [2304,768], bias [2304], and
 * writes separate Q/K/V tensors laid out [n_rows,12,64]. Dot products are
 * accumulated in fp32 before casting to the requested output type.
 */
int uocr_metal_context_diagnostic_sam_qkv_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   const uint16_t *qkv_weight_f16,
                                   const uint16_t *qkv_bias_f16,
                                   uint32_t n_rows,
                                   uocr_metal_dense_output_type output_type,
                                   void *q_out,
                                   void *k_out,
                                   void *v_out,
                                   char *error,
                                   size_t error_size);

/* Diagnostic SAM window partition helper. Converts BHWC fp16 patch grids
 * [grid_h,grid_w,768] into upstream window_partition() order
 * [n_windows,14,14,768], padding only the bottom/right edges with zeros when
 * grid_h/grid_w are not multiples of 14. The padded dimensions and window
 * count are returned for block wiring and parity checks.
 */
int uocr_metal_context_diagnostic_sam_window_partition_f16(uocr_metal_context *ctx,
                                                const uint16_t *input_bhwc_f16,
                                                uint32_t grid_w,
                                                uint32_t grid_h,
                                                uint16_t *out_windows_f16,
                                                uint32_t *out_n_windows,
                                                uint32_t *out_padded_w,
                                                uint32_t *out_padded_h,
                                                char *error,
                                                size_t error_size);

/* Diagnostic SAM window unpartition helper. Reverses the partition layout above
 * using upstream window_unpartition() semantics and crops away bottom/right
 * padding, returning BHWC fp16 [grid_h,grid_w,768].
 */
int uocr_metal_context_diagnostic_sam_window_unpartition_f16(uocr_metal_context *ctx,
                                                  const uint16_t *windows_f16,
                                                  uint32_t grid_w,
                                                  uint32_t grid_h,
                                                  uint16_t *out_bhwc_f16,
                                                  char *error,
                                                  size_t error_size);

/* Diagnostic SAM neck 1x1 convolution helper. Consumes transformer output in
 * BHWC fp16 layout [grid_h,grid_w,768], applies the first neck Conv2d weight
 * [256,768,1,1] with no bias, accumulates in fp32, and writes NCHW output
 * [256,grid_h,grid_w] for the following LayerNorm2d/3x3 neck stages.
 */
int uocr_metal_context_diagnostic_sam_neck_conv1x1_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_bhwc_f16,
                                            const uint16_t *weight_f16,
                                            uint32_t grid_w,
                                            uint32_t grid_h,
                                            uocr_metal_dense_output_type output_type,
                                            void *out_nchw,
                                            char *error,
                                            size_t error_size);

/* Diagnostic SAM neck 3x3 convolution helper. Consumes NCHW fp16 tensors
 * [256,grid_h,grid_w], applies Conv2d weight [256,256,3,3] with padding=1,
 * stride=1, no bias, accumulates in fp32, and emits NCHW output.
 */
int uocr_metal_context_diagnostic_sam_neck_conv3x3_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_nchw_f16,
                                            const uint16_t *weight_f16,
                                            uint32_t grid_w,
                                            uint32_t grid_h,
                                            uocr_metal_dense_output_type output_type,
                                            void *out_nchw,
                                            char *error,
                                            size_t error_size);

/* Diagnostic SAM neck LayerNorm2d helper for both neck normalization stages.
 * Normalizes NCHW fp16 tensors [256,grid_h,grid_w] across the channel dimension
 * for each spatial location, applies fp16 weight+bias [256], uses eps=1e-6,
 * and emits NCHW output.
 */
int uocr_metal_context_diagnostic_sam_layernorm2d_f16(uocr_metal_context *ctx,
                                           const uint16_t *input_nchw_f16,
                                           const uint16_t *weight_f16,
                                           const uint16_t *bias_f16,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uocr_metal_dense_output_type output_type,
                                           void *out_nchw,
                                           char *error,
                                           size_t error_size);

/* Diagnostic SAM net_2 stride-2 convolution helper. Consumes NCHW fp16
 * [256,grid_h,grid_w], applies Conv2d weight [512,256,3,3] with padding=1,
 * stride=2, no bias, accumulates in fp32, and emits NCHW output
 * [512,ceil(grid_h/2),ceil(grid_w/2)].
 */
int uocr_metal_context_diagnostic_sam_net2_conv3x3_stride2_f16(uocr_metal_context *ctx,
                                                    const uint16_t *input_nchw_f16,
                                                    const uint16_t *weight_f16,
                                                    uint32_t grid_w,
                                                    uint32_t grid_h,
                                                    uocr_metal_dense_output_type output_type,
                                                    void *out_nchw,
                                                    char *error,
                                                    size_t error_size);

/* Diagnostic SAM net_3 stride-2 convolution helper. Consumes NCHW fp16
 * [512,grid_h,grid_w], applies Conv2d weight [1024,512,3,3] with padding=1,
 * stride=2, no bias, accumulates in fp32, and emits NCHW output
 * [1024,ceil(grid_h/2),ceil(grid_w/2)].
 */
int uocr_metal_context_diagnostic_sam_net3_conv3x3_stride2_f16(uocr_metal_context *ctx,
                                                    const uint16_t *input_nchw_f16,
                                                    const uint16_t *weight_f16,
                                                    uint32_t grid_w,
                                                    uint32_t grid_h,
                                                    uocr_metal_dense_output_type output_type,
                                                    void *out_nchw,
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

/* Diagnostic SAM transformer block helper. Runs the upstream pre-norm block
 * over BHWC-flattened rows [grid_h*grid_w,768]. Non-global blocks partition
 * LayerNorm1 output into 14x14 windows with bottom/right padding, run rel-pos
 * attention per window, unpartition, then apply the two residuals. Global
 * blocks run rel-pos attention over the full grid. Internal activations are
 * kept fp16 for the current Metal vision bring-up path.
 */
int uocr_metal_context_diagnostic_sam_transformer_block_f16(uocr_metal_context *ctx,
                                                 const uint16_t *input_bhwc_f16,
                                                 const uocr_metal_sam_transformer_block_f16 *block,
                                                 uint32_t grid_w,
                                                 uint32_t grid_h,
                                                 int use_global_attention,
                                                 uocr_metal_dense_output_type output_type,
                                                 void *out_bhwc,
                                                 char *error,
                                                 size_t error_size);

/* Diagnostic 12-block SAM transformer helper. The blocks array must contain
 * UOCR_SAM_BLOCKS entries in layer order; global attention is selected for
 * blocks 2, 5, 8, and 11.
 */
int uocr_metal_context_diagnostic_sam_transformer_f16(uocr_metal_context *ctx,
                                           const uint16_t *input_bhwc_f16,
                                           const uocr_metal_sam_transformer_block_f16 *blocks,
                                           uint32_t block_count,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uocr_metal_dense_output_type output_type,
                                           void *out_bhwc,
                                           char *error,
                                           size_t error_size);

/* Diagnostic CLIP embedding helper for the SAM-output path. Consumes SAM net
 * output in NCHW fp16 layout [1024,grid_h,grid_w], flattens spatial features
 * in row-major order, prepends the fp16 CLIP class embedding [1024], and emits
 * token-major [1 + grid_h*grid_w,1024] embeddings. This deliberately does not
 * use the raw-pixel CLIP patch_embedding weights.
 */
int uocr_metal_context_diagnostic_clip_embed_sam_f16(uocr_metal_context *ctx,
                                          const uint16_t *sam_nchw_f16,
                                          const uint16_t *class_embedding_f16,
                                          uint32_t grid_w,
                                          uint32_t grid_h,
                                          uocr_metal_dense_output_type output_type,
                                          void *out_tokens,
                                          char *error,
                                          size_t error_size);

/* Diagnostic CLIP absolute-position helper. Adds position_embedding.weight
 * [257,1024] to token-major CLIP embeddings [1+grid_h*grid_w,1024]. The CLS
 * position is copied directly; spatial positions are resized from the 16x16
 * source grid with upstream bicubic interpolation, antialias=true, and
 * align_corners=false semantics when the target grid is 10x10.
 */
int uocr_metal_context_diagnostic_clip_add_abs_pos_f16(uocr_metal_context *ctx,
                                            const uint16_t *tokens_f16,
                                            const uint16_t *pos_embed_f16,
                                            uint32_t grid_w,
                                            uint32_t grid_h,
                                            uocr_metal_dense_output_type output_type,
                                            void *out_tokens,
                                            char *error,
                                            size_t error_size);

/* Diagnostic CLIP pre-LayerNorm helper. Normalizes token-major rows over the
 * last dimension using fp32 mean/variance, applies fp16 weight+bias from
 * vision_model.pre_layrnorm, and uses upstream epsilon 1e-5. The supported
 * token counts are 257 (16x16+CLS global view) and 101 (10x10+CLS local crop).
 */
int uocr_metal_context_diagnostic_clip_pre_layernorm_f16(uocr_metal_context *ctx,
                                              const uint16_t *input_f16,
                                              const uint16_t *weight_f16,
                                              const uint16_t *bias_f16,
                                              uint32_t token_count,
                                              uocr_metal_layernorm_output_type output_type,
                                              void *out,
                                              char *error,
                                              size_t error_size);

/* Diagnostic CLIP transformer LayerNorm helper for layer_norm1/layer_norm2 in
 * each ViT block. It uses the same fp32 mean/variance path as pre-LayerNorm,
 * hidden size 1024, and upstream epsilon 1e-5.
 */
int uocr_metal_context_diagnostic_clip_layernorm_f16(uocr_metal_context *ctx,
                                          const uint16_t *input_f16,
                                          const uint16_t *weight_f16,
                                          const uint16_t *bias_f16,
                                          uint32_t token_count,
                                          uocr_metal_layernorm_output_type output_type,
                                          void *out,
                                          char *error,
                                          size_t error_size);

/* Diagnostic CLIP transformer QKV helper. Computes the biased qkv linear
 * projection with input [tokens,1024], weight [3072,1024], bias [3072], and
 * writes separate Q/K/V tensors laid out [tokens,16,64]. Dot products are
 * accumulated in fp32 before casting to the requested output type.
 */
int uocr_metal_context_diagnostic_clip_qkv_f16(uocr_metal_context *ctx,
                                    const uint16_t *input_f16,
                                    const uint16_t *qkv_weight_f16,
                                    const uint16_t *qkv_bias_f16,
                                    uint32_t token_count,
                                    uocr_metal_dense_output_type output_type,
                                    void *q_out,
                                    void *k_out,
                                    void *v_out,
                                    char *error,
                                    size_t error_size);

/* Diagnostic CLIP full self-attention helper. Q/K/V are fp16 tensors laid out
 * [tokens,16,64]. The helper computes unmasked scaled dot-product attention
 * over all 257 or 101 tokens and writes [tokens,1024].
 */
int uocr_metal_context_diagnostic_clip_attention_f16(uocr_metal_context *ctx,
                                          const uint16_t *q_f16,
                                          const uint16_t *k_f16,
                                          const uint16_t *v_f16,
                                          uint32_t token_count,
                                          uocr_metal_dense_output_type output_type,
                                          void *out,
                                          char *error,
                                          size_t error_size);

/* Diagnostic CLIP attention output projection helper. Computes the biased
 * [1024,1024] out_proj linear over [tokens,1024] attention results for either
 * 257-token global or 101-token local CLIP sequences.
 */
int uocr_metal_context_diagnostic_clip_output_projection_f16(uocr_metal_context *ctx,
                                                  const uint16_t *input_f16,
                                                  const uint16_t *weight_f16,
                                                  const uint16_t *bias_f16,
                                                  uint32_t token_count,
                                                  uocr_metal_dense_output_type output_type,
                                                  void *out,
                                                  char *error,
                                                  size_t error_size);

/* Diagnostic CLIP MLP QuickGELU activation helper. Input/output are
 * [tokens,4096] tensors where tokens is 257 or 101. Computes
 * x * sigmoid(1.702*x) in fp32 before casting to the requested output type.
 */
int uocr_metal_context_diagnostic_clip_quickgelu_f16(uocr_metal_context *ctx,
                                          const uint16_t *input_f16,
                                          uint32_t token_count,
                                          uocr_metal_dense_output_type output_type,
                                          void *out,
                                          char *error,
                                          size_t error_size);

/* Diagnostic CLIP MLP helper. Computes fc1 [4096,1024] + bias, QuickGELU,
 * then fc2 [1024,4096] + bias for 257-token global or 101-token local CLIP
 * sequences. Intermediate activations are kept in fp16 to match the current
 * Metal fp16 vision bring-up path.
 */
int uocr_metal_context_diagnostic_clip_mlp_f16(uocr_metal_context *ctx,
                                    const uint16_t *input_f16,
                                    const uint16_t *fc1_weight_f16,
                                    const uint16_t *fc1_bias_f16,
                                    const uint16_t *fc2_weight_f16,
                                    const uint16_t *fc2_bias_f16,
                                    uint32_t token_count,
                                    uocr_metal_dense_output_type output_type,
                                    void *out,
                                    char *error,
                                    size_t error_size);

/* Diagnostic CLIP residual helper for transformer block wiring. Adds two fp16
 * [tokens,1024] tensors with fp32 arithmetic before casting. Use it for both
 * x + out_proj(attention(norm1(x))) and x + MLP(norm2(x)).
 */
int uocr_metal_context_diagnostic_clip_residual_add_f16(uocr_metal_context *ctx,
                                             const uint16_t *base_f16,
                                             const uint16_t *update_f16,
                                             uint32_t token_count,
                                             uocr_metal_dense_output_type output_type,
                                             void *out,
                                             char *error,
                                             size_t error_size);

/* Diagnostic CLIP transformer block helper. Runs the upstream pre-norm block:
 * x = x + out_proj(attention(qkv(layer_norm1(x))));
 * x = x + mlp(layer_norm2(x)). Internal block activations are kept fp16.
 */
int uocr_metal_context_diagnostic_clip_transformer_block_f16(uocr_metal_context *ctx,
                                                  const uint16_t *input_f16,
                                                  const uocr_metal_clip_transformer_block_f16 *block,
                                                  uint32_t token_count,
                                                  uocr_metal_dense_output_type output_type,
                                                  void *out,
                                                  char *error,
                                                  size_t error_size);

/* Diagnostic 24-block CLIP transformer helper for the SAM-output vision path.
 * The blocks array must contain UOCR_CLIP_BLOCKS entries in layer order.
 */
int uocr_metal_context_diagnostic_clip_transformer_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_f16,
                                            const uocr_metal_clip_transformer_block_f16 *blocks,
                                            uint32_t block_count,
                                            uint32_t token_count,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size);

/* Diagnostic visual projector-input formatter. Concatenates CLIP token features
 * without CLS and SAM net_3 NCHW feature-map tokens into token-major
 * [grid_h*grid_w,2048] rows: first 1024 CLIP channels, then 1024 SAM channels.
 */
int uocr_metal_context_diagnostic_clip_sam_concat_f16(uocr_metal_context *ctx,
                                           const uint16_t *clip_tokens_f16,
                                           const uint16_t *sam_nchw_f16,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uocr_metal_dense_output_type output_type,
                                           void *out,
                                           char *error,
                                           size_t error_size);

/* Diagnostic visual projector helper. Applies model.projector.layers as a
 * biased fp16 linear mapping from token-major [rows,2048] concat features to
 * [rows,1280] decoder-hidden visual features.
 */
int uocr_metal_context_diagnostic_visual_projector_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_f16,
                                            const uint16_t *weight_f16,
                                            const uint16_t *bias_f16,
                                            uint32_t n_rows,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size);

/* Diagnostic SAM window-attention helper for non-global transformer blocks.
 * Q/K/V are fp16 tensors laid out as [n_windows,14*14,12,64] (equivalent to
 * window-major rows with flattened [head,dim] channels). The helper computes
 * per-window, per-head scaled dot-product attention with scale 1/sqrt(64),
 * including all 196 tokens supplied by the caller. Padding/window partitioning
 * remains a separate stage; callers should pass padded window tokens when
 * matching upstream window_partition() semantics.
 */
int uocr_metal_context_diagnostic_sam_window_attention_f16(uocr_metal_context *ctx,
                                                const uint16_t *q_f16,
                                                const uint16_t *k_f16,
                                                const uint16_t *v_f16,
                                                uint32_t n_windows,
                                                uocr_metal_dense_output_type output_type,
                                                void *out,
                                                char *error,
                                                size_t error_size);

/* Diagnostic SAM global-attention helper for transformer blocks 2, 5, 8, and
 * 11. Q/K/V are fp16 tensors laid out as [grid_h*grid_w,12,64], and every
 * spatial token attends to every other token. The helper supports any positive
 * grid up to the SAM 64x64 patch grid; production views use 64x64 global and
 * 40x40 local grids. Relative-position bias is handled by the rel-pos helper
 * below.
 */
int uocr_metal_context_diagnostic_sam_global_attention_f16(uocr_metal_context *ctx,
                                                const uint16_t *q_f16,
                                                const uint16_t *k_f16,
                                                const uint16_t *v_f16,
                                                uint32_t grid_w,
                                                uint32_t grid_h,
                                                uocr_metal_dense_output_type output_type,
                                                void *out,
                                                char *error,
                                                size_t error_size);

/* Diagnostic SAM attention helper with decomposed relative-position bias from
 * rel_pos_h and rel_pos_w. Q/K/V are fp16 tensors laid out as
 * [n_windows,grid_h*grid_w,12,64]. The score is
 * (q dot k) / sqrt(64) + q dot rel_h[q_y-k_y] + q dot rel_w[q_x-k_x], matching
 * upstream add_decomposed_rel_pos() + scaled_dot_product_attention() semantics.
 * rel_pos_* tables are [rel_pos_*_length,64] and are linearly interpolated at
 * runtime when the source length differs from 2*grid_{h,w}-1 (for example,
 * 64x64-trained global tables on a 40x40 local view).
 */
int uocr_metal_context_diagnostic_sam_rel_pos_attention_f16(uocr_metal_context *ctx,
                                                 const uint16_t *q_f16,
                                                 const uint16_t *k_f16,
                                                 const uint16_t *v_f16,
                                                 const uint16_t *rel_pos_h_f16,
                                                 const uint16_t *rel_pos_w_f16,
                                                 uint32_t n_windows,
                                                 uint32_t grid_w,
                                                 uint32_t grid_h,
                                                 uint32_t rel_pos_h_length,
                                                 uint32_t rel_pos_w_length,
                                                 uocr_metal_dense_output_type output_type,
                                                 void *out,
                                                 char *error,
                                                 size_t error_size);

/* Diagnostic SAM residual helper for transformer block wiring. Adds two fp16
 * BHWC-flattened row tensors [n_rows,768] with fp32 arithmetic before casting
 * to the requested output type. Use this for shortcut + attention update after
 * window unpartition and for the final x + MLP(norm2(x)) residual.
 */
int uocr_metal_context_diagnostic_sam_residual_add_f16(uocr_metal_context *ctx,
                                            const uint16_t *base_f16,
                                            const uint16_t *update_f16,
                                            uint32_t n_rows,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size);

/* Diagnostic SAM attention-output residual helper. Computes
 * residual + (attention_context @ proj_weight.T + proj_bias) for rows shaped
 * [n_rows,768]. This matches the SAM Attention.proj step plus the first block
 * shortcut residual when the caller supplies tensors in the same row order.
 */
int uocr_metal_context_diagnostic_sam_attention_project_residual_f16(uocr_metal_context *ctx,
                                                          const uint16_t *attention_context_f16,
                                                          const uint16_t *proj_weight_f16,
                                                          const uint16_t *proj_bias_f16,
                                                          const uint16_t *residual_f16,
                                                          uint32_t n_rows,
                                                          uocr_metal_dense_output_type output_type,
                                                          void *out,
                                                          char *error,
                                                          size_t error_size);

/* Diagnostic SAM transformer MLP helper. Computes
 * lin2(GELU(lin1(input))) with input [n_rows,768], lin1 weight/bias
 * [3072,768]/[3072], lin2 weight/bias [768,3072]/[768]. GELU uses PyTorch's
 * default erf formulation with a small Metal-side erf approximation. Dot
 * products accumulate in fp32 and the post-GELU intermediate is stored as fp16
 * before lin2, matching the current fp16 activation policy for vision bring-up.
 */
int uocr_metal_context_diagnostic_sam_mlp_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   const uint16_t *lin1_weight_f16,
                                   const uint16_t *lin1_bias_f16,
                                   const uint16_t *lin2_weight_f16,
                                   const uint16_t *lin2_bias_f16,
                                   uint32_t n_rows,
                                   uocr_metal_dense_output_type output_type,
                                   void *out,
                                   char *error,
                                   size_t error_size);
#endif /* UOCR_METAL_ENABLE_DIAGNOSTIC_API */

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

#if UOCR_METAL_ENABLE_DIAGNOSTIC_API
/* Diagnostic RMSNorm helper for synthetic tests. The kernel accumulates the
 * row variance in fp32 and applies fp16 learned weights.
 */
int uocr_metal_context_diagnostic_rmsnorm_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   const uint16_t *weight_f16,
                                   uint32_t n_rows,
                                   uint32_t hidden_size,
                                   float eps,
                                   uocr_metal_rmsnorm_output_type output_type,
                                   void *out,
                                   char *error,
                                   size_t error_size);
#endif /* UOCR_METAL_ENABLE_DIAGNOSTIC_API */

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

/* Apply no-repeat-ngram bans on Metal to a row-major fp32 logits buffer and
 * copy the mutated logits back to the caller. configs_or_null==NULL disables
 * banning. This mirrors uocr_no_repeat_ngram_apply_batch() semantics for the
 * decode path (no whitelist), while moving the scan/mutation to the GPU.
 */
int uocr_metal_context_apply_no_repeat_ngram_f32(uocr_metal_context *ctx,
                                                 float *logits_f32,
                                                 uint32_t n_rows,
                                                 uint32_t vocab_size,
                                                 const uocr_no_repeat_ngram_config *configs_or_null,
                                                 char *error,
                                                 size_t error_size);

/* Decode-time greedy selection helper. It applies no-repeat-ngram bans on
 * Metal, preserving the caller-visible in-place logits mutation, then runs the
 * Metal argmax kernel. no_repeat_or_null==NULL disables banning.
 */
int uocr_metal_context_select_greedy_f32(uocr_metal_context *ctx,
                                         float *logits_f32,
                                         uint32_t n_rows,
                                         uint32_t vocab_size,
                                         const uocr_no_repeat_ngram_config *no_repeat_or_null,
                                         uint32_t *token_ids_out,
                                         float *scores_out_f32_or_null,
                                         char *error,
                                         size_t error_size);

/* Decode-time final-token selection helper. This wires the mapped final
 * RMSNorm, mapped LM head, optional no-repeat-ngram bans, and Metal greedy
 * argmax into one synchronous stage for bring-up/parity tests. The caller
 * provides fp16 normalized-hidden scratch sized [n_rows,1280] and fp32 logits
 * scratch sized [n_rows,129280]; a later production decode loop should bind
 * persistent Metal arenas directly for every intermediate.
 */
int uocr_metal_context_select_next_token_f16(uocr_metal_context *ctx,
                                             const uint16_t *hidden_f16,
                                             uint32_t n_rows,
                                             const uocr_no_repeat_ngram_config *no_repeat_or_null,
                                             uint16_t *normed_scratch_f16,
                                             float *logits_scratch_f32,
                                             uint32_t *token_ids_out,
                                             float *scores_out_f32_or_null,
                                             char *error,
                                             size_t error_size);

#if UOCR_METAL_ENABLE_DIAGNOSTIC_API
/* Diagnostic fp16 dense helper for synthetic tests. Computes
 * out[row, out_col] = dot(input[row, :], weight[out_col, :]) + optional bias,
 * where weights are row-major [out_features, in_features]. Dot products are
 * accumulated in fp32.
 */
int uocr_metal_context_diagnostic_dense_f16(uocr_metal_context *ctx,
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
int uocr_metal_context_diagnostic_attention_qkvo_f16(uocr_metal_context *ctx,
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

/* Diagnostic attention output helper for synthetic decoder tests. Computes
 * residual + (attention_context @ o_proj_weight.T), where tensors are shaped
 * [n_tokens, 1280] and o_proj_weight is row-major [1280,1280]. Dot products
 * are accumulated in fp32 before the residual add.
 */
int uocr_metal_context_diagnostic_attention_output_residual_f16(uocr_metal_context *ctx,
                                                     const uint16_t *attention_context_f16,
                                                     const uint16_t *o_weight_f16,
                                                     const uint16_t *residual_f16,
                                                     uint32_t n_tokens,
                                                     uocr_metal_dense_output_type output_type,
                                                     void *out,
                                                     char *error,
                                                     size_t error_size);

/* Diagnostic dense layer-0 MLP helper for synthetic decoder tests. Computes
 * down_proj(SiLU(input @ gate_proj.T) * (input @ up_proj.T)) with shapes
 * input/residual [n_tokens,1280], gate/up [6848,1280], down [1280,6848].
 * Gate/up/down dot products accumulate in fp32, the SwiGLU intermediate is
 * stored as fp16, and residual_f16_or_null is added after down_proj when set.
 */
int uocr_metal_context_diagnostic_dense_swiglu_f16(uocr_metal_context *ctx,
                                        const uint16_t *input_f16,
                                        const uint16_t *gate_weight_f16,
                                        const uint16_t *up_weight_f16,
                                        const uint16_t *down_weight_f16,
                                        const uint16_t *residual_f16_or_null,
                                        uint32_t n_tokens,
                                        uocr_metal_dense_output_type output_type,
                                        void *out,
                                        char *error,
                                        size_t error_size);

/* Diagnostic shared-expert MLP helper for synthetic MoE decoder tests.
 * Computes shared_experts(input) for layers 1..11 with shapes
 * input [n_tokens,1280], gate/up [1792,1280], down [1280,1792]. Dot products
 * accumulate in fp32 and the SwiGLU intermediate is stored as fp16, matching
 * the dense/routed MLP diagnostic policy. The caller adds this output to the
 * routed expert result in the following MoE-combine stage.
 */
int uocr_metal_context_diagnostic_moe_shared_experts_f16(uocr_metal_context *ctx,
                                              const uint16_t *input_f16,
                                              const uint16_t *shared_gate_weight_f16,
                                              const uint16_t *shared_up_weight_f16,
                                              const uint16_t *shared_down_weight_f16,
                                              uint32_t n_tokens,
                                              uocr_metal_dense_output_type output_type,
                                              void *out,
                                              char *error,
                                              size_t error_size);

/* Diagnostic MoE router helper for synthetic decoder tests. Computes router
 * logits = hidden @ router_weight.T for [n_tokens,1280] hidden states and
 * [64,1280] fp16 router weights, accumulates logits in fp32, softmaxes over
 * 64 experts in fp32, and returns greedy top-6 probabilities without
 * renormalization or DS4-specific router transforms. logits_out_f32_or_null
 * and probs_out_f32_or_null are optional; top ids/weights are required.
 */
int uocr_metal_context_diagnostic_moe_router_f16(uocr_metal_context *ctx,
                                      const uint16_t *input_f16,
                                      const uint16_t *router_weight_f16,
                                      uint32_t n_tokens,
                                      float *logits_out_f32_or_null,
                                      float *probs_out_f32_or_null,
                                      uint32_t *top_expert_ids_out,
                                      float *top_weights_out,
                                      char *error,
                                      size_t error_size);

/* Diagnostic selected-routed-expert decode helper for synthetic decoder tests.
 * input_f16 is one hidden row [1280]. selected_* weights are compacted in
 * selected-rank-major order according to top_expert_ids: gate/up are
 * [6,896,1280], down is [6,1280,896]. Computes the routed sum of the six
 * fp16 experts with fp32 dot accumulation, fp16 SwiGLU intermediates, and
 * fp32 router weights applied after each expert down projection. Shared
 * experts and decoder residual are intentionally separate plan items.
 */
int uocr_metal_context_diagnostic_moe_selected_experts_decode_f16(uocr_metal_context *ctx,
                                                       const uint16_t *input_f16,
                                                       const uint32_t *top_expert_ids,
                                                       const float *top_weights_f32,
                                                       const uint16_t *selected_gate_weight_f16,
                                                       const uint16_t *selected_up_weight_f16,
                                                       const uint16_t *selected_down_weight_f16,
                                                       uocr_metal_dense_output_type output_type,
                                                       void *out,
                                                       char *error,
                                                       size_t error_size);

/* Diagnostic token-batched routed-expert prefill helper. Weights are
 * expert-major [expert,out_row,input_col] for gate/up and
 * [expert,hidden_row,intermediate_col] for down. top_expert_ids/top_weights are
 * row-major [n_tokens,top_k]. This is the first prefill-oriented routed path:
 * it avoids CPU-side per-token selected-weight compaction and lets kernels read
 * expert-major slabs directly before later optimizing true expert grouping.
 */
int uocr_metal_context_diagnostic_moe_selected_experts_prefill_f16(uocr_metal_context *ctx,
                                                        const uint16_t *input_f16,
                                                        const uint32_t *top_expert_ids,
                                                        const float *top_weights_f32,
                                                        const uint16_t *expert_gate_weight_f16,
                                                        const uint16_t *expert_up_weight_f16,
                                                        const uint16_t *expert_down_weight_f16,
                                                        uint32_t n_tokens,
                                                        uint32_t hidden_size,
                                                        uint32_t intermediate_size,
                                                        uint32_t expert_count,
                                                        uint32_t top_k,
                                                        uocr_metal_dense_output_type output_type,
                                                        void *out,
                                                        char *error,
                                                        size_t error_size);

/* Diagnostic MoE combine helper for synthetic decoder tests. Computes the
 * elementwise sum of routed expert output and shared expert output for
 * [n_tokens,1280] fp16 rows. residual_f16_or_null is optional and is added
 * after the routed+shared MoE result when a full decoder-layer output is being
 * diagnosed; pass NULL to match the DeepSeekV2MoE module output itself.
 */
int uocr_metal_context_diagnostic_moe_combine_f16(uocr_metal_context *ctx,
                                       const uint16_t *routed_f16,
                                       const uint16_t *shared_f16,
                                       const uint16_t *residual_f16_or_null,
                                       uint32_t n_tokens,
                                       uocr_metal_dense_output_type output_type,
                                       void *out,
                                       char *error,
                                       size_t error_size);

/* Diagnostic RoPE helper for synthetic decoder tests. Applies Unlimited-OCR's
 * Llama-style split-half RoPE to projected Q and K tensors shaped
 * [n_tokens, 10 heads, 128 dim], using monotonically increasing positions
 * starting at position_start and theta=10000.
 */
int uocr_metal_context_diagnostic_rope_qk_f16(uocr_metal_context *ctx,
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
int uocr_metal_context_diagnostic_prefill_attention_f16(uocr_metal_context *ctx,
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
int uocr_metal_context_diagnostic_prefill_attention_varlen_f16(uocr_metal_context *ctx,
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
int uocr_metal_context_diagnostic_decode_attention_f16(uocr_metal_context *ctx,
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
int uocr_metal_context_diagnostic_write_kv_cache_f16(uocr_metal_context *ctx,
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
#endif /* UOCR_METAL_ENABLE_DIAGNOSTIC_API */

int uocr_metal_smoke_test(const char *resource_path, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_METAL_H */
