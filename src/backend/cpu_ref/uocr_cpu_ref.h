#ifndef UOCR_CPU_REF_H
#define UOCR_CPU_REF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *uocr_cpu_ref_backend_name(void);

float uocr_cpu_ref_bf16_bits_to_f32(uint16_t bits);
float uocr_cpu_ref_f16_bits_to_f32(uint16_t bits);
uint16_t uocr_cpu_ref_f32_to_f16_bits(float value);

int uocr_cpu_ref_bf16_to_f32_array(const uint16_t *input, size_t count, float *out);
int uocr_cpu_ref_f16_to_f32_array(const uint16_t *input, size_t count, float *out);
int uocr_cpu_ref_f32_to_f16_array(const float *input, size_t count, uint16_t *out);

int uocr_cpu_ref_rmsnorm_f32(const float *input,
                             const float *weight,
                             uint32_t rows,
                             uint32_t cols,
                             float eps,
                             float *out);

int uocr_cpu_ref_rope_split_half_f32(const float *q,
                                     const float *k,
                                     uint32_t n_tokens,
                                     uint32_t heads,
                                     uint32_t head_dim,
                                     uint32_t start_position,
                                     float theta,
                                     float *q_out,
                                     float *k_out);

int uocr_cpu_ref_causal_sdpa_f32(const float *q,
                                 const float *k,
                                 const float *v,
                                 uint32_t n_tokens,
                                 uint32_t heads,
                                 uint32_t head_dim,
                                 float scale,
                                 float *out);

typedef struct uocr_cpu_ref_kv_cache_layout {
    uint32_t prompt_token_capacity;
    uint32_t generated_ring_window;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t cache_token_capacity;
    uint64_t token_stride_floats;
    uint64_t total_floats;
} uocr_cpu_ref_kv_cache_layout;

typedef struct uocr_cpu_ref_decode_attention_plan {
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
} uocr_cpu_ref_decode_attention_plan;

int uocr_cpu_ref_kv_cache_layout_init(uint32_t prompt_token_capacity,
                                      uint32_t generated_ring_window,
                                      uint32_t heads,
                                      uint32_t head_dim,
                                      uocr_cpu_ref_kv_cache_layout *out_layout);

int uocr_cpu_ref_kv_cache_token_for_position(uint32_t prompt_length,
                                             const uocr_cpu_ref_kv_cache_layout *layout,
                                             uint32_t position,
                                             uint32_t *out_cache_token);

int uocr_cpu_ref_kv_cache_write_token_f32(float *k_cache,
                                          float *v_cache,
                                          const uocr_cpu_ref_kv_cache_layout *layout,
                                          uint32_t prompt_length,
                                          uint32_t position,
                                          const float *k_token,
                                          const float *v_token);

int uocr_cpu_ref_kv_cache_read_token_f32(const float *k_cache,
                                         const float *v_cache,
                                         const uocr_cpu_ref_kv_cache_layout *layout,
                                         uint32_t cache_token,
                                         float *k_token_out,
                                         float *v_token_out);

int uocr_cpu_ref_kv_cache_decode_attention_plan(uint32_t prompt_length,
                                                const uocr_cpu_ref_kv_cache_layout *layout,
                                                uint32_t generated_count,
                                                uocr_cpu_ref_decode_attention_plan *out_plan);

int uocr_cpu_ref_kv_cache_decode_attention_index_to_token(const uocr_cpu_ref_decode_attention_plan *plan,
                                                          uint32_t attention_index,
                                                          uint32_t *out_cache_token);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_CPU_REF_H */
