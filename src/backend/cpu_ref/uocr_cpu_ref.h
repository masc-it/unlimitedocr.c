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

#ifdef __cplusplus
}
#endif

#endif /* UOCR_CPU_REF_H */
