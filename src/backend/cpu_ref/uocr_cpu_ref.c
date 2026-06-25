#include "backend/cpu_ref/uocr_cpu_ref.h"

#include <math.h>
#include <string.h>

const char *uocr_cpu_ref_backend_name(void) {
    return "cpu-ref";
}

static uint32_t f32_to_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_to_f32(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float uocr_cpu_ref_bf16_bits_to_f32(uint16_t bits) {
    return bits_to_f32((uint32_t)bits << 16u);
}

float uocr_cpu_ref_f16_bits_to_f32(uint16_t bits16) {
    const uint32_t sign = ((uint32_t)bits16 & 0x8000u) << 16u;
    uint32_t mantissa = (uint32_t)bits16 & 0x03ffu;
    int exponent = (int)((bits16 >> 10u) & 0x001fu);
    uint32_t bits32 = 0u;

    if (exponent == 0) {
        if (mantissa == 0u) {
            bits32 = sign;
        } else {
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            mantissa &= ~0x0400u;
            bits32 = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        bits32 = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits32 = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
    }

    return bits_to_f32(bits32);
}

uint16_t uocr_cpu_ref_f32_to_f16_bits(float value) {
    const uint32_t bits32 = f32_to_bits(value);
    const uint32_t sign = (bits32 >> 16u) & 0x8000u;
    const uint32_t exponent = (bits32 >> 23u) & 0xffu;
    const uint32_t mantissa = bits32 & 0x007fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0u) {
            return (uint16_t)(sign | 0x7c00u);
        }
        return (uint16_t)(sign | 0x7e00u);
    }

    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return (uint16_t)sign;
        }
        const uint32_t mant = mantissa | 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mant >> shift;
        const uint32_t round_bit = 1u << (shift - 1u);
        const uint32_t remainder = mant & (round_bit - 1u);
        if ((mant & round_bit) != 0u && (remainder != 0u || (rounded & 1u) != 0u)) {
            ++rounded;
        }
        return (uint16_t)(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13u;
    const uint32_t round_bit = 0x00001000u;
    const uint32_t remainder = mantissa & (round_bit - 1u);
    if ((mantissa & round_bit) != 0u && (remainder != 0u || (rounded_mantissa & 1u) != 0u)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0u;
            ++half_exponent;
            if (half_exponent >= 31) {
                return (uint16_t)(sign | 0x7c00u);
            }
        }
    }

    return (uint16_t)(sign | ((uint32_t)half_exponent << 10u) | rounded_mantissa);
}

int uocr_cpu_ref_bf16_to_f32_array(const uint16_t *input, size_t count, float *out) {
    if (count == 0u) {
        return 1;
    }
    if (input == NULL || out == NULL) {
        return 0;
    }
    for (size_t index = 0u; index < count; ++index) {
        out[index] = uocr_cpu_ref_bf16_bits_to_f32(input[index]);
    }
    return 1;
}

int uocr_cpu_ref_f16_to_f32_array(const uint16_t *input, size_t count, float *out) {
    if (count == 0u) {
        return 1;
    }
    if (input == NULL || out == NULL) {
        return 0;
    }
    for (size_t index = 0u; index < count; ++index) {
        out[index] = uocr_cpu_ref_f16_bits_to_f32(input[index]);
    }
    return 1;
}

int uocr_cpu_ref_f32_to_f16_array(const float *input, size_t count, uint16_t *out) {
    if (count == 0u) {
        return 1;
    }
    if (input == NULL || out == NULL) {
        return 0;
    }
    for (size_t index = 0u; index < count; ++index) {
        out[index] = uocr_cpu_ref_f32_to_f16_bits(input[index]);
    }
    return 1;
}

static size_t token_head_offset(uint32_t token, uint32_t head, uint32_t heads, uint32_t head_dim) {
    return ((size_t)token * heads + head) * head_dim;
}

int uocr_cpu_ref_rmsnorm_f32(const float *input,
                             const float *weight,
                             uint32_t rows,
                             uint32_t cols,
                             float eps,
                             float *out) {
    if (input == NULL || weight == NULL || out == NULL || rows == 0u || cols == 0u || eps < 0.0f) {
        return 0;
    }

    for (uint32_t row = 0u; row < rows; ++row) {
        const size_t row_base = (size_t)row * cols;
        float sum_squares = 0.0f;
        for (uint32_t col = 0u; col < cols; ++col) {
            const float value = input[row_base + col];
            sum_squares = fmaf(value, value, sum_squares);
        }
        const float mean_square = sum_squares / (float)cols;
        const float inv_rms = 1.0f / sqrtf(mean_square + eps);
        for (uint32_t col = 0u; col < cols; ++col) {
            out[row_base + col] = input[row_base + col] * inv_rms * weight[col];
        }
    }
    return 1;
}

int uocr_cpu_ref_rope_split_half_f32(const float *q,
                                     const float *k,
                                     uint32_t n_tokens,
                                     uint32_t heads,
                                     uint32_t head_dim,
                                     uint32_t start_position,
                                     float theta,
                                     float *q_out,
                                     float *k_out) {
    if (q == NULL || k == NULL || q_out == NULL || k_out == NULL || n_tokens == 0u || heads == 0u ||
        head_dim == 0u || (head_dim % 2u) != 0u || theta <= 0.0f) {
        return 0;
    }

    const uint32_t half_dim = head_dim / 2u;
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        const float position = (float)(start_position + token);
        for (uint32_t head = 0u; head < heads; ++head) {
            const size_t base = token_head_offset(token, head, heads, head_dim);
            for (uint32_t dim = 0u; dim < half_dim; ++dim) {
                const float inv_freq = powf(theta, -2.0f * (float)dim / (float)head_dim);
                const float angle = position * inv_freq;
                const float cos_value = cosf(angle);
                const float sin_value = sinf(angle);

                const float q0 = q[base + dim];
                const float q1 = q[base + half_dim + dim];
                q_out[base + dim] = q0 * cos_value - q1 * sin_value;
                q_out[base + half_dim + dim] = q1 * cos_value + q0 * sin_value;

                const float k0 = k[base + dim];
                const float k1 = k[base + half_dim + dim];
                k_out[base + dim] = k0 * cos_value - k1 * sin_value;
                k_out[base + half_dim + dim] = k1 * cos_value + k0 * sin_value;
            }
        }
    }
    return 1;
}

static float attention_dot_f32(const float *q,
                               const float *k,
                               uint32_t query_token,
                               uint32_t key_token,
                               uint32_t head,
                               uint32_t heads,
                               uint32_t head_dim) {
    const size_t q_base = token_head_offset(query_token, head, heads, head_dim);
    const size_t k_base = token_head_offset(key_token, head, heads, head_dim);
    float dot = 0.0f;
    for (uint32_t dim = 0u; dim < head_dim; ++dim) {
        dot = fmaf(q[q_base + dim], k[k_base + dim], dot);
    }
    return dot;
}

int uocr_cpu_ref_causal_sdpa_f32(const float *q,
                                 const float *k,
                                 const float *v,
                                 uint32_t n_tokens,
                                 uint32_t heads,
                                 uint32_t head_dim,
                                 float scale,
                                 float *out) {
    if (q == NULL || k == NULL || v == NULL || out == NULL || n_tokens == 0u || heads == 0u || head_dim == 0u) {
        return 0;
    }

    const float score_scale = scale > 0.0f ? scale : 1.0f / sqrtf((float)head_dim);
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        for (uint32_t head = 0u; head < heads; ++head) {
            float max_score = -INFINITY;
            for (uint32_t key_token = 0u; key_token <= token; ++key_token) {
                const float score = attention_dot_f32(q, k, token, key_token, head, heads, head_dim) * score_scale;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float denom = 0.0f;
            for (uint32_t key_token = 0u; key_token <= token; ++key_token) {
                const float score = attention_dot_f32(q, k, token, key_token, head, heads, head_dim) * score_scale;
                denom += expf(score - max_score);
            }
            if (!(denom > 0.0f)) {
                return 0;
            }

            const size_t out_base = token_head_offset(token, head, heads, head_dim);
            for (uint32_t dim = 0u; dim < head_dim; ++dim) {
                float value = 0.0f;
                for (uint32_t key_token = 0u; key_token <= token; ++key_token) {
                    const float score = attention_dot_f32(q, k, token, key_token, head, heads, head_dim) * score_scale;
                    const float weight = expf(score - max_score) / denom;
                    const size_t v_base = token_head_offset(key_token, head, heads, head_dim);
                    value = fmaf(weight, v[v_base + dim], value);
                }
                out[out_base + dim] = value;
            }
        }
    }
    return 1;
}
