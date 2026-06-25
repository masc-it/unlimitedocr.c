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

static int checked_add_u32(uint32_t a, uint32_t b, uint32_t *out) {
    if (out == NULL || a > UINT32_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || (a != 0u && b > UINT64_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static float dense_dot_row_f32(const float *input_row,
                               const float *weight_row,
                               uint32_t input_dim) {
    float sum = 0.0f;
    for (uint32_t col = 0u; col < input_dim; ++col) {
        sum = fmaf(input_row[col], weight_row[col], sum);
    }
    return sum;
}

static float silu_f32(float value) {
    return value / (1.0f + expf(-value));
}

int uocr_cpu_ref_dense_swiglu_f32(const float *input,
                                  const float *gate_weight,
                                  const float *up_weight,
                                  const float *down_weight,
                                  uint32_t rows,
                                  uint32_t hidden_dim,
                                  uint32_t intermediate_dim,
                                  float *swiglu_workspace,
                                  float *out) {
    if (input == NULL || gate_weight == NULL || up_weight == NULL || down_weight == NULL ||
        swiglu_workspace == NULL || out == NULL || rows == 0u || hidden_dim == 0u || intermediate_dim == 0u) {
        return 0;
    }

    uint64_t input_count = 0u;
    uint64_t workspace_count = 0u;
    if (!checked_mul_u64((uint64_t)rows, (uint64_t)hidden_dim, &input_count) ||
        !checked_mul_u64((uint64_t)rows, (uint64_t)intermediate_dim, &workspace_count) ||
        input_count > (uint64_t)SIZE_MAX || workspace_count > (uint64_t)SIZE_MAX) {
        return 0;
    }

    for (uint32_t row = 0u; row < rows; ++row) {
        const float *input_row = input + (size_t)row * hidden_dim;
        float *workspace_row = swiglu_workspace + (size_t)row * intermediate_dim;
        for (uint32_t inter = 0u; inter < intermediate_dim; ++inter) {
            const float *gate_row = gate_weight + (size_t)inter * hidden_dim;
            const float *up_row = up_weight + (size_t)inter * hidden_dim;
            const float gate = dense_dot_row_f32(input_row, gate_row, hidden_dim);
            const float up = dense_dot_row_f32(input_row, up_row, hidden_dim);
            workspace_row[inter] = silu_f32(gate) * up;
        }
    }

    for (uint32_t row = 0u; row < rows; ++row) {
        const float *workspace_row = swiglu_workspace + (size_t)row * intermediate_dim;
        float *out_row = out + (size_t)row * hidden_dim;
        for (uint32_t hidden = 0u; hidden < hidden_dim; ++hidden) {
            const float *down_row = down_weight + (size_t)hidden * intermediate_dim;
            out_row[hidden] = dense_dot_row_f32(workspace_row, down_row, intermediate_dim);
        }
    }

    return 1;
}

static int checked_product3_fits_size(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t ab = 0u;
    uint64_t abc = 0u;
    return checked_mul_u64(a, b, &ab) && checked_mul_u64(ab, c, &abc) && abc <= (uint64_t)SIZE_MAX;
}

static int moe_validate_common(const float *input,
                               const float *router_weight,
                               uint32_t rows,
                               uint32_t hidden_dim,
                               uint32_t n_experts,
                               uint32_t top_k,
                               float routed_scaling_factor,
                               float *router_logits,
                               float *router_probs,
                               uint32_t *top_expert_ids,
                               float *top_expert_weights) {
    if (input == NULL || router_weight == NULL || router_logits == NULL || router_probs == NULL ||
        top_expert_ids == NULL || top_expert_weights == NULL || rows == 0u || hidden_dim == 0u ||
        n_experts == 0u || top_k == 0u || top_k > n_experts || routed_scaling_factor < 0.0f ||
        !isfinite(routed_scaling_factor)) {
        return 0;
    }
    return checked_product3_fits_size((uint64_t)rows, (uint64_t)n_experts, 1u) &&
           checked_product3_fits_size((uint64_t)rows, (uint64_t)top_k, 1u) &&
           checked_product3_fits_size((uint64_t)n_experts, (uint64_t)hidden_dim, 1u);
}

static int expert_already_selected(const uint32_t *ids, uint32_t count, uint32_t expert) {
    for (uint32_t i = 0u; i < count; ++i) {
        if (ids[i] == expert) {
            return 1;
        }
    }
    return 0;
}

int uocr_cpu_ref_moe_router_f32(const float *input,
                                const float *router_weight,
                                uint32_t rows,
                                uint32_t hidden_dim,
                                uint32_t n_experts,
                                uint32_t top_k,
                                float routed_scaling_factor,
                                float *router_logits,
                                float *router_probs,
                                uint32_t *top_expert_ids,
                                float *top_expert_weights) {
    if (!moe_validate_common(input,
                             router_weight,
                             rows,
                             hidden_dim,
                             n_experts,
                             top_k,
                             routed_scaling_factor,
                             router_logits,
                             router_probs,
                             top_expert_ids,
                             top_expert_weights)) {
        return 0;
    }

    for (uint32_t row = 0u; row < rows; ++row) {
        const float *input_row = input + (size_t)row * hidden_dim;
        float *logits_row = router_logits + (size_t)row * n_experts;
        float *probs_row = router_probs + (size_t)row * n_experts;
        float max_logit = -INFINITY;
        for (uint32_t expert = 0u; expert < n_experts; ++expert) {
            const float *router_row = router_weight + (size_t)expert * hidden_dim;
            const float logit = dense_dot_row_f32(input_row, router_row, hidden_dim);
            logits_row[expert] = logit;
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
        if (!isfinite(max_logit)) {
            return 0;
        }

        float denom = 0.0f;
        for (uint32_t expert = 0u; expert < n_experts; ++expert) {
            const float prob = expf(logits_row[expert] - max_logit);
            probs_row[expert] = prob;
            denom += prob;
        }
        if (!(denom > 0.0f) || !isfinite(denom)) {
            return 0;
        }
        for (uint32_t expert = 0u; expert < n_experts; ++expert) {
            probs_row[expert] /= denom;
        }

        uint32_t *ids_row = top_expert_ids + (size_t)row * top_k;
        float *weights_row = top_expert_weights + (size_t)row * top_k;
        for (uint32_t slot = 0u; slot < top_k; ++slot) {
            uint32_t best_expert = UINT32_MAX;
            float best_prob = -INFINITY;
            for (uint32_t expert = 0u; expert < n_experts; ++expert) {
                const float prob = probs_row[expert];
                if (expert_already_selected(ids_row, slot, expert)) {
                    continue;
                }
                if (prob > best_prob || (prob == best_prob && expert < best_expert)) {
                    best_prob = prob;
                    best_expert = expert;
                }
            }
            if (best_expert == UINT32_MAX || !(best_prob >= 0.0f)) {
                return 0;
            }
            ids_row[slot] = best_expert;
            weights_row[slot] = best_prob * routed_scaling_factor;
        }
    }
    return 1;
}

int uocr_cpu_ref_moe_swiglu_f32(const float *input,
                                const float *router_weight,
                                const float *expert_gate_weight,
                                const float *expert_up_weight,
                                const float *expert_down_weight,
                                const float *shared_gate_weight,
                                const float *shared_up_weight,
                                const float *shared_down_weight,
                                uint32_t rows,
                                uint32_t hidden_dim,
                                uint32_t n_experts,
                                uint32_t top_k,
                                uint32_t expert_intermediate_dim,
                                uint32_t shared_intermediate_dim,
                                float routed_scaling_factor,
                                float *router_logits,
                                float *router_probs,
                                uint32_t *top_expert_ids,
                                float *top_expert_weights,
                                float *expert_workspace,
                                float *shared_workspace,
                                float *out) {
    if (expert_gate_weight == NULL || expert_up_weight == NULL || expert_down_weight == NULL ||
        shared_gate_weight == NULL || shared_up_weight == NULL || shared_down_weight == NULL ||
        expert_workspace == NULL || shared_workspace == NULL || out == NULL || expert_intermediate_dim == 0u ||
        shared_intermediate_dim == 0u) {
        return 0;
    }
    if (!moe_validate_common(input,
                             router_weight,
                             rows,
                             hidden_dim,
                             n_experts,
                             top_k,
                             routed_scaling_factor,
                             router_logits,
                             router_probs,
                             top_expert_ids,
                             top_expert_weights)) {
        return 0;
    }
    if (!checked_product3_fits_size((uint64_t)n_experts, (uint64_t)expert_intermediate_dim, (uint64_t)hidden_dim) ||
        !checked_product3_fits_size((uint64_t)n_experts, (uint64_t)hidden_dim, (uint64_t)expert_intermediate_dim) ||
        !checked_product3_fits_size((uint64_t)rows, (uint64_t)top_k, (uint64_t)expert_intermediate_dim) ||
        !checked_product3_fits_size((uint64_t)rows, (uint64_t)shared_intermediate_dim, 1u) ||
        !checked_product3_fits_size((uint64_t)shared_intermediate_dim, (uint64_t)hidden_dim, 1u) ||
        !checked_product3_fits_size((uint64_t)hidden_dim, (uint64_t)shared_intermediate_dim, 1u)) {
        return 0;
    }

    if (!uocr_cpu_ref_moe_router_f32(input,
                                     router_weight,
                                     rows,
                                     hidden_dim,
                                     n_experts,
                                     top_k,
                                     routed_scaling_factor,
                                     router_logits,
                                     router_probs,
                                     top_expert_ids,
                                     top_expert_weights)) {
        return 0;
    }

    for (uint32_t row = 0u; row < rows; ++row) {
        const float *input_row = input + (size_t)row * hidden_dim;
        float *out_row = out + (size_t)row * hidden_dim;
        for (uint32_t hidden = 0u; hidden < hidden_dim; ++hidden) {
            out_row[hidden] = 0.0f;
        }

        for (uint32_t slot = 0u; slot < top_k; ++slot) {
            const uint32_t expert = top_expert_ids[(size_t)row * top_k + slot];
            const float route_weight = top_expert_weights[(size_t)row * top_k + slot];
            if (expert >= n_experts) {
                return 0;
            }
            float *expert_intermediate = expert_workspace + ((size_t)row * top_k + slot) * expert_intermediate_dim;
            for (uint32_t inter = 0u; inter < expert_intermediate_dim; ++inter) {
                const size_t expert_inter_offset = ((size_t)expert * expert_intermediate_dim + inter) * hidden_dim;
                const float gate = dense_dot_row_f32(input_row, expert_gate_weight + expert_inter_offset, hidden_dim);
                const float up = dense_dot_row_f32(input_row, expert_up_weight + expert_inter_offset, hidden_dim);
                expert_intermediate[inter] = silu_f32(gate) * up;
            }
            for (uint32_t hidden = 0u; hidden < hidden_dim; ++hidden) {
                const size_t down_offset = ((size_t)expert * hidden_dim + hidden) * expert_intermediate_dim;
                const float down = dense_dot_row_f32(expert_intermediate,
                                                     expert_down_weight + down_offset,
                                                     expert_intermediate_dim);
                out_row[hidden] = fmaf(route_weight, down, out_row[hidden]);
            }
        }

        float *shared_intermediate = shared_workspace + (size_t)row * shared_intermediate_dim;
        for (uint32_t inter = 0u; inter < shared_intermediate_dim; ++inter) {
            const float gate = dense_dot_row_f32(input_row,
                                                 shared_gate_weight + (size_t)inter * hidden_dim,
                                                 hidden_dim);
            const float up = dense_dot_row_f32(input_row,
                                               shared_up_weight + (size_t)inter * hidden_dim,
                                               hidden_dim);
            shared_intermediate[inter] = silu_f32(gate) * up;
        }
        for (uint32_t hidden = 0u; hidden < hidden_dim; ++hidden) {
            const float shared = dense_dot_row_f32(shared_intermediate,
                                                   shared_down_weight + (size_t)hidden * shared_intermediate_dim,
                                                   shared_intermediate_dim);
            out_row[hidden] += shared;
        }
    }
    return 1;
}

int uocr_cpu_ref_kv_cache_layout_init(uint32_t prompt_token_capacity,
                                      uint32_t generated_ring_window,
                                      uint32_t heads,
                                      uint32_t head_dim,
                                      uocr_cpu_ref_kv_cache_layout *out_layout) {
    if (out_layout == NULL || prompt_token_capacity == 0u || generated_ring_window == 0u || heads == 0u ||
        head_dim == 0u) {
        return 0;
    }

    uint32_t cache_token_capacity = 0u;
    if (!checked_add_u32(prompt_token_capacity, generated_ring_window, &cache_token_capacity)) {
        return 0;
    }

    uint64_t token_stride = 0u;
    if (!checked_mul_u64((uint64_t)heads, (uint64_t)head_dim, &token_stride)) {
        return 0;
    }
    uint64_t total_floats = 0u;
    if (!checked_mul_u64((uint64_t)cache_token_capacity, token_stride, &total_floats)) {
        return 0;
    }

    memset(out_layout, 0, sizeof(*out_layout));
    out_layout->prompt_token_capacity = prompt_token_capacity;
    out_layout->generated_ring_window = generated_ring_window;
    out_layout->heads = heads;
    out_layout->head_dim = head_dim;
    out_layout->cache_token_capacity = cache_token_capacity;
    out_layout->token_stride_floats = token_stride;
    out_layout->total_floats = total_floats;
    return 1;
}

int uocr_cpu_ref_kv_cache_token_for_position(uint32_t prompt_length,
                                             const uocr_cpu_ref_kv_cache_layout *layout,
                                             uint32_t position,
                                             uint32_t *out_cache_token) {
    if (layout == NULL || out_cache_token == NULL || prompt_length == 0u ||
        prompt_length > layout->prompt_token_capacity || layout->generated_ring_window == 0u) {
        return 0;
    }
    if (position < prompt_length) {
        *out_cache_token = position;
        return 1;
    }

    const uint32_t generated_index = position - prompt_length;
    uint32_t cache_token = 0u;
    if (!checked_add_u32(prompt_length, generated_index % layout->generated_ring_window, &cache_token) ||
        cache_token >= layout->cache_token_capacity) {
        return 0;
    }
    *out_cache_token = cache_token;
    return 1;
}

static int kv_cache_token_offset(const uocr_cpu_ref_kv_cache_layout *layout,
                                 uint32_t cache_token,
                                 uint64_t *out_offset) {
    if (layout == NULL || out_offset == NULL || cache_token >= layout->cache_token_capacity ||
        layout->token_stride_floats == 0u) {
        return 0;
    }
    return checked_mul_u64((uint64_t)cache_token, layout->token_stride_floats, out_offset);
}

int uocr_cpu_ref_kv_cache_write_token_f32(float *k_cache,
                                          float *v_cache,
                                          const uocr_cpu_ref_kv_cache_layout *layout,
                                          uint32_t prompt_length,
                                          uint32_t position,
                                          const float *k_token,
                                          const float *v_token) {
    if (k_cache == NULL || v_cache == NULL || layout == NULL || k_token == NULL || v_token == NULL) {
        return 0;
    }

    uint32_t cache_token = 0u;
    uint64_t offset = 0u;
    if (!uocr_cpu_ref_kv_cache_token_for_position(prompt_length, layout, position, &cache_token) ||
        !kv_cache_token_offset(layout, cache_token, &offset) ||
        offset > UINT64_MAX - layout->token_stride_floats || offset + layout->token_stride_floats > layout->total_floats) {
        return 0;
    }

    memcpy(k_cache + offset, k_token, (size_t)layout->token_stride_floats * sizeof(float));
    memcpy(v_cache + offset, v_token, (size_t)layout->token_stride_floats * sizeof(float));
    return 1;
}

int uocr_cpu_ref_kv_cache_read_token_f32(const float *k_cache,
                                         const float *v_cache,
                                         const uocr_cpu_ref_kv_cache_layout *layout,
                                         uint32_t cache_token,
                                         float *k_token_out,
                                         float *v_token_out) {
    if (k_cache == NULL || v_cache == NULL || layout == NULL || k_token_out == NULL || v_token_out == NULL) {
        return 0;
    }

    uint64_t offset = 0u;
    if (!kv_cache_token_offset(layout, cache_token, &offset) ||
        offset > UINT64_MAX - layout->token_stride_floats || offset + layout->token_stride_floats > layout->total_floats) {
        return 0;
    }

    memcpy(k_token_out, k_cache + offset, (size_t)layout->token_stride_floats * sizeof(float));
    memcpy(v_token_out, v_cache + offset, (size_t)layout->token_stride_floats * sizeof(float));
    return 1;
}

int uocr_cpu_ref_kv_cache_decode_attention_plan(uint32_t prompt_length,
                                                const uocr_cpu_ref_kv_cache_layout *layout,
                                                uint32_t generated_count,
                                                uocr_cpu_ref_decode_attention_plan *out_plan) {
    if (layout == NULL || out_plan == NULL || prompt_length == 0u || prompt_length > layout->prompt_token_capacity ||
        layout->generated_ring_window == 0u || generated_count > UINT32_MAX - prompt_length) {
        return 0;
    }

    const uint32_t live_generated = generated_count < layout->generated_ring_window ? generated_count
                                                                                    : layout->generated_ring_window;
    uint32_t attention_length = 0u;
    if (!checked_add_u32(prompt_length, live_generated, &attention_length)) {
        return 0;
    }

    const uint32_t first_generated_index = generated_count - live_generated;
    uint32_t first_generated_position = 0u;
    if (!checked_add_u32(prompt_length, first_generated_index, &first_generated_position)) {
        return 0;
    }
    const uint32_t query_position = generated_count == 0u ? prompt_length : prompt_length + generated_count - 1u;

    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->prompt_length = prompt_length;
    out_plan->prompt_token_capacity = layout->prompt_token_capacity;
    out_plan->cache_token_capacity = layout->cache_token_capacity;
    out_plan->generated_count = generated_count;
    out_plan->live_generated = live_generated;
    out_plan->first_generated_index = first_generated_index;
    out_plan->first_generated_position = first_generated_position;
    out_plan->query_position = query_position;
    out_plan->attention_length = attention_length;
    out_plan->generated_ring_window = layout->generated_ring_window;
    return 1;
}

int uocr_cpu_ref_kv_cache_decode_attention_index_to_token(const uocr_cpu_ref_decode_attention_plan *plan,
                                                          uint32_t attention_index,
                                                          uint32_t *out_cache_token) {
    if (plan == NULL || out_cache_token == NULL || plan->prompt_length == 0u || plan->generated_ring_window == 0u ||
        attention_index >= plan->attention_length) {
        return 0;
    }
    if (attention_index < plan->prompt_length) {
        *out_cache_token = attention_index;
        return 1;
    }

    const uint32_t generated_offset = attention_index - plan->prompt_length;
    if (generated_offset >= plan->live_generated) {
        return 0;
    }
    const uint32_t logical_generated = plan->first_generated_index + generated_offset;
    uint32_t cache_token = 0u;
    if (!checked_add_u32(plan->prompt_length, logical_generated % plan->generated_ring_window, &cache_token) ||
        cache_token >= plan->cache_token_capacity) {
        return 0;
    }
    *out_cache_token = cache_token;
    return 1;
}
