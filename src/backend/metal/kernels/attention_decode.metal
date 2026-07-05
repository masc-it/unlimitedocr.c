// attention_decode.metal - Decode attention params
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrDecodeAttentionParams {
    uint batch_slots;
    uint cache_token_capacity;
    uint layer;
    uint slot;
    uint prompt_length;
    uint generated_count;
    uint attention_length;
    uint first_generated;
    uint heads;
    uint head_dim;
    uint ring_window;
    float scale;
};

/* OCR adapts the gradients.c sdpa_decode prefix/window rule to a compact
 * attention stream: [all prompt prefix tokens] followed by the live generated
 * window. The generated window is a 128-token ring whose logical first token is
 * supplied by the host as first_generated.
 */
static inline uint uocr_decode_attention_cache_token(constant UocrDecodeAttentionParams &params,
                                                     uint attention_index,
                                                     uint ring_window) {
    if (attention_index < params.prompt_length) {
        return attention_index;
    }
    const uint generated_index = params.first_generated + (attention_index - params.prompt_length);
    return params.prompt_length + (generated_index % ring_window);
}

static inline ulong uocr_decode_attention_cache_index(constant UocrDecodeAttentionParams &params,
                                                      uint heads,
                                                      uint head_dim,
                                                      uint cache_token,
                                                      uint head,
                                                      uint dim) {
    return (((ulong(params.layer) * ulong(params.batch_slots) + ulong(params.slot)) *
             ulong(params.cache_token_capacity) + ulong(cache_token)) *
            ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
}

#define UOCR_FLASH_Q_PER_TG 4u
#define UOCR_FLASH_MAX_LANE_VALUES 4u
#define UOCR_FLASH_NEG_INF (-3.4028234663852886e38f)

static inline uint uocr_flash_lane_dim(uint lane, uint component, uint simd_width) {
    return lane + component * simd_width;
}

template <typename out_t>
static inline void uocr_sam_window_attention_flash_impl(device const half *q_src,
                                                        device const half *k_src,
                                                        device const half *v_src,
                                                        device out_t *dst,
                                                        constant UocrSamWindowAttentionParams &params,
                                                        uint3 tg,
                                                        ushort lane_u16,
                                                        ushort simdgroup_u16,
                                                        ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint batch_size = uocr_sam_window_attention_batch_size(params);
    const uint logical_window = tg.z;
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || params.windows == 0u || batch >= batch_size || window >= params.windows ||
        query_token >= params.tokens_per_window || head >= params.heads ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong q_index = uocr_sam_window_attention_index(params, batch, window, query_token, head, dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token < params.tokens_per_window; ++key_token) {
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong k_index = uocr_sam_window_attention_index(params, batch, window, key_token, head, dim);
                local_dot += qv[i] * float(k_src[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong v_index = uocr_sam_window_attention_index(params, batch, window, key_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong dst_index = uocr_sam_window_attention_index(params, batch, window, query_token, head, dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

// Decode flash attention (template + f16/f32 kernels)
static inline void uocr_decode_attention_flash_impl(device const half *q_src,
                                                    device const half *k_cache,
                                                    device const half *v_cache,
                                                    device out_t *dst,
                                                    constant UocrDecodeAttentionParams &params,
                                                    uint head,
                                                    ushort lane_u16,
                                                    ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const uint ring_window = uocr_fc_ring_window_or(params.ring_window);
    const float scale = uocr_fc_attention_scale_or(params.scale);
    if (head >= heads || params.attention_length == 0u ||
        head_dim == 0u || head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong q_index = ulong(head) * ulong(head_dim) + ulong(dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint attention_index = 0u; attention_index < params.attention_length; ++attention_index) {
        const uint cache_token = uocr_decode_attention_cache_token(params, attention_index, ring_window);
        if (cache_token >= params.cache_token_capacity) {
            continue;
        }
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong k_index = uocr_decode_attention_cache_index(params, heads, head_dim, cache_token, head, dim);
                local_dot += qv[i] * float(k_cache[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong v_index = uocr_decode_attention_cache_index(params, heads, head_dim, cache_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_cache[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong dst_index = ulong(head) * ulong(head_dim) + ulong(dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_decode_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                   device const half *k_cache [[buffer(1)]],
                                                   device const half *v_cache [[buffer(2)]],
                                                   device half *dst [[buffer(3)]],
                                                   constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                                   uint head [[threadgroup_position_in_grid]],
                                                   ushort lane [[thread_index_in_simdgroup]],
                                                   ushort simd_width [[threads_per_simdgroup]]) {
    uocr_decode_attention_flash_impl(q_src, k_cache, v_cache, dst, params, head, lane, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_decode_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                   device const half *k_cache [[buffer(1)]],
                                                   device const half *v_cache [[buffer(2)]],
                                                   device float *dst [[buffer(3)]],
                                                   constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                                   uint head [[threadgroup_position_in_grid]],
                                                   ushort lane [[thread_index_in_simdgroup]],
                                                   ushort simd_width [[threads_per_simdgroup]]) {
    uocr_decode_attention_flash_impl(q_src, k_cache, v_cache, dst, params, head, lane, simd_width);
}
