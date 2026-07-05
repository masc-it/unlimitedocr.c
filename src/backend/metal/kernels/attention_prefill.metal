// attention_prefill.metal - Prefill attention (varlen)
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrPrefillAttentionParams {
    uint n_tokens;
    uint heads;
    uint head_dim;
    float scale;
};

struct UocrPrefillAttentionVarlenParams {
    uint total_tokens;
    uint batch;
    uint heads;
    uint head_dim;
    float scale;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline void uocr_prefill_varlen_find_sequence(device const uint *cu,
                                                     constant UocrPrefillAttentionVarlenParams &params,
                                                     uint token,
                                                     thread uint &seq_start,
                                                     thread uint &seq_end) {
    seq_start = 0u;
    seq_end = 0u;
    for (uint b = 0u; b < params.batch; ++b) {
        const uint start = cu[b];
        const uint end = cu[b + 1u];
        if (token >= start && token < end) {
            seq_start = start;
            seq_end = end;
            return;
        }
    }
}

kernel void uocr_prefill_attention_varlen_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                     device const half *k_src [[buffer(1)]],
                                                     device const half *v_src [[buffer(2)]],
                                                     device const uint *cu [[buffer(3)]],
                                                     device half *dst [[buffer(4)]],
                                                     constant UocrPrefillAttentionVarlenParams &params [[buffer(5)]],
                                                     threadgroup float *partials [[threadgroup(0)]],
                                                     uint group_index [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]],
                                                     uint ntg [[threads_per_threadgroup]],
                                                     uint simd_width [[threads_per_simdgroup]]) {
    const uint token = group_index / params.heads;
    const uint head = group_index - token * params.heads;
    if (token >= params.total_tokens || head >= params.heads || ntg < params.head_dim) {
        return;
    }
    const bool dim_valid = tid < params.head_dim;

    uint seq_start;
    uint seq_end;
    uocr_prefill_varlen_find_sequence(cu, params, token, seq_start, seq_end);
    if (seq_end <= seq_start) {
        return;
    }

    const ulong q_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
    const float q = dim_valid ? float(q_src[q_index]) : 0.0f;
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    for (uint key_token = seq_start; key_token <= token; ++key_token) {
        const ulong k_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        const float local_dot = dim_valid ? q * float(k_src[k_index]) : 0.0f;
        const float score = uocr_threadgroup_sum(local_dot, partials, tid, ntg, simd_width) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        if (dim_valid) {
            acc = acc * corr + e * float(v_src[v_index]);
        }
        l = l * corr + e;
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (dim_valid) {
        const ulong dst_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        dst[dst_index] = half(l > 0.0f ? acc / l : 0.0f);
    }
}

// Prefill flash attention (template + f16/f32 kernels)
template <typename out_t>
static inline void uocr_prefill_attention_flash_impl(device const half *q_src,
                                                     device const half *k_src,
                                                     device const half *v_src,
                                                     device out_t *dst,
                                                     constant UocrPrefillAttentionParams &params,
                                                     uint2 tg,
                                                     ushort lane_u16,
                                                     ushort simdgroup_u16,
                                                     ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const float scale = uocr_fc_attention_scale_or(params.scale);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || query_token >= params.n_tokens || head >= heads ||
        head_dim == 0u || head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong q_index = (ulong(query_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token <= query_token; ++key_token) {
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong k_index = (ulong(key_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
                local_dot += qv[i] * float(k_src[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong v_index = (ulong(key_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong dst_index = (ulong(query_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

[[max_total_threads_per_threadgroup(512)]] kernel void uocr_prefill_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                    device const half *k_src [[buffer(1)]],
                                                    device const half *v_src [[buffer(2)]],
                                                    device half *dst [[buffer(3)]],
                                                    constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                                    uint2 tg [[threadgroup_position_in_grid]],
                                                    ushort lane [[thread_index_in_simdgroup]],
                                                    ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                    ushort simd_width [[threads_per_simdgroup]]) {
    uocr_prefill_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup, simd_width);
}

[[max_total_threads_per_threadgroup(512)]] kernel void uocr_prefill_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                    device const half *k_src [[buffer(1)]],
                                                    device const half *v_src [[buffer(2)]],
                                                    device float *dst [[buffer(3)]],
                                                    constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                                    uint2 tg [[threadgroup_position_in_grid]],
                                                    ushort lane [[thread_index_in_simdgroup]],
                                                    ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                    ushort simd_width [[threads_per_simdgroup]]) {
    uocr_prefill_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup, simd_width);
}
