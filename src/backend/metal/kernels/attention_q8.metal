// attention_q8.metal - Fused Q8_0 attention projection kernels
#include "common.metal"

struct UocrAttentionQkvQ8Params {
    uint n_tokens;
    uint hidden_size;
    uint group_size;
    uint groups_per_row;
};

static inline float uocr_attention_qkv_q8_dot(device const half *src,
                                              device const char *qweight,
                                              device const half *qscale,
                                              constant UocrAttentionQkvQ8Params &params,
                                              uint token,
                                              uint out_col,
                                              uint tid,
                                              uint ntg,
                                              uint simd_width,
                                              threadgroup float *partials) {
    const uint hidden_size = params.hidden_size;
    const uint group_size = params.group_size;
    const uint groups_per_row = params.groups_per_row;
    const uint src_base = token * hidden_size;
    const uint weight_base = out_col * hidden_size;
    const uint scale_base = out_col * groups_per_row;
    float sum = 0.0f;
    for (uint k = tid; k < hidden_size; k += ntg) {
        const float scale = float(qscale[scale_base + (k / group_size)]);
        const float q = float(int(qweight[weight_base + k]));
        sum += float(src[src_base + k]) * (q * scale);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_attention_qkv_q8_0_to_f16_h1280_g64(
    device const half *src [[buffer(0)]],
    device const char *q_qweight [[buffer(1)]],
    device const char *k_qweight [[buffer(2)]],
    device const char *v_qweight [[buffer(3)]],
    device const half *q_qscale [[buffer(4)]],
    device const half *k_qscale [[buffer(5)]],
    device const half *v_qscale [[buffer(6)]],
    device half *q_dst [[buffer(7)]],
    device half *k_dst [[buffer(8)]],
    device half *v_dst [[buffer(9)]],
    constant UocrAttentionQkvQ8Params &params [[buffer(10)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = params.hidden_size;
    const uint values_per_projection = params.n_tokens * hidden_size;
    if (output_index >= 3u * values_per_projection) {
        return;
    }
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint token = element / hidden_size;
    const uint out_col = element - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    device const char *qweight = q_qweight;
    device const half *qscale = q_qscale;
    device half *dst = q_dst;
    if (projection == 1u) {
        qweight = k_qweight;
        qscale = k_qscale;
        dst = k_dst;
    } else if (projection == 2u) {
        qweight = v_qweight;
        qscale = v_qscale;
        dst = v_dst;
    }

    const float value = uocr_attention_qkv_q8_dot(src, qweight, qscale, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        dst[token * hidden_size + out_col] = half(value);
    }
}

struct UocrAttentionOutputQ8Params {
    uint n_tokens;
    uint hidden_size;
    uint group_size;
    uint groups_per_row;
};

static inline float uocr_attention_output_q8_dot(device const half *src,
                                                 device const char *qweight,
                                                 device const half *qscale,
                                                 constant UocrAttentionOutputQ8Params &params,
                                                 uint token,
                                                 uint out_col,
                                                 uint tid,
                                                 uint ntg,
                                                 uint simd_width,
                                                 threadgroup float *partials) {
    const uint hidden_size = params.hidden_size;
    const uint group_size = params.group_size;
    const uint groups_per_row = params.groups_per_row;
    const uint src_base = token * hidden_size;
    const uint weight_base = out_col * hidden_size;
    const uint scale_base = out_col * groups_per_row;
    float sum = 0.0f;
    for (uint k = tid; k < hidden_size; k += ntg) {
        const float scale = float(qscale[scale_base + (k / group_size)]);
        const float q = float(int(qweight[weight_base + k]));
        sum += float(src[src_base + k]) * (q * scale);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_attention_output_residual_q8_0_to_f16_h1280_g64(
    device const half *src [[buffer(0)]],
    device const char *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device half *dst [[buffer(4)]],
    constant UocrAttentionOutputQ8Params &params [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = params.hidden_size;
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    const float projected = uocr_attention_output_q8_dot(src, qweight, qscale, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        const uint dst_index = token * hidden_size + out_col;
        dst[dst_index] = half(projected + float(residual[dst_index]));
    }
}
