// decode_gemv_q8.metal - Simdgroup-per-row Q8_0 GEMV kernels for decode.
//
// Single-token decode is weight-bandwidth-bound.  The previous decode kernels
// used one threadgroup (with a tree reduction) per output element and scalar
// int8 loads, reaching only 10-25% of streaming bandwidth.  These kernels
// assign one simdgroup per output row with char4 weight / half4 activation
// loads and a single simd_sum - no threadgroup barriers.
#include "common.metal"

struct UocrDecodeGemvQ8Params {
    uint k;              // input features (multiple of 4; groups of 64)
    uint n;              // output rows
    uint groups_per_row; // k / 64
    uint has_residual;
};

// Shared with the prefill bucketed Q8 MoE kernels.
struct UocrMoeDecodeGemvQ8Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint expert_stride_values;
    uint up_offset_values;
    uint down_offset_values;
    uint expert_scale_stride_values;
    uint up_scale_offset_values;
    uint down_scale_offset_values;
    uint group_size;
};

// Lane-partial dot product over one Q8 row (no reduction; caller simd_sums).
// char4 runs stay within one 64-element scale group (4 divides 64).
static inline float uocr_decode_gemv_q8_lane_dot(device const half *x,
                                                 device const char *w_row,
                                                 device const half *s_row,
                                                 uint k,
                                                 uint lane,
                                                 uint simd_width) {
    device const char4 *w4 = (device const char4 *)w_row;
    device const half4 *x4 = (device const half4 *)x;
    float sum = 0.0f;
    const uint chunks = k / 4u;
    for (uint c = lane; c < chunks; c += simd_width) {
        const char4 w = w4[c];
        const half4 xv = x4[c];
        const float scale = float(s_row[(c * 4u) / 64u]);
        sum += (float(xv.x) * float(w.x) + float(xv.y) * float(w.y) +
                float(xv.z) * float(w.z) + float(xv.w) * float(w.w)) * scale;
    }
    return sum;
}

// Fused decode QKV: one simdgroup per output row across the three packed
// projections (row / n_per_projection selects Q, K or V).
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_decode_qkv_gemv_q8_0_to_f16(
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
    constant UocrDecodeGemvQ8Params &params [[buffer(10)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint rows_per_tg = ntg / simd_width;
    const uint row = tgpig * rows_per_tg + sgitg;
    if (row >= 3u * params.n) {
        return;
    }
    const uint projection = row / params.n;
    const uint local_row = row - projection * params.n;
    device const char *qweight = projection == 0u ? q_qweight : (projection == 1u ? k_qweight : v_qweight);
    device const half *qscale = projection == 0u ? q_qscale : (projection == 1u ? k_qscale : v_qscale);
    device half *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);

    const float sum = simd_sum(uocr_decode_gemv_q8_lane_dot(src,
                                                            qweight + (ulong)local_row * params.k,
                                                            qscale + (ulong)local_row * params.groups_per_row,
                                                            params.k,
                                                            lane,
                                                            simd_width));
    if (lane == 0u) {
        dst[local_row] = half(sum);
    }
}

// Decode linear with optional residual (attention O and dense/shared down).
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_decode_linear_residual_gemv_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device half *dst [[buffer(4)]],
    constant UocrDecodeGemvQ8Params &params [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint rows_per_tg = ntg / simd_width;
    const uint row = tgpig * rows_per_tg + sgitg;
    if (row >= params.n) {
        return;
    }
    float sum = simd_sum(uocr_decode_gemv_q8_lane_dot(src,
                                                      qweight + (ulong)row * params.k,
                                                      qscale + (ulong)row * params.groups_per_row,
                                                      params.k,
                                                      lane,
                                                      simd_width));
    if (lane == 0u) {
        if (params.has_residual != 0u) {
            sum += float(residual[row]);
        }
        dst[row] = half(sum);
    }
}

// Decode SwiGLU gate/up: one simdgroup computes both dots for its column.
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_decode_swiglu_gate_up_gemv_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *gate_qweight [[buffer(1)]],
    device const char *up_qweight [[buffer(2)]],
    device const half *gate_qscale [[buffer(3)]],
    device const half *up_qscale [[buffer(4)]],
    device half *mid [[buffer(5)]],
    constant UocrDecodeGemvQ8Params &params [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint rows_per_tg = ntg / simd_width;
    const uint row = tgpig * rows_per_tg + sgitg;
    if (row >= params.n) {
        return;
    }
    const float gate = simd_sum(uocr_decode_gemv_q8_lane_dot(src,
                                                             gate_qweight + (ulong)row * params.k,
                                                             gate_qscale + (ulong)row * params.groups_per_row,
                                                             params.k,
                                                             lane,
                                                             simd_width));
    const float up = simd_sum(uocr_decode_gemv_q8_lane_dot(src,
                                                           up_qweight + (ulong)row * params.k,
                                                           up_qscale + (ulong)row * params.groups_per_row,
                                                           params.k,
                                                           lane,
                                                           simd_width));
    if (lane == 0u) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[row] = half(silu * up);
    }
}

// Decode routed-MoE gate/up: one simdgroup per (rank, column).
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_moe_decode_gate_up_gemv_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const char *expert_qweight [[buffer(2)]],
    device const half *expert_qscale [[buffer(3)]],
    device half *mid [[buffer(4)]],
    constant UocrMoeDecodeGemvQ8Params &params [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint rows_per_tg = ntg / simd_width;
    const uint row = tgpig * rows_per_tg + sgitg;
    const uint intermediate_size = params.intermediate_size;
    if (row >= params.top_k * intermediate_size) {
        return;
    }
    const uint rank = row / intermediate_size;
    const uint col = row - rank * intermediate_size;
    const uint expert = top_expert_ids[rank];
    if (expert >= params.expert_count) {
        if (lane == 0u) {
            mid[row] = half(0.0f);
        }
        return;
    }
    const uint K = params.hidden_size;
    const uint groups_per_row = K / params.group_size;
    device const char *gate_qweight = expert_qweight + (ulong)expert * params.expert_stride_values + (ulong)col * K;
    device const char *up_qweight = gate_qweight + params.up_offset_values;
    device const half *gate_qscale = expert_qscale + (ulong)expert * params.expert_scale_stride_values +
                                     (ulong)col * groups_per_row;
    device const half *up_qscale = gate_qscale + params.up_scale_offset_values;

    const float gate = simd_sum(uocr_decode_gemv_q8_lane_dot(src, gate_qweight, gate_qscale, K, lane, simd_width));
    const float up = simd_sum(uocr_decode_gemv_q8_lane_dot(src, up_qweight, up_qscale, K, lane, simd_width));
    if (lane == 0u) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[row] = half(silu * up);
    }
}

// Decode routed-MoE down + weighted combine: one simdgroup per output column;
// each simdgroup accumulates the weighted lane-partials over all top-k
// experts and reduces once.
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_moe_decode_down_combine_gemv_q8_0_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const float *top_weights [[buffer(2)]],
    device const char *expert_qweight [[buffer(3)]],
    device const half *expert_qscale [[buffer(4)]],
    device const half *shared [[buffer(5)]],
    device const half *residual [[buffer(6)]],
    device half *dst [[buffer(7)]],
    constant UocrMoeDecodeGemvQ8Params &params [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint rows_per_tg = ntg / simd_width;
    const uint col = tgpig * rows_per_tg + sgitg;
    if (col >= params.hidden_size) {
        return;
    }
    const uint K = params.intermediate_size;
    const uint groups_per_row = K / params.group_size;
    float lane_sum = 0.0f;
    for (uint rank = 0u; rank < params.top_k; ++rank) {
        const uint expert = top_expert_ids[rank];
        if (expert >= params.expert_count) {
            continue;
        }
        device const char *down_qweight = expert_qweight + (ulong)expert * params.expert_stride_values +
                                          params.down_offset_values + (ulong)col * K;
        device const half *down_qscale = expert_qscale + (ulong)expert * params.expert_scale_stride_values +
                                         params.down_scale_offset_values + (ulong)col * groups_per_row;
        lane_sum += top_weights[rank] *
                    uocr_decode_gemv_q8_lane_dot(mid + (ulong)rank * K, down_qweight, down_qscale, K, lane, simd_width);
    }
    const float routed = simd_sum(lane_sum);
    if (lane == 0u) {
        dst[col] = half(routed + float(shared[col]) + float(residual[col]));
    }
}
