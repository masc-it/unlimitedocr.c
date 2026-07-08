// decode_gemv_q4.metal - Simdgroup-per-row Q4_0 GEMV kernels for decode.
//
// Same regime as decode_gemv_q8.metal: single-token decode is
// weight-bandwidth-bound, one simdgroup owns one output row, lanes reduce
// with a single simd_sum and no threadgroup barriers.  Q4_0 halves the weight
// bytes; the group-half-split nibble packing (see docs/plan_q4.md §1.1) keeps
// the unpack vectorized: a 4-byte load yields two float4 halves that each
// pair with an aligned half4 activation load.  Measured ~2x vs the Q8 decode
// GEMV on M1 Pro.
#include "common.metal"

// Mirrors UocrMoeDecodeGemvQ8Params (decode_gemv_q8.metal is an earlier
// fragment); stride/offset fields are PACKED BYTE counts for Q4 qweights and
// value counts for scales.
struct UocrMoeDecodeGemvQ4Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint expert_stride_bytes;
    uint up_offset_bytes;
    uint down_offset_bytes;
    uint expert_scale_stride_values;
    uint up_scale_offset_values;
    uint down_scale_offset_values;
    uint group_size;
};

// Lane-partial dot product over one Q4_0 row (no reduction; caller
// simd_sums).  Group-half split: uchar4 chunk c covers weights
// k = g*64 + j..j+3 (low nibbles) and k+32..k+35 (high nibbles) with
// g = c/8, j = (c%8)*4 — both halves stay inside one 64-wide scale group.
static inline float uocr_decode_gemv_q4_lane_dot(device const half *x,
                                                 device const uchar *w_row,
                                                 device const half *s_row,
                                                 uint k,
                                                 uint lane,
                                                 uint simd_width) {
    device const uchar4 *w4 = (device const uchar4 *)w_row;
    device const half4 *x4 = (device const half4 *)x;
    float sum = 0.0f;
    const uint chunks = k / 8u; // 4 packed bytes -> 8 weights
    for (uint c = lane; c < chunks; c += simd_width) {
        const uchar4 w = w4[c];
        const uint group = c / 8u;       // 8 uchar4 chunks per 64-wide group
        const uint j4 = c - group * 8u;  // half4 index inside the low half
        const float scale = float(s_row[group]);
        const float4 lo = float4(int4(w) & 0xF) - 8.0f;
        const float4 hi = float4(int4(w) >> 4) - 8.0f;
        const uint xbase = group * 16u + j4; // 16 half4 per 64-wide group
        float d = dot(float4(x4[xbase]), lo);
        d += dot(float4(x4[xbase + 8u]), hi);
        sum += d * scale;
    }
    return sum;
}

// Decode routed-MoE gate/up: one simdgroup per (rank, column).
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_moe_decode_gate_up_gemv_q4_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const uchar *expert_qweight [[buffer(2)]],
    device const half *expert_qscale [[buffer(3)]],
    device half *mid [[buffer(4)]],
    constant UocrMoeDecodeGemvQ4Params &params [[buffer(5)]],
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
    device const uchar *gate_qweight = expert_qweight + (ulong)expert * params.expert_stride_bytes +
                                       (ulong)col * (K / 2u);
    device const uchar *up_qweight = gate_qweight + params.up_offset_bytes;
    device const half *gate_qscale = expert_qscale + (ulong)expert * params.expert_scale_stride_values +
                                     (ulong)col * groups_per_row;
    device const half *up_qscale = gate_qscale + params.up_scale_offset_values;

    const float gate = simd_sum(uocr_decode_gemv_q4_lane_dot(src, gate_qweight, gate_qscale, K, lane, simd_width));
    const float up = simd_sum(uocr_decode_gemv_q4_lane_dot(src, up_qweight, up_qscale, K, lane, simd_width));
    if (lane == 0u) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[row] = half(silu * up);
    }
}

// Decode routed-MoE down + weighted combine: one simdgroup per output column;
// each simdgroup accumulates the weighted lane-partials over all top-k
// experts and reduces once.
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_moe_decode_down_combine_gemv_q4_0_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const float *top_weights [[buffer(2)]],
    device const uchar *expert_qweight [[buffer(3)]],
    device const half *expert_qscale [[buffer(4)]],
    device const half *shared [[buffer(5)]],
    device const half *residual [[buffer(6)]],
    device half *dst [[buffer(7)]],
    constant UocrMoeDecodeGemvQ4Params &params [[buffer(8)]],
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
        device const uchar *down_qweight = expert_qweight + (ulong)expert * params.expert_stride_bytes +
                                           params.down_offset_bytes + (ulong)col * (K / 2u);
        device const half *down_qscale = expert_qscale + (ulong)expert * params.expert_scale_stride_values +
                                         params.down_scale_offset_values + (ulong)col * groups_per_row;
        lane_sum += top_weights[rank] *
                    uocr_decode_gemv_q4_lane_dot(mid + (ulong)rank * K, down_qweight, down_qscale, K, lane, simd_width);
    }
    const float routed = simd_sum(lane_sum);
    if (lane == 0u) {
        dst[col] = half(routed + float(shared[col]) + float(residual[col]));
    }
}
