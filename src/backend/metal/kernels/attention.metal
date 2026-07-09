// attention.metal - Shared attention projection/output kernels
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrAttentionProjectionParams {
    uint n_tokens;
    uint hidden_size;
    uint projection_count;
    uint reserved;
};

// Decode Q/K/V projection: one SIMD-group owns one output row and reduces with
// simd_sum, avoiding the 256-thread threadgroup reductions used by the generic
// multi-token projection kernel above.
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_attention_qkv_decode_gemv_f16_to_f16(
    device const half *src [[buffer(0)]],
    device const half *q_weight [[buffer(1)]],
    device const half *k_weight [[buffer(2)]],
    device const half *v_weight [[buffer(3)]],
    device half *q_dst [[buffer(4)]],
    device half *k_dst [[buffer(5)]],
    device half *v_dst [[buffer(6)]],
    constant UocrAttentionProjectionParams &params [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort lane_u16 [[thread_index_in_simdgroup]],
    ushort simdgroup_u16 [[simdgroup_index_in_threadgroup]],
    ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint rows_per_tg = 128u / simd_width;
    const uint row = tg * rows_per_tg + uint(simdgroup_u16);
    const uint hidden_size = params.hidden_size;
    const uint projection_count = min(params.projection_count, 3u);
    const uint total_rows = hidden_size * projection_count;
    if (row >= total_rows) {
        return;
    }
    const uint projection = row / hidden_size;
    const uint out_col = row - projection * hidden_size;

    device const half *weight = q_weight;
    device half *dst = q_dst;
    if (projection == 1u) {
        weight = k_weight;
        dst = k_dst;
    } else if (projection == 2u) {
        weight = v_weight;
        dst = v_dst;
    }

    const float value = simd_sum(uocr_decode_gemv_f16_lane_dot(src,
                                                              weight + ulong(out_col) * hidden_size,
                                                              hidden_size,
                                                              lane,
                                                              simd_width));
    if (lane == 0u) {
        dst[out_col] = half(value);
    }
}

struct UocrAttentionOutputParams {
    uint n_tokens;
    uint hidden_size;
    uint reserved0;
    uint reserved1;
};

// Decode attention output projection + residual epilogue, with one SIMD-group
// per output row.
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_attention_output_residual_decode_gemv_f16_to_f16(
    device const half *src [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *residual [[buffer(2)]],
    device half *dst [[buffer(3)]],
    constant UocrAttentionOutputParams &params [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort lane_u16 [[thread_index_in_simdgroup]],
    ushort simdgroup_u16 [[simdgroup_index_in_threadgroup]],
    ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint rows_per_tg = 128u / simd_width;
    const uint out_col = tg * rows_per_tg + uint(simdgroup_u16);
    const uint hidden_size = params.hidden_size;
    if (out_col >= hidden_size) {
        return;
    }

    const float projected = simd_sum(uocr_decode_gemv_f16_lane_dot(src,
                                                                  weight + ulong(out_col) * hidden_size,
                                                                  hidden_size,
                                                                  lane,
                                                                  simd_width));
    if (lane == 0u) {
        dst[out_col] = half(projected + float(residual[out_col]));
    }
}
