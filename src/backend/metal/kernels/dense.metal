// dense.metal - Bias add / dense decode kernels
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrBiasAddParams {
    uint rows;
    uint cols;
    uint reserved0;
    uint reserved1;
};

kernel void uocr_bias_add_f16_inplace(device half *dst [[buffer(0)]],
                                      device const half *bias [[buffer(1)]],
                                      constant UocrBiasAddParams &params [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    const uint vec_cols = uocr_div_up_u32(params.cols, UOCR_HALF4_WIDTH);
    const uint vector_count = params.rows * vec_cols;
    if (gid >= vector_count || vec_cols == 0u) {
        return;
    }
    const uint row = gid / vec_cols;
    const uint col = (gid - row * vec_cols) * UOCR_HALF4_WIDTH;
    const ulong base = (ulong(row) * ulong(params.cols)) + ulong(col);
    if (col + (UOCR_HALF4_WIDTH - 1u) < params.cols) {
        const half4 values = uocr_load_half4(dst, base);
        const half4 bias_values = uocr_load_half4(bias, ulong(col));
        uocr_store_half4(dst, base, half4(float4(values) + float4(bias_values)));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.cols; ++i) {
            const ulong idx = base + ulong(i);
            dst[idx] = half(float(dst[idx]) + float(bias[col + i]));
        }
    }
}

/*
 * Fused bias + residual add: dst[row, col] = src[row, col] + bias[col] +
 * residual[row, col].  Replaces the separate bias-add and residual-add passes
 * after the vision fp16 attention output projection.
 *
 * Dispatch: rows * (cols / 4) half4 vectors; cols must be a multiple of 4.
 */
kernel void uocr_bias_residual_add_f16(device const half *src [[buffer(0)]],
                                       device const half *bias [[buffer(1)]],
                                       device const half *residual [[buffer(2)]],
                                       device half *dst [[buffer(3)]],
                                       constant UocrBiasAddParams &params [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]]) {
    const uint vec_cols = params.cols / UOCR_HALF4_WIDTH;
    const uint vector_count = params.rows * vec_cols;
    if (gid >= vector_count || vec_cols == 0u) {
        return;
    }
    const uint row = gid / vec_cols;
    const uint col = (gid - row * vec_cols) * UOCR_HALF4_WIDTH;
    const ulong base = (ulong(row) * ulong(params.cols)) + ulong(col);
    const half4 values = uocr_load_half4(src, base);
    const half4 bias_values = uocr_load_half4(bias, ulong(col));
    const half4 residual_values = uocr_load_half4(residual, base);
    uocr_store_half4(dst, base, half4(float4(values) + float4(bias_values) + float4(residual_values)));
}

// Shared activation: uocr_quickgelu
static inline float uocr_quickgelu(float x) {
    return x / (1.0f + exp(-1.702f * x));
}

kernel void uocr_bias_quickgelu_f16_inplace(device half *dst [[buffer(0)]],
                                            device const half *bias [[buffer(1)]],
                                            constant UocrBiasAddParams &params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    const uint value_count = params.rows * params.cols;
    if (gid >= value_count) {
        return;
    }
    const uint col = gid - (gid / params.cols) * params.cols;
    const float value = float(dst[gid]) + float(bias[col]);
    dst[gid] = half(uocr_quickgelu(value));
}

// Dense SwiGLU kernels
struct UocrDenseSwigluParams {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint has_residual;
};

struct UocrDecodeGemvF16Params {
    uint k;
    uint n;
    uint has_residual;
    uint reserved;
};

// Decode-only fp16 SwiGLU gate/up GEMV.  One simdgroup owns one output row,
// matching the optimized Q8 decode regime but reading fp16 weights directly.
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_decode_swiglu_gate_up_gemv_f16_to_f16(
    device const half *src [[buffer(0)]],
    device const half *gate_weight [[buffer(1)]],
    device const half *up_weight [[buffer(2)]],
    device half *mid [[buffer(3)]],
    constant UocrDecodeGemvF16Params &params [[buffer(4)]],
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
    device const half *gate_row = gate_weight + (ulong)row * params.k;
    device const half *up_row = up_weight + (ulong)row * params.k;
    const float gate = simd_sum(uocr_decode_gemv_f16_lane_dot(src, gate_row, params.k, lane, simd_width));
    const float up = simd_sum(uocr_decode_gemv_f16_lane_dot(src, up_row, params.k, lane, simd_width));
    if (lane == 0u) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[row] = half(silu * up);
    }
}

// Decode-only fp16 linear GEMV with optional residual add (dense/shared down).
[[max_total_threads_per_threadgroup(128)]] kernel void uocr_decode_linear_residual_gemv_f16_to_f16(
    device const half *src [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *residual [[buffer(2)]],
    device half *dst [[buffer(3)]],
    constant UocrDecodeGemvF16Params &params [[buffer(4)]],
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
    float value = simd_sum(uocr_decode_gemv_f16_lane_dot(src,
                                                          weight + (ulong)row * params.k,
                                                          params.k,
                                                          lane,
                                                          simd_width));
    if (lane == 0u) {
        if (params.has_residual != 0u) {
            value += float(residual[row]);
        }
        dst[row] = half(value);
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_gate_up_f16(device const half *src [[buffer(0)]],
                                          device const half *gate_weight [[buffer(1)]],
                                          device const half *up_weight [[buffer(2)]],
                                          device half *mid [[buffer(3)]],
                                          constant UocrDenseSwigluParams &params [[buffer(4)]],
                                          threadgroup float *partials [[threadgroup(0)]],
                                          uint output_index [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]],
                                          uint simd_width [[threads_per_simdgroup]]) {
    const uint token = output_index / params.intermediate_size;
    const uint out_col = output_index - token * params.intermediate_size;
    if (token >= params.n_tokens || out_col >= params.intermediate_size) {
        return;
    }

    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint src_base = token * hidden_size;
    const uint weight_base = out_col * hidden_size;
    for (uint k = tid; k < hidden_size; k += ntg) {
        const float x = float(src[src_base + k]);
        gate_sum += x * float(gate_weight[weight_base + k]);
        up_sum += x * float(up_weight[weight_base + k]);
    }
    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, gate, up);

    if (tid == 0) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * params.intermediate_size + out_col] = half(silu * up);
    }
}

static inline void uocr_dense_swiglu_partition256_sum2(float value0,
                                                       float value1,
                                                       threadgroup float *partials,
                                                       uint group,
                                                       uint groups,
                                                       uint local_tid,
                                                       uint simd_width,
                                                       thread float &out0,
                                                       thread float &out1) {
    const uint threads_per_column = 256u;
    const uint lane = uocr_simd_lane_from_tid(local_tid, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(local_tid, simd_width);
    const uint simdgroups = uocr_simdgroups_for_threadgroup(threads_per_column, simd_width);
    threadgroup float *partials0 = partials + group * simdgroups;
    threadgroup float *partials1 = partials + groups * simdgroups + group * simdgroups;
    const float simd_total0 = simd_sum(value0);
    const float simd_total1 = simd_sum(value1);
    if (lane == 0u) {
        partials0[simdgroup] = simd_total0;
        partials1[simdgroup] = simd_total1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = simdgroups;
    while (active > 1u) {
        const uint upper = (active + 1u) >> 1u;
        if (local_tid < active - upper) {
            partials0[local_tid] += partials0[local_tid + upper];
            partials1[local_tid] += partials1[local_tid + upper];
        }
        active = upper;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out0 = partials0[0];
    out1 = partials1[0];
}

// Decode-only shared-expert gate/up tiling.  Four columns share one
// threadgroup while each column still receives the same 256-thread
// dot-product partition/reduction shape as uocr_dense_swiglu_gate_up_f16,
// preserving the fp32 accumulation pattern before the fp16 SwiGLU
// intermediate store.  Threadgroup hidden staging was tested separately and
// left out because the extra uniform barrier outweighed reuse on the current
// GPU; the tiled columns still hit the same hidden row in cache.
[[max_total_threads_per_threadgroup(1024)]] kernel void uocr_dense_swiglu_shared_gate_up_tile4_f16(device const half *src [[buffer(0)]],
                                          device const half *gate_weight [[buffer(1)]],
                                          device const half *up_weight [[buffer(2)]],
                                          device half *mid [[buffer(3)]],
                                          constant UocrDenseSwigluParams &params [[buffer(4)]],
                                          threadgroup float *partials [[threadgroup(0)]],
                                          uint output_tile [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]],
                                          uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint threads_per_column = 256u;
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint intermediate_size = uocr_fc_moe_shared_intermediate_or(params.intermediate_size);
    const uint tiles_per_token = uocr_div_up_u32(intermediate_size, tile_columns);
    const uint token = output_tile / tiles_per_token;
    const uint tile = output_tile - token * tiles_per_token;
    if (token >= params.n_tokens || ntg != tile_columns * threads_per_column || hidden_size != 1280u) {
        return;
    }

    const uint src_base = token * hidden_size;
    const uint group = tid / threads_per_column;
    const uint local_tid = tid - group * threads_per_column;
    const uint out_col = tile * tile_columns + group;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (out_col < intermediate_size) {
        const uint weight_base = out_col * hidden_size;
        for (uint k = local_tid; k < hidden_size; k += threads_per_column) {
            const float x = float(src[src_base + k]);
            gate_sum += x * float(gate_weight[weight_base + k]);
            up_sum += x * float(up_weight[weight_base + k]);
        }
    }
    float gate = 0.0f;
    float up = 0.0f;
    uocr_dense_swiglu_partition256_sum2(gate_sum,
                                        up_sum,
                                        partials,
                                        group,
                                        tile_columns,
                                        local_tid,
                                        simd_width,
                                        gate,
                                        up);

    if (local_tid == 0u && out_col < intermediate_size) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * intermediate_size + out_col] = half(silu * up);
    }
}

static inline float uocr_dense_swiglu_down_dot_f16(device const half *mid,
                                                   device const half *down_weight,
                                                   constant UocrDenseSwigluParams &params,
                                                   uint token,
                                                   uint out_col,
                                                   uint tid,
                                                   uint ntg,
                                              uint simd_width,
                                                   threadgroup float *partials) {
    float sum = 0.0f;
    const uint mid_base = token * params.intermediate_size;
    const uint weight_base = out_col * params.intermediate_size;
    for (uint k = tid; k < params.intermediate_size; k += ntg) {
        sum += float(mid[mid_base + k]) * float(down_weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_down_f16_to_f16(device const half *mid [[buffer(0)]],
                                              device const half *down_weight [[buffer(1)]],
                                              device const half *residual [[buffer(2)]],
                                              device half *dst [[buffer(3)]],
                                              constant UocrDenseSwigluParams &params [[buffer(4)]],
                                              threadgroup float *partials [[threadgroup(0)]],
                                              uint output_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]],
                                              uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    float value = uocr_dense_swiglu_down_dot_f16(mid, down_weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        const uint dst_index = token * hidden_size + out_col;
        if (params.has_residual != 0u) {
            value += float(residual[dst_index]);
        }
        dst[dst_index] = half(value);
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_down_f16_to_f32(device const half *mid [[buffer(0)]],
                                              device const half *down_weight [[buffer(1)]],
                                              device const half *residual [[buffer(2)]],
                                              device float *dst [[buffer(3)]],
                                              constant UocrDenseSwigluParams &params [[buffer(4)]],
                                              threadgroup float *partials [[threadgroup(0)]],
                                              uint output_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]],
                                              uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    float value = uocr_dense_swiglu_down_dot_f16(mid, down_weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        const uint dst_index = token * hidden_size + out_col;
        if (params.has_residual != 0u) {
            value += float(residual[dst_index]);
        }
        dst[dst_index] = value;
    }
}
