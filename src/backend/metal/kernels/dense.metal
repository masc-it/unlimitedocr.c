// dense.metal - Dense projection params, bias add
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrDenseParams {
    uint input_rows;
    uint in_features;
    uint out_features;
    uint has_bias;
};



static inline float uocr_dense_dot_f16(device const half *src,
                                       device const half *weight,
                                       constant UocrDenseParams &params,
                                       uint row,
                                       uint out_col,
                                       uint tid,
                                       uint ntg,
                                       uint simd_width,
                                       threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.in_features;
    const uint weight_base = out_col * params.in_features;
    for (uint k = tid; k < params.in_features; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

kernel void uocr_dense_f16_to_f16(device const half *src [[buffer(0)]],
                                  device const half *weight [[buffer(1)]],
                                  device const half *bias [[buffer(2)]],
                                  device half *dst [[buffer(3)]],
                                  constant UocrDenseParams &params [[buffer(4)]],
                                  threadgroup float *partials [[threadgroup(0)]],
                                  uint output_index [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint ntg [[threads_per_threadgroup]],
                                  uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_f16(src, weight, params, row, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = half(value);
    }
}
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
