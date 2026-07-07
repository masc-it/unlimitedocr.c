// dense_q8.metal - Fused Q8_0 dense/shared MLP SwiGLU kernels
#include "common.metal"

struct UocrDenseSwigluQ8Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint has_residual;
    uint group_size;
    uint groups_per_row;
};

static inline void uocr_dense_swiglu_q8_gate_up_dot(
    device const half *src,
    device const char *gate_qweight,
    device const char *up_qweight,
    device const half *gate_qscale,
    device const half *up_qscale,
    constant UocrDenseSwigluQ8Params &params,
    uint token,
    uint out_col,
    uint tid,
    uint ntg,
    uint simd_width,
    threadgroup float *partials,
    thread float &out_gate,
    thread float &out_up) {
    const uint hidden_size = params.hidden_size;
    const uint group_size = params.group_size;
    const uint groups_per_row = params.groups_per_row;
    const uint src_base = token * hidden_size;
    const uint weight_base = out_col * hidden_size;
    const uint scale_base = out_col * groups_per_row;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint k = tid; k < hidden_size; k += ntg) {
        const float x = float(src[src_base + k]);
        const uint group = k / group_size;
        const float g_scale = float(gate_qscale[scale_base + group]);
        const float u_scale = float(up_qscale[scale_base + group]);
        gate_sum += x * (float(int(gate_qweight[weight_base + k])) * g_scale);
        up_sum += x * (float(int(up_qweight[weight_base + k])) * u_scale);
    }
    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, out_gate, out_up);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_gate_up_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *gate_qweight [[buffer(1)]],
    device const char *up_qweight [[buffer(2)]],
    device const half *gate_qscale [[buffer(3)]],
    device const half *up_qscale [[buffer(4)]],
    device half *mid [[buffer(5)]],
    constant UocrDenseSwigluQ8Params &params [[buffer(6)]],
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

    float gate = 0.0f;
    float up = 0.0f;
    uocr_dense_swiglu_q8_gate_up_dot(src, gate_qweight, up_qweight, gate_qscale, up_qscale,
                                     params, token, out_col, tid, ntg, simd_width, partials, gate, up);
    if (tid == 0u) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * params.intermediate_size + out_col] = half(silu * up);
    }
}

static inline void uocr_dense_swiglu_q8_partition256_sum2(
    float value0, float value1,
    threadgroup float *partials,
    uint group, uint groups,
    uint local_tid, uint simd_width,
    thread float &out0, thread float &out1) {
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

[[max_total_threads_per_threadgroup(1024)]] kernel void uocr_dense_swiglu_shared_gate_up_tile4_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *gate_qweight [[buffer(1)]],
    device const char *up_qweight [[buffer(2)]],
    device const half *gate_qscale [[buffer(3)]],
    device const half *up_qscale [[buffer(4)]],
    device half *mid [[buffer(5)]],
    constant UocrDenseSwigluQ8Params &params [[buffer(6)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint threads_per_column = 256u;
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint intermediate_size = uocr_fc_moe_shared_intermediate_or(params.intermediate_size);
    const uint group_size = params.group_size;
    const uint groups_per_row = params.groups_per_row;
    const uint tiles_per_token = uocr_div_up_u32(intermediate_size, tile_columns);
    const uint token = output_tile / tiles_per_token;
    const uint tile = output_tile - token * tiles_per_token;
    if (token >= params.n_tokens || ntg != tile_columns * threads_per_column || hidden_size != 1280u) {
        return;
    }

    const uint src_base = token * hidden_size;
    const uint col_group = tid / threads_per_column;
    const uint local_tid = tid - col_group * threads_per_column;
    const uint out_col = tile * tile_columns + col_group;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (out_col < intermediate_size) {
        const uint weight_base = out_col * hidden_size;
        const uint scale_base = out_col * groups_per_row;
        for (uint k = local_tid; k < hidden_size; k += threads_per_column) {
            const float x = float(src[src_base + k]);
            const uint group = k / group_size;
            const float g_scale = float(gate_qscale[scale_base + group]);
            const float u_scale = float(up_qscale[scale_base + group]);
            gate_sum += x * (float(int(gate_qweight[weight_base + k])) * g_scale);
            up_sum += x * (float(int(up_qweight[weight_base + k])) * u_scale);
        }
    }
    float gate = 0.0f;
    float up = 0.0f;
    uocr_dense_swiglu_q8_partition256_sum2(gate_sum, up_sum, partials,
                                           col_group, tile_columns, local_tid, simd_width, gate, up);

    if (local_tid == 0u && out_col < intermediate_size) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * intermediate_size + out_col] = half(silu * up);
    }
}

static inline float uocr_dense_swiglu_q8_down_dot(
    device const half *mid,
    device const char *down_qweight,
    device const half *down_qscale,
    constant UocrDenseSwigluQ8Params &params,
    uint token,
    uint out_col,
    uint tid,
    uint ntg,
    uint simd_width,
    threadgroup float *partials) {
    const uint intermediate_size = params.intermediate_size;
    const uint group_size = params.group_size;
    const uint groups_per_row = params.groups_per_row;
    const uint mid_base = token * intermediate_size;
    const uint weight_base = out_col * intermediate_size;
    const uint scale_base = out_col * groups_per_row;
    float sum = 0.0f;
    for (uint k = tid; k < intermediate_size; k += ntg) {
        const float scale = float(down_qscale[scale_base + (k / group_size)]);
        const float q = float(int(down_qweight[weight_base + k]));
        sum += float(mid[mid_base + k]) * (q * scale);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_down_q8_0_to_f16(
    device const half *mid [[buffer(0)]],
    device const char *down_qweight [[buffer(1)]],
    device const half *down_qscale [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device half *dst [[buffer(4)]],
    constant UocrDenseSwigluQ8Params &params [[buffer(5)]],
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

    float value = uocr_dense_swiglu_q8_down_dot(mid, down_qweight, down_qscale, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        const uint dst_index = token * hidden_size + out_col;
        if (params.has_residual != 0u) {
            value += float(residual[dst_index]);
        }
        dst[dst_index] = half(value);
    }
}
