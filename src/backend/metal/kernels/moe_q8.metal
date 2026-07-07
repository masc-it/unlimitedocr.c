// moe_q8.metal - Fused Q8_0 routed MoE selected-expert kernels
#include "common.metal"

struct UocrMoePrefillInterleavedQ8Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint expert_stride_values;     // qweight values per expert (gate+up+down)
    uint up_offset_values;         // qweight offset to up block
    uint down_offset_values;       // qweight offset to down block
    uint expert_scale_stride_values;  // qscale values per expert (gate+up+down)
    uint up_scale_offset_values;      // qscale offset to up block
    uint down_scale_offset_values;    // qscale offset to down block
    uint group_size;
};

struct UocrMoeDecodeInterleavedQ8Params {
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
    uint reserved;
};

// Decode routed-expert gate/up (single-token, non-tiled).
// One threadgroup per (token, rank, out_col).  Dequantizes gate/up qweights
// inside the dot loop, applies SwiGLU, writes fp16 mid.
kernel void uocr_moe_decode_interleaved_gate_up_one_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const char *expert_qweight [[buffer(2)]],
    device const half *expert_qscale [[buffer(3)]],
    device half *mid [[buffer(4)]],
    constant UocrMoePrefillInterleavedQ8Params &params [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint per_token = params.top_k * params.intermediate_size;
    const uint token = output_index / per_token;
    const uint token_rem = output_index - token * per_token;
    const uint rank = token_rem / params.intermediate_size;
    const uint out_col = token_rem - rank * params.intermediate_size;
    if (token >= params.n_tokens || rank >= params.top_k || out_col >= params.intermediate_size) {
        return;
    }

    const uint expert = top_expert_ids[token * params.top_k + rank];
    if (expert >= params.expert_count) {
        if (tid == 0u) {
            mid[(ulong(token) * ulong(params.top_k) + ulong(rank)) * ulong(params.intermediate_size) + ulong(out_col)] = half(0.0f);
        }
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    const uint hidden_size = params.hidden_size;
    const uint group_size = params.group_size;
    const uint groups_per_row_hidden = hidden_size / group_size;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const ulong src_base = ulong(token) * ulong(hidden_size);
    const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values);
    const ulong expert_scale_base = ulong(expert) * ulong(params.expert_scale_stride_values);
    const ulong gate_w_base = expert_base + ulong(out_col) * ulong(hidden_size);
    const ulong up_w_base = expert_base + ulong(params.up_offset_values) + ulong(out_col) * ulong(hidden_size);
    const ulong gate_s_base = expert_scale_base + ulong(out_col) * ulong(groups_per_row_hidden);
    const ulong up_s_base = expert_scale_base + ulong(params.up_scale_offset_values) + ulong(out_col) * ulong(groups_per_row_hidden);
    for (uint k = tid; k < hidden_size; k += ntg) {
        const float x = float(src[src_base + ulong(k)]);
        const uint group = k / group_size;
        const float g_scale = float(expert_qscale[gate_s_base + ulong(group)]);
        const float u_scale = float(expert_qscale[up_s_base + ulong(group)]);
        gate_sum += x * (float(int(expert_qweight[gate_w_base + ulong(k)])) * g_scale);
        up_sum += x * (float(int(expert_qweight[up_w_base + ulong(k)])) * u_scale);
    }
    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, gate, up);

    if (tid == 0u) {
        const float silu = gate / (1.0f + exp(-gate));
        const ulong mid_index = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                                    ulong(params.intermediate_size) + ulong(out_col);
        mid[mid_index] = half(silu * up);
    }
}

// Decode routed-expert gate/up tiled (4 columns per threadgroup).
[[max_total_threads_per_threadgroup(256)]]
kernel void uocr_moe_decode_interleaved_gate_up_tile4_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const char *expert_qweight [[buffer(2)]],
    device const half *expert_qscale [[buffer(3)]],
    device half *mid [[buffer(4)]],
    constant UocrMoePrefillInterleavedQ8Params &params [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint tiles_per_rank = uocr_div_up_u32(params.intermediate_size, tile_columns);
    const uint tiles_per_token = params.top_k * tiles_per_rank;
    const uint token = output_tile / tiles_per_token;
    const uint token_rem = output_tile - token * tiles_per_token;
    const uint rank = token_rem / tiles_per_rank;
    const uint tile = token_rem - rank * tiles_per_rank;
    if (token >= params.n_tokens || rank >= params.top_k || ntg != 256u) {
        return;
    }

    const uint out_col0 = tile * tile_columns;
    const uint out_col1 = out_col0 + 1u;
    const uint out_col2 = out_col0 + 2u;
    const uint out_col3 = out_col0 + 3u;
    const bool valid0 = out_col0 < params.intermediate_size;
    const bool valid1 = out_col1 < params.intermediate_size;
    const bool valid2 = out_col2 < params.intermediate_size;
    const bool valid3 = out_col3 < params.intermediate_size;
    const ulong mid_base = (ulong(token) * ulong(params.top_k) + ulong(rank)) * ulong(params.intermediate_size);
    const uint expert = top_expert_ids[token * params.top_k + rank];
    if (expert >= params.expert_count) {
        if (tid == 0u) {
            if (valid0) mid[mid_base + ulong(out_col0)] = half(0.0f);
            if (valid1) mid[mid_base + ulong(out_col1)] = half(0.0f);
            if (valid2) mid[mid_base + ulong(out_col2)] = half(0.0f);
            if (valid3) mid[mid_base + ulong(out_col3)] = half(0.0f);
        }
        return;
    }

    const uint hidden_size = params.hidden_size;
    const uint group_size = params.group_size;
    const uint groups_per_row_hidden = hidden_size / group_size;
    const ulong src_base = ulong(token) * ulong(hidden_size);
    const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values);
    const ulong expert_scale_base = ulong(expert) * ulong(params.expert_scale_stride_values);

    float gate0 = 0.0f, up0 = 0.0f;
    float gate1 = 0.0f, up1 = 0.0f;
    float gate2 = 0.0f, up2 = 0.0f;
    float gate3 = 0.0f, up3 = 0.0f;
    for (uint k = tid; k < hidden_size; k += ntg) {
        const ulong kk = ulong(k);
        const float x = float(src[src_base + kk]);
        const uint group = k / group_size;
        if (valid0) {
            const ulong gw = expert_base + ulong(out_col0) * ulong(hidden_size) + kk;
            const ulong gs = expert_scale_base + ulong(out_col0) * ulong(groups_per_row_hidden) + ulong(group);
            const ulong uw = expert_base + ulong(params.up_offset_values) + ulong(out_col0) * ulong(hidden_size) + kk;
            const ulong us = expert_scale_base + ulong(params.up_scale_offset_values) + ulong(out_col0) * ulong(groups_per_row_hidden) + ulong(group);
            const float g_s = float(expert_qscale[gs]);
            const float u_s = float(expert_qscale[us]);
            gate0 += x * (float(int(expert_qweight[gw])) * g_s);
            up0 += x * (float(int(expert_qweight[uw])) * u_s);
        }
        if (valid1) {
            const ulong gw = expert_base + ulong(out_col1) * ulong(hidden_size) + kk;
            const ulong gs = expert_scale_base + ulong(out_col1) * ulong(groups_per_row_hidden) + ulong(group);
            const ulong uw = expert_base + ulong(params.up_offset_values) + ulong(out_col1) * ulong(hidden_size) + kk;
            const ulong us = expert_scale_base + ulong(params.up_scale_offset_values) + ulong(out_col1) * ulong(groups_per_row_hidden) + ulong(group);
            const float g_s = float(expert_qscale[gs]);
            const float u_s = float(expert_qscale[us]);
            gate1 += x * (float(int(expert_qweight[gw])) * g_s);
            up1 += x * (float(int(expert_qweight[uw])) * u_s);
        }
        if (valid2) {
            const ulong gw = expert_base + ulong(out_col2) * ulong(hidden_size) + kk;
            const ulong gs = expert_scale_base + ulong(out_col2) * ulong(groups_per_row_hidden) + ulong(group);
            const ulong uw = expert_base + ulong(params.up_offset_values) + ulong(out_col2) * ulong(hidden_size) + kk;
            const ulong us = expert_scale_base + ulong(params.up_scale_offset_values) + ulong(out_col2) * ulong(groups_per_row_hidden) + ulong(group);
            const float g_s = float(expert_qscale[gs]);
            const float u_s = float(expert_qscale[us]);
            gate2 += x * (float(int(expert_qweight[gw])) * g_s);
            up2 += x * (float(int(expert_qweight[uw])) * u_s);
        }
        if (valid3) {
            const ulong gw = expert_base + ulong(out_col3) * ulong(hidden_size) + kk;
            const ulong gs = expert_scale_base + ulong(out_col3) * ulong(groups_per_row_hidden) + ulong(group);
            const ulong uw = expert_base + ulong(params.up_offset_values) + ulong(out_col3) * ulong(hidden_size) + kk;
            const ulong us = expert_scale_base + ulong(params.up_scale_offset_values) + ulong(out_col3) * ulong(groups_per_row_hidden) + ulong(group);
            const float g_s = float(expert_qscale[gs]);
            const float u_s = float(expert_qscale[us]);
            gate3 += x * (float(int(expert_qweight[gw])) * g_s);
            up3 += x * (float(int(expert_qweight[uw])) * u_s);
        }
    }

    float gate = 0.0f, up = 0.0f;
    uocr_threadgroup_sum2(gate0, up0, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid0) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col0)] = half(silu * up);
    }
    uocr_threadgroup_sum2(gate1, up1, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid1) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col1)] = half(silu * up);
    }
    uocr_threadgroup_sum2(gate2, up2, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid2) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col2)] = half(silu * up);
    }
    uocr_threadgroup_sum2(gate3, up3, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid3) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col3)] = half(silu * up);
    }
}

// Routed-expert down/combine (works for both decode and prefill).
// One threadgroup per (token, out_col).  Uses the prefill params struct
// which has n_tokens.  For decode (n_tokens==1), token is always 0.
[[max_total_threads_per_threadgroup(256)]]
kernel void uocr_moe_decode_interleaved_down_sum_combine_one_q8_0_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const float *top_weights [[buffer(2)]],
    device const char *expert_qweight [[buffer(3)]],
    device const half *expert_qscale [[buffer(4)]],
    device const half *shared [[buffer(5)]],
    device const half *residual [[buffer(6)]],
    device half *dst [[buffer(7)]],
    constant UocrMoePrefillInterleavedQ8Params &params [[buffer(8)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint intermediate_size = params.intermediate_size;
    const uint top_k = params.top_k;
    const uint hidden_size = params.hidden_size;
    const uint group_size = params.group_size;
    const uint groups_per_row_inter = intermediate_size / group_size;
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }
    float sum = 0.0f;
    for (uint rank = 0u; rank < top_k; ++rank) {
        const uint expert = top_expert_ids[token * top_k + rank];
        if (expert >= params.expert_count) {
            continue;
        }
        float expert_sum = 0.0f;
        const ulong mid_base = (ulong(token) * ulong(top_k) + ulong(rank)) * ulong(intermediate_size);
        const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values);
        const ulong expert_scale_base = ulong(expert) * ulong(params.expert_scale_stride_values);
        const ulong down_w_base = expert_base + ulong(params.down_offset_values) + ulong(out_col) * ulong(intermediate_size);
        const ulong down_s_base = expert_scale_base + ulong(params.down_scale_offset_values) + ulong(out_col) * ulong(groups_per_row_inter);
        for (uint k = tid; k < intermediate_size; k += ntg) {
            const float scale = float(expert_qscale[down_s_base + ulong(k / group_size)]);
            const float q = float(int(expert_qweight[down_w_base + ulong(k)]));
            expert_sum += float(mid[mid_base + ulong(k)]) * (q * scale);
        }
        sum += expert_sum * top_weights[token * top_k + rank];
    }
    const float routed = uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
    if (tid == 0u) {
        const uint dst_index = token * hidden_size + out_col;
        dst[dst_index] = half(routed + float(shared[dst_index]) + float(residual[dst_index]));
    }
}

// Decode routed-expert down/combine tiled (4 columns per threadgroup).
// Uses the prefill params struct for token indexing.
[[max_total_threads_per_threadgroup(256)]]
kernel void uocr_moe_decode_interleaved_down_sum_combine_tile4_q8_0_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const float *top_weights [[buffer(2)]],
    device const char *expert_qweight [[buffer(3)]],
    device const half *expert_qscale [[buffer(4)]],
    device const half *shared [[buffer(5)]],
    device const half *residual [[buffer(6)]],
    device half *dst [[buffer(7)]],
    constant UocrMoePrefillInterleavedQ8Params &params [[buffer(8)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint out_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint hidden_size = params.hidden_size;
    const uint intermediate_size = params.intermediate_size;
    const uint top_k = params.top_k;
    const uint group_size = params.group_size;
    const uint groups_per_row_inter = intermediate_size / group_size;
    const uint tiles_per_token = (hidden_size + tile_columns - 1u) / tile_columns;
    const uint token = out_tile / tiles_per_token;
    const uint tile = out_tile - token * tiles_per_token;
    const uint out_col0 = tile * tile_columns;
    if (token >= params.n_tokens || out_col0 >= hidden_size || ntg != 256u) {
        return;
    }
    const uint out_col1 = out_col0 + 1u;
    const uint out_col2 = out_col0 + 2u;
    const uint out_col3 = out_col0 + 3u;
    const bool valid0 = out_col0 < hidden_size;
    const bool valid1 = out_col1 < hidden_size;
    const bool valid2 = out_col2 < hidden_size;
    const bool valid3 = out_col3 < hidden_size;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    for (uint rank = 0u; rank < top_k; ++rank) {
        const uint expert = top_expert_ids[token * top_k + rank];
        if (expert >= params.expert_count) {
            continue;
        }
        const ulong mid_base = (ulong(token) * ulong(top_k) + ulong(rank)) * ulong(intermediate_size);
        const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values);
        const ulong expert_scale_base = ulong(expert) * ulong(params.expert_scale_stride_values);
        float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
        for (uint k = tid; k < intermediate_size; k += ntg) {
            const ulong kk = ulong(k);
            const float m = float(mid[mid_base + kk]);
            const uint group = k / group_size;
            if (valid0) {
                const ulong dw = expert_base + ulong(params.down_offset_values) + ulong(out_col0) * ulong(intermediate_size) + kk;
                const ulong ds = expert_scale_base + ulong(params.down_scale_offset_values) + ulong(out_col0) * ulong(groups_per_row_inter) + ulong(group);
                e0 += m * (float(int(expert_qweight[dw])) * float(expert_qscale[ds]));
            }
            if (valid1) {
                const ulong dw = expert_base + ulong(params.down_offset_values) + ulong(out_col1) * ulong(intermediate_size) + kk;
                const ulong ds = expert_scale_base + ulong(params.down_scale_offset_values) + ulong(out_col1) * ulong(groups_per_row_inter) + ulong(group);
                e1 += m * (float(int(expert_qweight[dw])) * float(expert_qscale[ds]));
            }
            if (valid2) {
                const ulong dw = expert_base + ulong(params.down_offset_values) + ulong(out_col2) * ulong(intermediate_size) + kk;
                const ulong ds = expert_scale_base + ulong(params.down_scale_offset_values) + ulong(out_col2) * ulong(groups_per_row_inter) + ulong(group);
                e2 += m * (float(int(expert_qweight[dw])) * float(expert_qscale[ds]));
            }
            if (valid3) {
                const ulong dw = expert_base + ulong(params.down_offset_values) + ulong(out_col3) * ulong(intermediate_size) + kk;
                const ulong ds = expert_scale_base + ulong(params.down_scale_offset_values) + ulong(out_col3) * ulong(groups_per_row_inter) + ulong(group);
                e3 += m * (float(int(expert_qweight[dw])) * float(expert_qscale[ds]));
            }
        }
        sum0 += e0 * top_weights[token * top_k + rank];
        sum1 += e1 * top_weights[token * top_k + rank];
        sum2 += e2 * top_weights[token * top_k + rank];
        sum3 += e3 * top_weights[token * top_k + rank];
    }

    float v0 = uocr_threadgroup_sum(sum0, partials, tid, ntg, simd_width);
    if (tid == 0u && valid0) { const uint idx = token * hidden_size + out_col0; dst[idx] = half(v0 + float(shared[idx]) + float(residual[idx])); }
    float v1 = uocr_threadgroup_sum(sum1, partials, tid, ntg, simd_width);
    if (tid == 0u && valid1) { const uint idx = token * hidden_size + out_col1; dst[idx] = half(v1 + float(shared[idx]) + float(residual[idx])); }
    float v2 = uocr_threadgroup_sum(sum2, partials, tid, ntg, simd_width);
    if (tid == 0u && valid2) { const uint idx = token * hidden_size + out_col2; dst[idx] = half(v2 + float(shared[idx]) + float(residual[idx])); }
    float v3 = uocr_threadgroup_sum(sum3, partials, tid, ntg, simd_width);
    if (tid == 0u && valid3) { const uint idx = token * hidden_size + out_col3; dst[idx] = half(v3 + float(shared[idx]) + float(residual[idx])); }
}
