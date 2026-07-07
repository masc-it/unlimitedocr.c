// sam_q8.metal - Fused Q8_0 SAM transformer linear kernels
// Requires uocr_gelu_erf from sam.metal in the combined translation unit.
#include "common.metal"

#define UOCR_SAM_Q8_ACT_NONE 0u
#define UOCR_SAM_Q8_ACT_GELU_ERF 1u

struct UocrSamLinearQ8Params {
    uint n_rows;
    uint input_size;
    uint output_size;
    uint group_size;
    uint groups_per_row;
    uint activation;
    uint reserved0;
    uint reserved1;
};

// Fused Q8_0 linear: dst[row, out_col] = act(dot(src[row, :], w[out_col, :]) + bias[out_col]).
// Four output columns per 1024-thread threadgroup; fp32 accumulation.
[[max_total_threads_per_threadgroup(1024)]] kernel void uocr_sam_linear_tile4_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device half *dst [[buffer(4)]],
    constant UocrSamLinearQ8Params &params [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint threads_per_column = 256u;
    const uint tiles_per_row = uocr_div_up_u32(params.output_size, tile_columns);
    const uint row = output_tile / tiles_per_row;
    const uint tile = output_tile - row * tiles_per_row;
    const uint col_group = tid / threads_per_column;
    const uint local_tid = tid - col_group * threads_per_column;
    const uint out_col = tile * tile_columns + col_group;
    if (row >= params.n_rows || col_group >= tile_columns) {
        return;
    }

    const uint input_size = params.input_size;
    const uint group_size = params.group_size;
    const uint groups_per_row = params.groups_per_row;
    const uint src_base = row * input_size;

    float sum = 0.0f;
    if (out_col < params.output_size) {
        const uint weight_base = out_col * input_size;
        const uint scale_base = out_col * groups_per_row;
        for (uint k = local_tid; k < input_size; k += threads_per_column) {
            const float x = float(src[src_base + k]);
            const float scale = float(qscale[scale_base + (k / group_size)]);
            const float w = float(int(qweight[weight_base + k])) * scale;
            sum += x * w;
        }
    }

    threadgroup float *column_partials = partials + col_group * threads_per_column;
    float value = uocr_threadgroup_sum(sum,
                                       column_partials,
                                       local_tid,
                                       threads_per_column,
                                       simd_width);
    if (local_tid == 0u && out_col < params.output_size) {
        value += float(bias[out_col]);
        if (params.activation == UOCR_SAM_Q8_ACT_GELU_ERF) {
            value = uocr_gelu_erf(value);
        }
        dst[row * params.output_size + out_col] = half(value);
    }
}
