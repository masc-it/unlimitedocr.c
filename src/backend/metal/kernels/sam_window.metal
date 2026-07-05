// sam_window.metal - SAM window partition/unpartition
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrSamWindowPartitionParams {
    uint grid_width;
    uint grid_height;
    uint padded_width;
    uint padded_height;
    uint windows_per_row;
    uint windows_per_col;
    uint window_size;
    uint hidden_size;
    uint batch_size;
};

kernel void uocr_sam_window_partition_f16(device const half *src [[buffer(0)]],
                                          device half *dst [[buffer(1)]],
                                          constant UocrSamWindowPartitionParams &params [[buffer(2)]],
                                          uint gid [[thread_position_in_grid]]) {
    const uint padded_tokens = params.padded_width * params.padded_height;
    const uint values_per_view = padded_tokens * params.hidden_size;
    const uint batch_size = params.batch_size == 0u ? 1u : params.batch_size;
    const uint values = values_per_view * batch_size;
    if (gid >= values) {
        return;
    }

    const uint batch = gid / values_per_view;
    const uint local_gid = gid - batch * values_per_view;
    const uint channel = local_gid % params.hidden_size;
    const uint flat_token = local_gid / params.hidden_size;
    const uint token_in_window = flat_token % (params.window_size * params.window_size);
    const uint window = flat_token / (params.window_size * params.window_size);
    const uint window_x = window % params.windows_per_row;
    const uint window_y = window / params.windows_per_row;
    const uint local_y = token_in_window / params.window_size;
    const uint local_x = token_in_window - local_y * params.window_size;
    const uint src_y = window_y * params.window_size + local_y;
    const uint src_x = window_x * params.window_size + local_x;

    if (src_y < params.grid_height && src_x < params.grid_width) {
        const uint src_spatial = src_y * params.grid_width + src_x;
        dst[gid] = src[(batch * params.grid_width * params.grid_height + src_spatial) * params.hidden_size + channel];
    } else {
        dst[gid] = half(0.0f);
    }
}

kernel void uocr_sam_window_unpartition_f16(device const half *src [[buffer(0)]],
                                            device half *dst [[buffer(1)]],
                                            constant UocrSamWindowPartitionParams &params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    const uint values_per_view = params.grid_width * params.grid_height * params.hidden_size;
    const uint batch_size = params.batch_size == 0u ? 1u : params.batch_size;
    const uint values = values_per_view * batch_size;
    if (gid >= values) {
        return;
    }

    const uint batch = gid / values_per_view;
    const uint local_gid = gid - batch * values_per_view;
    const uint channel = local_gid % params.hidden_size;
    const uint flat_token = local_gid / params.hidden_size;
    const uint y = flat_token / params.grid_width;
    const uint x = flat_token - y * params.grid_width;
    const uint window_y = y / params.window_size;
    const uint window_x = x / params.window_size;
    const uint local_y = y - window_y * params.window_size;
    const uint local_x = x - window_x * params.window_size;
    const uint window = window_y * params.windows_per_row + window_x;
    const uint token_in_window = local_y * params.window_size + local_x;

    const uint window_tokens = params.window_size * params.window_size;
    dst[gid] = src[((batch * params.windows_per_row * params.windows_per_col + window) * window_tokens + token_in_window) *
                   params.hidden_size +
                   channel];
}
