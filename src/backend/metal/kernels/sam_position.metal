// sam_position.metal - SAM positional encoding (abs pos)
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrSamAbsPosParams {
    uint source_grid;
    uint target_width;
    uint target_height;
    uint channels;
    uint batch_size;
};

static inline float uocr_cubic_convolution_weight(float x) {
    constexpr float a = -0.75f;
    const float ax = fabs(x);
    if (ax < 1.0f) {
        return ((a + 2.0f) * ax - (a + 3.0f)) * ax * ax + 1.0f;
    }
    if (ax < 2.0f) {
        return (((a * ax - 5.0f * a) * ax + 8.0f * a) * ax - 4.0f * a);
    }
    return 0.0f;
}

static inline float uocr_sam_abs_pos_axis_weight(int source_index, float center, float filter_scale) {
    return uocr_cubic_convolution_weight((float(source_index) - center) / filter_scale);
}

static float uocr_sam_abs_pos_bicubic_antialias(device const half *pos_embed,
                                                constant UocrSamAbsPosParams &params,
                                                uint out_y,
                                                uint out_x,
                                                uint channel) {
    if (params.target_width == params.source_grid && params.target_height == params.source_grid) {
        return float(pos_embed[((out_y * params.source_grid + out_x) * params.channels) + channel]);
    }

    const float scale_y = float(params.source_grid) / float(params.target_height);
    const float scale_x = float(params.source_grid) / float(params.target_width);
    const float filter_scale_y = max(scale_y, 1.0f);
    const float filter_scale_x = max(scale_x, 1.0f);
    const float support_y = 2.0f * filter_scale_y;
    const float support_x = 2.0f * filter_scale_x;
    const float center_y = (float(out_y) + 0.5f) * scale_y - 0.5f;
    const float center_x = (float(out_x) + 0.5f) * scale_x - 0.5f;

    const int src_grid = int(params.source_grid);
    const int y0 = max(0, int(floor(center_y - support_y)));
    const int y1 = min(src_grid - 1, int(ceil(center_y + support_y)));
    const int x0 = max(0, int(floor(center_x - support_x)));
    const int x1 = min(src_grid - 1, int(ceil(center_x + support_x)));

    float acc = 0.0f;
    float weight_sum = 0.0f;
    for (int iy = y0; iy <= y1; ++iy) {
        const float wy = uocr_sam_abs_pos_axis_weight(iy, center_y, filter_scale_y);
        for (int ix = x0; ix <= x1; ++ix) {
            const float wx = uocr_sam_abs_pos_axis_weight(ix, center_x, filter_scale_x);
            const float w = wy * wx;
            acc += float(pos_embed[((uint(iy) * params.source_grid + uint(ix)) * params.channels) + channel]) * w;
            weight_sum += w;
        }
    }
    return weight_sum != 0.0f ? acc / weight_sum : 0.0f;
}

kernel void uocr_sam_add_abs_pos_f16(device const half *patch_bhwc [[buffer(0)]],
                                     device const half *pos_embed [[buffer(1)]],
                                     device half *dst_bhwc [[buffer(2)]],
                                     constant UocrSamAbsPosParams &params [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    const uint values_per_view = params.target_width * params.target_height * params.channels;
    const uint batch_size = params.batch_size == 0u ? 1u : params.batch_size;
    const uint total = values_per_view * batch_size;
    if (gid >= total) {
        return;
    }
    const uint local_gid = gid % values_per_view;
    const uint channel = local_gid % params.channels;
    const uint patch_index = local_gid / params.channels;
    const uint out_y = patch_index / params.target_width;
    const uint out_x = patch_index - out_y * params.target_width;
    const float pos = uocr_sam_abs_pos_bicubic_antialias(pos_embed, params, out_y, out_x, channel);
    dst_bhwc[gid] = half(float(patch_bhwc[gid]) + pos);
}

// SAM rel pos interpolation (UocrSamRelPosInterpolateParams, uocr_sam_interpolate_rel_pos_f16)
struct UocrSamRelPosInterpolateParams {
    uint source_length;
    uint target_length;
    uint head_dim;
    uint reserved;
};

kernel void uocr_sam_interpolate_rel_pos_f16(device const half *src [[buffer(0)]],
                                             device half *dst [[buffer(1)]],
                                             constant UocrSamRelPosInterpolateParams &params [[buffer(2)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint total = params.target_length * params.head_dim;
    if (gid >= total || params.source_length == 0u || params.target_length == 0u || params.head_dim == 0u) {
        return;
    }
    const uint target_index = gid / params.head_dim;
    const uint dim = gid - target_index * params.head_dim;
    dst[gid] = half(uocr_sam_rel_pos_table_value(src,
                                                 params.source_length,
                                                 params.target_length,
                                                 target_index,
                                                 dim,
                                                 params.head_dim));
}
