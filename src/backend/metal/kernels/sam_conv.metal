// sam_conv.metal - SAM convolution kernels (neck conv, conv3x3 stride2)
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrSamNeckConv1x1Params {
    uint grid_width;
    uint grid_height;
    uint in_channels;
    uint out_channels;
    uint batch_size;
};

static inline float uocr_sam_neck_conv1x1_tile_value_f16(device const half *src_bhwc,
                                                         device const half *weight,
                                                         constant UocrSamNeckConv1x1Params &params,
                                                         uint spatial,
                                                         uint out_channel,
                                                         uint batch) {
    float sum = 0.0f;
    const uint spatial_size = params.grid_width * params.grid_height;
    const ulong src_base = (ulong(batch) * ulong(spatial_size) + ulong(spatial)) * ulong(params.in_channels);
    const uint weight_base = out_channel * params.in_channels;
    for (uint c = 0u; c < params.in_channels; ++c) {
        sum += float(src_bhwc[src_base + ulong(c)]) * float(weight[weight_base + c]);
    }
    return sum;
}

kernel void uocr_sam_neck_conv1x1_f16_to_f16(device const half *src_bhwc [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device half *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv1x1Params &params [[buffer(3)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.out_channels || spatial >= spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_neck_conv1x1_tile_value_f16(src_bhwc, weight, params, spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.out_channels) + ulong(out_channel)) * ulong(spatial_size) + ulong(spatial)] = half(value);
}
struct UocrSamNeckConv3x3Params {
    uint grid_width;
    uint grid_height;
    uint channels;
    uint kernel_size;
    uint batch_size;
};

static inline float uocr_sam_neck_conv3x3_tile_value_f16(device const half *src_nchw,
                                                         device const half *weight,
                                                         constant UocrSamNeckConv3x3Params &params,
                                                         uint spatial,
                                                         uint out_channel,
                                                         uint batch) {
    const uint x = spatial % params.grid_width;
    const uint y = spatial / params.grid_width;
    const uint spatial_size = params.grid_width * params.grid_height;
    const int kernel_radius = int(params.kernel_size >> 1u);
    float sum = 0.0f;
    for (uint in_channel = 0u; in_channel < params.channels; ++in_channel) {
        for (uint ky = 0u; ky < params.kernel_size; ++ky) {
            const int sy = int(y) + int(ky) - kernel_radius;
            if (sy < 0 || sy >= int(params.grid_height)) {
                continue;
            }
            for (uint kx = 0u; kx < params.kernel_size; ++kx) {
                const int sx = int(x) + int(kx) - kernel_radius;
                if (sx < 0 || sx >= int(params.grid_width)) {
                    continue;
                }
                const ulong src_index = (ulong(batch) * ulong(params.channels) + ulong(in_channel)) * ulong(spatial_size) +
                                        ulong(uint(sy)) * ulong(params.grid_width) + ulong(uint(sx));
                const ulong weight_index = ((ulong(out_channel) * ulong(params.channels) + ulong(in_channel)) *
                                                ulong(params.kernel_size) + ulong(ky)) *
                                               ulong(params.kernel_size) + ulong(kx);
                sum += float(src_nchw[src_index]) * float(weight[weight_index]);
            }
        }
    }
    return sum;
}

kernel void uocr_sam_neck_conv3x3_f16_to_f16(device const half *src_nchw [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device half *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv3x3Params &params [[buffer(3)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.channels || spatial >= spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_neck_conv3x3_tile_value_f16(src_nchw, weight, params, spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.channels) + ulong(out_channel)) * ulong(spatial_size) + ulong(spatial)] = half(value);
}
struct UocrSamConv3x3Stride2Params {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint in_channels;
    uint out_channels;
    uint kernel_size;
    uint stride;
    uint batch_size;
};

static inline float uocr_sam_conv3x3_stride2_tile_value_f16(device const half *src_nchw,
                                                            device const half *weight,
                                                            constant UocrSamConv3x3Stride2Params &params,
                                                            uint output_spatial,
                                                            uint out_channel,
                                                            uint batch) {
    const uint out_x = output_spatial % params.output_width;
    const uint out_y = output_spatial / params.output_width;
    const uint input_spatial_size = params.input_width * params.input_height;
    const int kernel_radius = int(params.kernel_size >> 1u);
    float sum = 0.0f;
    for (uint in_channel = 0u; in_channel < params.in_channels; ++in_channel) {
        for (uint ky = 0u; ky < params.kernel_size; ++ky) {
            const int sy = int(out_y * params.stride) + int(ky) - kernel_radius;
            if (sy < 0 || sy >= int(params.input_height)) {
                continue;
            }
            for (uint kx = 0u; kx < params.kernel_size; ++kx) {
                const int sx = int(out_x * params.stride) + int(kx) - kernel_radius;
                if (sx < 0 || sx >= int(params.input_width)) {
                    continue;
                }
                const ulong src_index = (ulong(batch) * ulong(params.in_channels) + ulong(in_channel)) * ulong(input_spatial_size) +
                                        ulong(uint(sy)) * ulong(params.input_width) + ulong(uint(sx));
                const ulong weight_index = ((ulong(out_channel) * ulong(params.in_channels) + ulong(in_channel)) *
                                                ulong(params.kernel_size) + ulong(ky)) *
                                               ulong(params.kernel_size) + ulong(kx);
                sum += float(src_nchw[src_index]) * float(weight[weight_index]);
            }
        }
    }
    return sum;
}

kernel void uocr_sam_conv3x3_stride2_f16_to_f16(device const half *src_nchw [[buffer(0)]],
                                                device const half *weight [[buffer(1)]],
                                                device half *dst_nchw [[buffer(2)]],
                                                constant UocrSamConv3x3Stride2Params &params [[buffer(3)]],
                                                uint3 gid [[thread_position_in_grid]]) {
    const uint output_spatial_size = params.output_width * params.output_height;
    const uint output_spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.out_channels || output_spatial >= output_spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_conv3x3_stride2_tile_value_f16(src_nchw, weight, params, output_spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.out_channels) + ulong(out_channel)) * ulong(output_spatial_size) + ulong(output_spatial)] = half(value);
}
