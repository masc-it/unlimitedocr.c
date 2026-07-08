// sam.metal - SAM patch embedding
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrSamPatchEmbedParams {
    uint width;
    uint height;
    uint out_width;
    uint out_height;
    uint has_bias;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline uint uocr_sam_patch_weight_index(uint out_channel, uint in_channel, uint ky, uint kx) {
    return (((out_channel * 3u + in_channel) * 16u + ky) * 16u + kx);
}

kernel void uocr_sam_patch_embed_f16_input(device const half *pixels [[buffer(0)]],
                                           device const half *weight [[buffer(1)]],
                                           device const half *bias [[buffer(2)]],
                                           device half *dst_bhwc [[buffer(3)]],
                                           constant UocrSamPatchEmbedParams &params [[buffer(4)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint3 block [[threadgroup_position_in_grid]],
                                           uint3 tid3 [[thread_position_in_threadgroup]],
                                           uint3 ntg3 [[threads_per_threadgroup]],
                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint tid = tid3.x;
    const uint ntg = ntg3.x;
    const uint out_channel = block.x;
    const uint patch_index = block.y;
    const uint batch_index = block.z;
    const uint patch_count = params.out_width * params.out_height;
    if (out_channel >= 768u || patch_index >= patch_count) {
        return;
    }

    const uint patch_y = patch_index / params.out_width;
    const uint patch_x = patch_index - patch_y * params.out_width;
    const uint input_y0 = patch_y * 16u;
    const uint input_x0 = patch_x * 16u;

    float acc = 0.0f;
    for (uint linear = tid; linear < 3u * 16u * 16u; linear += ntg) {
        const uint c = linear / (16u * 16u);
        const uint rem = linear - c * 16u * 16u;
        const uint ky = rem / 16u;
        const uint kx = rem - ky * 16u;
        const uint pixel_index = batch_index * (3u * params.height * params.width) +
                                 c * params.height * params.width +
                                 (input_y0 + ky) * params.width + input_x0 + kx;
        const float x = float(pixels[pixel_index]);
        const float w = float(weight[uocr_sam_patch_weight_index(out_channel, c, ky, kx)]);
        acc += x * w;
    }
    const float total = uocr_threadgroup_sum(acc, partials, tid, ntg, simd_width);
    if (tid == 0u) {
        const float value = total + (params.has_bias != 0u ? float(bias[out_channel]) : 0.0f);
        dst_bhwc[(batch_index * patch_count + patch_index) * 768u + out_channel] = half(value);
    }
}

// SAM layernorm2d and residual
struct UocrSamLayerNorm2dParams {
    uint grid_width;
    uint grid_height;
    uint channels;
    float eps;
    uint batch_size;
};

static inline float uocr_sam_layernorm2d_value(device const half *src_nchw,
                                               device const half *weight,
                                               device const half *bias,
                                               constant UocrSamLayerNorm2dParams &params,
                                               uint spatial,
                                               uint channel,
                                               uint batch,
                                               threadgroup float *partials,
                                               uint tid,
                                               uint ntg,
                                               uint simd_width) {
    const uint spatial_size = params.grid_width * params.grid_height;
    float value = 0.0f;
    if (tid < params.channels) {
        value = float(src_nchw[(ulong(batch) * ulong(params.channels) + ulong(tid)) * ulong(spatial_size) + ulong(spatial)]);
    }
    float value_sum = 0.0f;
    float square_sum = 0.0f;
    uocr_threadgroup_sum2(tid < params.channels ? value : 0.0f,
                          tid < params.channels ? value * value : 0.0f,
                          partials,
                          partials + ntg,
                          tid,
                          ntg,
                          simd_width,
                          value_sum,
                          square_sum);

    const float inv_channels = 1.0f / float(params.channels);
    const float mean = value_sum * inv_channels;
    const float variance = max(square_sum * inv_channels - mean * mean, 0.0f);
    return (value - mean) * rsqrt(variance + params.eps) * float(weight[channel]) + float(bias[channel]);
}

kernel void uocr_sam_layernorm2d_f16_to_f16(device const half *src_nchw [[buffer(0)]],
                                            device const half *weight [[buffer(1)]],
                                            device const half *bias [[buffer(2)]],
                                            device half *dst_nchw [[buffer(3)]],
                                            constant UocrSamLayerNorm2dParams &params [[buffer(4)]],
                                            threadgroup float *partials [[threadgroup(0)]],
                                            uint3 block [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint3 ntg3 [[threads_per_threadgroup]],
                                            uint simd_width [[threads_per_simdgroup]]) {
    const uint ntg = ntg3.x;
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = block.x;
    const uint batch = block.z;
    if (spatial >= spatial_size || tid >= params.channels || batch >= params.batch_size) {
        return;
    }
    const float value = uocr_sam_layernorm2d_value(src_nchw, weight, bias, params, spatial, tid, batch, partials, tid, ntg, simd_width);
    dst_nchw[(ulong(batch) * ulong(params.channels) + ulong(tid)) * ulong(spatial_size) + ulong(spatial)] = half(value);
}
/*
 * BHWC variant of layernorm2d.  Normalizes over channels at each spatial
 * position, with channels contiguous in memory for better cache behavior.
 *
 * src/dst:  [batch, spatial_size, channels]  in BHWC layout.
 * Dispatch: (spatial, 1, batch)  with threads = channels.
 */
kernel void uocr_sam_layernorm2d_bhwc_f16_to_f16(device const half *src_bhwc [[buffer(0)]],
                                                  device const half *weight [[buffer(1)]],
                                                  device const half *bias [[buffer(2)]],
                                                  device half *dst_bhwc [[buffer(3)]],
                                                  constant UocrSamLayerNorm2dParams &params [[buffer(4)]],
                                                  threadgroup float *partials [[threadgroup(0)]],
                                                  uint3 block [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint3 ntg3 [[threads_per_threadgroup]],
                                                  uint simd_width [[threads_per_simdgroup]]) {
    const uint ntg = ntg3.x;
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = block.x;
    const uint batch = block.z;
    /* All threads must participate in threadgroup_barrier inside
     * uocr_threadgroup_sum2. Only skip threads past valid range. */
    if (spatial >= spatial_size || batch >= params.batch_size) {
        return;
    }

    const ulong src_base = (ulong(batch) * ulong(spatial_size) + ulong(spatial)) * ulong(params.channels);
    float value = 0.0f;
    if (tid < params.channels) {
        value = float(src_bhwc[src_base + ulong(tid)]);
    }
    float value_sum = 0.0f;
    float square_sum = 0.0f;
    uocr_threadgroup_sum2(tid < params.channels ? value : 0.0f,
                          tid < params.channels ? value * value : 0.0f,
                          partials,
                          partials + ntg,
                          tid,
                          ntg,
                          simd_width,
                          value_sum,
                          square_sum);

    const float inv_channels = 1.0f / float(params.channels);
    const float mean = value_sum * inv_channels;
    const float variance = max(square_sum * inv_channels - mean * mean, 0.0f);
    if (tid < params.channels) {
        const float result = (value - mean) * rsqrt(variance + params.eps) * float(weight[tid]) + float(bias[tid]);
        dst_bhwc[src_base + ulong(tid)] = half(result);
    }
}

// SAM MLP activation helpers
static inline float uocr_erf_approx(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
                            0.254829592f) *
                               t * exp(-(ax * ax));
    return sign * y;
}

static inline float uocr_gelu_erf(float x) {
    return 0.5f * x * (1.0f + uocr_erf_approx(x * 0.70710678118654752440f));
}

kernel void uocr_bias_gelu_f16_inplace(device half *dst [[buffer(0)]],
                                       device const half *bias [[buffer(1)]],
                                       constant UocrBiasAddParams &params [[buffer(2)]],
                                       uint gid [[thread_position_in_grid]]) {
    const uint value_count = params.rows * params.cols;
    if (gid >= value_count) {
        return;
    }
    const uint col = gid - (gid / params.cols) * params.cols;
    const float value = float(dst[gid]) + float(bias[col]);
    dst[gid] = half(uocr_gelu_erf(value));
}
