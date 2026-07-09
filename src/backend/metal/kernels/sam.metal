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

// SAM layernorm2d and residual
struct UocrSamLayerNorm2dParams {
    uint grid_width;
    uint grid_height;
    uint channels;
    float eps;
    uint batch_size;
};

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
