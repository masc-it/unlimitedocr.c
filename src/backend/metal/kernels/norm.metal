// norm.metal - RMS norm and layer norm kernels
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrRmsNormParams {
    uint n_rows;
    uint hidden_size;
    float eps;
    uint reserved;
};

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rmsnorm_f16_to_f16(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device half *dst [[buffer(2)]],
                                    constant UocrRmsNormParams &params [[buffer(3)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const float eps = uocr_fc_rms_eps_or(params.eps);
    float sum = 0.0f;
    const uint row_base = row * hidden_size;
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        sum += x * x;
    }
    const float total = uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
    const float scale = 1.0f / sqrt(total / float(hidden_size) + eps);
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        const float w = float(weight[col]);
        dst[row_base + col] = half(x * scale * w);
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rmsnorm_f16_to_f32(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device float *dst [[buffer(2)]],
                                    constant UocrRmsNormParams &params [[buffer(3)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const float eps = uocr_fc_rms_eps_or(params.eps);
    float sum = 0.0f;
    const uint row_base = row * hidden_size;
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        sum += x * x;
    }
    const float total = uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
    const float scale = 1.0f / sqrt(total / float(hidden_size) + eps);
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        const float w = float(weight[col]);
        dst[row_base + col] = x * scale * w;
    }
}

static inline float uocr_layernorm_reduce_sum(device const half *src,
                                              constant UocrRmsNormParams &params,
                                              uint row_base,
                                              uint tid,
                                              uint ntg,
                                              uint simd_width,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        sum += float(src[row_base + col]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

static inline float uocr_layernorm_reduce_var(device const half *src,
                                              constant UocrRmsNormParams &params,
                                              uint row_base,
                                              float mean,
                                              uint tid,
                                              uint ntg,
                                              uint simd_width,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float centered = float(src[row_base + col]) - mean;
        sum += centered * centered;
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width) / float(params.hidden_size);
}

kernel void uocr_layernorm_f16_to_f16(device const half *src [[buffer(0)]],
                                      device const half *weight [[buffer(1)]],
                                      device const half *bias [[buffer(2)]],
                                      device half *dst [[buffer(3)]],
                                      constant UocrRmsNormParams &params [[buffer(4)]],
                                      threadgroup float *partials [[threadgroup(0)]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint ntg [[threads_per_threadgroup]],
                                      uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint row_base = row * params.hidden_size;
    const float mean = uocr_layernorm_reduce_sum(src, params, row_base, tid, ntg, simd_width, partials) / float(params.hidden_size);
    const float variance = uocr_layernorm_reduce_var(src, params, row_base, mean, tid, ntg, simd_width, partials);
    const float scale = 1.0f / sqrt(variance + params.eps);
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float normalized = (float(src[row_base + col]) - mean) * scale;
        dst[row_base + col] = half(normalized * float(weight[col]) + float(bias[col]));
    }
}

kernel void uocr_layernorm_f16_to_f32(device const half *src [[buffer(0)]],
                                      device const half *weight [[buffer(1)]],
                                      device const half *bias [[buffer(2)]],
                                      device float *dst [[buffer(3)]],
                                      constant UocrRmsNormParams &params [[buffer(4)]],
                                      threadgroup float *partials [[threadgroup(0)]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint ntg [[threads_per_threadgroup]],
                                      uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint row_base = row * params.hidden_size;
    const float mean = uocr_layernorm_reduce_sum(src, params, row_base, tid, ntg, simd_width, partials) / float(params.hidden_size);
    const float variance = uocr_layernorm_reduce_var(src, params, row_base, mean, tid, ntg, simd_width, partials);
    const float scale = 1.0f / sqrt(variance + params.eps);
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float normalized = (float(src[row_base + col]) - mean) * scale;
        dst[row_base + col] = normalized * float(weight[col]) + float(bias[col]);
    }
}
