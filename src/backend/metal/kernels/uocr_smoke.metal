#include <metal_stdlib>
using namespace metal;

kernel void uocr_smoke_u32(device const uint *src [[buffer(0)]],
                           device uint *dst [[buffer(1)]],
                           constant uint &n [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    dst[gid] = src[gid] ^ 0xa5a5a5a5u;
}

struct UocrTouchParams {
    uint n_words;
    uint stride_words;
    uint dst_base;
    uint reserved;
};

kernel void uocr_touch_u32(device const uint *src [[buffer(0)]],
                           device uint *dst [[buffer(1)]],
                           constant UocrTouchParams &params [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint idx = gid * params.stride_words;
    if (idx >= params.n_words) {
        return;
    }
    dst[params.dst_base + gid] = src[idx] ^ idx;
}

struct UocrGetRowsParams {
    uint table_rows;
    uint row_width;
    uint n_row_ids;
    uint reserved;
};

kernel void uocr_get_rows_f16_to_f16(device const half *table [[buffer(0)]],
                                     device const int *row_ids [[buffer(1)]],
                                     device half *dst [[buffer(2)]],
                                     constant UocrGetRowsParams &params [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint out_row = gid.y;
    if (col >= params.row_width || out_row >= params.n_row_ids) {
        return;
    }
    const int row = row_ids[out_row];
    if (row < 0 || (uint)row >= params.table_rows) {
        dst[out_row * params.row_width + col] = half(0.0);
        return;
    }
    dst[out_row * params.row_width + col] = table[(uint)row * params.row_width + col];
}

kernel void uocr_get_rows_f16_to_f32(device const half *table [[buffer(0)]],
                                     device const int *row_ids [[buffer(1)]],
                                     device float *dst [[buffer(2)]],
                                     constant UocrGetRowsParams &params [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint out_row = gid.y;
    if (col >= params.row_width || out_row >= params.n_row_ids) {
        return;
    }
    const int row = row_ids[out_row];
    if (row < 0 || (uint)row >= params.table_rows) {
        dst[out_row * params.row_width + col] = 0.0f;
        return;
    }
    dst[out_row * params.row_width + col] = float(table[(uint)row * params.row_width + col]);
}

struct UocrPromptAssemblyParams {
    uint table_rows;
    uint hidden_size;
    uint n_tokens;
    uint image_span_start;
    uint image_span_length;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_assemble_prompt_text_f16(device const half *embedding_table [[buffer(0)]],
                                          device const int *input_ids [[buffer(1)]],
                                          device half *dst [[buffer(2)]],
                                          constant UocrPromptAssemblyParams &params [[buffer(3)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= params.hidden_size || token >= params.n_tokens) {
        return;
    }
    const int row = input_ids[token];
    if (row < 0 || (uint)row >= params.table_rows) {
        dst[token * params.hidden_size + col] = half(0.0);
        return;
    }
    dst[token * params.hidden_size + col] = embedding_table[(uint)row * params.hidden_size + col];
}

kernel void uocr_assemble_prompt_with_image_f16(device const half *embedding_table [[buffer(0)]],
                                                device const int *input_ids [[buffer(1)]],
                                                device const half *image_features [[buffer(2)]],
                                                device half *dst [[buffer(3)]],
                                                constant UocrPromptAssemblyParams &params [[buffer(4)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= params.hidden_size || token >= params.n_tokens) {
        return;
    }
    if (token >= params.image_span_start && token < params.image_span_start + params.image_span_length) {
        const uint image_row = token - params.image_span_start;
        dst[token * params.hidden_size + col] = image_features[image_row * params.hidden_size + col];
        return;
    }
    const int row = input_ids[token];
    if (row < 0 || (uint)row >= params.table_rows) {
        dst[token * params.hidden_size + col] = half(0.0);
        return;
    }
    dst[token * params.hidden_size + col] = embedding_table[(uint)row * params.hidden_size + col];
}

struct UocrRmsNormParams {
    uint n_rows;
    uint hidden_size;
    float eps;
    uint reserved;
};

kernel void uocr_rmsnorm_f16_to_f16(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device half *dst [[buffer(2)]],
                                    constant UocrRmsNormParams &params [[buffer(3)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    float sum = 0.0f;
    const uint row_base = row * params.hidden_size;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        sum += x * x;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = 1.0f / sqrt(partials[0] / float(params.hidden_size) + params.eps);
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        const float w = float(weight[col]);
        dst[row_base + col] = half(x * scale * w);
    }
}

kernel void uocr_rmsnorm_f16_to_f32(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device float *dst [[buffer(2)]],
                                    constant UocrRmsNormParams &params [[buffer(3)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    float sum = 0.0f;
    const uint row_base = row * params.hidden_size;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        sum += x * x;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = 1.0f / sqrt(partials[0] / float(params.hidden_size) + params.eps);
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        const float w = float(weight[col]);
        dst[row_base + col] = x * scale * w;
    }
}

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
                                       threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.in_features;
    const uint weight_base = out_col * params.in_features;
    for (uint k = tid; k < params.in_features; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partials[0];
}

kernel void uocr_dense_f16_to_f16(device const half *src [[buffer(0)]],
                                  device const half *weight [[buffer(1)]],
                                  device const half *bias [[buffer(2)]],
                                  device half *dst [[buffer(3)]],
                                  constant UocrDenseParams &params [[buffer(4)]],
                                  threadgroup float *partials [[threadgroup(0)]],
                                  uint output_index [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_f16(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = half(value);
    }
}

kernel void uocr_dense_f16_to_f32(device const half *src [[buffer(0)]],
                                  device const half *weight [[buffer(1)]],
                                  device const half *bias [[buffer(2)]],
                                  device float *dst [[buffer(3)]],
                                  constant UocrDenseParams &params [[buffer(4)]],
                                  threadgroup float *partials [[threadgroup(0)]],
                                  uint output_index [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_f16(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = value;
    }
}
