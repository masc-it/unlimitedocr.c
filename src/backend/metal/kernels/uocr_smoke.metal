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

struct UocrGetRowsQ8Params {
    uint table_rows;
    uint logical_width;
    uint physical_width;
    uint n_row_ids;
    uint row_size;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline float uocr_q8_0_load_value(device const uchar *table, uint row_size, uint row, uint col) {
    constexpr uint qk = 32u;
    constexpr uint block_bytes = 34u;
    const uint block = col / qk;
    const uint in_block = col - block * qk;
    device const uchar *packed = table + row * row_size + block * block_bytes;
    const ushort scale_bits = ushort(packed[0]) | (ushort(packed[1]) << 8u);
    const half scale = as_type<half>(scale_bits);
    int q = int(packed[2u + in_block]);
    if (q >= 128) {
        q -= 256;
    }
    return float(scale) * float(q);
}

static inline uchar2 uocr_q4_k_get_scale_min(device const uchar *scales, uint group) {
    if (group < 4u) {
        return uchar2(uchar(scales[group] & 63u), uchar(scales[group + 4u] & 63u));
    }
    const uint k = group - 4u;
    return uchar2(uchar((scales[8u + k] & 0x0fu) | ((scales[k] & 0xc0u) >> 2u)),
                  uchar((scales[8u + k] >> 4u) | ((scales[4u + k] & 0xc0u) >> 2u)));
}

static inline float uocr_q4_k_load_value(device const uchar *table, uint row_size, uint row, uint col) {
    constexpr uint qk = 256u;
    constexpr uint block_bytes = 144u;
    const uint block = col / qk;
    const uint in_block = col - block * qk;
    device const uchar *packed = table + row * row_size + block * block_bytes;
    const ushort d_bits = ushort(packed[0]) | (ushort(packed[1]) << 8u);
    const ushort dmin_bits = ushort(packed[2]) | (ushort(packed[3]) << 8u);
    const float d = float(as_type<half>(d_bits));
    const float dmin = float(as_type<half>(dmin_bits));
    device const uchar *scales = packed + 4u;
    device const uchar *qs = packed + 16u;

    const uint il = in_block / 16u;
    const uint offset = in_block - il * 16u;
    const uint group = il / 2u;
    const uchar2 scale_min = uocr_q4_k_get_scale_min(scales, group);
    const uint q_base = (il / 4u) * 32u + 16u * (il & 1u);
    const uchar q_byte = qs[q_base + offset];
    const uint q = (il & 2u) == 0u ? uint(q_byte & 0x0fu) : uint(q_byte >> 4u);
    return d * float(scale_min.x) * float(q) - dmin * float(scale_min.y);
}

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

kernel void uocr_get_rows_q8_0_to_f16(device const uchar *table [[buffer(0)]],
                                      device const int *row_ids [[buffer(1)]],
                                      device half *dst [[buffer(2)]],
                                      constant UocrGetRowsQ8Params &params [[buffer(3)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint out_row = gid.y;
    if (col >= params.logical_width || out_row >= params.n_row_ids) {
        return;
    }
    const int row = row_ids[out_row];
    if (row < 0 || (uint)row >= params.table_rows) {
        dst[out_row * params.logical_width + col] = half(0.0);
        return;
    }
    dst[out_row * params.logical_width + col] = half(uocr_q8_0_load_value(table, params.row_size, uint(row), col));
}

kernel void uocr_get_rows_q8_0_to_f32(device const uchar *table [[buffer(0)]],
                                      device const int *row_ids [[buffer(1)]],
                                      device float *dst [[buffer(2)]],
                                      constant UocrGetRowsQ8Params &params [[buffer(3)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint out_row = gid.y;
    if (col >= params.logical_width || out_row >= params.n_row_ids) {
        return;
    }
    const int row = row_ids[out_row];
    if (row < 0 || (uint)row >= params.table_rows) {
        dst[out_row * params.logical_width + col] = 0.0f;
        return;
    }
    dst[out_row * params.logical_width + col] = uocr_q8_0_load_value(table, params.row_size, uint(row), col);
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

kernel void uocr_assemble_prompt_text_skip_image_f16(device const half *embedding_table [[buffer(0)]],
                                                     device const int *input_ids [[buffer(1)]],
                                                     device half *dst [[buffer(2)]],
                                                     constant UocrPromptAssemblyParams &params [[buffer(3)]],
                                                     uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= params.hidden_size || token >= params.n_tokens) {
        return;
    }
    const uint image_span_end = params.image_span_start + params.image_span_length;
    if (token >= params.image_span_start && token < image_span_end) {
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

struct UocrVisualFormatGlobalParams {
    uint hidden_size;
    uint grid_size;
    uint visual_tokens_per_view;
    uint view_count;
    uint dst_token_base;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_format_global_visual_rows_f16(device const half *projected_rows [[buffer(0)]],
                                               device const half *image_newline [[buffer(1)]],
                                               device const half *view_separator [[buffer(2)]],
                                               device half *dst_rows [[buffer(3)]],
                                               constant UocrVisualFormatGlobalParams &params [[buffer(4)]],
                                               uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint visual_row_linear = gid.y;
    if (col >= params.hidden_size || visual_row_linear >= params.view_count * params.visual_tokens_per_view) {
        return;
    }
    const uint grid = params.grid_size;
    const uint grid_tokens = grid * grid;
    const uint view_index = visual_row_linear / params.visual_tokens_per_view;
    const uint visual_row = visual_row_linear - view_index * params.visual_tokens_per_view;
    half value;
    if (visual_row == grid * (grid + 1u)) {
        value = view_separator[col];
    } else {
        const uint row_in_view = visual_row / (grid + 1u);
        const uint col_in_view = visual_row - row_in_view * (grid + 1u);
        if (col_in_view == grid) {
            value = image_newline[col];
        } else {
            const uint src_row = view_index * grid_tokens + row_in_view * grid + col_in_view;
            value = projected_rows[src_row * params.hidden_size + col];
        }
    }
    const uint dst_row = params.dst_token_base + visual_row_linear;
    dst_rows[dst_row * params.hidden_size + col] = value;
}

struct UocrVisualFormatLocalParams {
    uint hidden_size;
    uint grid_size;
    uint crop_grid_w;
    uint chunk_first_view;
    uint chunk_view_count;
    uint dst_token_base;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_fill_local_visual_newlines_f16(device const half *image_newline [[buffer(0)]],
                                                device half *dst_rows [[buffer(1)]],
                                                constant UocrVisualFormatLocalParams &params [[buffer(2)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint local_row = gid.y;
    if (col >= params.hidden_size || params.crop_grid_w == 0u || local_row >= params.chunk_view_count * params.grid_size) {
        return;
    }
    const uint stitched_row_stride = params.crop_grid_w * params.grid_size + 1u;
    const uint dst_row = params.dst_token_base + local_row * stitched_row_stride + params.crop_grid_w * params.grid_size;
    dst_rows[dst_row * params.hidden_size + col] = image_newline[col];
}

kernel void uocr_format_local_visual_rows_f16(device const half *projected_rows [[buffer(0)]],
                                              device half *dst_rows [[buffer(1)]],
                                              constant UocrVisualFormatLocalParams &params [[buffer(2)]],
                                              uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint src_row = gid.y;
    const uint grid = params.grid_size;
    const uint tokens_per_view = grid * grid;
    if (col >= params.hidden_size || tokens_per_view == 0u || src_row >= params.chunk_view_count * tokens_per_view ||
        params.crop_grid_w == 0u) {
        return;
    }
    const uint chunk_view_index = src_row / tokens_per_view;
    const uint token_in_view = src_row - chunk_view_index * tokens_per_view;
    const uint local_view = params.chunk_first_view + chunk_view_index;
    const uint crop_y = local_view / params.crop_grid_w;
    const uint crop_x = local_view - crop_y * params.crop_grid_w;
    const uint row_in_view = token_in_view / grid;
    const uint col_in_view = token_in_view - row_in_view * grid;
    const uint stitched_row_stride = params.crop_grid_w * grid + 1u;
    const uint dst_row = params.dst_token_base +
                         (crop_y * grid + row_in_view) * stitched_row_stride +
                         crop_x * grid + col_in_view;
    dst_rows[dst_row * params.hidden_size + col] = projected_rows[src_row * params.hidden_size + col];
}

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
                                           uint2 block [[threadgroup_position_in_grid]],
                                           uint2 tid2 [[thread_position_in_threadgroup]],
                                           uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint tid = tid2.x;
    const uint ntg = ntg2.x;
    const uint out_channel = block.x;
    const uint patch_index = block.y;
    if (out_channel >= 768u || patch_index >= params.out_width * params.out_height) {
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
        const uint pixel_index = c * params.height * params.width + (input_y0 + ky) * params.width + input_x0 + kx;
        const float x = float(pixels[pixel_index]);
        const float w = float(weight[uocr_sam_patch_weight_index(out_channel, c, ky, kx)]);
        acc += x * w;
    }
    partials[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        const float value = partials[0] + (params.has_bias != 0u ? float(bias[out_channel]) : 0.0f);
        dst_bhwc[patch_index * 768u + out_channel] = half(value);
    }
}

kernel void uocr_sam_patch_embed_f32_input(device const float *pixels [[buffer(0)]],
                                           device const half *weight [[buffer(1)]],
                                           device const half *bias [[buffer(2)]],
                                           device half *dst_bhwc [[buffer(3)]],
                                           constant UocrSamPatchEmbedParams &params [[buffer(4)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint2 block [[threadgroup_position_in_grid]],
                                           uint2 tid2 [[thread_position_in_threadgroup]],
                                           uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint tid = tid2.x;
    const uint ntg = ntg2.x;
    const uint out_channel = block.x;
    const uint patch_index = block.y;
    if (out_channel >= 768u || patch_index >= params.out_width * params.out_height) {
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
        const uint pixel_index = c * params.height * params.width + (input_y0 + ky) * params.width + input_x0 + kx;
        const float x = pixels[pixel_index];
        const float w = float(weight[uocr_sam_patch_weight_index(out_channel, c, ky, kx)]);
        acc += x * w;
    }
    partials[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        const float value = partials[0] + (params.has_bias != 0u ? float(bias[out_channel]) : 0.0f);
        dst_bhwc[patch_index * 768u + out_channel] = half(value);
    }
}

struct UocrSamAbsPosParams {
    uint source_grid;
    uint target_width;
    uint target_height;
    uint channels;
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
    const uint total = params.target_width * params.target_height * params.channels;
    if (gid >= total) {
        return;
    }
    const uint channel = gid % params.channels;
    const uint patch_index = gid / params.channels;
    const uint out_y = patch_index / params.target_width;
    const uint out_x = patch_index - out_y * params.target_width;
    const float pos = uocr_sam_abs_pos_bicubic_antialias(pos_embed, params, out_y, out_x, channel);
    dst_bhwc[gid] = half(float(patch_bhwc[gid]) + pos);
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

static inline float uocr_layernorm_reduce_sum(device const half *src,
                                              constant UocrRmsNormParams &params,
                                              uint row_base,
                                              uint tid,
                                              uint ntg,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        sum += float(src[row_base + col]);
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

static inline float uocr_layernorm_reduce_var(device const half *src,
                                              constant UocrRmsNormParams &params,
                                              uint row_base,
                                              float mean,
                                              uint tid,
                                              uint ntg,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float centered = float(src[row_base + col]) - mean;
        sum += centered * centered;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partials[0] / float(params.hidden_size);
}

kernel void uocr_layernorm_f16_to_f16(device const half *src [[buffer(0)]],
                                      device const half *weight [[buffer(1)]],
                                      device const half *bias [[buffer(2)]],
                                      device half *dst [[buffer(3)]],
                                      constant UocrRmsNormParams &params [[buffer(4)]],
                                      threadgroup float *partials [[threadgroup(0)]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint ntg [[threads_per_threadgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint row_base = row * params.hidden_size;
    const float mean = uocr_layernorm_reduce_sum(src, params, row_base, tid, ntg, partials) / float(params.hidden_size);
    const float variance = uocr_layernorm_reduce_var(src, params, row_base, mean, tid, ntg, partials);
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
                                      uint ntg [[threads_per_threadgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint row_base = row * params.hidden_size;
    const float mean = uocr_layernorm_reduce_sum(src, params, row_base, tid, ntg, partials) / float(params.hidden_size);
    const float variance = uocr_layernorm_reduce_var(src, params, row_base, mean, tid, ntg, partials);
    const float scale = 1.0f / sqrt(variance + params.eps);
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float normalized = (float(src[row_base + col]) - mean) * scale;
        dst[row_base + col] = normalized * float(weight[col]) + float(bias[col]);
    }
}

struct UocrDenseParams {
    uint input_rows;
    uint in_features;
    uint out_features;
    uint has_bias;
};

struct UocrDenseQ8Params {
    uint input_rows;
    uint logical_in_features;
    uint physical_in_features;
    uint out_features;
    uint weight_row_size;
    uint has_bias;
    uint reserved0;
    uint reserved1;
};

struct UocrDenseQ4Params {
    uint input_rows;
    uint logical_in_features;
    uint physical_in_features;
    uint out_features;
    uint weight_row_size;
    uint has_bias;
    uint reserved0;
    uint reserved1;
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

struct UocrBiasAddParams {
    uint rows;
    uint cols;
    uint reserved0;
    uint reserved1;
};

kernel void uocr_bias_add_f16_inplace(device half *dst [[buffer(0)]],
                                      device const half *bias [[buffer(1)]],
                                      constant UocrBiasAddParams &params [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    const uint value_count = params.rows * params.cols;
    if (gid >= value_count) {
        return;
    }
    const uint col = gid - (gid / params.cols) * params.cols;
    dst[gid] = half(float(dst[gid]) + float(bias[col]));
}

static inline float uocr_dense_dot_q8_0(device const half *src,
                                        device const uchar *weight,
                                        constant UocrDenseQ8Params &params,
                                        uint row,
                                        uint out_col,
                                        uint tid,
                                        uint ntg,
                                        threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.logical_in_features;
    for (uint k = tid; k < params.logical_in_features; k += ntg) {
        sum += float(src[src_base + k]) * uocr_q8_0_load_value(weight, params.weight_row_size, out_col, k);
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

kernel void uocr_dense_q8_0_to_f16(device const half *src [[buffer(0)]],
                                   device const uchar *weight [[buffer(1)]],
                                   device const half *bias [[buffer(2)]],
                                   device half *dst [[buffer(3)]],
                                   constant UocrDenseQ8Params &params [[buffer(4)]],
                                   threadgroup float *partials [[threadgroup(0)]],
                                   uint output_index [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_q8_0(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = half(value);
    }
}

kernel void uocr_dense_q8_0_to_f32(device const half *src [[buffer(0)]],
                                   device const uchar *weight [[buffer(1)]],
                                   device const half *bias [[buffer(2)]],
                                   device float *dst [[buffer(3)]],
                                   constant UocrDenseQ8Params &params [[buffer(4)]],
                                   threadgroup float *partials [[threadgroup(0)]],
                                   uint output_index [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_q8_0(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = value;
    }
}

static inline float uocr_dense_dot_q4_k(device const half *src,
                                        device const uchar *weight,
                                        constant UocrDenseQ4Params &params,
                                        uint row,
                                        uint out_col,
                                        uint tid,
                                        uint ntg,
                                        threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.logical_in_features;
    for (uint k = tid; k < params.logical_in_features; k += ntg) {
        sum += float(src[src_base + k]) * uocr_q4_k_load_value(weight, params.weight_row_size, out_col, k);
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

kernel void uocr_dense_q4_k_to_f16(device const half *src [[buffer(0)]],
                                   device const uchar *weight [[buffer(1)]],
                                   device const half *bias [[buffer(2)]],
                                   device half *dst [[buffer(3)]],
                                   constant UocrDenseQ4Params &params [[buffer(4)]],
                                   threadgroup float *partials [[threadgroup(0)]],
                                   uint output_index [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_q4_k(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = half(value);
    }
}

kernel void uocr_dense_q4_k_to_f32(device const half *src [[buffer(0)]],
                                   device const uchar *weight [[buffer(1)]],
                                   device const half *bias [[buffer(2)]],
                                   device float *dst [[buffer(3)]],
                                   constant UocrDenseQ4Params &params [[buffer(4)]],
                                   threadgroup float *partials [[threadgroup(0)]],
                                   uint output_index [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_q4_k(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = value;
    }
}

kernel void uocr_sam_qkv_f16_to_f16(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device const half *bias [[buffer(2)]],
                                    device half *q_dst [[buffer(3)]],
                                    device half *k_dst [[buffer(4)]],
                                    device half *v_dst [[buffer(5)]],
                                    constant UocrDenseParams &params [[buffer(6)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint output_index [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device half *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = half(value);
    }
}

kernel void uocr_sam_qkv_f16_to_f32(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device const half *bias [[buffer(2)]],
                                    device float *q_dst [[buffer(3)]],
                                    device float *k_dst [[buffer(4)]],
                                    device float *v_dst [[buffer(5)]],
                                    constant UocrDenseParams &params [[buffer(6)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint output_index [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device float *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = value;
    }
}

kernel void uocr_clip_qkv_f16_to_f16(device const half *src [[buffer(0)]],
                                     device const half *weight [[buffer(1)]],
                                     device const half *bias [[buffer(2)]],
                                     device half *q_dst [[buffer(3)]],
                                     device half *k_dst [[buffer(4)]],
                                     device half *v_dst [[buffer(5)]],
                                     constant UocrDenseParams &params [[buffer(6)]],
                                     threadgroup float *partials [[threadgroup(0)]],
                                     uint output_index [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint ntg [[threads_per_threadgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device half *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = half(value);
    }
}

kernel void uocr_clip_qkv_f16_to_f32(device const half *src [[buffer(0)]],
                                     device const half *weight [[buffer(1)]],
                                     device const half *bias [[buffer(2)]],
                                     device float *q_dst [[buffer(3)]],
                                     device float *k_dst [[buffer(4)]],
                                     device float *v_dst [[buffer(5)]],
                                     constant UocrDenseParams &params [[buffer(6)]],
                                     threadgroup float *partials [[threadgroup(0)]],
                                     uint output_index [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint ntg [[threads_per_threadgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device float *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = value;
    }
}

static inline float uocr_quickgelu(float x) {
    return x / (1.0f + exp(-1.702f * x));
}

kernel void uocr_bias_quickgelu_f16_inplace(device half *dst [[buffer(0)]],
                                            device const half *bias [[buffer(1)]],
                                            constant UocrBiasAddParams &params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    const uint value_count = params.rows * params.cols;
    if (gid >= value_count) {
        return;
    }
    const uint col = gid - (gid / params.cols) * params.cols;
    const float value = float(dst[gid]) + float(bias[col]);
    dst[gid] = half(uocr_quickgelu(value));
}

kernel void uocr_clip_quickgelu_f16_to_f16(device const half *src [[buffer(0)]],
                                           device half *dst [[buffer(1)]],
                                           constant uint &value_count [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    dst[gid] = half(uocr_quickgelu(float(src[gid])));
}

kernel void uocr_clip_quickgelu_f16_to_f32(device const half *src [[buffer(0)]],
                                           device float *dst [[buffer(1)]],
                                           constant uint &value_count [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    dst[gid] = uocr_quickgelu(float(src[gid]));
}

kernel void uocr_clip_residual_add_f16_to_f16(device const half *base [[buffer(0)]],
                                              device const half *update [[buffer(1)]],
                                              device half *dst [[buffer(2)]],
                                              constant uint &value_count [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    dst[gid] = half(float(base[gid]) + float(update[gid]));
}

kernel void uocr_clip_residual_add_f16_to_f32(device const half *base [[buffer(0)]],
                                              device const half *update [[buffer(1)]],
                                              device float *dst [[buffer(2)]],
                                              constant uint &value_count [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    dst[gid] = float(base[gid]) + float(update[gid]);
}

struct UocrClipSamConcatParams {
    uint grid_width;
    uint grid_height;
    uint hidden_size;
    uint projector_in_size;
};

kernel void uocr_clip_sam_concat_f16_to_f16(device const half *clip_tokens [[buffer(0)]],
                                            device const half *sam_nchw [[buffer(1)]],
                                            device half *dst [[buffer(2)]],
                                            constant UocrClipSamConcatParams &params [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint spatial = gid.y;
    const uint spatial_size = params.grid_width * params.grid_height;
    if (col >= params.projector_in_size || spatial >= spatial_size) {
        return;
    }
    const uint dst_index = spatial * params.projector_in_size + col;
    if (col < params.hidden_size) {
        dst[dst_index] = clip_tokens[(spatial + 1u) * params.hidden_size + col];
        return;
    }
    const uint channel = col - params.hidden_size;
    dst[dst_index] = sam_nchw[channel * spatial_size + spatial];
}

kernel void uocr_clip_sam_concat_f16_to_f32(device const half *clip_tokens [[buffer(0)]],
                                            device const half *sam_nchw [[buffer(1)]],
                                            device float *dst [[buffer(2)]],
                                            constant UocrClipSamConcatParams &params [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint spatial = gid.y;
    const uint spatial_size = params.grid_width * params.grid_height;
    if (col >= params.projector_in_size || spatial >= spatial_size) {
        return;
    }
    const uint dst_index = spatial * params.projector_in_size + col;
    if (col < params.hidden_size) {
        dst[dst_index] = float(clip_tokens[(spatial + 1u) * params.hidden_size + col]);
        return;
    }
    const uint channel = col - params.hidden_size;
    dst[dst_index] = float(sam_nchw[channel * spatial_size + spatial]);
}

struct UocrSamWindowAttentionParams {
    uint windows;
    uint tokens_per_window;
    uint heads;
    uint head_dim;
    float scale;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline ulong uocr_sam_window_attention_index(constant UocrSamWindowAttentionParams &params,
                                                    uint window,
                                                    uint token,
                                                    uint head,
                                                    uint dim) {
    return (((ulong(window) * ulong(params.tokens_per_window) + ulong(token)) * ulong(params.heads) + ulong(head)) *
            ulong(params.head_dim)) +
           ulong(dim);
}

struct UocrSamRelPosAttentionParams {
    uint windows;
    uint grid_width;
    uint grid_height;
    uint tokens_per_window;
    uint heads;
    uint head_dim;
    uint rel_pos_h_length;
    uint rel_pos_w_length;
    float scale;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline ulong uocr_sam_rel_pos_attention_index(constant UocrSamRelPosAttentionParams &params,
                                                     uint window,
                                                     uint token,
                                                     uint head,
                                                     uint dim) {
    return (((ulong(window) * ulong(params.tokens_per_window) + ulong(token)) * ulong(params.heads) + ulong(head)) *
            ulong(params.head_dim)) +
           ulong(dim);
}

static inline float uocr_sam_rel_pos_table_value(device const half *rel_pos,
                                                 uint source_length,
                                                 uint target_length,
                                                 uint target_index,
                                                 uint dim,
                                                 uint head_dim) {
    if (source_length == target_length) {
        return float(rel_pos[ulong(target_index) * ulong(head_dim) + ulong(dim)]);
    }

    const float source_x = (float(target_index) + 0.5f) * float(source_length) / float(target_length) - 0.5f;
    int index0 = int(floor(source_x));
    int index1 = index0 + 1;
    const float t = source_x - floor(source_x);
    index0 = clamp(index0, 0, int(source_length) - 1);
    index1 = clamp(index1, 0, int(source_length) - 1);
    const float v0 = float(rel_pos[ulong(uint(index0)) * ulong(head_dim) + ulong(dim)]);
    const float v1 = float(rel_pos[ulong(uint(index1)) * ulong(head_dim) + ulong(dim)]);
    return v0 * (1.0f - t) + v1 * t;
}

struct UocrSamWindowPartitionParams {
    uint grid_width;
    uint grid_height;
    uint padded_width;
    uint padded_height;
    uint windows_per_row;
    uint windows_per_col;
    uint window_size;
    uint hidden_size;
};

kernel void uocr_sam_window_partition_f16(device const half *src [[buffer(0)]],
                                          device half *dst [[buffer(1)]],
                                          constant UocrSamWindowPartitionParams &params [[buffer(2)]],
                                          uint gid [[thread_position_in_grid]]) {
    const uint padded_tokens = params.padded_width * params.padded_height;
    const uint values = padded_tokens * params.hidden_size;
    if (gid >= values) {
        return;
    }

    const uint channel = gid % params.hidden_size;
    const uint flat_token = gid / params.hidden_size;
    const uint token_in_window = flat_token % (params.window_size * params.window_size);
    const uint window = flat_token / (params.window_size * params.window_size);
    const uint window_x = window % params.windows_per_row;
    const uint window_y = window / params.windows_per_row;
    const uint local_y = token_in_window / params.window_size;
    const uint local_x = token_in_window - local_y * params.window_size;
    const uint src_y = window_y * params.window_size + local_y;
    const uint src_x = window_x * params.window_size + local_x;

    if (src_y < params.grid_height && src_x < params.grid_width) {
        dst[gid] = src[(src_y * params.grid_width + src_x) * params.hidden_size + channel];
    } else {
        dst[gid] = half(0.0f);
    }
}

kernel void uocr_sam_window_unpartition_f16(device const half *src [[buffer(0)]],
                                            device half *dst [[buffer(1)]],
                                            constant UocrSamWindowPartitionParams &params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    const uint values = params.grid_width * params.grid_height * params.hidden_size;
    if (gid >= values) {
        return;
    }

    const uint channel = gid % params.hidden_size;
    const uint flat_token = gid / params.hidden_size;
    const uint y = flat_token / params.grid_width;
    const uint x = flat_token - y * params.grid_width;
    const uint window_y = y / params.window_size;
    const uint window_x = x / params.window_size;
    const uint local_y = y - window_y * params.window_size;
    const uint local_x = x - window_x * params.window_size;
    const uint window = window_y * params.windows_per_row + window_x;
    const uint token_in_window = local_y * params.window_size + local_x;

    dst[gid] = src[(window * params.window_size * params.window_size + token_in_window) * params.hidden_size + channel];
}

struct UocrSamNeckConv1x1Params {
    uint grid_width;
    uint grid_height;
    uint in_channels;
    uint out_channels;
};

static inline float uocr_sam_neck_conv1x1_tile_value_f16(device const half *src_bhwc,
                                                         device const half *weight,
                                                         constant UocrSamNeckConv1x1Params &params,
                                                         uint spatial,
                                                         uint out_channel) {
    float sum = 0.0f;
    const uint src_base = spatial * params.in_channels;
    const uint weight_base = out_channel * params.in_channels;
    for (uint c = 0u; c < params.in_channels; ++c) {
        sum += float(src_bhwc[src_base + c]) * float(weight[weight_base + c]);
    }
    return sum;
}

kernel void uocr_sam_neck_conv1x1_f16_to_f16(device const half *src_bhwc [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device half *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv1x1Params &params [[buffer(3)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    if (out_channel >= params.out_channels || spatial >= spatial_size) {
        return;
    }

    const float value = uocr_sam_neck_conv1x1_tile_value_f16(src_bhwc, weight, params, spatial, out_channel);
    dst_nchw[out_channel * spatial_size + spatial] = half(value);
}

kernel void uocr_sam_neck_conv1x1_f16_to_f32(device const half *src_bhwc [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device float *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv1x1Params &params [[buffer(3)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    if (out_channel >= params.out_channels || spatial >= spatial_size) {
        return;
    }

    const float value = uocr_sam_neck_conv1x1_tile_value_f16(src_bhwc, weight, params, spatial, out_channel);
    dst_nchw[out_channel * spatial_size + spatial] = value;
}

struct UocrSamNeckConv3x3Params {
    uint grid_width;
    uint grid_height;
    uint channels;
    uint kernel_size;
};

static inline float uocr_sam_neck_conv3x3_tile_value_f16(device const half *src_nchw,
                                                         device const half *weight,
                                                         constant UocrSamNeckConv3x3Params &params,
                                                         uint spatial,
                                                         uint out_channel) {
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
                const ulong src_index = ulong(in_channel) * ulong(spatial_size) +
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
                                             uint2 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    if (out_channel >= params.channels || spatial >= spatial_size) {
        return;
    }

    const float value = uocr_sam_neck_conv3x3_tile_value_f16(src_nchw, weight, params, spatial, out_channel);
    dst_nchw[out_channel * spatial_size + spatial] = half(value);
}

kernel void uocr_sam_neck_conv3x3_f16_to_f32(device const half *src_nchw [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device float *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv3x3Params &params [[buffer(3)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    if (out_channel >= params.channels || spatial >= spatial_size) {
        return;
    }

    const float value = uocr_sam_neck_conv3x3_tile_value_f16(src_nchw, weight, params, spatial, out_channel);
    dst_nchw[out_channel * spatial_size + spatial] = value;
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
};

static inline float uocr_sam_conv3x3_stride2_tile_value_f16(device const half *src_nchw,
                                                            device const half *weight,
                                                            constant UocrSamConv3x3Stride2Params &params,
                                                            uint output_spatial,
                                                            uint out_channel) {
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
                const ulong src_index = ulong(in_channel) * ulong(input_spatial_size) +
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
                                                uint2 gid [[thread_position_in_grid]]) {
    const uint output_spatial_size = params.output_width * params.output_height;
    const uint output_spatial = gid.x;
    const uint out_channel = gid.y;
    if (out_channel >= params.out_channels || output_spatial >= output_spatial_size) {
        return;
    }

    const float value = uocr_sam_conv3x3_stride2_tile_value_f16(src_nchw, weight, params, output_spatial, out_channel);
    dst_nchw[out_channel * output_spatial_size + output_spatial] = half(value);
}

kernel void uocr_sam_conv3x3_stride2_f16_to_f32(device const half *src_nchw [[buffer(0)]],
                                                device const half *weight [[buffer(1)]],
                                                device float *dst_nchw [[buffer(2)]],
                                                constant UocrSamConv3x3Stride2Params &params [[buffer(3)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    const uint output_spatial_size = params.output_width * params.output_height;
    const uint output_spatial = gid.x;
    const uint out_channel = gid.y;
    if (out_channel >= params.out_channels || output_spatial >= output_spatial_size) {
        return;
    }

    const float value = uocr_sam_conv3x3_stride2_tile_value_f16(src_nchw, weight, params, output_spatial, out_channel);
    dst_nchw[out_channel * output_spatial_size + output_spatial] = value;
}

struct UocrClipEmbedSamParams {
    uint grid_width;
    uint grid_height;
    uint hidden_size;
    uint token_count;
};

static inline float uocr_clip_embedding_from_sam_value(device const half *sam_nchw,
                                                       device const half *class_embedding,
                                                       constant UocrClipEmbedSamParams &params,
                                                       uint token,
                                                       uint channel) {
    if (token == 0u) {
        return float(class_embedding[channel]);
    }
    const uint spatial = token - 1u;
    const uint y = spatial / params.grid_width;
    const uint x = spatial - y * params.grid_width;
    const ulong spatial_size = ulong(params.grid_width) * ulong(params.grid_height);
    const ulong src_index = ulong(channel) * spatial_size + ulong(y) * ulong(params.grid_width) + ulong(x);
    return float(sam_nchw[src_index]);
}

kernel void uocr_clip_embed_sam_f16_to_f16(device const half *sam_nchw [[buffer(0)]],
                                           device const half *class_embedding [[buffer(1)]],
                                           device half *dst_tokens [[buffer(2)]],
                                           constant UocrClipEmbedSamParams &params [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    const uint channel = gid.x;
    const uint token = gid.y;
    if (channel >= params.hidden_size || token >= params.token_count) {
        return;
    }
    const float value = uocr_clip_embedding_from_sam_value(sam_nchw, class_embedding, params, token, channel);
    dst_tokens[ulong(token) * ulong(params.hidden_size) + ulong(channel)] = half(value);
}

kernel void uocr_clip_embed_sam_f16_to_f32(device const half *sam_nchw [[buffer(0)]],
                                           device const half *class_embedding [[buffer(1)]],
                                           device float *dst_tokens [[buffer(2)]],
                                           constant UocrClipEmbedSamParams &params [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    const uint channel = gid.x;
    const uint token = gid.y;
    if (channel >= params.hidden_size || token >= params.token_count) {
        return;
    }
    dst_tokens[ulong(token) * ulong(params.hidden_size) + ulong(channel)] =
        uocr_clip_embedding_from_sam_value(sam_nchw, class_embedding, params, token, channel);
}

struct UocrClipAbsPosParams {
    uint source_grid;
    uint target_width;
    uint target_height;
    uint hidden_size;
};

static float uocr_clip_abs_pos_bicubic_antialias(device const half *pos_embed,
                                                 constant UocrClipAbsPosParams &params,
                                                 uint token,
                                                 uint channel) {
    if (token == 0u) {
        return float(pos_embed[channel]);
    }
    const uint spatial = token - 1u;
    const uint out_y = spatial / params.target_width;
    const uint out_x = spatial - out_y * params.target_width;
    if (params.target_width == params.source_grid && params.target_height == params.source_grid) {
        return float(pos_embed[(token * params.hidden_size) + channel]);
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
            const ulong source_token = 1ul + ulong(uint(iy)) * ulong(params.source_grid) + ulong(uint(ix));
            acc += float(pos_embed[source_token * ulong(params.hidden_size) + ulong(channel)]) * w;
            weight_sum += w;
        }
    }
    return weight_sum != 0.0f ? acc / weight_sum : 0.0f;
}

kernel void uocr_clip_add_abs_pos_f16_to_f16(device const half *tokens [[buffer(0)]],
                                             device const half *pos_embed [[buffer(1)]],
                                             device half *dst_tokens [[buffer(2)]],
                                             constant UocrClipAbsPosParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint token_count = 1u + params.target_width * params.target_height;
    const uint total = token_count * params.hidden_size;
    if (gid >= total) {
        return;
    }
    const uint channel = gid % params.hidden_size;
    const uint token = gid / params.hidden_size;
    const float pos = uocr_clip_abs_pos_bicubic_antialias(pos_embed, params, token, channel);
    dst_tokens[gid] = half(float(tokens[gid]) + pos);
}

kernel void uocr_clip_add_abs_pos_f16_to_f32(device const half *tokens [[buffer(0)]],
                                             device const half *pos_embed [[buffer(1)]],
                                             device float *dst_tokens [[buffer(2)]],
                                             constant UocrClipAbsPosParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint token_count = 1u + params.target_width * params.target_height;
    const uint total = token_count * params.hidden_size;
    if (gid >= total) {
        return;
    }
    const uint channel = gid % params.hidden_size;
    const uint token = gid / params.hidden_size;
    const float pos = uocr_clip_abs_pos_bicubic_antialias(pos_embed, params, token, channel);
    dst_tokens[gid] = float(tokens[gid]) + pos;
}

struct UocrSamLayerNorm2dParams {
    uint grid_width;
    uint grid_height;
    uint channels;
    float eps;
};

static inline float uocr_sam_layernorm2d_value(device const half *src_nchw,
                                               device const half *weight,
                                               device const half *bias,
                                               constant UocrSamLayerNorm2dParams &params,
                                               uint spatial,
                                               uint channel,
                                               threadgroup float *partials,
                                               uint tid,
                                               uint ntg) {
    const uint spatial_size = params.grid_width * params.grid_height;
    float value = 0.0f;
    if (tid < params.channels) {
        value = float(src_nchw[tid * spatial_size + spatial]);
    }
    partials[tid] = tid < params.channels ? value : 0.0f;
    partials[ntg + tid] = tid < params.channels ? value * value : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
            partials[ntg + tid] += partials[ntg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_channels = 1.0f / float(params.channels);
    const float mean = partials[0] * inv_channels;
    const float variance = max(partials[ntg] * inv_channels - mean * mean, 0.0f);
    return (value - mean) * rsqrt(variance + params.eps) * float(weight[channel]) + float(bias[channel]);
}

kernel void uocr_sam_layernorm2d_f16_to_f16(device const half *src_nchw [[buffer(0)]],
                                            device const half *weight [[buffer(1)]],
                                            device const half *bias [[buffer(2)]],
                                            device half *dst_nchw [[buffer(3)]],
                                            constant UocrSamLayerNorm2dParams &params [[buffer(4)]],
                                            threadgroup float *partials [[threadgroup(0)]],
                                            uint spatial [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint ntg [[threads_per_threadgroup]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    if (spatial >= spatial_size || tid >= params.channels) {
        return;
    }
    const float value = uocr_sam_layernorm2d_value(src_nchw, weight, bias, params, spatial, tid, partials, tid, ntg);
    dst_nchw[tid * spatial_size + spatial] = half(value);
}

kernel void uocr_sam_layernorm2d_f16_to_f32(device const half *src_nchw [[buffer(0)]],
                                            device const half *weight [[buffer(1)]],
                                            device const half *bias [[buffer(2)]],
                                            device float *dst_nchw [[buffer(3)]],
                                            constant UocrSamLayerNorm2dParams &params [[buffer(4)]],
                                            threadgroup float *partials [[threadgroup(0)]],
                                            uint spatial [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint ntg [[threads_per_threadgroup]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    if (spatial >= spatial_size || tid >= params.channels) {
        return;
    }
    const float value = uocr_sam_layernorm2d_value(src_nchw, weight, bias, params, spatial, tid, partials, tid, ntg);
    dst_nchw[tid * spatial_size + spatial] = value;
}

struct UocrSamResidualParams {
    uint n_rows;
    uint hidden_size;
    uint reserved0;
    uint reserved1;
};

kernel void uocr_sam_residual_add_f16_to_f16(device const half *base [[buffer(0)]],
                                             device const half *update [[buffer(1)]],
                                             device half *dst [[buffer(2)]],
                                             constant UocrSamResidualParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint values = params.n_rows * params.hidden_size;
    if (gid >= values) {
        return;
    }
    dst[gid] = half(float(base[gid]) + float(update[gid]));
}

kernel void uocr_sam_residual_add_f16_to_f32(device const half *base [[buffer(0)]],
                                             device const half *update [[buffer(1)]],
                                             device float *dst [[buffer(2)]],
                                             constant UocrSamResidualParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint values = params.n_rows * params.hidden_size;
    if (gid >= values) {
        return;
    }
    dst[gid] = float(base[gid]) + float(update[gid]);
}

static inline float uocr_sam_attention_project_dot_f16(device const half *src,
                                                       device const half *weight,
                                                       constant UocrSamResidualParams &params,
                                                       uint row,
                                                       uint out_col,
                                                       uint tid,
                                                       uint ntg,
                                                       threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partials[0];
}

kernel void uocr_sam_attention_project_residual_f16_to_f16(device const half *src [[buffer(0)]],
                                                           device const half *weight [[buffer(1)]],
                                                           device const half *bias [[buffer(2)]],
                                                           device const half *residual [[buffer(3)]],
                                                           device half *dst [[buffer(4)]],
                                                           constant UocrSamResidualParams &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_attention_project_dot_f16(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0u) {
        const uint dst_index = row * params.hidden_size + out_col;
        dst[dst_index] = half(value + float(bias[out_col]) + float(residual[dst_index]));
    }
}

kernel void uocr_sam_attention_project_residual_f16_to_f32(device const half *src [[buffer(0)]],
                                                           device const half *weight [[buffer(1)]],
                                                           device const half *bias [[buffer(2)]],
                                                           device const half *residual [[buffer(3)]],
                                                           device float *dst [[buffer(4)]],
                                                           constant UocrSamResidualParams &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_attention_project_dot_f16(src, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0u) {
        const uint dst_index = row * params.hidden_size + out_col;
        dst[dst_index] = value + float(bias[out_col]) + float(residual[dst_index]);
    }
}

struct UocrSamMlpParams {
    uint n_rows;
    uint hidden_size;
    uint intermediate_size;
    uint reserved;
};

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

kernel void uocr_sam_mlp_lin1_gelu_f16(device const half *src [[buffer(0)]],
                                       device const half *weight [[buffer(1)]],
                                       device const half *bias [[buffer(2)]],
                                       device half *mid [[buffer(3)]],
                                       constant UocrSamMlpParams &params [[buffer(4)]],
                                       threadgroup float *partials [[threadgroup(0)]],
                                       uint output_index [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.intermediate_size;
    const uint out_col = output_index - row * params.intermediate_size;
    if (row >= params.n_rows || out_col >= params.intermediate_size) {
        return;
    }

    float sum = 0.0f;
    const uint src_base = row * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float projected = partials[0] + float(bias[out_col]);
        mid[row * params.intermediate_size + out_col] = half(uocr_gelu_erf(projected));
    }
}

static inline float uocr_sam_mlp_lin2_dot_f16(device const half *mid,
                                              device const half *weight,
                                              constant UocrSamMlpParams &params,
                                              uint row,
                                              uint out_col,
                                              uint tid,
                                              uint ntg,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    const uint mid_base = row * params.intermediate_size;
    const uint weight_base = out_col * params.intermediate_size;
    for (uint k = tid; k < params.intermediate_size; k += ntg) {
        sum += float(mid[mid_base + k]) * float(weight[weight_base + k]);
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partials[0];
}

kernel void uocr_sam_mlp_lin2_f16_to_f16(device const half *mid [[buffer(0)]],
                                         device const half *weight [[buffer(1)]],
                                         device const half *bias [[buffer(2)]],
                                         device half *dst [[buffer(3)]],
                                         constant UocrSamMlpParams &params [[buffer(4)]],
                                         threadgroup float *partials [[threadgroup(0)]],
                                         uint output_index [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_mlp_lin2_dot_f16(mid, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0u) {
        dst[row * params.hidden_size + out_col] = half(value + float(bias[out_col]));
    }
}

kernel void uocr_sam_mlp_lin2_f16_to_f32(device const half *mid [[buffer(0)]],
                                         device const half *weight [[buffer(1)]],
                                         device const half *bias [[buffer(2)]],
                                         device float *dst [[buffer(3)]],
                                         constant UocrSamMlpParams &params [[buffer(4)]],
                                         threadgroup float *partials [[threadgroup(0)]],
                                         uint output_index [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint ntg [[threads_per_threadgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_mlp_lin2_dot_f16(mid, weight, params, row, out_col, tid, ntg, partials);
    if (tid == 0u) {
        dst[row * params.hidden_size + out_col] = value + float(bias[out_col]);
    }
}

struct UocrArgmaxParams {
    uint rows;
    uint vocab_size;
    uint reserved0;
    uint reserved1;
};

struct UocrNoRepeatNgramParams {
    uint rows;
    uint vocab_size;
    uint max_candidates;
    uint reserved0;
};

struct UocrNoRepeatCollectParams {
    uint sequence_len;
    uint ngram_size;
    uint window;
    uint candidate_count;
    uint vocab_size;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct UocrLmHeadArgmaxParams {
    uint vocab_size;
    uint hidden_size;
    uint tile_tokens;
    uint lanes_per_token;
    uint banned_count;
    uint partial_count;
    uint reserved0;
    uint reserved1;
};

struct UocrArgmaxPairsParams {
    uint count;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_no_repeat_ngram_f32(device float *logits [[buffer(0)]],
                                     device const int *sequences [[buffer(1)]],
                                     device const uint *row_offsets [[buffer(2)]],
                                     device const uint *sequence_lengths [[buffer(3)]],
                                     device const uint *ngram_sizes [[buffer(4)]],
                                     device const uint *windows [[buffer(5)]],
                                     constant UocrNoRepeatNgramParams &params [[buffer(6)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint candidate_offset = gid.x;
    const uint row = gid.y;
    if (row >= params.rows || candidate_offset >= params.max_candidates) {
        return;
    }

    const uint ngram_size = ngram_sizes[row];
    const uint sequence_len = sequence_lengths[row];
    if (ngram_size == 0u || sequence_len < ngram_size) {
        return;
    }

    const uint window = windows[row];
    const uint effective_window = (window == 0u || window > sequence_len) ? sequence_len : window;
    const uint search_start = sequence_len - effective_window;
    const uint search_end = sequence_len - ngram_size + 1u;
    if (search_end <= search_start) {
        return;
    }

    const uint candidate_count = search_end - search_start;
    if (candidate_offset >= candidate_count) {
        return;
    }

    const uint idx = search_start + candidate_offset;
    const uint current_prefix_start = ngram_size > 1u ? sequence_len - (ngram_size - 1u) : sequence_len;
    const uint row_offset = row_offsets[row];
    for (uint j = 0u; j + 1u < ngram_size; ++j) {
        if (sequences[row_offset + idx + j] != sequences[row_offset + current_prefix_start + j]) {
            return;
        }
    }

    const int token_id = sequences[row_offset + idx + ngram_size - 1u];
    if (token_id >= 0 && uint(token_id) < params.vocab_size) {
        logits[ulong(row) * ulong(params.vocab_size) + ulong(uint(token_id))] = -INFINITY;
    }
}

static inline bool uocr_argmax_better(float score, uint id, float best_score, uint best_id) {
    if (isnan(score)) {
        return false;
    }
    if (best_id == 0xffffffffu) {
        return true;
    }
    if (score > best_score) {
        return true;
    }
    if (score < best_score) {
        return false;
    }
    return id < best_id;
}

static inline bool uocr_argmax_pair_better(float score, uint id, float best_score, uint best_id) {
    if (id == 0xffffffffu) {
        return false;
    }
    return uocr_argmax_better(score, id, best_score, best_id);
}

kernel void uocr_no_repeat_collect_banned_i32(device const int *sequence [[buffer(0)]],
                                             device uchar *ban_flags [[buffer(1)]],
                                             constant UocrNoRepeatCollectParams &params [[buffer(2)]],
                                             uint candidate_offset [[thread_position_in_grid]]) {
    if (candidate_offset >= params.candidate_count || params.ngram_size == 0u ||
        params.sequence_len < params.ngram_size) {
        return;
    }

    const uint effective_window = (params.window == 0u || params.window > params.sequence_len) ?
                                      params.sequence_len :
                                      params.window;
    const uint search_start = params.sequence_len - effective_window;
    const uint search_end = params.sequence_len - params.ngram_size + 1u;
    if (search_end <= search_start) {
        return;
    }

    const uint candidate_count = search_end - search_start;
    if (candidate_offset >= candidate_count) {
        return;
    }

    const uint idx = search_start + candidate_offset;
    const uint current_prefix_start = params.ngram_size > 1u ?
                                          params.sequence_len - (params.ngram_size - 1u) :
                                          params.sequence_len;
    for (uint j = 0u; j + 1u < params.ngram_size; ++j) {
        if (sequence[idx + j] != sequence[current_prefix_start + j]) {
            return;
        }
    }

    const int token_id = sequence[idx + params.ngram_size - 1u];
    if (token_id >= 0 && uint(token_id) < params.vocab_size) {
        ban_flags[uint(token_id)] = uchar(1u);
    }
}

kernel void uocr_lm_head_argmax_f16(device const half *hidden [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device const uchar *ban_flags [[buffer(2)]],
                                    device float *partial_scores_out [[buffer(3)]],
                                    device uint *partial_ids_out [[buffer(4)]],
                                    constant UocrLmHeadArgmaxParams &params [[buffer(5)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    threadgroup uint *partial_ids [[threadgroup(1)]],
                                    threadgroup half *hidden_tg [[threadgroup(2)]],
                                    uint tile [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]]) {
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        hidden_tg[k] = hidden[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lanes = params.lanes_per_token;
    const uint local_token = tid / lanes;
    const uint lane = tid - local_token * lanes;
    const uint token_id = tile * params.tile_tokens + local_token;

    float sum = 0.0f;
    if (local_token < params.tile_tokens && token_id < params.vocab_size) {
        const ulong weight_base = ulong(token_id) * ulong(params.hidden_size);
        for (uint k = lane; k < params.hidden_size; k += lanes) {
            sum += float(hidden_tg[k]) * float(weight[weight_base + ulong(k)]);
        }
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lanes >> 1; stride > 0u; stride >>= 1) {
        if (local_token < params.tile_tokens && lane < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_token < params.tile_tokens && lane == 0u) {
        float score = partials[tid];
        uint id = token_id;
        if (token_id >= params.vocab_size) {
            id = 0xffffffffu;
            score = -INFINITY;
        } else if (params.banned_count != 0u && ban_flags[token_id] != uchar(0u)) {
            score = -INFINITY;
        } else if (isnan(score)) {
            id = 0xffffffffu;
            score = -INFINITY;
        }
        partials[local_token] = score;
        partial_ids[local_token] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.tile_tokens >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            const float score = partials[tid + stride];
            const uint id = partial_ids[tid + stride];
            if (uocr_argmax_pair_better(score, id, partials[tid], partial_ids[tid])) {
                partials[tid] = score;
                partial_ids[tid] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u && tile < params.partial_count) {
        partial_scores_out[tile] = partial_ids[0] == 0xffffffffu ? -INFINITY : partials[0];
        partial_ids_out[tile] = partial_ids[0];
    }
}

kernel void uocr_argmax_pairs_f32(device const float *partial_scores [[buffer(0)]],
                                  device const uint *partial_ids_in [[buffer(1)]],
                                  device int *token_id_out [[buffer(2)]],
                                  device float *score_out [[buffer(3)]],
                                  constant UocrArgmaxPairsParams &params [[buffer(4)]],
                                  threadgroup float *scores [[threadgroup(0)]],
                                  threadgroup uint *ids [[threadgroup(1)]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint ntg [[threads_per_threadgroup]]) {
    float best_score = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint i = tid; i < params.count; i += ntg) {
        const float score = partial_scores[i];
        const uint id = partial_ids_in[i];
        if (uocr_argmax_pair_better(score, id, best_score, best_id)) {
            best_score = score;
            best_id = id;
        }
    }
    scores[tid] = best_score;
    ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            const float score = scores[tid + stride];
            const uint id = ids[tid + stride];
            if (uocr_argmax_pair_better(score, id, scores[tid], ids[tid])) {
                scores[tid] = score;
                ids[tid] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const uint id = ids[0] == 0xffffffffu ? 0u : ids[0];
        token_id_out[0] = int(id);
        score_out[0] = ids[0] == 0xffffffffu ? -INFINITY : scores[0];
    }
}

kernel void uocr_argmax_f32(device const float *logits [[buffer(0)]],
                            device uint *token_ids [[buffer(1)]],
                            device float *scores [[buffer(2)]],
                            constant UocrArgmaxParams &params [[buffer(3)]],
                            threadgroup float *partial_scores [[threadgroup(0)]],
                            threadgroup uint *partial_ids [[threadgroup(1)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint ntg [[threads_per_threadgroup]]) {
    if (row >= params.rows) {
        return;
    }

    float best_score = -INFINITY;
    uint best_id = 0xffffffffu;
    const ulong row_base = ulong(row) * ulong(params.vocab_size);
    for (uint col = tid; col < params.vocab_size; col += ntg) {
        const float score = logits[row_base + ulong(col)];
        if (uocr_argmax_better(score, col, best_score, best_id)) {
            best_score = score;
            best_id = col;
        }
    }
    partial_scores[tid] = best_score;
    partial_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float score = partial_scores[tid + stride];
            const uint id = partial_ids[tid + stride];
            if (uocr_argmax_better(score, id, partial_scores[tid], partial_ids[tid])) {
                partial_scores[tid] = score;
                partial_ids[tid] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const uint id = partial_ids[0] == 0xffffffffu ? 0u : partial_ids[0];
        token_ids[row] = id;
        scores[row] = partial_ids[0] == 0xffffffffu ? -INFINITY : partial_scores[0];
    }
}

struct UocrAttentionProjectionParams {
    uint n_tokens;
    uint hidden_size;
    uint projection_count;
    uint reserved;
};

static inline float uocr_attention_projection_dot_f16(device const half *src,
                                                      device const half *weight,
                                                      constant UocrAttentionProjectionParams &params,
                                                      uint token,
                                                      uint out_col,
                                                      uint tid,
                                                      uint ntg,
                                                      threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = token * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
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

kernel void uocr_attention_qkvo_f16_to_f16(device const half *src [[buffer(0)]],
                                           device const half *q_weight [[buffer(1)]],
                                           device const half *k_weight [[buffer(2)]],
                                           device const half *v_weight [[buffer(3)]],
                                           device const half *o_weight [[buffer(4)]],
                                           device half *q_dst [[buffer(5)]],
                                           device half *k_dst [[buffer(6)]],
                                           device half *v_dst [[buffer(7)]],
                                           device half *o_dst [[buffer(8)]],
                                           constant UocrAttentionProjectionParams &params [[buffer(9)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint output_index [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint ntg [[threads_per_threadgroup]]) {
    const uint values_per_projection = params.n_tokens * params.hidden_size;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint token = element / params.hidden_size;
    const uint out_col = element - token * params.hidden_size;
    if (projection >= params.projection_count || token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }

    device const half *weight = q_weight;
    device half *dst = q_dst;
    if (projection == 1u) {
        weight = k_weight;
        dst = k_dst;
    } else if (projection == 2u) {
        weight = v_weight;
        dst = v_dst;
    } else if (projection == 3u) {
        weight = o_weight;
        dst = o_dst;
    }

    const float value = uocr_attention_projection_dot_f16(src, weight, params, token, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[token * params.hidden_size + out_col] = half(value);
    }
}

kernel void uocr_attention_qkvo_f16_to_f32(device const half *src [[buffer(0)]],
                                           device const half *q_weight [[buffer(1)]],
                                           device const half *k_weight [[buffer(2)]],
                                           device const half *v_weight [[buffer(3)]],
                                           device const half *o_weight [[buffer(4)]],
                                           device float *q_dst [[buffer(5)]],
                                           device float *k_dst [[buffer(6)]],
                                           device float *v_dst [[buffer(7)]],
                                           device float *o_dst [[buffer(8)]],
                                           constant UocrAttentionProjectionParams &params [[buffer(9)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint output_index [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint ntg [[threads_per_threadgroup]]) {
    const uint values_per_projection = params.n_tokens * params.hidden_size;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint token = element / params.hidden_size;
    const uint out_col = element - token * params.hidden_size;
    if (projection >= params.projection_count || token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }

    device const half *weight = q_weight;
    device float *dst = q_dst;
    if (projection == 1u) {
        weight = k_weight;
        dst = k_dst;
    } else if (projection == 2u) {
        weight = v_weight;
        dst = v_dst;
    } else if (projection == 3u) {
        weight = o_weight;
        dst = o_dst;
    }

    const float value = uocr_attention_projection_dot_f16(src, weight, params, token, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[token * params.hidden_size + out_col] = value;
    }
}

struct UocrAttentionOutputParams {
    uint n_tokens;
    uint hidden_size;
    uint reserved0;
    uint reserved1;
};

static inline float uocr_attention_output_dot_f16(device const half *src,
                                                  device const half *weight,
                                                  constant UocrAttentionOutputParams &params,
                                                  uint token,
                                                  uint out_col,
                                                  uint tid,
                                                  uint ntg,
                                                  threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = token * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
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

kernel void uocr_attention_output_residual_f16_to_f16(device const half *src [[buffer(0)]],
                                                      device const half *weight [[buffer(1)]],
                                                      device const half *residual [[buffer(2)]],
                                                      device half *dst [[buffer(3)]],
                                                      constant UocrAttentionOutputParams &params [[buffer(4)]],
                                                      threadgroup float *partials [[threadgroup(0)]],
                                                      uint output_index [[threadgroup_position_in_grid]],
                                                      uint tid [[thread_index_in_threadgroup]],
                                                      uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }

    const float projected = uocr_attention_output_dot_f16(src, weight, params, token, out_col, tid, ntg, partials);
    if (tid == 0) {
        const uint dst_index = token * params.hidden_size + out_col;
        dst[dst_index] = half(projected + float(residual[dst_index]));
    }
}

kernel void uocr_attention_output_residual_f16_to_f32(device const half *src [[buffer(0)]],
                                                      device const half *weight [[buffer(1)]],
                                                      device const half *residual [[buffer(2)]],
                                                      device float *dst [[buffer(3)]],
                                                      constant UocrAttentionOutputParams &params [[buffer(4)]],
                                                      threadgroup float *partials [[threadgroup(0)]],
                                                      uint output_index [[threadgroup_position_in_grid]],
                                                      uint tid [[thread_index_in_threadgroup]],
                                                      uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }

    const float projected = uocr_attention_output_dot_f16(src, weight, params, token, out_col, tid, ntg, partials);
    if (tid == 0) {
        const uint dst_index = token * params.hidden_size + out_col;
        dst[dst_index] = projected + float(residual[dst_index]);
    }
}

struct UocrDenseSwigluParams {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint has_residual;
};

kernel void uocr_dense_swiglu_gate_up_f16(device const half *src [[buffer(0)]],
                                          device const half *gate_weight [[buffer(1)]],
                                          device const half *up_weight [[buffer(2)]],
                                          device half *mid [[buffer(3)]],
                                          constant UocrDenseSwigluParams &params [[buffer(4)]],
                                          threadgroup float *partials [[threadgroup(0)]],
                                          uint output_index [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.intermediate_size;
    const uint out_col = output_index - token * params.intermediate_size;
    if (token >= params.n_tokens || out_col >= params.intermediate_size) {
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint src_base = token * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        const float x = float(src[src_base + k]);
        gate_sum += x * float(gate_weight[weight_base + k]);
        up_sum += x * float(up_weight[weight_base + k]);
    }
    gate_partials[tid] = gate_sum;
    up_partials[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_partials[tid] += gate_partials[tid + stride];
            up_partials[tid] += up_partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float gate = gate_partials[0];
        const float up = up_partials[0];
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * params.intermediate_size + out_col] = half(silu * up);
    }
}

kernel void uocr_dense_swiglu_gate_up_q8_0(device const half *src [[buffer(0)]],
                                           device const uchar *gate_weight [[buffer(1)]],
                                           device const uchar *up_weight [[buffer(2)]],
                                           device half *mid [[buffer(3)]],
                                           constant UocrDenseQ8Params &params [[buffer(4)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint output_index [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.out_features;
    const uint out_col = output_index - token * params.out_features;
    if (token >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint src_base = token * params.logical_in_features;
    for (uint k = tid; k < params.logical_in_features; k += ntg) {
        const float x = float(src[src_base + k]);
        gate_sum += x * uocr_q8_0_load_value(gate_weight, params.weight_row_size, out_col, k);
        up_sum += x * uocr_q8_0_load_value(up_weight, params.weight_row_size, out_col, k);
    }
    gate_partials[tid] = gate_sum;
    up_partials[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_partials[tid] += gate_partials[tid + stride];
            up_partials[tid] += up_partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float gate = gate_partials[0];
        const float up = up_partials[0];
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * params.out_features + out_col] = half(silu * up);
    }
}

static inline float uocr_dense_swiglu_down_dot_f16(device const half *mid,
                                                   device const half *down_weight,
                                                   constant UocrDenseSwigluParams &params,
                                                   uint token,
                                                   uint out_col,
                                                   uint tid,
                                                   uint ntg,
                                                   threadgroup float *partials) {
    float sum = 0.0f;
    const uint mid_base = token * params.intermediate_size;
    const uint weight_base = out_col * params.intermediate_size;
    for (uint k = tid; k < params.intermediate_size; k += ntg) {
        sum += float(mid[mid_base + k]) * float(down_weight[weight_base + k]);
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

kernel void uocr_dense_swiglu_down_f16_to_f16(device const half *mid [[buffer(0)]],
                                              device const half *down_weight [[buffer(1)]],
                                              device const half *residual [[buffer(2)]],
                                              device half *dst [[buffer(3)]],
                                              constant UocrDenseSwigluParams &params [[buffer(4)]],
                                              threadgroup float *partials [[threadgroup(0)]],
                                              uint output_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_dense_swiglu_down_dot_f16(mid, down_weight, params, token, out_col, tid, ntg, partials);
    if (tid == 0) {
        const uint dst_index = token * params.hidden_size + out_col;
        if (params.has_residual != 0u) {
            value += float(residual[dst_index]);
        }
        dst[dst_index] = half(value);
    }
}

kernel void uocr_dense_swiglu_down_f16_to_f32(device const half *mid [[buffer(0)]],
                                              device const half *down_weight [[buffer(1)]],
                                              device const half *residual [[buffer(2)]],
                                              device float *dst [[buffer(3)]],
                                              constant UocrDenseSwigluParams &params [[buffer(4)]],
                                              threadgroup float *partials [[threadgroup(0)]],
                                              uint output_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_dense_swiglu_down_dot_f16(mid, down_weight, params, token, out_col, tid, ntg, partials);
    if (tid == 0) {
        const uint dst_index = token * params.hidden_size + out_col;
        if (params.has_residual != 0u) {
            value += float(residual[dst_index]);
        }
        dst[dst_index] = value;
    }
}

struct UocrMoeRouterParams {
    uint n_tokens;
    uint hidden_size;
    uint experts;
    uint top_k;
};

static inline bool uocr_moe_router_topk_better(threadgroup const float *scores, uint a, uint b) {
    const float va = scores[a];
    const float vb = scores[b];
    return va > vb || (va == vb && a < b);
}

static inline float uocr_moe_router_dot_f16(device const half *src,
                                            device const half *weight,
                                            constant UocrMoeRouterParams &params,
                                            uint token,
                                            uint expert,
                                            uint tid,
                                            uint ntg,
                                            threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = token * params.hidden_size;
    const uint weight_base = expert * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
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

kernel void uocr_moe_router_logits_f16_to_f32(device const half *src [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device float *logits [[buffer(2)]],
                                             constant UocrMoeRouterParams &params [[buffer(3)]],
                                             threadgroup float *partials [[threadgroup(0)]],
                                             uint output_index [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.experts;
    const uint expert = output_index - token * params.experts;
    if (token >= params.n_tokens || expert >= params.experts) {
        return;
    }

    const float value = uocr_moe_router_dot_f16(src, weight, params, token, expert, tid, ntg, partials);
    if (tid == 0) {
        logits[token * params.experts + expert] = value;
    }
}

// Unlimited-OCR routing contract: softmax(hidden @ router_weight.T) over all
// 64 experts, greedy top-6, raw selected probabilities, no top-k
// renormalization/scaling, and no DS4 softplus/sqrt/bias transforms.
kernel void uocr_moe_router_softmax_topk_f32(device const float *logits [[buffer(0)]],
                                             device float *probs [[buffer(1)]],
                                             device uint *top_expert_ids [[buffer(2)]],
                                             device float *top_weights [[buffer(3)]],
                                             constant UocrMoeRouterParams &params [[buffer(4)]],
                                             threadgroup float *scratch [[threadgroup(0)]],
                                             uint token [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint ntg [[threads_per_threadgroup]]) {
    if (token >= params.n_tokens) {
        return;
    }

    threadgroup float *scores = scratch;
    threadgroup float *partials = scratch + params.experts;
    threadgroup uint *indices = (threadgroup uint *)(scratch + params.experts + ntg);

    float local_max = -INFINITY;
    const uint row_base = token * params.experts;
    for (uint expert = tid; expert < params.experts; expert += ntg) {
        const float value = logits[row_base + expert];
        scores[expert] = value;
        local_max = max(local_max, value);
    }
    partials[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] = max(partials[tid], partials[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float max_logit = partials[0];
    float local_sum = 0.0f;
    for (uint expert = tid; expert < params.experts; expert += ntg) {
        const float value = exp(scores[expert] - max_logit);
        scores[expert] = value;
        local_sum += value;
    }
    partials[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_sum = 1.0f / partials[0];
    for (uint expert = tid; expert < params.experts; expert += ntg) {
        const float prob = scores[expert] * inv_sum;
        scores[expert] = prob;
        probs[row_base + expert] = prob;
        indices[expert] = expert;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // OCR only needs top-6 from 64 experts, but use the same bitonic
    // index-sort shape as DS4's router/argsort kernels instead of a serial
    // thread-0 scan. This keeps selection deterministic for equal
    // probabilities by preferring the lower expert id; the routed sum is
    // commutative, so descending score order is only a diagnostic convention.
    for (uint k = 2u; k <= params.experts; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            const uint other = tid ^ j;
            if (other > tid && other < params.experts) {
                const uint a = indices[tid];
                const uint b = indices[other];
                bool swap = false;
                if ((tid & k) == 0u) {
                    swap = uocr_moe_router_topk_better(scores, b, a);
                } else {
                    swap = uocr_moe_router_topk_better(scores, a, b);
                }
                if (swap) {
                    indices[tid] = b;
                    indices[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < params.top_k) {
        const uint expert = indices[tid];
        top_expert_ids[token * params.top_k + tid] = expert;
        top_weights[token * params.top_k + tid] = scores[expert];
    }
}

struct UocrMoeSelectedParams {
    uint hidden_size;
    uint intermediate_size;
    uint top_k;
    uint reserved;
};

struct UocrMoeSelectedQ4Params {
    uint hidden_size;
    uint physical_hidden_size;
    uint intermediate_size;
    uint top_k;
    uint gate_row_size;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct UocrMoeSelectedDownQ8Params {
    uint hidden_size;
    uint intermediate_size;
    uint physical_intermediate_size;
    uint top_k;
    uint down_row_size;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct UocrMoeSelectedDownQ4Params {
    uint hidden_size;
    uint intermediate_size;
    uint physical_intermediate_size;
    uint top_k;
    uint down_row_size;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_moe_selected_gate_up_f16(device const half *src [[buffer(0)]],
                                          device const half *gate_weight [[buffer(1)]],
                                          device const half *up_weight [[buffer(2)]],
                                          device half *mid [[buffer(3)]],
                                          constant UocrMoeSelectedParams &params [[buffer(4)]],
                                          threadgroup float *partials [[threadgroup(0)]],
                                          uint output_index [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]]) {
    const uint rank = output_index / params.intermediate_size;
    const uint out_col = output_index - rank * params.intermediate_size;
    if (rank >= params.top_k || out_col >= params.intermediate_size) {
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const ulong weight_base = (ulong(rank) * ulong(params.intermediate_size) + ulong(out_col)) *
                              ulong(params.hidden_size);
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        const float x = float(src[k]);
        gate_sum += x * float(gate_weight[weight_base + ulong(k)]);
        up_sum += x * float(up_weight[weight_base + ulong(k)]);
    }
    gate_partials[tid] = gate_sum;
    up_partials[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_partials[tid] += gate_partials[tid + stride];
            up_partials[tid] += up_partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float gate = gate_partials[0];
        const float up = up_partials[0];
        const float silu = gate / (1.0f + exp(-gate));
        mid[rank * params.intermediate_size + out_col] = half(silu * up);
    }
}

kernel void uocr_moe_selected_gate_up_q4_k(device const half *src [[buffer(0)]],
                                           device const uchar *gate_weight [[buffer(1)]],
                                           device const uchar *up_weight [[buffer(2)]],
                                           device half *mid [[buffer(3)]],
                                           constant UocrMoeSelectedQ4Params &params [[buffer(4)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint output_index [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint ntg [[threads_per_threadgroup]]) {
    const uint rank = output_index / params.intermediate_size;
    const uint out_col = output_index - rank * params.intermediate_size;
    if (rank >= params.top_k || out_col >= params.intermediate_size) {
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint weight_row = rank * params.intermediate_size + out_col;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        const float x = float(src[k]);
        gate_sum += x * uocr_q4_k_load_value(gate_weight, params.gate_row_size, weight_row, k);
        up_sum += x * uocr_q4_k_load_value(up_weight, params.gate_row_size, weight_row, k);
    }
    gate_partials[tid] = gate_sum;
    up_partials[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_partials[tid] += gate_partials[tid + stride];
            up_partials[tid] += up_partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float gate = gate_partials[0];
        const float up = up_partials[0];
        const float silu = gate / (1.0f + exp(-gate));
        mid[rank * params.intermediate_size + out_col] = half(silu * up);
    }
}

static inline float uocr_moe_selected_down_sum_dot_f16(device const half *mid,
                                                       device const half *down_weight,
                                                       device const float *top_weights,
                                                       constant UocrMoeSelectedParams &params,
                                                       uint out_col,
                                                       uint tid,
                                                       uint ntg,
                                                       threadgroup float *partials) {
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        float expert_sum = 0.0f;
        const ulong mid_base = ulong(rank) * ulong(params.intermediate_size);
        const ulong weight_base = (ulong(rank) * ulong(params.hidden_size) + ulong(out_col)) *
                                  ulong(params.intermediate_size);
        for (uint k = tid; k < params.intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) * float(down_weight[weight_base + ulong(k)]);
        }
        sum += expert_sum * top_weights[rank];
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

kernel void uocr_moe_selected_down_sum_f16_to_f16(device const half *mid [[buffer(0)]],
                                                  device const half *down_weight [[buffer(1)]],
                                                  device const float *top_weights [[buffer(2)]],
                                                  device half *dst [[buffer(3)]],
                                                  constant UocrMoeSelectedParams &params [[buffer(4)]],
                                                  threadgroup float *partials [[threadgroup(0)]],
                                                  uint out_col [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint ntg [[threads_per_threadgroup]]) {
    if (out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_selected_down_sum_dot_f16(mid, down_weight, top_weights, params, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[out_col] = half(value);
    }
}

kernel void uocr_moe_selected_down_sum_f16_to_f32(device const half *mid [[buffer(0)]],
                                                  device const half *down_weight [[buffer(1)]],
                                                  device const float *top_weights [[buffer(2)]],
                                                  device float *dst [[buffer(3)]],
                                                  constant UocrMoeSelectedParams &params [[buffer(4)]],
                                                  threadgroup float *partials [[threadgroup(0)]],
                                                  uint out_col [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint ntg [[threads_per_threadgroup]]) {
    if (out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_selected_down_sum_dot_f16(mid, down_weight, top_weights, params, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[out_col] = value;
    }
}

static inline float uocr_moe_selected_down_sum_dot_q8_0(device const half *mid,
                                                        device const uchar *down_weight,
                                                        device const float *top_weights,
                                                        constant UocrMoeSelectedDownQ8Params &params,
                                                        uint out_col,
                                                        uint tid,
                                                        uint ntg,
                                                        threadgroup float *partials) {
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        float expert_sum = 0.0f;
        const ulong mid_base = ulong(rank) * ulong(params.intermediate_size);
        const uint weight_row = rank * params.hidden_size + out_col;
        for (uint k = tid; k < params.intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) *
                          uocr_q8_0_load_value(down_weight, params.down_row_size, weight_row, k);
        }
        sum += expert_sum * top_weights[rank];
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

kernel void uocr_moe_selected_down_sum_q8_0_to_f16(device const half *mid [[buffer(0)]],
                                                   device const uchar *down_weight [[buffer(1)]],
                                                   device const float *top_weights [[buffer(2)]],
                                                   device half *dst [[buffer(3)]],
                                                   constant UocrMoeSelectedDownQ8Params &params [[buffer(4)]],
                                                   threadgroup float *partials [[threadgroup(0)]],
                                                   uint out_col [[threadgroup_position_in_grid]],
                                                   uint tid [[thread_index_in_threadgroup]],
                                                   uint ntg [[threads_per_threadgroup]]) {
    if (out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_selected_down_sum_dot_q8_0(mid, down_weight, top_weights, params, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[out_col] = half(value);
    }
}

kernel void uocr_moe_selected_down_sum_q8_0_to_f32(device const half *mid [[buffer(0)]],
                                                   device const uchar *down_weight [[buffer(1)]],
                                                   device const float *top_weights [[buffer(2)]],
                                                   device float *dst [[buffer(3)]],
                                                   constant UocrMoeSelectedDownQ8Params &params [[buffer(4)]],
                                                   threadgroup float *partials [[threadgroup(0)]],
                                                   uint out_col [[threadgroup_position_in_grid]],
                                                   uint tid [[thread_index_in_threadgroup]],
                                                   uint ntg [[threads_per_threadgroup]]) {
    if (out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_selected_down_sum_dot_q8_0(mid, down_weight, top_weights, params, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[out_col] = value;
    }
}

static inline float uocr_moe_selected_down_sum_dot_q4_k(device const half *mid,
                                                        device const uchar *down_weight,
                                                        device const float *top_weights,
                                                        constant UocrMoeSelectedDownQ4Params &params,
                                                        uint out_col,
                                                        uint tid,
                                                        uint ntg,
                                                        threadgroup float *partials) {
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        float expert_sum = 0.0f;
        const ulong mid_base = ulong(rank) * ulong(params.intermediate_size);
        const uint weight_row = rank * params.hidden_size + out_col;
        for (uint k = tid; k < params.intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) *
                          uocr_q4_k_load_value(down_weight, params.down_row_size, weight_row, k);
        }
        sum += expert_sum * top_weights[rank];
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

kernel void uocr_moe_selected_down_sum_q4_k_to_f16(device const half *mid [[buffer(0)]],
                                                   device const uchar *down_weight [[buffer(1)]],
                                                   device const float *top_weights [[buffer(2)]],
                                                   device half *dst [[buffer(3)]],
                                                   constant UocrMoeSelectedDownQ4Params &params [[buffer(4)]],
                                                   threadgroup float *partials [[threadgroup(0)]],
                                                   uint out_col [[threadgroup_position_in_grid]],
                                                   uint tid [[thread_index_in_threadgroup]],
                                                   uint ntg [[threads_per_threadgroup]]) {
    if (out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_selected_down_sum_dot_q4_k(mid, down_weight, top_weights, params, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[out_col] = half(value);
    }
}

kernel void uocr_moe_selected_down_sum_q4_k_to_f32(device const half *mid [[buffer(0)]],
                                                   device const uchar *down_weight [[buffer(1)]],
                                                   device const float *top_weights [[buffer(2)]],
                                                   device float *dst [[buffer(3)]],
                                                   constant UocrMoeSelectedDownQ4Params &params [[buffer(4)]],
                                                   threadgroup float *partials [[threadgroup(0)]],
                                                   uint out_col [[threadgroup_position_in_grid]],
                                                   uint tid [[thread_index_in_threadgroup]],
                                                   uint ntg [[threads_per_threadgroup]]) {
    if (out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_selected_down_sum_dot_q4_k(mid, down_weight, top_weights, params, out_col, tid, ntg, partials);
    if (tid == 0) {
        dst[out_col] = value;
    }
}

struct UocrMoePrefillSelectedParams {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct UocrMoePrefillSelectedQ4Params {
    uint n_tokens;
    uint hidden_size;
    uint physical_hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint gate_row_size;
    uint reserved0;
};

struct UocrMoePrefillSelectedDownQ8Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint physical_intermediate_size;
    uint expert_count;
    uint top_k;
    uint down_row_size;
    uint reserved0;
};

struct UocrMoePrefillSelectedDownQ4Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint physical_intermediate_size;
    uint expert_count;
    uint top_k;
    uint down_row_size;
    uint reserved0;
};

struct UocrMoePrefillInterleavedParams {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint expert_stride_values;
    uint up_offset_values;
    uint down_offset_values;
};

kernel void uocr_moe_prefill_selected_gate_up_f16(device const half *src [[buffer(0)]],
                                                  device const uint *top_expert_ids [[buffer(1)]],
                                                  device const half *gate_weight [[buffer(2)]],
                                                  device const half *up_weight [[buffer(3)]],
                                                  device half *mid [[buffer(4)]],
                                                  constant UocrMoePrefillSelectedParams &params [[buffer(5)]],
                                                  threadgroup float *partials [[threadgroup(0)]],
                                                  uint output_index [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint ntg [[threads_per_threadgroup]]) {
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
        if (tid == 0) {
            mid[(token * params.top_k + rank) * params.intermediate_size + out_col] = half(0.0f);
        }
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const ulong src_base = ulong(token) * ulong(params.hidden_size);
    const ulong weight_base = (ulong(expert) * ulong(params.intermediate_size) + ulong(out_col)) *
                              ulong(params.hidden_size);
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        const float x = float(src[src_base + ulong(k)]);
        gate_sum += x * float(gate_weight[weight_base + ulong(k)]);
        up_sum += x * float(up_weight[weight_base + ulong(k)]);
    }
    gate_partials[tid] = gate_sum;
    up_partials[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_partials[tid] += gate_partials[tid + stride];
            up_partials[tid] += up_partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float gate = gate_partials[0];
        const float up = up_partials[0];
        const float silu = gate / (1.0f + exp(-gate));
        const ulong mid_index = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                                    ulong(params.intermediate_size) +
                                ulong(out_col);
        mid[mid_index] = half(silu * up);
    }
}

kernel void uocr_moe_prefill_selected_gate_up_q4_k(device const half *src [[buffer(0)]],
                                                   device const uint *top_expert_ids [[buffer(1)]],
                                                   device const uchar *gate_weight [[buffer(2)]],
                                                   device const uchar *up_weight [[buffer(3)]],
                                                   device half *mid [[buffer(4)]],
                                                   constant UocrMoePrefillSelectedQ4Params &params [[buffer(5)]],
                                                   threadgroup float *partials [[threadgroup(0)]],
                                                   uint output_index [[threadgroup_position_in_grid]],
                                                   uint tid [[thread_index_in_threadgroup]],
                                                   uint ntg [[threads_per_threadgroup]]) {
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
        if (tid == 0) {
            mid[(token * params.top_k + rank) * params.intermediate_size + out_col] = half(0.0f);
        }
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint weight_row = expert * params.intermediate_size + out_col;
    const ulong src_base = ulong(token) * ulong(params.hidden_size);
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        const float x = float(src[src_base + ulong(k)]);
        gate_sum += x * uocr_q4_k_load_value(gate_weight, params.gate_row_size, weight_row, k);
        up_sum += x * uocr_q4_k_load_value(up_weight, params.gate_row_size, weight_row, k);
    }
    gate_partials[tid] = gate_sum;
    up_partials[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_partials[tid] += gate_partials[tid + stride];
            up_partials[tid] += up_partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float gate = gate_partials[0];
        const float up = up_partials[0];
        const float silu = gate / (1.0f + exp(-gate));
        const ulong mid_index = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                                    ulong(params.intermediate_size) +
                                ulong(out_col);
        mid[mid_index] = half(silu * up);
    }
}

static inline float uocr_moe_prefill_selected_down_dot_f16(device const half *mid,
                                                           device const uint *top_expert_ids,
                                                           device const float *top_weights,
                                                           device const half *down_weight,
                                                           constant UocrMoePrefillSelectedParams &params,
                                                           uint token,
                                                           uint out_col,
                                                           uint tid,
                                                           uint ntg,
                                                           threadgroup float *partials) {
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        const uint expert = top_expert_ids[token * params.top_k + rank];
        if (expert >= params.expert_count) {
            continue;
        }
        float expert_sum = 0.0f;
        const ulong mid_base = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                               ulong(params.intermediate_size);
        const ulong weight_base = (ulong(expert) * ulong(params.hidden_size) + ulong(out_col)) *
                                  ulong(params.intermediate_size);
        for (uint k = tid; k < params.intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) * float(down_weight[weight_base + ulong(k)]);
        }
        sum += expert_sum * top_weights[token * params.top_k + rank];
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

kernel void uocr_moe_prefill_selected_down_sum_f16_to_f16(device const half *mid [[buffer(0)]],
                                                          device const uint *top_expert_ids [[buffer(1)]],
                                                          device const float *top_weights [[buffer(2)]],
                                                          device const half *down_weight [[buffer(3)]],
                                                          device half *dst [[buffer(4)]],
                                                          constant UocrMoePrefillSelectedParams &params [[buffer(5)]],
                                                          threadgroup float *partials [[threadgroup(0)]],
                                                          uint output_index [[threadgroup_position_in_grid]],
                                                          uint tid [[thread_index_in_threadgroup]],
                                                          uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_selected_down_dot_f16(mid,
                                                               top_expert_ids,
                                                               top_weights,
                                                               down_weight,
                                                               params,
                                                               token,
                                                               out_col,
                                                               tid,
                                                               ntg,
                                                               partials);
    if (tid == 0) {
        dst[output_index] = half(value);
    }
}

kernel void uocr_moe_prefill_selected_down_sum_f16_to_f32(device const half *mid [[buffer(0)]],
                                                          device const uint *top_expert_ids [[buffer(1)]],
                                                          device const float *top_weights [[buffer(2)]],
                                                          device const half *down_weight [[buffer(3)]],
                                                          device float *dst [[buffer(4)]],
                                                          constant UocrMoePrefillSelectedParams &params [[buffer(5)]],
                                                          threadgroup float *partials [[threadgroup(0)]],
                                                          uint output_index [[threadgroup_position_in_grid]],
                                                          uint tid [[thread_index_in_threadgroup]],
                                                          uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_selected_down_dot_f16(mid,
                                                               top_expert_ids,
                                                               top_weights,
                                                               down_weight,
                                                               params,
                                                               token,
                                                               out_col,
                                                               tid,
                                                               ntg,
                                                               partials);
    if (tid == 0) {
        dst[output_index] = value;
    }
}

static inline float uocr_moe_prefill_selected_down_dot_q8_0(device const half *mid,
                                                            device const uint *top_expert_ids,
                                                            device const float *top_weights,
                                                            device const uchar *down_weight,
                                                            constant UocrMoePrefillSelectedDownQ8Params &params,
                                                            uint token,
                                                            uint out_col,
                                                            uint tid,
                                                            uint ntg,
                                                            threadgroup float *partials) {
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        const uint expert = top_expert_ids[token * params.top_k + rank];
        if (expert >= params.expert_count) {
            continue;
        }
        float expert_sum = 0.0f;
        const ulong mid_base = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                               ulong(params.intermediate_size);
        const uint weight_row = expert * params.hidden_size + out_col;
        for (uint k = tid; k < params.intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) *
                          uocr_q8_0_load_value(down_weight, params.down_row_size, weight_row, k);
        }
        sum += expert_sum * top_weights[token * params.top_k + rank];
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

kernel void uocr_moe_prefill_selected_down_sum_q8_0_to_f16(device const half *mid [[buffer(0)]],
                                                           device const uint *top_expert_ids [[buffer(1)]],
                                                           device const float *top_weights [[buffer(2)]],
                                                           device const uchar *down_weight [[buffer(3)]],
                                                           device half *dst [[buffer(4)]],
                                                           constant UocrMoePrefillSelectedDownQ8Params &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_selected_down_dot_q8_0(mid,
                                                                top_expert_ids,
                                                                top_weights,
                                                                down_weight,
                                                                params,
                                                                token,
                                                                out_col,
                                                                tid,
                                                                ntg,
                                                                partials);
    if (tid == 0) {
        dst[output_index] = half(value);
    }
}

kernel void uocr_moe_prefill_selected_down_sum_q8_0_to_f32(device const half *mid [[buffer(0)]],
                                                           device const uint *top_expert_ids [[buffer(1)]],
                                                           device const float *top_weights [[buffer(2)]],
                                                           device const uchar *down_weight [[buffer(3)]],
                                                           device float *dst [[buffer(4)]],
                                                           constant UocrMoePrefillSelectedDownQ8Params &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_selected_down_dot_q8_0(mid,
                                                                top_expert_ids,
                                                                top_weights,
                                                                down_weight,
                                                                params,
                                                                token,
                                                                out_col,
                                                                tid,
                                                                ntg,
                                                                partials);
    if (tid == 0) {
        dst[output_index] = value;
    }
}

static inline float uocr_moe_prefill_selected_down_dot_q4_k(device const half *mid,
                                                            device const uint *top_expert_ids,
                                                            device const float *top_weights,
                                                            device const uchar *down_weight,
                                                            constant UocrMoePrefillSelectedDownQ4Params &params,
                                                            uint token,
                                                            uint out_col,
                                                            uint tid,
                                                            uint ntg,
                                                            threadgroup float *partials) {
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        const uint expert = top_expert_ids[token * params.top_k + rank];
        if (expert >= params.expert_count) {
            continue;
        }
        float expert_sum = 0.0f;
        const ulong mid_base = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                               ulong(params.intermediate_size);
        const uint weight_row = expert * params.hidden_size + out_col;
        for (uint k = tid; k < params.intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) *
                          uocr_q4_k_load_value(down_weight, params.down_row_size, weight_row, k);
        }
        sum += expert_sum * top_weights[token * params.top_k + rank];
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

kernel void uocr_moe_prefill_selected_down_sum_q4_k_to_f16(device const half *mid [[buffer(0)]],
                                                           device const uint *top_expert_ids [[buffer(1)]],
                                                           device const float *top_weights [[buffer(2)]],
                                                           device const uchar *down_weight [[buffer(3)]],
                                                           device half *dst [[buffer(4)]],
                                                           constant UocrMoePrefillSelectedDownQ4Params &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_selected_down_dot_q4_k(mid,
                                                                top_expert_ids,
                                                                top_weights,
                                                                down_weight,
                                                                params,
                                                                token,
                                                                out_col,
                                                                tid,
                                                                ntg,
                                                                partials);
    if (tid == 0) {
        dst[output_index] = half(value);
    }
}

kernel void uocr_moe_prefill_selected_down_sum_q4_k_to_f32(device const half *mid [[buffer(0)]],
                                                           device const uint *top_expert_ids [[buffer(1)]],
                                                           device const float *top_weights [[buffer(2)]],
                                                           device const uchar *down_weight [[buffer(3)]],
                                                           device float *dst [[buffer(4)]],
                                                           constant UocrMoePrefillSelectedDownQ4Params &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_selected_down_dot_q4_k(mid,
                                                                top_expert_ids,
                                                                top_weights,
                                                                down_weight,
                                                                params,
                                                                token,
                                                                out_col,
                                                                tid,
                                                                ntg,
                                                                partials);
    if (tid == 0) {
        dst[output_index] = value;
    }
}

kernel void uocr_moe_prefill_interleaved_gate_up_f16(device const half *src [[buffer(0)]],
                                                     device const uint *top_expert_ids [[buffer(1)]],
                                                     device const half *expert_slab [[buffer(2)]],
                                                     device half *mid [[buffer(3)]],
                                                     constant UocrMoePrefillInterleavedParams &params [[buffer(4)]],
                                                     threadgroup float *partials [[threadgroup(0)]],
                                                     uint output_index [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]],
                                                     uint ntg [[threads_per_threadgroup]]) {
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
        if (tid == 0) {
            mid[(token * params.top_k + rank) * params.intermediate_size + out_col] = half(0.0f);
        }
        return;
    }

    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const ulong src_base = ulong(token) * ulong(params.hidden_size);
    const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values);
    const ulong gate_base = expert_base + ulong(out_col) * ulong(params.hidden_size);
    const ulong up_base = expert_base + ulong(params.up_offset_values) + ulong(out_col) * ulong(params.hidden_size);
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        const float x = float(src[src_base + ulong(k)]);
        gate_sum += x * float(expert_slab[gate_base + ulong(k)]);
        up_sum += x * float(expert_slab[up_base + ulong(k)]);
    }
    gate_partials[tid] = gate_sum;
    up_partials[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_partials[tid] += gate_partials[tid + stride];
            up_partials[tid] += up_partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float gate = gate_partials[0];
        const float up = up_partials[0];
        const float silu = gate / (1.0f + exp(-gate));
        const ulong mid_index = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                                    ulong(params.intermediate_size) +
                                ulong(out_col);
        mid[mid_index] = half(silu * up);
    }
}

static inline float uocr_moe_prefill_interleaved_down_dot_f16(device const half *mid,
                                                              device const uint *top_expert_ids,
                                                              device const float *top_weights,
                                                              device const half *expert_slab,
                                                              constant UocrMoePrefillInterleavedParams &params,
                                                              uint token,
                                                              uint out_col,
                                                              uint tid,
                                                              uint ntg,
                                                              threadgroup float *partials) {
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        const uint expert = top_expert_ids[token * params.top_k + rank];
        if (expert >= params.expert_count) {
            continue;
        }
        float expert_sum = 0.0f;
        const ulong mid_base = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                               ulong(params.intermediate_size);
        const ulong weight_base = ulong(expert) * ulong(params.expert_stride_values) +
                                  ulong(params.down_offset_values) +
                                  ulong(out_col) * ulong(params.intermediate_size);
        for (uint k = tid; k < params.intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) * float(expert_slab[weight_base + ulong(k)]);
        }
        sum += expert_sum * top_weights[token * params.top_k + rank];
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

kernel void uocr_moe_prefill_interleaved_down_sum_f16_to_f16(device const half *mid [[buffer(0)]],
                                                             device const uint *top_expert_ids [[buffer(1)]],
                                                             device const float *top_weights [[buffer(2)]],
                                                             device const half *expert_slab [[buffer(3)]],
                                                             device half *dst [[buffer(4)]],
                                                             constant UocrMoePrefillInterleavedParams &params [[buffer(5)]],
                                                             threadgroup float *partials [[threadgroup(0)]],
                                                             uint output_index [[threadgroup_position_in_grid]],
                                                             uint tid [[thread_index_in_threadgroup]],
                                                             uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_interleaved_down_dot_f16(mid,
                                                                  top_expert_ids,
                                                                  top_weights,
                                                                  expert_slab,
                                                                  params,
                                                                  token,
                                                                  out_col,
                                                                  tid,
                                                                  ntg,
                                                                  partials);
    if (tid == 0) {
        dst[output_index] = half(value);
    }
}

kernel void uocr_moe_prefill_interleaved_down_sum_f16_to_f32(device const half *mid [[buffer(0)]],
                                                             device const uint *top_expert_ids [[buffer(1)]],
                                                             device const float *top_weights [[buffer(2)]],
                                                             device const half *expert_slab [[buffer(3)]],
                                                             device float *dst [[buffer(4)]],
                                                             constant UocrMoePrefillInterleavedParams &params [[buffer(5)]],
                                                             threadgroup float *partials [[threadgroup(0)]],
                                                             uint output_index [[threadgroup_position_in_grid]],
                                                             uint tid [[thread_index_in_threadgroup]],
                                                             uint ntg [[threads_per_threadgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_prefill_interleaved_down_dot_f16(mid,
                                                                  top_expert_ids,
                                                                  top_weights,
                                                                  expert_slab,
                                                                  params,
                                                                  token,
                                                                  out_col,
                                                                  tid,
                                                                  ntg,
                                                                  partials);
    if (tid == 0) {
        dst[output_index] = value;
    }
}

struct UocrMoeCombineParams {
    uint n_tokens;
    uint hidden_size;
    uint has_residual;
    uint reserved;
};

static inline float uocr_moe_combine_value_f16(device const half *routed,
                                               device const half *shared,
                                               device const half *residual,
                                               constant UocrMoeCombineParams &params,
                                               uint gid) {
    float value = float(routed[gid]) + float(shared[gid]);
    if (params.has_residual != 0u) {
        value += float(residual[gid]);
    }
    return value;
}

kernel void uocr_moe_combine_f16_to_f16(device const half *routed [[buffer(0)]],
                                        device const half *shared [[buffer(1)]],
                                        device const half *residual [[buffer(2)]],
                                        device half *dst [[buffer(3)]],
                                        constant UocrMoeCombineParams &params [[buffer(4)]],
                                        uint gid [[thread_position_in_grid]]) {
    const uint total = params.n_tokens * params.hidden_size;
    if (gid >= total) {
        return;
    }
    dst[gid] = half(uocr_moe_combine_value_f16(routed, shared, residual, params, gid));
}

kernel void uocr_moe_combine_f16_to_f32(device const half *routed [[buffer(0)]],
                                        device const half *shared [[buffer(1)]],
                                        device const half *residual [[buffer(2)]],
                                        device float *dst [[buffer(3)]],
                                        constant UocrMoeCombineParams &params [[buffer(4)]],
                                        uint gid [[thread_position_in_grid]]) {
    const uint total = params.n_tokens * params.hidden_size;
    if (gid >= total) {
        return;
    }
    dst[gid] = uocr_moe_combine_value_f16(routed, shared, residual, params, gid);
}

struct UocrRopeQKParams {
    uint n_tokens;
    uint heads;
    uint head_dim;
    uint position_start;
    float freq_scale;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline void uocr_rope_pair(uint gid,
                                  constant UocrRopeQKParams &params,
                                  thread uint &token,
                                  thread uint &head,
                                  thread uint &a,
                                  thread uint &b,
                                  thread float &c,
                                  thread float &s) {
    const uint half_dim = params.head_dim >> 1u;
    const uint pair = gid % half_dim;
    const uint token_head = gid / half_dim;
    head = token_head % params.heads;
    token = token_head / params.heads;
    a = pair;
    b = pair + half_dim;
    const float angle = float(params.position_start + token) * exp2(float(pair) * params.freq_scale);
    c = cos(angle);
    s = sin(angle);
}

kernel void uocr_rope_qk_f16_to_f16(device const half *q_src [[buffer(0)]],
                                    device const half *k_src [[buffer(1)]],
                                    device half *q_dst [[buffer(2)]],
                                    device half *k_dst [[buffer(3)]],
                                    constant UocrRopeQKParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint half_dim = params.head_dim >> 1u;
    const uint total = params.n_tokens * params.heads * half_dim;
    if (gid >= total) {
        return;
    }

    uint token;
    uint head;
    uint a;
    uint b;
    float c;
    float s;
    uocr_rope_pair(gid, params, token, head, a, b, c, s);
    const ulong base = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim);
    const float q0 = float(q_src[base + ulong(a)]);
    const float q1 = float(q_src[base + ulong(b)]);
    const float k0 = float(k_src[base + ulong(a)]);
    const float k1 = float(k_src[base + ulong(b)]);
    q_dst[base + ulong(a)] = half(q0 * c - q1 * s);
    q_dst[base + ulong(b)] = half(q0 * s + q1 * c);
    k_dst[base + ulong(a)] = half(k0 * c - k1 * s);
    k_dst[base + ulong(b)] = half(k0 * s + k1 * c);
}

kernel void uocr_rope_qk_f16_to_f32(device const half *q_src [[buffer(0)]],
                                    device const half *k_src [[buffer(1)]],
                                    device float *q_dst [[buffer(2)]],
                                    device float *k_dst [[buffer(3)]],
                                    constant UocrRopeQKParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint half_dim = params.head_dim >> 1u;
    const uint total = params.n_tokens * params.heads * half_dim;
    if (gid >= total) {
        return;
    }

    uint token;
    uint head;
    uint a;
    uint b;
    float c;
    float s;
    uocr_rope_pair(gid, params, token, head, a, b, c, s);
    const ulong base = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim);
    const float q0 = float(q_src[base + ulong(a)]);
    const float q1 = float(q_src[base + ulong(b)]);
    const float k0 = float(k_src[base + ulong(a)]);
    const float k1 = float(k_src[base + ulong(b)]);
    q_dst[base + ulong(a)] = q0 * c - q1 * s;
    q_dst[base + ulong(b)] = q0 * s + q1 * c;
    k_dst[base + ulong(a)] = k0 * c - k1 * s;
    k_dst[base + ulong(b)] = k0 * s + k1 * c;
}

struct UocrPrefillAttentionParams {
    uint n_tokens;
    uint heads;
    uint head_dim;
    float scale;
};

struct UocrPrefillAttentionVarlenParams {
    uint total_tokens;
    uint batch;
    uint heads;
    uint head_dim;
    float scale;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline void uocr_prefill_varlen_find_sequence(device const uint *cu,
                                                     constant UocrPrefillAttentionVarlenParams &params,
                                                     uint token,
                                                     thread uint &seq_start,
                                                     thread uint &seq_end) {
    seq_start = 0u;
    seq_end = 0u;
    for (uint b = 0u; b < params.batch; ++b) {
        const uint start = cu[b];
        const uint end = cu[b + 1u];
        if (token >= start && token < end) {
            seq_start = start;
            seq_end = end;
            return;
        }
    }
}

kernel void uocr_prefill_attention_varlen_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                     device const half *k_src [[buffer(1)]],
                                                     device const half *v_src [[buffer(2)]],
                                                     device const uint *cu [[buffer(3)]],
                                                     device half *dst [[buffer(4)]],
                                                     constant UocrPrefillAttentionVarlenParams &params [[buffer(5)]],
                                                     threadgroup float *partials [[threadgroup(0)]],
                                                     uint group_index [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]],
                                                     uint ntg [[threads_per_threadgroup]]) {
    const uint token = group_index / params.heads;
    const uint head = group_index - token * params.heads;
    if (token >= params.total_tokens || head >= params.heads || tid >= params.head_dim || ntg < params.head_dim) {
        return;
    }

    uint seq_start;
    uint seq_end;
    uocr_prefill_varlen_find_sequence(cu, params, token, seq_start, seq_end);
    if (seq_end <= seq_start) {
        return;
    }

    const ulong q_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
    const float q = float(q_src[q_index]);
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    for (uint key_token = seq_start; key_token <= token; ++key_token) {
        const ulong k_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        partials[tid] = q * float(k_src[k_index]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float score = partials[0] * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        acc = acc * corr + e * float(v_src[v_index]);
        l = l * corr + e;
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong dst_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
    dst[dst_index] = half(l > 0.0f ? acc / l : 0.0f);
}

kernel void uocr_prefill_attention_varlen_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                     device const half *k_src [[buffer(1)]],
                                                     device const half *v_src [[buffer(2)]],
                                                     device const uint *cu [[buffer(3)]],
                                                     device float *dst [[buffer(4)]],
                                                     constant UocrPrefillAttentionVarlenParams &params [[buffer(5)]],
                                                     threadgroup float *partials [[threadgroup(0)]],
                                                     uint group_index [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]],
                                                     uint ntg [[threads_per_threadgroup]]) {
    const uint token = group_index / params.heads;
    const uint head = group_index - token * params.heads;
    if (token >= params.total_tokens || head >= params.heads || tid >= params.head_dim || ntg < params.head_dim) {
        return;
    }

    uint seq_start;
    uint seq_end;
    uocr_prefill_varlen_find_sequence(cu, params, token, seq_start, seq_end);
    if (seq_end <= seq_start) {
        return;
    }

    const ulong q_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
    const float q = float(q_src[q_index]);
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    for (uint key_token = seq_start; key_token <= token; ++key_token) {
        const ulong k_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        partials[tid] = q * float(k_src[k_index]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float score = partials[0] * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        acc = acc * corr + e * float(v_src[v_index]);
        l = l * corr + e;
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong dst_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
    dst[dst_index] = l > 0.0f ? acc / l : 0.0f;
}

struct UocrDecodeAttentionParams {
    uint batch_slots;
    uint cache_token_capacity;
    uint layer;
    uint slot;
    uint prompt_length;
    uint generated_count;
    uint attention_length;
    uint first_generated;
    uint heads;
    uint head_dim;
    uint ring_window;
    float scale;
};

/* OCR adapts the gradients.c sdpa_decode prefix/window rule to a compact
 * attention stream: [all prompt prefix tokens] followed by the live generated
 * window. The generated window is a 128-token ring whose logical first token is
 * supplied by the host as first_generated.
 */
static inline uint uocr_decode_attention_cache_token(constant UocrDecodeAttentionParams &params,
                                                     uint attention_index) {
    if (attention_index < params.prompt_length) {
        return attention_index;
    }
    const uint generated_index = params.first_generated + (attention_index - params.prompt_length);
    return params.prompt_length + (generated_index % params.ring_window);
}

static inline ulong uocr_decode_attention_cache_index(constant UocrDecodeAttentionParams &params,
                                                      uint cache_token,
                                                      uint head,
                                                      uint dim) {
    return (((ulong(params.layer) * ulong(params.batch_slots) + ulong(params.slot)) *
             ulong(params.cache_token_capacity) + ulong(cache_token)) *
            ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
}

#define UOCR_FLASH_SIMD_WIDTH 32u
#define UOCR_FLASH_Q_PER_TG 4u
#define UOCR_FLASH_MAX_LANE_VALUES 4u
#define UOCR_FLASH_NEG_INF (-3.4028234663852886e38f)

static inline uint uocr_flash_lane_dim(uint lane, uint component) {
    return lane + component * UOCR_FLASH_SIMD_WIDTH;
}

template <typename out_t>
static inline void uocr_sam_window_attention_flash_impl(device const half *q_src,
                                                        device const half *k_src,
                                                        device const half *v_src,
                                                        device out_t *dst,
                                                        constant UocrSamWindowAttentionParams &params,
                                                        uint3 tg,
                                                        ushort lane_u16,
                                                        ushort simdgroup_u16) {
    const uint lane = uint(lane_u16);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint window = tg.z;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || window >= params.windows ||
        query_token >= params.tokens_per_window || head >= params.heads ||
        params.head_dim == 0u || params.head_dim > UOCR_FLASH_SIMD_WIDTH * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong q_index = uocr_sam_window_attention_index(params, window, query_token, head, dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token < params.tokens_per_window; ++key_token) {
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong k_index = uocr_sam_window_attention_index(params, window, key_token, head, dim);
                local_dot += qv[i] * float(k_src[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong v_index = uocr_sam_window_attention_index(params, window, key_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong dst_index = uocr_sam_window_attention_index(params, window, query_token, head, dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

kernel void uocr_sam_window_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                       device const half *k_src [[buffer(1)]],
                                                       device const half *v_src [[buffer(2)]],
                                                       device half *dst [[buffer(3)]],
                                                       constant UocrSamWindowAttentionParams &params [[buffer(4)]],
                                                       uint3 tg [[threadgroup_position_in_grid]],
                                                       ushort lane [[thread_index_in_simdgroup]],
                                                       ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    uocr_sam_window_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup);
}

kernel void uocr_sam_window_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                       device const half *k_src [[buffer(1)]],
                                                       device const half *v_src [[buffer(2)]],
                                                       device float *dst [[buffer(3)]],
                                                       constant UocrSamWindowAttentionParams &params [[buffer(4)]],
                                                       uint3 tg [[threadgroup_position_in_grid]],
                                                       ushort lane [[thread_index_in_simdgroup]],
                                                       ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    uocr_sam_window_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup);
}

template <typename out_t>
static inline void uocr_sam_rel_pos_attention_flash_impl(device const half *q_src,
                                                         device const half *k_src,
                                                         device const half *v_src,
                                                         device const half *rel_pos_h,
                                                         device const half *rel_pos_w,
                                                         device out_t *dst,
                                                         constant UocrSamRelPosAttentionParams &params,
                                                         uint3 tg,
                                                         ushort lane_u16,
                                                         ushort simdgroup_u16) {
    const uint lane = uint(lane_u16);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint window = tg.z;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || window >= params.windows ||
        query_token >= params.tokens_per_window || head >= params.heads ||
        params.grid_width == 0u || params.grid_height == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height ||
        params.head_dim == 0u || params.head_dim > UOCR_FLASH_SIMD_WIDTH * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    const uint query_y = query_token / params.grid_width;
    const uint query_x = query_token - query_y * params.grid_width;
    const uint target_h_length = 2u * params.grid_height - 1u;
    const uint target_w_length = 2u * params.grid_width - 1u;

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong q_index = uocr_sam_rel_pos_attention_index(params, window, query_token, head, dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token < params.tokens_per_window; ++key_token) {
        const uint key_y = key_token / params.grid_width;
        const uint key_x = key_token - key_y * params.grid_width;
        const uint rel_h_index = uint(int(query_y) - int(key_y) + int(params.grid_height) - 1);
        const uint rel_w_index = uint(int(query_x) - int(key_x) + int(params.grid_width) - 1);
        float local_score = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong k_index = uocr_sam_rel_pos_attention_index(params, window, key_token, head, dim);
                const float rh = uocr_sam_rel_pos_table_value(rel_pos_h,
                                                              params.rel_pos_h_length,
                                                              target_h_length,
                                                              rel_h_index,
                                                              dim,
                                                              params.head_dim);
                const float rw = uocr_sam_rel_pos_table_value(rel_pos_w,
                                                              params.rel_pos_w_length,
                                                              target_w_length,
                                                              rel_w_index,
                                                              dim,
                                                              params.head_dim);
                local_score += qv[i] * (float(k_src[k_index]) * params.scale + rh + rw);
            }
        }
        const float score = simd_sum(local_score);
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong v_index = uocr_sam_rel_pos_attention_index(params, window, key_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong dst_index = uocr_sam_rel_pos_attention_index(params, window, query_token, head, dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

kernel void uocr_sam_rel_pos_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                        device const half *k_src [[buffer(1)]],
                                                        device const half *v_src [[buffer(2)]],
                                                        device const half *rel_pos_h [[buffer(3)]],
                                                        device const half *rel_pos_w [[buffer(4)]],
                                                        device half *dst [[buffer(5)]],
                                                        constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                        uint3 tg [[threadgroup_position_in_grid]],
                                                        ushort lane [[thread_index_in_simdgroup]],
                                                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    uocr_sam_rel_pos_attention_flash_impl(q_src, k_src, v_src, rel_pos_h, rel_pos_w, dst, params, tg, lane, simdgroup);
}

kernel void uocr_sam_rel_pos_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                        device const half *k_src [[buffer(1)]],
                                                        device const half *v_src [[buffer(2)]],
                                                        device const half *rel_pos_h [[buffer(3)]],
                                                        device const half *rel_pos_w [[buffer(4)]],
                                                        device float *dst [[buffer(5)]],
                                                        constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                        uint3 tg [[threadgroup_position_in_grid]],
                                                        ushort lane [[thread_index_in_simdgroup]],
                                                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    uocr_sam_rel_pos_attention_flash_impl(q_src, k_src, v_src, rel_pos_h, rel_pos_w, dst, params, tg, lane, simdgroup);
}

template <typename out_t>
static inline void uocr_prefill_attention_flash_impl(device const half *q_src,
                                                     device const half *k_src,
                                                     device const half *v_src,
                                                     device out_t *dst,
                                                     constant UocrPrefillAttentionParams &params,
                                                     uint2 tg,
                                                     ushort lane_u16,
                                                     ushort simdgroup_u16) {
    const uint lane = uint(lane_u16);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || query_token >= params.n_tokens || head >= params.heads ||
        params.head_dim == 0u || params.head_dim > UOCR_FLASH_SIMD_WIDTH * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong q_index = (ulong(query_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token <= query_token; ++key_token) {
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong k_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
                local_dot += qv[i] * float(k_src[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong dst_index = (ulong(query_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

kernel void uocr_prefill_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                    device const half *k_src [[buffer(1)]],
                                                    device const half *v_src [[buffer(2)]],
                                                    device half *dst [[buffer(3)]],
                                                    constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                                    uint2 tg [[threadgroup_position_in_grid]],
                                                    ushort lane [[thread_index_in_simdgroup]],
                                                    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    uocr_prefill_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup);
}

kernel void uocr_prefill_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                    device const half *k_src [[buffer(1)]],
                                                    device const half *v_src [[buffer(2)]],
                                                    device float *dst [[buffer(3)]],
                                                    constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                                    uint2 tg [[threadgroup_position_in_grid]],
                                                    ushort lane [[thread_index_in_simdgroup]],
                                                    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    uocr_prefill_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup);
}

template <typename out_t>
static inline void uocr_decode_attention_flash_impl(device const half *q_src,
                                                    device const half *k_cache,
                                                    device const half *v_cache,
                                                    device out_t *dst,
                                                    constant UocrDecodeAttentionParams &params,
                                                    uint head,
                                                    ushort lane_u16) {
    const uint lane = uint(lane_u16);
    if (head >= params.heads || params.attention_length == 0u ||
        params.head_dim == 0u || params.head_dim > UOCR_FLASH_SIMD_WIDTH * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong q_index = ulong(head) * ulong(params.head_dim) + ulong(dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint attention_index = 0u; attention_index < params.attention_length; ++attention_index) {
        const uint cache_token = uocr_decode_attention_cache_token(params, attention_index);
        if (cache_token >= params.cache_token_capacity) {
            continue;
        }
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong k_index = uocr_decode_attention_cache_index(params, cache_token, head, dim);
                local_dot += qv[i] * float(k_cache[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i);
            if (dim < params.head_dim) {
                const ulong v_index = uocr_decode_attention_cache_index(params, cache_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_cache[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i);
        if (dim < params.head_dim) {
            const ulong dst_index = ulong(head) * ulong(params.head_dim) + ulong(dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

kernel void uocr_decode_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                   device const half *k_cache [[buffer(1)]],
                                                   device const half *v_cache [[buffer(2)]],
                                                   device half *dst [[buffer(3)]],
                                                   constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                                   uint head [[threadgroup_position_in_grid]],
                                                   ushort lane [[thread_index_in_simdgroup]]) {
    uocr_decode_attention_flash_impl(q_src, k_cache, v_cache, dst, params, head, lane);
}

kernel void uocr_decode_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                   device const half *k_cache [[buffer(1)]],
                                                   device const half *v_cache [[buffer(2)]],
                                                   device float *dst [[buffer(3)]],
                                                   constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                                   uint head [[threadgroup_position_in_grid]],
                                                   ushort lane [[thread_index_in_simdgroup]]) {
    uocr_decode_attention_flash_impl(q_src, k_cache, v_cache, dst, params, head, lane);
}

struct UocrKVCacheWriteParams {
    uint n_tokens;
    uint batch_slots;
    uint cache_token_capacity;
    uint layer;
    uint slot;
    uint prompt_length;
    uint position_start;
    uint heads;
    uint head_dim;
    uint ring_window;
    uint reserved0;
    uint reserved1;
};

kernel void uocr_kv_cache_write_f16(device const half *k_src [[buffer(0)]],
                                    device const half *v_src [[buffer(1)]],
                                    device half *k_cache [[buffer(2)]],
                                    device half *v_cache [[buffer(3)]],
                                    constant UocrKVCacheWriteParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint head_area = params.heads * params.head_dim;
    const uint total = params.n_tokens * head_area;
    if (gid >= total) {
        return;
    }

    const uint dim = gid % params.head_dim;
    const uint head = (gid / params.head_dim) % params.heads;
    const uint token = gid / head_area;
    const uint position = params.position_start + token;
    uint cache_token = position;
    if (position >= params.prompt_length) {
        cache_token = params.prompt_length + ((position - params.prompt_length) % params.ring_window);
    }
    if (cache_token >= params.cache_token_capacity) {
        return;
    }

    const ulong src_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
    const ulong dst_index = (((ulong(params.layer) * ulong(params.batch_slots) + ulong(params.slot)) *
                              ulong(params.cache_token_capacity) + ulong(cache_token)) *
                             ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
    k_cache[dst_index] = k_src[src_index];
    v_cache[dst_index] = v_src[src_index];
}
