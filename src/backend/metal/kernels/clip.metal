// clip.metal - CLIP residual/embedding kernels
// Extracted from uocr_smoke.metal
//
#include "common.metal"
kernel void uocr_clip_residual_add_f16_to_f16(device const half *base [[buffer(0)]],
                                              device const half *update [[buffer(1)]],
                                              device half *dst [[buffer(2)]],
                                              constant uint &value_count [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]]) {
    const uint value_base = gid * UOCR_HALF4_WIDTH;
    if (value_base >= value_count) {
        return;
    }
    if (value_base + (UOCR_HALF4_WIDTH - 1u) < value_count) {
        const half4 base_values = uocr_load_half4(base, ulong(value_base));
        const half4 update_values = uocr_load_half4(update, ulong(value_base));
        uocr_store_half4(dst, ulong(value_base), half4(float4(base_values) + float4(update_values)));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && value_base + i < value_count; ++i) {
            const uint idx = value_base + i;
            dst[idx] = half(float(base[idx]) + float(update[idx]));
        }
    }
}

// CLIP embed SAM and absolute position
struct UocrClipEmbedSamParams {
    uint grid_width;
    uint grid_height;
    uint hidden_size;
    uint token_count;
    uint batch_size;
};

static inline float uocr_clip_embedding_from_sam_value(device const half *sam_nchw,
                                                       device const half *class_embedding,
                                                       constant UocrClipEmbedSamParams &params,
                                                       uint token,
                                                       uint channel,
                                                       uint batch) {
    if (token == 0u) {
        return float(class_embedding[channel]);
    }
    const uint spatial = token - 1u;
    const uint y = spatial / params.grid_width;
    const uint x = spatial - y * params.grid_width;
    const ulong spatial_size = ulong(params.grid_width) * ulong(params.grid_height);
    const ulong src_index = ulong(batch) * spatial_size * ulong(params.hidden_size) +
                            ulong(channel) * spatial_size + ulong(y) * ulong(params.grid_width) + ulong(x);
    return float(sam_nchw[src_index]);
}

kernel void uocr_clip_embed_sam_f16_to_f16(device const half *sam_nchw [[buffer(0)]],
                                           device const half *class_embedding [[buffer(1)]],
                                           device half *dst_tokens [[buffer(2)]],
                                           constant UocrClipEmbedSamParams &params [[buffer(3)]],
                                           uint3 gid [[thread_position_in_grid]]) {
    const uint channel = gid.x;
    const uint token = gid.y;
    const uint batch = gid.z;
    if (channel >= params.hidden_size || token >= params.token_count || batch >= params.batch_size) {
        return;
    }
    const float value = uocr_clip_embedding_from_sam_value(sam_nchw, class_embedding, params, token, channel, batch);
    dst_tokens[(ulong(batch) * ulong(params.token_count) + ulong(token)) * ulong(params.hidden_size) + ulong(channel)] = half(value);
}
struct UocrClipAbsPosParams {
    uint source_grid;
    uint target_width;
    uint target_height;
    uint hidden_size;
    uint batch_size;
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
    const uint values_per_view = token_count * params.hidden_size;
    const uint total = values_per_view * params.batch_size;
    if (gid >= total) {
        return;
    }
    const uint view_value = gid % values_per_view;
    const uint channel = view_value % params.hidden_size;
    const uint token = view_value / params.hidden_size;
    const float pos = uocr_clip_abs_pos_bicubic_antialias(pos_embed, params, token, channel);
    dst_tokens[gid] = half(float(tokens[gid]) + pos);
}
