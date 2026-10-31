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

// Noncausal CLIP attention, including the class token. Reuse each K/V tile
// across 16 queries, with FP32 probabilities and accumulation. Only tile tails
// beyond the 101/257-token sequence are masked; no device workspace is added.
[[max_total_threads_per_threadgroup(128)]]
kernel void uocr_clip_attention_mma_f16(
    device const half *q_src [[buffer(0)]],
    device const half *k_src [[buffer(1)]],
    device const half *v_src [[buffer(2)]],
    device half *dst [[buffer(3)]],
    constant UocrSamWindowAttentionParams &params [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint query_base = tg.y * 16u;
    const uint head = tg.x;
    const uint logical_window = tg.z;
    const uint tokens = params.tokens_per_window;
    if ((tokens != 101u && tokens != 257u) || params.head_dim != 64u ||
        head >= params.heads || query_base >= tokens ||
        logical_window >= params.windows * uocr_sam_window_attention_batch_size(params)) {
        return;
    }
    threadgroup half q_tile[16u * 64u];
    threadgroup half kv_tile[32u * 64u];
    threadgroup float probabilities[16u * 32u];
    // Scores and the rescaled output use this storage at separate barriers.
    threadgroup float work[16u * 64u];
    threadgroup float maxima[16u], sums[16u], correction[16u];
    const ulong base = (ulong(logical_window) * ulong(tokens) * ulong(params.heads) + ulong(head)) * 64ul;
    const uint stride = params.heads * 64u;
    for (uint i = tid; i < 16u * 64u; i += 128u) {
        const uint row = query_base + i / 64u;
        q_tile[i] = row < tokens ? q_src[base + ulong(row) * stride + i % 64u] : half(0);
    }
    if (tid < 16u) {
        maxima[tid] = UOCR_FLASH_NEG_INF;
        sums[tid] = 0.0f;
    }
    simdgroup_float8x8 acc[4];
    for (uint j = 0u; j < 4u; ++j) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint query_row = (uint(simdgroup) / 2u) * 8u;
    const uint key_col = (uint(simdgroup) % 2u) * 16u;
    const uint output_col = (uint(simdgroup) % 2u) * 32u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint key_base = 0u; key_base < tokens; key_base += 32u) {
        for (uint i = tid; i < 32u * 64u; i += 128u) {
            const uint key = key_base + i / 64u;
            kv_tile[i] = key < tokens ? k_src[base + ulong(key) * stride + i % 64u] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 score[2];
        for (uint j = 0u; j < 2u; ++j) score[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0u; d < 64u; d += 8u) {
            simdgroup_half8x8 q, k;
            simdgroup_load(q, q_tile + query_row * 64u + d, 64u);
            for (uint j = 0u; j < 2u; ++j) {
                simdgroup_load(k, kv_tile + (key_col + j * 8u) * 64u + d, 64u, ulong2(0), true);
                simdgroup_multiply_accumulate(score[j], q, k, score[j]);
            }
        }
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(score[j], work + query_row * 32u + key_col + j * 8u, 32u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One SIMD group reduces each row's 32 scores; four rows per group.
        for (uint r = 0u; r < 4u; ++r) {
            const uint row = uint(simdgroup) * 4u + r;
            const uint key = key_base + uint(lane);
            float s = work[row * 32u + uint(lane)] * params.scale;
            if (key >= tokens) s = UOCR_FLASH_NEG_INF;
            const float m = max(maxima[row], simd_max(s));
            const float rescale = exp(maxima[row] - m);
            const float e = exp(s - m);
            const float sum = sums[row] * rescale + simd_sum(e);
            probabilities[row * 32u + uint(lane)] = e;
            if (lane == 0u) {
                maxima[row] = m;
                sums[row] = sum;
                correction[row] = rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 4u; ++j) {
            simdgroup_store(acc[j], work + query_row * 64u + output_col + j * 8u, 64u);
        }
        // K is no longer consumed; reuse its tile for V.
        for (uint i = tid; i < 32u * 64u; i += 128u) {
            const uint key = key_base + i / 64u;
            kv_tile[i] = key < tokens ? v_src[base + ulong(key) * stride + i % 64u] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 16u * 64u; i += 128u) work[i] *= correction[i / 64u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 4u; ++j) {
            simdgroup_load(acc[j], work + query_row * 64u + output_col + j * 8u, 64u);
        }
        for (uint t = 0u; t < 32u; t += 8u) {
            simdgroup_float8x8 p;
            simdgroup_half8x8 v;
            simdgroup_load(p, probabilities + query_row * 32u + t, 32u);
            for (uint j = 0u; j < 4u; ++j) {
                simdgroup_load(v, kv_tile + t * 64u + output_col + j * 8u, 64u);
                simdgroup_multiply_accumulate(acc[j], p, v, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0u; j < 4u; ++j) {
        simdgroup_store(acc[j], work + query_row * 64u + output_col + j * 8u, 64u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16u * 64u; i += 128u) {
        const uint row = query_base + i / 64u;
        const float inv_sum = sums[i / 64u] > 0.0f ? 1.0f / sums[i / 64u] : 0.0f;
        if (row < tokens) dst[base + ulong(row) * stride + i % 64u] = half(work[i] * inv_sum);
    }
}
