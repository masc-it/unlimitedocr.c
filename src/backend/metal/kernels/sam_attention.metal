// sam_attention.metal - SAM QKV projection
// Extracted from uocr_smoke.metal
//
#include "common.metal"
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
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device half *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = half(value);
    }
}

// SAM window/rel pos attention params and helpers
struct UocrSamWindowAttentionParams {
    uint windows;
    uint tokens_per_window;
    uint heads;
    uint head_dim;
    float scale;
    uint batch_size;
    uint reserved1;
    uint reserved2;
};

static inline uint uocr_sam_window_attention_batch_size(constant UocrSamWindowAttentionParams &params) {
    return params.batch_size == 0u ? 1u : params.batch_size;
}

static inline ulong uocr_sam_window_attention_index(constant UocrSamWindowAttentionParams &params,
                                                    uint batch,
                                                    uint window,
                                                    uint token,
                                                    uint head,
                                                    uint dim) {
    return (((((ulong(batch) * ulong(params.windows) + ulong(window)) * ulong(params.tokens_per_window) + ulong(token)) *
              ulong(params.heads) +
              ulong(head)) *
             ulong(params.head_dim)) +
            ulong(dim));
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
    uint batch_size;
    uint reserved1;
    uint reserved2;
};

static inline uint uocr_sam_rel_pos_attention_batch_size(constant UocrSamRelPosAttentionParams &params) {
    return params.batch_size == 0u ? 1u : params.batch_size;
}

static inline ulong uocr_sam_rel_pos_attention_index(constant UocrSamRelPosAttentionParams &params,
                                                     uint batch,
                                                     uint window,
                                                     uint token,
                                                     uint head,
                                                     uint dim) {
    return (((((ulong(batch) * ulong(params.windows) + ulong(window)) * ulong(params.tokens_per_window) + ulong(token)) *
              ulong(params.heads) +
              ulong(head)) *
             ulong(params.head_dim)) +
            ulong(dim));
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

// SAM attention project/residual
static inline float uocr_sam_attention_project_dot_f16(device const half *src,
                                                       device const half *weight,
                                                       constant UocrSamResidualParams &params,
                                                       uint row,
                                                       uint out_col,
                                                       uint tid,
                                                       uint ntg,
                                              uint simd_width,
                                                       threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
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
                                                           uint ntg [[threads_per_threadgroup]],
                                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_attention_project_dot_f16(src, weight, params, row, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        const uint dst_index = row * params.hidden_size + out_col;
        dst[dst_index] = half(value + float(bias[out_col]) + float(residual[dst_index]));
    }
}

// SAM flash attention: window, rel_pos, global attention, tiled variants
kernel void uocr_sam_window_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                       device const half *k_src [[buffer(1)]],
                                                       device const half *v_src [[buffer(2)]],
                                                       device half *dst [[buffer(3)]],
                                                       constant UocrSamWindowAttentionParams &params [[buffer(4)]],
                                                       uint3 tg [[threadgroup_position_in_grid]],
                                                       ushort lane [[thread_index_in_simdgroup]],
                                                       ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                       ushort simd_width [[threads_per_simdgroup]]) {
    uocr_sam_window_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup, simd_width);
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
                                                         ushort simdgroup_u16,
                                                         ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint batch_size = uocr_sam_rel_pos_attention_batch_size(params);
    const uint logical_window = tg.z;
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || params.windows == 0u || batch >= batch_size || window >= params.windows ||
        query_token >= params.tokens_per_window || head >= params.heads ||
        params.grid_width == 0u || params.grid_height == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    const uint query_y = query_token / params.grid_width;
    const uint query_x = query_token - query_y * params.grid_width;
    const uint target_h_length = 2u * params.grid_height - 1u;
    const uint target_w_length = 2u * params.grid_width - 1u;

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong q_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
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
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong k_index = uocr_sam_rel_pos_attention_index(params, batch, window, key_token, head, dim);
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
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong v_index = uocr_sam_rel_pos_attention_index(params, batch, window, key_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong dst_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
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
                                                        ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                        ushort simd_width [[threads_per_simdgroup]]) {
    uocr_sam_rel_pos_attention_flash_impl(q_src, k_src, v_src, rel_pos_h, rel_pos_w, dst, params, tg, lane, simdgroup, simd_width);
}
kernel void uocr_sam_rel_pos_logits_f16_to_f32(device const half *q_src [[buffer(0)]],
                                               device const half *rel_pos_h [[buffer(1)]],
                                               device const half *rel_pos_w [[buffer(2)]],
                                               device float *rel_h_logits [[buffer(3)]],
                                               device float *rel_w_logits [[buffer(4)]],
                                               constant UocrSamRelPosAttentionParams &params [[buffer(5)]],
                                               uint3 tg [[threadgroup_position_in_grid]],
                                               ushort lane_u16 [[thread_index_in_simdgroup]],
                                               ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint head = tg.x;
    const uint query_token = tg.y;
    const uint logical_window = tg.z;
    const uint batch_size = uocr_sam_rel_pos_attention_batch_size(params);
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    const uint target_h_length = 2u * params.grid_height - 1u;
    const uint target_w_length = 2u * params.grid_width - 1u;
    if (params.windows == 0u || batch >= batch_size || window >= params.windows || head >= params.heads ||
        query_token >= params.tokens_per_window || params.grid_width == 0u || params.grid_height == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height ||
        params.rel_pos_h_length != target_h_length || params.rel_pos_w_length != target_w_length ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    const uint query_y = query_token / params.grid_width;
    const uint query_x = query_token - query_y * params.grid_width;
    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        qv[i] = dim < params.head_dim ? float(q_src[uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim)]) : 0.0f;
    }

    const ulong base_h = (((ulong(logical_window) * ulong(params.heads) + ulong(head)) * ulong(params.tokens_per_window) +
                           ulong(query_token)) * ulong(params.grid_height));
    for (uint key_y = 0u; key_y < params.grid_height; ++key_y) {
        const uint rel_h_index = uint(int(query_y) - int(key_y) + int(params.grid_height) - 1);
        float local = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                local += qv[i] * float(rel_pos_h[ulong(rel_h_index) * ulong(params.head_dim) + ulong(dim)]);
            }
        }
        const float score = simd_sum(local);
        if (lane == 0u) {
            rel_h_logits[base_h + ulong(key_y)] = score;
        }
    }

    const ulong base_w = (((ulong(logical_window) * ulong(params.heads) + ulong(head)) * ulong(params.tokens_per_window) +
                           ulong(query_token)) * ulong(params.grid_width));
    for (uint key_x = 0u; key_x < params.grid_width; ++key_x) {
        const uint rel_w_index = uint(int(query_x) - int(key_x) + int(params.grid_width) - 1);
        float local = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                local += qv[i] * float(rel_pos_w[ulong(rel_w_index) * ulong(params.head_dim) + ulong(dim)]);
            }
        }
        const float score = simd_sum(local);
        if (lane == 0u) {
            rel_w_logits[base_w + ulong(key_x)] = score;
        }
    }
}

#define UOCR_SAM_REL_POS_TILE_KEYS 16u

kernel void uocr_sam_global_attention_pack_qkv_f16(device const half *q_src [[buffer(0)]],
                                                   device const half *k_src [[buffer(1)]],
                                                   device const half *v_src [[buffer(2)]],
                                                   device half *q_pack [[buffer(3)]],
                                                   device half *k_pack [[buffer(4)]],
                                                   device half *vt_pack [[buffer(5)]],
                                                   constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                   uint gid [[thread_position_in_grid]]) {
    const uint total = params.batch_size * params.windows * params.heads * params.tokens_per_window * params.head_dim;
    if (gid >= total || params.windows == 0u || params.tokens_per_window == 0u || params.heads == 0u || params.head_dim == 0u) {
        return;
    }
    uint rem = gid;
    const uint dim = rem % params.head_dim;
    rem /= params.head_dim;
    const uint token = rem % params.tokens_per_window;
    rem /= params.tokens_per_window;
    const uint head = rem % params.heads;
    rem /= params.heads;
    const uint window = rem % params.windows;
    const uint batch = rem / params.windows;
    const uint logical_head = (batch * params.windows + window) * params.heads + head;
    const ulong src_index = uocr_sam_rel_pos_attention_index(params, batch, window, token, head, dim);
    const ulong pack_index = (ulong(logical_head) * ulong(params.tokens_per_window) + ulong(token)) * ulong(params.head_dim) + ulong(dim);
    q_pack[pack_index] = q_src[src_index];
    k_pack[pack_index] = k_src[src_index];
    vt_pack[(ulong(logical_head) * ulong(params.head_dim) + ulong(dim)) * ulong(params.tokens_per_window) + ulong(token)] = v_src[src_index];
}

kernel void uocr_sam_global_attention_softmax_f32_to_f16(device const float *logits [[buffer(0)]],
                                                         device const float *rel_h_logits [[buffer(1)]],
                                                         device const float *rel_w_logits [[buffer(2)]],
                                                         device half *probs [[buffer(3)]],
                                                         constant UocrSamRelPosAttentionParams &params [[buffer(4)]],
                                                         threadgroup float *partials [[threadgroup(0)]],
                                                         uint3 tg [[threadgroup_position_in_grid]],
                                                         uint tid [[thread_index_in_threadgroup]],
                                                         uint3 ntg3 [[threads_per_threadgroup]],
                                                         ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint ntg = ntg3.x;
    const uint query = tg.x;
    const uint logical_head = tg.y;
    const uint logical_heads = params.batch_size * params.windows * params.heads;
    if (query >= params.tokens_per_window || logical_head >= logical_heads || params.grid_width == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height) {
        return;
    }
    const uint simd_width = uint(simd_width_u16);
    const ulong row_base = (ulong(logical_head) * ulong(params.tokens_per_window) + ulong(query)) * ulong(params.tokens_per_window);
    const ulong rel_base = ulong(logical_head) * ulong(params.tokens_per_window) + ulong(query);

    float local_max = UOCR_FLASH_NEG_INF;
    for (uint key = tid; key < params.tokens_per_window; key += ntg) {
        const uint key_y = key / params.grid_width;
        const uint key_x = key - key_y * params.grid_width;
        const float score = logits[row_base + ulong(key)] * params.scale +
                            rel_h_logits[rel_base * ulong(params.grid_height) + ulong(key_y)] +
                            rel_w_logits[rel_base * ulong(params.grid_width) + ulong(key_x)];
        local_max = max(local_max, score);
    }
    const float row_max = uocr_threadgroup_max(local_max, partials, tid, ntg, simd_width);

    float local_sum = 0.0f;
    for (uint key = tid; key < params.tokens_per_window; key += ntg) {
        const uint key_y = key / params.grid_width;
        const uint key_x = key - key_y * params.grid_width;
        const float score = logits[row_base + ulong(key)] * params.scale +
                            rel_h_logits[rel_base * ulong(params.grid_height) + ulong(key_y)] +
                            rel_w_logits[rel_base * ulong(params.grid_width) + ulong(key_x)];
        local_sum += exp(score - row_max);
    }
    const float row_sum = uocr_threadgroup_sum(local_sum, partials, tid, ntg, simd_width);
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

    for (uint key = tid; key < params.tokens_per_window; key += ntg) {
        const uint key_y = key / params.grid_width;
        const uint key_x = key - key_y * params.grid_width;
        const float score = logits[row_base + ulong(key)] * params.scale +
                            rel_h_logits[rel_base * ulong(params.grid_height) + ulong(key_y)] +
                            rel_w_logits[rel_base * ulong(params.grid_width) + ulong(key_x)];
        probs[row_base + ulong(key)] = half(exp(score - row_max) * inv_sum);
    }
}

kernel void uocr_sam_global_attention_unpack_f16(device const half *out_pack [[buffer(0)]],
                                                 device half *dst [[buffer(1)]],
                                                 constant UocrSamRelPosAttentionParams &params [[buffer(2)]],
                                                 uint gid [[thread_position_in_grid]]) {
    const uint total = params.batch_size * params.windows * params.heads * params.tokens_per_window * params.head_dim;
    if (gid >= total || params.windows == 0u || params.tokens_per_window == 0u || params.heads == 0u || params.head_dim == 0u) {
        return;
    }
    uint rem = gid;
    const uint dim = rem % params.head_dim;
    rem /= params.head_dim;
    const uint token = rem % params.tokens_per_window;
    rem /= params.tokens_per_window;
    const uint logical_head = rem;
    const uint head = logical_head % params.heads;
    const uint logical_window = logical_head / params.heads;
    const uint batch = logical_window / params.windows;
    const uint window = logical_window - batch * params.windows;
    const ulong pack_index = (ulong(logical_head) * ulong(params.tokens_per_window) + ulong(token)) * ulong(params.head_dim) + ulong(dim);
    dst[uocr_sam_rel_pos_attention_index(params, batch, window, token, head, dim)] = out_pack[pack_index];
}

kernel void uocr_sam_rel_pos_attention_tiled_precomputed_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                                     device const half *k_src [[buffer(1)]],
                                                                     device const half *v_src [[buffer(2)]],
                                                                     device const float *rel_h_logits [[buffer(3)]],
                                                                     device const float *rel_w_logits [[buffer(4)]],
                                                                     device half *dst [[buffer(5)]],
                                                                     constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                                     threadgroup half *k_tile [[threadgroup(0)]],
                                                                     threadgroup half *v_tile [[threadgroup(1)]],
                                                                     uint3 tg [[threadgroup_position_in_grid]],
                                                                     uint3 tid3 [[thread_position_in_threadgroup]],
                                                                     uint3 ntg3 [[threads_per_threadgroup]],
                                                                     ushort lane_u16 [[thread_index_in_simdgroup]],
                                                                     ushort simdgroup_u16 [[simdgroup_index_in_threadgroup]],
                                                                     ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint tid = tid3.x;
    const uint ntg = ntg3.x;
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint batch_size = uocr_sam_rel_pos_attention_batch_size(params);
    const uint logical_window = tg.z;
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    const uint target_h_length = 2u * params.grid_height - 1u;
    const uint target_w_length = 2u * params.grid_width - 1u;
    if (params.windows == 0u || batch >= batch_size || window >= params.windows || head >= params.heads ||
        params.grid_width == 0u || params.grid_height == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height ||
        params.rel_pos_h_length != target_h_length || params.rel_pos_w_length != target_w_length ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    const bool query_valid = query_in_block < UOCR_FLASH_Q_PER_TG && query_token < params.tokens_per_window;

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (query_valid && dim < params.head_dim) {
            qv[i] = float(q_src[uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim)]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    const ulong rel_base = (ulong(logical_window) * ulong(params.heads) + ulong(head)) * ulong(params.tokens_per_window) +
                           ulong(query_valid ? query_token : 0u);
    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint tile_start = 0u; tile_start < params.tokens_per_window; tile_start += UOCR_SAM_REL_POS_TILE_KEYS) {
        const uint tile_count = min(UOCR_SAM_REL_POS_TILE_KEYS, params.tokens_per_window - tile_start);
        const uint tile_values = tile_count * params.head_dim;
        for (uint tile_index = tid; tile_index < tile_values; tile_index += ntg) {
            const uint key_offset = tile_index / params.head_dim;
            const uint dim = tile_index - key_offset * params.head_dim;
            const uint key_token = tile_start + key_offset;
            const ulong kv_index = uocr_sam_rel_pos_attention_index(params, batch, window, key_token, head, dim);
            k_tile[tile_index] = k_src[kv_index];
            v_tile[tile_index] = v_src[kv_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key_offset = 0u; key_offset < tile_count; ++key_offset) {
            const uint key_token = tile_start + key_offset;
            const uint key_y = key_token / params.grid_width;
            const uint key_x = key_token - key_y * params.grid_width;
            float local_score = 0.0f;
            for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
                const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
                if (query_valid && dim < params.head_dim) {
                    local_score += qv[i] * float(k_tile[key_offset * params.head_dim + dim]) * params.scale;
                }
            }
            float score = query_valid ? simd_sum(local_score) : UOCR_FLASH_NEG_INF;
            if (query_valid) {
                score += rel_h_logits[rel_base * ulong(params.grid_height) + ulong(key_y)] +
                         rel_w_logits[rel_base * ulong(params.grid_width) + ulong(key_x)];
            }
            const float mnew = max(m, score);
            const float corr = exp(m - mnew);
            const float e = exp(score - mnew);
            for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
                const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
                if (query_valid && dim < params.head_dim) {
                    acc[i] = acc[i] * corr + e * float(v_tile[key_offset * params.head_dim + dim]);
                }
            }
            l = l * corr + e;
            m = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (query_valid && dim < params.head_dim) {
            const ulong dst_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
            dst[dst_index] = half(acc[i] * inv_l);
        }
    }
}

kernel void uocr_sam_rel_pos_attention_tiled_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                               device const half *k_src [[buffer(1)]],
                                                               device const half *v_src [[buffer(2)]],
                                                               device const half *rel_pos_h [[buffer(3)]],
                                                               device const half *rel_pos_w [[buffer(4)]],
                                                               device half *dst [[buffer(5)]],
                                                               constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                               threadgroup half *k_tile [[threadgroup(0)]],
                                                               threadgroup half *v_tile [[threadgroup(1)]],
                                                               uint3 tg [[threadgroup_position_in_grid]],
                                                               uint3 tid3 [[thread_position_in_threadgroup]],
                                                               uint3 ntg3 [[threads_per_threadgroup]],
                                                               ushort lane_u16 [[thread_index_in_simdgroup]],
                                                               ushort simdgroup_u16 [[simdgroup_index_in_threadgroup]],
                                                               ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint tid = tid3.x;
    const uint ntg = ntg3.x;
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint batch_size = uocr_sam_rel_pos_attention_batch_size(params);
    const uint logical_window = tg.z;
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    const uint target_h_length = 2u * params.grid_height - 1u;
    const uint target_w_length = 2u * params.grid_width - 1u;
    if (params.windows == 0u || batch >= batch_size || window >= params.windows || head >= params.heads ||
        params.grid_width == 0u || params.grid_height == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height ||
        params.rel_pos_h_length != target_h_length || params.rel_pos_w_length != target_w_length ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    const bool query_valid = query_in_block < UOCR_FLASH_Q_PER_TG && query_token < params.tokens_per_window;
    const uint query_y = query_valid ? query_token / params.grid_width : 0u;
    const uint query_x = query_valid ? query_token - query_y * params.grid_width : 0u;

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (query_valid && dim < params.head_dim) {
            const ulong q_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint tile_start = 0u; tile_start < params.tokens_per_window; tile_start += UOCR_SAM_REL_POS_TILE_KEYS) {
        const uint tile_count = min(UOCR_SAM_REL_POS_TILE_KEYS, params.tokens_per_window - tile_start);
        const uint tile_values = tile_count * params.head_dim;
        for (uint tile_index = tid; tile_index < tile_values; tile_index += ntg) {
            const uint key_offset = tile_index / params.head_dim;
            const uint dim = tile_index - key_offset * params.head_dim;
            const uint key_token = tile_start + key_offset;
            const ulong kv_index = uocr_sam_rel_pos_attention_index(params, batch, window, key_token, head, dim);
            k_tile[tile_index] = k_src[kv_index];
            v_tile[tile_index] = v_src[kv_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key_offset = 0u; key_offset < tile_count; ++key_offset) {
            const uint key_token = tile_start + key_offset;
            const uint key_y = key_token / params.grid_width;
            const uint key_x = key_token - key_y * params.grid_width;
            const uint rel_h_index = query_valid ? uint(int(query_y) - int(key_y) + int(params.grid_height) - 1) : 0u;
            const uint rel_w_index = query_valid ? uint(int(query_x) - int(key_x) + int(params.grid_width) - 1) : 0u;
            float local_score = 0.0f;
            for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
                const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
                if (query_valid && dim < params.head_dim) {
                    const float rh = float(rel_pos_h[ulong(rel_h_index) * ulong(params.head_dim) + ulong(dim)]);
                    const float rw = float(rel_pos_w[ulong(rel_w_index) * ulong(params.head_dim) + ulong(dim)]);
                    local_score += qv[i] * (float(k_tile[key_offset * params.head_dim + dim]) * params.scale + rh + rw);
                }
            }
            const float score = query_valid ? simd_sum(local_score) : UOCR_FLASH_NEG_INF;
            const float mnew = max(m, score);
            const float corr = exp(m - mnew);
            const float e = exp(score - mnew);
            for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
                const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
                if (query_valid && dim < params.head_dim) {
                    acc[i] = acc[i] * corr + e * float(v_tile[key_offset * params.head_dim + dim]);
                }
            }
            l = l * corr + e;
            m = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (query_valid && dim < params.head_dim) {
            const ulong dst_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
            dst[dst_index] = half(acc[i] * inv_l);
        }
    }
}

template <typename out_t>
