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

static inline float uocr_prefill_attention_score(device const half *q_src,
                                                 device const half *k_src,
                                                 constant UocrPrefillAttentionParams &params,
                                                 uint query_token,
                                                 uint key_token,
                                                 uint head) {
    const ulong q_base = (ulong(query_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim);
    const ulong k_base = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim);
    float score = 0.0f;
    for (uint dim = 0u; dim < params.head_dim; ++dim) {
        score += float(q_src[q_base + ulong(dim)]) * float(k_src[k_base + ulong(dim)]);
    }
    return score * params.scale;
}

static inline void uocr_prefill_reduce_max(threadgroup float *partials,
                                           uint tid,
                                           uint ntg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] = max(partials[tid], partials[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

static inline void uocr_prefill_reduce_sum(threadgroup float *partials,
                                           uint tid,
                                           uint ntg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void uocr_prefill_attention_f16_to_f16(device const half *q_src [[buffer(0)]],
                                              device const half *k_src [[buffer(1)]],
                                              device const half *v_src [[buffer(2)]],
                                              device half *dst [[buffer(3)]],
                                              constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                              threadgroup float *scratch [[threadgroup(0)]],
                                              uint group_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]]) {
    const uint query_token = group_index / params.heads;
    const uint head = group_index - query_token * params.heads;
    if (query_token >= params.n_tokens || head >= params.heads) {
        return;
    }

    threadgroup float *scores = scratch;
    threadgroup float *partials = scratch + params.n_tokens;
    float local_max = -3.4028234663852886e38f;
    for (uint key_token = tid; key_token <= query_token; key_token += ntg) {
        const float score = uocr_prefill_attention_score(q_src, k_src, params, query_token, key_token, head);
        scores[key_token] = score;
        local_max = max(local_max, score);
    }
    partials[tid] = local_max;
    uocr_prefill_reduce_max(partials, tid, ntg);
    const float max_score = partials[0];

    float local_denominator = 0.0f;
    for (uint key_token = tid; key_token <= query_token; key_token += ntg) {
        const float unnormalized = exp(scores[key_token] - max_score);
        scores[key_token] = unnormalized;
        local_denominator += unnormalized;
    }
    partials[tid] = local_denominator;
    uocr_prefill_reduce_sum(partials, tid, ntg);
    const float denominator = partials[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < params.head_dim; dim += ntg) {
        float value = 0.0f;
        for (uint key_token = 0u; key_token <= query_token; ++key_token) {
            const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
            value += (scores[key_token] / denominator) * float(v_src[v_index]);
        }
        const ulong dst_index = (ulong(query_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
        dst[dst_index] = half(value);
    }
}

kernel void uocr_prefill_attention_f16_to_f32(device const half *q_src [[buffer(0)]],
                                              device const half *k_src [[buffer(1)]],
                                              device const half *v_src [[buffer(2)]],
                                              device float *dst [[buffer(3)]],
                                              constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                              threadgroup float *scratch [[threadgroup(0)]],
                                              uint group_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]]) {
    const uint query_token = group_index / params.heads;
    const uint head = group_index - query_token * params.heads;
    if (query_token >= params.n_tokens || head >= params.heads) {
        return;
    }

    threadgroup float *scores = scratch;
    threadgroup float *partials = scratch + params.n_tokens;
    float local_max = -3.4028234663852886e38f;
    for (uint key_token = tid; key_token <= query_token; key_token += ntg) {
        const float score = uocr_prefill_attention_score(q_src, k_src, params, query_token, key_token, head);
        scores[key_token] = score;
        local_max = max(local_max, score);
    }
    partials[tid] = local_max;
    uocr_prefill_reduce_max(partials, tid, ntg);
    const float max_score = partials[0];

    float local_denominator = 0.0f;
    for (uint key_token = tid; key_token <= query_token; key_token += ntg) {
        const float unnormalized = exp(scores[key_token] - max_score);
        scores[key_token] = unnormalized;
        local_denominator += unnormalized;
    }
    partials[tid] = local_denominator;
    uocr_prefill_reduce_sum(partials, tid, ntg);
    const float denominator = partials[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < params.head_dim; dim += ntg) {
        float value = 0.0f;
        for (uint key_token = 0u; key_token <= query_token; ++key_token) {
            const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
            value += (scores[key_token] / denominator) * float(v_src[v_index]);
        }
        const ulong dst_index = (ulong(query_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(dim);
        dst[dst_index] = value;
    }
}

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

kernel void uocr_decode_attention_f16_to_f16(device const half *q_src [[buffer(0)]],
                                             device const half *k_cache [[buffer(1)]],
                                             device const half *v_cache [[buffer(2)]],
                                             device half *dst [[buffer(3)]],
                                             constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                             threadgroup float *partials [[threadgroup(0)]],
                                             uint head [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint ntg [[threads_per_threadgroup]]) {
    if (head >= params.heads || tid >= params.head_dim || ntg < params.head_dim || params.attention_length == 0u) {
        return;
    }

    const ulong q_index = ulong(head) * ulong(params.head_dim) + ulong(tid);
    const float q = float(q_src[q_index]);
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    for (uint attention_index = 0u; attention_index < params.attention_length; ++attention_index) {
        const uint cache_token = uocr_decode_attention_cache_token(params, attention_index);
        if (cache_token >= params.cache_token_capacity) {
            partials[tid] = 0.0f;
        } else {
            const ulong k_index = uocr_decode_attention_cache_index(params, cache_token, head, tid);
            partials[tid] = q * float(k_cache[k_index]);
        }
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
        const ulong v_index = uocr_decode_attention_cache_index(params, cache_token, head, tid);
        acc = acc * corr + e * float(v_cache[v_index]);
        l = l * corr + e;
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dst[q_index] = half(l > 0.0f ? acc / l : 0.0f);
}

kernel void uocr_decode_attention_f16_to_f32(device const half *q_src [[buffer(0)]],
                                             device const half *k_cache [[buffer(1)]],
                                             device const half *v_cache [[buffer(2)]],
                                             device float *dst [[buffer(3)]],
                                             constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                             threadgroup float *partials [[threadgroup(0)]],
                                             uint head [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint ntg [[threads_per_threadgroup]]) {
    if (head >= params.heads || tid >= params.head_dim || ntg < params.head_dim || params.attention_length == 0u) {
        return;
    }

    const ulong q_index = ulong(head) * ulong(params.head_dim) + ulong(tid);
    const float q = float(q_src[q_index]);
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    for (uint attention_index = 0u; attention_index < params.attention_length; ++attention_index) {
        const uint cache_token = uocr_decode_attention_cache_token(params, attention_index);
        if (cache_token >= params.cache_token_capacity) {
            partials[tid] = 0.0f;
        } else {
            const ulong k_index = uocr_decode_attention_cache_index(params, cache_token, head, tid);
            partials[tid] = q * float(k_cache[k_index]);
        }
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
        const ulong v_index = uocr_decode_attention_cache_index(params, cache_token, head, tid);
        acc = acc * corr + e * float(v_cache[v_index]);
        l = l * corr + e;
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dst[q_index] = l > 0.0f ? acc / l : 0.0f;
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
