// lm_head_q8.metal - Fused Q8_0 LM-head argmax kernels
#include "common.metal"

struct UocrLmHeadArgmaxQ8Params {
    uint vocab_size;
    uint hidden_size;
    uint tile_tokens;
    uint lanes_per_token;
    uint group_size;
    uint groups_per_row;
    uint partial_count;
    uint reserved0;
};

static inline bool uocr_lm_head_q8_argmax_better(float score,
                                                 uint id,
                                                 float best_score,
                                                 uint best_id) {
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

static inline bool uocr_lm_head_q8_argmax_pair_better(float score,
                                                      uint id,
                                                      float best_score,
                                                      uint best_id) {
    if (id == 0xffffffffu) {
        return false;
    }
    return uocr_lm_head_q8_argmax_better(score, id, best_score, best_id);
}

static inline void uocr_lm_head_q8_threadgroup_argmax_pair(thread float &best_score,
                                                           thread uint &best_id,
                                                           threadgroup float *scores,
                                                           threadgroup uint *ids,
                                                           uint tid,
                                                           uint ntg,
                                                           uint simd_width) {
    const uint lane = uocr_simd_lane_from_tid(tid, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(tid, simd_width);
    const uint simdgroups = uocr_simdgroups_for_threadgroup(ntg, simd_width);

    for (uint offset = simd_width >> 1u; offset > 0u; offset >>= 1u) {
        const float other_score = simd_shuffle_down(best_score, ushort(offset));
        const uint other_id = simd_shuffle_down(best_id, ushort(offset));
        if (lane + offset < simd_width &&
            uocr_lm_head_q8_argmax_pair_better(other_score, other_id, best_score, best_id)) {
            best_score = other_score;
            best_id = other_id;
        }
    }

    if (lane == 0u) {
        scores[simdgroup] = best_score;
        ids[simdgroup] = best_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = simdgroups;
    while (active > 1u) {
        const uint upper = (active + 1u) >> 1u;
        if (tid < active - upper) {
            const float score = scores[tid + upper];
            const uint id = ids[tid + upper];
            if (uocr_lm_head_q8_argmax_pair_better(score, id, scores[tid], ids[tid])) {
                scores[tid] = score;
                ids[tid] = id;
            }
        }
        active = upper;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    best_score = scores[0];
    best_id = ids[0];
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_lm_head_argmax_q8_0_to_f16(
    device const half *hidden [[buffer(0)]],
    device const char *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device float *partial_scores_out [[buffer(3)]],
    device uint *partial_ids_out [[buffer(4)]],
    constant UocrLmHeadArgmaxQ8Params &params [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    threadgroup uint *partial_ids [[threadgroup(1)]],
    threadgroup half *hidden_tg [[threadgroup(2)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint vocab_size = uocr_fc_vocab_size_or(params.vocab_size);
    const uint tile_tokens = uocr_fc_lm_head_tile_tokens_or(params.tile_tokens);
    const uint lanes = uocr_fc_lm_head_lanes_per_token_or(params.lanes_per_token);
    const uint group_size = params.group_size;
    const uint groups_per_row = params.groups_per_row;

    for (uint k = tid; k < hidden_size; k += ntg) {
        hidden_tg[k] = hidden[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint local_token = tid / lanes;
    const uint lane = tid - local_token * lanes;
    const uint token_id = tile * tile_tokens + local_token;

    float sum = 0.0f;
    if (local_token < tile_tokens && token_id < vocab_size && group_size != 0u && groups_per_row != 0u) {
        const ulong weight_base = ulong(token_id) * ulong(hidden_size);
        const ulong scale_base = ulong(token_id) * ulong(groups_per_row);
        for (uint group = 0u; group < groups_per_row; ++group) {
            const uint group_start = group * group_size;
            const uint group_end = min(group_start + group_size, hidden_size);
            const float scale = float(qscale[scale_base + ulong(group)]);
            /* group_size and lane strides are multiples of 4: char4/half4 runs
             * stay aligned and inside the group. */
            for (uint k = group_start + lane * 4u; k + 3u < group_end; k += lanes * 4u) {
                const char4 w = *(device const char4 *)(qweight + weight_base + ulong(k));
                const half4 x = *(threadgroup const half4 *)(hidden_tg + k);
                sum += (float(x.x) * float(w.x) + float(x.y) * float(w.y) +
                        float(x.z) * float(w.z) + float(x.w) * float(w.w)) * scale;
            }
        }
    }

    for (uint stride = lanes >> 1u; stride > 0u; stride >>= 1u) {
        const float other = simd_shuffle_down(sum, ushort(stride));
        if (local_token < tile_tokens && lane < stride) {
            sum += other;
        }
    }

    if (local_token < tile_tokens && lane == 0u) {
        float score = sum;
        uint id = token_id;
        if (token_id >= vocab_size || isnan(score)) {
            id = 0xffffffffu;
            score = -INFINITY;
        }
        partials[local_token] = score;
        partial_ids[local_token] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best_score = tid < tile_tokens ? partials[tid] : -INFINITY;
    uint best_id = tid < tile_tokens ? partial_ids[tid] : 0xffffffffu;
    uocr_lm_head_q8_threadgroup_argmax_pair(best_score, best_id, partials, partial_ids, tid, ntg, simd_width);

    if (tid == 0u && tile < params.partial_count) {
        partial_scores_out[tile] = best_id == 0xffffffffu ? -INFINITY : best_score;
        partial_ids_out[tile] = best_id;
    }
}
