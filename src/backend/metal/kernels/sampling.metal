// sampling.metal - Sampling and argmax kernels
// Extracted from uocr_smoke.metal
//
#include "common.metal"
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

static inline void uocr_threadgroup_argmax_pair(thread float &best_score,
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
        if (lane + offset < simd_width && uocr_argmax_pair_better(other_score, other_id, best_score, best_id)) {
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
            if (uocr_argmax_pair_better(score, id, scores[tid], ids[tid])) {
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

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_lm_head_argmax_f16(device const half *hidden [[buffer(0)]],
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
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint vocab_size = uocr_fc_vocab_size_or(params.vocab_size);
    const uint tile_tokens = uocr_fc_lm_head_tile_tokens_or(params.tile_tokens);
    const uint lanes = uocr_fc_lm_head_lanes_per_token_or(params.lanes_per_token);
    for (uint k = tid; k < hidden_size; k += ntg) {
        hidden_tg[k] = hidden[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint local_token = tid / lanes;
    const uint lane = tid - local_token * lanes;
    const uint token_id = tile * tile_tokens + local_token;

    float sum = 0.0f;
    if (local_token < tile_tokens && token_id < vocab_size) {
        const ulong weight_base = ulong(token_id) * ulong(hidden_size);
        /* Vectorized half4 dot: process 4 channels per iteration */
        for (uint k = lane * 4u; k + 3u < hidden_size; k += lanes * 4u) {
            half4 hv = half4(*reinterpret_cast<threadgroup const packed_half4 *>(hidden_tg + k));
            half4 wv = half4(*reinterpret_cast<device const packed_half4 *>(weight + weight_base + ulong(k)));
            sum += dot(float4(hv), float4(wv));
        }
        /* Remainder: scalar tail for leftover channels */
        const uint groups = hidden_size / (lanes * 4u);
        for (uint k = groups * (lanes * 4u) + lane; k < hidden_size; k += lanes) {
            sum += float(hidden_tg[k]) * float(weight[weight_base + ulong(k)]);
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
        if (token_id >= vocab_size) {
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

    float best_score = tid < tile_tokens ? partials[tid] : -INFINITY;
    uint best_id = tid < tile_tokens ? partial_ids[tid] : 0xffffffffu;
    uocr_threadgroup_argmax_pair(best_score, best_id, partials, partial_ids, tid, ntg, simd_width);

    if (tid == 0u && tile < params.partial_count) {
        partial_scores_out[tile] = best_id == 0xffffffffu ? -INFINITY : best_score;
        partial_ids_out[tile] = best_id;
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
                                  uint ntg [[threads_per_threadgroup]],
                                  uint simd_width [[threads_per_simdgroup]]) {
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
    uocr_threadgroup_argmax_pair(best_score, best_id, scores, ids, tid, ntg, simd_width);

    if (tid == 0u) {
        const uint id = best_id == 0xffffffffu ? 0u : best_id;
        token_id_out[0] = int(id);
        score_out[0] = best_id == 0xffffffffu ? -INFINITY : best_score;
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
                            uint ntg [[threads_per_threadgroup]],
                            uint simd_width [[threads_per_simdgroup]]) {
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
    uocr_threadgroup_argmax_pair(best_score, best_id, partial_scores, partial_ids, tid, ntg, simd_width);

    if (tid == 0) {
        const uint id = best_id == 0xffffffffu ? 0u : best_id;
        token_ids[row] = id;
        scores[row] = best_id == 0xffffffffu ? -INFINITY : best_score;
    }
}

struct UocrArgmaxBannedParams {
    uint vocab_size;
    uint has_banned;
    uint reserved0;
    uint reserved1;
};

/**
 * Single-row argmax over f32 logits with optional ban_flags.
 * Outputs one token id and score via the last threadgroup reduction winner.
 */
kernel void uocr_argmax_f32_banned(device const float *logits [[buffer(0)]],
                                   device const uchar *ban_flags [[buffer(1)]],
                                   device int *token_id_out [[buffer(2)]],
                                   device float *score_out [[buffer(3)]],
                                   constant UocrArgmaxBannedParams &params [[buffer(4)]],
                                   threadgroup float *scores [[threadgroup(0)]],
                                   threadgroup uint *ids [[threadgroup(1)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]],
                                   uint simd_width [[threads_per_simdgroup]]) {
    float best_score = -INFINITY;
    uint best_id = 0xffffffffu;

    for (uint col = tid; col < params.vocab_size; col += ntg) {
        float score = logits[col];
        if (params.has_banned != 0u && ban_flags[col] != uchar(0u)) {
            score = -INFINITY;
        }
        if (uocr_argmax_better(score, col, best_score, best_id)) {
            best_score = score;
            best_id = col;
        }
    }

    uocr_threadgroup_argmax_pair(best_score, best_id, scores, ids, tid, ntg, simd_width);

    if (tid == 0u) {
        token_id_out[0] = int(best_id == 0xffffffffu ? 0u : best_id);
        score_out[0] = best_id == 0xffffffffu ? -INFINITY : best_score;
    }
}
