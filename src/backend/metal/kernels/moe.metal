// moe.metal - MoE routing, selected, interleaved, and combine kernels
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrMoeRouterParams {
    uint n_tokens;
    uint hidden_size;
    uint experts;
    uint top_k;
};

static inline bool uocr_moe_router_topk_bits_better(uint score_bits,
                                                     uint inverse_expert,
                                                     uint best_score_bits,
                                                     uint best_inverse_expert) {
    return score_bits > best_score_bits ||
           (score_bits == best_score_bits && inverse_expert > best_inverse_expert);
}

static inline float uocr_moe_router_dot_f16(device const half *src,
                                            device const half *weight,
                                            constant UocrMoeRouterParams &params,
                                            uint token,
                                            uint expert,
                                            uint tid,
                                            uint ntg,
                                            uint simd_width,
                                            threadgroup float *partials) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    float sum = 0.0f;
    const uint src_base = token * hidden_size;
    const uint weight_base = expert * hidden_size;
    for (uint k = tid; k < hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

static inline float uocr_moe_router_group256_sum(float value,
                                                 threadgroup float *partials,
                                                 uint group,
                                                 uint local_tid,
                                                 uint simd_width) {
    const uint threads_per_expert = 256u;
    const uint lane = uocr_simd_lane_from_tid(local_tid, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(local_tid, simd_width);
    const uint simdgroups = uocr_simdgroups_for_threadgroup(threads_per_expert, simd_width);
    threadgroup float *group_partials = partials + group * simdgroups;
    const float simd_total = simd_sum(value);
    if (lane == 0u) {
        group_partials[simdgroup] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = simdgroups;
    while (active > 1u) {
        const uint upper = (active + 1u) >> 1u;
        if (local_tid < active - upper) {
            group_partials[local_tid] += group_partials[local_tid + upper];
        }
        active = upper;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return group_partials[0];
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_moe_router_logits_f16_to_f32(device const half *src [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device float *logits [[buffer(2)]],
                                             constant UocrMoeRouterParams &params [[buffer(3)]],
                                             threadgroup float *partials [[threadgroup(0)]],
                                             uint output_index [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint ntg [[threads_per_threadgroup]],
                                             uint simd_width [[threads_per_simdgroup]]) {
    const uint experts = uocr_fc_moe_experts_or(params.experts);
    const uint token = output_index / experts;
    const uint expert = output_index - token * experts;
    if (token >= params.n_tokens || expert >= experts) {
        return;
    }

    const float value = uocr_moe_router_dot_f16(src, weight, params, token, expert, tid, ntg, simd_width, partials);
    if (tid == 0) {
        logits[token * experts + expert] = value;
    }
}

// Decode-only router fusion for one hidden row.  Each 256-thread partition
// computes one expert dot-product with the same per-expert accumulation shape
// as uocr_moe_router_logits_f16_to_f32, then the first SIMD-group performs the
// existing softmax/top-6 contract from threadgroup-resident logits.  The
// default host path uses this only for the fixed OCR decode shape.
[[max_total_threads_per_threadgroup(1024)]] kernel void uocr_moe_router_decode_fused_f16(device const half *src [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device uint *top_expert_ids [[buffer(2)]],
                                             device float *top_weights [[buffer(3)]],
                                             constant UocrMoeRouterParams &params [[buffer(4)]],
                                             threadgroup half *hidden [[threadgroup(0)]],
                                             threadgroup float *scratch [[threadgroup(1)]],
                                             uint token [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint ntg [[threads_per_threadgroup]],
                                             uint simd_width [[threads_per_simdgroup]]) {
    if (token >= params.n_tokens) {
        return;
    }
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint experts = uocr_fc_moe_experts_or(params.experts);
    const uint top_k = uocr_fc_moe_top_k_or(params.top_k);
    const uint threads_per_expert = 256u;
    if (hidden_size != 1280u || experts != 64u || top_k != 6u || ntg < threads_per_expert) {
        return;
    }

    const uint src_base = token * hidden_size;
    for (uint k = tid; k < hidden_size; k += ntg) {
        hidden[k] = src[src_base + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float *logits = scratch;
    threadgroup float *partials = scratch + experts;
    const uint experts_per_batch = ntg / threads_per_expert;
    const uint expert_group = tid / threads_per_expert;
    const uint local_tid = tid - expert_group * threads_per_expert;
    for (uint expert_base = 0u; expert_base < experts; expert_base += experts_per_batch) {
        const uint expert = expert_base + expert_group;
        float sum = 0.0f;
        if (expert < experts) {
            const uint weight_base = expert * hidden_size;
            for (uint k = local_tid; k < hidden_size; k += threads_per_expert) {
                sum += float(hidden[k]) * float(weight[weight_base + k]);
            }
        }
        const float dot = uocr_moe_router_group256_sum(sum, partials, expert_group, local_tid, simd_width);
        if (local_tid == 0u && expert < experts) {
            logits[expert] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint lane = uocr_simd_lane_from_tid(tid, simd_width);
    if (tid < simd_width) {
        const bool active_lane = lane < 16u;
        const uint expert0 = lane;
        const uint expert1 = lane + 16u;
        const uint expert2 = lane + 32u;
        const uint expert3 = lane + 48u;
        const float logit0 = active_lane ? logits[expert0] : -INFINITY;
        const float logit1 = active_lane ? logits[expert1] : -INFINITY;
        const float logit2 = active_lane ? logits[expert2] : -INFINITY;
        const float logit3 = active_lane ? logits[expert3] : -INFINITY;
        const float local_max = max(max(logit0, logit1), max(logit2, logit3));
        const float max_logit = simd_max(local_max);
        const float exp0 = active_lane ? exp(logit0 - max_logit) : 0.0f;
        const float exp1 = active_lane ? exp(logit1 - max_logit) : 0.0f;
        const float exp2 = active_lane ? exp(logit2 - max_logit) : 0.0f;
        const float exp3 = active_lane ? exp(logit3 - max_logit) : 0.0f;
        const float total_sum = simd_sum(exp0 + exp1 + exp2 + exp3);
        const float inv_sum = 1.0f / total_sum;
        const float prob0 = exp0 * inv_sum;
        const float prob1 = exp1 * inv_sum;
        const float prob2 = exp2 * inv_sum;
        const float prob3 = exp3 * inv_sum;

        const uint score_bits0 = active_lane ? as_type<uint>(prob0) : 0u;
        const uint score_bits1 = active_lane ? as_type<uint>(prob1) : 0u;
        const uint score_bits2 = active_lane ? as_type<uint>(prob2) : 0u;
        const uint score_bits3 = active_lane ? as_type<uint>(prob3) : 0u;
        const uint inverse0 = active_lane ? (0xffffffffu - expert0) : 0u;
        const uint inverse1 = active_lane ? (0xffffffffu - expert1) : 0u;
        const uint inverse2 = active_lane ? (0xffffffffu - expert2) : 0u;
        const uint inverse3 = active_lane ? (0xffffffffu - expert3) : 0u;
        uint selected0 = 0xffffffffu;
        uint selected1 = 0xffffffffu;
        uint selected2 = 0xffffffffu;
        uint selected3 = 0xffffffffu;
        uint selected4 = 0xffffffffu;
        uint selected5 = 0xffffffffu;
        for (uint rank = 0u; rank < 6u; ++rank) {
            uint local_score_bits = 0u;
            uint local_inverse_expert = 0u;
            if (expert0 != selected0 && expert0 != selected1 && expert0 != selected2 &&
                expert0 != selected3 && expert0 != selected4 && expert0 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits0, inverse0, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits0;
                local_inverse_expert = inverse0;
            }
            if (expert1 != selected0 && expert1 != selected1 && expert1 != selected2 &&
                expert1 != selected3 && expert1 != selected4 && expert1 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits1, inverse1, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits1;
                local_inverse_expert = inverse1;
            }
            if (expert2 != selected0 && expert2 != selected1 && expert2 != selected2 &&
                expert2 != selected3 && expert2 != selected4 && expert2 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits2, inverse2, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits2;
                local_inverse_expert = inverse2;
            }
            if (expert3 != selected0 && expert3 != selected1 && expert3 != selected2 &&
                expert3 != selected3 && expert3 != selected4 && expert3 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits3, inverse3, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits3;
                local_inverse_expert = inverse3;
            }
            const uint simd_score_bits = simd_max(local_score_bits);
            const uint simd_inverse_expert = simd_max(local_score_bits == simd_score_bits ? local_inverse_expert : 0u);
            const uint best_expert = 0xffffffffu - simd_inverse_expert;
            if (rank == 0u) {
                selected0 = best_expert;
            } else if (rank == 1u) {
                selected1 = best_expert;
            } else if (rank == 2u) {
                selected2 = best_expert;
            } else if (rank == 3u) {
                selected3 = best_expert;
            } else if (rank == 4u) {
                selected4 = best_expert;
            } else {
                selected5 = best_expert;
            }
            if (lane == 0u) {
                top_expert_ids[token * 6u + rank] = best_expert;
                top_weights[token * 6u + rank] = as_type<float>(simd_score_bits);
            }
        }
    }
}

// Unlimited-OCR routing contract: softmax(hidden @ router_weight.T) over all
// 64 experts, greedy top-6, raw selected probabilities, no top-k
// renormalization/scaling, and no DS4 softplus/sqrt/bias transforms.
[[max_total_threads_per_threadgroup(64)]] kernel void uocr_moe_router_softmax_topk_f32(device const float *logits [[buffer(0)]],
                                             device float *probs [[buffer(1)]],
                                             device uint *top_expert_ids [[buffer(2)]],
                                             device float *top_weights [[buffer(3)]],
                                             constant UocrMoeRouterParams &params [[buffer(4)]],
                                             threadgroup float *scratch [[threadgroup(0)]],
                                             uint token [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint ntg [[threads_per_threadgroup]],
                                             uint simd_width [[threads_per_simdgroup]]) {
    if (token >= params.n_tokens) {
        return;
    }

    const uint experts = uocr_fc_moe_experts_or(params.experts);
    const uint top_k = uocr_fc_moe_top_k_or(params.top_k);
    const uint row_base = token * experts;
    const uint lane = uocr_simd_lane_from_tid(tid, simd_width);

    if (experts == 64u && top_k == 6u && ntg == 16u && ntg <= simd_width) {
        const uint expert0 = tid;
        const uint expert1 = tid + ntg;
        const uint expert2 = tid + 2u * ntg;
        const uint expert3 = tid + 3u * ntg;
        const float logit0 = logits[row_base + expert0];
        const float logit1 = logits[row_base + expert1];
        const float logit2 = logits[row_base + expert2];
        const float logit3 = logits[row_base + expert3];
        const float local_max = max(max(logit0, logit1), max(logit2, logit3));
        const float max_logit = simd_max(local_max);
        const float exp0 = exp(logit0 - max_logit);
        const float exp1 = exp(logit1 - max_logit);
        const float exp2 = exp(logit2 - max_logit);
        const float exp3 = exp(logit3 - max_logit);
        const float total_sum = simd_sum(exp0 + exp1 + exp2 + exp3);
        const float inv_sum = 1.0f / total_sum;
        const float prob0 = exp0 * inv_sum;
        const float prob1 = exp1 * inv_sum;
        const float prob2 = exp2 * inv_sum;
        const float prob3 = exp3 * inv_sum;
        probs[row_base + expert0] = prob0;
        probs[row_base + expert1] = prob1;
        probs[row_base + expert2] = prob2;
        probs[row_base + expert3] = prob3;

        const uint score_bits0 = as_type<uint>(prob0);
        const uint score_bits1 = as_type<uint>(prob1);
        const uint score_bits2 = as_type<uint>(prob2);
        const uint score_bits3 = as_type<uint>(prob3);
        const uint inverse0 = 0xffffffffu - expert0;
        const uint inverse1 = 0xffffffffu - expert1;
        const uint inverse2 = 0xffffffffu - expert2;
        const uint inverse3 = 0xffffffffu - expert3;
        uint selected0 = 0xffffffffu;
        uint selected1 = 0xffffffffu;
        uint selected2 = 0xffffffffu;
        uint selected3 = 0xffffffffu;
        uint selected4 = 0xffffffffu;
        uint selected5 = 0xffffffffu;
        for (uint rank = 0u; rank < 6u; ++rank) {
            uint local_score_bits = 0u;
            uint local_inverse_expert = 0u;
            if (expert0 != selected0 && expert0 != selected1 && expert0 != selected2 &&
                expert0 != selected3 && expert0 != selected4 && expert0 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits0, inverse0, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits0;
                local_inverse_expert = inverse0;
            }
            if (expert1 != selected0 && expert1 != selected1 && expert1 != selected2 &&
                expert1 != selected3 && expert1 != selected4 && expert1 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits1, inverse1, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits1;
                local_inverse_expert = inverse1;
            }
            if (expert2 != selected0 && expert2 != selected1 && expert2 != selected2 &&
                expert2 != selected3 && expert2 != selected4 && expert2 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits2, inverse2, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits2;
                local_inverse_expert = inverse2;
            }
            if (expert3 != selected0 && expert3 != selected1 && expert3 != selected2 &&
                expert3 != selected3 && expert3 != selected4 && expert3 != selected5 &&
                uocr_moe_router_topk_bits_better(score_bits3, inverse3, local_score_bits, local_inverse_expert)) {
                local_score_bits = score_bits3;
                local_inverse_expert = inverse3;
            }
            const uint simd_score_bits = simd_max(local_score_bits);
            const uint simd_inverse_expert = simd_max(local_score_bits == simd_score_bits ? local_inverse_expert : 0u);
            const uint best_expert = 0xffffffffu - simd_inverse_expert;
            if (rank == 0u) {
                selected0 = best_expert;
            } else if (rank == 1u) {
                selected1 = best_expert;
            } else if (rank == 2u) {
                selected2 = best_expert;
            } else if (rank == 3u) {
                selected3 = best_expert;
            } else if (rank == 4u) {
                selected4 = best_expert;
            } else {
                selected5 = best_expert;
            }
            if (lane == 0u) {
                top_expert_ids[token * 6u + rank] = best_expert;
                top_weights[token * 6u + rank] = as_type<float>(simd_score_bits);
            }
        }
        return;
    }

    threadgroup float *scores = scratch;
    threadgroup float *partials = scratch + experts;
    threadgroup uint *indices = (threadgroup uint *)(scratch + experts + ntg);

    float local_max = -INFINITY;
    for (uint expert = tid; expert < experts; expert += ntg) {
        const float value = logits[row_base + expert];
        scores[expert] = value;
        local_max = max(local_max, value);
    }
    const float max_logit = uocr_threadgroup_max(local_max, partials, tid, ntg, simd_width);
    float local_sum = 0.0f;
    for (uint expert = tid; expert < experts; expert += ntg) {
        const float value = exp(scores[expert] - max_logit);
        scores[expert] = value;
        local_sum += value;
    }
    const float total_sum = uocr_threadgroup_sum(local_sum, partials, tid, ntg, simd_width);
    const float inv_sum = 1.0f / total_sum;
    uint owned_score_bits = 0u;
    for (uint expert = tid; expert < experts; expert += ntg) {
        const float prob = scores[expert] * inv_sum;
        scores[expert] = prob;
        probs[row_base + expert] = prob;
        indices[expert] = expert;
        if (expert == tid) {
            owned_score_bits = as_type<uint>(prob);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // OCR only needs top-6 from 64 experts.  Select each rank with a
    // deterministic repeated argmax instead of sorting the whole row.  The
    // tuple key is (score_bits, inverse_expert_id): for nonnegative softmax
    // probabilities, unsigned score bits preserve higher-score priority and
    // exact ties select the lower expert id.  Prefer a one-SIMD greedy top-6
    // path: each lane owns multiple experts, SIMD reductions find each rank,
    // and the selected ids stay in registers.  This replaces the previous
    // full-row bitonic sort and avoids top-k threadgroup barriers.
    if (ntg == experts && top_k <= 6u) {
        const uint simdgroups = uocr_simdgroups_for_threadgroup(ntg, simd_width);
        const uint simdgroup = uocr_simdgroup_from_tid(tid, simd_width);
        threadgroup uint *selected_ids = indices;
        threadgroup uint *group_score_bits = indices + 6u;
        threadgroup uint *group_inverse_experts = group_score_bits + simdgroups;
        const uint inverse_expert = 0xffffffffu - tid;
        for (uint rank = 0u; rank < top_k; ++rank) {
            bool already_selected = false;
            for (uint prev = 0u; prev < rank; ++prev) {
                if (selected_ids[prev] == tid) {
                    already_selected = true;
                }
            }
            const uint local_score_bits = already_selected ? 0u : owned_score_bits;
            const uint local_inverse_expert = already_selected ? 0u : inverse_expert;
            const uint simd_score_bits = simd_max(local_score_bits);
            const uint simd_inverse_expert = simd_max(local_score_bits == simd_score_bits ? local_inverse_expert : 0u);
            if (lane == 0u) {
                group_score_bits[simdgroup] = simd_score_bits;
                group_inverse_experts[simdgroup] = simd_inverse_expert;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0u) {
                uint best_score_bits = 0u;
                uint best_inverse_expert = 0u;
                for (uint group = 0u; group < simdgroups; ++group) {
                    const uint score_bits = group_score_bits[group];
                    const uint group_inverse_expert = group_inverse_experts[group];
                    if (uocr_moe_router_topk_bits_better(score_bits,
                                                         group_inverse_expert,
                                                         best_score_bits,
                                                         best_inverse_expert)) {
                        best_score_bits = score_bits;
                        best_inverse_expert = group_inverse_expert;
                    }
                }
                const uint best_expert = 0xffffffffu - best_inverse_expert;
                selected_ids[rank] = best_expert;
                top_expert_ids[token * top_k + rank] = best_expert;
                top_weights[token * top_k + rank] = as_type<float>(best_score_bits);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    if (ntg <= simd_width && top_k <= 6u) {
        uint selected0 = 0xffffffffu;
        uint selected1 = 0xffffffffu;
        uint selected2 = 0xffffffffu;
        uint selected3 = 0xffffffffu;
        uint selected4 = 0xffffffffu;
        uint selected5 = 0xffffffffu;
        for (uint rank = 0u; rank < top_k; ++rank) {
            uint local_score_bits = 0u;
            uint local_inverse_expert = 0u;
            for (uint expert = tid; expert < experts; expert += ntg) {
                const bool already_selected =
                    (rank > 0u && expert == selected0) ||
                    (rank > 1u && expert == selected1) ||
                    (rank > 2u && expert == selected2) ||
                    (rank > 3u && expert == selected3) ||
                    (rank > 4u && expert == selected4) ||
                    (rank > 5u && expert == selected5);
                if (!already_selected) {
                    const uint score_bits = as_type<uint>(scores[expert]);
                    const uint inverse_expert = 0xffffffffu - expert;
                    if (uocr_moe_router_topk_bits_better(score_bits,
                                                         inverse_expert,
                                                         local_score_bits,
                                                         local_inverse_expert)) {
                        local_score_bits = score_bits;
                        local_inverse_expert = inverse_expert;
                    }
                }
            }
            const uint simd_score_bits = simd_max(local_score_bits);
            const uint simd_inverse_expert = simd_max(local_score_bits == simd_score_bits ? local_inverse_expert : 0u);
            const uint best_expert = 0xffffffffu - simd_inverse_expert;
            if (rank == 0u) {
                selected0 = best_expert;
            } else if (rank == 1u) {
                selected1 = best_expert;
            } else if (rank == 2u) {
                selected2 = best_expert;
            } else if (rank == 3u) {
                selected3 = best_expert;
            } else if (rank == 4u) {
                selected4 = best_expert;
            } else {
                selected5 = best_expert;
            }
            if (lane == 0u) {
                top_expert_ids[token * top_k + rank] = best_expert;
                top_weights[token * top_k + rank] = scores[best_expert];
            }
        }
        return;
    }

    const uint simdgroups = uocr_simdgroups_for_threadgroup(ntg, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(tid, simd_width);
    threadgroup uint *selected_ids = indices;
    const uint selected_words = (top_k + 1u) & ~1u;
    threadgroup uint *group_score_bits = indices + selected_words;
    threadgroup uint *group_inverse_experts = group_score_bits + simdgroups;
    for (uint rank = 0u; rank < top_k; ++rank) {
        uint local_score_bits = 0u;
        uint local_inverse_expert = 0u;
        for (uint expert = tid; expert < experts; expert += ntg) {
            bool already_selected = false;
            for (uint prev = 0u; prev < rank; ++prev) {
                if (selected_ids[prev] == expert) {
                    already_selected = true;
                }
            }
            if (!already_selected) {
                const uint score_bits = as_type<uint>(scores[expert]);
                const uint inverse_expert = 0xffffffffu - expert;
                if (uocr_moe_router_topk_bits_better(score_bits,
                                                     inverse_expert,
                                                     local_score_bits,
                                                     local_inverse_expert)) {
                    local_score_bits = score_bits;
                    local_inverse_expert = inverse_expert;
                }
            }
        }

        const uint simd_score_bits = simd_max(local_score_bits);
        const uint simd_inverse_expert = simd_max(local_score_bits == simd_score_bits ? local_inverse_expert : 0u);
        if (lane == 0u) {
            group_score_bits[simdgroup] = simd_score_bits;
            group_inverse_experts[simdgroup] = simd_inverse_expert;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint best_score_bits = 0u;
            uint best_inverse_expert = 0u;
            for (uint group = 0u; group < simdgroups; ++group) {
                const uint score_bits = group_score_bits[group];
                const uint inverse_expert = group_inverse_experts[group];
                if (uocr_moe_router_topk_bits_better(score_bits,
                                                     inverse_expert,
                                                     best_score_bits,
                                                     best_inverse_expert)) {
                    best_score_bits = score_bits;
                    best_inverse_expert = inverse_expert;
                }
            }
            const uint best_expert = 0xffffffffu - best_inverse_expert;
            selected_ids[rank] = best_expert;
            top_expert_ids[token * top_k + rank] = best_expert;
            top_weights[token * top_k + rank] = scores[best_expert];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct UocrMoeSelectedParams {
    uint hidden_size;
    uint intermediate_size;
    uint top_k;
    uint reserved;
};




kernel void uocr_moe_selected_gate_up_f16(device const half *src [[buffer(0)]],
                                          device const half *gate_weight [[buffer(1)]],
                                          device const half *up_weight [[buffer(2)]],
                                          device half *mid [[buffer(3)]],
                                          constant UocrMoeSelectedParams &params [[buffer(4)]],
                                          threadgroup float *partials [[threadgroup(0)]],
                                          uint output_index [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]],
                                          uint simd_width [[threads_per_simdgroup]]) {
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
    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, gate, up);

    if (tid == 0) {
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
                                              uint simd_width,
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
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

kernel void uocr_moe_selected_down_sum_f16_to_f16(device const half *mid [[buffer(0)]],
                                                  device const half *down_weight [[buffer(1)]],
                                                  device const float *top_weights [[buffer(2)]],
                                                  device half *dst [[buffer(3)]],
                                                  constant UocrMoeSelectedParams &params [[buffer(4)]],
                                                  threadgroup float *partials [[threadgroup(0)]],
                                                  uint out_col [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint ntg [[threads_per_threadgroup]],
                                                  uint simd_width [[threads_per_simdgroup]]) {
    if (out_col >= params.hidden_size) {
        return;
    }
    const float value = uocr_moe_selected_down_sum_dot_f16(mid, down_weight, top_weights, params, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        dst[out_col] = half(value);
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

struct UocrMoeDecodeInterleavedParams {
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint expert_stride_values;
    uint up_offset_values;
    uint down_offset_values;
    uint reserved;
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
                                                  uint ntg [[threads_per_threadgroup]],
                                                  uint simd_width [[threads_per_simdgroup]]) {
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
    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, gate, up);

    if (tid == 0) {
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
                                              uint simd_width,
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
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
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
                                                          uint ntg [[threads_per_threadgroup]],
                                                          uint simd_width [[threads_per_simdgroup]]) {
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
                                                               simd_width,
                                                               partials);
    if (tid == 0) {
        dst[output_index] = half(value);
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
                                                     uint ntg [[threads_per_threadgroup]],
                                                     uint simd_width [[threads_per_simdgroup]]) {
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
    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, gate, up);

    if (tid == 0) {
        const float silu = gate / (1.0f + exp(-gate));
        const ulong mid_index = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                                    ulong(params.intermediate_size) +
                                ulong(out_col);
        mid[mid_index] = half(silu * up);
    }
}

kernel void uocr_moe_decode_interleaved_gate_up_one_f16(device const half *src [[buffer(0)]],
                                                        device const uint *top_expert_ids [[buffer(1)]],
                                                        device const half *expert_slab [[buffer(2)]],
                                                        device half *mid [[buffer(3)]],
                                                        constant UocrMoePrefillInterleavedParams &params [[buffer(4)]],
                                                        threadgroup float *partials [[threadgroup(0)]],
                                                        uint output_index [[threadgroup_position_in_grid]],
                                                        uint tid [[thread_index_in_threadgroup]],
                                                        uint ntg [[threads_per_threadgroup]],
                                                        uint simd_width [[threads_per_simdgroup]]) {
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
    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, gate, up);

    if (tid == 0) {
        const float silu = gate / (1.0f + exp(-gate));
        const ulong mid_index = (ulong(token) * ulong(params.top_k) + ulong(rank)) *
                                    ulong(params.intermediate_size) +
                                ulong(out_col);
        mid[mid_index] = half(silu * up);
    }
}

// Decode-only routed-expert gate/up tiling.  Each threadgroup covers one
// selected rank and four adjacent intermediate columns.  The 256 lanes walk
// hidden indices in the same order as uocr_moe_decode_interleaved_gate_up_one_f16
// for every column while reusing each hidden load across the four columns.  Each
// column is then reduced with uocr_threadgroup_sum2 using the same threadgroup
// width and reduction tree as the scalar decode kernel, preserving the fp32
// accumulation feeding the fp16 SwiGLU intermediate store.  The expert-id branch
// is uniform for the whole threadgroup, and all valid/tail columns execute the
// same sequence of barriers.
[[max_total_threads_per_threadgroup(256)]] kernel void uocr_moe_decode_interleaved_gate_up_tile4_f16(device const half *src [[buffer(0)]],
                                                        device const uint *top_expert_ids [[buffer(1)]],
                                                        device const half *expert_slab [[buffer(2)]],
                                                        device half *mid [[buffer(3)]],
                                                        constant UocrMoePrefillInterleavedParams &params [[buffer(4)]],
                                                        threadgroup float *partials [[threadgroup(0)]],
                                                        uint output_tile [[threadgroup_position_in_grid]],
                                                        uint tid [[thread_index_in_threadgroup]],
                                                        uint ntg [[threads_per_threadgroup]],
                                                        uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint tiles_per_rank = uocr_div_up_u32(params.intermediate_size, tile_columns);
    const uint tiles_per_token = params.top_k * tiles_per_rank;
    const uint token = output_tile / tiles_per_token;
    const uint token_rem = output_tile - token * tiles_per_token;
    const uint rank = token_rem / tiles_per_rank;
    const uint tile = token_rem - rank * tiles_per_rank;
    if (token >= params.n_tokens || rank >= params.top_k || ntg != 256u) {
        return;
    }

    const uint out_col0 = tile * tile_columns;
    const uint out_col1 = out_col0 + 1u;
    const uint out_col2 = out_col0 + 2u;
    const uint out_col3 = out_col0 + 3u;
    const bool valid0 = out_col0 < params.intermediate_size;
    const bool valid1 = out_col1 < params.intermediate_size;
    const bool valid2 = out_col2 < params.intermediate_size;
    const bool valid3 = out_col3 < params.intermediate_size;
    const ulong mid_base = (ulong(token) * ulong(params.top_k) + ulong(rank)) * ulong(params.intermediate_size);
    const uint expert = top_expert_ids[token * params.top_k + rank];
    if (expert >= params.expert_count) {
        if (tid == 0u) {
            if (valid0) mid[mid_base + ulong(out_col0)] = half(0.0f);
            if (valid1) mid[mid_base + ulong(out_col1)] = half(0.0f);
            if (valid2) mid[mid_base + ulong(out_col2)] = half(0.0f);
            if (valid3) mid[mid_base + ulong(out_col3)] = half(0.0f);
        }
        return;
    }

    float gate0 = 0.0f;
    float up0 = 0.0f;
    float gate1 = 0.0f;
    float up1 = 0.0f;
    float gate2 = 0.0f;
    float up2 = 0.0f;
    float gate3 = 0.0f;
    float up3 = 0.0f;
    const ulong src_base = ulong(token) * ulong(params.hidden_size);
    const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values);
    const ulong gate_base0 = expert_base + ulong(out_col0) * ulong(params.hidden_size);
    const ulong gate_base1 = expert_base + ulong(out_col1) * ulong(params.hidden_size);
    const ulong gate_base2 = expert_base + ulong(out_col2) * ulong(params.hidden_size);
    const ulong gate_base3 = expert_base + ulong(out_col3) * ulong(params.hidden_size);
    const ulong up_base0 = expert_base + ulong(params.up_offset_values) + ulong(out_col0) * ulong(params.hidden_size);
    const ulong up_base1 = expert_base + ulong(params.up_offset_values) + ulong(out_col1) * ulong(params.hidden_size);
    const ulong up_base2 = expert_base + ulong(params.up_offset_values) + ulong(out_col2) * ulong(params.hidden_size);
    const ulong up_base3 = expert_base + ulong(params.up_offset_values) + ulong(out_col3) * ulong(params.hidden_size);
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        const ulong kk = ulong(k);
        const float x = float(src[src_base + kk]);
        if (valid0) {
            gate0 += x * float(expert_slab[gate_base0 + kk]);
            up0 += x * float(expert_slab[up_base0 + kk]);
        }
        if (valid1) {
            gate1 += x * float(expert_slab[gate_base1 + kk]);
            up1 += x * float(expert_slab[up_base1 + kk]);
        }
        if (valid2) {
            gate2 += x * float(expert_slab[gate_base2 + kk]);
            up2 += x * float(expert_slab[up_base2 + kk]);
        }
        if (valid3) {
            gate3 += x * float(expert_slab[gate_base3 + kk]);
            up3 += x * float(expert_slab[up_base3 + kk]);
        }
    }

    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate0, up0, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid0) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col0)] = half(silu * up);
    }
    uocr_threadgroup_sum2(gate1, up1, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid1) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col1)] = half(silu * up);
    }
    uocr_threadgroup_sum2(gate2, up2, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid2) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col2)] = half(silu * up);
    }
    uocr_threadgroup_sum2(gate3, up3, partials, partials + ntg, tid, ntg, simd_width, gate, up);
    if (tid == 0u && valid3) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[mid_base + ulong(out_col3)] = half(silu * up);
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
                                              uint simd_width,
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
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

static inline float uocr_moe_decode_interleaved_down_dot_f16(device const half *mid,
                                                             device const uint *top_expert_ids,
                                                             device const float *top_weights,
                                                             device const half *expert_slab,
                                                             constant UocrMoeDecodeInterleavedParams &params,
                                                             uint out_col,
                                                             uint tid,
                                                             uint ntg,
                                                             uint simd_width,
                                                             threadgroup float *partials) {
    const uint intermediate_size = uocr_fc_moe_expert_intermediate_or(params.intermediate_size);
    const uint top_k = uocr_fc_moe_top_k_or(params.top_k);
    float sum = 0.0f;
    for (uint rank = 0; rank < top_k; ++rank) {
        const uint expert = top_expert_ids[rank];
        float expert_sum = 0.0f;
        const ulong mid_base = ulong(rank) * ulong(intermediate_size);
        const ulong weight_base = ulong(expert) * ulong(params.expert_stride_values) +
                                  ulong(params.down_offset_values) +
                                  ulong(out_col) * ulong(intermediate_size);
        for (uint k = tid; k < intermediate_size; k += ntg) {
            expert_sum += float(mid[mid_base + ulong(k)]) * float(expert_slab[weight_base + ulong(k)]);
        }
        sum += expert_sum * top_weights[rank];
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
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
                                                             uint ntg [[threads_per_threadgroup]],
                                                             uint simd_width [[threads_per_simdgroup]]) {
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
                                                                  simd_width,
                                                                  partials);
    if (tid == 0) {
        dst[output_index] = half(value);
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_moe_decode_interleaved_down_sum_combine_f16_to_f16(device const half *mid [[buffer(0)]],
                                                                                                                device const uint *top_expert_ids [[buffer(1)]],
                                                                                                                device const float *top_weights [[buffer(2)]],
                                                                                                                device const half *expert_slab [[buffer(3)]],
                                                                                                                device const half *shared [[buffer(4)]],
                                                                                                                device const half *residual [[buffer(5)]],
                                                                                                                device half *dst [[buffer(6)]],
                                                                                                                constant UocrMoePrefillInterleavedParams &params [[buffer(7)]],
                                                                                                                threadgroup float *partials [[threadgroup(0)]],
                                                                                                                uint output_index [[threadgroup_position_in_grid]],
                                                                                                                uint tid [[thread_index_in_threadgroup]],
                                                                                                                uint ntg [[threads_per_threadgroup]],
                                                                                                                uint simd_width [[threads_per_simdgroup]]) {
    const uint token = output_index / params.hidden_size;
    const uint out_col = output_index - token * params.hidden_size;
    if (token >= params.n_tokens || out_col >= params.hidden_size) {
        return;
    }
    const float routed_value = uocr_moe_prefill_interleaved_down_dot_f16(mid,
                                                                         top_expert_ids,
                                                                         top_weights,
                                                                         expert_slab,
                                                                         params,
                                                                         token,
                                                                         out_col,
                                                                         tid,
                                                                         ntg,
                                                                         simd_width,
                                                                         partials);
    if (tid == 0) {
        // Preserve the existing fp16 behavior of the unfused path: routed down
        // projection is rounded to half before MoE combine reads it back.
        const float routed = float(half(routed_value));
        dst[output_index] = half(routed + float(shared[output_index]) + float(residual[output_index]));
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_moe_decode_interleaved_down_sum_combine_one_f16_to_f16(device const half *mid [[buffer(0)]],
                                                                                                                    device const uint *top_expert_ids [[buffer(1)]],
                                                                                                                    device const float *top_weights [[buffer(2)]],
                                                                                                                    device const half *expert_slab [[buffer(3)]],
                                                                                                                    device const half *shared [[buffer(4)]],
                                                                                                                    device const half *residual [[buffer(5)]],
                                                                                                                    device half *dst [[buffer(6)]],
                                                                                                                    constant UocrMoeDecodeInterleavedParams &params [[buffer(7)]],
                                                                                                                    threadgroup float *partials [[threadgroup(0)]],
                                                                                                                    uint out_col [[threadgroup_position_in_grid]],
                                                                                                                    uint tid [[thread_index_in_threadgroup]],
                                                                                                                    uint ntg [[threads_per_threadgroup]],
                                                                                                                    uint simd_width [[threads_per_simdgroup]]) {
    const float routed_value = uocr_moe_decode_interleaved_down_dot_f16(mid,
                                                                        top_expert_ids,
                                                                        top_weights,
                                                                        expert_slab,
                                                                        params,
                                                                        out_col,
                                                                        tid,
                                                                        ntg,
                                                                        simd_width,
                                                                        partials);
    if (tid == 0) {
        // Preserve the existing fp16 behavior of the unfused path: routed down
        // projection is rounded to half before MoE combine reads it back.
        const float routed = float(half(routed_value));
        dst[out_col] = half(routed + float(shared[out_col]) + float(residual[out_col]));
    }
}

/**
 * Decode-specialized down-combine that computes 4 adjacent output columns per
 * threadgroup, reducing threadgroup count by ~4x and reusing each mid[rank,k]
 * load across four down-projection weight columns.
 */
[[max_total_threads_per_threadgroup(256)]]
kernel void uocr_moe_decode_interleaved_down_sum_combine_tile4_f16_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const float *top_weights [[buffer(2)]],
    device const half *expert_slab [[buffer(3)]],
    device const half *shared [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    device half *dst [[buffer(6)]],
    constant UocrMoeDecodeInterleavedParams &params [[buffer(7)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint out_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint intermediate_size = uocr_fc_moe_expert_intermediate_or(params.intermediate_size);
    const uint top_k = uocr_fc_moe_top_k_or(params.top_k);
    const uint expert_count = uocr_fc_moe_experts_or(params.expert_count);

    const uint out0 = out_tile * tile_columns;
    const uint out1 = out0 + 1u;
    const uint out2 = out0 + 2u;
    const uint out3 = out0 + 3u;

    const bool valid0 = out0 < hidden_size;
    const bool valid1 = out1 < hidden_size;
    const bool valid2 = out2 < hidden_size;
    const bool valid3 = out3 < hidden_size;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (uint rank = 0u; rank < top_k; ++rank) {
        const uint expert = top_expert_ids[rank];
        if (expert >= expert_count) {
            continue;
        }

        const float route_weight = top_weights[rank];
        const ulong mid_base = ulong(rank) * ulong(intermediate_size);
        const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values) +
                                  ulong(params.down_offset_values);

        const ulong w0 = expert_base + ulong(out0) * ulong(intermediate_size);
        const ulong w1 = expert_base + ulong(out1) * ulong(intermediate_size);
        const ulong w2 = expert_base + ulong(out2) * ulong(intermediate_size);
        const ulong w3 = expert_base + ulong(out3) * ulong(intermediate_size);

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        for (uint k = tid; k < intermediate_size; k += ntg) {
            const ulong kk = ulong(k);
            const float m = float(mid[mid_base + kk]);

            if (valid0) acc0 += m * float(expert_slab[w0 + kk]);
            if (valid1) acc1 += m * float(expert_slab[w1 + kk]);
            if (valid2) acc2 += m * float(expert_slab[w2 + kk]);
            if (valid3) acc3 += m * float(expert_slab[w3 + kk]);
        }

        sum0 += acc0 * route_weight;
        sum1 += acc1 * route_weight;
        sum2 += acc2 * route_weight;
        sum3 += acc3 * route_weight;
    }

    const float routed0 = uocr_threadgroup_sum(sum0, partials, tid, ntg, simd_width);
    if (tid == 0u && valid0) {
        dst[out0] = half(float(half(routed0)) + float(shared[out0]) + float(residual[out0]));
    }

    const float routed1 = uocr_threadgroup_sum(sum1, partials, tid, ntg, simd_width);
    if (tid == 0u && valid1) {
        dst[out1] = half(float(half(routed1)) + float(shared[out1]) + float(residual[out1]));
    }

    const float routed2 = uocr_threadgroup_sum(sum2, partials, tid, ntg, simd_width);
    if (tid == 0u && valid2) {
        dst[out2] = half(float(half(routed2)) + float(shared[out2]) + float(residual[out2]));
    }

    const float routed3 = uocr_threadgroup_sum(sum3, partials, tid, ntg, simd_width);
    if (tid == 0u && valid3) {
        dst[out3] = half(float(half(routed3)) + float(shared[out3]) + float(residual[out3]));
    }
}

/**
 * Decode-specialized down-combine that computes 8 adjacent output columns per
 * threadgroup, reducing threadgroup count by ~8x vs scalar and reusing each
 * mid[rank,k] load across eight down-projection weight columns.
 */
[[max_total_threads_per_threadgroup(256)]]
kernel void uocr_moe_decode_interleaved_down_sum_combine_tile8_f16_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const float *top_weights [[buffer(2)]],
    device const half *expert_slab [[buffer(3)]],
    device const half *shared [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    device half *dst [[buffer(6)]],
    constant UocrMoeDecodeInterleavedParams &params [[buffer(7)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint out_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 8u;
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint intermediate_size = uocr_fc_moe_expert_intermediate_or(params.intermediate_size);
    const uint top_k = uocr_fc_moe_top_k_or(params.top_k);
    const uint expert_count = uocr_fc_moe_experts_or(params.expert_count);

    const uint out0 = out_tile * tile_columns;
    const uint out1 = out0 + 1u;
    const uint out2 = out0 + 2u;
    const uint out3 = out0 + 3u;
    const uint out4 = out0 + 4u;
    const uint out5 = out0 + 5u;
    const uint out6 = out0 + 6u;
    const uint out7 = out0 + 7u;

    const bool valid0 = out0 < hidden_size;
    const bool valid1 = out1 < hidden_size;
    const bool valid2 = out2 < hidden_size;
    const bool valid3 = out3 < hidden_size;
    const bool valid4 = out4 < hidden_size;
    const bool valid5 = out5 < hidden_size;
    const bool valid6 = out6 < hidden_size;
    const bool valid7 = out7 < hidden_size;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    for (uint rank = 0u; rank < top_k; ++rank) {
        const uint expert = top_expert_ids[rank];
        if (expert >= expert_count) {
            continue;
        }

        const float route_weight = top_weights[rank];
        const ulong mid_base = ulong(rank) * ulong(intermediate_size);
        const ulong expert_base = ulong(expert) * ulong(params.expert_stride_values) +
                                  ulong(params.down_offset_values);

        const ulong w0 = expert_base + ulong(out0) * ulong(intermediate_size);
        const ulong w1 = expert_base + ulong(out1) * ulong(intermediate_size);
        const ulong w2 = expert_base + ulong(out2) * ulong(intermediate_size);
        const ulong w3 = expert_base + ulong(out3) * ulong(intermediate_size);
        const ulong w4 = expert_base + ulong(out4) * ulong(intermediate_size);
        const ulong w5 = expert_base + ulong(out5) * ulong(intermediate_size);
        const ulong w6 = expert_base + ulong(out6) * ulong(intermediate_size);
        const ulong w7 = expert_base + ulong(out7) * ulong(intermediate_size);

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        float acc4 = 0.0f;
        float acc5 = 0.0f;
        float acc6 = 0.0f;
        float acc7 = 0.0f;

        for (uint k = tid; k < intermediate_size; k += ntg) {
            const ulong kk = ulong(k);
            const float m = float(mid[mid_base + kk]);

            if (valid0) acc0 += m * float(expert_slab[w0 + kk]);
            if (valid1) acc1 += m * float(expert_slab[w1 + kk]);
            if (valid2) acc2 += m * float(expert_slab[w2 + kk]);
            if (valid3) acc3 += m * float(expert_slab[w3 + kk]);
            if (valid4) acc4 += m * float(expert_slab[w4 + kk]);
            if (valid5) acc5 += m * float(expert_slab[w5 + kk]);
            if (valid6) acc6 += m * float(expert_slab[w6 + kk]);
            if (valid7) acc7 += m * float(expert_slab[w7 + kk]);
        }

        sum0 += acc0 * route_weight;
        sum1 += acc1 * route_weight;
        sum2 += acc2 * route_weight;
        sum3 += acc3 * route_weight;
        sum4 += acc4 * route_weight;
        sum5 += acc5 * route_weight;
        sum6 += acc6 * route_weight;
        sum7 += acc7 * route_weight;
    }

    const float routed0 = uocr_threadgroup_sum(sum0, partials, tid, ntg, simd_width);
    if (tid == 0u && valid0) {
        dst[out0] = half(float(half(routed0)) + float(shared[out0]) + float(residual[out0]));
    }

    const float routed1 = uocr_threadgroup_sum(sum1, partials, tid, ntg, simd_width);
    if (tid == 0u && valid1) {
        dst[out1] = half(float(half(routed1)) + float(shared[out1]) + float(residual[out1]));
    }

    const float routed2 = uocr_threadgroup_sum(sum2, partials, tid, ntg, simd_width);
    if (tid == 0u && valid2) {
        dst[out2] = half(float(half(routed2)) + float(shared[out2]) + float(residual[out2]));
    }

    const float routed3 = uocr_threadgroup_sum(sum3, partials, tid, ntg, simd_width);
    if (tid == 0u && valid3) {
        dst[out3] = half(float(half(routed3)) + float(shared[out3]) + float(residual[out3]));
    }

    const float routed4 = uocr_threadgroup_sum(sum4, partials, tid, ntg, simd_width);
    if (tid == 0u && valid4) {
        dst[out4] = half(float(half(routed4)) + float(shared[out4]) + float(residual[out4]));
    }

    const float routed5 = uocr_threadgroup_sum(sum5, partials, tid, ntg, simd_width);
    if (tid == 0u && valid5) {
        dst[out5] = half(float(half(routed5)) + float(shared[out5]) + float(residual[out5]));
    }

    const float routed6 = uocr_threadgroup_sum(sum6, partials, tid, ntg, simd_width);
    if (tid == 0u && valid6) {
        dst[out6] = half(float(half(routed6)) + float(shared[out6]) + float(residual[out6]));
    }

    const float routed7 = uocr_threadgroup_sum(sum7, partials, tid, ntg, simd_width);
    if (tid == 0u && valid7) {
        dst[out7] = half(float(half(routed7)) + float(shared[out7]) + float(residual[out7]));
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

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_moe_combine_f16_to_f16(device const half *routed [[buffer(0)]],
                                        device const half *shared [[buffer(1)]],
                                        device const half *residual [[buffer(2)]],
                                        device half *dst [[buffer(3)]],
                                        constant UocrMoeCombineParams &params [[buffer(4)]],
                                        uint gid [[thread_position_in_grid]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint total = params.n_tokens * hidden_size;
    const uint value_base = gid * UOCR_HALF4_WIDTH;
    if (value_base >= total) {
        return;
    }
    if (value_base + (UOCR_HALF4_WIDTH - 1u) < total) {
        float4 value = float4(uocr_load_half4(routed, ulong(value_base))) +
                       float4(uocr_load_half4(shared, ulong(value_base)));
        if (params.has_residual != 0u) {
            value += float4(uocr_load_half4(residual, ulong(value_base)));
        }
        uocr_store_half4(dst, ulong(value_base), half4(value));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && value_base + i < total; ++i) {
            const uint idx = value_base + i;
            dst[idx] = half(uocr_moe_combine_value_f16(routed, shared, residual, params, idx));
        }
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_moe_combine_f16_to_f32(device const half *routed [[buffer(0)]],
                                        device const half *shared [[buffer(1)]],
                                        device const half *residual [[buffer(2)]],
                                        device float *dst [[buffer(3)]],
                                        constant UocrMoeCombineParams &params [[buffer(4)]],
                                        uint gid [[thread_position_in_grid]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint total = params.n_tokens * hidden_size;
    if (gid >= total) {
        return;
    }
    dst[gid] = uocr_moe_combine_value_f16(routed, shared, residual, params, gid);
}
