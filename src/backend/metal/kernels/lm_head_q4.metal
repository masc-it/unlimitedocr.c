// lm_head_q4.metal - Fused Q4_0 LM-head argmax kernel.
//
// Structure differs from the Q8 kernel: instead of 8 lanes per token with the
// hidden vector staged in threadgroup memory (which caps Q4 at ~1.1x because
// the TG-load instruction rate does not halve with the weight bytes), one
// simdgroup computes one vocab row at a time, using group-half-split unpacking
// and reading the 2.5 KB hidden vector through the
// device cache. Eight SIMD groups per 256-thread group visit four rows each.
// A SIMD reduction produces one partial per 32 rows, preserving the Q8
// partial layout and the existing selection scratch/finalize pass.
#include "common.metal"

struct UocrLmHeadArgmaxQ4Params {
    uint vocab_size;
    uint hidden_size;
    uint tile_tokens;      // rows per threadgroup (32)
    uint lanes_per_token;  // unused; kept for ABI symmetry with Q8
    uint group_size;
    uint groups_per_row;
    uint partial_count;
    uint reserved0;
};

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_lm_head_argmax_q4_0_to_f16(
    device const half *hidden [[buffer(0)]],
    device const uchar *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device float *partial_scores_out [[buffer(3)]],
    device uint *partial_ids_out [[buffer(4)]],
    constant UocrLmHeadArgmaxQ4Params &params [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    threadgroup uint *partial_ids [[threadgroup(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]],
    ushort lane_u16 [[thread_index_in_simdgroup]],
    ushort simdgroup_u16 [[simdgroup_index_in_threadgroup]],
    ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simdgroup = uint(simdgroup_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint groups = ntg / simd_width;
    const uint hidden_size = params.hidden_size;
    const uint vocab_size = params.vocab_size;

    // Each SIMD group visits four rows with the original lane-dot order.
    for (uint local = simdgroup; local < 32u; local += groups) {
        const uint row = tile * 32u + local;
        float score = -INFINITY;
        uint id = 0xffffffffu;
        if (row < vocab_size && uocr_lm_token_allowed(row, params.reserved0) && params.group_size == 64u) {
            device const uchar *w_row = qweight + (ulong)row * (ulong)(hidden_size / 2u);
            device const half *s_row = qscale + (ulong)row * (ulong)params.groups_per_row;
            device const uchar4 *w4 = (device const uchar4 *)w_row;
            device const half4 *x4 = (device const half4 *)hidden;
            float sum = 0.0f;
            const uint chunks = hidden_size / 8u; // 4 packed bytes -> 8 weights
            for (uint c = lane; c < chunks; c += simd_width) {
                const uchar4 w = w4[c];
                const uint group = c / 8u;       // 8 uchar4 chunks per 64-wide group
                const uint j4 = c - group * 8u;  // half4 index inside the low half
                const float scale = float(s_row[group]);
                const uint xbase = group * 16u + j4; // 16 half4 per 64-wide group
                float d = dot(float4(x4[xbase]), float4(int4(w) & 0xF) - 8.0f);
                d += dot(float4(x4[xbase + 8u]), float4(int4(w) >> 4) - 8.0f);
                sum += d * scale;
            }
            const float total = simd_sum(sum);
            if (!isnan(total)) {
                score = total;
                id = row;
            }
        }

        if (lane == 0u) {
            partials[local] = score;
            partial_ids[local] = id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Preserve the 32-row partial layout and lowest-ID tie break. Fetch the
    // winning row's score so signed-zero ties retain the selected score bits.
    if (simdgroup == 0u) {
        const float best = simd_max(partials[lane]);
        const uint winner = simd_min(partials[lane] == best ? partial_ids[lane] : 0xffffffffu);
        if (lane == 0u && tile < params.partial_count) {
            partial_ids_out[tile] = winner;
            partial_scores_out[tile] = winner == 0xffffffffu ? -INFINITY : partials[winner - tile * 32u];
        }
    }
}
