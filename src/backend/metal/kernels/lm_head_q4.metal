// lm_head_q4.metal - Fused Q4_0 LM-head argmax kernel.
//
// Structure differs from the Q8 kernel: instead of 8 lanes per token with the
// hidden vector staged in threadgroup memory (which caps Q4 at ~1.1x because
// the TG-load instruction rate does not halve with the weight bytes), one
// simdgroup owns one vocab row and streams the packed Q4 weights with the
// group-half-split unpack, reading the 2.5 KB hidden vector through the
// device cache.  32 simdgroups per 1024-thread threadgroup produce one
// argmax partial per 32 rows - the same partial layout as the Q8 kernel
// (vocab / 32), so the selection scratch and finalize pass are unchanged.
// Measured ~1.2x vs the Q8 argmax on M1 Pro (~0.15 ms/token).
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

[[max_total_threads_per_threadgroup(1024)]] kernel void uocr_lm_head_argmax_q4_0_to_f16(
    device const half *hidden [[buffer(0)]],
    device const uchar *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device float *partial_scores_out [[buffer(3)]],
    device uint *partial_ids_out [[buffer(4)]],
    constant UocrLmHeadArgmaxQ4Params &params [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    threadgroup uint *partial_ids [[threadgroup(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    ushort lane_u16 [[thread_index_in_simdgroup]],
    ushort simdgroup_u16 [[simdgroup_index_in_threadgroup]],
    ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simdgroup = uint(simdgroup_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint rows_per_tg = ntg / simd_width;
    const uint row = tile * rows_per_tg + simdgroup;
    const uint hidden_size = params.hidden_size;
    const uint vocab_size = params.vocab_size;

    float score = -INFINITY;
    uint id = 0xffffffffu;
    if (row < vocab_size && params.group_size == 64u) {
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
        partials[simdgroup] = score;
        partial_ids[simdgroup] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Rows ascend with simdgroup index, so a strict > keeps the lowest id on
     * ties - matching the Q8 argmax tie-break. */
    if (tid == 0u && tile < params.partial_count) {
        float best = -INFINITY;
        uint best_id = 0xffffffffu;
        for (uint i = 0u; i < rows_per_tg; ++i) {
            if (partial_ids[i] == 0xffffffffu) {
                continue;
            }
            if (best_id == 0xffffffffu || partials[i] > best ||
                (partials[i] == best && partial_ids[i] < best_id)) {
                best = partials[i];
                best_id = partial_ids[i];
            }
        }
        partial_scores_out[tile] = best_id == 0xffffffffu ? -INFINITY : best;
        partial_ids_out[tile] = best_id;
    }
}
