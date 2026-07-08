// gemm_q4.metal - Expert-bucketed tiled Q4_0 GEMM kernels for routed-MoE
// prefill.
//
// Same 64x32x32 simdgroup-MMA regime as gemm_q8.metal; only the B-side
// staging differs: Q4_0 weights are dequantized on stage from the
// group-half-split nibble packing (docs/plan_q4.md §1.1).  A BK=32 K-tile is
// exactly one nibble half of a 64-wide scale group, so each staged run reads
// contiguous bytes and extracts a single nibble side with one scale lookup.
//
// Reuses uocr_gemm_q8_stage_a_gathered / uocr_gemm_q8_mma_tile and the
// bucket lookup helpers from gemm_q8.metal (earlier fragment).
#include "common.metal"

// Mirrors UocrMoeBucketedGemmQ8Params; qweight stride/offset fields are
// PACKED BYTE counts for Q4, scale fields stay value counts.
struct UocrMoeBucketedGemmQ4Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint expert_stride_bytes;
    uint up_offset_bytes;
    uint down_offset_bytes;
    uint expert_scale_stride_values;
    uint up_scale_offset_values;
    uint down_scale_offset_values;
    uint group_size;
};

// Dequantize a Q4_0 W tile transposed to [BK][BN] (k-major for B-side MMA
// loads).  Each thread handles one (column, 8-wide k run); with BK == 32 and
// k_base a multiple of 32, the run stays inside one nibble half of one
// 64-wide scale group.
static inline void uocr_gemm_q4_stage_b(device const uchar *qweight,
                                        device const half *qscale,
                                        uint K,
                                        uint groups_per_row,
                                        uint col_base,
                                        uint k_base,
                                        uint tid,
                                        threadgroup half *sb) {
    const uint wn = tid / 4u;
    const uint wk0 = (tid % 4u) * 8u;
    const uint wcol = col_base + wn;
    const uint k0 = k_base + wk0;
    const uint group = k0 / 64u;
    const float scale = float(qscale[wcol * groups_per_row + group]);
    const uint j = k0 - group * 64u; // 0..63; BK tiles keep the 8-run in one half
    const bool high = j >= 32u;
    device const uchar *w = qweight + (ulong)wcol * (K / 2u) + group * 32u + (high ? j - 32u : j);
    for (uint kk = 0u; kk < 8u; ++kk) {
        const uint b = uint(w[kk]);
        const int q = int(high ? (b >> 4u) : (b & 0xFu)) - 8;
        sb[(wk0 + kk) * UOCR_GEMM_Q8_BN + wn] = half(float(q) * scale);
    }
}

// Vision GEMM: Q4_0 twin of uocr_vision_gemm_q8_0_to_f16 (bias + activation
// + optional QKV-split epilogue).  Requires n % 32 == 0 and k % 64 == 0 (the
// Q4 B-side staging has no column guard); the host validates both.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_vision_gemm_q4_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uchar *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device half *dst0 [[buffer(4)]],
    device half *dst1 [[buffer(5)]],
    device half *dst2 [[buffer(6)]],
    constant UocrVisionGemmQ8Params &params [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb = sa + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK;
    threadgroup float *sc = smem;

    const uint K = params.k;
    const uint row_base = tgpig.y * UOCR_GEMM_Q8_BM;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;

    simdgroup_float8x8 mc[4][2];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sg_row = (sgitg / 2u) * 32u;
    const uint sg_col = (sgitg % 2u) * 16u;

    for (uint k_base = 0u; k_base < K; k_base += UOCR_GEMM_Q8_BK) {
        uocr_gemm_q8_stage_a(src, params.m, K, row_base, k_base, tid, sa);
        uocr_gemm_q4_stage_b(qweight, qscale, K, params.groups_per_row, col_base, k_base, tid, sb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uocr_gemm_q8_mma_tile(sa, sb, sg_row, sg_col, mc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(mc[i][j], sc + (sg_row + i * 8u) * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN; i += UOCR_GEMM_Q8_THREADS) {
        const uint mrow = i / UOCR_GEMM_Q8_BN;
        const uint n = i - mrow * UOCR_GEMM_Q8_BN;
        const uint row = row_base + mrow;
        const uint col = col_base + n;
        if (row >= params.m || col >= params.n) {
            continue;
        }
        float value = sc[i] + float(bias[col]);
        if (params.activation == UOCR_VISION_Q8_ACT_QUICKGELU) {
            value = uocr_quickgelu(value);
        } else if (params.activation == UOCR_VISION_Q8_ACT_GELU_ERF) {
            value = uocr_gelu_erf(value);
        }
        if (params.split_size == 0u) {
            dst0[(ulong)row * params.n + col] = half(value);
        } else {
            const uint projection = col / params.split_size;
            const uint local_col = col - projection * params.split_size;
            device half *dst = projection == 0u ? dst0 : (projection == 1u ? dst1 : dst2);
            dst[(ulong)row * params.split_size + local_col] = half(value);
        }
    }
}

// Decoder prefill GEMM with optional residual add (dense/shared MLP down
// projection).  Q4_0 twin of uocr_decoder_gemm_residual_q8_0_to_f16; reuses
// UocrDecoderGemmQ8Params (gemm_q8.metal is an earlier fragment).
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_decoder_gemm_residual_q4_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uchar *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device half *dst [[buffer(4)]],
    constant UocrDecoderGemmQ8Params &params [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb = sa + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK;
    threadgroup float *sc = smem;

    const uint K = params.k;
    const uint row_base = tgpig.y * UOCR_GEMM_Q8_BM;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;

    simdgroup_float8x8 mc[4][2];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sg_row = (sgitg / 2u) * 32u;
    const uint sg_col = (sgitg % 2u) * 16u;

    for (uint k_base = 0u; k_base < K; k_base += UOCR_GEMM_Q8_BK) {
        uocr_gemm_q8_stage_a(src, params.m, K, row_base, k_base, tid, sa);
        uocr_gemm_q4_stage_b(qweight, qscale, K, params.groups_per_row, col_base, k_base, tid, sb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uocr_gemm_q8_mma_tile(sa, sb, sg_row, sg_col, mc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(mc[i][j], sc + (sg_row + i * 8u) * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN; i += UOCR_GEMM_Q8_THREADS) {
        const uint mrow = i / UOCR_GEMM_Q8_BN;
        const uint col = col_base + (i - mrow * UOCR_GEMM_Q8_BN);
        const uint row = row_base + mrow;
        if (row < params.m && col < params.n) {
            const ulong dst_index = (ulong)row * params.n + col;
            float value = sc[i];
            if (params.has_residual != 0u) {
                value += float(residual[dst_index]);
            }
            dst[dst_index] = half(value);
        }
    }
}

// Decoder prefill fused SwiGLU gate/up (dense/shared MLP): Q4_0 twin of
// uocr_decoder_swiglu_gate_up_gemm_q8_0_to_f16.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_decoder_swiglu_gate_up_gemm_q4_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uchar *gate_qweight [[buffer(1)]],
    device const uchar *up_qweight [[buffer(2)]],
    device const half *gate_qscale [[buffer(3)]],
    device const half *up_qscale [[buffer(4)]],
    device half *mid [[buffer(5)]],
    constant UocrDecoderGemmQ8Params &params [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[2u * UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb_gate = sa + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK;
    threadgroup half *sb_up = sb_gate + UOCR_GEMM_Q8_BK * UOCR_GEMM_Q8_BN;
    threadgroup float *sc_gate = smem;
    threadgroup float *sc_up = smem + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN;

    const uint K = params.k;
    const uint row_base = tgpig.y * UOCR_GEMM_Q8_BM;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;

    simdgroup_float8x8 mc_gate[4][2];
    simdgroup_float8x8 mc_up[4][2];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            mc_gate[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            mc_up[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sg_row = (sgitg / 2u) * 32u;
    const uint sg_col = (sgitg % 2u) * 16u;

    for (uint k_base = 0u; k_base < K; k_base += UOCR_GEMM_Q8_BK) {
        uocr_gemm_q8_stage_a(src, params.m, K, row_base, k_base, tid, sa);
        uocr_gemm_q4_stage_b(gate_qweight, gate_qscale, K, params.groups_per_row, col_base, k_base, tid, sb_gate);
        uocr_gemm_q4_stage_b(up_qweight, up_qscale, K, params.groups_per_row, col_base, k_base, tid, sb_up);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uocr_gemm_q8_mma_tile(sa, sb_gate, sg_row, sg_col, mc_gate);
        uocr_gemm_q8_mma_tile(sa, sb_up, sg_row, sg_col, mc_up);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(mc_gate[i][j], sc_gate + (sg_row + i * 8u) * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
            simdgroup_store(mc_up[i][j], sc_up + (sg_row + i * 8u) * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN; i += UOCR_GEMM_Q8_THREADS) {
        const uint mrow = i / UOCR_GEMM_Q8_BN;
        const uint col = col_base + (i - mrow * UOCR_GEMM_Q8_BN);
        const uint row = row_base + mrow;
        if (row < params.m && col < params.n) {
            const float gate = sc_gate[i];
            const float silu = gate / (1.0f + exp(-gate));
            mid[(ulong)row * params.n + col] = half(silu * sc_up[i]);
        }
    }
}

// Bucketed gate/up GEMM: mid[pair, col] = silu(src_t @ Wg_e^T) * (src_t @ Wu_e^T)
// for every pair (t, e) in the tile's bucket.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_moe_prefill_bucketed_swiglu_gate_up_gemm_q4_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uint *pair_rows [[buffer(1)]],
    device const uint *expert_offsets [[buffer(2)]],
    device const uint *tile_prefix [[buffer(3)]],
    device const uchar *expert_qweight [[buffer(4)]],
    device const half *expert_qscale [[buffer(5)]],
    device half *mid [[buffer(6)]],
    constant UocrMoeBucketedGemmQ4Params &params [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[2u * UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN];
    threadgroup uint tile_pairs[UOCR_GEMM_Q8_BM];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb_gate = sa + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK;
    threadgroup half *sb_up = sb_gate + UOCR_GEMM_Q8_BK * UOCR_GEMM_Q8_BN;
    threadgroup float *sc_gate = smem;
    threadgroup float *sc_up = smem + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN;

    uint expert = 0u;
    uint bucket_start = 0u;
    uint bucket_rows = 0u;
    if (!uocr_moe_bucket_tile_lookup(tile_prefix, expert_offsets, params.expert_count,
                                     tgpig.y, UOCR_GEMM_Q8_BM, expert, bucket_start, bucket_rows)) {
        return;
    }
    uocr_moe_bucket_stage_rows(pair_rows, bucket_start, bucket_rows, tid, tile_pairs);

    const uint K = params.hidden_size;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;
    device const uchar *gate_qweight = expert_qweight + (ulong)expert * params.expert_stride_bytes;
    device const uchar *up_qweight = gate_qweight + params.up_offset_bytes;
    device const half *gate_qscale = expert_qscale + (ulong)expert * params.expert_scale_stride_values;
    device const half *up_qscale = gate_qscale + params.up_scale_offset_values;
    const uint groups_per_row = K / params.group_size;

    simdgroup_float8x8 mc_gate[4][2];
    simdgroup_float8x8 mc_up[4][2];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            mc_gate[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            mc_up[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sg_row = (sgitg / 2u) * 32u;
    const uint sg_col = (sgitg % 2u) * 16u;

    for (uint k_base = 0u; k_base < K; k_base += UOCR_GEMM_Q8_BK) {
        uocr_gemm_q8_stage_a_gathered(src, K, k_base, params.top_k, tile_pairs, tid, sa);
        uocr_gemm_q4_stage_b(gate_qweight, gate_qscale, K, groups_per_row, col_base, k_base, tid, sb_gate);
        uocr_gemm_q4_stage_b(up_qweight, up_qscale, K, groups_per_row, col_base, k_base, tid, sb_up);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uocr_gemm_q8_mma_tile(sa, sb_gate, sg_row, sg_col, mc_gate);
        uocr_gemm_q8_mma_tile(sa, sb_up, sg_row, sg_col, mc_up);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(mc_gate[i][j], sc_gate + (sg_row + i * 8u) * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
            simdgroup_store(mc_up[i][j], sc_up + (sg_row + i * 8u) * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN; i += UOCR_GEMM_Q8_THREADS) {
        const uint mrow = i / UOCR_GEMM_Q8_BN;
        const uint col = col_base + (i - mrow * UOCR_GEMM_Q8_BN);
        const uint pair = tile_pairs[mrow];
        if (pair != 0xFFFFFFFFu && col < params.intermediate_size) {
            const float gate = sc_gate[i];
            const float silu = gate / (1.0f + exp(-gate));
            mid[(ulong)pair * params.intermediate_size + col] = half(silu * sc_up[i]);
        }
    }
}

// Bucketed down GEMM: down_out[pair, col] = mid[pair, :] @ Wd_e^T.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_moe_prefill_bucketed_down_gemm_q4_0_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *pair_rows [[buffer(1)]],
    device const uint *expert_offsets [[buffer(2)]],
    device const uint *tile_prefix [[buffer(3)]],
    device const uchar *expert_qweight [[buffer(4)]],
    device const half *expert_qscale [[buffer(5)]],
    device half *down_out [[buffer(6)]],
    constant UocrMoeBucketedGemmQ4Params &params [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN];
    threadgroup uint tile_pairs[UOCR_GEMM_Q8_BM];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb = sa + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK;
    threadgroup float *sc = smem;

    uint expert = 0u;
    uint bucket_start = 0u;
    uint bucket_rows = 0u;
    if (!uocr_moe_bucket_tile_lookup(tile_prefix, expert_offsets, params.expert_count,
                                     tgpig.y, UOCR_GEMM_Q8_BM, expert, bucket_start, bucket_rows)) {
        return;
    }
    uocr_moe_bucket_stage_rows(pair_rows, bucket_start, bucket_rows, tid, tile_pairs);

    const uint K = params.intermediate_size;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;
    device const uchar *down_qweight = expert_qweight + (ulong)expert * params.expert_stride_bytes + params.down_offset_bytes;
    device const half *down_qscale = expert_qscale + (ulong)expert * params.expert_scale_stride_values + params.down_scale_offset_values;
    const uint groups_per_row = K / params.group_size;

    simdgroup_float8x8 mc[4][2];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sg_row = (sgitg / 2u) * 32u;
    const uint sg_col = (sgitg % 2u) * 16u;

    for (uint k_base = 0u; k_base < K; k_base += UOCR_GEMM_Q8_BK) {
        uocr_gemm_q8_stage_a_gathered(mid, K, k_base, 1u, tile_pairs, tid, sa);
        uocr_gemm_q4_stage_b(down_qweight, down_qscale, K, groups_per_row, col_base, k_base, tid, sb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uocr_gemm_q8_mma_tile(sa, sb, sg_row, sg_col, mc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(mc[i][j], sc + (sg_row + i * 8u) * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN; i += UOCR_GEMM_Q8_THREADS) {
        const uint mrow = i / UOCR_GEMM_Q8_BN;
        const uint col = col_base + (i - mrow * UOCR_GEMM_Q8_BN);
        const uint pair = tile_pairs[mrow];
        if (pair != 0xFFFFFFFFu && col < params.hidden_size) {
            down_out[(ulong)pair * params.hidden_size + col] = half(sc[i]);
        }
    }
}
