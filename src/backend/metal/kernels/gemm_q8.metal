// gemm_q8.metal - Tiled fused Q8_0 GEMM kernels for multi-row workloads.
//
// GEMV-style kernels (one threadgroup reduction per output element) are right
// for single-token decode, but multi-row GEMMs (vision encoders, decoder
// prefill) must reuse each dequantized weight tile across a block of rows.
// These kernels stage Q8 weight tiles once per 64-row block in threadgroup
// memory and use simdgroup fp16 MMA with fp32 accumulators.
//
// Requires uocr_quickgelu (dense.metal) and uocr_gelu_erf (sam.metal) in the
// combined translation unit.
#include "common.metal"

#define UOCR_GEMM_Q8_BM 64u
#define UOCR_GEMM_Q8_BN 32u
#define UOCR_GEMM_Q8_BK 32u
#define UOCR_GEMM_Q8_THREADS 128u

#define UOCR_VISION_Q8_ACT_NONE 0u
#define UOCR_VISION_Q8_ACT_QUICKGELU 1u
#define UOCR_VISION_Q8_ACT_GELU_ERF 2u

// Stage the fp16 A tile [BM][BK] (zero-filled past m rows).
static inline void uocr_gemm_q8_stage_a(device const half *src,
                                        uint m,
                                        uint K,
                                        uint row_base,
                                        uint k_base,
                                        uint tid,
                                        threadgroup half *sa) {
    for (uint i = tid; i < (UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK) / UOCR_HALF4_WIDTH; i += UOCR_GEMM_Q8_THREADS) {
        const uint chunks_per_row = UOCR_GEMM_Q8_BK / UOCR_HALF4_WIDTH;
        const uint mrow = i / chunks_per_row;
        const uint kk = (i - mrow * chunks_per_row) * UOCR_HALF4_WIDTH;
        const uint row = row_base + mrow;
        const half4 value = row < m ? uocr_load_half4(src, (ulong)row * K + k_base + kk) : uocr_zero_half4();
        *reinterpret_cast<threadgroup half4 *>(sa + mrow * UOCR_GEMM_Q8_BK + kk) = value;
    }
}

// Dequantize a W tile transposed to [BK][BN] (k-major for B-side MMA loads).
// Each thread handles one (column, 8-wide k run); aligned 8-runs never cross a
// 64-element scale group.
static inline void uocr_gemm_q8_stage_b(device const char *qweight,
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
    const float scale = float(qscale[wcol * groups_per_row + k0 / 64u]);
    device const char *w = qweight + (ulong)wcol * K + k0;
    for (uint kk = 0u; kk < 8u; ++kk) {
        sb[(wk0 + kk) * UOCR_GEMM_Q8_BN + wn] = half(float(int(w[kk])) * scale);
    }
}

// MMA over one staged K tile into the per-simdgroup accumulators.
static inline void uocr_gemm_q8_mma_tile(threadgroup const half *sa,
                                         threadgroup const half *sb,
                                         uint sg_row,
                                         uint sg_col,
                                         thread simdgroup_float8x8 (&mc)[4][2]) {
    for (uint kk = 0u; kk < UOCR_GEMM_Q8_BK; kk += 8u) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];
        for (uint i = 0u; i < 4u; ++i) {
            simdgroup_load(ma[i], sa + (sg_row + i * 8u) * UOCR_GEMM_Q8_BK + kk, UOCR_GEMM_Q8_BK);
        }
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_load(mb[j], sb + kk * UOCR_GEMM_Q8_BN + sg_col + j * 8u, UOCR_GEMM_Q8_BN);
        }
        for (uint i = 0u; i < 4u; ++i) {
            for (uint j = 0u; j < 2u; ++j) {
                simdgroup_multiply_accumulate(mc[i][j], ma[i], mb[j], mc[i][j]);
            }
        }
    }
}

// Full tile accumulation: sc[BM][BN] fp32 = A[rows][K] @ W[cols][K]^T.
// sa/sb staging may alias sc; sc is only written after the final barrier.
static inline void uocr_gemm_q8_accumulate(device const half *src,
                                           uint m,
                                           uint K,
                                           device const char *qweight,
                                           device const half *qscale,
                                           uint groups_per_row,
                                           uint row_base,
                                           uint col_base,
                                           uint tid,
                                           uint sgitg,
                                           threadgroup half *sa,
                                           threadgroup half *sb,
                                           threadgroup float *sc) {
    simdgroup_float8x8 mc[4][2];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sg_row = (sgitg / 2u) * 32u;
    const uint sg_col = (sgitg % 2u) * 16u;

    for (uint k_base = 0u; k_base < K; k_base += UOCR_GEMM_Q8_BK) {
        uocr_gemm_q8_stage_a(src, m, K, row_base, k_base, tid, sa);
        uocr_gemm_q8_stage_b(qweight, qscale, K, groups_per_row, col_base, k_base, tid, sb);
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
}

struct UocrVisionGemmQ8Params {
    uint m;              // token rows
    uint k;              // input features (multiple of 32; groups of 64)
    uint n;              // output features (multiple of 32)
    uint groups_per_row; // k / 64
    uint activation;     // UOCR_VISION_Q8_ACT_*
    uint split_size;     // 0: single dst; else per-projection width (fused QKV)
    uint reserved0;
    uint reserved1;
};

// Vision GEMM: C[64x32] = A @ W^T with bias + activation (+ optional QKV
// split) epilogue.  128 threads = 4 simdgroups in a 2x2 grid of 32x16
// sub-tiles.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_vision_gemm_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *qweight [[buffer(1)]],
    device const half *qscale [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device half *dst0 [[buffer(4)]],
    device half *dst1 [[buffer(5)]],
    device half *dst2 [[buffer(6)]],
    constant UocrVisionGemmQ8Params &params [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
    // Aliased staging: [sa 64x32 half | sb 32x32 half] while accumulating,
    // reused as the fp32 output tile sc[64x32] for the epilogue.
    threadgroup float smem[UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb = sa + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK;
    threadgroup float *sc = smem;

    const uint row_base = tgpig.y * UOCR_GEMM_Q8_BM;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;
    uocr_gemm_q8_accumulate(src, params.m, params.k, qweight, qscale, params.groups_per_row,
                            row_base, col_base, tid, sgitg, sa, sb, sc);

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

struct UocrDecoderGemmQ8Params {
    uint m;              // token rows
    uint k;              // input features
    uint n;              // output features
    uint groups_per_row; // k / 64
    uint has_residual;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

// Decoder prefill QKV: grid.z selects the projection (0=Q, 1=K, 2=V); each
// projection is an independent [m,1280] GEMM without bias.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_decoder_qkv_gemm_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *q_qweight [[buffer(1)]],
    device const char *k_qweight [[buffer(2)]],
    device const char *v_qweight [[buffer(3)]],
    device const half *q_qscale [[buffer(4)]],
    device const half *k_qscale [[buffer(5)]],
    device const half *v_qscale [[buffer(6)]],
    device half *q_dst [[buffer(7)]],
    device half *k_dst [[buffer(8)]],
    device half *v_dst [[buffer(9)]],
    constant UocrDecoderGemmQ8Params &params [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb = sa + UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK;
    threadgroup float *sc = smem;

    device const char *qweight = tgpig.z == 0u ? q_qweight : (tgpig.z == 1u ? k_qweight : v_qweight);
    device const half *qscale = tgpig.z == 0u ? q_qscale : (tgpig.z == 1u ? k_qscale : v_qscale);
    device half *dst = tgpig.z == 0u ? q_dst : (tgpig.z == 1u ? k_dst : v_dst);

    const uint row_base = tgpig.y * UOCR_GEMM_Q8_BM;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;
    uocr_gemm_q8_accumulate(src, params.m, params.k, qweight, qscale, params.groups_per_row,
                            row_base, col_base, tid, sgitg, sa, sb, sc);

    for (uint i = tid; i < UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BN; i += UOCR_GEMM_Q8_THREADS) {
        const uint mrow = i / UOCR_GEMM_Q8_BN;
        const uint col = col_base + (i - mrow * UOCR_GEMM_Q8_BN);
        const uint row = row_base + mrow;
        if (row < params.m && col < params.n) {
            dst[(ulong)row * params.n + col] = half(sc[i]);
        }
    }
}

// Decoder prefill GEMM with optional residual add (attention output
// projection and dense/shared MLP down projection).
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_decoder_gemm_residual_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *qweight [[buffer(1)]],
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

    const uint row_base = tgpig.y * UOCR_GEMM_Q8_BM;
    const uint col_base = tgpig.x * UOCR_GEMM_Q8_BN;
    uocr_gemm_q8_accumulate(src, params.m, params.k, qweight, qscale, params.groups_per_row,
                            row_base, col_base, tid, sgitg, sa, sb, sc);

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

// Decoder prefill fused SwiGLU gate/up: mid = silu(A @ Wg^T) * (A @ Wu^T).
// Staging occupies the low 8 KB while accumulating; the epilogue reuses the
// full 16 KB as two fp32 output tiles.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_decoder_swiglu_gate_up_gemm_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const char *gate_qweight [[buffer(1)]],
    device const char *up_qweight [[buffer(2)]],
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
        uocr_gemm_q8_stage_b(gate_qweight, gate_qscale, K, params.groups_per_row, col_base, k_base, tid, sb_gate);
        uocr_gemm_q8_stage_b(up_qweight, up_qscale, K, params.groups_per_row, col_base, k_base, tid, sb_up);
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

// ── Expert-bucketed routed-MoE prefill ─────────────────────────────────────
//
// Prefill routes n_tokens*top_k (token, expert) pairs.  To reuse dequantized
// expert weight tiles across rows, pairs are first bucketed by expert; the
// tiled GEMM then runs per bucket with gathered A rows (mul_mm_id style).

struct UocrMoeBucketQ8Params {
    uint n_tokens;
    uint top_k;
    uint expert_count;
    uint rows_per_tile;
};

// Mirrors UocrMoePrefillInterleavedQ8Params (moe_q8.metal is a later fragment).
struct UocrMoeBucketedGemmQ8Params {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint expert_count;
    uint top_k;
    uint expert_stride_values;
    uint up_offset_values;
    uint down_offset_values;
    uint expert_scale_stride_values;
    uint up_scale_offset_values;
    uint down_scale_offset_values;
    uint group_size;
};

// Single-threadgroup bucketing pass: histogram, prefix sums, scatter.
// pair_rows[expert_offsets[e] .. expert_offsets[e+1]) lists pair ids
// (token*top_k + rank) routed to expert e; tile_prefix is the running count
// of 64-row GEMM tiles per bucket.  Pairs with invalid expert ids are
// dropped (the combine kernel re-checks validity).
[[max_total_threads_per_threadgroup(256)]] kernel void uocr_moe_prefill_bucket_pairs(
    device const uint *top_expert_ids [[buffer(0)]],
    device uint *pair_rows [[buffer(1)]],
    device uint *expert_offsets [[buffer(2)]],
    device uint *tile_prefix [[buffer(3)]],
    constant UocrMoeBucketQ8Params &params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]) {
    threadgroup atomic_uint counts[64];
    threadgroup atomic_uint cursors[64];
    const uint expert_count = min(params.expert_count, 64u);
    const uint pair_count = params.n_tokens * params.top_k;

    for (uint e = tid; e < expert_count; e += ntg) {
        atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = tid; p < pair_count; p += ntg) {
        const uint expert = top_expert_ids[p];
        if (expert < expert_count) {
            atomic_fetch_add_explicit(&counts[expert], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint offset = 0u;
        uint tiles = 0u;
        for (uint e = 0u; e < expert_count; ++e) {
            const uint count = atomic_load_explicit(&counts[e], memory_order_relaxed);
            expert_offsets[e] = offset;
            tile_prefix[e] = tiles;
            atomic_store_explicit(&cursors[e], offset, memory_order_relaxed);
            offset += count;
            tiles += (count + params.rows_per_tile - 1u) / params.rows_per_tile;
        }
        expert_offsets[expert_count] = offset;
        tile_prefix[expert_count] = tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = tid; p < pair_count; p += ntg) {
        const uint expert = top_expert_ids[p];
        if (expert < expert_count) {
            const uint pos = atomic_fetch_add_explicit(&cursors[expert], 1u, memory_order_relaxed);
            pair_rows[pos] = p;
        }
    }
}

// Map a GEMM tile slot to (expert, first bucket position) via tile_prefix.
// Returns false for overdispatched slots past the last tile.
static inline bool uocr_moe_bucket_tile_lookup(device const uint *tile_prefix,
                                               device const uint *expert_offsets,
                                               uint expert_count,
                                               uint tile_slot,
                                               uint rows_per_tile,
                                               thread uint &out_expert,
                                               thread uint &out_bucket_start,
                                               thread uint &out_bucket_rows) {
    if (tile_slot >= tile_prefix[expert_count]) {
        return false;
    }
    uint lo = 0u;
    uint hi = expert_count;
    while (lo + 1u < hi) {
        const uint mid_e = (lo + hi) / 2u;
        if (tile_prefix[mid_e] <= tile_slot) {
            lo = mid_e;
        } else {
            hi = mid_e;
        }
    }
    const uint local_tile = tile_slot - tile_prefix[lo];
    out_expert = lo;
    out_bucket_start = expert_offsets[lo] + local_tile * rows_per_tile;
    out_bucket_rows = expert_offsets[lo + 1u] - expert_offsets[lo] - local_tile * rows_per_tile;
    return true;
}

// Stage the tile's pair ids once per threadgroup (UINT_MAX marks padding).
static inline void uocr_moe_bucket_stage_rows(device const uint *pair_rows,
                                              uint bucket_start,
                                              uint bucket_rows,
                                              uint tid,
                                              threadgroup uint *tile_pairs) {
    if (tid < UOCR_GEMM_Q8_BM) {
        tile_pairs[tid] = tid < bucket_rows ? pair_rows[bucket_start + tid] : 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Stage a gathered A tile: source row = tile_pairs[i] / row_divisor
// (divisor top_k gathers token rows from src; divisor 1 gathers pair rows
// from mid).  Padding rows are zero-filled.
static inline void uocr_gemm_q8_stage_a_gathered(device const half *src,
                                                 uint K,
                                                 uint k_base,
                                                 uint row_divisor,
                                                 threadgroup const uint *tile_pairs,
                                                 uint tid,
                                                 threadgroup half *sa) {
    for (uint i = tid; i < (UOCR_GEMM_Q8_BM * UOCR_GEMM_Q8_BK) / UOCR_HALF4_WIDTH; i += UOCR_GEMM_Q8_THREADS) {
        const uint chunks_per_row = UOCR_GEMM_Q8_BK / UOCR_HALF4_WIDTH;
        const uint mrow = i / chunks_per_row;
        const uint kk = (i - mrow * chunks_per_row) * UOCR_HALF4_WIDTH;
        const uint pair = tile_pairs[mrow];
        const half4 value = pair != 0xFFFFFFFFu
                                ? uocr_load_half4(src, (ulong)(pair / row_divisor) * K + k_base + kk)
                                : uocr_zero_half4();
        *reinterpret_cast<threadgroup half4 *>(sa + mrow * UOCR_GEMM_Q8_BK + kk) = value;
    }
}

// Bucketed gate/up GEMM: mid[pair, col] = silu(src_t @ Wg_e^T) * (src_t @ Wu_e^T)
// for every pair (t, e) in the tile's bucket.
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_moe_prefill_bucketed_swiglu_gate_up_gemm_q8_0_to_f16(
    device const half *src [[buffer(0)]],
    device const uint *pair_rows [[buffer(1)]],
    device const uint *expert_offsets [[buffer(2)]],
    device const uint *tile_prefix [[buffer(3)]],
    device const char *expert_qweight [[buffer(4)]],
    device const half *expert_qscale [[buffer(5)]],
    device half *mid [[buffer(6)]],
    constant UocrMoeBucketedGemmQ8Params &params [[buffer(7)]],
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
    device const char *gate_qweight = expert_qweight + (ulong)expert * params.expert_stride_values;
    device const char *up_qweight = gate_qweight + params.up_offset_values;
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
        uocr_gemm_q8_stage_b(gate_qweight, gate_qscale, K, groups_per_row, col_base, k_base, tid, sb_gate);
        uocr_gemm_q8_stage_b(up_qweight, up_qscale, K, groups_per_row, col_base, k_base, tid, sb_up);
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
[[max_total_threads_per_threadgroup(UOCR_GEMM_Q8_THREADS)]] kernel void uocr_moe_prefill_bucketed_down_gemm_q8_0_to_f16(
    device const half *mid [[buffer(0)]],
    device const uint *pair_rows [[buffer(1)]],
    device const uint *expert_offsets [[buffer(2)]],
    device const uint *tile_prefix [[buffer(3)]],
    device const char *expert_qweight [[buffer(4)]],
    device const half *expert_qscale [[buffer(5)]],
    device half *down_out [[buffer(6)]],
    constant UocrMoeBucketedGemmQ8Params &params [[buffer(7)]],
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
    device const char *down_qweight = expert_qweight + (ulong)expert * params.expert_stride_values + params.down_offset_values;
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
        uocr_gemm_q8_stage_b(down_qweight, down_qscale, K, groups_per_row, col_base, k_base, tid, sb);
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

// Weighted combine: dst[t, c] = shared + residual + sum_rank w[t,rank] * down_out[pair, c].
kernel void uocr_moe_prefill_bucketed_combine_f16(
    device const half *down_out [[buffer(0)]],
    device const uint *top_expert_ids [[buffer(1)]],
    device const float *top_weights [[buffer(2)]],
    device const half *shared [[buffer(3)]],
    device const half *residual [[buffer(4)]],
    device half *dst [[buffer(5)]],
    constant UocrMoeBucketedGemmQ8Params &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint hidden_size = params.hidden_size;
    const uint token = gid / hidden_size;
    const uint col = gid - token * hidden_size;
    if (token >= params.n_tokens) {
        return;
    }
    float sum = 0.0f;
    for (uint rank = 0u; rank < params.top_k; ++rank) {
        const uint pair = token * params.top_k + rank;
        if (top_expert_ids[pair] < params.expert_count) {
            sum += top_weights[pair] * float(down_out[(ulong)pair * hidden_size + col]);
        }
    }
    dst[gid] = half(sum + float(shared[gid]) + float(residual[gid]));
}
