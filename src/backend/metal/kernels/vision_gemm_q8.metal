// vision_gemm_q8.metal - Tiled fused Q8_0 GEMM for the vision encoders.
//
// Vision Q8 call sites (projector, CLIP QKV/O/fc1/fc2, SAM lin1/lin2) are
// multi-row GEMMs (hundreds to thousands of token rows).  A GEMV-style kernel
// re-reads the full weight matrix for every row; this kernel dequantizes each
// Q8 weight tile once per 64-row block into threadgroup memory and uses
// simdgroup fp16 MMA with fp32 accumulators.
//
// Requires uocr_quickgelu (dense.metal) and uocr_gelu_erf (sam.metal) in the
// combined translation unit.
#include "common.metal"

#define UOCR_VISION_GEMM_BM 64u
#define UOCR_VISION_GEMM_BN 32u
#define UOCR_VISION_GEMM_BK 32u
#define UOCR_VISION_GEMM_THREADS 128u

#define UOCR_VISION_Q8_ACT_NONE 0u
#define UOCR_VISION_Q8_ACT_QUICKGELU 1u
#define UOCR_VISION_Q8_ACT_GELU_ERF 2u

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

// C[64x32] = A[64xK] @ W[32xK]^T with bias + activation epilogue.
// 128 threads = 4 simdgroups in a 2x2 grid of 32x16 sub-tiles; each simdgroup
// holds 4x2 8x8 fp32 accumulators.
[[max_total_threads_per_threadgroup(UOCR_VISION_GEMM_THREADS)]] kernel void uocr_vision_gemm_q8_0_to_f16(
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
    threadgroup float smem[UOCR_VISION_GEMM_BM * UOCR_VISION_GEMM_BN];
    threadgroup half *sa = (threadgroup half *)smem;
    threadgroup half *sb = sa + UOCR_VISION_GEMM_BM * UOCR_VISION_GEMM_BK;
    threadgroup float *sc = smem;

    const uint K = params.k;
    const uint row_base = tgpig.y * UOCR_VISION_GEMM_BM;
    const uint col_base = tgpig.x * UOCR_VISION_GEMM_BN;

    simdgroup_float8x8 mc[4][2];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sg_row = (sgitg / 2u) * 32u; // 0 or 32
    const uint sg_col = (sgitg % 2u) * 16u; // 0 or 16

    // Weight-tile dequant assignment: one (column, 8-wide k run) per thread.
    // Aligned 8-runs never cross a 64-element scale group.
    const uint wn = tid / 4u;          // 0..31: column within tile
    const uint wk0 = (tid % 4u) * 8u;  // 0,8,16,24: k offset within tile
    const uint wcol = col_base + wn;
    const uint wscale_base = wcol * params.groups_per_row;

    for (uint k_base = 0u; k_base < K; k_base += UOCR_VISION_GEMM_BK) {
        // Stage A tile [BM][BK] as half4 chunks; zero-fill rows past m.
        for (uint i = tid; i < (UOCR_VISION_GEMM_BM * UOCR_VISION_GEMM_BK) / UOCR_HALF4_WIDTH; i += UOCR_VISION_GEMM_THREADS) {
            const uint chunks_per_row = UOCR_VISION_GEMM_BK / UOCR_HALF4_WIDTH;
            const uint mrow = i / chunks_per_row;
            const uint kk = (i - mrow * chunks_per_row) * UOCR_HALF4_WIDTH;
            const uint row = row_base + mrow;
            const half4 value = row < params.m ? uocr_load_half4(src, (ulong)row * K + k_base + kk) : uocr_zero_half4();
            *reinterpret_cast<threadgroup half4 *>(sa + mrow * UOCR_VISION_GEMM_BK + kk) = value;
        }
        // Dequantize W tile transposed to [BK][BN] (k-major for B-side MMA loads).
        {
            const uint k0 = k_base + wk0;
            const float scale = float(qscale[wscale_base + k0 / 64u]);
            device const char *w = qweight + wcol * K + k0;
            for (uint kk = 0u; kk < 8u; ++kk) {
                sb[(wk0 + kk) * UOCR_VISION_GEMM_BN + wn] = half(float(int(w[kk])) * scale);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < UOCR_VISION_GEMM_BK; kk += 8u) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            for (uint i = 0u; i < 4u; ++i) {
                simdgroup_load(ma[i], sa + (sg_row + i * 8u) * UOCR_VISION_GEMM_BK + kk, UOCR_VISION_GEMM_BK);
            }
            for (uint j = 0u; j < 2u; ++j) {
                simdgroup_load(mb[j], sb + kk * UOCR_VISION_GEMM_BN + sg_col + j * 8u, UOCR_VISION_GEMM_BN);
            }
            for (uint i = 0u; i < 4u; ++i) {
                for (uint j = 0u; j < 2u; ++j) {
                    simdgroup_multiply_accumulate(mc[i][j], ma[i], mb[j], mc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(mc[i][j], sc + (sg_row + i * 8u) * UOCR_VISION_GEMM_BN + sg_col + j * 8u, UOCR_VISION_GEMM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: bias + activation + bounds-checked (optionally split) store.
    for (uint i = tid; i < UOCR_VISION_GEMM_BM * UOCR_VISION_GEMM_BN; i += UOCR_VISION_GEMM_THREADS) {
        const uint mrow = i / UOCR_VISION_GEMM_BN;
        const uint n = i - mrow * UOCR_VISION_GEMM_BN;
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
            dst0[row * params.n + col] = half(value);
        } else {
            const uint projection = col / params.split_size;
            const uint local_col = col - projection * params.split_size;
            device half *dst = projection == 0u ? dst0 : (projection == 1u ? dst1 : dst2);
            dst[row * params.split_size + local_col] = half(value);
        }
    }
}
