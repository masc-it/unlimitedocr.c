// common.metal - Shared includes, constants, helpers (static inline)
// Extracted from uocr_smoke.metal
//
#include <metal_stdlib>
#ifndef UOCR_METAL_ENABLE_MPP_TENSOROPS
#define UOCR_METAL_ENABLE_MPP_TENSOROPS 0
#endif
#if UOCR_METAL_ENABLE_MPP_TENSOROPS
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

// Shared infrastructure (function constants, helpers, threadgroup reductions)
#define UOCR_HALF4_WIDTH 4u
#define UOCR_FLASH_Q_PER_TG 4u
#define UOCR_FLASH_MAX_LANE_VALUES 4u
#define UOCR_FLASH_NEG_INF (-3.4028234663852886e38f)
#define UOCR_FC_HIDDEN_SIZE 0
#define UOCR_FC_ATTENTION_HEADS 1
#define UOCR_FC_HEAD_DIM 2
#define UOCR_FC_RING_WINDOW 3
#define UOCR_FC_MOE_EXPERTS 4
#define UOCR_FC_MOE_TOP_K 5
#define UOCR_FC_RMS_EPS 6
#define UOCR_FC_ROPE_FREQ_SCALE 7
#define UOCR_FC_ATTENTION_SCALE 8
#define UOCR_FC_ATTENTION_PROJECTION_COUNT 9
#define UOCR_FC_VOCAB_SIZE 10
#define UOCR_FC_LM_HEAD_TILE_TOKENS 11
#define UOCR_FC_LM_HEAD_LANES_PER_TOKEN 12
#define UOCR_FC_MOE_EXPERT_INTERMEDIATE 13
#define UOCR_FC_MOE_SHARED_INTERMEDIATE 14

constant uint uocr_fc_hidden_size [[function_constant(UOCR_FC_HIDDEN_SIZE)]];
constant uint uocr_fc_attention_heads [[function_constant(UOCR_FC_ATTENTION_HEADS)]];
constant uint uocr_fc_head_dim [[function_constant(UOCR_FC_HEAD_DIM)]];
constant uint uocr_fc_ring_window [[function_constant(UOCR_FC_RING_WINDOW)]];
constant uint uocr_fc_moe_experts [[function_constant(UOCR_FC_MOE_EXPERTS)]];
constant uint uocr_fc_moe_top_k [[function_constant(UOCR_FC_MOE_TOP_K)]];
constant float uocr_fc_rms_eps [[function_constant(UOCR_FC_RMS_EPS)]];
constant float uocr_fc_rope_freq_scale [[function_constant(UOCR_FC_ROPE_FREQ_SCALE)]];
constant float uocr_fc_attention_scale [[function_constant(UOCR_FC_ATTENTION_SCALE)]];
constant uint uocr_fc_attention_projection_count [[function_constant(UOCR_FC_ATTENTION_PROJECTION_COUNT)]];
constant uint uocr_fc_vocab_size [[function_constant(UOCR_FC_VOCAB_SIZE)]];
constant uint uocr_fc_lm_head_tile_tokens [[function_constant(UOCR_FC_LM_HEAD_TILE_TOKENS)]];
constant uint uocr_fc_lm_head_lanes_per_token [[function_constant(UOCR_FC_LM_HEAD_LANES_PER_TOKEN)]];
constant uint uocr_fc_moe_expert_intermediate [[function_constant(UOCR_FC_MOE_EXPERT_INTERMEDIATE)]];
constant uint uocr_fc_moe_shared_intermediate [[function_constant(UOCR_FC_MOE_SHARED_INTERMEDIATE)]];

static inline uint uocr_fc_hidden_size_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_hidden_size) ? uocr_fc_hidden_size : runtime_value;
}

static inline uint uocr_fc_attention_heads_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_attention_heads) ? uocr_fc_attention_heads : runtime_value;
}

static inline uint uocr_fc_head_dim_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_head_dim) ? uocr_fc_head_dim : runtime_value;
}

static inline uint uocr_fc_ring_window_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_ring_window) ? uocr_fc_ring_window : runtime_value;
}

static inline uint uocr_fc_moe_experts_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_moe_experts) ? uocr_fc_moe_experts : runtime_value;
}

static inline uint uocr_fc_moe_top_k_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_moe_top_k) ? uocr_fc_moe_top_k : runtime_value;
}

static inline uint uocr_fc_moe_expert_intermediate_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_moe_expert_intermediate) ? uocr_fc_moe_expert_intermediate : runtime_value;
}

static inline uint uocr_fc_moe_shared_intermediate_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_moe_shared_intermediate) ? uocr_fc_moe_shared_intermediate : runtime_value;
}

static inline float uocr_fc_rms_eps_or(float runtime_value) {
    return is_function_constant_defined(uocr_fc_rms_eps) ? uocr_fc_rms_eps : runtime_value;
}

static inline float uocr_fc_rope_freq_scale_or(float runtime_value) {
    return is_function_constant_defined(uocr_fc_rope_freq_scale) ? uocr_fc_rope_freq_scale : runtime_value;
}

static inline float uocr_fc_attention_scale_or(float runtime_value) {
    return is_function_constant_defined(uocr_fc_attention_scale) ? uocr_fc_attention_scale : runtime_value;
}

static inline uint uocr_fc_attention_projection_count_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_attention_projection_count) ? uocr_fc_attention_projection_count : runtime_value;
}

static inline uint uocr_fc_vocab_size_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_vocab_size) ? uocr_fc_vocab_size : runtime_value;
}

static inline uint uocr_fc_lm_head_tile_tokens_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_lm_head_tile_tokens) ? uocr_fc_lm_head_tile_tokens : runtime_value;
}

static inline uint uocr_fc_lm_head_lanes_per_token_or(uint runtime_value) {
    return is_function_constant_defined(uocr_fc_lm_head_lanes_per_token) ? uocr_fc_lm_head_lanes_per_token : runtime_value;
}

static inline uint uocr_div_up_u32(uint value, uint divisor) {
    return (value + divisor - 1u) / divisor;
}

static inline half4 uocr_load_half4(device const half *ptr, ulong index) {
    return half4(*reinterpret_cast<device const packed_half4 *>(ptr + index));
}

static inline void uocr_store_half4(device half *ptr, ulong index, half4 value) {
    *reinterpret_cast<device packed_half4 *>(ptr + index) = packed_half4(value);
}

static inline half4 uocr_zero_half4() {
    return half4(half(0.0h), half(0.0h), half(0.0h), half(0.0h));
}

/* Lane-local fp16 GEMV dot for decode kernels.  One simdgroup owns one
 * output row; each lane processes half4 chunks and the caller reduces with
 * simd_sum.  All decode K dimensions in Unlimited-OCR are multiples of 4. */
static inline float uocr_decode_gemv_f16_lane_dot(device const half *x,
                                                  device const half *w_row,
                                                  uint k,
                                                  uint lane,
                                                  uint simd_width) {
    float sum = 0.0f;
    const uint chunks = k / UOCR_HALF4_WIDTH;
    for (uint c = lane; c < chunks; c += simd_width) {
        const uint kk = c * UOCR_HALF4_WIDTH;
        const half4 xv = uocr_load_half4(x, ulong(kk));
        const half4 wv = uocr_load_half4(w_row, ulong(kk));
        sum += float(xv.x) * float(wv.x) + float(xv.y) * float(wv.y) +
               float(xv.z) * float(wv.z) + float(xv.w) * float(wv.w);
    }
    return sum;
}

static inline uint uocr_flash_lane_dim(uint lane, uint component, uint simd_width) {
    return lane + component * simd_width;
}

static inline uint uocr_simd_lane_from_tid(uint tid, uint simd_width) {
    return simd_width == 0u ? tid : tid - (tid / simd_width) * simd_width;
}

static inline uint uocr_simdgroup_from_tid(uint tid, uint simd_width) {
    return simd_width == 0u ? 0u : tid / simd_width;
}

static inline uint uocr_simdgroups_for_threadgroup(uint ntg, uint simd_width) {
    return simd_width == 0u ? 1u : (ntg + simd_width - 1u) / simd_width;
}

static inline float uocr_threadgroup_sum(float value,
                                         threadgroup float *partials,
                                         uint tid,
                                         uint ntg,
                                         uint simd_width) {
    const uint lane = uocr_simd_lane_from_tid(tid, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(tid, simd_width);
    const uint simdgroups = uocr_simdgroups_for_threadgroup(ntg, simd_width);
    const float simd_total = simd_sum(value);
    if (lane == 0u) {
        partials[simdgroup] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = simdgroups;
    while (active > 1u) {
        const uint upper = (active + 1u) >> 1u;
        if (tid < active - upper) {
            partials[tid] += partials[tid + upper];
        }
        active = upper;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partials[0];
}

static inline float uocr_threadgroup_max(float value,
                                         threadgroup float *partials,
                                         uint tid,
                                         uint ntg,
                                         uint simd_width) {
    const uint lane = uocr_simd_lane_from_tid(tid, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(tid, simd_width);
    const uint simdgroups = uocr_simdgroups_for_threadgroup(ntg, simd_width);
    const float simd_total = simd_max(value);
    if (lane == 0u) {
        partials[simdgroup] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = simdgroups;
    while (active > 1u) {
        const uint upper = (active + 1u) >> 1u;
        if (tid < active - upper) {
            partials[tid] = max(partials[tid], partials[tid + upper]);
        }
        active = upper;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partials[0];
}

static inline void uocr_threadgroup_sum2(float value0,
                                         float value1,
                                         threadgroup float *partials0,
                                         threadgroup float *partials1,
                                         uint tid,
                                         uint ntg,
                                         uint simd_width,
                                         thread float &out0,
                                         thread float &out1) {
    const uint lane = uocr_simd_lane_from_tid(tid, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(tid, simd_width);
    const uint simdgroups = uocr_simdgroups_for_threadgroup(ntg, simd_width);
    const float simd_total0 = simd_sum(value0);
    const float simd_total1 = simd_sum(value1);
    if (lane == 0u) {
        partials0[simdgroup] = simd_total0;
        partials1[simdgroup] = simd_total1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = simdgroups;
    while (active > 1u) {
        const uint upper = (active + 1u) >> 1u;
        if (tid < active - upper) {
            partials0[tid] += partials0[tid + upper];
            partials1[tid] += partials1[tid + upper];
        }
        active = upper;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out0 = partials0[0];
    out1 = partials1[0];
}
