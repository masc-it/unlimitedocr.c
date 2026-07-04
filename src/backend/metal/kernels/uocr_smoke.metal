#include <metal_stdlib>
#ifndef UOCR_METAL_ENABLE_MPP_TENSOROPS
#define UOCR_METAL_ENABLE_MPP_TENSOROPS 0
#endif
#if UOCR_METAL_ENABLE_MPP_TENSOROPS
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
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





#define UOCR_HALF4_WIDTH 4u
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

kernel void uocr_get_rows_f16_to_f16(device const half *table [[buffer(0)]],
                                     device const int *row_ids [[buffer(1)]],
                                     device half *dst [[buffer(2)]],
                                     constant UocrGetRowsParams &params [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint out_row = gid.y;
    if (col >= params.row_width || out_row >= params.n_row_ids) {
        return;
    }
    const ulong dst_base = (ulong(out_row) * ulong(params.row_width)) + ulong(col);
    const int row = row_ids[out_row];
    const bool full4 = col + (UOCR_HALF4_WIDTH - 1u) < params.row_width;
    if (row < 0 || (uint)row >= params.table_rows) {
        if (full4) {
            uocr_store_half4(dst, dst_base, uocr_zero_half4());
        } else {
            for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.row_width; ++i) {
                dst[dst_base + ulong(i)] = half(0.0f);
            }
        }
        return;
    }
    const ulong src_base = (ulong(uint(row)) * ulong(params.row_width)) + ulong(col);
    if (full4) {
        uocr_store_half4(dst, dst_base, uocr_load_half4(table, src_base));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.row_width; ++i) {
            dst[dst_base + ulong(i)] = table[src_base + ulong(i)];
        }
    }
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
    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint token = gid.y;
    if (col >= params.hidden_size || token >= params.n_tokens) {
        return;
    }
    const ulong dst_base = (ulong(token) * ulong(params.hidden_size)) + ulong(col);
    const bool full4 = col + (UOCR_HALF4_WIDTH - 1u) < params.hidden_size;
    const int row = input_ids[token];
    if (row < 0 || (uint)row >= params.table_rows) {
        if (full4) {
            uocr_store_half4(dst, dst_base, uocr_zero_half4());
        } else {
            for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.hidden_size; ++i) {
                dst[dst_base + ulong(i)] = half(0.0f);
            }
        }
        return;
    }
    const ulong src_base = (ulong(uint(row)) * ulong(params.hidden_size)) + ulong(col);
    if (full4) {
        uocr_store_half4(dst, dst_base, uocr_load_half4(embedding_table, src_base));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.hidden_size; ++i) {
            dst[dst_base + ulong(i)] = embedding_table[src_base + ulong(i)];
        }
    }
}

kernel void uocr_assemble_prompt_text_skip_image_f16(device const half *embedding_table [[buffer(0)]],
                                                     device const int *input_ids [[buffer(1)]],
                                                     device half *dst [[buffer(2)]],
                                                     constant UocrPromptAssemblyParams &params [[buffer(3)]],
                                                     uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint token = gid.y;
    if (col >= params.hidden_size || token >= params.n_tokens) {
        return;
    }
    const uint image_span_end = params.image_span_start + params.image_span_length;
    if (token >= params.image_span_start && token < image_span_end) {
        return;
    }
    const ulong dst_base = (ulong(token) * ulong(params.hidden_size)) + ulong(col);
    const bool full4 = col + (UOCR_HALF4_WIDTH - 1u) < params.hidden_size;
    const int row = input_ids[token];
    if (row < 0 || (uint)row >= params.table_rows) {
        if (full4) {
            uocr_store_half4(dst, dst_base, uocr_zero_half4());
        } else {
            for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.hidden_size; ++i) {
                dst[dst_base + ulong(i)] = half(0.0f);
            }
        }
        return;
    }
    const ulong src_base = (ulong(uint(row)) * ulong(params.hidden_size)) + ulong(col);
    if (full4) {
        uocr_store_half4(dst, dst_base, uocr_load_half4(embedding_table, src_base));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.hidden_size; ++i) {
            dst[dst_base + ulong(i)] = embedding_table[src_base + ulong(i)];
        }
    }
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

struct UocrVisualFormatGlobalParams {
    uint hidden_size;
    uint grid_size;
    uint visual_tokens_per_view;
    uint view_count;
    uint dst_token_base;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_format_global_visual_rows_f16(device const half *projected_rows [[buffer(0)]],
                                               device const half *image_newline [[buffer(1)]],
                                               device const half *view_separator [[buffer(2)]],
                                               device half *dst_rows [[buffer(3)]],
                                               constant UocrVisualFormatGlobalParams &params [[buffer(4)]],
                                               uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint visual_row_linear = gid.y;
    if (col >= params.hidden_size || visual_row_linear >= params.view_count * params.visual_tokens_per_view) {
        return;
    }
    const uint grid = params.grid_size;
    const uint grid_tokens = grid * grid;
    const uint view_index = visual_row_linear / params.visual_tokens_per_view;
    const uint visual_row = visual_row_linear - view_index * params.visual_tokens_per_view;
    const bool full4 = col + (UOCR_HALF4_WIDTH - 1u) < params.hidden_size;
    device const half *src = image_newline;
    ulong src_base = ulong(col);
    if (visual_row == grid * (grid + 1u)) {
        src = view_separator;
    } else {
        const uint row_in_view = visual_row / (grid + 1u);
        const uint col_in_view = visual_row - row_in_view * (grid + 1u);
        if (col_in_view != grid) {
            const uint src_row = view_index * grid_tokens + row_in_view * grid + col_in_view;
            src = projected_rows;
            src_base = (ulong(src_row) * ulong(params.hidden_size)) + ulong(col);
        }
    }
    const uint dst_row = params.dst_token_base + visual_row_linear;
    const ulong dst_base = (ulong(dst_row) * ulong(params.hidden_size)) + ulong(col);
    if (full4) {
        uocr_store_half4(dst_rows, dst_base, uocr_load_half4(src, src_base));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.hidden_size; ++i) {
            dst_rows[dst_base + ulong(i)] = src[src_base + ulong(i)];
        }
    }
}

struct UocrVisualFormatLocalParams {
    uint hidden_size;
    uint grid_size;
    uint crop_grid_w;
    uint chunk_first_view;
    uint chunk_view_count;
    uint dst_token_base;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_fill_local_visual_newlines_f16(device const half *image_newline [[buffer(0)]],
                                                device half *dst_rows [[buffer(1)]],
                                                constant UocrVisualFormatLocalParams &params [[buffer(2)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint local_row = gid.y;
    if (col >= params.hidden_size || params.crop_grid_w == 0u || local_row >= params.chunk_view_count * params.grid_size) {
        return;
    }
    const uint stitched_row_stride = params.crop_grid_w * params.grid_size + 1u;
    const uint dst_row = params.dst_token_base + local_row * stitched_row_stride + params.crop_grid_w * params.grid_size;
    const ulong dst_base = (ulong(dst_row) * ulong(params.hidden_size)) + ulong(col);
    const bool full4 = col + (UOCR_HALF4_WIDTH - 1u) < params.hidden_size;
    if (full4) {
        uocr_store_half4(dst_rows, dst_base, uocr_load_half4(image_newline, ulong(col)));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.hidden_size; ++i) {
            dst_rows[dst_base + ulong(i)] = image_newline[col + i];
        }
    }
}

kernel void uocr_format_local_visual_rows_f16(device const half *projected_rows [[buffer(0)]],
                                              device half *dst_rows [[buffer(1)]],
                                              constant UocrVisualFormatLocalParams &params [[buffer(2)]],
                                              uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint src_row = gid.y;
    const uint grid = params.grid_size;
    const uint tokens_per_view = grid * grid;
    if (col >= params.hidden_size || tokens_per_view == 0u || src_row >= params.chunk_view_count * tokens_per_view ||
        params.crop_grid_w == 0u) {
        return;
    }
    const uint chunk_view_index = src_row / tokens_per_view;
    const uint token_in_view = src_row - chunk_view_index * tokens_per_view;
    const uint local_view = params.chunk_first_view + chunk_view_index;
    const uint crop_y = local_view / params.crop_grid_w;
    const uint crop_x = local_view - crop_y * params.crop_grid_w;
    const uint row_in_view = token_in_view / grid;
    const uint col_in_view = token_in_view - row_in_view * grid;
    const uint stitched_row_stride = params.crop_grid_w * grid + 1u;
    const uint dst_row = params.dst_token_base +
                         (crop_y * grid + row_in_view) * stitched_row_stride +
                         crop_x * grid + col_in_view;
    const ulong src_base = (ulong(src_row) * ulong(params.hidden_size)) + ulong(col);
    const ulong dst_base = (ulong(dst_row) * ulong(params.hidden_size)) + ulong(col);
    const bool full4 = col + (UOCR_HALF4_WIDTH - 1u) < params.hidden_size;
    if (full4) {
        uocr_store_half4(dst_rows, dst_base, uocr_load_half4(projected_rows, src_base));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.hidden_size; ++i) {
            dst_rows[dst_base + ulong(i)] = projected_rows[src_base + ulong(i)];
        }
    }
}

struct UocrSamPatchEmbedParams {
    uint width;
    uint height;
    uint out_width;
    uint out_height;
    uint has_bias;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline uint uocr_sam_patch_weight_index(uint out_channel, uint in_channel, uint ky, uint kx) {
    return (((out_channel * 3u + in_channel) * 16u + ky) * 16u + kx);
}

kernel void uocr_sam_patch_embed_f16_input(device const half *pixels [[buffer(0)]],
                                           device const half *weight [[buffer(1)]],
                                           device const half *bias [[buffer(2)]],
                                           device half *dst_bhwc [[buffer(3)]],
                                           constant UocrSamPatchEmbedParams &params [[buffer(4)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint3 block [[threadgroup_position_in_grid]],
                                           uint3 tid3 [[thread_position_in_threadgroup]],
                                           uint3 ntg3 [[threads_per_threadgroup]],
                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint tid = tid3.x;
    const uint ntg = ntg3.x;
    const uint out_channel = block.x;
    const uint patch_index = block.y;
    const uint batch_index = block.z;
    const uint patch_count = params.out_width * params.out_height;
    if (out_channel >= 768u || patch_index >= patch_count) {
        return;
    }

    const uint patch_y = patch_index / params.out_width;
    const uint patch_x = patch_index - patch_y * params.out_width;
    const uint input_y0 = patch_y * 16u;
    const uint input_x0 = patch_x * 16u;

    float acc = 0.0f;
    for (uint linear = tid; linear < 3u * 16u * 16u; linear += ntg) {
        const uint c = linear / (16u * 16u);
        const uint rem = linear - c * 16u * 16u;
        const uint ky = rem / 16u;
        const uint kx = rem - ky * 16u;
        const uint pixel_index = batch_index * (3u * params.height * params.width) +
                                 c * params.height * params.width +
                                 (input_y0 + ky) * params.width + input_x0 + kx;
        const float x = float(pixels[pixel_index]);
        const float w = float(weight[uocr_sam_patch_weight_index(out_channel, c, ky, kx)]);
        acc += x * w;
    }
    const float total = uocr_threadgroup_sum(acc, partials, tid, ntg, simd_width);
    if (tid == 0u) {
        const float value = total + (params.has_bias != 0u ? float(bias[out_channel]) : 0.0f);
        dst_bhwc[(batch_index * patch_count + patch_index) * 768u + out_channel] = half(value);
    }
}

kernel void uocr_sam_patch_embed_f32_input(device const float *pixels [[buffer(0)]],
                                           device const half *weight [[buffer(1)]],
                                           device const half *bias [[buffer(2)]],
                                           device half *dst_bhwc [[buffer(3)]],
                                           constant UocrSamPatchEmbedParams &params [[buffer(4)]],
                                           threadgroup float *partials [[threadgroup(0)]],
                                           uint3 block [[threadgroup_position_in_grid]],
                                           uint3 tid3 [[thread_position_in_threadgroup]],
                                           uint3 ntg3 [[threads_per_threadgroup]],
                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint tid = tid3.x;
    const uint ntg = ntg3.x;
    const uint out_channel = block.x;
    const uint patch_index = block.y;
    const uint batch_index = block.z;
    const uint patch_count = params.out_width * params.out_height;
    if (out_channel >= 768u || patch_index >= patch_count) {
        return;
    }

    const uint patch_y = patch_index / params.out_width;
    const uint patch_x = patch_index - patch_y * params.out_width;
    const uint input_y0 = patch_y * 16u;
    const uint input_x0 = patch_x * 16u;

    float acc = 0.0f;
    for (uint linear = tid; linear < 3u * 16u * 16u; linear += ntg) {
        const uint c = linear / (16u * 16u);
        const uint rem = linear - c * 16u * 16u;
        const uint ky = rem / 16u;
        const uint kx = rem - ky * 16u;
        const uint pixel_index = batch_index * (3u * params.height * params.width) +
                                 c * params.height * params.width +
                                 (input_y0 + ky) * params.width + input_x0 + kx;
        const float x = pixels[pixel_index];
        const float w = float(weight[uocr_sam_patch_weight_index(out_channel, c, ky, kx)]);
        acc += x * w;
    }
    const float total = uocr_threadgroup_sum(acc, partials, tid, ntg, simd_width);
    if (tid == 0u) {
        const float value = total + (params.has_bias != 0u ? float(bias[out_channel]) : 0.0f);
        dst_bhwc[(batch_index * patch_count + patch_index) * 768u + out_channel] = half(value);
    }
}

struct UocrSamAbsPosParams {
    uint source_grid;
    uint target_width;
    uint target_height;
    uint channels;
    uint batch_size;
};

static inline float uocr_cubic_convolution_weight(float x) {
    constexpr float a = -0.75f;
    const float ax = fabs(x);
    if (ax < 1.0f) {
        return ((a + 2.0f) * ax - (a + 3.0f)) * ax * ax + 1.0f;
    }
    if (ax < 2.0f) {
        return (((a * ax - 5.0f * a) * ax + 8.0f * a) * ax - 4.0f * a);
    }
    return 0.0f;
}

static inline float uocr_sam_abs_pos_axis_weight(int source_index, float center, float filter_scale) {
    return uocr_cubic_convolution_weight((float(source_index) - center) / filter_scale);
}

static float uocr_sam_abs_pos_bicubic_antialias(device const half *pos_embed,
                                                constant UocrSamAbsPosParams &params,
                                                uint out_y,
                                                uint out_x,
                                                uint channel) {
    if (params.target_width == params.source_grid && params.target_height == params.source_grid) {
        return float(pos_embed[((out_y * params.source_grid + out_x) * params.channels) + channel]);
    }

    const float scale_y = float(params.source_grid) / float(params.target_height);
    const float scale_x = float(params.source_grid) / float(params.target_width);
    const float filter_scale_y = max(scale_y, 1.0f);
    const float filter_scale_x = max(scale_x, 1.0f);
    const float support_y = 2.0f * filter_scale_y;
    const float support_x = 2.0f * filter_scale_x;
    const float center_y = (float(out_y) + 0.5f) * scale_y - 0.5f;
    const float center_x = (float(out_x) + 0.5f) * scale_x - 0.5f;

    const int src_grid = int(params.source_grid);
    const int y0 = max(0, int(floor(center_y - support_y)));
    const int y1 = min(src_grid - 1, int(ceil(center_y + support_y)));
    const int x0 = max(0, int(floor(center_x - support_x)));
    const int x1 = min(src_grid - 1, int(ceil(center_x + support_x)));

    float acc = 0.0f;
    float weight_sum = 0.0f;
    for (int iy = y0; iy <= y1; ++iy) {
        const float wy = uocr_sam_abs_pos_axis_weight(iy, center_y, filter_scale_y);
        for (int ix = x0; ix <= x1; ++ix) {
            const float wx = uocr_sam_abs_pos_axis_weight(ix, center_x, filter_scale_x);
            const float w = wy * wx;
            acc += float(pos_embed[((uint(iy) * params.source_grid + uint(ix)) * params.channels) + channel]) * w;
            weight_sum += w;
        }
    }
    return weight_sum != 0.0f ? acc / weight_sum : 0.0f;
}

kernel void uocr_sam_add_abs_pos_f16(device const half *patch_bhwc [[buffer(0)]],
                                     device const half *pos_embed [[buffer(1)]],
                                     device half *dst_bhwc [[buffer(2)]],
                                     constant UocrSamAbsPosParams &params [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    const uint values_per_view = params.target_width * params.target_height * params.channels;
    const uint batch_size = params.batch_size == 0u ? 1u : params.batch_size;
    const uint total = values_per_view * batch_size;
    if (gid >= total) {
        return;
    }
    const uint local_gid = gid % values_per_view;
    const uint channel = local_gid % params.channels;
    const uint patch_index = local_gid / params.channels;
    const uint out_y = patch_index / params.target_width;
    const uint out_x = patch_index - out_y * params.target_width;
    const float pos = uocr_sam_abs_pos_bicubic_antialias(pos_embed, params, out_y, out_x, channel);
    dst_bhwc[gid] = half(float(patch_bhwc[gid]) + pos);
}

struct UocrRmsNormParams {
    uint n_rows;
    uint hidden_size;
    float eps;
    uint reserved;
};

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rmsnorm_f16_to_f16(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device half *dst [[buffer(2)]],
                                    constant UocrRmsNormParams &params [[buffer(3)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const float eps = uocr_fc_rms_eps_or(params.eps);
    float sum = 0.0f;
    const uint row_base = row * hidden_size;
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        sum += x * x;
    }
    const float total = uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
    const float scale = 1.0f / sqrt(total / float(hidden_size) + eps);
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        const float w = float(weight[col]);
        dst[row_base + col] = half(x * scale * w);
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rmsnorm_f16_to_f32(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device float *dst [[buffer(2)]],
                                    constant UocrRmsNormParams &params [[buffer(3)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const float eps = uocr_fc_rms_eps_or(params.eps);
    float sum = 0.0f;
    const uint row_base = row * hidden_size;
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        sum += x * x;
    }
    const float total = uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
    const float scale = 1.0f / sqrt(total / float(hidden_size) + eps);
    for (uint col = tid; col < hidden_size; col += ntg) {
        const float x = float(src[row_base + col]);
        const float w = float(weight[col]);
        dst[row_base + col] = x * scale * w;
    }
}

static inline float uocr_layernorm_reduce_sum(device const half *src,
                                              constant UocrRmsNormParams &params,
                                              uint row_base,
                                              uint tid,
                                              uint ntg,
                                              uint simd_width,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        sum += float(src[row_base + col]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

static inline float uocr_layernorm_reduce_var(device const half *src,
                                              constant UocrRmsNormParams &params,
                                              uint row_base,
                                              float mean,
                                              uint tid,
                                              uint ntg,
                                              uint simd_width,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float centered = float(src[row_base + col]) - mean;
        sum += centered * centered;
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width) / float(params.hidden_size);
}

kernel void uocr_layernorm_f16_to_f16(device const half *src [[buffer(0)]],
                                      device const half *weight [[buffer(1)]],
                                      device const half *bias [[buffer(2)]],
                                      device half *dst [[buffer(3)]],
                                      constant UocrRmsNormParams &params [[buffer(4)]],
                                      threadgroup float *partials [[threadgroup(0)]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint ntg [[threads_per_threadgroup]],
                                      uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint row_base = row * params.hidden_size;
    const float mean = uocr_layernorm_reduce_sum(src, params, row_base, tid, ntg, simd_width, partials) / float(params.hidden_size);
    const float variance = uocr_layernorm_reduce_var(src, params, row_base, mean, tid, ntg, simd_width, partials);
    const float scale = 1.0f / sqrt(variance + params.eps);
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float normalized = (float(src[row_base + col]) - mean) * scale;
        dst[row_base + col] = half(normalized * float(weight[col]) + float(bias[col]));
    }
}

kernel void uocr_layernorm_f16_to_f32(device const half *src [[buffer(0)]],
                                      device const half *weight [[buffer(1)]],
                                      device const half *bias [[buffer(2)]],
                                      device float *dst [[buffer(3)]],
                                      constant UocrRmsNormParams &params [[buffer(4)]],
                                      threadgroup float *partials [[threadgroup(0)]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint ntg [[threads_per_threadgroup]],
                                      uint simd_width [[threads_per_simdgroup]]) {
    if (row >= params.n_rows) {
        return;
    }

    const uint row_base = row * params.hidden_size;
    const float mean = uocr_layernorm_reduce_sum(src, params, row_base, tid, ntg, simd_width, partials) / float(params.hidden_size);
    const float variance = uocr_layernorm_reduce_var(src, params, row_base, mean, tid, ntg, simd_width, partials);
    const float scale = 1.0f / sqrt(variance + params.eps);
    for (uint col = tid; col < params.hidden_size; col += ntg) {
        const float normalized = (float(src[row_base + col]) - mean) * scale;
        dst[row_base + col] = normalized * float(weight[col]) + float(bias[col]);
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
                                       uint simd_width,
                                       threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.in_features;
    const uint weight_base = out_col * params.in_features;
    for (uint k = tid; k < params.in_features; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

kernel void uocr_dense_f16_to_f16(device const half *src [[buffer(0)]],
                                  device const half *weight [[buffer(1)]],
                                  device const half *bias [[buffer(2)]],
                                  device half *dst [[buffer(3)]],
                                  constant UocrDenseParams &params [[buffer(4)]],
                                  threadgroup float *partials [[threadgroup(0)]],
                                  uint output_index [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint ntg [[threads_per_threadgroup]],
                                  uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_f16(src, weight, params, row, out_col, tid, ntg, simd_width, partials);
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
                                  uint ntg [[threads_per_threadgroup]],
                                  uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.out_features;
    const uint out_col = output_index - row * params.out_features;
    if (row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    float value = uocr_dense_dot_f16(src, weight, params, row, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        if (params.has_bias != 0u) {
            value += float(bias[out_col]);
        }
        dst[row * params.out_features + out_col] = value;
    }
}

struct UocrBiasAddParams {
    uint rows;
    uint cols;
    uint reserved0;
    uint reserved1;
};

kernel void uocr_bias_add_f16_inplace(device half *dst [[buffer(0)]],
                                      device const half *bias [[buffer(1)]],
                                      constant UocrBiasAddParams &params [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    const uint vec_cols = uocr_div_up_u32(params.cols, UOCR_HALF4_WIDTH);
    const uint vector_count = params.rows * vec_cols;
    if (gid >= vector_count || vec_cols == 0u) {
        return;
    }
    const uint row = gid / vec_cols;
    const uint col = (gid - row * vec_cols) * UOCR_HALF4_WIDTH;
    const ulong base = (ulong(row) * ulong(params.cols)) + ulong(col);
    if (col + (UOCR_HALF4_WIDTH - 1u) < params.cols) {
        const half4 values = uocr_load_half4(dst, base);
        const half4 bias_values = uocr_load_half4(bias, ulong(col));
        uocr_store_half4(dst, base, half4(float4(values) + float4(bias_values)));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && col + i < params.cols; ++i) {
            const ulong idx = base + ulong(i);
            dst[idx] = half(float(dst[idx]) + float(bias[col + i]));
        }
    }
}







kernel void uocr_sam_qkv_f16_to_f16(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device const half *bias [[buffer(2)]],
                                    device half *q_dst [[buffer(3)]],
                                    device half *k_dst [[buffer(4)]],
                                    device half *v_dst [[buffer(5)]],
                                    constant UocrDenseParams &params [[buffer(6)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint output_index [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device half *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = half(value);
    }
}

kernel void uocr_sam_qkv_f16_to_f32(device const half *src [[buffer(0)]],
                                    device const half *weight [[buffer(1)]],
                                    device const half *bias [[buffer(2)]],
                                    device float *q_dst [[buffer(3)]],
                                    device float *k_dst [[buffer(4)]],
                                    device float *v_dst [[buffer(5)]],
                                    constant UocrDenseParams &params [[buffer(6)]],
                                    threadgroup float *partials [[threadgroup(0)]],
                                    uint output_index [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]],
                                    uint simd_width [[threads_per_simdgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device float *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = value;
    }
}

kernel void uocr_clip_qkv_f16_to_f16(device const half *src [[buffer(0)]],
                                     device const half *weight [[buffer(1)]],
                                     device const half *bias [[buffer(2)]],
                                     device half *q_dst [[buffer(3)]],
                                     device half *k_dst [[buffer(4)]],
                                     device half *v_dst [[buffer(5)]],
                                     constant UocrDenseParams &params [[buffer(6)]],
                                     threadgroup float *partials [[threadgroup(0)]],
                                     uint output_index [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint ntg [[threads_per_threadgroup]],
                                     uint simd_width [[threads_per_simdgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device half *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = half(value);
    }
}

kernel void uocr_clip_qkv_f16_to_f32(device const half *src [[buffer(0)]],
                                     device const half *weight [[buffer(1)]],
                                     device const half *bias [[buffer(2)]],
                                     device float *q_dst [[buffer(3)]],
                                     device float *k_dst [[buffer(4)]],
                                     device float *v_dst [[buffer(5)]],
                                     constant UocrDenseParams &params [[buffer(6)]],
                                     threadgroup float *partials [[threadgroup(0)]],
                                     uint output_index [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint ntg [[threads_per_threadgroup]],
                                     uint simd_width [[threads_per_simdgroup]]) {
    const uint values_per_projection = params.input_rows * params.out_features;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint row = element / params.out_features;
    const uint out_col = element - row * params.out_features;
    if (projection >= 3u || row >= params.input_rows || out_col >= params.out_features) {
        return;
    }

    const uint packed_col = projection * params.out_features + out_col;
    float value = uocr_dense_dot_f16(src, weight, params, row, packed_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        value += float(bias[packed_col]);
        device float *dst = projection == 0u ? q_dst : (projection == 1u ? k_dst : v_dst);
        dst[row * params.out_features + out_col] = value;
    }
}

static inline float uocr_quickgelu(float x) {
    return x / (1.0f + exp(-1.702f * x));
}

kernel void uocr_bias_quickgelu_f16_inplace(device half *dst [[buffer(0)]],
                                            device const half *bias [[buffer(1)]],
                                            constant UocrBiasAddParams &params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    const uint value_count = params.rows * params.cols;
    if (gid >= value_count) {
        return;
    }
    const uint col = gid - (gid / params.cols) * params.cols;
    const float value = float(dst[gid]) + float(bias[col]);
    dst[gid] = half(uocr_quickgelu(value));
}

kernel void uocr_clip_quickgelu_f16_to_f16(device const half *src [[buffer(0)]],
                                           device half *dst [[buffer(1)]],
                                           constant uint &value_count [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    dst[gid] = half(uocr_quickgelu(float(src[gid])));
}

kernel void uocr_clip_quickgelu_f16_to_f32(device const half *src [[buffer(0)]],
                                           device float *dst [[buffer(1)]],
                                           constant uint &value_count [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    dst[gid] = uocr_quickgelu(float(src[gid]));
}

kernel void uocr_clip_residual_add_f16_to_f16(device const half *base [[buffer(0)]],
                                              device const half *update [[buffer(1)]],
                                              device half *dst [[buffer(2)]],
                                              constant uint &value_count [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]]) {
    const uint value_base = gid * UOCR_HALF4_WIDTH;
    if (value_base >= value_count) {
        return;
    }
    if (value_base + (UOCR_HALF4_WIDTH - 1u) < value_count) {
        const half4 base_values = uocr_load_half4(base, ulong(value_base));
        const half4 update_values = uocr_load_half4(update, ulong(value_base));
        uocr_store_half4(dst, ulong(value_base), half4(float4(base_values) + float4(update_values)));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && value_base + i < value_count; ++i) {
            const uint idx = value_base + i;
            dst[idx] = half(float(base[idx]) + float(update[idx]));
        }
    }
}

kernel void uocr_clip_residual_add_f16_to_f32(device const half *base [[buffer(0)]],
                                              device const half *update [[buffer(1)]],
                                              device float *dst [[buffer(2)]],
                                              constant uint &value_count [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    dst[gid] = float(base[gid]) + float(update[gid]);
}

struct UocrClipSamConcatParams {
    uint grid_width;
    uint grid_height;
    uint hidden_size;
    uint projector_in_size;
};

kernel void uocr_clip_sam_concat_f16_to_f16(device const half *clip_tokens [[buffer(0)]],
                                            device const half *sam_nchw [[buffer(1)]],
                                            device half *dst [[buffer(2)]],
                                            constant UocrClipSamConcatParams &params [[buffer(3)]],
                                            uint3 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint spatial = gid.y;
    const uint batch = gid.z;
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint clip_tokens_per_view = spatial_size + 1u;
    if (col >= params.projector_in_size || spatial >= spatial_size) {
        return;
    }
    const uint dst_index = (batch * spatial_size + spatial) * params.projector_in_size + col;
    if (col < params.hidden_size) {
        dst[dst_index] = clip_tokens[(batch * clip_tokens_per_view + spatial + 1u) * params.hidden_size + col];
        return;
    }
    const uint channel = col - params.hidden_size;
    dst[dst_index] = sam_nchw[batch * (params.projector_in_size - params.hidden_size) * spatial_size +
                              channel * spatial_size + spatial];
}

kernel void uocr_clip_sam_concat_f16_to_f32(device const half *clip_tokens [[buffer(0)]],
                                            device const half *sam_nchw [[buffer(1)]],
                                            device float *dst [[buffer(2)]],
                                            constant UocrClipSamConcatParams &params [[buffer(3)]],
                                            uint3 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint spatial = gid.y;
    const uint batch = gid.z;
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint clip_tokens_per_view = spatial_size + 1u;
    if (col >= params.projector_in_size || spatial >= spatial_size) {
        return;
    }
    const uint dst_index = (batch * spatial_size + spatial) * params.projector_in_size + col;
    if (col < params.hidden_size) {
        dst[dst_index] = float(clip_tokens[(batch * clip_tokens_per_view + spatial + 1u) * params.hidden_size + col]);
        return;
    }
    const uint channel = col - params.hidden_size;
    dst[dst_index] = float(sam_nchw[batch * (params.projector_in_size - params.hidden_size) * spatial_size +
                              channel * spatial_size + spatial]);
}

struct UocrSamWindowAttentionParams {
    uint windows;
    uint tokens_per_window;
    uint heads;
    uint head_dim;
    float scale;
    uint batch_size;
    uint reserved1;
    uint reserved2;
};

static inline uint uocr_sam_window_attention_batch_size(constant UocrSamWindowAttentionParams &params) {
    return params.batch_size == 0u ? 1u : params.batch_size;
}

static inline ulong uocr_sam_window_attention_index(constant UocrSamWindowAttentionParams &params,
                                                    uint batch,
                                                    uint window,
                                                    uint token,
                                                    uint head,
                                                    uint dim) {
    return (((((ulong(batch) * ulong(params.windows) + ulong(window)) * ulong(params.tokens_per_window) + ulong(token)) *
              ulong(params.heads) +
              ulong(head)) *
             ulong(params.head_dim)) +
            ulong(dim));
}

struct UocrSamRelPosAttentionParams {
    uint windows;
    uint grid_width;
    uint grid_height;
    uint tokens_per_window;
    uint heads;
    uint head_dim;
    uint rel_pos_h_length;
    uint rel_pos_w_length;
    float scale;
    uint batch_size;
    uint reserved1;
    uint reserved2;
};

static inline uint uocr_sam_rel_pos_attention_batch_size(constant UocrSamRelPosAttentionParams &params) {
    return params.batch_size == 0u ? 1u : params.batch_size;
}

static inline ulong uocr_sam_rel_pos_attention_index(constant UocrSamRelPosAttentionParams &params,
                                                     uint batch,
                                                     uint window,
                                                     uint token,
                                                     uint head,
                                                     uint dim) {
    return (((((ulong(batch) * ulong(params.windows) + ulong(window)) * ulong(params.tokens_per_window) + ulong(token)) *
              ulong(params.heads) +
              ulong(head)) *
             ulong(params.head_dim)) +
            ulong(dim));
}

static inline float uocr_sam_rel_pos_table_value(device const half *rel_pos,
                                                 uint source_length,
                                                 uint target_length,
                                                 uint target_index,
                                                 uint dim,
                                                 uint head_dim) {
    if (source_length == target_length) {
        return float(rel_pos[ulong(target_index) * ulong(head_dim) + ulong(dim)]);
    }

    const float source_x = (float(target_index) + 0.5f) * float(source_length) / float(target_length) - 0.5f;
    int index0 = int(floor(source_x));
    int index1 = index0 + 1;
    const float t = source_x - floor(source_x);
    index0 = clamp(index0, 0, int(source_length) - 1);
    index1 = clamp(index1, 0, int(source_length) - 1);
    const float v0 = float(rel_pos[ulong(uint(index0)) * ulong(head_dim) + ulong(dim)]);
    const float v1 = float(rel_pos[ulong(uint(index1)) * ulong(head_dim) + ulong(dim)]);
    return v0 * (1.0f - t) + v1 * t;
}

struct UocrSamRelPosInterpolateParams {
    uint source_length;
    uint target_length;
    uint head_dim;
    uint reserved;
};

kernel void uocr_sam_interpolate_rel_pos_f16(device const half *src [[buffer(0)]],
                                             device half *dst [[buffer(1)]],
                                             constant UocrSamRelPosInterpolateParams &params [[buffer(2)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint total = params.target_length * params.head_dim;
    if (gid >= total || params.source_length == 0u || params.target_length == 0u || params.head_dim == 0u) {
        return;
    }
    const uint target_index = gid / params.head_dim;
    const uint dim = gid - target_index * params.head_dim;
    dst[gid] = half(uocr_sam_rel_pos_table_value(src,
                                                 params.source_length,
                                                 params.target_length,
                                                 target_index,
                                                 dim,
                                                 params.head_dim));
}

struct UocrSamWindowPartitionParams {
    uint grid_width;
    uint grid_height;
    uint padded_width;
    uint padded_height;
    uint windows_per_row;
    uint windows_per_col;
    uint window_size;
    uint hidden_size;
    uint batch_size;
};

kernel void uocr_sam_window_partition_f16(device const half *src [[buffer(0)]],
                                          device half *dst [[buffer(1)]],
                                          constant UocrSamWindowPartitionParams &params [[buffer(2)]],
                                          uint gid [[thread_position_in_grid]]) {
    const uint padded_tokens = params.padded_width * params.padded_height;
    const uint values_per_view = padded_tokens * params.hidden_size;
    const uint batch_size = params.batch_size == 0u ? 1u : params.batch_size;
    const uint values = values_per_view * batch_size;
    if (gid >= values) {
        return;
    }

    const uint batch = gid / values_per_view;
    const uint local_gid = gid - batch * values_per_view;
    const uint channel = local_gid % params.hidden_size;
    const uint flat_token = local_gid / params.hidden_size;
    const uint token_in_window = flat_token % (params.window_size * params.window_size);
    const uint window = flat_token / (params.window_size * params.window_size);
    const uint window_x = window % params.windows_per_row;
    const uint window_y = window / params.windows_per_row;
    const uint local_y = token_in_window / params.window_size;
    const uint local_x = token_in_window - local_y * params.window_size;
    const uint src_y = window_y * params.window_size + local_y;
    const uint src_x = window_x * params.window_size + local_x;

    if (src_y < params.grid_height && src_x < params.grid_width) {
        const uint src_spatial = src_y * params.grid_width + src_x;
        dst[gid] = src[(batch * params.grid_width * params.grid_height + src_spatial) * params.hidden_size + channel];
    } else {
        dst[gid] = half(0.0f);
    }
}

kernel void uocr_sam_window_unpartition_f16(device const half *src [[buffer(0)]],
                                            device half *dst [[buffer(1)]],
                                            constant UocrSamWindowPartitionParams &params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    const uint values_per_view = params.grid_width * params.grid_height * params.hidden_size;
    const uint batch_size = params.batch_size == 0u ? 1u : params.batch_size;
    const uint values = values_per_view * batch_size;
    if (gid >= values) {
        return;
    }

    const uint batch = gid / values_per_view;
    const uint local_gid = gid - batch * values_per_view;
    const uint channel = local_gid % params.hidden_size;
    const uint flat_token = local_gid / params.hidden_size;
    const uint y = flat_token / params.grid_width;
    const uint x = flat_token - y * params.grid_width;
    const uint window_y = y / params.window_size;
    const uint window_x = x / params.window_size;
    const uint local_y = y - window_y * params.window_size;
    const uint local_x = x - window_x * params.window_size;
    const uint window = window_y * params.windows_per_row + window_x;
    const uint token_in_window = local_y * params.window_size + local_x;

    const uint window_tokens = params.window_size * params.window_size;
    dst[gid] = src[((batch * params.windows_per_row * params.windows_per_col + window) * window_tokens + token_in_window) *
                   params.hidden_size +
                   channel];
}

struct UocrSamNeckConv1x1Params {
    uint grid_width;
    uint grid_height;
    uint in_channels;
    uint out_channels;
    uint batch_size;
};

static inline float uocr_sam_neck_conv1x1_tile_value_f16(device const half *src_bhwc,
                                                         device const half *weight,
                                                         constant UocrSamNeckConv1x1Params &params,
                                                         uint spatial,
                                                         uint out_channel,
                                                         uint batch) {
    float sum = 0.0f;
    const uint spatial_size = params.grid_width * params.grid_height;
    const ulong src_base = (ulong(batch) * ulong(spatial_size) + ulong(spatial)) * ulong(params.in_channels);
    const uint weight_base = out_channel * params.in_channels;
    for (uint c = 0u; c < params.in_channels; ++c) {
        sum += float(src_bhwc[src_base + ulong(c)]) * float(weight[weight_base + c]);
    }
    return sum;
}

kernel void uocr_sam_neck_conv1x1_f16_to_f16(device const half *src_bhwc [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device half *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv1x1Params &params [[buffer(3)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.out_channels || spatial >= spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_neck_conv1x1_tile_value_f16(src_bhwc, weight, params, spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.out_channels) + ulong(out_channel)) * ulong(spatial_size) + ulong(spatial)] = half(value);
}

kernel void uocr_sam_neck_conv1x1_f16_to_f32(device const half *src_bhwc [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device float *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv1x1Params &params [[buffer(3)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.out_channels || spatial >= spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_neck_conv1x1_tile_value_f16(src_bhwc, weight, params, spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.out_channels) + ulong(out_channel)) * ulong(spatial_size) + ulong(spatial)] = value;
}

struct UocrSamNeckConv3x3Params {
    uint grid_width;
    uint grid_height;
    uint channels;
    uint kernel_size;
    uint batch_size;
};

static inline float uocr_sam_neck_conv3x3_tile_value_f16(device const half *src_nchw,
                                                         device const half *weight,
                                                         constant UocrSamNeckConv3x3Params &params,
                                                         uint spatial,
                                                         uint out_channel,
                                                         uint batch) {
    const uint x = spatial % params.grid_width;
    const uint y = spatial / params.grid_width;
    const uint spatial_size = params.grid_width * params.grid_height;
    const int kernel_radius = int(params.kernel_size >> 1u);
    float sum = 0.0f;
    for (uint in_channel = 0u; in_channel < params.channels; ++in_channel) {
        for (uint ky = 0u; ky < params.kernel_size; ++ky) {
            const int sy = int(y) + int(ky) - kernel_radius;
            if (sy < 0 || sy >= int(params.grid_height)) {
                continue;
            }
            for (uint kx = 0u; kx < params.kernel_size; ++kx) {
                const int sx = int(x) + int(kx) - kernel_radius;
                if (sx < 0 || sx >= int(params.grid_width)) {
                    continue;
                }
                const ulong src_index = (ulong(batch) * ulong(params.channels) + ulong(in_channel)) * ulong(spatial_size) +
                                        ulong(uint(sy)) * ulong(params.grid_width) + ulong(uint(sx));
                const ulong weight_index = ((ulong(out_channel) * ulong(params.channels) + ulong(in_channel)) *
                                                ulong(params.kernel_size) + ulong(ky)) *
                                               ulong(params.kernel_size) + ulong(kx);
                sum += float(src_nchw[src_index]) * float(weight[weight_index]);
            }
        }
    }
    return sum;
}

kernel void uocr_sam_neck_conv3x3_f16_to_f16(device const half *src_nchw [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device half *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv3x3Params &params [[buffer(3)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.channels || spatial >= spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_neck_conv3x3_tile_value_f16(src_nchw, weight, params, spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.channels) + ulong(out_channel)) * ulong(spatial_size) + ulong(spatial)] = half(value);
}

kernel void uocr_sam_neck_conv3x3_f16_to_f32(device const half *src_nchw [[buffer(0)]],
                                             device const half *weight [[buffer(1)]],
                                             device float *dst_nchw [[buffer(2)]],
                                             constant UocrSamNeckConv3x3Params &params [[buffer(3)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.channels || spatial >= spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_neck_conv3x3_tile_value_f16(src_nchw, weight, params, spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.channels) + ulong(out_channel)) * ulong(spatial_size) + ulong(spatial)] = value;
}

struct UocrSamConv3x3Stride2Params {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint in_channels;
    uint out_channels;
    uint kernel_size;
    uint stride;
    uint batch_size;
};

static inline float uocr_sam_conv3x3_stride2_tile_value_f16(device const half *src_nchw,
                                                            device const half *weight,
                                                            constant UocrSamConv3x3Stride2Params &params,
                                                            uint output_spatial,
                                                            uint out_channel,
                                                            uint batch) {
    const uint out_x = output_spatial % params.output_width;
    const uint out_y = output_spatial / params.output_width;
    const uint input_spatial_size = params.input_width * params.input_height;
    const int kernel_radius = int(params.kernel_size >> 1u);
    float sum = 0.0f;
    for (uint in_channel = 0u; in_channel < params.in_channels; ++in_channel) {
        for (uint ky = 0u; ky < params.kernel_size; ++ky) {
            const int sy = int(out_y * params.stride) + int(ky) - kernel_radius;
            if (sy < 0 || sy >= int(params.input_height)) {
                continue;
            }
            for (uint kx = 0u; kx < params.kernel_size; ++kx) {
                const int sx = int(out_x * params.stride) + int(kx) - kernel_radius;
                if (sx < 0 || sx >= int(params.input_width)) {
                    continue;
                }
                const ulong src_index = (ulong(batch) * ulong(params.in_channels) + ulong(in_channel)) * ulong(input_spatial_size) +
                                        ulong(uint(sy)) * ulong(params.input_width) + ulong(uint(sx));
                const ulong weight_index = ((ulong(out_channel) * ulong(params.in_channels) + ulong(in_channel)) *
                                                ulong(params.kernel_size) + ulong(ky)) *
                                               ulong(params.kernel_size) + ulong(kx);
                sum += float(src_nchw[src_index]) * float(weight[weight_index]);
            }
        }
    }
    return sum;
}

kernel void uocr_sam_conv3x3_stride2_f16_to_f16(device const half *src_nchw [[buffer(0)]],
                                                device const half *weight [[buffer(1)]],
                                                device half *dst_nchw [[buffer(2)]],
                                                constant UocrSamConv3x3Stride2Params &params [[buffer(3)]],
                                                uint3 gid [[thread_position_in_grid]]) {
    const uint output_spatial_size = params.output_width * params.output_height;
    const uint output_spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.out_channels || output_spatial >= output_spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_conv3x3_stride2_tile_value_f16(src_nchw, weight, params, output_spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.out_channels) + ulong(out_channel)) * ulong(output_spatial_size) + ulong(output_spatial)] = half(value);
}

kernel void uocr_sam_conv3x3_stride2_f16_to_f32(device const half *src_nchw [[buffer(0)]],
                                                device const half *weight [[buffer(1)]],
                                                device float *dst_nchw [[buffer(2)]],
                                                constant UocrSamConv3x3Stride2Params &params [[buffer(3)]],
                                                uint3 gid [[thread_position_in_grid]]) {
    const uint output_spatial_size = params.output_width * params.output_height;
    const uint output_spatial = gid.x;
    const uint out_channel = gid.y;
    const uint batch = gid.z;
    if (out_channel >= params.out_channels || output_spatial >= output_spatial_size || batch >= params.batch_size) {
        return;
    }

    const float value = uocr_sam_conv3x3_stride2_tile_value_f16(src_nchw, weight, params, output_spatial, out_channel, batch);
    dst_nchw[(ulong(batch) * ulong(params.out_channels) + ulong(out_channel)) * ulong(output_spatial_size) + ulong(output_spatial)] = value;
}

struct UocrClipEmbedSamParams {
    uint grid_width;
    uint grid_height;
    uint hidden_size;
    uint token_count;
    uint batch_size;
};

static inline float uocr_clip_embedding_from_sam_value(device const half *sam_nchw,
                                                       device const half *class_embedding,
                                                       constant UocrClipEmbedSamParams &params,
                                                       uint token,
                                                       uint channel,
                                                       uint batch) {
    if (token == 0u) {
        return float(class_embedding[channel]);
    }
    const uint spatial = token - 1u;
    const uint y = spatial / params.grid_width;
    const uint x = spatial - y * params.grid_width;
    const ulong spatial_size = ulong(params.grid_width) * ulong(params.grid_height);
    const ulong src_index = ulong(batch) * spatial_size * ulong(params.hidden_size) +
                            ulong(channel) * spatial_size + ulong(y) * ulong(params.grid_width) + ulong(x);
    return float(sam_nchw[src_index]);
}

kernel void uocr_clip_embed_sam_f16_to_f16(device const half *sam_nchw [[buffer(0)]],
                                           device const half *class_embedding [[buffer(1)]],
                                           device half *dst_tokens [[buffer(2)]],
                                           constant UocrClipEmbedSamParams &params [[buffer(3)]],
                                           uint3 gid [[thread_position_in_grid]]) {
    const uint channel = gid.x;
    const uint token = gid.y;
    const uint batch = gid.z;
    if (channel >= params.hidden_size || token >= params.token_count || batch >= params.batch_size) {
        return;
    }
    const float value = uocr_clip_embedding_from_sam_value(sam_nchw, class_embedding, params, token, channel, batch);
    dst_tokens[(ulong(batch) * ulong(params.token_count) + ulong(token)) * ulong(params.hidden_size) + ulong(channel)] = half(value);
}

kernel void uocr_clip_embed_sam_f16_to_f32(device const half *sam_nchw [[buffer(0)]],
                                           device const half *class_embedding [[buffer(1)]],
                                           device float *dst_tokens [[buffer(2)]],
                                           constant UocrClipEmbedSamParams &params [[buffer(3)]],
                                           uint3 gid [[thread_position_in_grid]]) {
    const uint channel = gid.x;
    const uint token = gid.y;
    const uint batch = gid.z;
    if (channel >= params.hidden_size || token >= params.token_count || batch >= params.batch_size) {
        return;
    }
    dst_tokens[(ulong(batch) * ulong(params.token_count) + ulong(token)) * ulong(params.hidden_size) + ulong(channel)] =
        uocr_clip_embedding_from_sam_value(sam_nchw, class_embedding, params, token, channel, batch);
}

struct UocrClipAbsPosParams {
    uint source_grid;
    uint target_width;
    uint target_height;
    uint hidden_size;
    uint batch_size;
};

static float uocr_clip_abs_pos_bicubic_antialias(device const half *pos_embed,
                                                 constant UocrClipAbsPosParams &params,
                                                 uint token,
                                                 uint channel) {
    if (token == 0u) {
        return float(pos_embed[channel]);
    }
    const uint spatial = token - 1u;
    const uint out_y = spatial / params.target_width;
    const uint out_x = spatial - out_y * params.target_width;
    if (params.target_width == params.source_grid && params.target_height == params.source_grid) {
        return float(pos_embed[(token * params.hidden_size) + channel]);
    }

    const float scale_y = float(params.source_grid) / float(params.target_height);
    const float scale_x = float(params.source_grid) / float(params.target_width);
    const float filter_scale_y = max(scale_y, 1.0f);
    const float filter_scale_x = max(scale_x, 1.0f);
    const float support_y = 2.0f * filter_scale_y;
    const float support_x = 2.0f * filter_scale_x;
    const float center_y = (float(out_y) + 0.5f) * scale_y - 0.5f;
    const float center_x = (float(out_x) + 0.5f) * scale_x - 0.5f;

    const int src_grid = int(params.source_grid);
    const int y0 = max(0, int(floor(center_y - support_y)));
    const int y1 = min(src_grid - 1, int(ceil(center_y + support_y)));
    const int x0 = max(0, int(floor(center_x - support_x)));
    const int x1 = min(src_grid - 1, int(ceil(center_x + support_x)));

    float acc = 0.0f;
    float weight_sum = 0.0f;
    for (int iy = y0; iy <= y1; ++iy) {
        const float wy = uocr_sam_abs_pos_axis_weight(iy, center_y, filter_scale_y);
        for (int ix = x0; ix <= x1; ++ix) {
            const float wx = uocr_sam_abs_pos_axis_weight(ix, center_x, filter_scale_x);
            const float w = wy * wx;
            const ulong source_token = 1ul + ulong(uint(iy)) * ulong(params.source_grid) + ulong(uint(ix));
            acc += float(pos_embed[source_token * ulong(params.hidden_size) + ulong(channel)]) * w;
            weight_sum += w;
        }
    }
    return weight_sum != 0.0f ? acc / weight_sum : 0.0f;
}

kernel void uocr_clip_add_abs_pos_f16_to_f16(device const half *tokens [[buffer(0)]],
                                             device const half *pos_embed [[buffer(1)]],
                                             device half *dst_tokens [[buffer(2)]],
                                             constant UocrClipAbsPosParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint token_count = 1u + params.target_width * params.target_height;
    const uint values_per_view = token_count * params.hidden_size;
    const uint total = values_per_view * params.batch_size;
    if (gid >= total) {
        return;
    }
    const uint view_value = gid % values_per_view;
    const uint channel = view_value % params.hidden_size;
    const uint token = view_value / params.hidden_size;
    const float pos = uocr_clip_abs_pos_bicubic_antialias(pos_embed, params, token, channel);
    dst_tokens[gid] = half(float(tokens[gid]) + pos);
}

kernel void uocr_clip_add_abs_pos_f16_to_f32(device const half *tokens [[buffer(0)]],
                                             device const half *pos_embed [[buffer(1)]],
                                             device float *dst_tokens [[buffer(2)]],
                                             constant UocrClipAbsPosParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint token_count = 1u + params.target_width * params.target_height;
    const uint values_per_view = token_count * params.hidden_size;
    const uint total = values_per_view * params.batch_size;
    if (gid >= total) {
        return;
    }
    const uint view_value = gid % values_per_view;
    const uint channel = view_value % params.hidden_size;
    const uint token = view_value / params.hidden_size;
    const float pos = uocr_clip_abs_pos_bicubic_antialias(pos_embed, params, token, channel);
    dst_tokens[gid] = float(tokens[gid]) + pos;
}

struct UocrSamLayerNorm2dParams {
    uint grid_width;
    uint grid_height;
    uint channels;
    float eps;
    uint batch_size;
};

static inline float uocr_sam_layernorm2d_value(device const half *src_nchw,
                                               device const half *weight,
                                               device const half *bias,
                                               constant UocrSamLayerNorm2dParams &params,
                                               uint spatial,
                                               uint channel,
                                               uint batch,
                                               threadgroup float *partials,
                                               uint tid,
                                               uint ntg,
                                               uint simd_width) {
    const uint spatial_size = params.grid_width * params.grid_height;
    float value = 0.0f;
    if (tid < params.channels) {
        value = float(src_nchw[(ulong(batch) * ulong(params.channels) + ulong(tid)) * ulong(spatial_size) + ulong(spatial)]);
    }
    float value_sum = 0.0f;
    float square_sum = 0.0f;
    uocr_threadgroup_sum2(tid < params.channels ? value : 0.0f,
                          tid < params.channels ? value * value : 0.0f,
                          partials,
                          partials + ntg,
                          tid,
                          ntg,
                          simd_width,
                          value_sum,
                          square_sum);

    const float inv_channels = 1.0f / float(params.channels);
    const float mean = value_sum * inv_channels;
    const float variance = max(square_sum * inv_channels - mean * mean, 0.0f);
    return (value - mean) * rsqrt(variance + params.eps) * float(weight[channel]) + float(bias[channel]);
}

kernel void uocr_sam_layernorm2d_f16_to_f16(device const half *src_nchw [[buffer(0)]],
                                            device const half *weight [[buffer(1)]],
                                            device const half *bias [[buffer(2)]],
                                            device half *dst_nchw [[buffer(3)]],
                                            constant UocrSamLayerNorm2dParams &params [[buffer(4)]],
                                            threadgroup float *partials [[threadgroup(0)]],
                                            uint3 block [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint3 ntg3 [[threads_per_threadgroup]],
                                            uint simd_width [[threads_per_simdgroup]]) {
    const uint ntg = ntg3.x;
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = block.x;
    const uint batch = block.z;
    if (spatial >= spatial_size || tid >= params.channels || batch >= params.batch_size) {
        return;
    }
    const float value = uocr_sam_layernorm2d_value(src_nchw, weight, bias, params, spatial, tid, batch, partials, tid, ntg, simd_width);
    dst_nchw[(ulong(batch) * ulong(params.channels) + ulong(tid)) * ulong(spatial_size) + ulong(spatial)] = half(value);
}

kernel void uocr_sam_layernorm2d_f16_to_f32(device const half *src_nchw [[buffer(0)]],
                                            device const half *weight [[buffer(1)]],
                                            device const half *bias [[buffer(2)]],
                                            device float *dst_nchw [[buffer(3)]],
                                            constant UocrSamLayerNorm2dParams &params [[buffer(4)]],
                                            threadgroup float *partials [[threadgroup(0)]],
                                            uint3 block [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint3 ntg3 [[threads_per_threadgroup]],
                                            uint simd_width [[threads_per_simdgroup]]) {
    const uint ntg = ntg3.x;
    const uint spatial_size = params.grid_width * params.grid_height;
    const uint spatial = block.x;
    const uint batch = block.z;
    if (spatial >= spatial_size || tid >= params.channels || batch >= params.batch_size) {
        return;
    }
    const float value = uocr_sam_layernorm2d_value(src_nchw, weight, bias, params, spatial, tid, batch, partials, tid, ntg, simd_width);
    dst_nchw[(ulong(batch) * ulong(params.channels) + ulong(tid)) * ulong(spatial_size) + ulong(spatial)] = value;
}

struct UocrSamResidualParams {
    uint n_rows;
    uint hidden_size;
    uint reserved0;
    uint reserved1;
};

kernel void uocr_sam_residual_add_f16_to_f16(device const half *base [[buffer(0)]],
                                             device const half *update [[buffer(1)]],
                                             device half *dst [[buffer(2)]],
                                             constant UocrSamResidualParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint values = params.n_rows * params.hidden_size;
    const uint value_base = gid * UOCR_HALF4_WIDTH;
    if (value_base >= values) {
        return;
    }
    if (value_base + (UOCR_HALF4_WIDTH - 1u) < values) {
        const half4 base_values = uocr_load_half4(base, ulong(value_base));
        const half4 update_values = uocr_load_half4(update, ulong(value_base));
        uocr_store_half4(dst, ulong(value_base), half4(float4(base_values) + float4(update_values)));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && value_base + i < values; ++i) {
            const uint idx = value_base + i;
            dst[idx] = half(float(base[idx]) + float(update[idx]));
        }
    }
}

kernel void uocr_sam_residual_add_f16_to_f32(device const half *base [[buffer(0)]],
                                             device const half *update [[buffer(1)]],
                                             device float *dst [[buffer(2)]],
                                             constant UocrSamResidualParams &params [[buffer(3)]],
                                             uint gid [[thread_position_in_grid]]) {
    const uint values = params.n_rows * params.hidden_size;
    if (gid >= values) {
        return;
    }
    dst[gid] = float(base[gid]) + float(update[gid]);
}

static inline float uocr_sam_attention_project_dot_f16(device const half *src,
                                                       device const half *weight,
                                                       constant UocrSamResidualParams &params,
                                                       uint row,
                                                       uint out_col,
                                                       uint tid,
                                                       uint ntg,
                                              uint simd_width,
                                                       threadgroup float *partials) {
    float sum = 0.0f;
    const uint src_base = row * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

kernel void uocr_sam_attention_project_residual_f16_to_f16(device const half *src [[buffer(0)]],
                                                           device const half *weight [[buffer(1)]],
                                                           device const half *bias [[buffer(2)]],
                                                           device const half *residual [[buffer(3)]],
                                                           device half *dst [[buffer(4)]],
                                                           constant UocrSamResidualParams &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]],
                                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_attention_project_dot_f16(src, weight, params, row, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        const uint dst_index = row * params.hidden_size + out_col;
        dst[dst_index] = half(value + float(bias[out_col]) + float(residual[dst_index]));
    }
}

kernel void uocr_sam_attention_project_residual_f16_to_f32(device const half *src [[buffer(0)]],
                                                           device const half *weight [[buffer(1)]],
                                                           device const half *bias [[buffer(2)]],
                                                           device const half *residual [[buffer(3)]],
                                                           device float *dst [[buffer(4)]],
                                                           constant UocrSamResidualParams &params [[buffer(5)]],
                                                           threadgroup float *partials [[threadgroup(0)]],
                                                           uint output_index [[threadgroup_position_in_grid]],
                                                           uint tid [[thread_index_in_threadgroup]],
                                                           uint ntg [[threads_per_threadgroup]],
                                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_attention_project_dot_f16(src, weight, params, row, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        const uint dst_index = row * params.hidden_size + out_col;
        dst[dst_index] = value + float(bias[out_col]) + float(residual[dst_index]);
    }
}

struct UocrSamMlpParams {
    uint n_rows;
    uint hidden_size;
    uint intermediate_size;
    uint reserved;
};

static inline float uocr_erf_approx(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
                            0.254829592f) *
                               t * exp(-(ax * ax));
    return sign * y;
}

static inline float uocr_gelu_erf(float x) {
    return 0.5f * x * (1.0f + uocr_erf_approx(x * 0.70710678118654752440f));
}

kernel void uocr_bias_gelu_f16_inplace(device half *dst [[buffer(0)]],
                                       device const half *bias [[buffer(1)]],
                                       constant UocrBiasAddParams &params [[buffer(2)]],
                                       uint gid [[thread_position_in_grid]]) {
    const uint value_count = params.rows * params.cols;
    if (gid >= value_count) {
        return;
    }
    const uint col = gid - (gid / params.cols) * params.cols;
    const float value = float(dst[gid]) + float(bias[col]);
    dst[gid] = half(uocr_gelu_erf(value));
}

kernel void uocr_sam_mlp_lin1_gelu_f16(device const half *src [[buffer(0)]],
                                       device const half *weight [[buffer(1)]],
                                       device const half *bias [[buffer(2)]],
                                       device half *mid [[buffer(3)]],
                                       constant UocrSamMlpParams &params [[buffer(4)]],
                                       threadgroup float *partials [[threadgroup(0)]],
                                       uint output_index [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint ntg [[threads_per_threadgroup]],
                                       uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.intermediate_size;
    const uint out_col = output_index - row * params.intermediate_size;
    if (row >= params.n_rows || out_col >= params.intermediate_size) {
        return;
    }

    float sum = 0.0f;
    const uint src_base = row * params.hidden_size;
    const uint weight_base = out_col * params.hidden_size;
    for (uint k = tid; k < params.hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    const float total = uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);

    if (tid == 0u) {
        const float projected = total + float(bias[out_col]);
        mid[row * params.intermediate_size + out_col] = half(uocr_gelu_erf(projected));
    }
}

static inline float uocr_sam_mlp_lin2_dot_f16(device const half *mid,
                                              device const half *weight,
                                              constant UocrSamMlpParams &params,
                                              uint row,
                                              uint out_col,
                                              uint tid,
                                              uint ntg,
                                              uint simd_width,
                                              threadgroup float *partials) {
    float sum = 0.0f;
    const uint mid_base = row * params.intermediate_size;
    const uint weight_base = out_col * params.intermediate_size;
    for (uint k = tid; k < params.intermediate_size; k += ntg) {
        sum += float(mid[mid_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

kernel void uocr_sam_mlp_lin2_f16_to_f16(device const half *mid [[buffer(0)]],
                                         device const half *weight [[buffer(1)]],
                                         device const half *bias [[buffer(2)]],
                                         device half *dst [[buffer(3)]],
                                         constant UocrSamMlpParams &params [[buffer(4)]],
                                         threadgroup float *partials [[threadgroup(0)]],
                                         uint output_index [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint ntg [[threads_per_threadgroup]],
                                         uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_mlp_lin2_dot_f16(mid, weight, params, row, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        dst[row * params.hidden_size + out_col] = half(value + float(bias[out_col]));
    }
}

kernel void uocr_sam_mlp_lin2_f16_to_f32(device const half *mid [[buffer(0)]],
                                         device const half *weight [[buffer(1)]],
                                         device const half *bias [[buffer(2)]],
                                         device float *dst [[buffer(3)]],
                                         constant UocrSamMlpParams &params [[buffer(4)]],
                                         threadgroup float *partials [[threadgroup(0)]],
                                         uint output_index [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint ntg [[threads_per_threadgroup]],
                                         uint simd_width [[threads_per_simdgroup]]) {
    const uint row = output_index / params.hidden_size;
    const uint out_col = output_index - row * params.hidden_size;
    if (row >= params.n_rows || out_col >= params.hidden_size) {
        return;
    }

    float value = uocr_sam_mlp_lin2_dot_f16(mid, weight, params, row, out_col, tid, ntg, simd_width, partials);
    if (tid == 0u) {
        dst[row * params.hidden_size + out_col] = value + float(bias[out_col]);
    }
}

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
        for (uint k = lane; k < hidden_size; k += lanes) {
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
                                              uint simd_width,
                                                      threadgroup float *partials) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    float sum = 0.0f;
    const uint src_base = token * hidden_size;
    const uint weight_base = out_col * hidden_size;
    for (uint k = tid; k < hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_attention_qkvo_f16_to_f16(device const half *src [[buffer(0)]],
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
                                           uint ntg [[threads_per_threadgroup]],
                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint projection_count = uocr_fc_attention_projection_count_or(params.projection_count);
    const uint values_per_projection = params.n_tokens * hidden_size;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint token = element / hidden_size;
    const uint out_col = element - token * hidden_size;
    if (projection >= projection_count || token >= params.n_tokens || out_col >= hidden_size) {
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

    const float value = uocr_attention_projection_dot_f16(src, weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        dst[token * hidden_size + out_col] = half(value);
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_attention_qkvo_f16_to_f32(device const half *src [[buffer(0)]],
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
                                           uint ntg [[threads_per_threadgroup]],
                                           uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint projection_count = uocr_fc_attention_projection_count_or(params.projection_count);
    const uint values_per_projection = params.n_tokens * hidden_size;
    const uint projection = output_index / values_per_projection;
    const uint element = output_index - projection * values_per_projection;
    const uint token = element / hidden_size;
    const uint out_col = element - token * hidden_size;
    if (projection >= projection_count || token >= params.n_tokens || out_col >= hidden_size) {
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

    const float value = uocr_attention_projection_dot_f16(src, weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        dst[token * hidden_size + out_col] = value;
    }
}

struct UocrAttentionOutputParams {
    uint n_tokens;
    uint hidden_size;
    uint reserved0;
    uint reserved1;
};

static inline float uocr_attention_output_dot_f16(device const half *src,
                                                  device const half *weight,
                                                  constant UocrAttentionOutputParams &params,
                                                  uint token,
                                                  uint out_col,
                                                  uint tid,
                                                  uint ntg,
                                              uint simd_width,
                                                  threadgroup float *partials) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    float sum = 0.0f;
    const uint src_base = token * hidden_size;
    const uint weight_base = out_col * hidden_size;
    for (uint k = tid; k < hidden_size; k += ntg) {
        sum += float(src[src_base + k]) * float(weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_attention_output_residual_f16_to_f16(device const half *src [[buffer(0)]],
                                                      device const half *weight [[buffer(1)]],
                                                      device const half *residual [[buffer(2)]],
                                                      device half *dst [[buffer(3)]],
                                                      constant UocrAttentionOutputParams &params [[buffer(4)]],
                                                      threadgroup float *partials [[threadgroup(0)]],
                                                      uint output_index [[threadgroup_position_in_grid]],
                                                      uint tid [[thread_index_in_threadgroup]],
                                                      uint ntg [[threads_per_threadgroup]],
                                                      uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    const float projected = uocr_attention_output_dot_f16(src, weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        const uint dst_index = token * hidden_size + out_col;
        dst[dst_index] = half(projected + float(residual[dst_index]));
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_attention_output_residual_f16_to_f32(device const half *src [[buffer(0)]],
                                                      device const half *weight [[buffer(1)]],
                                                      device const half *residual [[buffer(2)]],
                                                      device float *dst [[buffer(3)]],
                                                      constant UocrAttentionOutputParams &params [[buffer(4)]],
                                                      threadgroup float *partials [[threadgroup(0)]],
                                                      uint output_index [[threadgroup_position_in_grid]],
                                                      uint tid [[thread_index_in_threadgroup]],
                                                      uint ntg [[threads_per_threadgroup]],
                                                      uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    const float projected = uocr_attention_output_dot_f16(src, weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        const uint dst_index = token * hidden_size + out_col;
        dst[dst_index] = projected + float(residual[dst_index]);
    }
}

struct UocrDenseSwigluParams {
    uint n_tokens;
    uint hidden_size;
    uint intermediate_size;
    uint has_residual;
};

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_gate_up_f16(device const half *src [[buffer(0)]],
                                          device const half *gate_weight [[buffer(1)]],
                                          device const half *up_weight [[buffer(2)]],
                                          device half *mid [[buffer(3)]],
                                          constant UocrDenseSwigluParams &params [[buffer(4)]],
                                          threadgroup float *partials [[threadgroup(0)]],
                                          uint output_index [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]],
                                          uint simd_width [[threads_per_simdgroup]]) {
    const uint token = output_index / params.intermediate_size;
    const uint out_col = output_index - token * params.intermediate_size;
    if (token >= params.n_tokens || out_col >= params.intermediate_size) {
        return;
    }

    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    threadgroup float *gate_partials = partials;
    threadgroup float *up_partials = partials + ntg;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint src_base = token * hidden_size;
    const uint weight_base = out_col * hidden_size;
    for (uint k = tid; k < hidden_size; k += ntg) {
        const float x = float(src[src_base + k]);
        gate_sum += x * float(gate_weight[weight_base + k]);
        up_sum += x * float(up_weight[weight_base + k]);
    }
    float gate = 0.0f;
    float up = 0.0f;
    uocr_threadgroup_sum2(gate_sum, up_sum, gate_partials, up_partials, tid, ntg, simd_width, gate, up);

    if (tid == 0) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * params.intermediate_size + out_col] = half(silu * up);
    }
}

static inline void uocr_dense_swiglu_partition256_sum2(float value0,
                                                       float value1,
                                                       threadgroup float *partials,
                                                       uint group,
                                                       uint groups,
                                                       uint local_tid,
                                                       uint simd_width,
                                                       thread float &out0,
                                                       thread float &out1) {
    const uint threads_per_column = 256u;
    const uint lane = uocr_simd_lane_from_tid(local_tid, simd_width);
    const uint simdgroup = uocr_simdgroup_from_tid(local_tid, simd_width);
    const uint simdgroups = uocr_simdgroups_for_threadgroup(threads_per_column, simd_width);
    threadgroup float *partials0 = partials + group * simdgroups;
    threadgroup float *partials1 = partials + groups * simdgroups + group * simdgroups;
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
        if (local_tid < active - upper) {
            partials0[local_tid] += partials0[local_tid + upper];
            partials1[local_tid] += partials1[local_tid + upper];
        }
        active = upper;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out0 = partials0[0];
    out1 = partials1[0];
}

// Decode-only shared-expert gate/up tiling.  Four columns share one
// threadgroup while each column still receives the same 256-thread
// dot-product partition/reduction shape as uocr_dense_swiglu_gate_up_f16,
// preserving the fp32 accumulation pattern before the fp16 SwiGLU
// intermediate store.  Threadgroup hidden staging was tested separately and
// left out because the extra uniform barrier outweighed reuse on the current
// GPU; the tiled columns still hit the same hidden row in cache.
[[max_total_threads_per_threadgroup(1024)]] kernel void uocr_dense_swiglu_shared_gate_up_tile4_f16(device const half *src [[buffer(0)]],
                                          device const half *gate_weight [[buffer(1)]],
                                          device const half *up_weight [[buffer(2)]],
                                          device half *mid [[buffer(3)]],
                                          constant UocrDenseSwigluParams &params [[buffer(4)]],
                                          threadgroup float *partials [[threadgroup(0)]],
                                          uint output_tile [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]],
                                          uint simd_width [[threads_per_simdgroup]]) {
    const uint tile_columns = 4u;
    const uint threads_per_column = 256u;
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint intermediate_size = uocr_fc_moe_shared_intermediate_or(params.intermediate_size);
    const uint tiles_per_token = uocr_div_up_u32(intermediate_size, tile_columns);
    const uint token = output_tile / tiles_per_token;
    const uint tile = output_tile - token * tiles_per_token;
    if (token >= params.n_tokens || ntg != tile_columns * threads_per_column || hidden_size != 1280u) {
        return;
    }

    const uint src_base = token * hidden_size;
    const uint group = tid / threads_per_column;
    const uint local_tid = tid - group * threads_per_column;
    const uint out_col = tile * tile_columns + group;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (out_col < intermediate_size) {
        const uint weight_base = out_col * hidden_size;
        for (uint k = local_tid; k < hidden_size; k += threads_per_column) {
            const float x = float(src[src_base + k]);
            gate_sum += x * float(gate_weight[weight_base + k]);
            up_sum += x * float(up_weight[weight_base + k]);
        }
    }
    float gate = 0.0f;
    float up = 0.0f;
    uocr_dense_swiglu_partition256_sum2(gate_sum,
                                        up_sum,
                                        partials,
                                        group,
                                        tile_columns,
                                        local_tid,
                                        simd_width,
                                        gate,
                                        up);

    if (local_tid == 0u && out_col < intermediate_size) {
        const float silu = gate / (1.0f + exp(-gate));
        mid[token * intermediate_size + out_col] = half(silu * up);
    }
}

static inline float uocr_dense_swiglu_down_dot_f16(device const half *mid,
                                                   device const half *down_weight,
                                                   constant UocrDenseSwigluParams &params,
                                                   uint token,
                                                   uint out_col,
                                                   uint tid,
                                                   uint ntg,
                                              uint simd_width,
                                                   threadgroup float *partials) {
    float sum = 0.0f;
    const uint mid_base = token * params.intermediate_size;
    const uint weight_base = out_col * params.intermediate_size;
    for (uint k = tid; k < params.intermediate_size; k += ntg) {
        sum += float(mid[mid_base + k]) * float(down_weight[weight_base + k]);
    }
    return uocr_threadgroup_sum(sum, partials, tid, ntg, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_down_f16_to_f16(device const half *mid [[buffer(0)]],
                                              device const half *down_weight [[buffer(1)]],
                                              device const half *residual [[buffer(2)]],
                                              device half *dst [[buffer(3)]],
                                              constant UocrDenseSwigluParams &params [[buffer(4)]],
                                              threadgroup float *partials [[threadgroup(0)]],
                                              uint output_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]],
                                              uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    float value = uocr_dense_swiglu_down_dot_f16(mid, down_weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        const uint dst_index = token * hidden_size + out_col;
        if (params.has_residual != 0u) {
            value += float(residual[dst_index]);
        }
        dst[dst_index] = half(value);
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_dense_swiglu_down_f16_to_f32(device const half *mid [[buffer(0)]],
                                              device const half *down_weight [[buffer(1)]],
                                              device const half *residual [[buffer(2)]],
                                              device float *dst [[buffer(3)]],
                                              constant UocrDenseSwigluParams &params [[buffer(4)]],
                                              threadgroup float *partials [[threadgroup(0)]],
                                              uint output_index [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint ntg [[threads_per_threadgroup]],
                                              uint simd_width [[threads_per_simdgroup]]) {
    const uint hidden_size = uocr_fc_hidden_size_or(params.hidden_size);
    const uint token = output_index / hidden_size;
    const uint out_col = output_index - token * hidden_size;
    if (token >= params.n_tokens || out_col >= hidden_size) {
        return;
    }

    float value = uocr_dense_swiglu_down_dot_f16(mid, down_weight, params, token, out_col, tid, ntg, simd_width, partials);
    if (tid == 0) {
        const uint dst_index = token * hidden_size + out_col;
        if (params.has_residual != 0u) {
            value += float(residual[dst_index]);
        }
        dst[dst_index] = value;
    }
}

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

kernel void uocr_moe_selected_down_sum_f16_to_f32(device const half *mid [[buffer(0)]],
                                                  device const half *down_weight [[buffer(1)]],
                                                  device const float *top_weights [[buffer(2)]],
                                                  device float *dst [[buffer(3)]],
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
        dst[out_col] = value;
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

kernel void uocr_moe_prefill_selected_down_sum_f16_to_f32(device const half *mid [[buffer(0)]],
                                                          device const uint *top_expert_ids [[buffer(1)]],
                                                          device const float *top_weights [[buffer(2)]],
                                                          device const half *down_weight [[buffer(3)]],
                                                          device float *dst [[buffer(4)]],
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
        dst[output_index] = value;
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

kernel void uocr_moe_prefill_interleaved_down_sum_f16_to_f32(device const half *mid [[buffer(0)]],
                                                             device const uint *top_expert_ids [[buffer(1)]],
                                                             device const float *top_weights [[buffer(2)]],
                                                             device const half *expert_slab [[buffer(3)]],
                                                             device float *dst [[buffer(4)]],
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
        dst[output_index] = value;
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
                                  uint heads,
                                  uint head_dim,
                                  float freq_scale,
                                  thread uint &token,
                                  thread uint &head,
                                  thread uint &a,
                                  thread uint &b,
                                  thread float &c,
                                  thread float &s) {
    const uint half_dim = head_dim >> 1u;
    const uint pair = gid % half_dim;
    const uint token_head = gid / half_dim;
    head = token_head % heads;
    token = token_head / heads;
    a = pair;
    b = pair + half_dim;
    const float angle = float(params.position_start + token) * exp2(float(pair) * freq_scale);
    c = cos(angle);
    s = sin(angle);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rope_qk_f16_to_f16(device const half *q_src [[buffer(0)]],
                                    device const half *k_src [[buffer(1)]],
                                    device half *q_dst [[buffer(2)]],
                                    device half *k_dst [[buffer(3)]],
                                    constant UocrRopeQKParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const float freq_scale = uocr_fc_rope_freq_scale_or(params.freq_scale);
    const uint half_dim = head_dim >> 1u;
    const uint total = params.n_tokens * heads * half_dim;
    if (gid >= total) {
        return;
    }

    uint token;
    uint head;
    uint a;
    uint b;
    float c;
    float s;
    uocr_rope_pair(gid, params, heads, head_dim, freq_scale, token, head, a, b, c, s);
    const ulong base = (ulong(token) * ulong(heads) + ulong(head)) * ulong(head_dim);
    const float q0 = float(q_src[base + ulong(a)]);
    const float q1 = float(q_src[base + ulong(b)]);
    const float k0 = float(k_src[base + ulong(a)]);
    const float k1 = float(k_src[base + ulong(b)]);
    q_dst[base + ulong(a)] = half(q0 * c - q1 * s);
    q_dst[base + ulong(b)] = half(q0 * s + q1 * c);
    k_dst[base + ulong(a)] = half(k0 * c - k1 * s);
    k_dst[base + ulong(b)] = half(k0 * s + k1 * c);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rope_qk_f16_to_f32(device const half *q_src [[buffer(0)]],
                                    device const half *k_src [[buffer(1)]],
                                    device float *q_dst [[buffer(2)]],
                                    device float *k_dst [[buffer(3)]],
                                    constant UocrRopeQKParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const float freq_scale = uocr_fc_rope_freq_scale_or(params.freq_scale);
    const uint half_dim = head_dim >> 1u;
    const uint total = params.n_tokens * heads * half_dim;
    if (gid >= total) {
        return;
    }

    uint token;
    uint head;
    uint a;
    uint b;
    float c;
    float s;
    uocr_rope_pair(gid, params, heads, head_dim, freq_scale, token, head, a, b, c, s);
    const ulong base = (ulong(token) * ulong(heads) + ulong(head)) * ulong(head_dim);
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
                                                     uint ntg [[threads_per_threadgroup]],
                                                     uint simd_width [[threads_per_simdgroup]]) {
    const uint token = group_index / params.heads;
    const uint head = group_index - token * params.heads;
    if (token >= params.total_tokens || head >= params.heads || ntg < params.head_dim) {
        return;
    }
    const bool dim_valid = tid < params.head_dim;

    uint seq_start;
    uint seq_end;
    uocr_prefill_varlen_find_sequence(cu, params, token, seq_start, seq_end);
    if (seq_end <= seq_start) {
        return;
    }

    const ulong q_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
    const float q = dim_valid ? float(q_src[q_index]) : 0.0f;
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    for (uint key_token = seq_start; key_token <= token; ++key_token) {
        const ulong k_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        const float local_dot = dim_valid ? q * float(k_src[k_index]) : 0.0f;
        const float score = uocr_threadgroup_sum(local_dot, partials, tid, ntg, simd_width) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        if (dim_valid) {
            acc = acc * corr + e * float(v_src[v_index]);
        }
        l = l * corr + e;
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (dim_valid) {
        const ulong dst_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        dst[dst_index] = half(l > 0.0f ? acc / l : 0.0f);
    }
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
                                                     uint ntg [[threads_per_threadgroup]],
                                                     uint simd_width [[threads_per_simdgroup]]) {
    const uint token = group_index / params.heads;
    const uint head = group_index - token * params.heads;
    if (token >= params.total_tokens || head >= params.heads || ntg < params.head_dim) {
        return;
    }
    const bool dim_valid = tid < params.head_dim;

    uint seq_start;
    uint seq_end;
    uocr_prefill_varlen_find_sequence(cu, params, token, seq_start, seq_end);
    if (seq_end <= seq_start) {
        return;
    }

    const ulong q_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
    const float q = dim_valid ? float(q_src[q_index]) : 0.0f;
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    for (uint key_token = seq_start; key_token <= token; ++key_token) {
        const ulong k_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        const float local_dot = dim_valid ? q * float(k_src[k_index]) : 0.0f;
        const float score = uocr_threadgroup_sum(local_dot, partials, tid, ntg, simd_width) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        const ulong v_index = (ulong(key_token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        if (dim_valid) {
            acc = acc * corr + e * float(v_src[v_index]);
        }
        l = l * corr + e;
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (dim_valid) {
        const ulong dst_index = (ulong(token) * ulong(params.heads) + ulong(head)) * ulong(params.head_dim) + ulong(tid);
        dst[dst_index] = l > 0.0f ? acc / l : 0.0f;
    }
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

/* OCR adapts the gradients.c sdpa_decode prefix/window rule to a compact
 * attention stream: [all prompt prefix tokens] followed by the live generated
 * window. The generated window is a 128-token ring whose logical first token is
 * supplied by the host as first_generated.
 */
static inline uint uocr_decode_attention_cache_token(constant UocrDecodeAttentionParams &params,
                                                     uint attention_index,
                                                     uint ring_window) {
    if (attention_index < params.prompt_length) {
        return attention_index;
    }
    const uint generated_index = params.first_generated + (attention_index - params.prompt_length);
    return params.prompt_length + (generated_index % ring_window);
}

static inline ulong uocr_decode_attention_cache_index(constant UocrDecodeAttentionParams &params,
                                                      uint heads,
                                                      uint head_dim,
                                                      uint cache_token,
                                                      uint head,
                                                      uint dim) {
    return (((ulong(params.layer) * ulong(params.batch_slots) + ulong(params.slot)) *
             ulong(params.cache_token_capacity) + ulong(cache_token)) *
            ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
}

#define UOCR_FLASH_Q_PER_TG 4u
#define UOCR_FLASH_MAX_LANE_VALUES 4u
#define UOCR_FLASH_NEG_INF (-3.4028234663852886e38f)

static inline uint uocr_flash_lane_dim(uint lane, uint component, uint simd_width) {
    return lane + component * simd_width;
}

template <typename out_t>
static inline void uocr_sam_window_attention_flash_impl(device const half *q_src,
                                                        device const half *k_src,
                                                        device const half *v_src,
                                                        device out_t *dst,
                                                        constant UocrSamWindowAttentionParams &params,
                                                        uint3 tg,
                                                        ushort lane_u16,
                                                        ushort simdgroup_u16,
                                                        ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint batch_size = uocr_sam_window_attention_batch_size(params);
    const uint logical_window = tg.z;
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || params.windows == 0u || batch >= batch_size || window >= params.windows ||
        query_token >= params.tokens_per_window || head >= params.heads ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong q_index = uocr_sam_window_attention_index(params, batch, window, query_token, head, dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token < params.tokens_per_window; ++key_token) {
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong k_index = uocr_sam_window_attention_index(params, batch, window, key_token, head, dim);
                local_dot += qv[i] * float(k_src[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * params.scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong v_index = uocr_sam_window_attention_index(params, batch, window, key_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong dst_index = uocr_sam_window_attention_index(params, batch, window, query_token, head, dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

kernel void uocr_sam_window_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                       device const half *k_src [[buffer(1)]],
                                                       device const half *v_src [[buffer(2)]],
                                                       device half *dst [[buffer(3)]],
                                                       constant UocrSamWindowAttentionParams &params [[buffer(4)]],
                                                       uint3 tg [[threadgroup_position_in_grid]],
                                                       ushort lane [[thread_index_in_simdgroup]],
                                                       ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                       ushort simd_width [[threads_per_simdgroup]]) {
    uocr_sam_window_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup, simd_width);
}

kernel void uocr_sam_window_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                       device const half *k_src [[buffer(1)]],
                                                       device const half *v_src [[buffer(2)]],
                                                       device float *dst [[buffer(3)]],
                                                       constant UocrSamWindowAttentionParams &params [[buffer(4)]],
                                                       uint3 tg [[threadgroup_position_in_grid]],
                                                       ushort lane [[thread_index_in_simdgroup]],
                                                       ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                       ushort simd_width [[threads_per_simdgroup]]) {
    uocr_sam_window_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup, simd_width);
}

template <typename out_t>
static inline void uocr_sam_rel_pos_attention_flash_impl(device const half *q_src,
                                                         device const half *k_src,
                                                         device const half *v_src,
                                                         device const half *rel_pos_h,
                                                         device const half *rel_pos_w,
                                                         device out_t *dst,
                                                         constant UocrSamRelPosAttentionParams &params,
                                                         uint3 tg,
                                                         ushort lane_u16,
                                                         ushort simdgroup_u16,
                                                         ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint batch_size = uocr_sam_rel_pos_attention_batch_size(params);
    const uint logical_window = tg.z;
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || params.windows == 0u || batch >= batch_size || window >= params.windows ||
        query_token >= params.tokens_per_window || head >= params.heads ||
        params.grid_width == 0u || params.grid_height == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    const uint query_y = query_token / params.grid_width;
    const uint query_x = query_token - query_y * params.grid_width;
    const uint target_h_length = 2u * params.grid_height - 1u;
    const uint target_w_length = 2u * params.grid_width - 1u;

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong q_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token < params.tokens_per_window; ++key_token) {
        const uint key_y = key_token / params.grid_width;
        const uint key_x = key_token - key_y * params.grid_width;
        const uint rel_h_index = uint(int(query_y) - int(key_y) + int(params.grid_height) - 1);
        const uint rel_w_index = uint(int(query_x) - int(key_x) + int(params.grid_width) - 1);
        float local_score = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong k_index = uocr_sam_rel_pos_attention_index(params, batch, window, key_token, head, dim);
                const float rh = uocr_sam_rel_pos_table_value(rel_pos_h,
                                                              params.rel_pos_h_length,
                                                              target_h_length,
                                                              rel_h_index,
                                                              dim,
                                                              params.head_dim);
                const float rw = uocr_sam_rel_pos_table_value(rel_pos_w,
                                                              params.rel_pos_w_length,
                                                              target_w_length,
                                                              rel_w_index,
                                                              dim,
                                                              params.head_dim);
                local_score += qv[i] * (float(k_src[k_index]) * params.scale + rh + rw);
            }
        }
        const float score = simd_sum(local_score);
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < params.head_dim) {
                const ulong v_index = uocr_sam_rel_pos_attention_index(params, batch, window, key_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < params.head_dim) {
            const ulong dst_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

kernel void uocr_sam_rel_pos_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                        device const half *k_src [[buffer(1)]],
                                                        device const half *v_src [[buffer(2)]],
                                                        device const half *rel_pos_h [[buffer(3)]],
                                                        device const half *rel_pos_w [[buffer(4)]],
                                                        device half *dst [[buffer(5)]],
                                                        constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                        uint3 tg [[threadgroup_position_in_grid]],
                                                        ushort lane [[thread_index_in_simdgroup]],
                                                        ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                        ushort simd_width [[threads_per_simdgroup]]) {
    uocr_sam_rel_pos_attention_flash_impl(q_src, k_src, v_src, rel_pos_h, rel_pos_w, dst, params, tg, lane, simdgroup, simd_width);
}

kernel void uocr_sam_rel_pos_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                        device const half *k_src [[buffer(1)]],
                                                        device const half *v_src [[buffer(2)]],
                                                        device const half *rel_pos_h [[buffer(3)]],
                                                        device const half *rel_pos_w [[buffer(4)]],
                                                        device float *dst [[buffer(5)]],
                                                        constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                        uint3 tg [[threadgroup_position_in_grid]],
                                                        ushort lane [[thread_index_in_simdgroup]],
                                                        ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                        ushort simd_width [[threads_per_simdgroup]]) {
    uocr_sam_rel_pos_attention_flash_impl(q_src, k_src, v_src, rel_pos_h, rel_pos_w, dst, params, tg, lane, simdgroup, simd_width);
}

#define UOCR_SAM_REL_POS_TILE_KEYS 16u

kernel void uocr_sam_rel_pos_attention_tiled_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                               device const half *k_src [[buffer(1)]],
                                                               device const half *v_src [[buffer(2)]],
                                                               device const half *rel_pos_h [[buffer(3)]],
                                                               device const half *rel_pos_w [[buffer(4)]],
                                                               device half *dst [[buffer(5)]],
                                                               constant UocrSamRelPosAttentionParams &params [[buffer(6)]],
                                                               threadgroup half *k_tile [[threadgroup(0)]],
                                                               threadgroup half *v_tile [[threadgroup(1)]],
                                                               uint3 tg [[threadgroup_position_in_grid]],
                                                               uint3 tid3 [[thread_position_in_threadgroup]],
                                                               uint3 ntg3 [[threads_per_threadgroup]],
                                                               ushort lane_u16 [[thread_index_in_simdgroup]],
                                                               ushort simdgroup_u16 [[simdgroup_index_in_threadgroup]],
                                                               ushort simd_width_u16 [[threads_per_simdgroup]]) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint tid = tid3.x;
    const uint ntg = ntg3.x;
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    const uint batch_size = uocr_sam_rel_pos_attention_batch_size(params);
    const uint logical_window = tg.z;
    const uint batch = params.windows == 0u ? 0u : logical_window / params.windows;
    const uint window = params.windows == 0u ? 0u : logical_window - batch * params.windows;
    const uint target_h_length = 2u * params.grid_height - 1u;
    const uint target_w_length = 2u * params.grid_width - 1u;
    if (params.windows == 0u || batch >= batch_size || window >= params.windows || head >= params.heads ||
        params.grid_width == 0u || params.grid_height == 0u ||
        params.tokens_per_window != params.grid_width * params.grid_height ||
        params.rel_pos_h_length != target_h_length || params.rel_pos_w_length != target_w_length ||
        params.head_dim == 0u || params.head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    const bool query_valid = query_in_block < UOCR_FLASH_Q_PER_TG && query_token < params.tokens_per_window;
    const uint query_y = query_valid ? query_token / params.grid_width : 0u;
    const uint query_x = query_valid ? query_token - query_y * params.grid_width : 0u;

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (query_valid && dim < params.head_dim) {
            const ulong q_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint tile_start = 0u; tile_start < params.tokens_per_window; tile_start += UOCR_SAM_REL_POS_TILE_KEYS) {
        const uint tile_count = min(UOCR_SAM_REL_POS_TILE_KEYS, params.tokens_per_window - tile_start);
        const uint tile_values = tile_count * params.head_dim;
        for (uint tile_index = tid; tile_index < tile_values; tile_index += ntg) {
            const uint key_offset = tile_index / params.head_dim;
            const uint dim = tile_index - key_offset * params.head_dim;
            const uint key_token = tile_start + key_offset;
            const ulong kv_index = uocr_sam_rel_pos_attention_index(params, batch, window, key_token, head, dim);
            k_tile[tile_index] = k_src[kv_index];
            v_tile[tile_index] = v_src[kv_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key_offset = 0u; key_offset < tile_count; ++key_offset) {
            const uint key_token = tile_start + key_offset;
            const uint key_y = key_token / params.grid_width;
            const uint key_x = key_token - key_y * params.grid_width;
            const uint rel_h_index = query_valid ? uint(int(query_y) - int(key_y) + int(params.grid_height) - 1) : 0u;
            const uint rel_w_index = query_valid ? uint(int(query_x) - int(key_x) + int(params.grid_width) - 1) : 0u;
            float local_score = 0.0f;
            for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
                const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
                if (query_valid && dim < params.head_dim) {
                    const float rh = float(rel_pos_h[ulong(rel_h_index) * ulong(params.head_dim) + ulong(dim)]);
                    const float rw = float(rel_pos_w[ulong(rel_w_index) * ulong(params.head_dim) + ulong(dim)]);
                    local_score += qv[i] * (float(k_tile[key_offset * params.head_dim + dim]) * params.scale + rh + rw);
                }
            }
            const float score = query_valid ? simd_sum(local_score) : UOCR_FLASH_NEG_INF;
            const float mnew = max(m, score);
            const float corr = exp(m - mnew);
            const float e = exp(score - mnew);
            for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
                const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
                if (query_valid && dim < params.head_dim) {
                    acc[i] = acc[i] * corr + e * float(v_tile[key_offset * params.head_dim + dim]);
                }
            }
            l = l * corr + e;
            m = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (query_valid && dim < params.head_dim) {
            const ulong dst_index = uocr_sam_rel_pos_attention_index(params, batch, window, query_token, head, dim);
            dst[dst_index] = half(acc[i] * inv_l);
        }
    }
}

template <typename out_t>
static inline void uocr_prefill_attention_flash_impl(device const half *q_src,
                                                     device const half *k_src,
                                                     device const half *v_src,
                                                     device out_t *dst,
                                                     constant UocrPrefillAttentionParams &params,
                                                     uint2 tg,
                                                     ushort lane_u16,
                                                     ushort simdgroup_u16,
                                                     ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const float scale = uocr_fc_attention_scale_or(params.scale);
    const uint query_in_block = uint(simdgroup_u16);
    const uint head = tg.x;
    const uint query_token = tg.y * UOCR_FLASH_Q_PER_TG + query_in_block;
    if (query_in_block >= UOCR_FLASH_Q_PER_TG || query_token >= params.n_tokens || head >= heads ||
        head_dim == 0u || head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong q_index = (ulong(query_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint key_token = 0u; key_token <= query_token; ++key_token) {
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong k_index = (ulong(key_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
                local_dot += qv[i] * float(k_src[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong v_index = (ulong(key_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
                acc[i] = acc[i] * corr + e * float(v_src[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong dst_index = (ulong(query_token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

[[max_total_threads_per_threadgroup(512)]] kernel void uocr_prefill_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                    device const half *k_src [[buffer(1)]],
                                                    device const half *v_src [[buffer(2)]],
                                                    device half *dst [[buffer(3)]],
                                                    constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                                    uint2 tg [[threadgroup_position_in_grid]],
                                                    ushort lane [[thread_index_in_simdgroup]],
                                                    ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                    ushort simd_width [[threads_per_simdgroup]]) {
    uocr_prefill_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup, simd_width);
}

[[max_total_threads_per_threadgroup(512)]] kernel void uocr_prefill_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                    device const half *k_src [[buffer(1)]],
                                                    device const half *v_src [[buffer(2)]],
                                                    device float *dst [[buffer(3)]],
                                                    constant UocrPrefillAttentionParams &params [[buffer(4)]],
                                                    uint2 tg [[threadgroup_position_in_grid]],
                                                    ushort lane [[thread_index_in_simdgroup]],
                                                    ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                                    ushort simd_width [[threads_per_simdgroup]]) {
    uocr_prefill_attention_flash_impl(q_src, k_src, v_src, dst, params, tg, lane, simdgroup, simd_width);
}

template <typename out_t>
static inline void uocr_decode_attention_flash_impl(device const half *q_src,
                                                    device const half *k_cache,
                                                    device const half *v_cache,
                                                    device out_t *dst,
                                                    constant UocrDecodeAttentionParams &params,
                                                    uint head,
                                                    ushort lane_u16,
                                                    ushort simd_width_u16) {
    const uint lane = uint(lane_u16);
    const uint simd_width = uint(simd_width_u16);
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const uint ring_window = uocr_fc_ring_window_or(params.ring_window);
    const float scale = uocr_fc_attention_scale_or(params.scale);
    if (head >= heads || params.attention_length == 0u ||
        head_dim == 0u || head_dim > simd_width * UOCR_FLASH_MAX_LANE_VALUES) {
        return;
    }

    float qv[UOCR_FLASH_MAX_LANE_VALUES];
    float acc[UOCR_FLASH_MAX_LANE_VALUES];
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong q_index = ulong(head) * ulong(head_dim) + ulong(dim);
            qv[i] = float(q_src[q_index]);
        } else {
            qv[i] = 0.0f;
        }
        acc[i] = 0.0f;
    }

    float m = UOCR_FLASH_NEG_INF;
    float l = 0.0f;
    for (uint attention_index = 0u; attention_index < params.attention_length; ++attention_index) {
        const uint cache_token = uocr_decode_attention_cache_token(params, attention_index, ring_window);
        if (cache_token >= params.cache_token_capacity) {
            continue;
        }
        float local_dot = 0.0f;
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong k_index = uocr_decode_attention_cache_index(params, heads, head_dim, cache_token, head, dim);
                local_dot += qv[i] * float(k_cache[k_index]);
            }
        }
        const float score = simd_sum(local_dot) * scale;
        const float mnew = max(m, score);
        const float corr = exp(m - mnew);
        const float e = exp(score - mnew);
        for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
            const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
            if (dim < head_dim) {
                const ulong v_index = uocr_decode_attention_cache_index(params, heads, head_dim, cache_token, head, dim);
                acc[i] = acc[i] * corr + e * float(v_cache[v_index]);
            }
        }
        l = l * corr + e;
        m = mnew;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint i = 0u; i < UOCR_FLASH_MAX_LANE_VALUES; ++i) {
        const uint dim = uocr_flash_lane_dim(lane, i, simd_width);
        if (dim < head_dim) {
            const ulong dst_index = ulong(head) * ulong(head_dim) + ulong(dim);
            dst[dst_index] = out_t(acc[i] * inv_l);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_decode_attention_flash_f16_to_f16(device const half *q_src [[buffer(0)]],
                                                   device const half *k_cache [[buffer(1)]],
                                                   device const half *v_cache [[buffer(2)]],
                                                   device half *dst [[buffer(3)]],
                                                   constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                                   uint head [[threadgroup_position_in_grid]],
                                                   ushort lane [[thread_index_in_simdgroup]],
                                                   ushort simd_width [[threads_per_simdgroup]]) {
    uocr_decode_attention_flash_impl(q_src, k_cache, v_cache, dst, params, head, lane, simd_width);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_decode_attention_flash_f16_to_f32(device const half *q_src [[buffer(0)]],
                                                   device const half *k_cache [[buffer(1)]],
                                                   device const half *v_cache [[buffer(2)]],
                                                   device float *dst [[buffer(3)]],
                                                   constant UocrDecodeAttentionParams &params [[buffer(4)]],
                                                   uint head [[threadgroup_position_in_grid]],
                                                   ushort lane [[thread_index_in_simdgroup]],
                                                   ushort simd_width [[threads_per_simdgroup]]) {
    uocr_decode_attention_flash_impl(q_src, k_cache, v_cache, dst, params, head, lane, simd_width);
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

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_kv_cache_write_f16(device const half *k_src [[buffer(0)]],
                                    device const half *v_src [[buffer(1)]],
                                    device half *k_cache [[buffer(2)]],
                                    device half *v_cache [[buffer(3)]],
                                    constant UocrKVCacheWriteParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const uint ring_window = uocr_fc_ring_window_or(params.ring_window);
    const uint vec_head_dim = uocr_div_up_u32(head_dim, UOCR_HALF4_WIDTH);
    const uint head_area_vec = heads * vec_head_dim;
    const uint total_vec = params.n_tokens * head_area_vec;
    if (gid >= total_vec || vec_head_dim == 0u) {
        return;
    }

    const uint dim_vec = gid % vec_head_dim;
    const uint dim = dim_vec * UOCR_HALF4_WIDTH;
    const uint head = (gid / vec_head_dim) % heads;
    const uint token = gid / head_area_vec;
    const uint position = params.position_start + token;
    uint cache_token = position;
    if (position >= params.prompt_length) {
        cache_token = params.prompt_length + ((position - params.prompt_length) % ring_window);
    }
    if (cache_token >= params.cache_token_capacity) {
        return;
    }

    const ulong src_index = (ulong(token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
    const ulong dst_index = (((ulong(params.layer) * ulong(params.batch_slots) + ulong(params.slot)) *
                              ulong(params.cache_token_capacity) + ulong(cache_token)) *
                             ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
    if (dim + (UOCR_HALF4_WIDTH - 1u) < head_dim) {
        uocr_store_half4(k_cache, dst_index, uocr_load_half4(k_src, src_index));
        uocr_store_half4(v_cache, dst_index, uocr_load_half4(v_src, src_index));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && dim + i < head_dim; ++i) {
            k_cache[dst_index + ulong(i)] = k_src[src_index + ulong(i)];
            v_cache[dst_index + ulong(i)] = v_src[src_index + ulong(i)];
        }
    }
}

#if UOCR_METAL_ENABLE_MPP_TENSOROPS
/*
 * Experimental Metal 4 MPP TensorOps prototype. Production dispatch keeps using
 * MPSNDArrayMatrixMultiplication as the portable large-GEMM path; this kernel is
 * opt-in so MTLTensor host binding and device support can be evaluated without
 * affecting default builds.
 */
kernel void uocr_mpp_matmul2d_f16_64x32_prototype(tensor<device half, dextents<int, 2>> a [[buffer(0)]],
                                                   tensor<device half, dextents<int, 2>> b [[buffer(1)]],
                                                   tensor<device half, dextents<int, 2>> c [[buffer(2)]],
                                                   uint2 tgid [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(64, 32, 0);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> op;

    auto tile_a = a.slice(0, int(tgid.y) * 64);
    auto tile_b = b.slice(int(tgid.x) * 32, 0);
    auto tile_c = c.slice(int(tgid.x) * 32, int(tgid.y) * 64);
    op.run(tile_a, tile_b, tile_c);
}
#endif

/*
 * Transpose BHWC → NCHW for fp16 tensors.
 * input[batch][spatial][channels]  →  output[batch][channels][spatial]
 *
 * Dispatch:  (spatial, channels, batch)  as thread_position_in_grid.
 */
struct UocrTransposeBhwcToNchwDims {
    uint batch_size;
    uint spatial;
    uint channels;
};

kernel void uocr_bhwc_to_nchw_f16(device const half *bhwc [[buffer(0)]],
                                   device half *nchw [[buffer(1)]],
                                   constant UocrTransposeBhwcToNchwDims &dims [[buffer(2)]],
                                   uint3 gid [[thread_position_in_grid]]) {
    const uint s = gid.x;
    const uint c = gid.y;
    const uint b = gid.z;
    if (s >= dims.spatial || c >= dims.channels || b >= dims.batch_size) {
        return;
    }
    const ulong src = ((ulong(b) * ulong(dims.spatial) + ulong(s)) * ulong(dims.channels) + ulong(c));
    const ulong dst = ((ulong(b) * ulong(dims.channels) + ulong(c)) * ulong(dims.spatial) + ulong(s));
    nchw[dst] = bhwc[src];
}

/*
 * Extract non-overlapping 16x16 patches from an NCHW fp16 pixel buffer into
 * row-major [B*grid_h*grid_w, 768] layout for MPS matmul with the SAM patch
 * weight [768, 3*16*16].
 *
 * pixels  — NCHW fp16 [batch, 3, height, width]
 * patches_out — row-major fp16 [batch * grid_h * grid_w, 768], contiguous.
 * inner = channel * 256 + ky * 16 + kx  (0..767).
 *
 * Dispatch:  (inner, batch * patches)  as thread_position_in_grid.
 */
struct UocrExtractPatchesDims {
    uint height;
    uint width;
    uint grid_h;
    uint grid_w;
};

kernel void uocr_sam_extract_patches_f16(device const half *pixels [[buffer(0)]],
                                         device half *patches_out [[buffer(1)]],
                                         constant UocrExtractPatchesDims &dims [[buffer(2)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    const uint inner = gid.x;
    const uint patch = gid.y;
    const uint patches_per_view = dims.grid_h * dims.grid_w;
    if (inner >= 768u || patches_per_view == 0u) {
        return;
    }
    const uint batch = patch / patches_per_view;
    const uint patch_in_view = patch - batch * patches_per_view;
    const uint patch_y = patch_in_view / dims.grid_w;
    const uint patch_x = patch_in_view - patch_y * dims.grid_w;
    const uint c = inner / 256u;
    const uint rem = inner - c * 256u;
    const uint ky = rem / 16u;
    const uint kx = rem - ky * 16u;
    const uint py = patch_y * 16u + ky;
    const uint px = patch_x * 16u + kx;
    if (py >= dims.height || px >= dims.width) {
        return;
    }
    const ulong pixel_offset = (ulong(batch) * 3ul * ulong(dims.height) * ulong(dims.width) +
                                ulong(c) * ulong(dims.height) * ulong(dims.width) +
                                ulong(py) * ulong(dims.width) + ulong(px));
    const ulong patch_offset = ulong(patch) * 768ul + ulong(inner);
    patches_out[patch_offset] = pixels[pixel_offset];
}

struct UocrConv3x3Im2ColParams {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint in_channels;
    uint stride;
    uint batch_size;
    uint reserved;
};

/*
 * Converts NCHW fp16 input into row-major im2col:
 *
 * input:  [B, C, H, W]
 * cols:   [B * OH * OW, C * 3 * 3]
 *
 * Padding behavior matches the current direct kernels:
 * center = output_coord * stride, radius = 1, out-of-bounds = 0.
 */
kernel void uocr_sam_conv3x3_im2col_nchw_f16(device const half *src_nchw [[buffer(0)]],
                                              device half *cols [[buffer(1)]],
                                              constant UocrConv3x3Im2ColParams &params [[buffer(2)]],
                                              uint2 gid [[thread_position_in_grid]]) {
    const uint k = gid.x;
    const uint row = gid.y;

    const uint kernel_elems = 9u;
    const uint k_total = params.in_channels * kernel_elems;
    const uint out_spatial = params.output_width * params.output_height;
    const uint total_rows = params.batch_size * out_spatial;

    if (k >= k_total || row >= total_rows) {
        return;
    }

    const uint batch = row / out_spatial;
    const uint out_spatial_idx = row - batch * out_spatial;
    const uint out_y = out_spatial_idx / params.output_width;
    const uint out_x = out_spatial_idx - out_y * params.output_width;

    const uint in_channel = k / kernel_elems;
    const uint rem = k - in_channel * kernel_elems;
    const uint ky = rem / 3u;
    const uint kx = rem - ky * 3u;

    const int sy = int(out_y * params.stride) + int(ky) - 1;
    const int sx = int(out_x * params.stride) + int(kx) - 1;

    half value = half(0.0h);

    if (sy >= 0 && sy < int(params.input_height) &&
        sx >= 0 && sx < int(params.input_width)) {
        const uint input_spatial = params.input_width * params.input_height;
        const ulong src_index =
            (ulong(batch) * ulong(params.in_channels) + ulong(in_channel)) * ulong(input_spatial) +
            ulong(uint(sy)) * ulong(params.input_width) +
            ulong(uint(sx));
        value = src_nchw[src_index];
    }

    cols[ulong(row) * ulong(k_total) + ulong(k)] = value;
}
