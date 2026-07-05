// prompt_assembly.metal - Prompt assembly + visual row formatting
// Extracted from uocr_smoke.metal
//
#include "common.metal"
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
