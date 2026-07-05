// embedding.metal - Embedding lookup (UocrGetRowsParams, uocr_get_rows_f16_to_f16)
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrGetRowsParams {
    uint table_rows;
    uint row_width;
    uint n_row_ids;
    uint reserved;
};






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
