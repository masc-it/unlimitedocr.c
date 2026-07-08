// embedding_q4.metal - Q4_0 embedding lookup kernels
//
// Same shape contract as uocr_get_rows_q8_0_to_f16_h1280_g64; rows use the
// group-half-split nibble packing (docs/plan_q4.md §1.1): byte g*32 + j of a
// row holds weights k = g*64 + j (low nibble) and k + 32 (high nibble).  A
// half4-aligned column run stays inside one nibble half of one scale group.
#include "common.metal"

// Mirrors UocrGetRowsQ8Params (embedding_q8.metal is an earlier fragment).
struct UocrGetRowsQ4Params {
    uint table_rows;
    uint logical_width;
    uint physical_width;
    uint n_row_ids;
    uint row_size; // packed row bytes (width / 2)
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_get_rows_q4_0_to_f16_h1280_g64(device const uchar *qweight [[buffer(0)]],
                                                device const half *qscale [[buffer(1)]],
                                                device const int *row_ids [[buffer(2)]],
                                                device half *dst [[buffer(3)]],
                                                constant UocrGetRowsQ4Params &params [[buffer(4)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    constexpr uint kWidth = 1280u;
    constexpr uint kGroup = 64u;
    constexpr uint kGroupsPerRow = kWidth / kGroup;
    constexpr uint kRowBytes = kWidth / 2u;

    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint out_row = gid.y;
    if (col >= kWidth || out_row >= params.n_row_ids) {
        return;
    }

    const ulong dst_base = (ulong(out_row) * ulong(kWidth)) + ulong(col);
    const int row = row_ids[out_row];
    if (row < 0 || uint(row) >= params.table_rows || params.logical_width != kWidth ||
        params.physical_width != kWidth || params.row_size != kRowBytes) {
        uocr_store_half4(dst, dst_base, uocr_zero_half4());
        return;
    }

    const uint src_row = uint(row);
    const uint group = col / kGroup;
    const uint j = col - group * kGroup; // half4-aligned; run stays in one half
    const bool high = j >= 32u;
    const ulong q_base = (ulong(src_row) * ulong(kRowBytes)) + ulong(group * 32u + (high ? j - 32u : j));
    const float scale = float(qscale[ulong(src_row) * ulong(kGroupsPerRow) + ulong(group)]);
    const uchar4 w = *(device const uchar4 *)(qweight + q_base);
    const int4 q = high ? (int4(w) >> 4) : (int4(w) & 0xF);
    const float4 values = (float4(q) - 8.0f) * scale;
    uocr_store_half4(dst, dst_base, half4(values));
}
