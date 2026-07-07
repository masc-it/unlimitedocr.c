// embedding_q8.metal - Q8_0 embedding lookup kernels
#include "common.metal"

struct UocrGetRowsQ8Params {
    uint table_rows;
    uint logical_width;
    uint physical_width;
    uint n_row_ids;
    uint row_size;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void uocr_get_rows_q8_0_to_f16_h1280_g64(device const char *qweight [[buffer(0)]],
                                                device const half *qscale [[buffer(1)]],
                                                device const int *row_ids [[buffer(2)]],
                                                device half *dst [[buffer(3)]],
                                                constant UocrGetRowsQ8Params &params [[buffer(4)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    constexpr uint kWidth = 1280u;
    constexpr uint kGroup = 64u;
    constexpr uint kGroupsPerRow = kWidth / kGroup;

    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint out_row = gid.y;
    if (col >= kWidth || out_row >= params.n_row_ids) {
        return;
    }

    const ulong dst_base = (ulong(out_row) * ulong(kWidth)) + ulong(col);
    const int row = row_ids[out_row];
    if (row < 0 || uint(row) >= params.table_rows || params.logical_width != kWidth ||
        params.physical_width != kWidth || params.row_size != kWidth) {
        uocr_store_half4(dst, dst_base, uocr_zero_half4());
        return;
    }

    const uint src_row = uint(row);
    const ulong q_base = (ulong(src_row) * ulong(kWidth)) + ulong(col);
    const ulong scale_base = ulong(src_row) * ulong(kGroupsPerRow);
    half values[UOCR_HALF4_WIDTH];
    for (uint i = 0u; i < UOCR_HALF4_WIDTH; ++i) {
        const uint c = col + i;
        const uint group = c / kGroup;
        const float scale = float(qscale[scale_base + ulong(group)]);
        const float q = float(int(qweight[q_base + ulong(i)]));
        values[i] = half(q * scale);
    }
    uocr_store_half4(dst, dst_base, half4(values[0], values[1], values[2], values[3]));
}
