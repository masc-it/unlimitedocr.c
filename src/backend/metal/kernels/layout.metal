// layout.metal - Layout conversion, im2col, split QKV
// Extracted from uocr_smoke.metal
//
#include "common.metal"
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
struct UocrSplitQkvParams {
    uint rows;
    uint hidden_size;
};

/*
 * Split a packed QKV buffer into separate Q, K, V.
 *
 * qkv:  [rows, 3 * hidden_size]  row-major,  q | k | v  concatenated along cols
 * q:    [rows, hidden_size]
 * k:    [rows, hidden_size]
 * v:    [rows, hidden_size]
 *
 * Dispatch:  (col, row)  as thread_position_in_grid.
 */
/*
 * Fused packed-QKV bias add + split: q/k/v[row, col] = qkv[row, ...] + bias.
 *
 * qkv:  [rows, 3 * hidden_size]  raw matmul output (no bias)
 * bias: [3 * hidden_size]        packed q | k | v bias
 *
 * Dispatch: (hidden_size / 4, rows) as thread_position_in_grid; hidden_size
 * must be a multiple of 4 (CLIP 1024, SAM 768).
 */
kernel void uocr_split_qkv_bias_f16(device const half *qkv [[buffer(0)]],
                                    device const half *bias [[buffer(1)]],
                                    device half *q [[buffer(2)]],
                                    device half *k [[buffer(3)]],
                                    device half *v [[buffer(4)]],
                                    constant UocrSplitQkvParams &params [[buffer(5)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x * UOCR_HALF4_WIDTH;
    const uint row = gid.y;
    if (row >= params.rows || col >= params.hidden_size) {
        return;
    }

    const uint hidden = params.hidden_size;
    const ulong qkv_base = ulong(row) * ulong(hidden * 3u) + ulong(col);
    const ulong dst = ulong(row) * ulong(hidden) + ulong(col);

    const half4 qv = half4(float4(uocr_load_half4(qkv, qkv_base)) +
                           float4(uocr_load_half4(bias, ulong(col))));
    const half4 kv = half4(float4(uocr_load_half4(qkv, qkv_base + ulong(hidden))) +
                           float4(uocr_load_half4(bias, ulong(hidden + col))));
    const half4 vv = half4(float4(uocr_load_half4(qkv, qkv_base + ulong(hidden * 2u))) +
                           float4(uocr_load_half4(bias, ulong(hidden * 2u + col))));
    uocr_store_half4(q, dst, qv);
    uocr_store_half4(k, dst, kv);
    uocr_store_half4(v, dst, vv);
}

struct UocrConv3x3BhwcIm2ColParams {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint channels;
    uint stride;
    uint batch_size;
    uint reserved;
};

/*
 * BHWC im2col for 3x3 convolutions with half4 vectorized channel writes.
 *
 * input:  [B, H, W, C]  BHWC fp16
 * cols:   [B * OH * OW, 9 * C]  row-major, with kernel position as the
 *         slowest dimension within each row (matching repacked weight layout).
 *
 * Each thread handles 4 channels (half4), dispatched as:
 *   x = 9 * C4_count  (kernel_pos * C4_count + c4_group)
 *   y = total_rows
 */
kernel void uocr_conv3x3_im2col_bhwc_f16(device const half *src_bhwc [[buffer(0)]],
                                         device half *cols [[buffer(1)]],
                                         constant UocrConv3x3BhwcIm2ColParams &p [[buffer(2)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    const uint c4_index = gid.x;
    const uint row = gid.y;

    const uint c4_count = (p.channels + 3u) / 4u;
    const uint kernel_pos = c4_index / c4_count;
    const uint c_base = (c4_index - kernel_pos * c4_count) * 4u;

    const uint out_spatial = p.output_width * p.output_height;
    const uint total_rows = p.batch_size * out_spatial;

    if (kernel_pos >= 9u || row >= total_rows || c_base >= p.channels) {
        return;
    }

    const uint batch = row / out_spatial;
    const uint s = row - batch * out_spatial;
    const uint out_y = s / p.output_width;
    const uint out_x = s - out_y * p.output_width;

    const uint ky = kernel_pos / 3u;
    const uint kx = kernel_pos - ky * 3u;

    const int sy = int(out_y * p.stride) + int(ky) - 1;
    const int sx = int(out_x * p.stride) + int(kx) - 1;

    const ulong k_total = ulong(p.channels) * 9ul;
    const ulong dst_base = ulong(row) * k_total + ulong(kernel_pos) * ulong(p.channels) + ulong(c_base);

    half4 v = half4(half(0.0h));

    if (sy >= 0 && sy < int(p.input_height) && sx >= 0 && sx < int(p.input_width)) {
        const ulong src_base =
            ((ulong(batch) * ulong(p.input_height) + ulong(uint(sy))) * ulong(p.input_width) + ulong(uint(sx))) *
                ulong(p.channels) + ulong(c_base);

        if (c_base + 3u < p.channels) {
            v = half4(*reinterpret_cast<device const packed_half4 *>(src_bhwc + src_base));
            *reinterpret_cast<device packed_half4 *>(cols + dst_base) = packed_half4(v);
            return;
        }
        for (uint i = 0u; i < 4u && c_base + i < p.channels; ++i) {
            cols[dst_base + ulong(i)] = src_bhwc[src_base + ulong(i)];
        }
        return;
    }

    /* Out-of-bounds: write zero */
    if (c_base + 3u < p.channels) {
        *reinterpret_cast<device packed_half4 *>(cols + dst_base) = packed_half4(v);
    } else {
        for (uint i = 0u; i < 4u && c_base + i < p.channels; ++i) {
            cols[dst_base + ulong(i)] = half(0.0h);
        }
    }
}
