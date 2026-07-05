// clip_sam.metal - CLIP/SAM bridge (UocrClipSamConcatParams, uocr_clip_sam_concat_f16_to_f16)
// Extracted from uocr_smoke.metal
//
#include "common.metal"
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
