// mpp_prototypes.metal - MPP TensorOps matmul prototype
// Extracted from uocr_smoke.metal
//
#include "common.metal"
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
