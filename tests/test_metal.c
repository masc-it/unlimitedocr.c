#include "backend/metal/uocr_metal.h"
#include "model/uocr_constants.h"
#include "model/uocr_model_file.h"
#include "model/uocr_tensor_registry.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_request_validation.h"
#include "runtime/uocr_sequence.h"
#include "quant/uocr_quant.h"
#include "unlimitedocr.h"

#include "uocr_test_model_file.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef UOCR_TEST_METAL_RESOURCE_PATH
#define UOCR_TEST_METAL_RESOURCE_PATH NULL
#endif

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static float f16_bits_to_f32(uint16_t h);
static float clip_attention_expected(const uint16_t *q,
                                     const uint16_t *k,
                                     const uint16_t *v,
                                     uint32_t tokens,
                                     uint32_t token,
                                     uint32_t head,
                                     uint32_t dim);

static int test_metal_smoke(void) {
    if (!uocr_metal_is_available()) {
        printf("Metal device not available; skipping Metal smoke test\n");
        return 0;
    }

    const uint64_t recommended = uocr_metal_recommended_working_set_size();
    CHECK(recommended > 0u);
    const uint64_t default_budget = uocr_metal_default_memory_budget_bytes(recommended);
    CHECK(default_budget > 0u);
    CHECK(default_budget <= recommended);

    char error[1024];
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_smoke_test(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    return 0;
}

static int test_metal_compile_all_kernels(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(error[0] == '\0');

    const uint32_t function_count = uocr_metal_context_library_function_count(ctx);
    CHECK(function_count > 0u);
    CHECK(uocr_metal_context_pipeline_cache_count(ctx) == 0u);

    uint32_t pipeline_count = 0u;
    CHECK(uocr_metal_context_compile_all_pipelines(ctx, &pipeline_count, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(pipeline_count == function_count);
    CHECK(uocr_metal_context_pipeline_cache_count(ctx) == function_count);

    uint32_t second_count = 0u;
    CHECK(uocr_metal_context_compile_all_pipelines(ctx, &second_count, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(second_count == function_count);
    CHECK(uocr_metal_context_pipeline_cache_count(ctx) == function_count);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static size_t sam_patch_weight_index(uint32_t out_channel, uint32_t in_channel, uint32_t ky, uint32_t kx) {
    return (size_t)((((out_channel * 3u + in_channel) * UOCR_VISION_PATCH_SIZE + ky) * UOCR_VISION_PATCH_SIZE + kx));
}

static size_t sam_abs_pos_index(uint32_t grid, uint32_t y, uint32_t x, uint32_t channel) {
    return ((size_t)y * grid + x) * UOCR_SAM_HIDDEN_SIZE + channel;
}

static size_t clip_token_index(uint32_t token, uint32_t channel) {
    return (size_t)token * UOCR_CLIP_HIDDEN_SIZE + channel;
}

static size_t clip_pos_spatial_index(uint32_t source_grid, uint32_t y, uint32_t x, uint32_t channel) {
    return clip_token_index(1u + y * source_grid + x, channel);
}

static size_t sam_qkv_packed_col(uint32_t projection, uint32_t head, uint32_t dim) {
    return (size_t)projection * UOCR_SAM_HIDDEN_SIZE + (size_t)head * UOCR_SAM_HEAD_DIM + dim;
}

static size_t sam_qkv_weight_index(uint32_t projection, uint32_t head, uint32_t dim, uint32_t input_col) {
    return sam_qkv_packed_col(projection, head, dim) * UOCR_SAM_HIDDEN_SIZE + input_col;
}

static size_t sam_qkv_out_index(uint32_t row, uint32_t head, uint32_t dim) {
    return (size_t)row * UOCR_SAM_HIDDEN_SIZE + (size_t)head * UOCR_SAM_HEAD_DIM + dim;
}

static size_t clip_qkv_packed_col(uint32_t projection, uint32_t head, uint32_t dim) {
    return (size_t)projection * UOCR_CLIP_HIDDEN_SIZE + (size_t)head * UOCR_CLIP_HEAD_DIM + dim;
}

static size_t clip_qkv_weight_index(uint32_t projection, uint32_t head, uint32_t dim, uint32_t input_col) {
    return clip_qkv_packed_col(projection, head, dim) * UOCR_CLIP_HIDDEN_SIZE + input_col;
}

static size_t clip_qkv_out_index(uint32_t token, uint32_t head, uint32_t dim) {
    return (size_t)token * UOCR_CLIP_HIDDEN_SIZE + (size_t)head * UOCR_CLIP_HEAD_DIM + dim;
}

static size_t clip_projection_weight_index(uint32_t out_channel, uint32_t in_channel) {
    return (size_t)out_channel * UOCR_CLIP_HIDDEN_SIZE + in_channel;
}

static size_t visual_projector_weight_index(uint32_t out_channel, uint32_t in_channel) {
    return (size_t)out_channel * UOCR_PROJECTOR_IN_SIZE + in_channel;
}

static size_t clip_mlp_fc1_weight_index(uint32_t out_channel, uint32_t in_channel) {
    return (size_t)out_channel * UOCR_CLIP_HIDDEN_SIZE + in_channel;
}

static size_t clip_mlp_fc2_weight_index(uint32_t out_channel, uint32_t in_channel) {
    return (size_t)out_channel * UOCR_CLIP_MLP_INTERMEDIATE + in_channel;
}

static size_t sam_window_attention_index(uint32_t window, uint32_t token, uint32_t head, uint32_t dim) {
    return (((size_t)window * UOCR_SAM_WINDOW_TOKENS + token) * UOCR_SAM_ATTENTION_HEADS + head) *
               UOCR_SAM_HEAD_DIM +
           dim;
}

static size_t sam_bhwc_index(uint32_t grid_w, uint32_t y, uint32_t x, uint32_t channel) {
    return ((size_t)y * grid_w + x) * UOCR_SAM_HIDDEN_SIZE + channel;
}

static size_t sam_window_partition_index(uint32_t window, uint32_t token, uint32_t channel) {
    return ((size_t)window * UOCR_SAM_WINDOW_TOKENS + token) * UOCR_SAM_HIDDEN_SIZE + channel;
}

static size_t sam_neck_conv1x1_weight_index(uint32_t out_channel, uint32_t in_channel) {
    return (size_t)out_channel * UOCR_SAM_HIDDEN_SIZE + in_channel;
}

static size_t sam_neck_conv3x3_weight_index(uint32_t out_channel, uint32_t in_channel, uint32_t ky, uint32_t kx) {
    return (((size_t)out_channel * UOCR_SAM_NECK_CHANNELS + in_channel) * UOCR_SAM_NECK_KERNEL_SIZE + ky) *
               UOCR_SAM_NECK_KERNEL_SIZE +
           kx;
}

static size_t sam_net2_conv3x3_weight_index(uint32_t out_channel, uint32_t in_channel, uint32_t ky, uint32_t kx) {
    return (((size_t)out_channel * UOCR_SAM_NECK_CHANNELS + in_channel) * UOCR_SAM_NECK_KERNEL_SIZE + ky) *
               UOCR_SAM_NECK_KERNEL_SIZE +
           kx;
}

static size_t sam_net3_conv3x3_weight_index(uint32_t out_channel, uint32_t in_channel, uint32_t ky, uint32_t kx) {
    return (((size_t)out_channel * UOCR_SAM_NET2_CHANNELS + in_channel) * UOCR_SAM_NECK_KERNEL_SIZE + ky) *
               UOCR_SAM_NECK_KERNEL_SIZE +
           kx;
}

static size_t sam_neck_nchw_index(uint32_t grid_w, uint32_t grid_h, uint32_t out_channel, uint32_t y, uint32_t x) {
    return ((size_t)out_channel * grid_h + y) * grid_w + x;
}

static float sam_neck_conv1x1_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t y,
                                       uint32_t x,
                                       uint32_t out_channel);
static float sam_neck_conv3x3_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t y,
                                       uint32_t x,
                                       uint32_t out_channel);
static float sam_layernorm2d_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       const uint16_t *bias,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t y,
                                       uint32_t x,
                                       uint32_t channel);
static float sam_net2_conv3x3_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t out_y,
                                       uint32_t out_x,
                                       uint32_t out_channel);
static float sam_net3_conv3x3_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t out_y,
                                       uint32_t out_x,
                                       uint32_t out_channel);

static size_t sam_global_attention_index(uint32_t token, uint32_t head, uint32_t dim) {
    return ((size_t)token * UOCR_SAM_ATTENTION_HEADS + head) * UOCR_SAM_HEAD_DIM + dim;
}

static size_t sam_rel_pos_attention_index(uint32_t window, uint32_t token, uint32_t head, uint32_t dim, uint32_t tokens) {
    return (((size_t)window * tokens + token) * UOCR_SAM_ATTENTION_HEADS + head) * UOCR_SAM_HEAD_DIM + dim;
}

static size_t sam_mlp_lin1_weight_index(uint32_t out_col, uint32_t in_col) {
    return (size_t)out_col * UOCR_SAM_HIDDEN_SIZE + in_col;
}

static size_t sam_mlp_lin2_weight_index(uint32_t out_col, uint32_t in_col) {
    return (size_t)out_col * UOCR_SAM_MLP_INTERMEDIATE + in_col;
}

static size_t sam_attention_proj_weight_index(uint32_t out_col, uint32_t in_col) {
    return (size_t)out_col * UOCR_SAM_HIDDEN_SIZE + in_col;
}

static float sam_cubic_weight(float x) {
    const float a = -0.75f;
    const float ax = fabsf(x);
    if (ax < 1.0f) {
        return ((a + 2.0f) * ax - (a + 3.0f)) * ax * ax + 1.0f;
    }
    if (ax < 2.0f) {
        return (((a * ax - 5.0f * a) * ax + 8.0f * a) * ax - 4.0f * a);
    }
    return 0.0f;
}

static float sam_abs_pos_reference(const uint16_t *pos,
                                   uint32_t src_grid,
                                   uint32_t target_grid,
                                   uint32_t out_y,
                                   uint32_t out_x,
                                   uint32_t channel) {
    if (src_grid == target_grid) {
        return f16_bits_to_f32(pos[sam_abs_pos_index(src_grid, out_y, out_x, channel)]);
    }
    const float scale = (float)src_grid / (float)target_grid;
    const float filter_scale = fmaxf(scale, 1.0f);
    const float support = 2.0f * filter_scale;
    const float center_y = ((float)out_y + 0.5f) * scale - 0.5f;
    const float center_x = ((float)out_x + 0.5f) * scale - 0.5f;
    int y0 = (int)floorf(center_y - support);
    int y1 = (int)ceilf(center_y + support);
    int x0 = (int)floorf(center_x - support);
    int x1 = (int)ceilf(center_x + support);
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    if (y1 >= (int)src_grid) y1 = (int)src_grid - 1;
    if (x1 >= (int)src_grid) x1 = (int)src_grid - 1;

    float acc = 0.0f;
    float weight_sum = 0.0f;
    for (int iy = y0; iy <= y1; ++iy) {
        const float wy = sam_cubic_weight(((float)iy - center_y) / filter_scale);
        for (int ix = x0; ix <= x1; ++ix) {
            const float wx = sam_cubic_weight(((float)ix - center_x) / filter_scale);
            const float w = wy * wx;
            acc += f16_bits_to_f32(pos[sam_abs_pos_index(src_grid, (uint32_t)iy, (uint32_t)ix, channel)]) * w;
            weight_sum += w;
        }
    }
    return weight_sum != 0.0f ? acc / weight_sum : 0.0f;
}

static float clip_abs_pos_reference(const uint16_t *pos,
                                    uint32_t target_grid,
                                    uint32_t token,
                                    uint32_t channel) {
    if (token == 0u) {
        return f16_bits_to_f32(pos[clip_token_index(0u, channel)]);
    }
    const uint32_t src_grid = UOCR_CLIP_POSITION_GRID;
    const uint32_t spatial = token - 1u;
    const uint32_t out_y = spatial / target_grid;
    const uint32_t out_x = spatial - out_y * target_grid;
    if (src_grid == target_grid) {
        return f16_bits_to_f32(pos[clip_token_index(token, channel)]);
    }
    const float scale = (float)src_grid / (float)target_grid;
    const float filter_scale = fmaxf(scale, 1.0f);
    const float support = 2.0f * filter_scale;
    const float center_y = ((float)out_y + 0.5f) * scale - 0.5f;
    const float center_x = ((float)out_x + 0.5f) * scale - 0.5f;
    int y0 = (int)floorf(center_y - support);
    int y1 = (int)ceilf(center_y + support);
    int x0 = (int)floorf(center_x - support);
    int x1 = (int)ceilf(center_x + support);
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    if (y1 >= (int)src_grid) y1 = (int)src_grid - 1;
    if (x1 >= (int)src_grid) x1 = (int)src_grid - 1;

    float acc = 0.0f;
    float weight_sum = 0.0f;
    for (int iy = y0; iy <= y1; ++iy) {
        const float wy = sam_cubic_weight(((float)iy - center_y) / filter_scale);
        for (int ix = x0; ix <= x1; ++ix) {
            const float wx = sam_cubic_weight(((float)ix - center_x) / filter_scale);
            const float w = wy * wx;
            acc += f16_bits_to_f32(pos[clip_pos_spatial_index(src_grid, (uint32_t)iy, (uint32_t)ix, channel)]) * w;
            weight_sum += w;
        }
    }
    return weight_sum != 0.0f ? acc / weight_sum : 0.0f;
}

static int test_metal_named_scratch_buffers(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) == 0u);
    CHECK(uocr_metal_context_total_scratch_high_watermark(ctx) == 0u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 1024u, 0, error, sizeof(error)) == 1);
    const uint64_t first_capacity = uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(first_capacity >= 1024u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 512u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, first_capacity + 1u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) >= first_capacity + 1u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_LOGITS, 4096u, 1, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS) >= 4096u);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) >=
          uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) +
              uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS));

    uocr_metal_context_release_scratch(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == 0u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16u;
    uint32_t bits = 0u;
    uint32_t mantissa = (uint32_t)h & 0x03ffu;
    int exponent = (int)((h >> 10u) & 0x001fu);
    if (exponent == 0) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            mantissa &= ~0x0400u;
            bits = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
    }
    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0u) {
            return (uint16_t)(sign | 0x7c00u);
        }
        return (uint16_t)(sign | 0x7e00u);
    }

    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return (uint16_t)sign;
        }
        uint32_t mant = mantissa | 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mant >> shift;
        const uint32_t round_bit = 1u << (shift - 1u);
        const uint32_t remainder = mant & (round_bit - 1u);
        if ((mant & round_bit) != 0u && (remainder != 0u || (rounded & 1u) != 0u)) {
            ++rounded;
        }
        return (uint16_t)(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13u;
    const uint32_t round_bit = 0x00001000u;
    const uint32_t remainder = mantissa & (round_bit - 1u);
    if ((mantissa & round_bit) != 0u && (remainder != 0u || (rounded_mantissa & 1u) != 0u)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0u;
            ++half_exponent;
            if (half_exponent >= 31) {
                return (uint16_t)(sign | 0x7c00u);
            }
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10u) | rounded_mantissa);
}

static int test_metal_sam_patch_embed_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    enum { WIDTH = 32u, HEIGHT = 32u, GRID = 2u, PATCH = UOCR_VISION_PATCH_SIZE };
    float *pixels_f32 = (float *)calloc(3u * WIDTH * HEIGHT, sizeof(float));
    uint16_t *weights = (uint16_t *)calloc((size_t)UOCR_SAM_HIDDEN_SIZE * 3u * PATCH * PATCH, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(UOCR_SAM_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *out = (uint16_t *)calloc(GRID * GRID * UOCR_SAM_HIDDEN_SIZE, sizeof(uint16_t));
    CHECK(pixels_f32 != NULL);
    CHECK(weights != NULL);
    CHECK(bias != NULL);
    CHECK(out != NULL);

    for (uint32_t c = 0u; c < 3u; ++c) {
        for (uint32_t y = 0u; y < HEIGHT; ++y) {
            for (uint32_t x = 0u; x < WIDTH; ++x) {
                pixels_f32[(c * HEIGHT + y) * WIDTH + x] = (float)(c * 100u + y + x);
            }
        }
    }
    weights[sam_patch_weight_index(0u, 0u, 0u, 0u)] = f32_to_f16_bits(1.0f);
    weights[sam_patch_weight_index(1u, 1u, 15u, 15u)] = f32_to_f16_bits(1.0f);
    weights[sam_patch_weight_index(2u, 0u, 0u, 0u)] = f32_to_f16_bits(1.0f);
    weights[sam_patch_weight_index(2u, 2u, 1u, 2u)] = f32_to_f16_bits(1.0f);
    bias[2u] = f32_to_f16_bits(3.0f);
    bias[7u] = f32_to_f16_bits(42.0f);

    uint32_t grid_w = 0u;
    uint32_t grid_h = 0u;
    CHECK(uocr_metal_context_sam_patch_embed_f16(ctx,
                                                 pixels_f32,
                                                 UOCR_PIXEL_F32_NCHW,
                                                 WIDTH,
                                                 HEIGHT,
                                                 weights,
                                                 bias,
                                                 out,
                                                 &grid_w,
                                                 &grid_h,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(grid_w == GRID);
    CHECK(grid_h == GRID);

    for (uint32_t py = 0u; py < GRID; ++py) {
        for (uint32_t px = 0u; px < GRID; ++px) {
            const uint32_t patch_index = py * GRID + px;
            const uint32_t base = patch_index * UOCR_SAM_HIDDEN_SIZE;
            const float expected0 = (float)(py * 16u + px * 16u);
            const float expected1 = (float)(100u + py * 16u + 15u + px * 16u + 15u);
            const float expected2 = expected0 + (float)(200u + py * 16u + 1u + px * 16u + 2u) + 3.0f;
            CHECK(out[base + 0u] == f32_to_f16_bits(expected0));
            CHECK(out[base + 1u] == f32_to_f16_bits(expected1));
            CHECK(out[base + 2u] == f32_to_f16_bits(expected2));
            CHECK(out[base + 7u] == f32_to_f16_bits(42.0f));
            CHECK(out[base + 9u] == f32_to_f16_bits(0.0f));
        }
    }

    uint16_t *pixels_f16 = (uint16_t *)calloc(3u * PATCH * PATCH, sizeof(uint16_t));
    CHECK(pixels_f16 != NULL);
    memset(weights, 0, (size_t)UOCR_SAM_HIDDEN_SIZE * 3u * PATCH * PATCH * sizeof(uint16_t));
    memset(out, 0, GRID * GRID * UOCR_SAM_HIDDEN_SIZE * sizeof(uint16_t));
    for (uint32_t c = 0u; c < 3u; ++c) {
        for (uint32_t y = 0u; y < PATCH; ++y) {
            for (uint32_t x = 0u; x < PATCH; ++x) {
                pixels_f16[(c * PATCH + y) * PATCH + x] = f32_to_f16_bits((float)(c * 100u + y + x));
            }
        }
    }
    weights[sam_patch_weight_index(5u, 2u, 3u, 4u)] = f32_to_f16_bits(1.0f);
    CHECK(uocr_metal_context_sam_patch_embed_f16(ctx,
                                                 pixels_f16,
                                                 UOCR_PIXEL_F16_NCHW,
                                                 PATCH,
                                                 PATCH,
                                                 weights,
                                                 NULL,
                                                 out,
                                                 &grid_w,
                                                 &grid_h,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(grid_w == 1u);
    CHECK(grid_h == 1u);
    CHECK(out[5u] == f32_to_f16_bits(207.0f));
    CHECK(out[7u] == f32_to_f16_bits(0.0f));

    CHECK(uocr_metal_context_sam_patch_embed_f16(ctx,
                                                 pixels_f32,
                                                 UOCR_PIXEL_F32_NCHW,
                                                 17u,
                                                 HEIGHT,
                                                 weights,
                                                 bias,
                                                 out,
                                                 &grid_w,
                                                 &grid_h,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "divisible") != NULL);

    free(pixels_f16);
    free(out);
    free(bias);
    free(weights);
    free(pixels_f32);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static void layernorm_expected_hidden(const uint16_t *input,
                                      const uint16_t *weight,
                                      const uint16_t *bias,
                                      uint32_t row,
                                      uint32_t hidden_size,
                                      float eps,
                                      float *out) {
    double sum = 0.0;
    const size_t row_base = (size_t)row * hidden_size;
    for (uint32_t col = 0u; col < hidden_size; ++col) {
        sum += (double)f16_bits_to_f32(input[row_base + col]);
    }
    const float mean = (float)(sum / (double)hidden_size);
    double sq_sum = 0.0;
    for (uint32_t col = 0u; col < hidden_size; ++col) {
        const float centered = f16_bits_to_f32(input[row_base + col]) - mean;
        sq_sum += (double)(centered * centered);
    }
    const float inv_std = 1.0f / sqrtf((float)(sq_sum / (double)hidden_size) + eps);
    for (uint32_t col = 0u; col < hidden_size; ++col) {
        const float normalized = (f16_bits_to_f32(input[row_base + col]) - mean) * inv_std;
        out[col] = normalized * f16_bits_to_f32(weight[col]) + f16_bits_to_f32(bias[col]);
    }
}

static void sam_layernorm_expected(const uint16_t *input,
                                   const uint16_t *weight,
                                   const uint16_t *bias,
                                   uint32_t row,
                                   float eps,
                                   float *out) {
    layernorm_expected_hidden(input, weight, bias, row, UOCR_SAM_HIDDEN_SIZE, eps, out);
}

static void clip_layernorm_expected(const uint16_t *input,
                                    const uint16_t *weight,
                                    const uint16_t *bias,
                                    uint32_t row,
                                    float *out) {
    layernorm_expected_hidden(input, weight, bias, row, UOCR_CLIP_HIDDEN_SIZE, UOCR_CLIP_PRE_LAYERNORM_EPS, out);
}

static int test_metal_sam_abs_pos_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        SRC_GRID = UOCR_GLOBAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE,
        LOCAL_GRID = UOCR_LOCAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE
    };
    const size_t pos_values = (size_t)SRC_GRID * SRC_GRID * UOCR_SAM_HIDDEN_SIZE;
    const size_t global_values = pos_values;
    const size_t local_values = (size_t)LOCAL_GRID * LOCAL_GRID * UOCR_SAM_HIDDEN_SIZE;

    uint16_t *pos = (uint16_t *)calloc(pos_values, sizeof(uint16_t));
    uint16_t *global_patch = (uint16_t *)calloc(global_values, sizeof(uint16_t));
    uint16_t *global_out = (uint16_t *)calloc(global_values, sizeof(uint16_t));
    uint16_t *local_patch = (uint16_t *)calloc(local_values, sizeof(uint16_t));
    uint16_t *local_out = (uint16_t *)calloc(local_values, sizeof(uint16_t));
    CHECK(pos != NULL);
    CHECK(global_patch != NULL);
    CHECK(global_out != NULL);
    CHECK(local_patch != NULL);
    CHECK(local_out != NULL);

    for (uint32_t y = 0u; y < SRC_GRID; ++y) {
        for (uint32_t x = 0u; x < SRC_GRID; ++x) {
            for (uint32_t c = 0u; c < UOCR_SAM_HIDDEN_SIZE; ++c) {
                const float value = 0.001f * (float)y + 0.002f * (float)x + 0.0001f * (float)(c % 17u);
                pos[sam_abs_pos_index(SRC_GRID, y, x, c)] = f32_to_f16_bits(value);
                global_patch[sam_abs_pos_index(SRC_GRID, y, x, c)] =
                    f32_to_f16_bits(-0.25f + 0.00005f * (float)(c % 13u));
            }
        }
    }
    for (uint32_t y = 0u; y < LOCAL_GRID; ++y) {
        for (uint32_t x = 0u; x < LOCAL_GRID; ++x) {
            for (uint32_t c = 0u; c < UOCR_SAM_HIDDEN_SIZE; ++c) {
                local_patch[sam_abs_pos_index(LOCAL_GRID, y, x, c)] =
                    f32_to_f16_bits(0.125f + 0.00003f * (float)((x + y + c) % 19u));
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_add_abs_pos_f16(ctx,
                                                 global_patch,
                                                 pos,
                                                 SRC_GRID,
                                                 SRC_GRID,
                                                 global_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    const uint32_t direct_y[] = {0u, 17u, 63u};
    const uint32_t direct_x[] = {0u, 31u, 63u};
    const uint32_t direct_c[] = {0u, 5u, UOCR_SAM_HIDDEN_SIZE - 1u};
    for (size_t iy = 0u; iy < sizeof(direct_y) / sizeof(direct_y[0]); ++iy) {
        for (size_t ix = 0u; ix < sizeof(direct_x) / sizeof(direct_x[0]); ++ix) {
            for (size_t ic = 0u; ic < sizeof(direct_c) / sizeof(direct_c[0]); ++ic) {
                const size_t idx = sam_abs_pos_index(SRC_GRID, direct_y[iy], direct_x[ix], direct_c[ic]);
                const float expected = f16_bits_to_f32(global_patch[idx]) + f16_bits_to_f32(pos[idx]);
                CHECK(global_out[idx] == f32_to_f16_bits(expected));
            }
        }
    }

    CHECK(uocr_metal_context_sam_add_abs_pos_f16(ctx,
                                                 local_patch,
                                                 pos,
                                                 LOCAL_GRID,
                                                 LOCAL_GRID,
                                                 local_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    const uint32_t interp_y[] = {0u, 7u, LOCAL_GRID - 1u};
    const uint32_t interp_x[] = {0u, 13u, LOCAL_GRID - 1u};
    const uint32_t interp_c[] = {0u, 11u, UOCR_SAM_HIDDEN_SIZE - 1u};
    for (size_t iy = 0u; iy < sizeof(interp_y) / sizeof(interp_y[0]); ++iy) {
        for (size_t ix = 0u; ix < sizeof(interp_x) / sizeof(interp_x[0]); ++ix) {
            for (size_t ic = 0u; ic < sizeof(interp_c) / sizeof(interp_c[0]); ++ic) {
                const size_t idx = sam_abs_pos_index(LOCAL_GRID, interp_y[iy], interp_x[ix], interp_c[ic]);
                const float expected = f16_bits_to_f32(local_patch[idx]) +
                                       sam_abs_pos_reference(pos, SRC_GRID, LOCAL_GRID, interp_y[iy], interp_x[ix], interp_c[ic]);
                const float actual = f16_bits_to_f32(local_out[idx]);
                CHECK(fabsf(actual - expected) <= 1.5e-3f);
            }
        }
    }

    CHECK(uocr_metal_context_sam_add_abs_pos_f16(ctx,
                                                 local_patch,
                                                 pos,
                                                 LOCAL_GRID,
                                                 LOCAL_GRID + 1u,
                                                 local_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "square") != NULL);

    uocr_metal_context_destroy(ctx);
    free(local_out);
    free(local_patch);
    free(global_out);
    free(global_patch);
    free(pos);
    return 0;
}

static int test_metal_sam_layernorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 4u, HIDDEN = UOCR_SAM_HIDDEN_SIZE };
    uint16_t *input = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    float *out_f32 = (float *)calloc((size_t)ROWS * HIDDEN, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    float *expected = (float *)calloc(HIDDEN, sizeof(float));
    CHECK(input != NULL);
    CHECK(weight != NULL);
    CHECK(bias != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(expected != NULL);

    for (uint32_t col = 0u; col < HIDDEN; ++col) {
        weight[col] = f32_to_f16_bits(0.75f + 0.0125f * (float)(col % 17u));
        bias[col] = f32_to_f16_bits(-0.2f + 0.01f * (float)(col % 23u));
    }
    for (uint32_t col = 0u; col < HIDDEN; ++col) {
        input[col] = f32_to_f16_bits(-1.0f + 0.003f * (float)col);
        input[HIDDEN + col] = f32_to_f16_bits(((col & 1u) ? 0.5f : -0.5f) + 0.0007f * (float)(col % 31u));
        input[2u * HIDDEN + col] = f32_to_f16_bits(0.125f);
        input[3u * HIDDEN + col] = f32_to_f16_bits(sinf((float)col * 0.05f) * 0.25f + (float)(col % 5u) * 0.02f);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_layernorm_f16(ctx,
                                               input,
                                               weight,
                                               bias,
                                               ROWS,
                                               UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                               out_f32,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    const uint32_t cols[] = {0u, 1u, 17u, 255u, 511u, HIDDEN - 1u};
    for (uint32_t row = 0u; row < ROWS; ++row) {
        sam_layernorm_expected(input, weight, bias, row, 1.0e-6f, expected);
        for (size_t i = 0u; i < sizeof(cols) / sizeof(cols[0]); ++i) {
            const uint32_t col = cols[i];
            const float actual = out_f32[(size_t)row * HIDDEN + col];
            CHECK(fabsf(actual - expected[col]) <= 5.0e-4f);
        }
    }

    CHECK(uocr_metal_context_sam_layernorm_f16(ctx,
                                               input,
                                               weight,
                                               bias,
                                               ROWS,
                                               UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                               out_f16,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < ROWS; ++row) {
        sam_layernorm_expected(input, weight, bias, row, 1.0e-6f, expected);
        for (size_t i = 0u; i < sizeof(cols) / sizeof(cols[0]); ++i) {
            const uint32_t col = cols[i];
            const float actual = f16_bits_to_f32(out_f16[(size_t)row * HIDDEN + col]);
            CHECK(fabsf(actual - expected[col]) <= 2.0e-3f);
        }
    }

    CHECK(uocr_metal_context_sam_layernorm_f16(ctx,
                                               input,
                                               weight,
                                               bias,
                                               0u,
                                               UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                               out_f32,
                                               error,
                                               sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM LayerNorm") != NULL);

    uocr_metal_context_destroy(ctx);
    free(expected);
    free(out_f16);
    free(out_f32);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_sam_neck_conv1x1_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { GRID_W = 3u, GRID_H = 2u, IN = UOCR_SAM_HIDDEN_SIZE, OUT = UOCR_SAM_NECK_CHANNELS };
    const size_t input_values = (size_t)GRID_W * GRID_H * IN;
    const size_t weight_values = (size_t)OUT * IN;
    const size_t output_values = (size_t)OUT * GRID_H * GRID_W;
    uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc(weight_values, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(output_values, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(weight != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(UOCR_SAM_NECK_CHANNELS == 256u);

    for (uint32_t y = 0u; y < GRID_H; ++y) {
        for (uint32_t x = 0u; x < GRID_W; ++x) {
            for (uint32_t c = 0u; c < IN; ++c) {
                const int mod = (int)((y * 19u + x * 11u + c * 5u) % 43u) - 21;
                input[sam_bhwc_index(GRID_W, y, x, c)] = f32_to_f16_bits((float)mod * 0.01f);
            }
        }
    }

    const uint32_t active_out[] = {0u, 7u, 111u, OUT - 1u};
    for (size_t i = 0u; i < sizeof(active_out) / sizeof(active_out[0]); ++i) {
        const uint32_t out_channel = active_out[i];
        weight[sam_neck_conv1x1_weight_index(out_channel, (uint32_t)(1u + i))] =
            f32_to_f16_bits(0.125f - 0.0125f * (float)i);
        weight[sam_neck_conv1x1_weight_index(out_channel, (uint32_t)(33u + 7u * i))] =
            f32_to_f16_bits(-0.08f + 0.01f * (float)i);
        weight[sam_neck_conv1x1_weight_index(out_channel, IN - 1u - (uint32_t)i)] =
            f32_to_f16_bits(0.03125f * (float)(i + 1u));
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_neck_conv1x1_f16(ctx,
                                                   input,
                                                   weight,
                                                   GRID_W,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct neck_sample {
        uint32_t y;
        uint32_t x;
        uint32_t out_channel;
    } samples[] = {
        {0u, 0u, 0u},
        {0u, 2u, 7u},
        {1u, 1u, 111u},
        {1u, 2u, OUT - 1u},
        {1u, 0u, 3u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_neck_conv1x1_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 2.0e-5f);
    }

    CHECK(uocr_metal_context_sam_neck_conv1x1_f16(ctx,
                                                   input,
                                                   weight,
                                                   GRID_W,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   out_f16,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_neck_conv1x1_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 8.0e-4f);
    }

    CHECK(uocr_metal_context_sam_neck_conv1x1_f16(ctx,
                                                   input,
                                                   NULL,
                                                   GRID_W,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM neck 1x1") != NULL);

    CHECK(uocr_metal_context_sam_neck_conv1x1_f16(ctx,
                                                   input,
                                                   weight,
                                                   UOCR_SAM_MAX_GRID_SIZE + 1u,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM neck 1x1 grid") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_sam_neck_conv3x3_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { GRID_W = 4u, GRID_H = 3u, CHANNELS = UOCR_SAM_NECK_CHANNELS };
    const size_t input_values = (size_t)CHANNELS * GRID_H * GRID_W;
    const size_t weight_values = (size_t)CHANNELS * CHANNELS * UOCR_SAM_NECK_KERNEL_SIZE * UOCR_SAM_NECK_KERNEL_SIZE;
    const size_t output_values = input_values;
    uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc(weight_values, sizeof(uint16_t));
    uint16_t *ln_weight = (uint16_t *)calloc(CHANNELS, sizeof(uint16_t));
    uint16_t *ln_bias = (uint16_t *)calloc(CHANNELS, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(output_values, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
    float *ln_out_f32 = (float *)calloc(output_values, sizeof(float));
    uint16_t *ln_out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(weight != NULL);
    CHECK(ln_weight != NULL);
    CHECK(ln_bias != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(ln_out_f32 != NULL);
    CHECK(ln_out_f16 != NULL);
    CHECK(UOCR_SAM_NECK_CHANNELS == 256u);
    CHECK(UOCR_SAM_NECK_KERNEL_SIZE == 3u);

    for (uint32_t c = 0u; c < CHANNELS; ++c) {
        for (uint32_t y = 0u; y < GRID_H; ++y) {
            for (uint32_t x = 0u; x < GRID_W; ++x) {
                const int mod = (int)((c * 13u + y * 17u + x * 19u) % 79u) - 39;
                input[sam_neck_nchw_index(GRID_W, GRID_H, c, y, x)] = f32_to_f16_bits((float)mod * 0.0035f);
            }
        }
    }
    for (uint32_t out_channel = 0u; out_channel < CHANNELS; ++out_channel) {
        ln_weight[out_channel] = f32_to_f16_bits(0.85f + (float)(out_channel % 13u) * 0.015625f);
        ln_bias[out_channel] = f32_to_f16_bits(((int)(out_channel % 17u) - 8) * 0.0075f);
        for (uint32_t in_channel = 0u; in_channel < CHANNELS; ++in_channel) {
            for (uint32_t ky = 0u; ky < UOCR_SAM_NECK_KERNEL_SIZE; ++ky) {
                for (uint32_t kx = 0u; kx < UOCR_SAM_NECK_KERNEL_SIZE; ++kx) {
                    const int mod = (int)((out_channel * 7u + in_channel * 11u + ky * 13u + kx * 5u) % 53u) - 26;
                    weight[sam_neck_conv3x3_weight_index(out_channel, in_channel, ky, kx)] =
                        f32_to_f16_bits((float)mod * 0.00016f);
                }
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_neck_conv3x3_f16(ctx,
                                                   input,
                                                   weight,
                                                   GRID_W,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct conv3_sample {
        uint32_t y;
        uint32_t x;
        uint32_t out_channel;
    } samples[] = {
        {0u, 0u, 0u},
        {0u, GRID_W - 1u, 7u},
        {1u, 2u, 111u},
        {GRID_H - 1u, 1u, 200u},
        {GRID_H - 1u, GRID_W - 1u, CHANNELS - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_neck_conv3x3_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         GRID_H,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 5.0e-4f);
    }

    CHECK(uocr_metal_context_sam_neck_conv3x3_f16(ctx,
                                                   input,
                                                   weight,
                                                   GRID_W,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   out_f16,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_neck_conv3x3_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         GRID_H,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                  out_f16,
                                                  ln_weight,
                                                  ln_bias,
                                                  GRID_W,
                                                  GRID_H,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  ln_out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_layernorm2d_expected(out_f16,
                                                        ln_weight,
                                                        ln_bias,
                                                        GRID_W,
                                                        GRID_H,
                                                        samples[i].y,
                                                        samples[i].x,
                                                        samples[i].out_channel);
        CHECK(isfinite(expected));
        CHECK(fabsf(ln_out_f32[idx] - expected) <= 1.2e-4f);
    }

    CHECK(uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                  out_f16,
                                                  ln_weight,
                                                  ln_bias,
                                                  GRID_W,
                                                  GRID_H,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  ln_out_f16,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_layernorm2d_expected(out_f16,
                                                        ln_weight,
                                                        ln_bias,
                                                        GRID_W,
                                                        GRID_H,
                                                        samples[i].y,
                                                        samples[i].x,
                                                        samples[i].out_channel);
        CHECK(isfinite(expected));
        CHECK(fabsf(f16_bits_to_f32(ln_out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_neck_conv3x3_f16(ctx,
                                                   input,
                                                   NULL,
                                                   GRID_W,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM neck 3x3") != NULL);

    CHECK(uocr_metal_context_sam_neck_conv3x3_f16(ctx,
                                                   input,
                                                   weight,
                                                   UOCR_SAM_MAX_GRID_SIZE + 1u,
                                                   GRID_H,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM neck 3x3 grid") != NULL);

    uocr_metal_context_destroy(ctx);
    free(ln_out_f16);
    free(ln_out_f32);
    free(out_f16);
    free(out_f32);
    free(ln_bias);
    free(ln_weight);
    free(weight);
    free(input);
    return 0;
}

static float sam_layernorm2d_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       const uint16_t *bias,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t y,
                                       uint32_t x,
                                       uint32_t channel) {
    const uint32_t spatial = y * grid_w + x;
    const uint32_t spatial_size = grid_w * grid_h;
    double sum = 0.0;
    double sq_sum = 0.0;
    for (uint32_t c = 0u; c < UOCR_SAM_NECK_CHANNELS; ++c) {
        const float v = f16_bits_to_f32(input[(size_t)c * spatial_size + spatial]);
        sum += (double)v;
        sq_sum += (double)v * (double)v;
    }
    const float mean = (float)(sum / (double)UOCR_SAM_NECK_CHANNELS);
    float variance = (float)(sq_sum / (double)UOCR_SAM_NECK_CHANNELS) - mean * mean;
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    const float v = f16_bits_to_f32(input[sam_neck_nchw_index(grid_w, grid_h, channel, y, x)]);
    return (v - mean) / sqrtf(variance + 1.0e-6f) * f16_bits_to_f32(weight[channel]) + f16_bits_to_f32(bias[channel]);
}

static int test_metal_sam_layernorm2d_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { GRID_W = 4u, GRID_H = 3u, CHANNELS = UOCR_SAM_NECK_CHANNELS };
    const size_t value_count = (size_t)CHANNELS * GRID_H * GRID_W;
    uint16_t *input = (uint16_t *)calloc(value_count, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc(CHANNELS, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(CHANNELS, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(value_count, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(value_count, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(weight != NULL);
    CHECK(bias != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(UOCR_SAM_NECK_CHANNELS == 256u);

    for (uint32_t c = 0u; c < CHANNELS; ++c) {
        weight[c] = f32_to_f16_bits(0.75f + (float)(c % 11u) * 0.03125f);
        bias[c] = f32_to_f16_bits(((int)(c % 13u) - 6) * 0.0125f);
        for (uint32_t y = 0u; y < GRID_H; ++y) {
            for (uint32_t x = 0u; x < GRID_W; ++x) {
                const int mod = (int)((c * 17u + y * 23u + x * 5u) % 101u) - 50;
                input[sam_neck_nchw_index(GRID_W, GRID_H, c, y, x)] = f32_to_f16_bits((float)mod * 0.006f);
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                  input,
                                                  weight,
                                                  bias,
                                                  GRID_W,
                                                  GRID_H,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct ln2d_sample {
        uint32_t y;
        uint32_t x;
        uint32_t channel;
    } samples[] = {
        {0u, 0u, 0u},
        {0u, 3u, 7u},
        {1u, 2u, 111u},
        {2u, 1u, 200u},
        {2u, 3u, CHANNELS - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].channel, samples[i].y, samples[i].x);
        const float expected = sam_layernorm2d_expected(input,
                                                        weight,
                                                        bias,
                                                        GRID_W,
                                                        GRID_H,
                                                        samples[i].y,
                                                        samples[i].x,
                                                        samples[i].channel);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 8.0e-5f);
    }

    CHECK(uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                  input,
                                                  weight,
                                                  bias,
                                                  GRID_W,
                                                  GRID_H,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  out_f16,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(GRID_W, GRID_H, samples[i].channel, samples[i].y, samples[i].x);
        const float expected = sam_layernorm2d_expected(input,
                                                        weight,
                                                        bias,
                                                        GRID_W,
                                                        GRID_H,
                                                        samples[i].y,
                                                        samples[i].x,
                                                        samples[i].channel);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                  input,
                                                  weight,
                                                  NULL,
                                                  GRID_W,
                                                  GRID_H,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM LayerNorm2d") != NULL);

    CHECK(uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                  input,
                                                  weight,
                                                  bias,
                                                  0u,
                                                  GRID_H,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM LayerNorm2d grid") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_sam_net2_conv3x3_stride2_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        GRID_W = 5u,
        GRID_H = 4u,
        OUT_W = (GRID_W + UOCR_SAM_NET_STRIDE - 1u) / UOCR_SAM_NET_STRIDE,
        OUT_H = (GRID_H + UOCR_SAM_NET_STRIDE - 1u) / UOCR_SAM_NET_STRIDE,
    };
    const size_t input_values = (size_t)UOCR_SAM_NECK_CHANNELS * GRID_H * GRID_W;
    const size_t weight_values = (size_t)UOCR_SAM_NET2_CHANNELS * UOCR_SAM_NECK_CHANNELS *
                                 UOCR_SAM_NECK_KERNEL_SIZE * UOCR_SAM_NECK_KERNEL_SIZE;
    const size_t output_values = (size_t)UOCR_SAM_NET2_CHANNELS * OUT_H * OUT_W;
    uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc(weight_values, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(output_values, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(weight != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(UOCR_SAM_NECK_CHANNELS == 256u);
    CHECK(UOCR_SAM_NET2_CHANNELS == 512u);
    CHECK(UOCR_SAM_NECK_KERNEL_SIZE == 3u);
    CHECK(UOCR_SAM_NET_STRIDE == 2u);
    CHECK(OUT_W == 3u);
    CHECK(OUT_H == 2u);

    for (uint32_t c = 0u; c < UOCR_SAM_NECK_CHANNELS; ++c) {
        for (uint32_t y = 0u; y < GRID_H; ++y) {
            for (uint32_t x = 0u; x < GRID_W; ++x) {
                const int mod = (int)((c * 17u + y * 23u + x * 29u) % 89u) - 44;
                input[sam_neck_nchw_index(GRID_W, GRID_H, c, y, x)] = f32_to_f16_bits((float)mod * 0.003f);
            }
        }
    }
    for (uint32_t out_channel = 0u; out_channel < UOCR_SAM_NET2_CHANNELS; ++out_channel) {
        for (uint32_t in_channel = 0u; in_channel < UOCR_SAM_NECK_CHANNELS; ++in_channel) {
            for (uint32_t ky = 0u; ky < UOCR_SAM_NECK_KERNEL_SIZE; ++ky) {
                for (uint32_t kx = 0u; kx < UOCR_SAM_NECK_KERNEL_SIZE; ++kx) {
                    const int mod = (int)((out_channel * 5u + in_channel * 7u + ky * 11u + kx * 13u) % 67u) - 33;
                    weight[sam_net2_conv3x3_weight_index(out_channel, in_channel, ky, kx)] =
                        f32_to_f16_bits((float)mod * 0.00013f);
                }
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_net2_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           weight,
                                                           GRID_W,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct net2_sample {
        uint32_t y;
        uint32_t x;
        uint32_t out_channel;
    } samples[] = {
        {0u, 0u, 0u},
        {0u, OUT_W - 1u, 7u},
        {1u, 1u, 111u},
        {OUT_H - 1u, 0u, 300u},
        {OUT_H - 1u, OUT_W - 1u, UOCR_SAM_NET2_CHANNELS - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(OUT_W, OUT_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_net2_conv3x3_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         GRID_H,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 5.0e-4f);
    }

    CHECK(uocr_metal_context_sam_net2_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           weight,
                                                           GRID_W,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           out_f16,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(OUT_W, OUT_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_net2_conv3x3_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         GRID_H,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_net2_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           NULL,
                                                           GRID_W,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM net_2") != NULL);

    CHECK(uocr_metal_context_sam_net2_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           weight,
                                                           UOCR_SAM_MAX_GRID_SIZE + 1u,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM net_2 grid") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_sam_net3_conv3x3_stride2_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        GRID_W = 5u,
        GRID_H = 3u,
        OUT_W = (GRID_W + UOCR_SAM_NET_STRIDE - 1u) / UOCR_SAM_NET_STRIDE,
        OUT_H = (GRID_H + UOCR_SAM_NET_STRIDE - 1u) / UOCR_SAM_NET_STRIDE,
    };
    const size_t input_values = (size_t)UOCR_SAM_NET2_CHANNELS * GRID_H * GRID_W;
    const size_t weight_values = (size_t)UOCR_SAM_NET3_CHANNELS * UOCR_SAM_NET2_CHANNELS *
                                 UOCR_SAM_NECK_KERNEL_SIZE * UOCR_SAM_NECK_KERNEL_SIZE;
    const size_t output_values = (size_t)UOCR_SAM_NET3_CHANNELS * OUT_H * OUT_W;
    uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc(weight_values, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(output_values, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(weight != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(UOCR_SAM_NET2_CHANNELS == 512u);
    CHECK(UOCR_SAM_NET3_CHANNELS == 1024u);
    CHECK(UOCR_SAM_FEATURE_CHANNELS == 1024u);
    CHECK(UOCR_SAM_NECK_KERNEL_SIZE == 3u);
    CHECK(UOCR_SAM_NET_STRIDE == 2u);
    CHECK(OUT_W == 3u);
    CHECK(OUT_H == 2u);

    for (uint32_t c = 0u; c < UOCR_SAM_NET2_CHANNELS; ++c) {
        for (uint32_t y = 0u; y < GRID_H; ++y) {
            for (uint32_t x = 0u; x < GRID_W; ++x) {
                const int mod = (int)((c * 19u + y * 31u + x * 37u) % 101u) - 50;
                input[sam_neck_nchw_index(GRID_W, GRID_H, c, y, x)] = f32_to_f16_bits((float)mod * 0.0017f);
            }
        }
    }
    for (uint32_t out_channel = 0u; out_channel < UOCR_SAM_NET3_CHANNELS; ++out_channel) {
        for (uint32_t in_channel = 0u; in_channel < UOCR_SAM_NET2_CHANNELS; ++in_channel) {
            for (uint32_t ky = 0u; ky < UOCR_SAM_NECK_KERNEL_SIZE; ++ky) {
                for (uint32_t kx = 0u; kx < UOCR_SAM_NECK_KERNEL_SIZE; ++kx) {
                    const int mod = (int)((out_channel * 3u + in_channel * 5u + ky * 17u + kx * 23u) % 79u) - 39;
                    weight[sam_net3_conv3x3_weight_index(out_channel, in_channel, ky, kx)] =
                        f32_to_f16_bits((float)mod * 0.00009f);
                }
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_net3_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           weight,
                                                           GRID_W,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct net3_sample {
        uint32_t y;
        uint32_t x;
        uint32_t out_channel;
    } samples[] = {
        {0u, 0u, 0u},
        {0u, OUT_W - 1u, 13u},
        {1u, 1u, 255u},
        {OUT_H - 1u, 0u, 700u},
        {OUT_H - 1u, OUT_W - 1u, UOCR_SAM_NET3_CHANNELS - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(OUT_W, OUT_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_net3_conv3x3_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         GRID_H,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 8.0e-4f);
    }

    CHECK(uocr_metal_context_sam_net3_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           weight,
                                                           GRID_W,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           out_f16,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_neck_nchw_index(OUT_W, OUT_H, samples[i].out_channel, samples[i].y, samples[i].x);
        const float expected = sam_net3_conv3x3_expected(input,
                                                         weight,
                                                         GRID_W,
                                                         GRID_H,
                                                         samples[i].y,
                                                         samples[i].x,
                                                         samples[i].out_channel);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.5e-3f);
    }

    CHECK(uocr_metal_context_sam_net3_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           NULL,
                                                           GRID_W,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM net_3") != NULL);

    CHECK(uocr_metal_context_sam_net3_conv3x3_stride2_f16(ctx,
                                                           input,
                                                           weight,
                                                           UOCR_SAM_MAX_GRID_SIZE + 1u,
                                                           GRID_H,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM net_3 grid") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_clip_embed_sam_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_SAM_FEATURE_CHANNELS == UOCR_CLIP_HIDDEN_SIZE);
    CHECK(UOCR_CLIP_CLASS_TOKENS == 1u);
    CHECK(UOCR_CLIP_MAX_GRID_SIZE == UOCR_GLOBAL_GRID_QUERIES);
    CHECK(UOCR_CLIP_MAX_TOKENS == 257u);

    uint16_t *class_embedding = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    CHECK(class_embedding != NULL);
    for (uint32_t c = 0u; c < UOCR_CLIP_HIDDEN_SIZE; ++c) {
        const int mod = (int)((c * 7u) % 83u) - 41;
        class_embedding[c] = f32_to_f16_bits((float)mod * 0.001f);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t grids[] = {UOCR_GLOBAL_GRID_QUERIES, UOCR_LOCAL_GRID_QUERIES};
    for (size_t case_index = 0u; case_index < sizeof(grids) / sizeof(grids[0]); ++case_index) {
        const uint32_t grid = grids[case_index];
        const uint32_t tokens = UOCR_CLIP_CLASS_TOKENS + grid * grid;
        CHECK(tokens == 257u || tokens == 101u);
        const size_t input_values = (size_t)UOCR_CLIP_HIDDEN_SIZE * grid * grid;
        const size_t output_values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *sam_features = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(output_values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
        CHECK(sam_features != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t c = 0u; c < UOCR_CLIP_HIDDEN_SIZE; ++c) {
            for (uint32_t y = 0u; y < grid; ++y) {
                for (uint32_t x = 0u; x < grid; ++x) {
                    const int mod = (int)((c * 11u + y * 13u + x * 17u + grid) % 97u) - 48;
                    sam_features[sam_neck_nchw_index(grid, grid, c, y, x)] = f32_to_f16_bits((float)mod * 0.0013f);
                }
            }
        }

        CHECK(uocr_metal_context_clip_embed_sam_f16(ctx,
                                                     sam_features,
                                                     class_embedding,
                                                     grid,
                                                     grid,
                                                     UOCR_METAL_DENSE_OUTPUT_F32,
                                                     out_f32,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct clip_embed_sample {
            uint32_t token;
            uint32_t channel;
        } samples[] = {
            {0u, 0u},
            {0u, UOCR_CLIP_HIDDEN_SIZE - 1u},
            {1u, 3u},
            {grid, 17u},
            {1u + grid * (grid - 1u) + (grid - 1u), UOCR_CLIP_HIDDEN_SIZE - 1u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t out_index = (size_t)samples[i].token * UOCR_CLIP_HIDDEN_SIZE + samples[i].channel;
            uint16_t expected_bits = 0u;
            if (samples[i].token == 0u) {
                expected_bits = class_embedding[samples[i].channel];
            } else {
                const uint32_t spatial = samples[i].token - 1u;
                const uint32_t y = spatial / grid;
                const uint32_t x = spatial - y * grid;
                expected_bits = sam_features[sam_neck_nchw_index(grid, grid, samples[i].channel, y, x)];
            }
            CHECK(fabsf(out_f32[out_index] - f16_bits_to_f32(expected_bits)) <= 1.0e-6f);
        }

        CHECK(uocr_metal_context_clip_embed_sam_f16(ctx,
                                                     sam_features,
                                                     class_embedding,
                                                     grid,
                                                     grid,
                                                     UOCR_METAL_DENSE_OUTPUT_F16,
                                                     out_f16,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t out_index = (size_t)samples[i].token * UOCR_CLIP_HIDDEN_SIZE + samples[i].channel;
            uint16_t expected_bits = 0u;
            if (samples[i].token == 0u) {
                expected_bits = class_embedding[samples[i].channel];
            } else {
                const uint32_t spatial = samples[i].token - 1u;
                const uint32_t y = spatial / grid;
                const uint32_t x = spatial - y * grid;
                expected_bits = sam_features[sam_neck_nchw_index(grid, grid, samples[i].channel, y, x)];
            }
            CHECK(out_f16[out_index] == expected_bits);
        }

        free(out_f16);
        free(out_f32);
        free(sam_features);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_embed_sam_f16(ctx,
                                                 &one_value,
                                                 NULL,
                                                 1u,
                                                 1u,
                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP SAM embedding") != NULL);

    CHECK(uocr_metal_context_clip_embed_sam_f16(ctx,
                                                 &one_value,
                                                 &one_value,
                                                 UOCR_CLIP_MAX_GRID_SIZE + 1u,
                                                 1u,
                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP SAM embedding grid") != NULL);

    uocr_metal_context_destroy(ctx);
    free(class_embedding);
    return 0;
}

static int test_metal_clip_abs_pos_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_CLASS_TOKENS == 1u);
    CHECK(UOCR_CLIP_POSITION_GRID == 16u);
    CHECK(UOCR_CLIP_GLOBAL_TOKENS == 257u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);
    CHECK(UOCR_CLIP_MAX_TOKENS == UOCR_CLIP_GLOBAL_TOKENS);

    const size_t pos_values = (size_t)UOCR_CLIP_GLOBAL_TOKENS * UOCR_CLIP_HIDDEN_SIZE;
    uint16_t *pos = (uint16_t *)calloc(pos_values, sizeof(uint16_t));
    CHECK(pos != NULL);
    for (uint32_t token = 0u; token < UOCR_CLIP_GLOBAL_TOKENS; ++token) {
        for (uint32_t c = 0u; c < UOCR_CLIP_HIDDEN_SIZE; ++c) {
            const int mod = (int)((token * 19u + c * 7u) % 113u) - 56;
            pos[clip_token_index(token, c)] = f32_to_f16_bits((float)mod * 0.0012f);
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t grids[] = {UOCR_GLOBAL_GRID_QUERIES, UOCR_LOCAL_GRID_QUERIES};
    for (size_t case_index = 0u; case_index < sizeof(grids) / sizeof(grids[0]); ++case_index) {
        const uint32_t grid = grids[case_index];
        const uint32_t tokens = UOCR_CLIP_CLASS_TOKENS + grid * grid;
        CHECK(tokens == UOCR_CLIP_GLOBAL_TOKENS || tokens == UOCR_CLIP_LOCAL_TOKENS);
        const size_t values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *input = (uint16_t *)calloc(values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t c = 0u; c < UOCR_CLIP_HIDDEN_SIZE; ++c) {
                const int mod = (int)((token * 11u + c * 5u + grid * 3u) % 97u) - 48;
                input[clip_token_index(token, c)] = f32_to_f16_bits((float)mod * 0.0015f);
            }
        }

        CHECK(uocr_metal_context_clip_add_abs_pos_f16(ctx,
                                                       input,
                                                       pos,
                                                       grid,
                                                       grid,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct clip_abs_sample {
            uint32_t token;
            uint32_t channel;
        } samples[] = {
            {0u, 0u},
            {0u, UOCR_CLIP_HIDDEN_SIZE - 1u},
            {1u, 5u},
            {1u + grid * (grid / 2u) + grid / 3u, 77u},
            {tokens - 1u, UOCR_CLIP_HIDDEN_SIZE - 1u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
            const float expected = f16_bits_to_f32(input[idx]) +
                                   clip_abs_pos_reference(pos, grid, samples[i].token, samples[i].channel);
            CHECK(isfinite(expected));
            CHECK(fabsf(out_f32[idx] - expected) <= 5.0e-4f);
        }

        CHECK(uocr_metal_context_clip_add_abs_pos_f16(ctx,
                                                       input,
                                                       pos,
                                                       grid,
                                                       grid,
                                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                                       out_f16,
                                                       error,
                                                       sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
            const float expected = f16_bits_to_f32(input[idx]) +
                                   clip_abs_pos_reference(pos, grid, samples[i].token, samples[i].channel);
            CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 1.5e-3f);
        }

        free(out_f16);
        free(out_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_add_abs_pos_f16(ctx,
                                                   &one_value,
                                                   NULL,
                                                   UOCR_LOCAL_GRID_QUERIES,
                                                   UOCR_LOCAL_GRID_QUERIES,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   &one_out,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP abs pos") != NULL);

    CHECK(uocr_metal_context_clip_add_abs_pos_f16(ctx,
                                                   &one_value,
                                                   pos,
                                                   UOCR_LOCAL_GRID_QUERIES + 1u,
                                                   UOCR_LOCAL_GRID_QUERIES,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   &one_out,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal CLIP abs pos grid") != NULL);

    uocr_metal_context_destroy(ctx);
    free(pos);
    return 0;
}

static int test_metal_clip_pre_layernorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_PRE_LAYERNORM_EPS == 1.0e-5f);
    CHECK(UOCR_CLIP_GLOBAL_TOKENS == 257u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);

    uint16_t *weight = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    float *expected = (float *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(float));
    CHECK(weight != NULL);
    CHECK(bias != NULL);
    CHECK(expected != NULL);

    for (uint32_t col = 0u; col < UOCR_CLIP_HIDDEN_SIZE; ++col) {
        weight[col] = f32_to_f16_bits(0.85f + 0.006f * (float)(col % 29u));
        bias[col] = f32_to_f16_bits(-0.08f + 0.004f * (float)(col % 37u));
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t token_counts[] = {UOCR_CLIP_GLOBAL_TOKENS, UOCR_CLIP_LOCAL_TOKENS};
    for (size_t case_index = 0u; case_index < sizeof(token_counts) / sizeof(token_counts[0]); ++case_index) {
        const uint32_t tokens = token_counts[case_index];
        const size_t values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *input = (uint16_t *)calloc(values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t col = 0u; col < UOCR_CLIP_HIDDEN_SIZE; ++col) {
                float value = 0.0f;
                if (token == 0u) {
                    value = ((col & 1u) ? 0.00055f : -0.00055f) + 0.00001f * (float)(col % 3u);
                } else {
                    const int mod = (int)((token * 13u + col * 17u + tokens) % 127u) - 63;
                    value = (float)mod * 0.0011f;
                }
                input[clip_token_index(token, col)] = f32_to_f16_bits(value);
            }
        }

        CHECK(uocr_metal_context_clip_pre_layernorm_f16(ctx,
                                                         input,
                                                         weight,
                                                         bias,
                                                         tokens,
                                                         UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                                         out_f32,
                                                         error,
                                                         sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        const uint32_t rows[] = {0u, tokens / 2u, tokens - 1u};
        const uint32_t cols[] = {0u, 1u, 19u, 257u, 733u, UOCR_CLIP_HIDDEN_SIZE - 1u};
        for (size_t row_index = 0u; row_index < sizeof(rows) / sizeof(rows[0]); ++row_index) {
            const uint32_t row = rows[row_index];
            clip_layernorm_expected(input, weight, bias, row, expected);
            for (size_t col_index = 0u; col_index < sizeof(cols) / sizeof(cols[0]); ++col_index) {
                const uint32_t col = cols[col_index];
                const size_t idx = clip_token_index(row, col);
                CHECK(fabsf(out_f32[idx] - expected[col]) <= 7.0e-4f);
            }
        }

        CHECK(uocr_metal_context_clip_pre_layernorm_f16(ctx,
                                                         input,
                                                         weight,
                                                         bias,
                                                         tokens,
                                                         UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                                         out_f16,
                                                         error,
                                                         sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t row_index = 0u; row_index < sizeof(rows) / sizeof(rows[0]); ++row_index) {
            const uint32_t row = rows[row_index];
            clip_layernorm_expected(input, weight, bias, row, expected);
            for (size_t col_index = 0u; col_index < sizeof(cols) / sizeof(cols[0]); ++col_index) {
                const uint32_t col = cols[col_index];
                const size_t idx = clip_token_index(row, col);
                CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected[col]) <= 2.0e-3f);
            }
        }

        free(out_f16);
        free(out_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_pre_layernorm_f16(ctx,
                                                     &one_value,
                                                     NULL,
                                                     bias,
                                                     UOCR_CLIP_LOCAL_TOKENS,
                                                     UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                                     &one_out,
                                                     error,
                                                     sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP pre-LayerNorm") != NULL);

    CHECK(uocr_metal_context_clip_pre_layernorm_f16(ctx,
                                                     &one_value,
                                                     weight,
                                                     bias,
                                                     UOCR_CLIP_LOCAL_TOKENS - 1u,
                                                     UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                                     &one_out,
                                                     error,
                                                     sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP pre-LayerNorm token count") != NULL);

    uocr_metal_context_destroy(ctx);
    free(expected);
    free(bias);
    free(weight);
    return 0;
}

static int test_metal_clip_layernorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_BLOCKS == 24u);
    CHECK(UOCR_CLIP_ATTENTION_HEADS == 16u);
    CHECK(UOCR_CLIP_HEAD_DIM == 64u);
    CHECK(UOCR_CLIP_QKV_SIZE == 3072u);
    CHECK(UOCR_CLIP_MLP_INTERMEDIATE == 4096u);
    CHECK(UOCR_CLIP_LAYERNORM_EPS == 1.0e-5f);

    uint16_t *weight = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    float *expected = (float *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(float));
    CHECK(weight != NULL);
    CHECK(bias != NULL);
    CHECK(expected != NULL);

    for (uint32_t col = 0u; col < UOCR_CLIP_HIDDEN_SIZE; ++col) {
        weight[col] = f32_to_f16_bits(0.7f + 0.0035f * (float)(col % 41u));
        bias[col] = f32_to_f16_bits(0.03f - 0.0025f * (float)(col % 31u));
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t token_counts[] = {UOCR_CLIP_GLOBAL_TOKENS, UOCR_CLIP_LOCAL_TOKENS};
    for (size_t case_index = 0u; case_index < sizeof(token_counts) / sizeof(token_counts[0]); ++case_index) {
        const uint32_t tokens = token_counts[case_index];
        const size_t values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *input = (uint16_t *)calloc(values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t col = 0u; col < UOCR_CLIP_HIDDEN_SIZE; ++col) {
                const float wave = sinf((float)(token + 1u) * 0.03f + (float)col * 0.011f) * 0.08f;
                const float offset = (float)((token * 7u + col * 3u) % 17u) * 0.001f;
                input[clip_token_index(token, col)] = f32_to_f16_bits(wave + offset);
            }
        }

        CHECK(uocr_metal_context_clip_layernorm_f16(ctx,
                                                     input,
                                                     weight,
                                                     bias,
                                                     tokens,
                                                     UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                                     out_f32,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        const uint32_t rows[] = {0u, tokens / 3u, tokens - 1u};
        const uint32_t cols[] = {0u, 2u, 64u, 509u, 900u, UOCR_CLIP_HIDDEN_SIZE - 1u};
        for (size_t row_index = 0u; row_index < sizeof(rows) / sizeof(rows[0]); ++row_index) {
            const uint32_t row = rows[row_index];
            clip_layernorm_expected(input, weight, bias, row, expected);
            for (size_t col_index = 0u; col_index < sizeof(cols) / sizeof(cols[0]); ++col_index) {
                const uint32_t col = cols[col_index];
                const size_t idx = clip_token_index(row, col);
                CHECK(fabsf(out_f32[idx] - expected[col]) <= 7.0e-4f);
            }
        }

        CHECK(uocr_metal_context_clip_layernorm_f16(ctx,
                                                     input,
                                                     weight,
                                                     bias,
                                                     tokens,
                                                     UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                                     out_f16,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t row_index = 0u; row_index < sizeof(rows) / sizeof(rows[0]); ++row_index) {
            const uint32_t row = rows[row_index];
            clip_layernorm_expected(input, weight, bias, row, expected);
            for (size_t col_index = 0u; col_index < sizeof(cols) / sizeof(cols[0]); ++col_index) {
                const uint32_t col = cols[col_index];
                const size_t idx = clip_token_index(row, col);
                CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected[col]) <= 2.0e-3f);
            }
        }

        free(out_f16);
        free(out_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_layernorm_f16(ctx,
                                                 &one_value,
                                                 NULL,
                                                 bias,
                                                 UOCR_CLIP_LOCAL_TOKENS,
                                                 UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP LayerNorm") != NULL);

    CHECK(uocr_metal_context_clip_layernorm_f16(ctx,
                                                 &one_value,
                                                 weight,
                                                 bias,
                                                 UOCR_CLIP_MAX_TOKENS + 1u,
                                                 UOCR_METAL_LAYERNORM_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP LayerNorm token count") != NULL);

    uocr_metal_context_destroy(ctx);
    free(expected);
    free(bias);
    free(weight);
    return 0;
}

static int test_metal_clip_qkv_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_QKV_SIZE == 3072u);
    CHECK(UOCR_CLIP_ATTENTION_HEADS == 16u);
    CHECK(UOCR_CLIP_HEAD_DIM == 64u);
    CHECK(UOCR_CLIP_ATTENTION_HEADS * UOCR_CLIP_HEAD_DIM == UOCR_CLIP_HIDDEN_SIZE);

    const size_t weight_values = (size_t)UOCR_CLIP_QKV_SIZE * UOCR_CLIP_HIDDEN_SIZE;
    uint16_t *weight = (uint16_t *)calloc(weight_values, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(UOCR_CLIP_QKV_SIZE, sizeof(uint16_t));
    CHECK(weight != NULL);
    CHECK(bias != NULL);

    /* Q head 0 dim 0 = input[0] + 0.25. */
    weight[clip_qkv_weight_index(0u, 0u, 0u, 0u)] = f32_to_f16_bits(1.0f);
    bias[clip_qkv_packed_col(0u, 0u, 0u)] = f32_to_f16_bits(0.25f);
    /* Q head 15 dim 63 is bias-only. */
    bias[clip_qkv_packed_col(0u, UOCR_CLIP_ATTENTION_HEADS - 1u, UOCR_CLIP_HEAD_DIM - 1u)] =
        f32_to_f16_bits(-0.375f);
    /* K head 8 dim 17 = -1.5 * input[11] + 0.125. */
    weight[clip_qkv_weight_index(1u, 8u, 17u, 11u)] = f32_to_f16_bits(-1.5f);
    bias[clip_qkv_packed_col(1u, 8u, 17u)] = f32_to_f16_bits(0.125f);
    /* V head 15 dim 63 = input[5] - input[13] - 0.5. */
    weight[clip_qkv_weight_index(2u, UOCR_CLIP_ATTENTION_HEADS - 1u, UOCR_CLIP_HEAD_DIM - 1u, 5u)] =
        f32_to_f16_bits(1.0f);
    weight[clip_qkv_weight_index(2u, UOCR_CLIP_ATTENTION_HEADS - 1u, UOCR_CLIP_HEAD_DIM - 1u, 13u)] =
        f32_to_f16_bits(-1.0f);
    bias[clip_qkv_packed_col(2u, UOCR_CLIP_ATTENTION_HEADS - 1u, UOCR_CLIP_HEAD_DIM - 1u)] =
        f32_to_f16_bits(-0.5f);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t token_counts[] = {UOCR_CLIP_GLOBAL_TOKENS, UOCR_CLIP_LOCAL_TOKENS};
    for (size_t case_index = 0u; case_index < sizeof(token_counts) / sizeof(token_counts[0]); ++case_index) {
        const uint32_t tokens = token_counts[case_index];
        const size_t input_values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        float *q_f32 = (float *)calloc(input_values, sizeof(float));
        float *k_f32 = (float *)calloc(input_values, sizeof(float));
        float *v_f32 = (float *)calloc(input_values, sizeof(float));
        uint16_t *q_f16 = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        uint16_t *k_f16 = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        uint16_t *v_f16 = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(q_f32 != NULL);
        CHECK(k_f32 != NULL);
        CHECK(v_f32 != NULL);
        CHECK(q_f16 != NULL);
        CHECK(k_f16 != NULL);
        CHECK(v_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t col = 0u; col < UOCR_CLIP_HIDDEN_SIZE; ++col) {
                const int mod = (int)((token * 23u + col * 7u + tokens) % 251u) - 125;
                input[clip_token_index(token, col)] = f32_to_f16_bits((float)mod * 0.001f);
            }
        }

        CHECK(uocr_metal_context_clip_qkv_f16(ctx,
                                               input,
                                               weight,
                                               bias,
                                               tokens,
                                               UOCR_METAL_DENSE_OUTPUT_F32,
                                               q_f32,
                                               k_f32,
                                               v_f32,
                                               error,
                                               sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        const uint32_t sample_tokens[] = {0u, tokens / 2u, tokens - 1u};
        for (size_t i = 0u; i < sizeof(sample_tokens) / sizeof(sample_tokens[0]); ++i) {
            const uint32_t token = sample_tokens[i];
            const float x0 = f16_bits_to_f32(input[clip_token_index(token, 0u)]);
            const float x5 = f16_bits_to_f32(input[clip_token_index(token, 5u)]);
            const float x11 = f16_bits_to_f32(input[clip_token_index(token, 11u)]);
            const float x13 = f16_bits_to_f32(input[clip_token_index(token, 13u)]);
            CHECK(fabsf(q_f32[clip_qkv_out_index(token, 0u, 0u)] - (x0 + 0.25f)) <= 1.0e-5f);
            CHECK(fabsf(q_f32[clip_qkv_out_index(token, UOCR_CLIP_ATTENTION_HEADS - 1u, UOCR_CLIP_HEAD_DIM - 1u)] +
                        0.375f) <= 1.0e-5f);
            CHECK(fabsf(k_f32[clip_qkv_out_index(token, 8u, 17u)] - (-1.5f * x11 + 0.125f)) <= 1.0e-5f);
            CHECK(fabsf(v_f32[clip_qkv_out_index(token,
                                                 UOCR_CLIP_ATTENTION_HEADS - 1u,
                                                 UOCR_CLIP_HEAD_DIM - 1u)] -
                        (x5 - x13 - 0.5f)) <= 1.0e-5f);
            CHECK(fabsf(q_f32[clip_qkv_out_index(token, 3u, 9u)]) <= 1.0e-5f);
            CHECK(fabsf(k_f32[clip_qkv_out_index(token, 3u, 9u)]) <= 1.0e-5f);
            CHECK(fabsf(v_f32[clip_qkv_out_index(token, 3u, 9u)]) <= 1.0e-5f);
        }

        CHECK(uocr_metal_context_clip_qkv_f16(ctx,
                                               input,
                                               weight,
                                               bias,
                                               tokens,
                                               UOCR_METAL_DENSE_OUTPUT_F16,
                                               q_f16,
                                               k_f16,
                                               v_f16,
                                               error,
                                               sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(sample_tokens) / sizeof(sample_tokens[0]); ++i) {
            const uint32_t token = sample_tokens[i];
            const float x0 = f16_bits_to_f32(input[clip_token_index(token, 0u)]);
            const float x5 = f16_bits_to_f32(input[clip_token_index(token, 5u)]);
            const float x11 = f16_bits_to_f32(input[clip_token_index(token, 11u)]);
            const float x13 = f16_bits_to_f32(input[clip_token_index(token, 13u)]);
            CHECK(q_f16[clip_qkv_out_index(token, 0u, 0u)] == f32_to_f16_bits(x0 + 0.25f));
            CHECK(q_f16[clip_qkv_out_index(token, UOCR_CLIP_ATTENTION_HEADS - 1u, UOCR_CLIP_HEAD_DIM - 1u)] ==
                  f32_to_f16_bits(-0.375f));
            CHECK(k_f16[clip_qkv_out_index(token, 8u, 17u)] == f32_to_f16_bits(-1.5f * x11 + 0.125f));
            CHECK(v_f16[clip_qkv_out_index(token, UOCR_CLIP_ATTENTION_HEADS - 1u, UOCR_CLIP_HEAD_DIM - 1u)] ==
                  f32_to_f16_bits(x5 - x13 - 0.5f));
        }

        free(v_f16);
        free(k_f16);
        free(q_f16);
        free(v_f32);
        free(k_f32);
        free(q_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_qkv_f16(ctx,
                                           &one_value,
                                           NULL,
                                           bias,
                                           UOCR_CLIP_LOCAL_TOKENS,
                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                           &one_out,
                                           &one_out,
                                           &one_out,
                                           error,
                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP QKV") != NULL);

    CHECK(uocr_metal_context_clip_qkv_f16(ctx,
                                           &one_value,
                                           weight,
                                           bias,
                                           UOCR_CLIP_LOCAL_TOKENS - 1u,
                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                           &one_out,
                                           &one_out,
                                           &one_out,
                                           error,
                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP QKV token count") != NULL);

    uocr_metal_context_destroy(ctx);
    free(bias);
    free(weight);
    return 0;
}

static int test_metal_clip_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_ATTENTION_HEADS == 16u);
    CHECK(UOCR_CLIP_HEAD_DIM == 64u);
    CHECK(UOCR_CLIP_ATTENTION_HEADS * UOCR_CLIP_HEAD_DIM == UOCR_CLIP_HIDDEN_SIZE);
    CHECK(UOCR_CLIP_GLOBAL_TOKENS == 257u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t token_counts[] = {UOCR_CLIP_GLOBAL_TOKENS, UOCR_CLIP_LOCAL_TOKENS};
    for (size_t case_index = 0u; case_index < sizeof(token_counts) / sizeof(token_counts[0]); ++case_index) {
        const uint32_t tokens = token_counts[case_index];
        const size_t values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *q = (uint16_t *)calloc(values, sizeof(uint16_t));
        uint16_t *k = (uint16_t *)calloc(values, sizeof(uint16_t));
        uint16_t *v = (uint16_t *)calloc(values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(values, sizeof(uint16_t));
        CHECK(q != NULL);
        CHECK(k != NULL);
        CHECK(v != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t head = 0u; head < UOCR_CLIP_ATTENTION_HEADS; ++head) {
                for (uint32_t dim = 0u; dim < UOCR_CLIP_HEAD_DIM; ++dim) {
                    const int q_mod = (int)((token * 5u + head * 7u + dim * 11u + tokens) % 31u) - 15;
                    const int k_mod = (int)((token * 13u + head * 3u + dim * 17u + tokens) % 29u) - 14;
                    const int v_mod = (int)((token * 19u + head * 23u + dim * 2u + tokens) % 37u) - 18;
                    const size_t idx = clip_qkv_out_index(token, head, dim);
                    q[idx] = f32_to_f16_bits((float)q_mod * 0.003f);
                    k[idx] = f32_to_f16_bits((float)k_mod * 0.004f);
                    v[idx] = f32_to_f16_bits((float)v_mod * 0.005f);
                }
            }
        }

        CHECK(uocr_metal_context_clip_attention_f16(ctx,
                                                     q,
                                                     k,
                                                     v,
                                                     tokens,
                                                     UOCR_METAL_DENSE_OUTPUT_F32,
                                                     out_f32,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct clip_attention_sample {
            uint32_t token;
            uint32_t head;
            uint32_t dim;
        } samples[] = {
            {0u, 0u, 0u},
            {tokens / 3u, 7u, 19u},
            {tokens / 2u, 15u, 63u},
            {tokens - 1u, 3u, 5u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_qkv_out_index(samples[i].token, samples[i].head, samples[i].dim);
            const float expected = clip_attention_expected(q,
                                                           k,
                                                           v,
                                                           tokens,
                                                           samples[i].token,
                                                           samples[i].head,
                                                           samples[i].dim);
            CHECK(isfinite(expected));
            CHECK(fabsf(out_f32[idx] - expected) <= 1.2e-4f);
        }

        CHECK(uocr_metal_context_clip_attention_f16(ctx,
                                                     q,
                                                     k,
                                                     v,
                                                     tokens,
                                                     UOCR_METAL_DENSE_OUTPUT_F16,
                                                     out_f16,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_qkv_out_index(samples[i].token, samples[i].head, samples[i].dim);
            const float expected = clip_attention_expected(q,
                                                           k,
                                                           v,
                                                           tokens,
                                                           samples[i].token,
                                                           samples[i].head,
                                                           samples[i].dim);
            CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 1.5e-3f);
        }

        free(out_f16);
        free(out_f32);
        free(v);
        free(k);
        free(q);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_attention_f16(ctx,
                                                 &one_value,
                                                 NULL,
                                                 &one_value,
                                                 UOCR_CLIP_LOCAL_TOKENS,
                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP attention") != NULL);

    CHECK(uocr_metal_context_clip_attention_f16(ctx,
                                                 &one_value,
                                                 &one_value,
                                                 &one_value,
                                                 UOCR_CLIP_LOCAL_TOKENS - 1u,
                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP attention token count") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static float clip_output_projection_expected(const uint16_t *input,
                                             const uint16_t *weight,
                                             const uint16_t *bias,
                                             uint32_t token,
                                             uint32_t out_channel) {
    float sum = f16_bits_to_f32(bias[out_channel]);
    for (uint32_t k = 0u; k < UOCR_CLIP_HIDDEN_SIZE; ++k) {
        sum += f16_bits_to_f32(input[clip_token_index(token, k)]) *
               f16_bits_to_f32(weight[clip_projection_weight_index(out_channel, k)]);
    }
    return sum;
}

static int test_metal_clip_output_projection_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_GLOBAL_TOKENS == 257u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);

    const size_t weight_values = (size_t)UOCR_CLIP_HIDDEN_SIZE * UOCR_CLIP_HIDDEN_SIZE;
    uint16_t *weight = (uint16_t *)calloc(weight_values, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    CHECK(weight != NULL);
    CHECK(bias != NULL);

    const float diag_values[] = {0.25f, -0.5f, 0.75f, 1.0f, -0.25f, 0.125f, -0.375f};
    for (uint32_t channel = 0u; channel < UOCR_CLIP_HIDDEN_SIZE; ++channel) {
        weight[clip_projection_weight_index(channel, channel)] =
            f32_to_f16_bits(diag_values[channel % (uint32_t)(sizeof(diag_values) / sizeof(diag_values[0]))]);
        bias[channel] = f32_to_f16_bits(((float)((int)(channel % 17u) - 8)) * 0.01f);
    }
    weight[clip_projection_weight_index(3u, 7u)] = f32_to_f16_bits(-0.75f);
    weight[clip_projection_weight_index(511u, 2u)] = f32_to_f16_bits(1.25f);
    weight[clip_projection_weight_index(UOCR_CLIP_HIDDEN_SIZE - 1u, 0u)] = f32_to_f16_bits(-0.5f);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t token_counts[] = {UOCR_CLIP_GLOBAL_TOKENS, UOCR_CLIP_LOCAL_TOKENS};
    for (size_t case_index = 0u; case_index < sizeof(token_counts) / sizeof(token_counts[0]); ++case_index) {
        const uint32_t tokens = token_counts[case_index];
        const size_t values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *input = (uint16_t *)calloc(values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t channel = 0u; channel < UOCR_CLIP_HIDDEN_SIZE; ++channel) {
                const int mod = (int)((token * 7u + channel * 11u + tokens) % 43u) - 21;
                input[clip_token_index(token, channel)] = f32_to_f16_bits((float)mod * 0.01f);
            }
        }

        CHECK(uocr_metal_context_clip_output_projection_f16(ctx,
                                                            input,
                                                            weight,
                                                            bias,
                                                            tokens,
                                                            UOCR_METAL_DENSE_OUTPUT_F32,
                                                            out_f32,
                                                            error,
                                                            sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct clip_projection_sample {
            uint32_t token;
            uint32_t channel;
        } samples[] = {
            {0u, 0u},
            {tokens / 3u, 3u},
            {tokens / 2u, 511u},
            {tokens - 1u, UOCR_CLIP_HIDDEN_SIZE - 1u},
            {tokens - 1u, 17u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
            const float expected = clip_output_projection_expected(input,
                                                                   weight,
                                                                   bias,
                                                                   samples[i].token,
                                                                   samples[i].channel);
            CHECK(fabsf(out_f32[idx] - expected) <= 3.0e-5f);
        }

        CHECK(uocr_metal_context_clip_output_projection_f16(ctx,
                                                            input,
                                                            weight,
                                                            bias,
                                                            tokens,
                                                            UOCR_METAL_DENSE_OUTPUT_F16,
                                                            out_f16,
                                                            error,
                                                            sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
            const float expected = clip_output_projection_expected(input,
                                                                   weight,
                                                                   bias,
                                                                   samples[i].token,
                                                                   samples[i].channel);
            CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 1.5e-3f);
        }

        free(out_f16);
        free(out_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_output_projection_f16(ctx,
                                                        &one_value,
                                                        weight,
                                                        NULL,
                                                        UOCR_CLIP_LOCAL_TOKENS,
                                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                                        &one_out,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP output projection") != NULL);

    CHECK(uocr_metal_context_clip_output_projection_f16(ctx,
                                                        &one_value,
                                                        weight,
                                                        bias,
                                                        UOCR_CLIP_LOCAL_TOKENS - 1u,
                                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                                        &one_out,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP output projection token count") != NULL);

    CHECK(uocr_metal_context_clip_output_projection_f16(ctx,
                                                        &one_value,
                                                        weight,
                                                        bias,
                                                        UOCR_CLIP_LOCAL_TOKENS,
                                                        (uocr_metal_dense_output_type)99,
                                                        &one_out,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal CLIP output projection output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(bias);
    free(weight);
    return 0;
}

static float quickgelu_expected_from_f16(uint16_t value) {
    const float x = f16_bits_to_f32(value);
    return x / (1.0f + expf(-1.702f * x));
}

static int test_metal_clip_quickgelu_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_MLP_INTERMEDIATE == 4096u);
    CHECK(UOCR_CLIP_GLOBAL_TOKENS == 257u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t token_counts[] = {UOCR_CLIP_GLOBAL_TOKENS, UOCR_CLIP_LOCAL_TOKENS};
    for (size_t case_index = 0u; case_index < sizeof(token_counts) / sizeof(token_counts[0]); ++case_index) {
        const uint32_t tokens = token_counts[case_index];
        const size_t values = (size_t)tokens * UOCR_CLIP_MLP_INTERMEDIATE;
        uint16_t *input = (uint16_t *)calloc(values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t channel = 0u; channel < UOCR_CLIP_MLP_INTERMEDIATE; ++channel) {
                const int mod = (int)((token * 17u + channel * 31u + tokens) % 161u) - 80;
                input[(size_t)token * UOCR_CLIP_MLP_INTERMEDIATE + channel] = f32_to_f16_bits((float)mod * 0.05f);
            }
        }

        CHECK(uocr_metal_context_clip_quickgelu_f16(ctx,
                                                    input,
                                                    tokens,
                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct quickgelu_sample {
            uint32_t token;
            uint32_t channel;
        } samples[] = {
            {0u, 0u},
            {tokens / 3u, 17u},
            {tokens / 2u, 1023u},
            {tokens - 1u, 2048u},
            {tokens - 1u, UOCR_CLIP_MLP_INTERMEDIATE - 1u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = (size_t)samples[i].token * UOCR_CLIP_MLP_INTERMEDIATE + samples[i].channel;
            const float expected = quickgelu_expected_from_f16(input[idx]);
            CHECK(fabsf(out_f32[idx] - expected) <= 2.0e-5f);
        }

        CHECK(uocr_metal_context_clip_quickgelu_f16(ctx,
                                                    input,
                                                    tokens,
                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                    out_f16,
                                                    error,
                                                    sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = (size_t)samples[i].token * UOCR_CLIP_MLP_INTERMEDIATE + samples[i].channel;
            const float expected = quickgelu_expected_from_f16(input[idx]);
            CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
        }

        free(out_f16);
        free(out_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_quickgelu_f16(ctx,
                                                NULL,
                                                UOCR_CLIP_LOCAL_TOKENS,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                &one_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP QuickGELU") != NULL);

    CHECK(uocr_metal_context_clip_quickgelu_f16(ctx,
                                                &one_value,
                                                UOCR_CLIP_LOCAL_TOKENS - 1u,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                &one_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP QuickGELU token count") != NULL);

    CHECK(uocr_metal_context_clip_quickgelu_f16(ctx,
                                                &one_value,
                                                UOCR_CLIP_LOCAL_TOKENS,
                                                (uocr_metal_dense_output_type)99,
                                                &one_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal CLIP QuickGELU output type") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static float clip_mlp_expected(const uint16_t *input,
                               const uint16_t *fc1_weight,
                               const uint16_t *fc1_bias,
                               const uint16_t *fc2_weight,
                               const uint16_t *fc2_bias,
                               uint32_t token,
                               uint32_t out_channel) {
    float sum = f16_bits_to_f32(fc2_bias[out_channel]);
    for (uint32_t mid = 0u; mid < UOCR_CLIP_MLP_INTERMEDIATE; ++mid) {
        float lin1 = f16_bits_to_f32(fc1_bias[mid]);
        for (uint32_t k = 0u; k < UOCR_CLIP_HIDDEN_SIZE; ++k) {
            const uint16_t weight_bits = fc1_weight[clip_mlp_fc1_weight_index(mid, k)];
            if (weight_bits != 0u) {
                lin1 += f16_bits_to_f32(input[clip_token_index(token, k)]) * f16_bits_to_f32(weight_bits);
            }
        }
        const uint16_t lin1_f16 = f32_to_f16_bits(lin1);
        const uint16_t activated_f16 = f32_to_f16_bits(quickgelu_expected_from_f16(lin1_f16));
        const uint16_t fc2_weight_bits = fc2_weight[clip_mlp_fc2_weight_index(out_channel, mid)];
        if (fc2_weight_bits != 0u) {
            sum += f16_bits_to_f32(activated_f16) * f16_bits_to_f32(fc2_weight_bits);
        }
    }
    return sum;
}

static int test_metal_clip_mlp_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_MLP_INTERMEDIATE == 4096u);
    CHECK(UOCR_CLIP_GLOBAL_TOKENS == 257u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);

    const size_t fc1_weight_values = (size_t)UOCR_CLIP_MLP_INTERMEDIATE * UOCR_CLIP_HIDDEN_SIZE;
    const size_t fc2_weight_values = (size_t)UOCR_CLIP_HIDDEN_SIZE * UOCR_CLIP_MLP_INTERMEDIATE;
    uint16_t *fc1_weight = (uint16_t *)calloc(fc1_weight_values, sizeof(uint16_t));
    uint16_t *fc1_bias = (uint16_t *)calloc(UOCR_CLIP_MLP_INTERMEDIATE, sizeof(uint16_t));
    uint16_t *fc2_weight = (uint16_t *)calloc(fc2_weight_values, sizeof(uint16_t));
    uint16_t *fc2_bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    CHECK(fc1_weight != NULL);
    CHECK(fc1_bias != NULL);
    CHECK(fc2_weight != NULL);
    CHECK(fc2_bias != NULL);

    const uint32_t active_mids[] = {0u, 7u, 1023u, 2048u, UOCR_CLIP_MLP_INTERMEDIATE - 1u};
    const float fc1_scale[] = {0.25f, -0.5f, 0.75f, -0.125f, 0.375f};
    for (size_t i = 0u; i < sizeof(active_mids) / sizeof(active_mids[0]); ++i) {
        const uint32_t mid = active_mids[i];
        fc1_bias[mid] = f32_to_f16_bits(((float)((int)i - 2)) * 0.05f);
        fc1_weight[clip_mlp_fc1_weight_index(mid, (uint32_t)((i * 113u + 3u) % UOCR_CLIP_HIDDEN_SIZE))] =
            f32_to_f16_bits(fc1_scale[i]);
        fc1_weight[clip_mlp_fc1_weight_index(mid, (uint32_t)((i * 251u + 17u) % UOCR_CLIP_HIDDEN_SIZE))] =
            f32_to_f16_bits(-0.5f * fc1_scale[i]);
    }
    for (uint32_t channel = 0u; channel < UOCR_CLIP_HIDDEN_SIZE; ++channel) {
        fc2_bias[channel] = f32_to_f16_bits(((float)((int)(channel % 13u) - 6)) * 0.02f);
    }
    const uint32_t active_outputs[] = {0u, 3u, 17u, 511u, UOCR_CLIP_HIDDEN_SIZE - 1u};
    for (size_t out_i = 0u; out_i < sizeof(active_outputs) / sizeof(active_outputs[0]); ++out_i) {
        const uint32_t out_channel = active_outputs[out_i];
        for (size_t mid_i = 0u; mid_i < sizeof(active_mids) / sizeof(active_mids[0]); ++mid_i) {
            const int mod = (int)((out_i + 1u) * 7u + (mid_i + 3u) * 5u) % 11 - 5;
            fc2_weight[clip_mlp_fc2_weight_index(out_channel, active_mids[mid_i])] =
                f32_to_f16_bits((float)mod * 0.125f);
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    struct clip_mlp_case {
        uint32_t tokens;
        uocr_metal_dense_output_type output_type;
    } cases[] = {
        {UOCR_CLIP_GLOBAL_TOKENS, UOCR_METAL_DENSE_OUTPUT_F32},
        {UOCR_CLIP_LOCAL_TOKENS, UOCR_METAL_DENSE_OUTPUT_F16},
    };
    for (size_t case_index = 0u; case_index < sizeof(cases) / sizeof(cases[0]); ++case_index) {
        const uint32_t tokens = cases[case_index].tokens;
        const size_t input_values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(input_values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t channel = 0u; channel < UOCR_CLIP_HIDDEN_SIZE; ++channel) {
                const int mod = (int)((token * 11u + channel * 19u + tokens) % 73u) - 36;
                input[clip_token_index(token, channel)] = f32_to_f16_bits((float)mod * 0.01f);
            }
        }

        CHECK(uocr_metal_context_clip_mlp_f16(ctx,
                                              input,
                                              fc1_weight,
                                              fc1_bias,
                                              fc2_weight,
                                              fc2_bias,
                                              tokens,
                                              cases[case_index].output_type,
                                              cases[case_index].output_type == UOCR_METAL_DENSE_OUTPUT_F32 ?
                                                  (void *)out_f32 :
                                                  (void *)out_f16,
                                              error,
                                              sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct clip_mlp_sample {
            uint32_t token;
            uint32_t channel;
        } samples[] = {
            {0u, 0u},
            {tokens / 3u, 3u},
            {tokens / 2u, 17u},
            {tokens - 1u, 511u},
            {tokens - 1u, UOCR_CLIP_HIDDEN_SIZE - 1u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
            const float expected = clip_mlp_expected(input,
                                                     fc1_weight,
                                                     fc1_bias,
                                                     fc2_weight,
                                                     fc2_bias,
                                                     samples[i].token,
                                                     samples[i].channel);
            if (cases[case_index].output_type == UOCR_METAL_DENSE_OUTPUT_F32) {
                CHECK(fabsf(out_f32[idx] - expected) <= 2.5e-4f);
            } else {
                CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
            }
        }

        free(out_f16);
        free(out_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_mlp_f16(ctx,
                                          NULL,
                                          fc1_weight,
                                          fc1_bias,
                                          fc2_weight,
                                          fc2_bias,
                                          UOCR_CLIP_LOCAL_TOKENS,
                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                          &one_out,
                                          error,
                                          sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP MLP") != NULL);

    CHECK(uocr_metal_context_clip_mlp_f16(ctx,
                                          &one_value,
                                          fc1_weight,
                                          fc1_bias,
                                          fc2_weight,
                                          fc2_bias,
                                          UOCR_CLIP_LOCAL_TOKENS - 1u,
                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                          &one_out,
                                          error,
                                          sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP MLP token count") != NULL);

    CHECK(uocr_metal_context_clip_mlp_f16(ctx,
                                          &one_value,
                                          fc1_weight,
                                          fc1_bias,
                                          fc2_weight,
                                          fc2_bias,
                                          UOCR_CLIP_LOCAL_TOKENS,
                                          (uocr_metal_dense_output_type)99,
                                          &one_out,
                                          error,
                                          sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal CLIP MLP output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(fc2_bias);
    free(fc2_weight);
    free(fc1_bias);
    free(fc1_weight);
    return 0;
}

static int test_metal_clip_residual_add_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_GLOBAL_TOKENS == 257u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t token_counts[] = {UOCR_CLIP_GLOBAL_TOKENS, UOCR_CLIP_LOCAL_TOKENS};
    for (size_t case_index = 0u; case_index < sizeof(token_counts) / sizeof(token_counts[0]); ++case_index) {
        const uint32_t tokens = token_counts[case_index];
        const size_t values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        uint16_t *base = (uint16_t *)calloc(values, sizeof(uint16_t));
        uint16_t *update = (uint16_t *)calloc(values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(values, sizeof(uint16_t));
        CHECK(base != NULL);
        CHECK(update != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t channel = 0u; channel < UOCR_CLIP_HIDDEN_SIZE; ++channel) {
                const int base_mod = (int)((token * 7u + channel * 13u + tokens) % 59u) - 29;
                const int update_mod = (int)((token * 17u + channel * 5u + tokens) % 53u) - 26;
                base[clip_token_index(token, channel)] = f32_to_f16_bits((float)base_mod * 0.01f);
                update[clip_token_index(token, channel)] = f32_to_f16_bits((float)update_mod * 0.0125f);
            }
        }

        CHECK(uocr_metal_context_clip_residual_add_f16(ctx,
                                                       base,
                                                       update,
                                                       tokens,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct clip_residual_sample {
            uint32_t token;
            uint32_t channel;
        } samples[] = {
            {0u, 0u},
            {tokens / 3u, 17u},
            {tokens / 2u, 511u},
            {tokens - 1u, 3u},
            {tokens - 1u, UOCR_CLIP_HIDDEN_SIZE - 1u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
            const float expected = f16_bits_to_f32(base[idx]) + f16_bits_to_f32(update[idx]);
            CHECK(fabsf(out_f32[idx] - expected) <= 1.0e-7f);
        }

        CHECK(uocr_metal_context_clip_residual_add_f16(ctx,
                                                       base,
                                                       update,
                                                       tokens,
                                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                                       out_f16,
                                                       error,
                                                       sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
            const float expected = f16_bits_to_f32(base[idx]) + f16_bits_to_f32(update[idx]);
            CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 6.0e-4f);
        }

        free(out_f16);
        free(out_f32);
        free(update);
        free(base);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_residual_add_f16(ctx,
                                                   NULL,
                                                   &one_value,
                                                   UOCR_CLIP_LOCAL_TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   &one_out,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP residual") != NULL);

    CHECK(uocr_metal_context_clip_residual_add_f16(ctx,
                                                   &one_value,
                                                   &one_value,
                                                   UOCR_CLIP_LOCAL_TOKENS - 1u,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   &one_out,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP residual token count") != NULL);

    CHECK(uocr_metal_context_clip_residual_add_f16(ctx,
                                                   &one_value,
                                                   &one_value,
                                                   UOCR_CLIP_LOCAL_TOKENS,
                                                   (uocr_metal_dense_output_type)99,
                                                   &one_out,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal CLIP residual output type") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_clip_transformer_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_BLOCKS == 24u);
    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_QKV_SIZE == 3072u);
    CHECK(UOCR_CLIP_MLP_INTERMEDIATE == 4096u);
    CHECK(UOCR_CLIP_LOCAL_TOKENS == 101u);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t tokens = UOCR_CLIP_LOCAL_TOKENS;
    const size_t hidden_values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
    const size_t qkv_weight_values = (size_t)UOCR_CLIP_QKV_SIZE * UOCR_CLIP_HIDDEN_SIZE;
    const size_t proj_weight_values = (size_t)UOCR_CLIP_HIDDEN_SIZE * UOCR_CLIP_HIDDEN_SIZE;
    const size_t fc1_weight_values = (size_t)UOCR_CLIP_MLP_INTERMEDIATE * UOCR_CLIP_HIDDEN_SIZE;
    const size_t fc2_weight_values = (size_t)UOCR_CLIP_HIDDEN_SIZE * UOCR_CLIP_MLP_INTERMEDIATE;

    uint16_t *input = (uint16_t *)calloc(hidden_values, sizeof(uint16_t));
    uint16_t *ln1_weight = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *ln1_bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *qkv_weight = (uint16_t *)calloc(qkv_weight_values, sizeof(uint16_t));
    uint16_t *qkv_bias = (uint16_t *)calloc(UOCR_CLIP_QKV_SIZE, sizeof(uint16_t));
    uint16_t *out_proj_weight = (uint16_t *)calloc(proj_weight_values, sizeof(uint16_t));
    uint16_t *out_proj_bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *ln2_weight = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *ln2_bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *fc1_weight = (uint16_t *)calloc(fc1_weight_values, sizeof(uint16_t));
    uint16_t *fc1_bias = (uint16_t *)calloc(UOCR_CLIP_MLP_INTERMEDIATE, sizeof(uint16_t));
    uint16_t *fc2_weight = (uint16_t *)calloc(fc2_weight_values, sizeof(uint16_t));
    uint16_t *fc2_bias = (uint16_t *)calloc(UOCR_CLIP_HIDDEN_SIZE, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(hidden_values, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(hidden_values, sizeof(uint16_t));
    uint16_t *stack_out_f16 = (uint16_t *)calloc(hidden_values, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(ln1_weight != NULL);
    CHECK(ln1_bias != NULL);
    CHECK(qkv_weight != NULL);
    CHECK(qkv_bias != NULL);
    CHECK(out_proj_weight != NULL);
    CHECK(out_proj_bias != NULL);
    CHECK(ln2_weight != NULL);
    CHECK(ln2_bias != NULL);
    CHECK(fc1_weight != NULL);
    CHECK(fc1_bias != NULL);
    CHECK(fc2_weight != NULL);
    CHECK(fc2_bias != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(stack_out_f16 != NULL);

    for (uint32_t channel = 0u; channel < UOCR_CLIP_HIDDEN_SIZE; ++channel) {
        ln1_weight[channel] = f32_to_f16_bits(1.0f);
        ln2_weight[channel] = f32_to_f16_bits(1.0f);
        out_proj_bias[channel] = f32_to_f16_bits(((int)(channel % 17u) - 8) * 0.0025f);
        fc2_bias[channel] = f32_to_f16_bits(((int)(channel % 23u) - 11) * 0.00175f);
    }
    for (uint32_t token = 0u; token < tokens; ++token) {
        for (uint32_t channel = 0u; channel < UOCR_CLIP_HIDDEN_SIZE; ++channel) {
            const int mod = (int)((token * 7u + channel * 11u + tokens) % 83u) - 41;
            input[clip_token_index(token, channel)] = f32_to_f16_bits((float)mod * 0.003f);
        }
    }

    uocr_metal_clip_transformer_block_f16 block = {
        ln1_weight,
        ln1_bias,
        qkv_weight,
        qkv_bias,
        out_proj_weight,
        out_proj_bias,
        ln2_weight,
        ln2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    };

    CHECK(uocr_metal_context_clip_transformer_block_f16(ctx,
                                                        input,
                                                        &block,
                                                        tokens,
                                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                                        out_f32,
                                                        error,
                                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_clip_transformer_block_f16(ctx,
                                                        input,
                                                        &block,
                                                        tokens,
                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                        out_f16,
                                                        error,
                                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct clip_transformer_sample {
        uint32_t token;
        uint32_t channel;
    } samples[] = {
        {0u, 0u},
        {tokens / 4u, 17u},
        {tokens / 2u, 511u},
        {tokens - 1u, 7u},
        {tokens - 1u, UOCR_CLIP_HIDDEN_SIZE - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
        const float residual1 = f16_bits_to_f32(f32_to_f16_bits(f16_bits_to_f32(input[idx]) +
                                                               f16_bits_to_f32(out_proj_bias[samples[i].channel])));
        const float expected = residual1 + f16_bits_to_f32(fc2_bias[samples[i].channel]);
        CHECK(fabsf(out_f32[idx] - expected) <= 1.0e-7f);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 6.5e-4f);
    }

    memset(out_proj_bias, 0, UOCR_CLIP_HIDDEN_SIZE * sizeof(uint16_t));
    memset(fc2_bias, 0, UOCR_CLIP_HIDDEN_SIZE * sizeof(uint16_t));
    uocr_metal_clip_transformer_block_f16 blocks[UOCR_CLIP_BLOCKS];
    for (uint32_t i = 0u; i < UOCR_CLIP_BLOCKS; ++i) {
        blocks[i] = block;
    }
    CHECK(uocr_metal_context_clip_transformer_f16(ctx,
                                                  input,
                                                  blocks,
                                                  UOCR_CLIP_BLOCKS,
                                                  tokens,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  stack_out_f16,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = clip_token_index(samples[i].token, samples[i].channel);
        CHECK(stack_out_f16[idx] == input[idx]);
    }

    CHECK(uocr_metal_context_clip_transformer_block_f16(ctx,
                                                        input,
                                                        NULL,
                                                        tokens,
                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                        out_f16,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP transformer block") != NULL);

    CHECK(uocr_metal_context_clip_transformer_block_f16(ctx,
                                                        input,
                                                        &block,
                                                        tokens - 1u,
                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                        out_f16,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP transformer block token count") != NULL);

    CHECK(uocr_metal_context_clip_transformer_f16(ctx,
                                                  input,
                                                  blocks,
                                                  UOCR_CLIP_BLOCKS - 1u,
                                                  tokens,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  stack_out_f16,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP transformer block count") != NULL);

    free(stack_out_f16);
    free(out_f16);
    free(out_f32);
    free(fc2_bias);
    free(fc2_weight);
    free(fc1_bias);
    free(fc1_weight);
    free(ln2_bias);
    free(ln2_weight);
    free(out_proj_bias);
    free(out_proj_weight);
    free(qkv_bias);
    free(qkv_weight);
    free(ln1_bias);
    free(ln1_weight);
    free(input);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_clip_sam_concat_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_SAM_FEATURE_CHANNELS == 1024u);
    CHECK(UOCR_PROJECTOR_IN_SIZE == 2048u);
    CHECK(UOCR_PROJECTOR_IN_SIZE == UOCR_CLIP_HIDDEN_SIZE + UOCR_SAM_FEATURE_CHANNELS);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t grids[] = {UOCR_GLOBAL_GRID_QUERIES, UOCR_LOCAL_GRID_QUERIES};
    for (size_t case_index = 0u; case_index < sizeof(grids) / sizeof(grids[0]); ++case_index) {
        const uint32_t grid = grids[case_index];
        const uint32_t spatial = grid * grid;
        const uint32_t tokens = UOCR_CLIP_CLASS_TOKENS + spatial;
        const size_t clip_values = (size_t)tokens * UOCR_CLIP_HIDDEN_SIZE;
        const size_t sam_values = (size_t)spatial * UOCR_SAM_FEATURE_CHANNELS;
        const size_t output_values = (size_t)spatial * UOCR_PROJECTOR_IN_SIZE;

        uint16_t *clip_tokens = (uint16_t *)calloc(clip_values, sizeof(uint16_t));
        uint16_t *sam_nchw = (uint16_t *)calloc(sam_values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(output_values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
        CHECK(clip_tokens != NULL);
        CHECK(sam_nchw != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t c = 0u; c < UOCR_CLIP_HIDDEN_SIZE; ++c) {
            clip_tokens[clip_token_index(0u, c)] = f32_to_f16_bits(100.0f + (float)c);
        }
        for (uint32_t token = 1u; token < tokens; ++token) {
            for (uint32_t c = 0u; c < UOCR_CLIP_HIDDEN_SIZE; ++c) {
                const int mod = (int)((token * 13u + c * 7u + grid) % 127u) - 63;
                clip_tokens[clip_token_index(token, c)] = f32_to_f16_bits((float)mod * 0.003f);
            }
        }
        for (uint32_t y = 0u; y < grid; ++y) {
            for (uint32_t x = 0u; x < grid; ++x) {
                for (uint32_t c = 0u; c < UOCR_SAM_FEATURE_CHANNELS; ++c) {
                    const int mod = (int)((y * 17u + x * 19u + c * 5u + grid) % 131u) - 65;
                    sam_nchw[sam_neck_nchw_index(grid, grid, c, y, x)] = f32_to_f16_bits((float)mod * 0.0025f);
                }
            }
        }

        CHECK(uocr_metal_context_clip_sam_concat_f16(ctx,
                                                     clip_tokens,
                                                     sam_nchw,
                                                     grid,
                                                     grid,
                                                     UOCR_METAL_DENSE_OUTPUT_F32,
                                                     out_f32,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        CHECK(uocr_metal_context_clip_sam_concat_f16(ctx,
                                                     clip_tokens,
                                                     sam_nchw,
                                                     grid,
                                                     grid,
                                                     UOCR_METAL_DENSE_OUTPUT_F16,
                                                     out_f16,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct concat_sample {
            uint32_t spatial;
            uint32_t col;
        } samples[] = {
            {0u, 0u},
            {0u, UOCR_CLIP_HIDDEN_SIZE - 1u},
            {spatial / 3u, 17u},
            {spatial / 2u, UOCR_CLIP_HIDDEN_SIZE},
            {spatial - 1u, UOCR_PROJECTOR_IN_SIZE - 1u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const uint32_t spatial_index = samples[i].spatial;
            const uint32_t col = samples[i].col;
            const uint32_t y = spatial_index / grid;
            const uint32_t x = spatial_index - y * grid;
            uint16_t expected_bits;
            if (col < UOCR_CLIP_HIDDEN_SIZE) {
                expected_bits = clip_tokens[clip_token_index(spatial_index + UOCR_CLIP_CLASS_TOKENS, col)];
            } else {
                expected_bits = sam_nchw[sam_neck_nchw_index(grid, grid, col - UOCR_CLIP_HIDDEN_SIZE, y, x)];
            }
            const size_t out_index = (size_t)spatial_index * UOCR_PROJECTOR_IN_SIZE + col;
            CHECK(fabsf(out_f32[out_index] - f16_bits_to_f32(expected_bits)) <= 1.0e-7f);
            CHECK(out_f16[out_index] == expected_bits);
            CHECK(out_f16[out_index] != clip_tokens[clip_token_index(0u, col < UOCR_CLIP_HIDDEN_SIZE ? col : 0u)]);
        }

        free(out_f16);
        free(out_f32);
        free(sam_nchw);
        free(clip_tokens);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_clip_sam_concat_f16(ctx,
                                                 NULL,
                                                 &one_value,
                                                 UOCR_LOCAL_GRID_QUERIES,
                                                 UOCR_LOCAL_GRID_QUERIES,
                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal CLIP/SAM concat") != NULL);

    CHECK(uocr_metal_context_clip_sam_concat_f16(ctx,
                                                 &one_value,
                                                 &one_value,
                                                 12u,
                                                 12u,
                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal CLIP/SAM concat grid") != NULL);

    CHECK(uocr_metal_context_clip_sam_concat_f16(ctx,
                                                 &one_value,
                                                 &one_value,
                                                 UOCR_LOCAL_GRID_QUERIES,
                                                 UOCR_LOCAL_GRID_QUERIES,
                                                 (uocr_metal_dense_output_type)99,
                                                 &one_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal CLIP/SAM concat output type") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static float visual_projector_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       const uint16_t *bias,
                                       uint32_t row,
                                       uint32_t out_channel) {
    float sum = f16_bits_to_f32(bias[out_channel]);
    const size_t row_base = (size_t)row * UOCR_PROJECTOR_IN_SIZE;
    for (uint32_t k = 0u; k < UOCR_PROJECTOR_IN_SIZE; ++k) {
        const uint16_t weight_bits = weight[visual_projector_weight_index(out_channel, k)];
        if (weight_bits != 0u) {
            sum += f16_bits_to_f32(input[row_base + k]) * f16_bits_to_f32(weight_bits);
        }
    }
    return sum;
}

static int test_metal_visual_projector_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    CHECK(UOCR_PROJECTOR_IN_SIZE == 2048u);
    CHECK(UOCR_HIDDEN_SIZE == 1280u);
    CHECK(UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES == 256u);
    CHECK(UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES == 100u);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const size_t weight_values = (size_t)UOCR_HIDDEN_SIZE * UOCR_PROJECTOR_IN_SIZE;
    uint16_t *weight = (uint16_t *)calloc(weight_values, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    CHECK(weight != NULL);
    CHECK(bias != NULL);

    for (uint32_t out_channel = 0u; out_channel < UOCR_HIDDEN_SIZE; ++out_channel) {
        bias[out_channel] = f32_to_f16_bits(((int)(out_channel % 29u) - 14) * 0.0025f);
        weight[visual_projector_weight_index(out_channel, out_channel % UOCR_PROJECTOR_IN_SIZE)] =
            f32_to_f16_bits(((int)(out_channel % 9u) - 4) * 0.03125f);
        weight[visual_projector_weight_index(out_channel, (out_channel + UOCR_CLIP_HIDDEN_SIZE) % UOCR_PROJECTOR_IN_SIZE)] =
            f32_to_f16_bits(((int)(out_channel % 11u) - 5) * 0.015625f);
        weight[visual_projector_weight_index(out_channel, (out_channel * 37u + 11u) % UOCR_PROJECTOR_IN_SIZE)] =
            f32_to_f16_bits(((int)(out_channel % 13u) - 6) * 0.0078125f);
    }

    const uint32_t row_counts[] = {
        UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
        UOCR_LOCAL_GRID_QUERIES * UOCR_LOCAL_GRID_QUERIES,
    };
    for (size_t case_index = 0u; case_index < sizeof(row_counts) / sizeof(row_counts[0]); ++case_index) {
        const uint32_t rows = row_counts[case_index];
        const size_t input_values = (size_t)rows * UOCR_PROJECTOR_IN_SIZE;
        const size_t output_values = (size_t)rows * UOCR_HIDDEN_SIZE;
        uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
        float *out_f32 = (float *)calloc(output_values, sizeof(float));
        uint16_t *out_f16 = (uint16_t *)calloc(output_values, sizeof(uint16_t));
        CHECK(input != NULL);
        CHECK(out_f32 != NULL);
        CHECK(out_f16 != NULL);

        for (uint32_t row = 0u; row < rows; ++row) {
            for (uint32_t col = 0u; col < UOCR_PROJECTOR_IN_SIZE; ++col) {
                const int mod = (int)((row * 17u + col * 19u + rows) % 149u) - 74;
                input[(size_t)row * UOCR_PROJECTOR_IN_SIZE + col] = f32_to_f16_bits((float)mod * 0.002f);
            }
        }

        CHECK(uocr_metal_context_visual_projector_f16(ctx,
                                                      input,
                                                      weight,
                                                      bias,
                                                      rows,
                                                      UOCR_METAL_DENSE_OUTPUT_F32,
                                                      out_f32,
                                                      error,
                                                      sizeof(error)) == 1);
        CHECK(error[0] == '\0');

        struct visual_projector_sample {
            uint32_t row;
            uint32_t channel;
        } samples[] = {
            {0u, 0u},
            {rows / 4u, 17u},
            {rows / 2u, 511u},
            {rows - 1u, 777u},
            {rows - 1u, UOCR_HIDDEN_SIZE - 1u},
        };
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = (size_t)samples[i].row * UOCR_HIDDEN_SIZE + samples[i].channel;
            const float expected = visual_projector_expected(input, weight, bias, samples[i].row, samples[i].channel);
            CHECK(fabsf(out_f32[idx] - expected) <= 2.0e-5f);
        }

        CHECK(uocr_metal_context_visual_projector_f16(ctx,
                                                      input,
                                                      weight,
                                                      bias,
                                                      rows,
                                                      UOCR_METAL_DENSE_OUTPUT_F16,
                                                      out_f16,
                                                      error,
                                                      sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            const size_t idx = (size_t)samples[i].row * UOCR_HIDDEN_SIZE + samples[i].channel;
            const float expected = visual_projector_expected(input, weight, bias, samples[i].row, samples[i].channel);
            CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 1.0e-3f);
        }

        free(out_f16);
        free(out_f32);
        free(input);
    }

    uint16_t one_value = f32_to_f16_bits(0.0f);
    float one_out = 0.0f;
    CHECK(uocr_metal_context_visual_projector_f16(ctx,
                                                  NULL,
                                                  weight,
                                                  bias,
                                                  1u,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  &one_out,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal visual projector") != NULL);

    CHECK(uocr_metal_context_visual_projector_f16(ctx,
                                                  &one_value,
                                                  weight,
                                                  bias,
                                                  0u,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  &one_out,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal visual projector") != NULL);

    CHECK(uocr_metal_context_visual_projector_f16(ctx,
                                                  &one_value,
                                                  weight,
                                                  bias,
                                                  1u,
                                                  (uocr_metal_dense_output_type)99,
                                                  &one_out,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal visual projector output type") != NULL);

    free(bias);
    free(weight);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_sam_window_partition_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { GRID_W = 17u, GRID_H = 15u, PADDED_W = 28u, PADDED_H = 28u, WINDOWS = 4u };
    const size_t input_values = (size_t)GRID_W * GRID_H * UOCR_SAM_HIDDEN_SIZE;
    const size_t window_values = (size_t)WINDOWS * UOCR_SAM_WINDOW_TOKENS * UOCR_SAM_HIDDEN_SIZE;
    uint16_t *input = (uint16_t *)calloc(input_values, sizeof(uint16_t));
    uint16_t *windows = (uint16_t *)calloc(window_values, sizeof(uint16_t));
    uint16_t *roundtrip = (uint16_t *)calloc(input_values, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(windows != NULL);
    CHECK(roundtrip != NULL);
    CHECK(UOCR_SAM_WINDOW_SIZE == 14u);
    CHECK(UOCR_SAM_WINDOW_TOKENS == 196u);

    for (uint32_t y = 0u; y < GRID_H; ++y) {
        for (uint32_t x = 0u; x < GRID_W; ++x) {
            for (uint32_t c = 0u; c < UOCR_SAM_HIDDEN_SIZE; ++c) {
                const int mod = (int)((y * 31u + x * 17u + c * 3u) % 97u) - 48;
                input[sam_bhwc_index(GRID_W, y, x, c)] = f32_to_f16_bits((float)mod * 0.004f);
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t n_windows = 0u;
    uint32_t padded_w = 0u;
    uint32_t padded_h = 0u;
    CHECK(uocr_metal_context_sam_window_partition_f16(ctx,
                                                       input,
                                                       GRID_W,
                                                       GRID_H,
                                                       windows,
                                                       &n_windows,
                                                       &padded_w,
                                                       &padded_h,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(n_windows == WINDOWS);
    CHECK(padded_w == PADDED_W);
    CHECK(padded_h == PADDED_H);

    struct partition_sample {
        uint32_t window;
        uint32_t local_y;
        uint32_t local_x;
        uint32_t channel;
        uint32_t src_y;
        uint32_t src_x;
        int valid;
    } samples[] = {
        {0u, 0u, 0u, 0u, 0u, 0u, 1},
        {0u, 13u, 13u, UOCR_SAM_HIDDEN_SIZE - 1u, 13u, 13u, 1},
        {1u, 0u, 0u, 7u, 0u, 14u, 1},
        {1u, 0u, 3u, 11u, 0u, 17u, 0},
        {2u, 0u, 0u, 19u, 14u, 0u, 1},
        {2u, 1u, 0u, 23u, 15u, 0u, 0},
        {3u, 0u, 2u, 29u, 14u, 16u, 1},
        {3u, 1u, 0u, 31u, 15u, 14u, 0},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const uint32_t token = samples[i].local_y * UOCR_SAM_WINDOW_SIZE + samples[i].local_x;
        const size_t dst_idx = sam_window_partition_index(samples[i].window, token, samples[i].channel);
        const uint16_t expected = samples[i].valid ?
                                      input[sam_bhwc_index(GRID_W, samples[i].src_y, samples[i].src_x, samples[i].channel)] :
                                      f32_to_f16_bits(0.0f);
        CHECK(windows[dst_idx] == expected);
    }

    CHECK(uocr_metal_context_sam_window_unpartition_f16(ctx,
                                                         windows,
                                                         GRID_W,
                                                         GRID_H,
                                                         roundtrip,
                                                         error,
                                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < input_values; ++i) {
        CHECK(roundtrip[i] == input[i]);
    }

    CHECK(uocr_metal_context_sam_window_partition_f16(ctx,
                                                       input,
                                                       0u,
                                                       GRID_H,
                                                       windows,
                                                       &n_windows,
                                                       &padded_w,
                                                       &padded_h,
                                                       error,
                                                       sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM window partition") != NULL);

    CHECK(uocr_metal_context_sam_window_unpartition_f16(ctx,
                                                         NULL,
                                                         GRID_W,
                                                         GRID_H,
                                                         roundtrip,
                                                         error,
                                                         sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM window unpartition") != NULL);

    uocr_metal_context_destroy(ctx);
    free(roundtrip);
    free(windows);
    free(input);
    return 0;
}

static float sam_window_attention_expected(const uint16_t *q,
                                           const uint16_t *k,
                                           const uint16_t *v,
                                           uint32_t window,
                                           uint32_t token,
                                           uint32_t head,
                                           uint32_t dim) {
    float scores[UOCR_SAM_WINDOW_TOKENS];
    float max_score = -INFINITY;
    const float scale = 1.0f / sqrtf((float)UOCR_SAM_HEAD_DIM);
    for (uint32_t key = 0u; key < UOCR_SAM_WINDOW_TOKENS; ++key) {
        float score = 0.0f;
        for (uint32_t d = 0u; d < UOCR_SAM_HEAD_DIM; ++d) {
            score += f16_bits_to_f32(q[sam_window_attention_index(window, token, head, d)]) *
                     f16_bits_to_f32(k[sam_window_attention_index(window, key, head, d)]);
        }
        score *= scale;
        scores[key] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    float denominator = 0.0f;
    float value = 0.0f;
    for (uint32_t key = 0u; key < UOCR_SAM_WINDOW_TOKENS; ++key) {
        const float weight = expf(scores[key] - max_score);
        denominator += weight;
        value += weight * f16_bits_to_f32(v[sam_window_attention_index(window, key, head, dim)]);
    }
    return denominator > 0.0f ? value / denominator : 0.0f;
}

static float clip_attention_expected(const uint16_t *q,
                                     const uint16_t *k,
                                     const uint16_t *v,
                                     uint32_t tokens,
                                     uint32_t token,
                                     uint32_t head,
                                     uint32_t dim) {
    float scores[UOCR_CLIP_MAX_TOKENS];
    if (tokens > UOCR_CLIP_MAX_TOKENS) {
        return NAN;
    }
    float max_score = -INFINITY;
    const float scale = 1.0f / sqrtf((float)UOCR_CLIP_HEAD_DIM);
    for (uint32_t key = 0u; key < tokens; ++key) {
        float score = 0.0f;
        for (uint32_t d = 0u; d < UOCR_CLIP_HEAD_DIM; ++d) {
            score += f16_bits_to_f32(q[clip_qkv_out_index(token, head, d)]) *
                     f16_bits_to_f32(k[clip_qkv_out_index(key, head, d)]);
        }
        score *= scale;
        scores[key] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    float denominator = 0.0f;
    float value = 0.0f;
    for (uint32_t key = 0u; key < tokens; ++key) {
        const float weight = expf(scores[key] - max_score);
        denominator += weight;
        value += weight * f16_bits_to_f32(v[clip_qkv_out_index(key, head, dim)]);
    }
    return denominator > 0.0f ? value / denominator : 0.0f;
}

static float sam_global_attention_expected(const uint16_t *q,
                                           const uint16_t *k,
                                           const uint16_t *v,
                                           uint32_t tokens,
                                           uint32_t token,
                                           uint32_t head,
                                           uint32_t dim) {
    float scores[64];
    if (tokens > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
        return NAN;
    }
    float max_score = -INFINITY;
    const float scale = 1.0f / sqrtf((float)UOCR_SAM_HEAD_DIM);
    for (uint32_t key = 0u; key < tokens; ++key) {
        float score = 0.0f;
        for (uint32_t d = 0u; d < UOCR_SAM_HEAD_DIM; ++d) {
            score += f16_bits_to_f32(q[sam_global_attention_index(token, head, d)]) *
                     f16_bits_to_f32(k[sam_global_attention_index(key, head, d)]);
        }
        score *= scale;
        scores[key] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    float denominator = 0.0f;
    float value = 0.0f;
    for (uint32_t key = 0u; key < tokens; ++key) {
        const float weight = expf(scores[key] - max_score);
        denominator += weight;
        value += weight * f16_bits_to_f32(v[sam_global_attention_index(key, head, dim)]);
    }
    return denominator > 0.0f ? value / denominator : 0.0f;
}

static float sam_rel_pos_table_reference(const uint16_t *rel_pos,
                                         uint32_t source_length,
                                         uint32_t target_length,
                                         uint32_t target_index,
                                         uint32_t dim) {
    if (source_length == target_length) {
        return f16_bits_to_f32(rel_pos[(size_t)target_index * UOCR_SAM_HEAD_DIM + dim]);
    }
    const float source_x = ((float)target_index + 0.5f) * (float)source_length / (float)target_length - 0.5f;
    int index0 = (int)floorf(source_x);
    int index1 = index0 + 1;
    const float t = source_x - floorf(source_x);
    if (index0 < 0) index0 = 0;
    if (index1 < 0) index1 = 0;
    if (index0 >= (int)source_length) index0 = (int)source_length - 1;
    if (index1 >= (int)source_length) index1 = (int)source_length - 1;
    const float v0 = f16_bits_to_f32(rel_pos[(size_t)(uint32_t)index0 * UOCR_SAM_HEAD_DIM + dim]);
    const float v1 = f16_bits_to_f32(rel_pos[(size_t)(uint32_t)index1 * UOCR_SAM_HEAD_DIM + dim]);
    return v0 * (1.0f - t) + v1 * t;
}

static float sam_rel_pos_attention_expected(const uint16_t *q,
                                            const uint16_t *k,
                                            const uint16_t *v,
                                            const uint16_t *rel_pos_h,
                                            const uint16_t *rel_pos_w,
                                            uint32_t windows,
                                            uint32_t grid_w,
                                            uint32_t grid_h,
                                            uint32_t rel_pos_h_length,
                                            uint32_t rel_pos_w_length,
                                            uint32_t window,
                                            uint32_t token,
                                            uint32_t head,
                                            uint32_t dim) {
    (void)windows;
    const uint32_t tokens = grid_w * grid_h;
    float scores[64];
    if (tokens > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
        return NAN;
    }
    const uint32_t query_y = token / grid_w;
    const uint32_t query_x = token - query_y * grid_w;
    const uint32_t target_h_length = 2u * grid_h - 1u;
    const uint32_t target_w_length = 2u * grid_w - 1u;
    const float scale = 1.0f / sqrtf((float)UOCR_SAM_HEAD_DIM);
    float max_score = -INFINITY;
    for (uint32_t key = 0u; key < tokens; ++key) {
        const uint32_t key_y = key / grid_w;
        const uint32_t key_x = key - key_y * grid_w;
        const uint32_t rel_h_index = (uint32_t)((int)query_y - (int)key_y + (int)grid_h - 1);
        const uint32_t rel_w_index = (uint32_t)((int)query_x - (int)key_x + (int)grid_w - 1);
        float qk = 0.0f;
        float rel = 0.0f;
        for (uint32_t d = 0u; d < UOCR_SAM_HEAD_DIM; ++d) {
            const float q_value = f16_bits_to_f32(q[sam_rel_pos_attention_index(window, token, head, d, tokens)]);
            qk += q_value * f16_bits_to_f32(k[sam_rel_pos_attention_index(window, key, head, d, tokens)]);
            rel += q_value *
                   (sam_rel_pos_table_reference(rel_pos_h, rel_pos_h_length, target_h_length, rel_h_index, d) +
                    sam_rel_pos_table_reference(rel_pos_w, rel_pos_w_length, target_w_length, rel_w_index, d));
        }
        const float score = qk * scale + rel;
        scores[key] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    float denominator = 0.0f;
    float value = 0.0f;
    for (uint32_t key = 0u; key < tokens; ++key) {
        const float weight = expf(scores[key] - max_score);
        denominator += weight;
        value += weight * f16_bits_to_f32(v[sam_rel_pos_attention_index(window, key, head, dim, tokens)]);
    }
    return denominator > 0.0f ? value / denominator : 0.0f;
}

static float sam_mlp_gelu_expected(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

static float sam_mlp_expected(const uint16_t *input,
                              const uint16_t *lin1_weight,
                              const uint16_t *lin1_bias,
                              const uint16_t *lin2_weight,
                              const uint16_t *lin2_bias,
                              uint32_t row,
                              uint32_t out_col) {
    float value = f16_bits_to_f32(lin2_bias[out_col]);
    for (uint32_t hidden = 0u; hidden < UOCR_SAM_MLP_INTERMEDIATE; ++hidden) {
        float projected = f16_bits_to_f32(lin1_bias[hidden]);
        for (uint32_t col = 0u; col < UOCR_SAM_HIDDEN_SIZE; ++col) {
            projected += f16_bits_to_f32(input[(size_t)row * UOCR_SAM_HIDDEN_SIZE + col]) *
                         f16_bits_to_f32(lin1_weight[sam_mlp_lin1_weight_index(hidden, col)]);
        }
        const float gelu_f16 = f16_bits_to_f32(f32_to_f16_bits(sam_mlp_gelu_expected(projected)));
        value += gelu_f16 * f16_bits_to_f32(lin2_weight[sam_mlp_lin2_weight_index(out_col, hidden)]);
    }
    return value;
}

static float sam_attention_project_residual_expected(const uint16_t *context,
                                                     const uint16_t *weight,
                                                     const uint16_t *bias,
                                                     const uint16_t *residual,
                                                     uint32_t row,
                                                     uint32_t out_col) {
    float value = f16_bits_to_f32(bias[out_col]) +
                  f16_bits_to_f32(residual[(size_t)row * UOCR_SAM_HIDDEN_SIZE + out_col]);
    for (uint32_t col = 0u; col < UOCR_SAM_HIDDEN_SIZE; ++col) {
        value += f16_bits_to_f32(context[(size_t)row * UOCR_SAM_HIDDEN_SIZE + col]) *
                 f16_bits_to_f32(weight[sam_attention_proj_weight_index(out_col, col)]);
    }
    return value;
}

static float sam_neck_conv1x1_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t y,
                                       uint32_t x,
                                       uint32_t out_channel) {
    float value = 0.0f;
    for (uint32_t in_channel = 0u; in_channel < UOCR_SAM_HIDDEN_SIZE; ++in_channel) {
        value += f16_bits_to_f32(input[sam_bhwc_index(grid_w, y, x, in_channel)]) *
                 f16_bits_to_f32(weight[sam_neck_conv1x1_weight_index(out_channel, in_channel)]);
    }
    return value;
}

static float sam_neck_conv3x3_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t y,
                                       uint32_t x,
                                       uint32_t out_channel) {
    float value = 0.0f;
    for (uint32_t in_channel = 0u; in_channel < UOCR_SAM_NECK_CHANNELS; ++in_channel) {
        for (uint32_t ky = 0u; ky < UOCR_SAM_NECK_KERNEL_SIZE; ++ky) {
            const int32_t sy = (int32_t)y + (int32_t)ky - 1;
            if (sy < 0 || sy >= (int32_t)grid_h) {
                continue;
            }
            for (uint32_t kx = 0u; kx < UOCR_SAM_NECK_KERNEL_SIZE; ++kx) {
                const int32_t sx = (int32_t)x + (int32_t)kx - 1;
                if (sx < 0 || sx >= (int32_t)grid_w) {
                    continue;
                }
                value += f16_bits_to_f32(input[sam_neck_nchw_index(grid_w,
                                                                   grid_h,
                                                                   in_channel,
                                                                   (uint32_t)sy,
                                                                   (uint32_t)sx)]) *
                         f16_bits_to_f32(weight[sam_neck_conv3x3_weight_index(out_channel, in_channel, ky, kx)]);
            }
        }
    }
    return value;
}

static float sam_net2_conv3x3_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t out_y,
                                       uint32_t out_x,
                                       uint32_t out_channel) {
    float value = 0.0f;
    for (uint32_t in_channel = 0u; in_channel < UOCR_SAM_NECK_CHANNELS; ++in_channel) {
        for (uint32_t ky = 0u; ky < UOCR_SAM_NECK_KERNEL_SIZE; ++ky) {
            const int32_t sy = (int32_t)(out_y * UOCR_SAM_NET_STRIDE) + (int32_t)ky - 1;
            if (sy < 0 || sy >= (int32_t)grid_h) {
                continue;
            }
            for (uint32_t kx = 0u; kx < UOCR_SAM_NECK_KERNEL_SIZE; ++kx) {
                const int32_t sx = (int32_t)(out_x * UOCR_SAM_NET_STRIDE) + (int32_t)kx - 1;
                if (sx < 0 || sx >= (int32_t)grid_w) {
                    continue;
                }
                value += f16_bits_to_f32(input[sam_neck_nchw_index(grid_w,
                                                                   grid_h,
                                                                   in_channel,
                                                                   (uint32_t)sy,
                                                                   (uint32_t)sx)]) *
                         f16_bits_to_f32(weight[sam_net2_conv3x3_weight_index(out_channel, in_channel, ky, kx)]);
            }
        }
    }
    return value;
}

static float sam_net3_conv3x3_expected(const uint16_t *input,
                                       const uint16_t *weight,
                                       uint32_t grid_w,
                                       uint32_t grid_h,
                                       uint32_t out_y,
                                       uint32_t out_x,
                                       uint32_t out_channel) {
    float value = 0.0f;
    for (uint32_t in_channel = 0u; in_channel < UOCR_SAM_NET2_CHANNELS; ++in_channel) {
        for (uint32_t ky = 0u; ky < UOCR_SAM_NECK_KERNEL_SIZE; ++ky) {
            const int32_t sy = (int32_t)(out_y * UOCR_SAM_NET_STRIDE) + (int32_t)ky - 1;
            if (sy < 0 || sy >= (int32_t)grid_h) {
                continue;
            }
            for (uint32_t kx = 0u; kx < UOCR_SAM_NECK_KERNEL_SIZE; ++kx) {
                const int32_t sx = (int32_t)(out_x * UOCR_SAM_NET_STRIDE) + (int32_t)kx - 1;
                if (sx < 0 || sx >= (int32_t)grid_w) {
                    continue;
                }
                value += f16_bits_to_f32(input[sam_neck_nchw_index(grid_w,
                                                                   grid_h,
                                                                   in_channel,
                                                                   (uint32_t)sy,
                                                                   (uint32_t)sx)]) *
                         f16_bits_to_f32(weight[sam_net3_conv3x3_weight_index(out_channel, in_channel, ky, kx)]);
            }
        }
    }
    return value;
}

static int test_metal_sam_qkv_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3u, HIDDEN = UOCR_SAM_HIDDEN_SIZE, QKV = UOCR_SAM_QKV_SIZE };
    uint16_t *input = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc((size_t)QKV * HIDDEN, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(QKV, sizeof(uint16_t));
    float *q_f32 = (float *)calloc((size_t)ROWS * HIDDEN, sizeof(float));
    float *k_f32 = (float *)calloc((size_t)ROWS * HIDDEN, sizeof(float));
    float *v_f32 = (float *)calloc((size_t)ROWS * HIDDEN, sizeof(float));
    uint16_t *q_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *k_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *v_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(weight != NULL);
    CHECK(bias != NULL);
    CHECK(q_f32 != NULL);
    CHECK(k_f32 != NULL);
    CHECK(v_f32 != NULL);
    CHECK(q_f16 != NULL);
    CHECK(k_f16 != NULL);
    CHECK(v_f16 != NULL);

    for (uint32_t row = 0u; row < ROWS; ++row) {
        for (uint32_t col = 0u; col < HIDDEN; ++col) {
            input[(size_t)row * HIDDEN + col] = f32_to_f16_bits((float)(row * 100u + col) * 0.01f);
        }
    }

    /* Q head 0 dim 0 = input[0] + 0.5 */
    weight[sam_qkv_weight_index(0u, 0u, 0u, 0u)] = f32_to_f16_bits(1.0f);
    bias[sam_qkv_packed_col(0u, 0u, 0u)] = f32_to_f16_bits(0.5f);
    /* Q head 3 dim 4 is bias-only. */
    bias[sam_qkv_packed_col(0u, 3u, 4u)] = f32_to_f16_bits(3.0f);
    /* K head 2 dim 3 = 2 * input[5] - 1. */
    weight[sam_qkv_weight_index(1u, 2u, 3u, 5u)] = f32_to_f16_bits(2.0f);
    bias[sam_qkv_packed_col(1u, 2u, 3u)] = f32_to_f16_bits(-1.0f);
    /* V head 11 dim 63 = input[7] - input[9] + 0.25. */
    weight[sam_qkv_weight_index(2u, 11u, 63u, 7u)] = f32_to_f16_bits(1.0f);
    weight[sam_qkv_weight_index(2u, 11u, 63u, 9u)] = f32_to_f16_bits(-1.0f);
    bias[sam_qkv_packed_col(2u, 11u, 63u)] = f32_to_f16_bits(0.25f);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_qkv_f16(ctx,
                                         input,
                                         weight,
                                         bias,
                                         ROWS,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_f32,
                                         k_f32,
                                         v_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint32_t row = 0u; row < ROWS; ++row) {
        const float x0 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 0u]);
        const float x5 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 5u]);
        const float x7 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 7u]);
        const float x9 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 9u]);
        CHECK(fabsf(q_f32[sam_qkv_out_index(row, 0u, 0u)] - (x0 + 0.5f)) <= 1.0e-5f);
        CHECK(fabsf(q_f32[sam_qkv_out_index(row, 3u, 4u)] - 3.0f) <= 1.0e-5f);
        CHECK(fabsf(k_f32[sam_qkv_out_index(row, 2u, 3u)] - (2.0f * x5 - 1.0f)) <= 1.0e-5f);
        CHECK(fabsf(v_f32[sam_qkv_out_index(row, 11u, 63u)] - (x7 - x9 + 0.25f)) <= 1.0e-5f);
        CHECK(fabsf(q_f32[sam_qkv_out_index(row, 5u, 5u)]) <= 1.0e-5f);
        CHECK(fabsf(k_f32[sam_qkv_out_index(row, 5u, 5u)]) <= 1.0e-5f);
        CHECK(fabsf(v_f32[sam_qkv_out_index(row, 5u, 5u)]) <= 1.0e-5f);
    }

    CHECK(uocr_metal_context_sam_qkv_f16(ctx,
                                         input,
                                         weight,
                                         bias,
                                         ROWS,
                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                         q_f16,
                                         k_f16,
                                         v_f16,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < ROWS; ++row) {
        const float x0 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 0u]);
        const float x5 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 5u]);
        const float x7 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 7u]);
        const float x9 = f16_bits_to_f32(input[(size_t)row * HIDDEN + 9u]);
        CHECK(q_f16[sam_qkv_out_index(row, 0u, 0u)] == f32_to_f16_bits(x0 + 0.5f));
        CHECK(q_f16[sam_qkv_out_index(row, 3u, 4u)] == f32_to_f16_bits(3.0f));
        CHECK(k_f16[sam_qkv_out_index(row, 2u, 3u)] == f32_to_f16_bits(2.0f * x5 - 1.0f));
        CHECK(v_f16[sam_qkv_out_index(row, 11u, 63u)] == f32_to_f16_bits(x7 - x9 + 0.25f));
    }

    CHECK(uocr_metal_context_sam_qkv_f16(ctx,
                                         input,
                                         weight,
                                         bias,
                                         0u,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_f32,
                                         k_f32,
                                         v_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM QKV") != NULL);

    uocr_metal_context_destroy(ctx);
    free(v_f16);
    free(k_f16);
    free(q_f16);
    free(v_f32);
    free(k_f32);
    free(q_f32);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_sam_window_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        WINDOWS = 1u,
        VALUES = WINDOWS * UOCR_SAM_WINDOW_TOKENS * UOCR_SAM_HIDDEN_SIZE,
    };
    uint16_t *q = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    uint16_t *k = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    uint16_t *v = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(VALUES, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    CHECK(q != NULL);
    CHECK(k != NULL);
    CHECK(v != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);

    for (uint32_t window = 0u; window < WINDOWS; ++window) {
        for (uint32_t token = 0u; token < UOCR_SAM_WINDOW_TOKENS; ++token) {
            for (uint32_t head = 0u; head < UOCR_SAM_ATTENTION_HEADS; ++head) {
                for (uint32_t dim = 0u; dim < UOCR_SAM_HEAD_DIM; ++dim) {
                    const size_t idx = sam_window_attention_index(window, token, head, dim);
                    const int q_mod = (int)((token + 3u * dim + 5u * head + 7u * window) % 17u) - 8;
                    const int k_mod = (int)((3u * token + dim + 11u * head + 13u * window) % 19u) - 9;
                    const int v_mod = (int)((5u * token + 7u * dim + 2u * head + window) % 23u) - 11;
                    q[idx] = f32_to_f16_bits((float)q_mod * 0.015f);
                    k[idx] = f32_to_f16_bits((float)k_mod * 0.0125f);
                    v[idx] = f32_to_f16_bits((float)v_mod * 0.02f);
                }
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_window_attention_f16(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       WINDOWS,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct sample {
        uint32_t window;
        uint32_t token;
        uint32_t head;
        uint32_t dim;
    } samples[] = {
        {0u, 0u, 0u, 0u},
        {0u, 13u, 2u, 5u},
        {0u, 27u, 5u, 17u},
        {0u, 91u, 8u, 33u},
        {0u, UOCR_SAM_WINDOW_TOKENS - 1u, UOCR_SAM_ATTENTION_HEADS - 1u, UOCR_SAM_HEAD_DIM - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_window_attention_index(samples[i].window, samples[i].token, samples[i].head, samples[i].dim);
        const float expected = sam_window_attention_expected(q,
                                                             k,
                                                             v,
                                                             samples[i].window,
                                                             samples[i].token,
                                                             samples[i].head,
                                                             samples[i].dim);
        CHECK(fabsf(out_f32[idx] - expected) <= 2.0e-4f);
    }

    CHECK(uocr_metal_context_sam_window_attention_f16(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       WINDOWS,
                                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                                       out_f16,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_window_attention_index(samples[i].window, samples[i].token, samples[i].head, samples[i].dim);
        const float expected = sam_window_attention_expected(q,
                                                             k,
                                                             v,
                                                             samples[i].window,
                                                             samples[i].token,
                                                             samples[i].head,
                                                             samples[i].dim);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_window_attention_f16(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       0u,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM window attention") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_metal_sam_global_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { GRID_W = 6u, GRID_H = 5u, TOKENS = GRID_W * GRID_H, VALUES = TOKENS * UOCR_SAM_HIDDEN_SIZE };
    uint16_t *q = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    uint16_t *k = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    uint16_t *v = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(VALUES, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    CHECK(q != NULL);
    CHECK(k != NULL);
    CHECK(v != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);

    CHECK(UOCR_SAM_BLOCKS == 12u);
    CHECK(UOCR_SAM_WINDOW_REL_POS_SIZE == 27u);
    CHECK(UOCR_SAM_MAX_REL_POS_SIZE == 127u);
    for (uint32_t block = 0u; block < UOCR_SAM_BLOCKS; ++block) {
        const int expected_global = block == 2u || block == 5u || block == 8u || block == 11u;
        CHECK(uocr_sam_block_uses_global_attention(block) == expected_global);
    }
    CHECK(uocr_sam_block_uses_global_attention(UOCR_SAM_BLOCKS) == 0);

    for (uint32_t token = 0u; token < TOKENS; ++token) {
        for (uint32_t head = 0u; head < UOCR_SAM_ATTENTION_HEADS; ++head) {
            for (uint32_t dim = 0u; dim < UOCR_SAM_HEAD_DIM; ++dim) {
                const size_t idx = sam_global_attention_index(token, head, dim);
                const int q_mod = (int)((token + 2u * dim + 3u * head) % 23u) - 11;
                const int k_mod = (int)((5u * token + dim + 7u * head) % 29u) - 14;
                const int v_mod = (int)((3u * token + 11u * dim + head) % 31u) - 15;
                q[idx] = f32_to_f16_bits((float)q_mod * 0.01f);
                k[idx] = f32_to_f16_bits((float)k_mod * 0.009f);
                v[idx] = f32_to_f16_bits((float)v_mod * 0.0175f);
            }
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_global_attention_f16(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       GRID_W,
                                                       GRID_H,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct sample {
        uint32_t token;
        uint32_t head;
        uint32_t dim;
    } samples[] = {
        {0u, 0u, 0u},
        {3u, 1u, 7u},
        {11u, 4u, 19u},
        {23u, 9u, 41u},
        {TOKENS - 1u, UOCR_SAM_ATTENTION_HEADS - 1u, UOCR_SAM_HEAD_DIM - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_global_attention_index(samples[i].token, samples[i].head, samples[i].dim);
        const float expected = sam_global_attention_expected(q,
                                                             k,
                                                             v,
                                                             TOKENS,
                                                             samples[i].token,
                                                             samples[i].head,
                                                             samples[i].dim);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 2.0e-4f);
    }

    CHECK(uocr_metal_context_sam_global_attention_f16(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       GRID_W,
                                                       GRID_H,
                                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                                       out_f16,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_global_attention_index(samples[i].token, samples[i].head, samples[i].dim);
        const float expected = sam_global_attention_expected(q,
                                                             k,
                                                             v,
                                                             TOKENS,
                                                             samples[i].token,
                                                             samples[i].head,
                                                             samples[i].dim);
        CHECK(isfinite(expected));
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_global_attention_f16(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       0u,
                                                       GRID_H,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM global attention") != NULL);
    CHECK(uocr_metal_context_sam_global_attention_f16(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       UOCR_SAM_MAX_GRID_SIZE + 1u,
                                                       1u,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_metal_sam_rel_pos_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        WINDOWS = 2u,
        GRID_W = 4u,
        GRID_H = 3u,
        TOKENS = GRID_W * GRID_H,
        VALUES = WINDOWS * TOKENS * UOCR_SAM_HIDDEN_SIZE,
        REL_H = 2u * GRID_H - 1u,
        REL_W = 11u,
    };
    uint16_t *q = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    uint16_t *k = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    uint16_t *v = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    uint16_t *rel_h = (uint16_t *)calloc((size_t)REL_H * UOCR_SAM_HEAD_DIM, sizeof(uint16_t));
    uint16_t *rel_w = (uint16_t *)calloc((size_t)REL_W * UOCR_SAM_HEAD_DIM, sizeof(uint16_t));
    float *out_f32 = (float *)calloc(VALUES, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc(VALUES, sizeof(uint16_t));
    CHECK(q != NULL);
    CHECK(k != NULL);
    CHECK(v != NULL);
    CHECK(rel_h != NULL);
    CHECK(rel_w != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);

    for (uint32_t window = 0u; window < WINDOWS; ++window) {
        for (uint32_t token = 0u; token < TOKENS; ++token) {
            for (uint32_t head = 0u; head < UOCR_SAM_ATTENTION_HEADS; ++head) {
                for (uint32_t dim = 0u; dim < UOCR_SAM_HEAD_DIM; ++dim) {
                    const size_t idx = sam_rel_pos_attention_index(window, token, head, dim, TOKENS);
                    const int q_mod = (int)((window + 2u * token + 3u * head + dim) % 19u) - 9;
                    const int k_mod = (int)((5u * window + token + head + 2u * dim) % 23u) - 11;
                    const int v_mod = (int)((7u * window + 3u * token + head + dim) % 29u) - 14;
                    q[idx] = f32_to_f16_bits((float)q_mod * 0.0125f);
                    k[idx] = f32_to_f16_bits((float)k_mod * 0.01f);
                    v[idx] = f32_to_f16_bits((float)v_mod * 0.015f);
                }
            }
        }
    }
    for (uint32_t i = 0u; i < REL_H; ++i) {
        for (uint32_t dim = 0u; dim < UOCR_SAM_HEAD_DIM; ++dim) {
            const int mod = (int)((3u * i + dim) % 17u) - 8;
            rel_h[(size_t)i * UOCR_SAM_HEAD_DIM + dim] = f32_to_f16_bits((float)mod * 0.004f);
        }
    }
    for (uint32_t i = 0u; i < REL_W; ++i) {
        for (uint32_t dim = 0u; dim < UOCR_SAM_HEAD_DIM; ++dim) {
            const int mod = (int)((5u * i + 2u * dim) % 19u) - 9;
            rel_w[(size_t)i * UOCR_SAM_HEAD_DIM + dim] = f32_to_f16_bits((float)mod * 0.0035f);
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_rel_pos_attention_f16(ctx,
                                                        q,
                                                        k,
                                                        v,
                                                        rel_h,
                                                        rel_w,
                                                        WINDOWS,
                                                        GRID_W,
                                                        GRID_H,
                                                        REL_H,
                                                        REL_W,
                                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                                        out_f32,
                                                        error,
                                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct rel_sample {
        uint32_t window;
        uint32_t token;
        uint32_t head;
        uint32_t dim;
    } samples[] = {
        {0u, 0u, 0u, 0u},
        {0u, 5u, 2u, 7u},
        {1u, 3u, 4u, 19u},
        {1u, TOKENS - 1u, UOCR_SAM_ATTENTION_HEADS - 1u, UOCR_SAM_HEAD_DIM - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_rel_pos_attention_index(samples[i].window,
                                                       samples[i].token,
                                                       samples[i].head,
                                                       samples[i].dim,
                                                       TOKENS);
        const float expected = sam_rel_pos_attention_expected(q,
                                                              k,
                                                              v,
                                                              rel_h,
                                                              rel_w,
                                                              WINDOWS,
                                                              GRID_W,
                                                              GRID_H,
                                                              REL_H,
                                                              REL_W,
                                                              samples[i].window,
                                                              samples[i].token,
                                                              samples[i].head,
                                                              samples[i].dim);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 2.5e-4f);
    }

    CHECK(uocr_metal_context_sam_rel_pos_attention_f16(ctx,
                                                        q,
                                                        k,
                                                        v,
                                                        rel_h,
                                                        rel_w,
                                                        WINDOWS,
                                                        GRID_W,
                                                        GRID_H,
                                                        REL_H,
                                                        REL_W,
                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                        out_f16,
                                                        error,
                                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_rel_pos_attention_index(samples[i].window,
                                                       samples[i].token,
                                                       samples[i].head,
                                                       samples[i].dim,
                                                       TOKENS);
        const float expected = sam_rel_pos_attention_expected(q,
                                                              k,
                                                              v,
                                                              rel_h,
                                                              rel_w,
                                                              WINDOWS,
                                                              GRID_W,
                                                              GRID_H,
                                                              REL_H,
                                                              REL_W,
                                                              samples[i].window,
                                                              samples[i].token,
                                                              samples[i].head,
                                                              samples[i].dim);
        CHECK(isfinite(expected));
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_rel_pos_attention_f16(ctx,
                                                        q,
                                                        k,
                                                        v,
                                                        rel_h,
                                                        rel_w,
                                                        0u,
                                                        GRID_W,
                                                        GRID_H,
                                                        REL_H,
                                                        REL_W,
                                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                                        out_f32,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM relative-position attention") != NULL);
    CHECK(uocr_metal_context_sam_rel_pos_attention_f16(ctx,
                                                        q,
                                                        k,
                                                        v,
                                                        rel_h,
                                                        rel_w,
                                                        WINDOWS,
                                                        GRID_W,
                                                        GRID_H,
                                                        UOCR_SAM_MAX_REL_POS_SIZE + 1u,
                                                        REL_W,
                                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                                        out_f32,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "exceed") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(rel_w);
    free(rel_h);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_metal_sam_mlp_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3u, HIDDEN = UOCR_SAM_HIDDEN_SIZE, INTERMEDIATE = UOCR_SAM_MLP_INTERMEDIATE };
    uint16_t *input = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *lin1_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *lin1_bias = (uint16_t *)calloc(INTERMEDIATE, sizeof(uint16_t));
    uint16_t *lin2_weight = (uint16_t *)calloc((size_t)HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    uint16_t *lin2_bias = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    float *out_f32 = (float *)calloc((size_t)ROWS * HIDDEN, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(lin1_weight != NULL);
    CHECK(lin1_bias != NULL);
    CHECK(lin2_weight != NULL);
    CHECK(lin2_bias != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(UOCR_SAM_MLP_RATIO == 4u);
    CHECK(UOCR_SAM_MLP_INTERMEDIATE == 3072u);

    for (uint32_t row = 0u; row < ROWS; ++row) {
        for (uint32_t col = 0u; col < HIDDEN; ++col) {
            const int mod = (int)((7u * row + 3u * col) % 23u) - 11;
            input[(size_t)row * HIDDEN + col] = f32_to_f16_bits((float)mod * 0.0125f);
        }
    }

    const uint32_t active_hidden[] = {0u, 5u, 17u, 1024u, INTERMEDIATE - 1u};
    for (size_t i = 0u; i < sizeof(active_hidden) / sizeof(active_hidden[0]); ++i) {
        const uint32_t hidden = active_hidden[i];
        lin1_bias[hidden] = f32_to_f16_bits(((int)i - 2) * 0.035f);
        lin1_weight[sam_mlp_lin1_weight_index(hidden, (uint32_t)(1u + i))] = f32_to_f16_bits(0.18f - (float)i * 0.015f);
        lin1_weight[sam_mlp_lin1_weight_index(hidden, (uint32_t)(31u + 3u * i))] = f32_to_f16_bits(-0.11f + (float)i * 0.02f);
        lin1_weight[sam_mlp_lin1_weight_index(hidden, HIDDEN - 1u - (uint32_t)i)] = f32_to_f16_bits(0.045f * (float)(i + 1u));
    }

    const uint32_t active_out[] = {0u, 3u, 77u, HIDDEN - 1u};
    for (size_t o = 0u; o < sizeof(active_out) / sizeof(active_out[0]); ++o) {
        const uint32_t out_col = active_out[o];
        lin2_bias[out_col] = f32_to_f16_bits(((int)o - 1) * 0.02f);
        for (size_t i = 0u; i < sizeof(active_hidden) / sizeof(active_hidden[0]); ++i) {
            const float sign = ((o + i) & 1u) ? -1.0f : 1.0f;
            lin2_weight[sam_mlp_lin2_weight_index(out_col, active_hidden[i])] =
                f32_to_f16_bits(sign * (0.09f + 0.01f * (float)o + 0.006f * (float)i));
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_mlp_f16(ctx,
                                          input,
                                          lin1_weight,
                                          lin1_bias,
                                          lin2_weight,
                                          lin2_bias,
                                          ROWS,
                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                          out_f32,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct mlp_sample {
        uint32_t row;
        uint32_t out_col;
    } samples[] = {
        {0u, 0u},
        {1u, 3u},
        {2u, 77u},
        {2u, HIDDEN - 1u},
        {1u, 11u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const float expected = sam_mlp_expected(input,
                                                lin1_weight,
                                                lin1_bias,
                                                lin2_weight,
                                                lin2_bias,
                                                samples[i].row,
                                                samples[i].out_col);
        const size_t idx = (size_t)samples[i].row * HIDDEN + samples[i].out_col;
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 3.0e-4f);
    }

    CHECK(uocr_metal_context_sam_mlp_f16(ctx,
                                          input,
                                          lin1_weight,
                                          lin1_bias,
                                          lin2_weight,
                                          lin2_bias,
                                          ROWS,
                                          UOCR_METAL_DENSE_OUTPUT_F16,
                                          out_f16,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const float expected = sam_mlp_expected(input,
                                                lin1_weight,
                                                lin1_bias,
                                                lin2_weight,
                                                lin2_bias,
                                                samples[i].row,
                                                samples[i].out_col);
        const size_t idx = (size_t)samples[i].row * HIDDEN + samples[i].out_col;
        CHECK(isfinite(expected));
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 2.0e-3f);
    }

    CHECK(uocr_metal_context_sam_mlp_f16(ctx,
                                          input,
                                          lin1_weight,
                                          NULL,
                                          lin2_weight,
                                          lin2_bias,
                                          ROWS,
                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                          out_f32,
                                          error,
                                          sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM MLP") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(lin2_bias);
    free(lin2_weight);
    free(lin1_bias);
    free(lin1_weight);
    free(input);
    return 0;
}

static int test_metal_sam_residuals_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3u, HIDDEN = UOCR_SAM_HIDDEN_SIZE };
    uint16_t *context = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *update = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc((size_t)HIDDEN * HIDDEN, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    float *out_f32 = (float *)calloc((size_t)ROWS * HIDDEN, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    CHECK(context != NULL);
    CHECK(residual != NULL);
    CHECK(update != NULL);
    CHECK(weight != NULL);
    CHECK(bias != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);

    for (uint32_t row = 0u; row < ROWS; ++row) {
        for (uint32_t col = 0u; col < HIDDEN; ++col) {
            const size_t idx = (size_t)row * HIDDEN + col;
            context[idx] = f32_to_f16_bits(((int)((row * 11u + col * 5u) % 29u) - 14) * 0.0075f);
            residual[idx] = f32_to_f16_bits(((int)((row * 7u + col * 3u) % 31u) - 15) * 0.011f);
            update[idx] = f32_to_f16_bits(((int)((row * 13u + col * 2u) % 23u) - 11) * 0.009f);
        }
    }

    const uint32_t active_out[] = {0u, 17u, 101u, HIDDEN - 1u};
    for (size_t i = 0u; i < sizeof(active_out) / sizeof(active_out[0]); ++i) {
        const uint32_t out_col = active_out[i];
        bias[out_col] = f32_to_f16_bits(((int)i - 1) * 0.025f);
        weight[sam_attention_proj_weight_index(out_col, (uint32_t)(2u + i))] =
            f32_to_f16_bits(0.12f - 0.01f * (float)i);
        weight[sam_attention_proj_weight_index(out_col, (uint32_t)(37u + 5u * i))] =
            f32_to_f16_bits(-0.075f + 0.015f * (float)i);
        weight[sam_attention_proj_weight_index(out_col, HIDDEN - 1u - (uint32_t)i)] =
            f32_to_f16_bits(0.02f * (float)(i + 1u));
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_attention_project_residual_f16(ctx,
                                                                 context,
                                                                 weight,
                                                                 bias,
                                                                 residual,
                                                                 ROWS,
                                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                                 out_f32,
                                                                 error,
                                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct residual_sample {
        uint32_t row;
        uint32_t col;
    } samples[] = {
        {0u, 0u},
        {1u, 17u},
        {2u, 101u},
        {2u, HIDDEN - 1u},
        {1u, 23u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = (size_t)samples[i].row * HIDDEN + samples[i].col;
        const float expected = sam_attention_project_residual_expected(context,
                                                                       weight,
                                                                       bias,
                                                                       residual,
                                                                       samples[i].row,
                                                                       samples[i].col);
        CHECK(isfinite(expected));
        CHECK(fabsf(out_f32[idx] - expected) <= 2.0e-5f);
    }

    CHECK(uocr_metal_context_sam_attention_project_residual_f16(ctx,
                                                                 context,
                                                                 weight,
                                                                 bias,
                                                                 residual,
                                                                 ROWS,
                                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                                 out_f16,
                                                                 error,
                                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = (size_t)samples[i].row * HIDDEN + samples[i].col;
        const float expected = sam_attention_project_residual_expected(context,
                                                                       weight,
                                                                       bias,
                                                                       residual,
                                                                       samples[i].row,
                                                                       samples[i].col);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 1.5e-3f);
    }

    CHECK(uocr_metal_context_sam_residual_add_f16(ctx,
                                                   residual,
                                                   update,
                                                   ROWS,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = (size_t)samples[i].row * HIDDEN + samples[i].col;
        const float expected = f16_bits_to_f32(residual[idx]) + f16_bits_to_f32(update[idx]);
        CHECK(fabsf(out_f32[idx] - expected) <= 1.0e-7f);
    }

    CHECK(uocr_metal_context_sam_residual_add_f16(ctx,
                                                   residual,
                                                   update,
                                                   ROWS,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   out_f16,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = (size_t)samples[i].row * HIDDEN + samples[i].col;
        const float expected = f16_bits_to_f32(residual[idx]) + f16_bits_to_f32(update[idx]);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 8.0e-4f);
    }

    CHECK(uocr_metal_context_sam_residual_add_f16(ctx,
                                                   residual,
                                                   NULL,
                                                   ROWS,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM residual") != NULL);

    CHECK(uocr_metal_context_sam_attention_project_residual_f16(ctx,
                                                                 context,
                                                                 weight,
                                                                 bias,
                                                                 NULL,
                                                                 ROWS,
                                                                 UOCR_METAL_DENSE_OUTPUT_F32,
                                                                 out_f32,
                                                                 error,
                                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM attention residual") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(bias);
    free(weight);
    free(update);
    free(residual);
    free(context);
    return 0;
}

static int test_metal_sam_transformer_block_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        GRID_W = 3u,
        GRID_H = 2u,
        ROWS = GRID_W * GRID_H,
        HIDDEN = UOCR_SAM_HIDDEN_SIZE,
        QKV = UOCR_SAM_QKV_SIZE,
        MLP = UOCR_SAM_MLP_INTERMEDIATE,
    };
    CHECK(UOCR_SAM_BLOCKS == 12u);
    CHECK(UOCR_SAM_WINDOW_SIZE == 14u);
    CHECK(UOCR_SAM_WINDOW_REL_POS_SIZE == 27u);
    CHECK(UOCR_SAM_MAX_REL_POS_SIZE == 127u);

    uint16_t *input = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *norm1_weight = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    uint16_t *norm1_bias = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    uint16_t *qkv_weight = (uint16_t *)calloc((size_t)QKV * HIDDEN, sizeof(uint16_t));
    uint16_t *qkv_bias = (uint16_t *)calloc(QKV, sizeof(uint16_t));
    uint16_t *proj_weight = (uint16_t *)calloc((size_t)HIDDEN * HIDDEN, sizeof(uint16_t));
    uint16_t *proj_bias = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    uint16_t *rel_h = (uint16_t *)calloc((size_t)UOCR_SAM_MAX_REL_POS_SIZE * UOCR_SAM_HEAD_DIM, sizeof(uint16_t));
    uint16_t *rel_w = (uint16_t *)calloc((size_t)UOCR_SAM_MAX_REL_POS_SIZE * UOCR_SAM_HEAD_DIM, sizeof(uint16_t));
    uint16_t *norm2_weight = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    uint16_t *norm2_bias = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    uint16_t *lin1_weight = (uint16_t *)calloc((size_t)MLP * HIDDEN, sizeof(uint16_t));
    uint16_t *lin1_bias = (uint16_t *)calloc(MLP, sizeof(uint16_t));
    uint16_t *lin2_weight = (uint16_t *)calloc((size_t)HIDDEN * MLP, sizeof(uint16_t));
    uint16_t *lin2_bias = (uint16_t *)calloc(HIDDEN, sizeof(uint16_t));
    float *out_f32 = (float *)calloc((size_t)ROWS * HIDDEN, sizeof(float));
    uint16_t *out_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    uint16_t *stack_out_f16 = (uint16_t *)calloc((size_t)ROWS * HIDDEN, sizeof(uint16_t));
    CHECK(input != NULL);
    CHECK(norm1_weight != NULL);
    CHECK(norm1_bias != NULL);
    CHECK(qkv_weight != NULL);
    CHECK(qkv_bias != NULL);
    CHECK(proj_weight != NULL);
    CHECK(proj_bias != NULL);
    CHECK(rel_h != NULL);
    CHECK(rel_w != NULL);
    CHECK(norm2_weight != NULL);
    CHECK(norm2_bias != NULL);
    CHECK(lin1_weight != NULL);
    CHECK(lin1_bias != NULL);
    CHECK(lin2_weight != NULL);
    CHECK(lin2_bias != NULL);
    CHECK(out_f32 != NULL);
    CHECK(out_f16 != NULL);
    CHECK(stack_out_f16 != NULL);

    for (uint32_t channel = 0u; channel < HIDDEN; ++channel) {
        norm1_weight[channel] = f32_to_f16_bits(1.0f);
        norm2_weight[channel] = f32_to_f16_bits(1.0f);
        proj_bias[channel] = f32_to_f16_bits(((int)(channel % 17u) - 8) * 0.0025f);
        lin2_bias[channel] = f32_to_f16_bits(((int)(channel % 23u) - 11) * 0.00175f);
    }
    for (uint32_t y = 0u; y < GRID_H; ++y) {
        for (uint32_t x = 0u; x < GRID_W; ++x) {
            for (uint32_t channel = 0u; channel < HIDDEN; ++channel) {
                const int mod = (int)((y * 31u + x * 17u + channel * 7u) % 97u) - 48;
                input[sam_bhwc_index(GRID_W, y, x, channel)] = f32_to_f16_bits((float)mod * 0.0025f);
            }
        }
    }

    uocr_metal_sam_transformer_block_f16 block = {
        norm1_weight,
        norm1_bias,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        rel_h,
        rel_w,
        UOCR_SAM_MAX_REL_POS_SIZE,
        UOCR_SAM_MAX_REL_POS_SIZE,
        norm2_weight,
        norm2_bias,
        lin1_weight,
        lin1_bias,
        lin2_weight,
        lin2_bias,
    };

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_sam_transformer_block_f16(ctx,
                                                       input,
                                                       &block,
                                                       GRID_W,
                                                       GRID_H,
                                                       1,
                                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                                       out_f32,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    struct sam_transformer_sample {
        uint32_t y;
        uint32_t x;
        uint32_t channel;
    } samples[] = {
        {0u, 0u, 0u},
        {0u, 2u, 17u},
        {1u, 0u, 101u},
        {1u, 2u, HIDDEN - 1u},
    };
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_bhwc_index(GRID_W, samples[i].y, samples[i].x, samples[i].channel);
        const float residual1 = f16_bits_to_f32(f32_to_f16_bits(f16_bits_to_f32(input[idx]) +
                                                                f16_bits_to_f32(proj_bias[samples[i].channel])));
        const float expected = residual1 + f16_bits_to_f32(lin2_bias[samples[i].channel]);
        CHECK(fabsf(out_f32[idx] - expected) <= 1.0e-7f);
    }

    block.rel_pos_h_length = UOCR_SAM_WINDOW_REL_POS_SIZE;
    block.rel_pos_w_length = UOCR_SAM_WINDOW_REL_POS_SIZE;
    CHECK(uocr_metal_context_sam_transformer_block_f16(ctx,
                                                       input,
                                                       &block,
                                                       GRID_W,
                                                       GRID_H,
                                                       0,
                                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                                       out_f16,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_bhwc_index(GRID_W, samples[i].y, samples[i].x, samples[i].channel);
        const float residual1 = f16_bits_to_f32(f32_to_f16_bits(f16_bits_to_f32(input[idx]) +
                                                                f16_bits_to_f32(proj_bias[samples[i].channel])));
        const float expected = residual1 + f16_bits_to_f32(lin2_bias[samples[i].channel]);
        CHECK(fabsf(f16_bits_to_f32(out_f16[idx]) - expected) <= 6.5e-4f);
    }

    memset(proj_bias, 0, HIDDEN * sizeof(uint16_t));
    memset(lin2_bias, 0, HIDDEN * sizeof(uint16_t));
    block.rel_pos_h_length = UOCR_SAM_MAX_REL_POS_SIZE;
    block.rel_pos_w_length = UOCR_SAM_MAX_REL_POS_SIZE;
    uocr_metal_sam_transformer_block_f16 blocks[UOCR_SAM_BLOCKS];
    for (uint32_t i = 0u; i < UOCR_SAM_BLOCKS; ++i) {
        blocks[i] = block;
    }
    CHECK(uocr_metal_context_sam_transformer_f16(ctx,
                                                 input,
                                                 blocks,
                                                 UOCR_SAM_BLOCKS,
                                                 GRID_W,
                                                 GRID_H,
                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                 stack_out_f16,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (size_t i = 0u; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        const size_t idx = sam_bhwc_index(GRID_W, samples[i].y, samples[i].x, samples[i].channel);
        CHECK(stack_out_f16[idx] == input[idx]);
    }

    CHECK(uocr_metal_context_sam_transformer_block_f16(ctx,
                                                       input,
                                                       NULL,
                                                       GRID_W,
                                                       GRID_H,
                                                       1,
                                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                                       out_f16,
                                                       error,
                                                       sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM transformer block") != NULL);

    CHECK(uocr_metal_context_sam_transformer_f16(ctx,
                                                 input,
                                                 blocks,
                                                 UOCR_SAM_BLOCKS - 1u,
                                                 GRID_W,
                                                 GRID_H,
                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                 stack_out_f16,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal SAM transformer block count") != NULL);

    uocr_metal_context_destroy(ctx);
    free(stack_out_f16);
    free(out_f16);
    free(out_f32);
    free(lin2_bias);
    free(lin2_weight);
    free(lin1_bias);
    free(lin1_weight);
    free(norm2_bias);
    free(norm2_weight);
    free(rel_w);
    free(rel_h);
    free(proj_bias);
    free(proj_weight);
    free(qkv_bias);
    free(qkv_weight);
    free(norm1_bias);
    free(norm1_weight);
    free(input);
    return 0;
}

static int env_flag_enabled(const char *name) {
    const char *value = getenv(name);
    return value != NULL && (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0 ||
                             strcmp(value, "yes") == 0 || strcmp(value, "YES") == 0);
}

static uint32_t env_u32_or_default(const char *name, uint32_t default_value) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return default_value;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > (unsigned long)UINT32_MAX) {
        return default_value;
    }
    return (uint32_t)parsed;
}

static uint16_t bf16_bits_to_f16_bits(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16u;
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return f32_to_f16_bits(value);
}

static int read_safetensors_u64_le(FILE *f, uint64_t *out) {
    unsigned char bytes[8];
    if (fread(bytes, 1u, sizeof(bytes), f) != sizeof(bytes)) {
        return 0;
    }
    uint64_t value = 0u;
    for (uint32_t i = 0u; i < 8u; ++i) {
        value |= ((uint64_t)bytes[i]) << (8u * i);
    }
    *out = value;
    return 1;
}

static const char *parse_u64_after(const char *cursor, uint64_t *out) {
    if (cursor == NULL || out == NULL) {
        return NULL;
    }
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
        ++cursor;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long long value = strtoull(cursor, &end, 10);
    if (errno != 0 || end == cursor) {
        return NULL;
    }
    *out = (uint64_t)value;
    return end;
}

static int find_safetensors_tensor_range(const char *header,
                                         const char *tensor_name,
                                         uint64_t *out_start,
                                         uint64_t *out_end) {
    if (header == NULL || tensor_name == NULL || out_start == NULL || out_end == NULL) {
        return 0;
    }

    char key[256];
    if (snprintf(key, sizeof(key), "\"%s\"", tensor_name) >= (int)sizeof(key)) {
        return 0;
    }
    const char *entry = strstr(header, key);
    if (entry == NULL) {
        return 0;
    }
    const char *next_entry = strstr(entry + strlen(key), "},\"");
    const char *dtype = strstr(entry, "\"dtype\":\"BF16\"");
    const char *shape = strstr(entry, "\"shape\":[129280,1280]");
    const char *offsets = strstr(entry, "\"data_offsets\":[");
    if (dtype == NULL || shape == NULL || offsets == NULL ||
        (next_entry != NULL && (dtype > next_entry || shape > next_entry || offsets > next_entry))) {
        return 0;
    }
    offsets += strlen("\"data_offsets\":[");
    uint64_t start = 0u;
    uint64_t end = 0u;
    const char *cursor = parse_u64_after(offsets, &start);
    if (cursor == NULL || *cursor != ',') {
        return 0;
    }
    cursor = parse_u64_after(cursor + 1, &end);
    if (cursor == NULL || *cursor != ']' || end <= start) {
        return 0;
    }
    *out_start = start;
    *out_end = end;
    return 1;
}

static int make_safetensors_path(const char *hf_dir, char *out, size_t out_size) {
    static const char *candidates[] = {"model-00001-of-000001.safetensors", "model.safetensors"};
    if (hf_dir == NULL || hf_dir[0] == '\0' || out == NULL || out_size == 0u) {
        return 0;
    }
    for (uint32_t i = 0u; i < (uint32_t)(sizeof(candidates) / sizeof(candidates[0])); ++i) {
        if (snprintf(out, out_size, "%s/%s", hf_dir, candidates[i]) >= (int)out_size) {
            return 0;
        }
        if (access(out, R_OK) == 0) {
            return 1;
        }
    }
    return 0;
}

static int read_hf_embed_rows_as_f16(const char *hf_dir,
                                     const int32_t *token_ids,
                                     uint32_t n_tokens,
                                     uint16_t *out_f16,
                                     char *error,
                                     size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
    if (hf_dir == NULL || token_ids == NULL || out_f16 == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid HF embedding row request");
        return 0;
    }

    char path[4096];
    if (!make_safetensors_path(hf_dir, path, sizeof(path))) {
        snprintf(error, error_size, "no safetensors payload found under %s", hf_dir);
        return 0;
    }

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(error, error_size, "failed to open %s", path);
        return 0;
    }

    uint64_t header_len = 0u;
    if (!read_safetensors_u64_le(f, &header_len) || header_len == 0u || header_len > 256ull * 1024ull * 1024ull) {
        snprintf(error, error_size, "invalid safetensors header length in %s", path);
        fclose(f);
        return 0;
    }
    char *header = (char *)malloc((size_t)header_len + 1u);
    if (header == NULL) {
        snprintf(error, error_size, "failed to allocate safetensors header buffer");
        fclose(f);
        return 0;
    }
    if (fread(header, 1u, (size_t)header_len, f) != (size_t)header_len) {
        snprintf(error, error_size, "failed to read safetensors header from %s", path);
        free(header);
        fclose(f);
        return 0;
    }
    header[header_len] = '\0';

    uint64_t tensor_start = 0u;
    uint64_t tensor_end = 0u;
    if (!find_safetensors_tensor_range(header, "model.embed_tokens.weight", &tensor_start, &tensor_end)) {
        snprintf(error, error_size, "failed to locate model.embed_tokens.weight in %s", path);
        free(header);
        fclose(f);
        return 0;
    }
    free(header);

    const uint64_t data_start = 8u + header_len;
    const uint64_t row_bytes = (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t expected_tensor_bytes = (uint64_t)UOCR_VOCAB_SIZE * row_bytes;
    if (tensor_end - tensor_start != expected_tensor_bytes) {
        snprintf(error,
                 error_size,
                 "unexpected token embedding byte size %llu in %s",
                 (unsigned long long)(tensor_end - tensor_start),
                 path);
        fclose(f);
        return 0;
    }

    unsigned char row[UOCR_HIDDEN_SIZE * 2u];
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        if (token_ids[token] < 0 || (uint32_t)token_ids[token] >= UOCR_VOCAB_SIZE) {
            snprintf(error, error_size, "token id %d out of vocab", token_ids[token]);
            fclose(f);
            return 0;
        }
        const uint64_t row_offset = data_start + tensor_start + (uint64_t)(uint32_t)token_ids[token] * row_bytes;
        if (row_offset > (uint64_t)LONG_MAX || fseek(f, (long)row_offset, SEEK_SET) != 0) {
            snprintf(error, error_size, "failed to seek token embedding row %d", token_ids[token]);
            fclose(f);
            return 0;
        }
        if (fread(row, 1u, sizeof(row), f) != sizeof(row)) {
            snprintf(error, error_size, "failed to read token embedding row %d", token_ids[token]);
            fclose(f);
            return 0;
        }
        for (uint32_t col = 0u; col < (uint32_t)UOCR_HIDDEN_SIZE; ++col) {
            const uint16_t bf16 = (uint16_t)row[2u * col] | (uint16_t)((uint16_t)row[2u * col + 1u] << 8u);
            out_f16[token * (uint32_t)UOCR_HIDDEN_SIZE + col] = bf16_bits_to_f16_bits(bf16);
        }
    }

    fclose(f);
    return 1;
}

static int make_fixture_path(const char *root, const char *name, char *out, size_t out_size) {
    if (root == NULL || root[0] == '\0' || name == NULL || out == NULL || out_size == 0u) {
        return 0;
    }
    const int written = snprintf(out, out_size, "%s/%s", root, name);
    return written >= 0 && (size_t)written < out_size;
}

static int fixture_binary_exists(const char *root, const char *name) {
    char path[4096];
    if (!make_fixture_path(root, name, path, sizeof(path))) {
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    fclose(f);
    return 1;
}

static int read_binary_file(const char *path, unsigned char **out_bytes, size_t *out_size, char *error, size_t error_size) {
    if (out_bytes == NULL || out_size == NULL) {
        snprintf(error, error_size, "invalid binary read request");
        return 0;
    }
    *out_bytes = NULL;
    *out_size = 0u;
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(error, error_size, "failed to open %s", path);
        return 0;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        snprintf(error, error_size, "failed to seek %s", path);
        fclose(f);
        return 0;
    }
    const long size_long = ftell(f);
    if (size_long <= 0) {
        snprintf(error, error_size, "invalid empty binary file %s", path);
        fclose(f);
        return 0;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        snprintf(error, error_size, "failed to rewind %s", path);
        fclose(f);
        return 0;
    }
    unsigned char *bytes = (unsigned char *)malloc((size_t)size_long);
    if (bytes == NULL) {
        snprintf(error, error_size, "failed to allocate %ld bytes for %s", size_long, path);
        fclose(f);
        return 0;
    }
    if (fread(bytes, 1u, (size_t)size_long, f) != (size_t)size_long) {
        snprintf(error, error_size, "failed to read %s", path);
        free(bytes);
        fclose(f);
        return 0;
    }
    fclose(f);
    *out_bytes = bytes;
    *out_size = (size_t)size_long;
    return 1;
}

static int read_text_file_allow_empty(const char *path, char **out_text, size_t *out_size, char *error, size_t error_size) {
    if (out_text == NULL || out_size == NULL) {
        snprintf(error, error_size, "invalid text read request");
        return 0;
    }
    *out_text = NULL;
    *out_size = 0u;
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(error, error_size, "failed to open %s", path);
        return 0;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        snprintf(error, error_size, "failed to seek %s", path);
        fclose(f);
        return 0;
    }
    const long size_long = ftell(f);
    if (size_long < 0) {
        snprintf(error, error_size, "failed to tell %s", path);
        fclose(f);
        return 0;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        snprintf(error, error_size, "failed to rewind %s", path);
        fclose(f);
        return 0;
    }
    if ((uint64_t)size_long > (uint64_t)SIZE_MAX - 1u) {
        snprintf(error, error_size, "text file %s is too large", path);
        fclose(f);
        return 0;
    }
    const size_t alloc_size = (size_t)size_long + 1u;
    char *text = (char *)malloc(alloc_size);
    if (text == NULL) {
        snprintf(error, error_size, "failed to allocate %zu bytes for %s", alloc_size, path);
        fclose(f);
        return 0;
    }
    if (size_long > 0 && fread(text, 1u, (size_t)size_long, f) != (size_t)size_long) {
        snprintf(error, error_size, "failed to read %s", path);
        free(text);
        fclose(f);
        return 0;
    }
    fclose(f);
    text[(size_t)size_long] = '\0';
    *out_text = text;
    *out_size = (size_t)size_long;
    return 1;
}

static int read_generated_text_python_dump(const char *dump_dir, char **text_out, char *error, size_t error_size) {
    if (text_out == NULL) {
        snprintf(error, error_size, "invalid generated-text dump request");
        return 0;
    }
    *text_out = NULL;
    char path[4096];
    if (!make_fixture_path(dump_dir, "generated_text.txt", path, sizeof(path))) {
        snprintf(error, error_size, "generated-text dump path is too long");
        return 0;
    }
    size_t text_size = 0u;
    return read_text_file_allow_empty(path, text_out, &text_size, error, error_size);
}

static int read_prepared_tokens_python_dump(const char *dump_dir,
                                             int32_t **input_ids_out,
                                             uint8_t **image_mask_out,
                                             uint32_t *n_tokens_out,
                                             char *error,
                                             size_t error_size) {
    if (input_ids_out == NULL || image_mask_out == NULL || n_tokens_out == NULL) {
        snprintf(error, error_size, "invalid prepared-token dump request");
        return 0;
    }
    *input_ids_out = NULL;
    *image_mask_out = NULL;
    *n_tokens_out = 0u;

    char ids_path[4096];
    char mask_path[4096];
    if (!make_fixture_path(dump_dir, "input_ids_i32.bin", ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, "image_mask_u8.bin", mask_path, sizeof(mask_path))) {
        snprintf(error, error_size, "prepared-token dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    size_t ids_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size)) {
        return 0;
    }
    if (ids_size == 0u || ids_size % sizeof(int32_t) != 0u || ids_size / sizeof(int32_t) > (size_t)UINT32_MAX) {
        snprintf(error, error_size, "invalid input_ids_i32.bin size %zu", ids_size);
        free(ids_bytes);
        return 0;
    }
    const uint32_t n_tokens = (uint32_t)(ids_size / sizeof(int32_t));

    unsigned char *mask_bytes = NULL;
    size_t mask_size = 0u;
    if (!read_binary_file(mask_path, &mask_bytes, &mask_size, error, error_size)) {
        free(ids_bytes);
        return 0;
    }
    if (mask_size != (size_t)n_tokens) {
        snprintf(error,
                 error_size,
                 "image_mask_u8.bin size mismatch: expected %u bytes, got %zu",
                 n_tokens,
                 mask_size);
        free(mask_bytes);
        free(ids_bytes);
        return 0;
    }

    int32_t *ids = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    uint8_t *mask = (uint8_t *)malloc((size_t)n_tokens * sizeof(uint8_t));
    if (ids == NULL || mask == NULL) {
        snprintf(error, error_size, "failed to allocate prepared-token dump buffers");
        free(mask);
        free(ids);
        free(mask_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < n_tokens; ++i) {
        const unsigned char *p = ids_bytes + 4u * i;
        const uint32_t raw = (uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) |
                             ((uint32_t)p[3] << 24u);
        memcpy(&ids[i], &raw, sizeof(ids[i]));
        if (ids[i] < 0 || (uint32_t)ids[i] >= UOCR_VOCAB_SIZE) {
            snprintf(error, error_size, "input_ids_i32.bin token %u is out of vocabulary: %d", i, ids[i]);
            free(mask);
            free(ids);
            free(mask_bytes);
            free(ids_bytes);
            return 0;
        }
        if (mask_bytes[i] > 1u) {
            snprintf(error, error_size, "image_mask_u8.bin has invalid value %u at token %u", mask_bytes[i], i);
            free(mask);
            free(ids);
            free(mask_bytes);
            free(ids_bytes);
            return 0;
        }
        mask[i] = (uint8_t)mask_bytes[i];
    }

    free(mask_bytes);
    free(ids_bytes);
    *input_ids_out = ids;
    *image_mask_out = mask;
    *n_tokens_out = n_tokens;
    return 1;
}

static const char *json_find_key_range(const char *start, const char *end, const char *key) {
    if (start == NULL || end == NULL || key == NULL || end < start) {
        return NULL;
    }
    char pattern[128];
    const int written = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    const size_t pattern_len = (size_t)written;
    for (const char *p = start; p + pattern_len <= end; ++p) {
        if (memcmp(p, pattern, pattern_len) == 0) {
            return p;
        }
    }
    return NULL;
}

static const char *json_find_char_range(const char *start, const char *end, char ch) {
    if (start == NULL || end == NULL || end < start) {
        return NULL;
    }
    for (const char *p = start; p < end; ++p) {
        if (*p == ch) {
            return p;
        }
    }
    return NULL;
}

static const char *json_skip_ws_range(const char *p, const char *end) {
    while (p != NULL && p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
        ++p;
    }
    return p;
}

static int json_parse_u32_key_range(const char *start, const char *end, const char *key, uint32_t *out) {
    const char *p = json_find_key_range(start, end, key);
    if (p == NULL) {
        return 0;
    }
    p = json_find_char_range(p, end, ':');
    if (p == NULL) {
        return 0;
    }
    p = json_skip_ws_range(p + 1, end);
    if (p == NULL || p >= end) {
        return 0;
    }
    errno = 0;
    char *after = NULL;
    const unsigned long value = strtoul(p, &after, 10);
    if (after == p || errno != 0 || value > (unsigned long)UINT32_MAX) {
        return 0;
    }
    *out = (uint32_t)value;
    return 1;
}

static int json_parse_string_key_range(const char *start,
                                       const char *end,
                                       const char *key,
                                       char *out,
                                       size_t out_size) {
    if (out == NULL || out_size == 0u) {
        return 0;
    }
    const char *p = json_find_key_range(start, end, key);
    if (p == NULL) {
        return 0;
    }
    p = json_find_char_range(p, end, ':');
    if (p == NULL) {
        return 0;
    }
    p = json_skip_ws_range(p + 1, end);
    if (p == NULL || p >= end || *p != '"') {
        return 0;
    }
    const char *value_start = p + 1;
    const char *value_end = json_find_char_range(value_start, end, '"');
    if (value_end == NULL || value_end < value_start) {
        return 0;
    }
    const size_t len = (size_t)(value_end - value_start);
    if (len >= out_size) {
        return 0;
    }
    memcpy(out, value_start, len);
    out[len] = '\0';
    return 1;
}

static int read_prepared_view_stubs_from_manifest(const char *dump_dir,
                                                  uocr_image_view **views_out,
                                                  uint32_t *n_views_out,
                                                  uint32_t *crop_grid_w_out,
                                                  uint32_t *crop_grid_h_out,
                                                  char *error,
                                                  size_t error_size) {
    if (views_out == NULL || n_views_out == NULL || crop_grid_w_out == NULL || crop_grid_h_out == NULL) {
        snprintf(error, error_size, "invalid fixture view-manifest request");
        return 0;
    }
    *views_out = NULL;
    *n_views_out = 0u;
    *crop_grid_w_out = 0u;
    *crop_grid_h_out = 0u;

    char manifest_path[4096];
    if (!make_fixture_path(dump_dir, "manifest.json", manifest_path, sizeof(manifest_path))) {
        snprintf(error, error_size, "fixture manifest path is too long");
        return 0;
    }
    unsigned char *manifest_bytes = NULL;
    size_t manifest_size = 0u;
    if (!read_binary_file(manifest_path, &manifest_bytes, &manifest_size, error, error_size)) {
        return 0;
    }
    char *json = (char *)malloc(manifest_size + 1u);
    if (json == NULL) {
        snprintf(error, error_size, "failed to allocate fixture manifest buffer");
        free(manifest_bytes);
        return 0;
    }
    memcpy(json, manifest_bytes, manifest_size);
    json[manifest_size] = '\0';
    free(manifest_bytes);

    const char *json_start = json;
    const char *json_end = json + manifest_size;
    uint32_t crop_grid_w = 0u;
    uint32_t crop_grid_h = 0u;
    if (!json_parse_u32_key_range(json_start, json_end, "crop_grid_w", &crop_grid_w) ||
        !json_parse_u32_key_range(json_start, json_end, "crop_grid_h", &crop_grid_h)) {
        snprintf(error, error_size, "fixture manifest is missing crop grid metadata");
        free(json);
        return 0;
    }

    const char *views_key = json_find_key_range(json_start, json_end, "views");
    if (views_key == NULL) {
        snprintf(error, error_size, "fixture manifest is missing views metadata");
        free(json);
        return 0;
    }
    const char *array_start = json_find_char_range(views_key, json_end, '[');
    if (array_start == NULL) {
        snprintf(error, error_size, "fixture manifest views metadata is invalid");
        free(json);
        return 0;
    }
    const char *array_end = NULL;
    int bracket_depth = 0;
    for (const char *p = array_start; p < json_end; ++p) {
        if (*p == '[') {
            ++bracket_depth;
        } else if (*p == ']') {
            --bracket_depth;
            if (bracket_depth == 0) {
                array_end = p;
                break;
            }
        }
    }
    if (array_end == NULL) {
        snprintf(error, error_size, "fixture manifest views array is unterminated");
        free(json);
        return 0;
    }

    uint32_t n_views = 0u;
    for (const char *p = array_start + 1; p < array_end; ++p) {
        if (*p == '{') {
            ++n_views;
            const char *object_end = json_find_char_range(p + 1, array_end, '}');
            if (object_end == NULL) {
                snprintf(error, error_size, "fixture manifest view object is unterminated");
                free(json);
                return 0;
            }
            p = object_end;
        }
    }

    uocr_image_view *views = NULL;
    if (n_views != 0u) {
        views = (uocr_image_view *)calloc((size_t)n_views, sizeof(*views));
        if (views == NULL) {
            snprintf(error, error_size, "failed to allocate fixture view stubs");
            free(json);
            return 0;
        }
    }

    static const uint16_t pixel_sentinel = 0u;
    uint32_t view_index = 0u;
    for (const char *p = array_start + 1; p < array_end;) {
        if (*p != '{') {
            ++p;
            continue;
        }
        const char *object_start = p;
        const char *object_end = json_find_char_range(object_start + 1, array_end, '}');
        if (object_end == NULL || view_index >= n_views) {
            snprintf(error, error_size, "fixture manifest view object parse failure");
            free(views);
            free(json);
            return 0;
        }

        char kind[32];
        char format[32];
        uint32_t width = 0u;
        uint32_t height = 0u;
        if (!json_parse_string_key_range(object_start, object_end, "kind", kind, sizeof(kind)) ||
            !json_parse_string_key_range(object_start, object_end, "format", format, sizeof(format)) ||
            !json_parse_u32_key_range(object_start, object_end, "width", &width) ||
            !json_parse_u32_key_range(object_start, object_end, "height", &height)) {
            snprintf(error, error_size, "fixture manifest view %u is missing kind/format/shape", view_index);
            free(views);
            free(json);
            return 0;
        }

        views[view_index].pixels = &pixel_sentinel;
        views[view_index].width = width;
        views[view_index].height = height;
        if (strcmp(kind, "global") == 0) {
            views[view_index].kind = UOCR_VIEW_GLOBAL;
        } else if (strcmp(kind, "local") == 0) {
            views[view_index].kind = UOCR_VIEW_LOCAL;
        } else {
            snprintf(error, error_size, "fixture manifest view %u has unsupported kind %s", view_index, kind);
            free(views);
            free(json);
            return 0;
        }
        if (strcmp(format, "f16_nchw") == 0) {
            views[view_index].format = UOCR_PIXEL_F16_NCHW;
        } else if (strcmp(format, "f32_nchw") == 0) {
            views[view_index].format = UOCR_PIXEL_F32_NCHW;
        } else {
            snprintf(error, error_size, "fixture manifest view %u has unsupported format %s", view_index, format);
            free(views);
            free(json);
            return 0;
        }

        ++view_index;
        p = object_end + 1;
    }
    if (view_index != n_views) {
        snprintf(error, error_size, "fixture manifest view count changed while parsing");
        free(views);
        free(json);
        return 0;
    }

    free(json);
    *views_out = views;
    *n_views_out = n_views;
    *crop_grid_w_out = crop_grid_w;
    *crop_grid_h_out = crop_grid_h;
    return 1;
}

static int read_prompt_embedding_python_dump(const char *dump_dir,
                                             int32_t **input_ids_out,
                                             uint32_t *n_tokens_out,
                                             uint16_t **embeddings_out,
                                             char *error,
                                             size_t error_size) {
    if (input_ids_out == NULL || n_tokens_out == NULL || embeddings_out == NULL) {
        snprintf(error, error_size, "invalid prompt embedding dump request");
        return 0;
    }
    *input_ids_out = NULL;
    *n_tokens_out = 0u;
    *embeddings_out = NULL;

    char ids_path[4096];
    char embeddings_path[4096];
    if (!make_fixture_path(dump_dir, "input_ids_i32.bin", ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, "prompt_embeddings_f16.bin", embeddings_path, sizeof(embeddings_path))) {
        snprintf(error, error_size, "prompt embedding dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    size_t ids_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size)) {
        return 0;
    }
    if (ids_size % sizeof(int32_t) != 0u || ids_size / sizeof(int32_t) > (size_t)UINT32_MAX) {
        snprintf(error, error_size, "invalid input_ids_i32.bin size %zu", ids_size);
        free(ids_bytes);
        return 0;
    }
    const uint32_t n_tokens = (uint32_t)(ids_size / sizeof(int32_t));
    if (n_tokens == 0u) {
        snprintf(error, error_size, "prompt embedding dump has no tokens");
        free(ids_bytes);
        return 0;
    }

    uint64_t expected_embedding_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_embedding_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "prompt embedding dump size overflow");
        free(ids_bytes);
        return 0;
    }

    unsigned char *embedding_bytes = NULL;
    size_t embedding_size = 0u;
    if (!read_binary_file(embeddings_path, &embedding_bytes, &embedding_size, error, error_size)) {
        free(ids_bytes);
        return 0;
    }
    if ((uint64_t)embedding_size != expected_embedding_bytes) {
        snprintf(error,
                 error_size,
                 "prompt_embeddings_f16.bin size mismatch: expected %llu bytes, got %zu",
                 (unsigned long long)expected_embedding_bytes,
                 embedding_size);
        free(embedding_bytes);
        free(ids_bytes);
        return 0;
    }

    int32_t *ids = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    uint16_t *embeddings = (uint16_t *)malloc((size_t)expected_embedding_bytes);
    if (ids == NULL || embeddings == NULL) {
        snprintf(error, error_size, "failed to allocate prompt embedding dump buffers");
        free(embeddings);
        free(ids);
        free(embedding_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < n_tokens; ++i) {
        const unsigned char *p = ids_bytes + 4u * i;
        const uint32_t raw = (uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
        memcpy(&ids[i], &raw, sizeof(ids[i]));
    }
    const size_t embedding_values = embedding_size / sizeof(uint16_t);
    for (size_t i = 0u; i < embedding_values; ++i) {
        embeddings[i] = (uint16_t)embedding_bytes[2u * i] | (uint16_t)((uint16_t)embedding_bytes[2u * i + 1u] << 8u);
    }

    free(embedding_bytes);
    free(ids_bytes);
    *input_ids_out = ids;
    *n_tokens_out = n_tokens;
    *embeddings_out = embeddings;
    return 1;
}

static int read_text_layer0_python_dump(const char *dump_dir,
                                        int32_t **input_ids_out,
                                        uint32_t *n_tokens_out,
                                        uint16_t **prompt_embeddings_out,
                                        uint16_t **layer0_hidden_out,
                                        char *error,
                                        size_t error_size) {
    if (layer0_hidden_out == NULL) {
        snprintf(error, error_size, "invalid text layer0 dump request");
        return 0;
    }
    *layer0_hidden_out = NULL;
    if (!read_prompt_embedding_python_dump(dump_dir,
                                           input_ids_out,
                                           n_tokens_out,
                                           prompt_embeddings_out,
                                           error,
                                           error_size)) {
        return 0;
    }

    uint64_t expected_layer_bytes = (uint64_t)(*n_tokens_out) * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_layer_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "text layer0 dump size overflow");
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }

    char layer_path[4096];
    if (!make_fixture_path(dump_dir, "layer_0_hidden_f16.bin", layer_path, sizeof(layer_path))) {
        snprintf(error, error_size, "text layer0 dump path is too long");
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }

    unsigned char *layer_bytes = NULL;
    size_t layer_size = 0u;
    if (!read_binary_file(layer_path, &layer_bytes, &layer_size, error, error_size)) {
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }
    if ((uint64_t)layer_size != expected_layer_bytes) {
        snprintf(error,
                 error_size,
                 "layer_0_hidden_f16.bin size mismatch: expected %llu bytes, got %zu",
                 (unsigned long long)expected_layer_bytes,
                 layer_size);
        free(layer_bytes);
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }

    uint16_t *layer0 = (uint16_t *)malloc((size_t)expected_layer_bytes);
    if (layer0 == NULL) {
        snprintf(error, error_size, "failed to allocate text layer0 dump buffer");
        free(layer_bytes);
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }
    const size_t layer_values = layer_size / sizeof(uint16_t);
    for (size_t i = 0u; i < layer_values; ++i) {
        layer0[i] = (uint16_t)layer_bytes[2u * i] | (uint16_t)((uint16_t)layer_bytes[2u * i + 1u] << 8u);
    }
    free(layer_bytes);
    *layer0_hidden_out = layer0;
    return 1;
}

static int read_image_prompt_embedding_python_dump(const char *dump_dir,
                                                   int32_t **input_ids_out,
                                                   uint8_t **image_mask_out,
                                                   uint32_t *n_tokens_out,
                                                   uint16_t **visual_features_out,
                                                   uint32_t *image_span_start_out,
                                                   uint32_t *image_span_length_out,
                                                   uint16_t **prompt_embeddings_out,
                                                   char *error,
                                                   size_t error_size) {
    if (input_ids_out == NULL || image_mask_out == NULL || n_tokens_out == NULL || visual_features_out == NULL ||
        image_span_start_out == NULL || image_span_length_out == NULL || prompt_embeddings_out == NULL) {
        snprintf(error, error_size, "invalid image prompt embedding dump request");
        return 0;
    }
    *input_ids_out = NULL;
    *image_mask_out = NULL;
    *n_tokens_out = 0u;
    *visual_features_out = NULL;
    *image_span_start_out = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    *image_span_length_out = 0u;
    *prompt_embeddings_out = NULL;

    if (!read_prompt_embedding_python_dump(dump_dir,
                                           input_ids_out,
                                           n_tokens_out,
                                           prompt_embeddings_out,
                                           error,
                                           error_size)) {
        return 0;
    }

    char mask_path[4096];
    if (!make_fixture_path(dump_dir, "image_mask_u8.bin", mask_path, sizeof(mask_path))) {
        snprintf(error, error_size, "image prompt dump path is too long");
        goto fail;
    }
    unsigned char *mask_bytes = NULL;
    size_t mask_size = 0u;
    if (!read_binary_file(mask_path, &mask_bytes, &mask_size, error, error_size)) {
        goto fail;
    }
    if (mask_size != (size_t)(*n_tokens_out)) {
        snprintf(error,
                 error_size,
                 "image_mask_u8.bin size mismatch: expected %u bytes, got %zu",
                 *n_tokens_out,
                 mask_size);
        free(mask_bytes);
        goto fail;
    }
    uint32_t image_count = 0u;
    for (uint32_t i = 0u; i < *n_tokens_out; ++i) {
        if (mask_bytes[i] > 1u) {
            snprintf(error, error_size, "image_mask_u8.bin has invalid value %u at token %u", mask_bytes[i], i);
            free(mask_bytes);
            goto fail;
        }
        image_count += (uint32_t)mask_bytes[i];
    }
    if (image_count == 0u) {
        snprintf(error, error_size, "image prompt dump has no image placeholders");
        free(mask_bytes);
        goto fail;
    }

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = *input_ids_out;
    request.image_mask = mask_bytes;
    request.n_tokens = *n_tokens_out;
    request.max_new_tokens = 1u;
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    if (uocr_build_sequence_state(&request, &state, error, error_size) != UOCR_OK ||
        state.image_span_start == UOCR_SEQUENCE_NO_IMAGE_SPAN || state.image_span_length == 0u) {
        free(mask_bytes);
        goto fail;
    }
    if (state.image_span_length != image_count) {
        snprintf(error,
                 error_size,
                 "image placeholder count mismatch: span=%u count=%u",
                 state.image_span_length,
                 image_count);
        free(mask_bytes);
        goto fail;
    }

    if ((uint64_t)state.image_span_length > UINT64_MAX / (uint64_t)UOCR_HIDDEN_SIZE ||
        (uint64_t)state.image_span_length * (uint64_t)UOCR_HIDDEN_SIZE > UINT64_MAX / 2u) {
        snprintf(error, error_size, "image visual feature dump size overflow");
        free(mask_bytes);
        goto fail;
    }
    const uint64_t expected_visual_bytes =
        (uint64_t)state.image_span_length * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_visual_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "image visual feature dump size overflow");
        free(mask_bytes);
        goto fail;
    }

    char visual_path[4096];
    if (!make_fixture_path(dump_dir, "visual_features_f16.bin", visual_path, sizeof(visual_path))) {
        snprintf(error, error_size, "visual feature dump path is too long");
        free(mask_bytes);
        goto fail;
    }
    unsigned char *visual_bytes = NULL;
    size_t visual_size = 0u;
    if (!read_binary_file(visual_path, &visual_bytes, &visual_size, error, error_size)) {
        free(mask_bytes);
        goto fail;
    }
    if ((uint64_t)visual_size != expected_visual_bytes) {
        snprintf(error,
                 error_size,
                 "visual_features_f16.bin size mismatch: expected %llu bytes, got %zu",
                 (unsigned long long)expected_visual_bytes,
                 visual_size);
        free(visual_bytes);
        free(mask_bytes);
        goto fail;
    }
    uint16_t *visual = (uint16_t *)malloc((size_t)expected_visual_bytes);
    if (visual == NULL) {
        snprintf(error, error_size, "failed to allocate visual feature dump buffer");
        free(visual_bytes);
        free(mask_bytes);
        goto fail;
    }
    const size_t visual_values = visual_size / sizeof(uint16_t);
    for (size_t i = 0u; i < visual_values; ++i) {
        visual[i] = (uint16_t)visual_bytes[2u * i] | (uint16_t)((uint16_t)visual_bytes[2u * i + 1u] << 8u);
    }
    free(visual_bytes);

    *image_mask_out = mask_bytes;
    *visual_features_out = visual;
    *image_span_start_out = state.image_span_start;
    *image_span_length_out = state.image_span_length;
    return 1;

fail:
    free(*prompt_embeddings_out);
    free(*input_ids_out);
    *prompt_embeddings_out = NULL;
    *input_ids_out = NULL;
    *n_tokens_out = 0u;
    return 0;
}

static const uint16_t *model_tensor_f16_payload(const uocr_model_file *model,
                                                uint32_t tensor_id,
                                                uint64_t expected_values,
                                                char *error,
                                                size_t error_size) {
    if (model == NULL) {
        snprintf(error, error_size, "invalid model tensor lookup");
        return NULL;
    }
    const uocr_tensor_entry *tensor = uocr_model_file_find_tensor(model, tensor_id);
    if (tensor == NULL) {
        snprintf(error, error_size, "model tensor %u is missing", tensor_id);
        return NULL;
    }
    uint64_t expected_bytes = 0u;
    if (expected_values > UINT64_MAX / 2u) {
        snprintf(error, error_size, "model tensor %u byte-size overflow", tensor_id);
        return NULL;
    }
    expected_bytes = expected_values * 2u;
    if (tensor->qtype != UOCR_TENSOR_F16 || tensor->payload_size != expected_bytes ||
        tensor->payload_offset > (uint64_t)model->size || tensor->payload_size > (uint64_t)model->size - tensor->payload_offset) {
        snprintf(error,
                 error_size,
                 "model tensor %u is not the expected fp16 payload: qtype=%u bytes=%llu expected=%llu",
                 tensor_id,
                 tensor->qtype,
                 (unsigned long long)tensor->payload_size,
                 (unsigned long long)expected_bytes);
        return NULL;
    }
    return (const uint16_t *)(const void *)(model->data + tensor->payload_offset);
}

static int read_layer_hidden_python_dump(const char *dump_dir,
                                         const char *filename,
                                         uint32_t n_tokens,
                                         uint16_t **hidden_out,
                                         char *error,
                                         size_t error_size) {
    if (hidden_out == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid layer hidden dump request");
        return 0;
    }
    *hidden_out = NULL;
    uint64_t expected_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "layer hidden dump size overflow");
        return 0;
    }

    char path[4096];
    if (!make_fixture_path(dump_dir, filename, path, sizeof(path))) {
        snprintf(error, error_size, "layer hidden dump path is too long");
        return 0;
    }

    unsigned char *bytes = NULL;
    size_t size = 0u;
    if (!read_binary_file(path, &bytes, &size, error, error_size)) {
        return 0;
    }
    if ((uint64_t)size != expected_bytes) {
        snprintf(error,
                 error_size,
                 "%s size mismatch: expected %llu bytes, got %zu",
                 filename,
                 (unsigned long long)expected_bytes,
                 size);
        free(bytes);
        return 0;
    }

    uint16_t *hidden = (uint16_t *)malloc((size_t)expected_bytes);
    if (hidden == NULL) {
        snprintf(error, error_size, "failed to allocate layer hidden dump buffer");
        free(bytes);
        return 0;
    }
    const size_t values = size / sizeof(uint16_t);
    for (size_t i = 0u; i < values; ++i) {
        hidden[i] = (uint16_t)bytes[2u * i] | (uint16_t)((uint16_t)bytes[2u * i + 1u] << 8u);
    }
    free(bytes);
    *hidden_out = hidden;
    return 1;
}

static int run_metal_attention_block_from_model_f16(uocr_metal_context *ctx,
                                                    const uocr_model_file *model,
                                                    uint32_t layer,
                                                    const uint16_t *hidden_in,
                                                    uint32_t n_tokens,
                                                    uint16_t *attn_hidden_out,
                                                    uint16_t *mlp_input_out,
                                                    char *error,
                                                    size_t error_size) {
    if (ctx == NULL || model == NULL || hidden_in == NULL || attn_hidden_out == NULL || mlp_input_out == NULL ||
        n_tokens == 0u || layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid Metal attention-block parity request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal attention-block parity size overflow");
        return 0;
    }

    const uint16_t *input_norm_weight = model_tensor_f16_payload(model,
                                                                 uocr_tensor_id_layer_input_norm(layer),
                                                                 UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 error_size);
    const uint16_t *q_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_Q),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *k_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_K),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *v_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_V),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *o_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_O),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *post_norm_weight = model_tensor_f16_payload(model,
                                                                uocr_tensor_id_layer_post_attn_norm(layer),
                                                                UOCR_HIDDEN_SIZE,
                                                                error,
                                                                error_size);
    if (input_norm_weight == NULL || q_weight == NULL || k_weight == NULL || v_weight == NULL || o_weight == NULL ||
        post_norm_weight == NULL) {
        return 0;
    }

    uint16_t *normed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *v = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *unused_o = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *context = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    if (normed == NULL || q == NULL || k == NULL || v == NULL || unused_o == NULL || q_rope == NULL ||
        k_rope == NULL || context == NULL) {
        snprintf(error, error_size, "failed to allocate Metal attention-block parity buffers");
        free(context);
        free(k_rope);
        free(q_rope);
        free(unused_o);
        free(v);
        free(k);
        free(q);
        free(normed);
        return 0;
    }

    int ok = 0;
    if (uocr_metal_context_rmsnorm_f16(ctx,
                                       hidden_in,
                                       input_norm_weight,
                                       n_tokens,
                                       UOCR_HIDDEN_SIZE,
                                       UOCR_RMS_NORM_EPS,
                                       UOCR_METAL_RMSNORM_OUTPUT_F16,
                                       normed,
                                       error,
                                       error_size) != 1 ||
        uocr_metal_context_attention_qkvo_f16(ctx,
                                              normed,
                                              q_weight,
                                              k_weight,
                                              v_weight,
                                              o_weight,
                                              n_tokens,
                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                              q,
                                              k,
                                              v,
                                              unused_o,
                                              error,
                                              error_size) != 1 ||
        uocr_metal_context_rope_qk_f16(ctx,
                                       q,
                                       k,
                                       n_tokens,
                                       0u,
                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                       q_rope,
                                       k_rope,
                                       error,
                                       error_size) != 1 ||
        uocr_metal_context_prefill_attention_f16(ctx,
                                                 q_rope,
                                                 k_rope,
                                                 v,
                                                 n_tokens,
                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                 context,
                                                 error,
                                                 error_size) != 1 ||
        uocr_metal_context_attention_output_residual_f16(ctx,
                                                         context,
                                                         o_weight,
                                                         hidden_in,
                                                         n_tokens,
                                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                                         attn_hidden_out,
                                                         error,
                                                         error_size) != 1 ||
        uocr_metal_context_rmsnorm_f16(ctx,
                                       attn_hidden_out,
                                       post_norm_weight,
                                       n_tokens,
                                       UOCR_HIDDEN_SIZE,
                                       UOCR_RMS_NORM_EPS,
                                       UOCR_METAL_RMSNORM_OUTPUT_F16,
                                       mlp_input_out,
                                       error,
                                       error_size) != 1) {
        ok = 0;
    } else {
        ok = 1;
    }

    free(context);
    free(k_rope);
    free(q_rope);
    free(unused_o);
    free(v);
    free(k);
    free(q);
    free(normed);
    return ok;
}

static int run_metal_dense_decoder_layer0_from_model_f16(uocr_metal_context *ctx,
                                                           const uocr_model_file *model,
                                                           const uint16_t *hidden_in,
                                                           uint32_t n_tokens,
                                                           uint16_t *hidden_out,
                                                           char *error,
                                                           size_t error_size) {
    if (ctx == NULL || model == NULL || hidden_in == NULL || hidden_out == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid Metal dense decoder-layer0 request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal dense decoder-layer0 size overflow");
        return 0;
    }

    const uint16_t *gate_weight = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           error_size);
    const uint16_t *up_weight = model_tensor_f16_payload(model,
                                                         uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         error_size);
    const uint16_t *down_weight = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE,
                                                           error,
                                                           error_size);
    if (gate_weight == NULL || up_weight == NULL || down_weight == NULL) {
        return 0;
    }

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    if (attn_hidden == NULL || mlp_input == NULL) {
        snprintf(error, error_size, "failed to allocate Metal dense decoder-layer0 buffers");
        free(mlp_input);
        free(attn_hidden);
        return 0;
    }

    int ok = 0;
    if (run_metal_attention_block_from_model_f16(ctx,
                                                model,
                                                0u,
                                                hidden_in,
                                                n_tokens,
                                                attn_hidden,
                                                mlp_input,
                                                error,
                                                error_size) != 1 ||
        uocr_metal_context_dense_swiglu_f16(ctx,
                                            mlp_input,
                                            gate_weight,
                                            up_weight,
                                            down_weight,
                                            attn_hidden,
                                            n_tokens,
                                            UOCR_METAL_DENSE_OUTPUT_F16,
                                            hidden_out,
                                            error,
                                            error_size) != 1) {
        ok = 0;
    } else {
        ok = 1;
    }

    free(mlp_input);
    free(attn_hidden);
    return ok;
}

static int copy_selected_moe_expert_weights(const uocr_model_file *model,
                                            uint32_t layer,
                                            const uint32_t *top_expert_ids,
                                            uint16_t *selected_gate,
                                            uint16_t *selected_up,
                                            uint16_t *selected_down,
                                            char *error,
                                            size_t error_size) {
    if (model == NULL || top_expert_ids == NULL || selected_gate == NULL || selected_up == NULL || selected_down == NULL ||
        layer == 0u || layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid selected MoE expert copy request");
        return 0;
    }
    const uint64_t expert_values = (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * (uint64_t)UOCR_HIDDEN_SIZE;
    if (expert_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "selected MoE expert copy size overflow");
        return 0;
    }
    const size_t expert_bytes = (size_t)expert_values * sizeof(uint16_t);
    for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
        const uint32_t expert = top_expert_ids[rank];
        if (expert >= UOCR_ROUTED_EXPERTS) {
            snprintf(error, error_size, "selected MoE expert id %u out of range", expert);
            return 0;
        }
        const uint16_t *gate = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_GATE),
                                                        expert_values,
                                                        error,
                                                        error_size);
        const uint16_t *up = model_tensor_f16_payload(model,
                                                      uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_UP),
                                                      expert_values,
                                                      error,
                                                      error_size);
        const uint16_t *down = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_DOWN),
                                                        expert_values,
                                                        error,
                                                        error_size);
        if (gate == NULL || up == NULL || down == NULL) {
            return 0;
        }
        memcpy(selected_gate + (size_t)rank * (size_t)expert_values, gate, expert_bytes);
        memcpy(selected_up + (size_t)rank * (size_t)expert_values, up, expert_bytes);
        memcpy(selected_down + (size_t)rank * (size_t)expert_values, down, expert_bytes);
    }
    return 1;
}

static int read_text_logits_topk_python_dump(const char *dump_dir,
                                              int32_t **ids_out,
                                              float **scores_out,
                                              uint32_t *top_k_out,
                                              char *error,
                                              size_t error_size) {
    if (ids_out == NULL || scores_out == NULL || top_k_out == NULL) {
        snprintf(error, error_size, "invalid logits top-k dump request");
        return 0;
    }
    *ids_out = NULL;
    *scores_out = NULL;
    *top_k_out = 0u;

    char ids_path[4096];
    char scores_path[4096];
    if (!make_fixture_path(dump_dir, "logits_topk_ids_i32.bin", ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, "logits_topk_scores_f32.bin", scores_path, sizeof(scores_path))) {
        snprintf(error, error_size, "logits top-k dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    unsigned char *scores_bytes = NULL;
    size_t ids_size = 0u;
    size_t scores_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size) ||
        !read_binary_file(scores_path, &scores_bytes, &scores_size, error, error_size)) {
        free(scores_bytes);
        free(ids_bytes);
        return 0;
    }
    if (ids_size == 0u || ids_size % sizeof(int32_t) != 0u || scores_size != ids_size) {
        snprintf(error,
                 error_size,
                 "logits top-k size mismatch: ids=%zu bytes scores=%zu bytes",
                 ids_size,
                 scores_size);
        free(scores_bytes);
        free(ids_bytes);
        return 0;
    }
    const uint32_t top_k = (uint32_t)(ids_size / sizeof(int32_t));
    int32_t *ids = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    float *scores = (float *)malloc((size_t)top_k * sizeof(float));
    if (ids == NULL || scores == NULL) {
        snprintf(error, error_size, "failed to allocate logits top-k buffers");
        free(scores);
        free(ids);
        free(scores_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < top_k; ++i) {
        const uint32_t id_bits = (uint32_t)ids_bytes[4u * i] | ((uint32_t)ids_bytes[4u * i + 1u] << 8u) |
                                 ((uint32_t)ids_bytes[4u * i + 2u] << 16u) |
                                 ((uint32_t)ids_bytes[4u * i + 3u] << 24u);
        const uint32_t score_bits = (uint32_t)scores_bytes[4u * i] | ((uint32_t)scores_bytes[4u * i + 1u] << 8u) |
                                    ((uint32_t)scores_bytes[4u * i + 2u] << 16u) |
                                    ((uint32_t)scores_bytes[4u * i + 3u] << 24u);
        ids[i] = (int32_t)id_bits;
        memcpy(scores + i, &score_bits, sizeof(float));
        if (ids[i] < 0 || ids[i] >= (int32_t)UOCR_VOCAB_SIZE || !isfinite(scores[i])) {
            snprintf(error, error_size, "invalid logits top-k entry at rank %u", i);
            free(scores);
            free(ids);
            free(scores_bytes);
            free(ids_bytes);
            return 0;
        }
    }
    free(scores_bytes);
    free(ids_bytes);
    *ids_out = ids;
    *scores_out = scores;
    *top_k_out = top_k;
    return 1;
}

static int read_text_generated_ids_python_dump(const char *dump_dir,
                                               int32_t **ids_out,
                                               uint32_t *n_ids_out,
                                               char *error,
                                               size_t error_size) {
    if (ids_out == NULL || n_ids_out == NULL) {
        snprintf(error, error_size, "invalid generated-id dump request");
        return 0;
    }
    *ids_out = NULL;
    *n_ids_out = 0u;

    char path[4096];
    if (!make_fixture_path(dump_dir, "generated_ids_i32.bin", path, sizeof(path))) {
        snprintf(error, error_size, "generated-id dump path is too long");
        return 0;
    }
    unsigned char *bytes = NULL;
    size_t size = 0u;
    if (!read_binary_file(path, &bytes, &size, error, error_size)) {
        return 0;
    }
    if (size == 0u || size % sizeof(int32_t) != 0u) {
        snprintf(error, error_size, "generated-id dump size mismatch: %zu bytes", size);
        free(bytes);
        return 0;
    }
    const uint32_t n_ids = (uint32_t)(size / sizeof(int32_t));
    int32_t *ids = (int32_t *)malloc((size_t)n_ids * sizeof(int32_t));
    if (ids == NULL) {
        snprintf(error, error_size, "failed to allocate generated-id dump buffer");
        free(bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < n_ids; ++i) {
        const uint32_t id_bits = (uint32_t)bytes[4u * i] | ((uint32_t)bytes[4u * i + 1u] << 8u) |
                                 ((uint32_t)bytes[4u * i + 2u] << 16u) |
                                 ((uint32_t)bytes[4u * i + 3u] << 24u);
        ids[i] = (int32_t)id_bits;
        if (ids[i] < 0 || ids[i] >= (int32_t)UOCR_VOCAB_SIZE) {
            snprintf(error, error_size, "invalid generated token id at index %u", i);
            free(ids);
            free(bytes);
            return 0;
        }
    }
    free(bytes);
    *ids_out = ids;
    *n_ids_out = n_ids;
    return 1;
}

static int read_router_topk_python_dump(const char *dump_dir,
                                         uint32_t layer,
                                         uint32_t n_tokens,
                                         uint32_t **ids_out,
                                         float **weights_out,
                                         char *error,
                                         size_t error_size) {
    if (ids_out == NULL || weights_out == NULL || n_tokens == 0u || layer == 0u || layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid router top-k dump request");
        return 0;
    }
    *ids_out = NULL;
    *weights_out = NULL;

    char ids_name[64];
    char weights_name[64];
    snprintf(ids_name, sizeof(ids_name), "layer_%u_router_top_ids_u32.bin", layer);
    snprintf(weights_name, sizeof(weights_name), "layer_%u_router_top_weights_f32.bin", layer);
    char ids_path[4096];
    char weights_path[4096];
    if (!make_fixture_path(dump_dir, ids_name, ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, weights_name, weights_path, sizeof(weights_path))) {
        snprintf(error, error_size, "router top-k dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    unsigned char *weights_bytes = NULL;
    size_t ids_size = 0u;
    size_t weights_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size) ||
        !read_binary_file(weights_path, &weights_bytes, &weights_size, error, error_size)) {
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }

    uint64_t expected_values = 0u;
    if ((uint64_t)n_tokens > UINT64_MAX / (uint64_t)UOCR_MOE_TOP_K ||
        (expected_values = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K) > (uint64_t)SIZE_MAX / sizeof(uint32_t)) {
        snprintf(error, error_size, "router top-k dump size overflow");
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }
    const uint64_t expected_bytes = expected_values * sizeof(uint32_t);
    if ((uint64_t)ids_size != expected_bytes || (uint64_t)weights_size != expected_bytes) {
        snprintf(error,
                 error_size,
                 "router top-k dump size mismatch for layer %u: ids=%zu weights=%zu expected=%llu",
                 layer,
                 ids_size,
                 weights_size,
                 (unsigned long long)expected_bytes);
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }

    uint32_t *ids = (uint32_t *)malloc((size_t)expected_values * sizeof(uint32_t));
    float *weights = (float *)malloc((size_t)expected_values * sizeof(float));
    if (ids == NULL || weights == NULL) {
        snprintf(error, error_size, "failed to allocate router top-k dump buffers");
        free(weights);
        free(ids);
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint64_t i = 0u; i < expected_values; ++i) {
        const uint32_t id_bits = (uint32_t)ids_bytes[4u * i] | ((uint32_t)ids_bytes[4u * i + 1u] << 8u) |
                                 ((uint32_t)ids_bytes[4u * i + 2u] << 16u) |
                                 ((uint32_t)ids_bytes[4u * i + 3u] << 24u);
        const uint32_t weight_bits = (uint32_t)weights_bytes[4u * i] | ((uint32_t)weights_bytes[4u * i + 1u] << 8u) |
                                     ((uint32_t)weights_bytes[4u * i + 2u] << 16u) |
                                     ((uint32_t)weights_bytes[4u * i + 3u] << 24u);
        ids[i] = id_bits;
        memcpy(weights + i, &weight_bits, sizeof(float));
        if (ids[i] >= UOCR_ROUTED_EXPERTS || !isfinite(weights[i])) {
            snprintf(error, error_size, "invalid router top-k entry at flat index %llu", (unsigned long long)i);
            free(weights);
            free(ids);
            free(weights_bytes);
            free(ids_bytes);
            return 0;
        }
    }
    free(weights_bytes);
    free(ids_bytes);
    *ids_out = ids;
    *weights_out = weights;
    return 1;
}

static void compute_logits_topk_f32(const float *logits,
                                    uint32_t vocab_size,
                                    uint32_t top_k,
                                    int32_t *ids_out,
                                    float *scores_out) {
    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        uint32_t best_id = 0u;
        float best_score = -INFINITY;
        for (uint32_t token = 0u; token < vocab_size; ++token) {
            int already_selected = 0;
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if ((uint32_t)ids_out[prev] == token) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected || isnan(logits[token])) {
                continue;
            }
            if (logits[token] > best_score || (logits[token] == best_score && token < best_id)) {
                best_score = logits[token];
                best_id = token;
            }
        }
        ids_out[rank] = (int32_t)best_id;
        scores_out[rank] = best_score;
    }
}

static int compare_hidden_f16(const char *label,
                              const uint16_t *actual,
                              const uint16_t *expected,
                              uint32_t n_tokens,
                              float max_abs_tolerance,
                              double mean_abs_tolerance) {
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    uint64_t max_index = 0u;
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        const float diff = fabsf(f16_bits_to_f32(actual[i]) - f16_bits_to_f32(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            max_index = i;
        }
        sum_abs += (double)diff;
    }
    const double mean_abs = sum_abs / (double)hidden_values;
    if (max_abs > max_abs_tolerance || mean_abs > mean_abs_tolerance) {
        fprintf(stderr,
                "%s mismatch: max_abs=%g mean_abs=%g at token %llu col %llu actual=%g expected=%g\n",
                label,
                (double)max_abs,
                mean_abs,
                (unsigned long long)(max_index / (uint64_t)UOCR_HIDDEN_SIZE),
                (unsigned long long)(max_index % (uint64_t)UOCR_HIDDEN_SIZE),
                (double)f16_bits_to_f32(actual[max_index]),
                (double)f16_bits_to_f32(expected[max_index]));
        return 0;
    }
    return 1;
}

static int copy_compact_moe_expert_weights(const uocr_model_file *model,
                                           uint32_t layer,
                                           const uint32_t *top_expert_ids,
                                           uint32_t n_tokens,
                                           uint32_t *remapped_top_expert_ids,
                                           uint32_t *expert_count_out,
                                           uint16_t **gate_out,
                                           uint16_t **up_out,
                                           uint16_t **down_out,
                                           char *error,
                                           size_t error_size) {
    if (model == NULL || top_expert_ids == NULL || remapped_top_expert_ids == NULL || expert_count_out == NULL ||
        gate_out == NULL || up_out == NULL || down_out == NULL || n_tokens == 0u || layer == 0u ||
        layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid compact MoE expert copy request");
        return 0;
    }
    *expert_count_out = 0u;
    *gate_out = NULL;
    *up_out = NULL;
    *down_out = NULL;

    uint32_t experts[UOCR_ROUTED_EXPERTS];
    uint32_t expert_count = 0u;
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
            const size_t index = (size_t)token * UOCR_MOE_TOP_K + rank;
            const uint32_t expert = top_expert_ids[index];
            if (expert >= UOCR_ROUTED_EXPERTS) {
                snprintf(error, error_size, "compact MoE expert id %u out of range", expert);
                return 0;
            }
            uint32_t remapped = UINT32_MAX;
            for (uint32_t i = 0u; i < expert_count; ++i) {
                if (experts[i] == expert) {
                    remapped = i;
                    break;
                }
            }
            if (remapped == UINT32_MAX) {
                if (expert_count >= UOCR_ROUTED_EXPERTS) {
                    snprintf(error, error_size, "compact MoE expert table overflow");
                    return 0;
                }
                remapped = expert_count;
                experts[expert_count++] = expert;
            }
            remapped_top_expert_ids[index] = remapped;
        }
    }

    const uint64_t expert_values = (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * (uint64_t)UOCR_HIDDEN_SIZE;
    uint64_t total_values = 0u;
    if (expert_count == 0u || expert_values > UINT64_MAX / (uint64_t)expert_count ||
        (total_values = expert_values * (uint64_t)expert_count) > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "compact MoE expert copy size overflow");
        return 0;
    }
    const size_t expert_bytes = (size_t)expert_values * sizeof(uint16_t);
    const size_t total_bytes = (size_t)total_values * sizeof(uint16_t);
    uint16_t *gate = (uint16_t *)malloc(total_bytes);
    uint16_t *up = (uint16_t *)malloc(total_bytes);
    uint16_t *down = (uint16_t *)malloc(total_bytes);
    if (gate == NULL || up == NULL || down == NULL) {
        snprintf(error, error_size, "failed to allocate compact MoE expert weights");
        free(down);
        free(up);
        free(gate);
        return 0;
    }

    for (uint32_t compact = 0u; compact < expert_count; ++compact) {
        const uint32_t expert = experts[compact];
        const uint16_t *src_gate = model_tensor_f16_payload(model,
                                                            uocr_tensor_id_moe_expert(layer,
                                                                                      expert,
                                                                                      UOCR_TENSOR_PROJ_GATE),
                                                            expert_values,
                                                            error,
                                                            error_size);
        const uint16_t *src_up = model_tensor_f16_payload(model,
                                                          uocr_tensor_id_moe_expert(layer,
                                                                                    expert,
                                                                                    UOCR_TENSOR_PROJ_UP),
                                                          expert_values,
                                                          error,
                                                          error_size);
        const uint16_t *src_down = model_tensor_f16_payload(model,
                                                            uocr_tensor_id_moe_expert(layer,
                                                                                      expert,
                                                                                      UOCR_TENSOR_PROJ_DOWN),
                                                            expert_values,
                                                            error,
                                                            error_size);
        if (src_gate == NULL || src_up == NULL || src_down == NULL) {
            free(down);
            free(up);
            free(gate);
            return 0;
        }
        memcpy(gate + (size_t)compact * (size_t)expert_values, src_gate, expert_bytes);
        memcpy(up + (size_t)compact * (size_t)expert_values, src_up, expert_bytes);
        memcpy(down + (size_t)compact * (size_t)expert_values, src_down, expert_bytes);
    }

    *expert_count_out = expert_count;
    *gate_out = gate;
    *up_out = up;
    *down_out = down;
    return 1;
}

static int run_metal_moe_decoder_layer_from_model_f16(uocr_metal_context *ctx,
                                                      const uocr_model_file *model,
                                                      uint32_t layer,
                                                      const uint16_t *hidden_in,
                                                      uint32_t n_tokens,
                                                      uint16_t *hidden_out,
                                                      char *error,
                                                      size_t error_size) {
    if (ctx == NULL || model == NULL || hidden_in == NULL || hidden_out == NULL || n_tokens == 0u || layer == 0u ||
        layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid Metal MoE decoder-layer parity request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal MoE decoder-layer parity size overflow");
        return 0;
    }

    const uint16_t *router_weight = model_tensor_f16_payload(model,
                                                             uocr_tensor_id_moe_router(layer),
                                                             (uint64_t)UOCR_ROUTED_EXPERTS * (uint64_t)UOCR_HIDDEN_SIZE,
                                                             error,
                                                             error_size);
    const uint16_t *shared_gate = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           error_size);
    const uint16_t *shared_up = model_tensor_f16_payload(model,
                                                         uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         error_size);
    const uint16_t *shared_down = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE,
                                                           error,
                                                           error_size);
    if (router_weight == NULL || shared_gate == NULL || shared_up == NULL || shared_down == NULL) {
        return 0;
    }

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *routed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *shared = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint32_t *top_ids = (uint32_t *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(uint32_t));
    float *top_weights = (float *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(float));
    if (attn_hidden == NULL || mlp_input == NULL || routed == NULL || shared == NULL || top_ids == NULL ||
        top_weights == NULL) {
        snprintf(error, error_size, "failed to allocate Metal MoE decoder-layer parity buffers");
        free(top_weights);
        free(top_ids);
        free(shared);
        free(routed);
        free(mlp_input);
        free(attn_hidden);
        return 0;
    }

    int ok = 0;
    if (run_metal_attention_block_from_model_f16(ctx,
                                                model,
                                                layer,
                                                hidden_in,
                                                n_tokens,
                                                attn_hidden,
                                                mlp_input,
                                                error,
                                                error_size) != 1 ||
        uocr_metal_context_moe_router_f16(ctx,
                                          mlp_input,
                                          router_weight,
                                          n_tokens,
                                          NULL,
                                          NULL,
                                          top_ids,
                                          top_weights,
                                          error,
                                          error_size) != 1 ||
        uocr_metal_context_moe_shared_experts_f16(ctx,
                                                  mlp_input,
                                                  shared_gate,
                                                  shared_up,
                                                  shared_down,
                                                  n_tokens,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  shared,
                                                  error,
                                                  error_size) != 1) {
        goto cleanup;
    }

    uint32_t unique_experts[UOCR_ROUTED_EXPERTS];
    uint32_t unique_count = 0u;
    for (uint32_t i = 0u; i < n_tokens * UOCR_MOE_TOP_K; ++i) {
        uint32_t seen = 0u;
        for (uint32_t j = 0u; j < unique_count; ++j) {
            if (unique_experts[j] == top_ids[i]) {
                seen = 1u;
                break;
            }
        }
        if (!seen && unique_count < UOCR_ROUTED_EXPERTS) {
            unique_experts[unique_count++] = top_ids[i];
        }
    }
    const uint64_t compact_weight_bytes = (uint64_t)unique_count * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE *
                                          (uint64_t)UOCR_HIDDEN_SIZE * 2u * 3u;
    if (compact_weight_bytes <= 192ull * 1024ull * 1024ull) {
        uint32_t *remapped_ids = (uint32_t *)malloc((size_t)n_tokens * UOCR_MOE_TOP_K * sizeof(uint32_t));
        uint16_t *compact_gate = NULL;
        uint16_t *compact_up = NULL;
        uint16_t *compact_down = NULL;
        uint32_t compact_count = 0u;
        if (remapped_ids == NULL) {
            snprintf(error, error_size, "failed to allocate compact MoE remap buffer");
            goto cleanup;
        }
        if (copy_compact_moe_expert_weights(model,
                                            layer,
                                            top_ids,
                                            n_tokens,
                                            remapped_ids,
                                            &compact_count,
                                            &compact_gate,
                                            &compact_up,
                                            &compact_down,
                                            error,
                                            error_size) != 1 ||
            uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                                mlp_input,
                                                                remapped_ids,
                                                                top_weights,
                                                                compact_gate,
                                                                compact_up,
                                                                compact_down,
                                                                n_tokens,
                                                                UOCR_HIDDEN_SIZE,
                                                                UOCR_MOE_EXPERT_INTERMEDIATE,
                                                                compact_count,
                                                                UOCR_MOE_TOP_K,
                                                                UOCR_METAL_DENSE_OUTPUT_F16,
                                                                routed,
                                                                error,
                                                                error_size) != 1) {
            free(compact_down);
            free(compact_up);
            free(compact_gate);
            free(remapped_ids);
            goto cleanup;
        }
        free(compact_down);
        free(compact_up);
        free(compact_gate);
        free(remapped_ids);
    } else {
        const uint64_t expert_values = (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE *
                                       (uint64_t)UOCR_HIDDEN_SIZE;
        if (expert_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
            snprintf(error, error_size, "selected MoE fallback size overflow");
            goto cleanup;
        }
        uint16_t *selected_gate = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
        uint16_t *selected_up = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
        uint16_t *selected_down = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
        if (selected_gate == NULL || selected_up == NULL || selected_down == NULL) {
            snprintf(error, error_size, "failed to allocate selected MoE fallback weights");
            free(selected_down);
            free(selected_up);
            free(selected_gate);
            goto cleanup;
        }
        for (uint32_t token = 0u; token < n_tokens; ++token) {
            if (copy_selected_moe_expert_weights(model,
                                                 layer,
                                                 top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                                 selected_gate,
                                                 selected_up,
                                                 selected_down,
                                                 error,
                                                 error_size) != 1 ||
                uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                                   mlp_input + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                   top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                                                   top_weights + (size_t)token * UOCR_MOE_TOP_K,
                                                                   selected_gate,
                                                                   selected_up,
                                                                   selected_down,
                                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                                   routed + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                   error,
                                                                   error_size) != 1) {
                free(selected_down);
                free(selected_up);
                free(selected_gate);
                goto cleanup;
            }
        }
        free(selected_down);
        free(selected_up);
        free(selected_gate);
    }

    if (uocr_metal_context_moe_combine_f16(ctx,
                                           routed,
                                           shared,
                                           attn_hidden,
                                           n_tokens,
                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                           hidden_out,
                                           error,
                                           error_size) != 1) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(top_weights);
    free(top_ids);
    free(shared);
    free(routed);
    free(mlp_input);
    free(attn_hidden);
    return ok;
}

static int run_metal_decoder_prefill_from_model_f16(uocr_metal_context *ctx,
                                                     const uocr_model_file *model,
                                                     const uint16_t *prompt_embeddings_f16,
                                                     uint32_t n_tokens,
                                                     uint16_t *final_hidden_out,
                                                     char *error,
                                                     size_t error_size) {
    if (ctx == NULL || model == NULL || prompt_embeddings_f16 == NULL || final_hidden_out == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid Metal decoder prefill request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal decoder prefill size overflow");
        return 0;
    }
    const size_t hidden_bytes = (size_t)hidden_values * sizeof(uint16_t);
    uint16_t *ping = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *pong = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    if (ping == NULL || pong == NULL) {
        snprintf(error, error_size, "failed to allocate Metal decoder prefill ping-pong buffers");
        free(pong);
        free(ping);
        return 0;
    }

    int ok = 0;
    if (run_metal_dense_decoder_layer0_from_model_f16(ctx,
                                                      model,
                                                      prompt_embeddings_f16,
                                                      n_tokens,
                                                      ping,
                                                      error,
                                                      error_size) != 1) {
        goto cleanup;
    }

    uint16_t *current = ping;
    uint16_t *next = pong;
    for (uint32_t layer = 1u; layer < UOCR_DECODER_LAYERS; ++layer) {
        if (run_metal_moe_decoder_layer_from_model_f16(ctx,
                                                       model,
                                                       layer,
                                                       current,
                                                       n_tokens,
                                                       next,
                                                       error,
                                                       error_size) != 1) {
            goto cleanup;
        }
        uint16_t *tmp = current;
        current = next;
        next = tmp;
    }
    memcpy(final_hidden_out, current, hidden_bytes);
    ok = 1;

cleanup:
    free(pong);
    free(ping);
    return ok;
}

static int test_metal_text_prompt_embedding_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_PROMPT_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_PROMPT_DUMP_DIR are not both set; skipping Python prompt dump parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *expected = NULL;
    CHECK(read_prompt_embedding_python_dump(dump_dir, &input_ids, &n_tokens, &expected, error, sizeof(error)) == 1);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint64_t output_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    CHECK(output_bytes <= (uint64_t)SIZE_MAX);
    uint16_t *actual = (uint16_t *)malloc((size_t)output_bytes);
    CHECK(actual != NULL);
    memset(actual, 0, (size_t)output_bytes);
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            actual,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    const uint64_t values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    for (uint64_t i = 0u; i < values; ++i) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "Python prompt dump mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual[i],
                    expected[i]);
            free(actual);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected);
            free(input_ids);
            return 1;
        }
    }

    free(actual);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(input_ids);
    return 0;
}

static int test_metal_image_prompt_embedding_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image prompt embedding parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no visual_features_f16.bin; skipping image prompt embedding parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    const uint64_t output_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    CHECK(output_bytes <= (uint64_t)SIZE_MAX);
    uint16_t *actual = (uint16_t *)malloc((size_t)output_bytes);
    CHECK(actual != NULL);
    memset(actual, 0, (size_t)output_bytes);

    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            image_span_start,
                                                            image_span_length,
                                                            visual_features,
                                                            actual,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    const uint64_t values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    for (uint64_t i = 0u; i < values; ++i) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "Python image prompt dump mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual[i],
                    expected[i]);
            free(actual);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    free(actual);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_prompt_decoder_smoke_python_dump(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image prompt decoder smoke\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no visual_features_f16.bin; skipping image prompt decoder smoke\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected_prompt = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected_prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image decoder smoke\n",
               n_tokens,
               n_tokens);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));
    uint16_t *prompt = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *final_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(prompt != NULL);
    CHECK(final_hidden != NULL);
    CHECK(normed != NULL);
    CHECK(logits != NULL);

    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            image_span_start,
                                                            image_span_length,
                                                            visual_features,
                                                            prompt,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(memcmp(prompt, expected_prompt, (size_t)hidden_values * sizeof(uint16_t)) == 0);

    CHECK(run_metal_decoder_prefill_from_model_f16(ctx,
                                                   &model,
                                                   prompt,
                                                   n_tokens,
                                                   final_hidden,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    uint32_t token_id = UINT32_MAX;
    float score = 0.0f;
    const uint16_t *last_hidden = final_hidden + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   last_hidden,
                                                   1u,
                                                   NULL,
                                                   normed,
                                                   logits,
                                                   &token_id,
                                                   &score,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(token_id < (uint32_t)UOCR_VOCAB_SIZE);
    CHECK(isfinite(score));

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = n_tokens;
    request.max_new_tokens = 1u;
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    int32_t generated[1] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)token_id, generated, 1u) == UOCR_OK);
    CHECK(generated[0] == (int32_t)token_id);
    CHECK(state.generated_count == 1u);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    free(logits);
    free(normed);
    free(final_hidden);
    free(prompt);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_decoder_layers_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image decoder layer parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no visual_features_f16.bin; skipping image decoder layer parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no layer_11_hidden_f16.bin; skipping image decoder layer parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected_prompt = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected_prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image decoder layer parity\n",
               n_tokens,
               n_tokens);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));
    const size_t hidden_bytes = (size_t)hidden_values * sizeof(uint16_t);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *actual_prompt = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(actual_prompt != NULL);
    CHECK(actual != NULL);

    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            image_span_start,
                                                            image_span_length,
                                                            visual_features,
                                                            actual_prompt,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        if (actual_prompt[i] != expected_prompt[i]) {
            fprintf(stderr,
                    "image decoder prompt mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual_prompt[i],
                    expected_prompt[i]);
            free(actual);
            free(actual_prompt);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    uint16_t *expected = NULL;
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_0_hidden_f16.bin",
                                        n_tokens,
                                        &expected,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(run_metal_dense_decoder_layer0_from_model_f16(ctx,
                                                        &model,
                                                        actual_prompt,
                                                        n_tokens,
                                                        actual,
                                                        error,
                                                        sizeof(error)) == 1);
    if (!compare_hidden_f16("image layer0", actual, expected, n_tokens, 7.5e-3f, 7.5e-4)) {
        free(expected);
        free(actual);
        free(actual_prompt);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 1;
    }
    free(expected);

    for (uint32_t layer = 1u; layer < UOCR_DECODER_LAYERS; ++layer) {
        char prev_filename[64];
        char expected_filename[64];
        char label[64];
        snprintf(prev_filename, sizeof(prev_filename), "layer_%u_hidden_f16.bin", layer - 1u);
        snprintf(expected_filename, sizeof(expected_filename), "layer_%u_hidden_f16.bin", layer);
        snprintf(label, sizeof(label), "image layer%u", layer);
        uint16_t *prev_expected = NULL;
        expected = NULL;
        CHECK(read_layer_hidden_python_dump(dump_dir, prev_filename, n_tokens, &prev_expected, error, sizeof(error)) == 1);
        CHECK(read_layer_hidden_python_dump(dump_dir, expected_filename, n_tokens, &expected, error, sizeof(error)) == 1);
        memset(actual, 0, hidden_bytes);
        CHECK(run_metal_moe_decoder_layer_from_model_f16(ctx,
                                                         &model,
                                                         layer,
                                                         prev_expected,
                                                         n_tokens,
                                                         actual,
                                                         error,
                                                         sizeof(error)) == 1);
        const int close = compare_hidden_f16(label, actual, expected, n_tokens, 1.25e-2f, 1.25e-3);
        free(expected);
        free(prev_expected);
        if (!close) {
            free(actual);
            free(actual_prompt);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }
    CHECK(error[0] == '\0');

    free(actual);
    free(actual_prompt);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_router_topk_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image router top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_1_router_top_ids_u32.bin") ||
        !fixture_binary_exists(dump_dir, "layer_1_router_top_weights_f32.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no router top-k files; skipping image router top-k parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    (void)image_span_start;
    (void)image_span_length;
    (void)visual_features;
    (void)image_mask;
    (void)input_ids;

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image router top-k parity\n",
               n_tokens,
               n_tokens);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint32_t *actual_ids = (uint32_t *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(uint32_t));
    float *actual_weights = (float *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(float));
    CHECK(attn_hidden != NULL);
    CHECK(mlp_input != NULL);
    CHECK(actual_ids != NULL);
    CHECK(actual_weights != NULL);

    for (uint32_t layer = 1u; layer < UOCR_DECODER_LAYERS; ++layer) {
        char prev_filename[64];
        snprintf(prev_filename, sizeof(prev_filename), "layer_%u_hidden_f16.bin", layer - 1u);
        uint16_t *prev_hidden = NULL;
        CHECK(read_layer_hidden_python_dump(dump_dir, prev_filename, n_tokens, &prev_hidden, error, sizeof(error)) == 1);
        uint32_t *expected_ids = NULL;
        float *expected_weights = NULL;
        CHECK(read_router_topk_python_dump(dump_dir,
                                           layer,
                                           n_tokens,
                                           &expected_ids,
                                           &expected_weights,
                                           error,
                                           sizeof(error)) == 1);
        const uint16_t *router_weight = model_tensor_f16_payload(&model,
                                                                 uocr_tensor_id_moe_router(layer),
                                                                 (uint64_t)UOCR_ROUTED_EXPERTS *
                                                                     (uint64_t)UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 sizeof(error));
        CHECK(router_weight != NULL);
        memset(attn_hidden, 0, (size_t)hidden_values * sizeof(uint16_t));
        memset(mlp_input, 0, (size_t)hidden_values * sizeof(uint16_t));
        memset(actual_ids, 0, (size_t)n_tokens * UOCR_MOE_TOP_K * sizeof(uint32_t));
        memset(actual_weights, 0, (size_t)n_tokens * UOCR_MOE_TOP_K * sizeof(float));
        CHECK(run_metal_attention_block_from_model_f16(ctx,
                                                       &model,
                                                       layer,
                                                       prev_hidden,
                                                       n_tokens,
                                                       attn_hidden,
                                                       mlp_input,
                                                       error,
                                                       sizeof(error)) == 1);
        CHECK(uocr_metal_context_moe_router_f16(ctx,
                                                mlp_input,
                                                router_weight,
                                                n_tokens,
                                                NULL,
                                                NULL,
                                                actual_ids,
                                                actual_weights,
                                                error,
                                                sizeof(error)) == 1);
        const uint64_t values = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K;
        for (uint64_t i = 0u; i < values; ++i) {
            if (actual_ids[i] != expected_ids[i]) {
                fprintf(stderr,
                        "image router top-k id mismatch layer %u token %llu rank %llu: actual=%u expected=%u\n",
                        layer,
                        (unsigned long long)(i / (uint64_t)UOCR_MOE_TOP_K),
                        (unsigned long long)(i % (uint64_t)UOCR_MOE_TOP_K),
                        actual_ids[i],
                        expected_ids[i]);
                free(expected_weights);
                free(expected_ids);
                free(prev_hidden);
                free(actual_weights);
                free(actual_ids);
                free(mlp_input);
                free(attn_hidden);
                uocr_metal_context_destroy(ctx);
                uocr_model_file_close(&model);
                free(prompt);
                free(visual_features);
                free(image_mask);
                free(input_ids);
                return 1;
            }
            const float diff = fabsf(actual_weights[i] - expected_weights[i]);
            if (diff > 5.0e-5f) {
                fprintf(stderr,
                        "image router top-k weight mismatch layer %u token %llu rank %llu expert %u: actual=%g expected=%g diff=%g\n",
                        layer,
                        (unsigned long long)(i / (uint64_t)UOCR_MOE_TOP_K),
                        (unsigned long long)(i % (uint64_t)UOCR_MOE_TOP_K),
                        actual_ids[i],
                        (double)actual_weights[i],
                        (double)expected_weights[i],
                        (double)diff);
                free(expected_weights);
                free(expected_ids);
                free(prev_hidden);
                free(actual_weights);
                free(actual_ids);
                free(mlp_input);
                free(attn_hidden);
                uocr_metal_context_destroy(ctx);
                uocr_model_file_close(&model);
                free(prompt);
                free(visual_features);
                free(image_mask);
                free(input_ids);
                return 1;
            }
        }
        free(expected_weights);
        free(expected_ids);
        free(prev_hidden);
    }
    CHECK(error[0] == '\0');

    free(actual_weights);
    free(actual_ids);
    free(mlp_input);
    free(attn_hidden);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_logits_topk_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image logits top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "logits_topk_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "logits_topk_scores_f32.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no logits_topk files; skipping image logits top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no layer_11_hidden_f16.bin; skipping image logits top-k parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    float *expected_scores = NULL;
    uint32_t top_k = 0u;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    (void)image_span_start;
    (void)image_span_length;

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image logits top-k parity\n",
               n_tokens,
               n_tokens);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_logits_topk_python_dump(dump_dir,
                                            &expected_ids,
                                            &expected_scores,
                                            &top_k,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(top_k > 0u && top_k <= 128u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    int32_t *actual_ids = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    float *actual_scores = (float *)malloc((size_t)top_k * sizeof(float));
    CHECK(normed != NULL);
    CHECK(logits != NULL);
    CHECK(actual_ids != NULL);
    CHECK(actual_scores != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               last_hidden,
                                               1u,
                                               UOCR_METAL_RMSNORM_OUTPUT_F16,
                                               normed,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(uocr_metal_context_lm_head_f16(ctx, normed, 1u, logits, error, sizeof(error)) == 1);
    compute_logits_topk_f32(logits, UOCR_VOCAB_SIZE, top_k, actual_ids, actual_scores);
    CHECK(error[0] == '\0');

    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        if (actual_ids[rank] != expected_ids[rank]) {
            fprintf(stderr,
                    "image logits top-k id mismatch at rank %u: actual=%d expected=%d actual_score=%g expected_score=%g\n",
                    rank,
                    actual_ids[rank],
                    expected_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank]);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
        const float diff = fabsf(actual_scores[rank] - expected_scores[rank]);
        if (diff > 2.5e-2f) {
            fprintf(stderr,
                    "image logits top-k score mismatch at rank %u token %d: actual=%g expected=%g diff=%g\n",
                    rank,
                    actual_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank],
                    (double)diff);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    free(actual_scores);
    free(actual_ids);
    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_scores);
    free(expected_ids);
    free(layer11);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image generated-id/text parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "generated_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "generated_text.txt")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no generated id/text files; skipping image generated-id/text parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no layer_11_hidden_f16.bin; skipping image generated-id/text parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    uint32_t n_generated = 0u;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    (void)image_span_start;
    (void)image_span_length;

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image generated-id/text parity\n",
               n_tokens,
               n_tokens);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_generated_ids_python_dump(dump_dir,
                                              &expected_ids,
                                              &n_generated,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(n_generated == 1u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    uint32_t actual_token = UINT32_MAX;
    float actual_score = 0.0f;
    CHECK(normed != NULL);
    CHECK(logits != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   last_hidden,
                                                   1u,
                                                   NULL,
                                                   normed,
                                                   logits,
                                                   &actual_token,
                                                   &actual_score,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    if ((int32_t)actual_token != expected_ids[0]) {
        fprintf(stderr,
                "image generated-id mismatch: actual=%u expected=%d score=%g\n",
                actual_token,
                expected_ids[0],
                (double)actual_score);
        free(logits);
        free(normed);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_ids);
        free(layer11);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 1;
    }

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = n_tokens;
    request.max_new_tokens = n_generated;
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    int32_t generated[1] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)actual_token, generated, 1u) == UOCR_OK);
    CHECK(generated[0] == expected_ids[0]);
    CHECK(state.generated_count == 1u);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_ids);
    free(layer11);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_integrated_image_embedding_python_dump_prefill(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping integrated image-embedding prefill parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin") ||
        !fixture_binary_exists(dump_dir, "prompt_embeddings_f16.bin") ||
        !fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no integrated image-embedding fixture files; skipping integrated image-embedding prefill parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected_prompt = NULL;
    uint16_t *expected_final_hidden = NULL;
    uocr_image_view *views = NULL;
    uint32_t n_views = 0u;
    uint32_t crop_grid_w = 0u;
    uint32_t crop_grid_h = 0u;

    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected_prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(read_prepared_view_stubs_from_manifest(dump_dir,
                                                 &views,
                                                 &n_views,
                                                 &crop_grid_w,
                                                 &crop_grid_h,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &expected_final_hidden,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    uocr_prepared_request prepared;
    memset(&prepared, 0, sizeof(prepared));
    prepared.input_ids = input_ids;
    prepared.image_mask = image_mask;
    prepared.n_tokens = n_tokens;
    prepared.views = views;
    prepared.n_views = n_views;
    prepared.crop_grid_w = crop_grid_w;
    prepared.crop_grid_h = crop_grid_h;
    prepared.max_new_tokens = 0u;
    uocr_request_limits limits;
    memset(&limits, 0, sizeof(limits));
    limits.max_prompt_tokens = n_tokens;
    limits.max_gen_tokens = 1u;
    CHECK(uocr_validate_prepared_request(&prepared, &limits, error, sizeof(error)) == UOCR_OK);
    CHECK(uocr_count_image_placeholders(&prepared) == image_span_length);

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run integrated image-embedding prefill parity\n",
               n_tokens,
               n_tokens);
        free(expected_final_hidden);
        free(views);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, n_tokens, error, sizeof(error)) == 1);

    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.image_features_f16 = visual_features;
    request.n_tokens = n_tokens;
    request.max_new_tokens = 0u;
    request.slot = 0u;
    request.image_span_start = image_span_start;
    request.image_span_length = image_span_length;

    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count == 0u);

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / 2u);
    uint16_t *actual_prompt = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual_final_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(actual_prompt != NULL);
    CHECK(actual_final_hidden != NULL);
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx,
                                                   0u,
                                                   n_tokens,
                                                   actual_prompt,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(uocr_metal_context_read_decoder_final_hidden_f16(ctx,
                                                           0u,
                                                           n_tokens,
                                                           actual_final_hidden,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint64_t i = 0u; i < hidden_values; ++i) {
        if (actual_prompt[i] != expected_prompt[i]) {
            fprintf(stderr,
                    "integrated image prompt mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual_prompt[i],
                    expected_prompt[i]);
            free(actual_final_hidden);
            free(actual_prompt);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_final_hidden);
            free(views);
            free(expected_prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }
    if (!compare_hidden_f16("integrated image final hidden",
                            actual_final_hidden,
                            expected_final_hidden,
                            n_tokens,
                            1.25e-2f,
                            1.25e-3)) {
        free(actual_final_hidden);
        free(actual_prompt);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_final_hidden);
        free(views);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 1;
    }

    int32_t generated[2] = {0, 0};
    request.max_new_tokens = 2u;
    memset(&result, 0, sizeof(result));
    result.generated_ids = generated;
    result.generated_capacity = 2u;
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count >= 1u);
    CHECK(result.generated_count <= 2u);
    for (uint32_t i = 0u; i < result.generated_count; ++i) {
        CHECK(generated[i] >= 0);
        CHECK((uint32_t)generated[i] < UOCR_VOCAB_SIZE);
    }

    free(actual_final_hidden);
    free(actual_prompt);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_final_hidden);
    free(views);
    free(expected_prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_integrated_image_embedding_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping integrated image-embedding generated-id/text parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin") ||
        !fixture_binary_exists(dump_dir, "prompt_embeddings_f16.bin") ||
        !fixture_binary_exists(dump_dir, "generated_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "generated_text.txt")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no integrated image-embedding generated id/text fixture files; skipping integrated image-embedding generated-id/text parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int exit_code = 1;
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    uocr_image_view *views = NULL;
    uint32_t n_views = 0u;
    uint32_t crop_grid_w = 0u;
    uint32_t crop_grid_h = 0u;
    int32_t *expected_ids = NULL;
    uint32_t n_expected = 0u;
    char *expected_text = NULL;
    int32_t *generated = NULL;
    float *scores = NULL;
    uocr_model_file model;
    memset(&model, 0, sizeof(model));
    int model_open = 0;
    uocr_metal_context *ctx = NULL;

    if (read_image_prompt_embedding_python_dump(dump_dir,
                                                &input_ids,
                                                &image_mask,
                                                &n_tokens,
                                                &visual_features,
                                                &image_span_start,
                                                &image_span_length,
                                                &prompt,
                                                error,
                                                sizeof(error)) != 1 ||
        read_prepared_view_stubs_from_manifest(dump_dir,
                                               &views,
                                               &n_views,
                                               &crop_grid_w,
                                               &crop_grid_h,
                                               error,
                                               sizeof(error)) != 1 ||
        read_text_generated_ids_python_dump(dump_dir, &expected_ids, &n_expected, error, sizeof(error)) != 1 ||
        read_generated_text_python_dump(dump_dir, &expected_text, error, sizeof(error)) != 1) {
        fprintf(stderr, "failed to read integrated image-embedding generated-id fixture: %s\n", error);
        goto cleanup;
    }
    if (n_tokens == 0u || image_span_start == UOCR_SEQUENCE_NO_IMAGE_SPAN || image_span_length == 0u ||
        n_expected == 0u) {
        fprintf(stderr,
                "invalid integrated image-embedding generated-id fixture: tokens=%u span_start=%u span_length=%u generated=%u\n",
                n_tokens,
                image_span_start,
                image_span_length,
                n_expected);
        goto cleanup;
    }

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run integrated image-embedding generated-id/text parity\n",
               n_tokens,
               n_tokens);
        exit_code = 0;
        goto cleanup;
    }

    uocr_prepared_request prepared;
    memset(&prepared, 0, sizeof(prepared));
    prepared.input_ids = input_ids;
    prepared.image_mask = image_mask;
    prepared.n_tokens = n_tokens;
    prepared.views = views;
    prepared.n_views = n_views;
    prepared.crop_grid_w = crop_grid_w;
    prepared.crop_grid_h = crop_grid_h;
    prepared.max_new_tokens = n_expected;
    prepared.no_repeat_ngram_size = 0u;
    prepared.no_repeat_window = 0u;
    uocr_request_limits limits;
    memset(&limits, 0, sizeof(limits));
    limits.max_prompt_tokens = n_tokens;
    limits.max_gen_tokens = n_expected;
    if (uocr_validate_prepared_request(&prepared, &limits, error, sizeof(error)) != UOCR_OK) {
        fprintf(stderr, "integrated image-embedding generated-id fixture validation failed: %s\n", error);
        goto cleanup;
    }
    if (uocr_count_image_placeholders(&prepared) != image_span_length) {
        fprintf(stderr,
                "integrated image-embedding generated-id fixture image-span mismatch: placeholders=%u span=%u\n",
                uocr_count_image_placeholders(&prepared),
                image_span_length);
        goto cleanup;
    }

    if (uocr_model_file_open(model_path, &model, error, sizeof(error)) != 0) {
        fprintf(stderr, "failed to open model for integrated image-embedding generated-id parity: %s\n", error);
        goto cleanup;
    }
    model_open = 1;
    ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    if (ctx == NULL) {
        fprintf(stderr, "failed to create Metal context for integrated image-embedding generated-id parity: %s\n", error);
        goto cleanup;
    }
    if (uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) != 1 ||
        uocr_metal_context_allocate_runtime_arenas(ctx, 1u, n_tokens, error, sizeof(error)) != 1) {
        fprintf(stderr, "failed to prepare Metal context for integrated image-embedding generated-id parity: %s\n", error);
        goto cleanup;
    }

    generated = (int32_t *)calloc((size_t)n_expected, sizeof(int32_t));
    scores = (float *)calloc((size_t)n_expected, sizeof(float));
    if (generated == NULL || scores == NULL) {
        fprintf(stderr, "failed to allocate integrated image-embedding generated-id output buffers\n");
        goto cleanup;
    }

    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.image_features_f16 = visual_features;
    request.n_tokens = n_tokens;
    request.max_new_tokens = n_expected;
    request.slot = 0u;
    request.image_span_start = image_span_start;
    request.image_span_length = image_span_length;
    request.no_repeat_ngram_size = 0u;
    request.no_repeat_window = 0u;

    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    result.generated_ids = generated;
    result.generated_scores_f32_or_null = scores;
    result.generated_capacity = n_expected;
    if (uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) != 1) {
        fprintf(stderr, "integrated image-embedding generated-id generation failed: %s\n", error);
        goto cleanup;
    }
    if (result.generated_count != n_expected) {
        fprintf(stderr,
                "integrated image-embedding generated-id count mismatch: actual=%u expected=%u decoded_text='%s'\n",
                result.generated_count,
                n_expected,
                expected_text != NULL ? expected_text : "");
        goto cleanup;
    }
    for (uint32_t i = 0u; i < n_expected; ++i) {
        if (generated[i] != expected_ids[i]) {
            fprintf(stderr,
                    "integrated image-embedding generated-id mismatch at %u: actual=%d expected=%d score=%g decoded_text='%s'\n",
                    i,
                    generated[i],
                    expected_ids[i],
                    (double)scores[i],
                    expected_text != NULL ? expected_text : "");
            goto cleanup;
        }
    }

    /* C does not own tokenizer decoding in v1. The fixture's generated_text.txt is
       loaded above, and exact generated-id parity makes that decoded text parity
       deterministic for the Python-owned tokenizer/frontend path. */
    exit_code = 0;

cleanup:
    if (ctx != NULL) {
        uocr_metal_context_destroy(ctx);
    }
    if (model_open) {
        uocr_model_file_close(&model);
    }
    free(scores);
    free(generated);
    free(expected_text);
    free(expected_ids);
    free(views);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return exit_code;
}

static int test_metal_text_prompt_embedding_full_model_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *hf_dir = getenv("UOCR_HF_DIR");
    if (model_path == NULL || model_path[0] == '\0' || hf_dir == NULL || hf_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_HF_DIR are not both set; skipping full prompt parity\n");
        return 0;
    }

    enum { TOKENS = 6 };
    const int32_t token_ids[TOKENS] = {0, 1, 2, 42, 1000, 128000};
    uint16_t expected[TOKENS * UOCR_HIDDEN_SIZE];
    char error[1024];
    memset(error, 0, sizeof(error));
    CHECK(read_hf_embed_rows_as_f16(hf_dir, token_ids, TOKENS, expected, error, sizeof(error)) == 1);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t actual[TOKENS * UOCR_HIDDEN_SIZE];
    memset(actual, 0, sizeof(actual));
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            token_ids,
                                                            TOKENS,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            actual,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * UOCR_HIDDEN_SIZE); ++i) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "prompt embedding mismatch at flat index %u: actual=0x%04x expected=0x%04x\n",
                    i,
                    actual[i],
                    expected[i]);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            return 1;
        }
    }

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    return 0;
}

static int test_metal_text_layer0_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text layer0 parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *expected = NULL;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &expected, error, sizeof(error)) == 1);
    (void)input_ids;

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(n_tokens > 0u);
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / 2u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    const uint16_t *input_norm_weight = model_tensor_f16_payload(&model,
                                                                 uocr_tensor_id_layer_input_norm(0u),
                                                                 UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 sizeof(error));
    const uint16_t *q_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_Q),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *k_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_K),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *v_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_V),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *o_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_O),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *post_norm_weight = model_tensor_f16_payload(&model,
                                                                uocr_tensor_id_layer_post_attn_norm(0u),
                                                                UOCR_HIDDEN_SIZE,
                                                                error,
                                                                sizeof(error));
    const uint16_t *gate_weight = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           sizeof(error));
    const uint16_t *up_weight = model_tensor_f16_payload(&model,
                                                         uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         sizeof(error));
    const uint16_t *down_weight = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE,
                                                           error,
                                                           sizeof(error));
    CHECK(input_norm_weight != NULL);
    CHECK(q_weight != NULL);
    CHECK(k_weight != NULL);
    CHECK(v_weight != NULL);
    CHECK(o_weight != NULL);
    CHECK(post_norm_weight != NULL);
    CHECK(gate_weight != NULL);
    CHECK(up_weight != NULL);
    CHECK(down_weight != NULL);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *normed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *v = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *unused_o = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *context = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(normed != NULL);
    CHECK(q != NULL);
    CHECK(k != NULL);
    CHECK(v != NULL);
    CHECK(unused_o != NULL);
    CHECK(q_rope != NULL);
    CHECK(k_rope != NULL);
    CHECK(context != NULL);
    CHECK(attn_hidden != NULL);
    CHECK(mlp_input != NULL);
    CHECK(actual != NULL);

    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         prompt,
                                         input_norm_weight,
                                         n_tokens,
                                         UOCR_HIDDEN_SIZE,
                                         UOCR_RMS_NORM_EPS,
                                         UOCR_METAL_RMSNORM_OUTPUT_F16,
                                         normed,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                normed,
                                                q_weight,
                                                k_weight,
                                                v_weight,
                                                o_weight,
                                                n_tokens,
                                                UOCR_METAL_DENSE_OUTPUT_F16,
                                                q,
                                                k,
                                                v,
                                                unused_o,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         n_tokens,
                                         0u,
                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                         q_rope,
                                         k_rope,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q_rope,
                                                   k_rope,
                                                   v,
                                                   n_tokens,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   context,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           o_weight,
                                                           prompt,
                                                           n_tokens,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           attn_hidden,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         attn_hidden,
                                         post_norm_weight,
                                         n_tokens,
                                         UOCR_HIDDEN_SIZE,
                                         UOCR_RMS_NORM_EPS,
                                         UOCR_METAL_RMSNORM_OUTPUT_F16,
                                         mlp_input,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              mlp_input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              attn_hidden,
                                              n_tokens,
                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                              actual,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    float max_abs = 0.0f;
    double sum_abs = 0.0;
    uint64_t max_index = 0u;
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        const float diff = fabsf(f16_bits_to_f32(actual[i]) - f16_bits_to_f32(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            max_index = i;
        }
        sum_abs += (double)diff;
    }
    const double mean_abs = sum_abs / (double)hidden_values;
    if (max_abs > 7.5e-3f || mean_abs > 7.5e-4) {
        fprintf(stderr,
                "text layer0 mismatch: max_abs=%g mean_abs=%g at token %llu col %llu actual=%g expected=%g\n",
                (double)max_abs,
                mean_abs,
                (unsigned long long)(max_index / (uint64_t)UOCR_HIDDEN_SIZE),
                (unsigned long long)(max_index % (uint64_t)UOCR_HIDDEN_SIZE),
                (double)f16_bits_to_f32(actual[max_index]),
                (double)f16_bits_to_f32(expected[max_index]));
        free(actual);
        free(mlp_input);
        free(attn_hidden);
        free(context);
        free(k_rope);
        free(q_rope);
        free(unused_o);
        free(v);
        free(k);
        free(q);
        free(normed);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected);
        free(prompt);
        free(input_ids);
        return 1;
    }

    free(actual);
    free(mlp_input);
    free(attn_hidden);
    free(context);
    free(k_rope);
    free(q_rope);
    free(unused_o);
    free(v);
    free(k);
    free(q);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_layer1_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text layer1 parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    uint16_t *expected = NULL;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir, "layer_1_hidden_f16.bin", n_tokens, &expected, error, sizeof(error)) == 1);
    (void)input_ids;
    (void)prompt;

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    const uint64_t expert_values = (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE *
                                   (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(n_tokens > 0u);
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));
    CHECK(expert_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    const uint16_t *router_weight = model_tensor_f16_payload(&model,
                                                             uocr_tensor_id_moe_router(1u),
                                                             (uint64_t)UOCR_ROUTED_EXPERTS * (uint64_t)UOCR_HIDDEN_SIZE,
                                                             error,
                                                             sizeof(error));
    const uint16_t *shared_gate = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_moe_shared(1u, UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           sizeof(error));
    const uint16_t *shared_up = model_tensor_f16_payload(&model,
                                                         uocr_tensor_id_moe_shared(1u, UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         sizeof(error));
    const uint16_t *shared_down = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_moe_shared(1u, UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE,
                                                           error,
                                                           sizeof(error));
    CHECK(router_weight != NULL);
    CHECK(shared_gate != NULL);
    CHECK(shared_up != NULL);
    CHECK(shared_down != NULL);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *routed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *shared = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint32_t *top_ids = (uint32_t *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(uint32_t));
    float *top_weights = (float *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(float));
    uint16_t *selected_gate = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
    uint16_t *selected_up = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
    uint16_t *selected_down = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
    CHECK(attn_hidden != NULL);
    CHECK(mlp_input != NULL);
    CHECK(routed != NULL);
    CHECK(shared != NULL);
    CHECK(actual != NULL);
    CHECK(top_ids != NULL);
    CHECK(top_weights != NULL);
    CHECK(selected_gate != NULL);
    CHECK(selected_up != NULL);
    CHECK(selected_down != NULL);

    CHECK(run_metal_attention_block_from_model_f16(ctx,
                                                   &model,
                                                   1u,
                                                   layer0,
                                                   n_tokens,
                                                   attn_hidden,
                                                   mlp_input,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            mlp_input,
                                            router_weight,
                                            n_tokens,
                                            NULL,
                                            NULL,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    mlp_input,
                                                    shared_gate,
                                                    shared_up,
                                                    shared_down,
                                                    n_tokens,
                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                    shared,
                                                    error,
                                                    sizeof(error)) == 1);
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        CHECK(copy_selected_moe_expert_weights(&model,
                                               1u,
                                               top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                               selected_gate,
                                               selected_up,
                                               selected_down,
                                               error,
                                               sizeof(error)) == 1);
        CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                                 mlp_input + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                 top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                                                 top_weights + (size_t)token * UOCR_MOE_TOP_K,
                                                                 selected_gate,
                                                                 selected_up,
                                                                 selected_down,
                                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                                 routed + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 sizeof(error)) == 1);
    }
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             attn_hidden,
                                             n_tokens,
                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                             actual,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    float max_abs = 0.0f;
    double sum_abs = 0.0;
    uint64_t max_index = 0u;
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        const float diff = fabsf(f16_bits_to_f32(actual[i]) - f16_bits_to_f32(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            max_index = i;
        }
        sum_abs += (double)diff;
    }
    const double mean_abs = sum_abs / (double)hidden_values;
    if (max_abs > 1.25e-2f || mean_abs > 1.25e-3) {
        fprintf(stderr,
                "text layer1 mismatch: max_abs=%g mean_abs=%g at token %llu col %llu actual=%g expected=%g\n",
                (double)max_abs,
                mean_abs,
                (unsigned long long)(max_index / (uint64_t)UOCR_HIDDEN_SIZE),
                (unsigned long long)(max_index % (uint64_t)UOCR_HIDDEN_SIZE),
                (double)f16_bits_to_f32(actual[max_index]),
                (double)f16_bits_to_f32(expected[max_index]));
        free(selected_down);
        free(selected_up);
        free(selected_gate);
        free(top_weights);
        free(top_ids);
        free(actual);
        free(shared);
        free(routed);
        free(mlp_input);
        free(attn_hidden);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected);
        free(layer0);
        free(prompt);
        free(input_ids);
        return 1;
    }

    free(selected_down);
    free(selected_up);
    free(selected_gate);
    free(top_weights);
    free(top_ids);
    free(actual);
    free(shared);
    free(routed);
    free(mlp_input);
    free(attn_hidden);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_remaining_layers_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text remaining-layer parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no layer_11_hidden_f16.bin; skipping text remaining-layer parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    (void)input_ids;
    (void)prompt;

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(n_tokens > 0u);
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *input_hidden = (uint16_t *)malloc((size_t)hidden_values * sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(input_hidden != NULL);
    CHECK(actual != NULL);
    memcpy(input_hidden, layer0, (size_t)hidden_values * sizeof(uint16_t));

    for (uint32_t layer = 2u; layer < UOCR_DECODER_LAYERS; ++layer) {
        char prev_filename[64];
        char expected_filename[64];
        snprintf(prev_filename, sizeof(prev_filename), "layer_%u_hidden_f16.bin", layer - 1u);
        snprintf(expected_filename, sizeof(expected_filename), "layer_%u_hidden_f16.bin", layer);
        uint16_t *prev_expected = NULL;
        uint16_t *expected = NULL;
        CHECK(read_layer_hidden_python_dump(dump_dir, prev_filename, n_tokens, &prev_expected, error, sizeof(error)) == 1);
        CHECK(read_layer_hidden_python_dump(dump_dir, expected_filename, n_tokens, &expected, error, sizeof(error)) == 1);
        memcpy(input_hidden, prev_expected, (size_t)hidden_values * sizeof(uint16_t));
        memset(actual, 0, (size_t)hidden_values * sizeof(uint16_t));
        CHECK(run_metal_moe_decoder_layer_from_model_f16(ctx,
                                                         &model,
                                                         layer,
                                                         input_hidden,
                                                         n_tokens,
                                                         actual,
                                                         error,
                                                         sizeof(error)) == 1);
        char label[64];
        snprintf(label, sizeof(label), "text layer%u", layer);
        const int close = compare_hidden_f16(label, actual, expected, n_tokens, 1.25e-2f, 1.25e-3);
        free(expected);
        free(prev_expected);
        if (!close) {
            free(actual);
            free(input_hidden);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(layer0);
            free(prompt);
            free(input_ids);
            return 1;
        }
    }
    CHECK(error[0] == '\0');

    free(actual);
    free(input_hidden);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_logits_topk_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text logits top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "logits_topk_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "logits_topk_scores_f32.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no logits_topk files; skipping text logits top-k parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    float *expected_scores = NULL;
    uint32_t top_k = 0u;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_logits_topk_python_dump(dump_dir,
                                            &expected_ids,
                                            &expected_scores,
                                            &top_k,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(top_k > 0u && top_k <= 128u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    int32_t *actual_ids = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    float *actual_scores = (float *)malloc((size_t)top_k * sizeof(float));
    CHECK(normed != NULL);
    CHECK(logits != NULL);
    CHECK(actual_ids != NULL);
    CHECK(actual_scores != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               last_hidden,
                                               1u,
                                               UOCR_METAL_RMSNORM_OUTPUT_F16,
                                               normed,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(uocr_metal_context_lm_head_f16(ctx, normed, 1u, logits, error, sizeof(error)) == 1);
    compute_logits_topk_f32(logits, UOCR_VOCAB_SIZE, top_k, actual_ids, actual_scores);
    CHECK(error[0] == '\0');

    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        if (actual_ids[rank] != expected_ids[rank]) {
            fprintf(stderr,
                    "text logits top-k id mismatch at rank %u: actual=%d expected=%d actual_score=%g expected_score=%g\n",
                    rank,
                    actual_ids[rank],
                    expected_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank]);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(layer0);
            free(prompt);
            free(input_ids);
            return 1;
        }
        const float diff = fabsf(actual_scores[rank] - expected_scores[rank]);
        if (diff > 2.5e-2f) {
            fprintf(stderr,
                    "text logits top-k score mismatch at rank %u token %d: actual=%g expected=%g diff=%g\n",
                    rank,
                    actual_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank],
                    (double)diff);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(layer0);
            free(prompt);
            free(input_ids);
            return 1;
        }
    }

    free(actual_scores);
    free(actual_ids);
    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_scores);
    free(expected_ids);
    free(layer11);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text generated-id parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "generated_ids_i32.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no generated_ids_i32.bin; skipping text generated-id parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    uint32_t n_generated = 0u;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_generated_ids_python_dump(dump_dir,
                                              &expected_ids,
                                              &n_generated,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(n_generated == 1u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    uint32_t actual_token = UINT32_MAX;
    float actual_score = 0.0f;
    CHECK(normed != NULL);
    CHECK(logits != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   last_hidden,
                                                   1u,
                                                   NULL,
                                                   normed,
                                                   logits,
                                                   &actual_token,
                                                   &actual_score,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    if ((int32_t)actual_token != expected_ids[0]) {
        fprintf(stderr,
                "text generated-id mismatch: actual=%u expected=%d score=%g\n",
                actual_token,
                expected_ids[0],
                (double)actual_score);
        free(logits);
        free(normed);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_ids);
        free(layer11);
        free(layer0);
        free(prompt);
        free(input_ids);
        return 1;
    }

    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_ids);
    free(layer11);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static void write_q8_scale_le(uint8_t *dst, uint16_t scale_bits) {
    dst[0] = (uint8_t)(scale_bits & 0xffu);
    dst[1] = (uint8_t)(scale_bits >> 8u);
}

static float q8_0_expected_value(const uint8_t *table, uint32_t row_size, uint32_t row, uint32_t col) {
    const uint32_t block = col / 32u;
    const uint32_t in_block = col % 32u;
    const uint8_t *packed = table + (size_t)row * row_size + (size_t)block * 34u;
    const uint16_t scale_bits = (uint16_t)((uint16_t)packed[0] | ((uint16_t)packed[1] << 8u));
    const int8_t q = (int8_t)packed[2u + in_block];
    return f16_bits_to_f32(scale_bits) * (float)q;
}

static void set_q8_0_q(uint8_t *table, uint32_t row_size, uint32_t row, uint32_t col, int8_t q) {
    const uint32_t block = col / 32u;
    const uint32_t in_block = col % 32u;
    uint8_t *packed = table + (size_t)row * row_size + (size_t)block * 34u;
    packed[2u + in_block] = (uint8_t)q;
}

static void init_q4_k_rows(uint8_t *table, uint32_t rows, uint32_t row_size, uint32_t physical_cols, uint16_t d_bits) {
    memset(table, 0, (size_t)rows * row_size);
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t block = 0u; block < physical_cols / UOCR_Q4_K_BLOCK_SIZE; ++block) {
            uint8_t *packed = table + (size_t)row * row_size + (size_t)block * UOCR_Q4_K_TYPE_SIZE;
            write_q8_scale_le(packed, d_bits);
            write_q8_scale_le(packed + 2u, 0u);
            for (uint32_t i = 0u; i < 4u; ++i) {
                packed[4u + i] = 1u;
                packed[8u + i] = 0u;
                packed[12u + i] = 1u;
            }
        }
    }
}

static uint8_t q4_k_scale_value(const uint8_t *scales, uint32_t group) {
    if (group < 4u) {
        return (uint8_t)(scales[group] & 63u);
    }
    const uint32_t k = group - 4u;
    return (uint8_t)((scales[8u + k] & 0x0fu) | ((scales[k] & 0xc0u) >> 2u));
}

static uint8_t q4_k_min_value(const uint8_t *scales, uint32_t group) {
    if (group < 4u) {
        return (uint8_t)(scales[group + 4u] & 63u);
    }
    const uint32_t k = group - 4u;
    return (uint8_t)((scales[8u + k] >> 4u) | ((scales[4u + k] & 0xc0u) >> 2u));
}

static float q4_k_expected_value(const uint8_t *table, uint32_t row_size, uint32_t row, uint32_t col) {
    const uint32_t block = col / UOCR_Q4_K_BLOCK_SIZE;
    const uint32_t in_block = col % UOCR_Q4_K_BLOCK_SIZE;
    const uint8_t *packed = table + (size_t)row * row_size + (size_t)block * UOCR_Q4_K_TYPE_SIZE;
    const uint16_t d_bits = (uint16_t)((uint16_t)packed[0] | ((uint16_t)packed[1] << 8u));
    const uint16_t dmin_bits = (uint16_t)((uint16_t)packed[2] | ((uint16_t)packed[3] << 8u));
    const uint8_t *scales = packed + 4u;
    const uint8_t *qs = packed + 16u;
    const uint32_t il = in_block / 16u;
    const uint32_t offset = in_block % 16u;
    const uint32_t group = il / 2u;
    const uint32_t q_base = (il / 4u) * 32u + 16u * (il & 1u);
    const uint8_t q_byte = qs[q_base + offset];
    const uint32_t q = (il & 2u) == 0u ? (uint32_t)(q_byte & 0x0fu) : (uint32_t)(q_byte >> 4u);
    return f16_bits_to_f32(d_bits) * (float)q4_k_scale_value(scales, group) * (float)q -
           f16_bits_to_f32(dmin_bits) * (float)q4_k_min_value(scales, group);
}

static void set_q4_k_scale_min(uint8_t *table,
                                 uint32_t row_size,
                                 uint32_t row,
                                 uint32_t block,
                                 uint32_t group,
                                 uint8_t scale,
                                 uint8_t min) {
    uint8_t *packed = table + (size_t)row * row_size + (size_t)block * UOCR_Q4_K_TYPE_SIZE;
    uint8_t *scales = packed + 4u;
    scale = (uint8_t)(scale & 63u);
    min = (uint8_t)(min & 63u);
    if (group < 4u) {
        scales[group] = (uint8_t)((scales[group] & 0xc0u) | scale);
        scales[group + 4u] = (uint8_t)((scales[group + 4u] & 0xc0u) | min);
    } else {
        const uint32_t k = group - 4u;
        scales[8u + k] = (uint8_t)((scale & 0x0fu) | (uint8_t)((min & 0x0fu) << 4u));
        scales[k] = (uint8_t)((scales[k] & 0x3fu) | (uint8_t)((scale & 0x30u) << 2u));
        scales[4u + k] = (uint8_t)((scales[4u + k] & 0x3fu) | (uint8_t)((min & 0x30u) << 2u));
    }
}

static void set_q4_k_q(uint8_t *table, uint32_t row_size, uint32_t row, uint32_t col, uint8_t q) {
    const uint32_t block = col / UOCR_Q4_K_BLOCK_SIZE;
    const uint32_t in_block = col % UOCR_Q4_K_BLOCK_SIZE;
    const uint32_t il = in_block / 16u;
    const uint32_t offset = in_block % 16u;
    const uint32_t q_base = (il / 4u) * 32u + 16u * (il & 1u);
    uint8_t *packed = table + (size_t)row * row_size + (size_t)block * UOCR_Q4_K_TYPE_SIZE;
    uint8_t *dst = packed + 16u + q_base + offset;
    q = (uint8_t)(q & 0x0fu);
    if ((il & 2u) == 0u) {
        *dst = (uint8_t)((*dst & 0xf0u) | q);
    } else {
        *dst = (uint8_t)((*dst & 0x0fu) | (uint8_t)(q << 4u));
    }
}

static int test_metal_get_rows_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TABLE_ROWS = 5, ROW_WIDTH = 8, OUT_ROWS = 4 };
    const uint16_t values[] = {
        0x0000u, /* 0.0 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x3800u, /* 0.5 */
        0xb800u, /* -0.5 */
        0x4200u, /* 3.0 */
        0xc200u, /* -3.0 */
        0x4400u  /* 4.0 */
    };
    uint16_t table[TABLE_ROWS * ROW_WIDTH];
    for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * ROW_WIDTH); ++i) {
        table[i] = values[(i * 3u + 1u) % (sizeof(values) / sizeof(values[0]))];
    }
    const int32_t row_ids[OUT_ROWS] = {3, 0, 4, 1};

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t out_f16[OUT_ROWS * ROW_WIDTH];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          row_ids,
                                          OUT_ROWS,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                          out_f16,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)ROW_WIDTH; ++col) {
            const uint16_t expected = table[(uint32_t)row_ids[row] * (uint32_t)ROW_WIDTH + col];
            CHECK(out_f16[row * (uint32_t)ROW_WIDTH + col] == expected);
        }
    }

    float out_f32[OUT_ROWS * ROW_WIDTH];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          row_ids,
                                          OUT_ROWS,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F32,
                                          out_f32,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)ROW_WIDTH; ++col) {
            const uint16_t expected_bits = table[(uint32_t)row_ids[row] * (uint32_t)ROW_WIDTH + col];
            CHECK(out_f32[row * (uint32_t)ROW_WIDTH + col] == f16_bits_to_f32(expected_bits));
        }
    }

    const int32_t bad_row_ids[1] = {TABLE_ROWS};
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          bad_row_ids,
                                          1u,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                          out_f16,
                                          error,
                                          sizeof(error)) == 0);
    CHECK(strstr(error, "outside table rows") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_get_rows_q8_0(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TABLE_ROWS = 4, LOGICAL_WIDTH = 33, PHYSICAL_WIDTH = 64, OUT_ROWS = 3 };
    enum { ROW_SIZE = (PHYSICAL_WIDTH / 32) * 34 };
    uint8_t table[TABLE_ROWS * ROW_SIZE];
    memset(table, 0, sizeof(table));

    for (uint32_t row = 0u; row < (uint32_t)TABLE_ROWS; ++row) {
        uint8_t *row_base = table + row * (uint32_t)ROW_SIZE;
        write_q8_scale_le(row_base, row == 0u ? f32_to_f16_bits(0.0f) : f32_to_f16_bits(0.25f * (float)row));
        int8_t *q0 = (int8_t *)(void *)(row_base + 2u);
        for (uint32_t col = 0u; col < 32u; ++col) {
            q0[col] = (int8_t)(((int)(row * 17u + col * 3u) % 41) - 20);
        }
        uint8_t *block1 = row_base + 34u;
        write_q8_scale_le(block1, f32_to_f16_bits(0.125f * (float)(row + 1u)));
        int8_t *q1 = (int8_t *)(void *)(block1 + 2u);
        q1[0] = (int8_t)(12 - (int)row);
        q1[1] = -99; /* physical padding; must not appear in logical output */
    }

    const int32_t row_ids[OUT_ROWS] = {2, 1, 3};
    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    float out_f32[OUT_ROWS * LOGICAL_WIDTH];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_get_rows_q8_0(ctx,
                                           table,
                                           TABLE_ROWS,
                                           LOGICAL_WIDTH,
                                           PHYSICAL_WIDTH,
                                           row_ids,
                                           OUT_ROWS,
                                           UOCR_METAL_GET_ROWS_OUTPUT_F32,
                                           out_f32,
                                           error,
                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t out_row = 0u; out_row < (uint32_t)OUT_ROWS; ++out_row) {
        for (uint32_t col = 0u; col < (uint32_t)LOGICAL_WIDTH; ++col) {
            const float expected = q8_0_expected_value(table, ROW_SIZE, (uint32_t)row_ids[out_row], col);
            const float actual = out_f32[out_row * (uint32_t)LOGICAL_WIDTH + col];
            CHECK(fabsf(actual - expected) <= 1.0e-6f);
        }
    }

    uint16_t out_f16[OUT_ROWS * LOGICAL_WIDTH];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_get_rows_q8_0(ctx,
                                           table,
                                           TABLE_ROWS,
                                           LOGICAL_WIDTH,
                                           PHYSICAL_WIDTH,
                                           row_ids,
                                           OUT_ROWS,
                                           UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                           out_f16,
                                           error,
                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t out_row = 0u; out_row < (uint32_t)OUT_ROWS; ++out_row) {
        for (uint32_t col = 0u; col < (uint32_t)LOGICAL_WIDTH; ++col) {
            const float expected = q8_0_expected_value(table, ROW_SIZE, (uint32_t)row_ids[out_row], col);
            CHECK(out_f16[out_row * (uint32_t)LOGICAL_WIDTH + col] == f32_to_f16_bits(expected));
        }
    }

    const int32_t bad_row_ids[1] = {TABLE_ROWS};
    CHECK(uocr_metal_context_get_rows_q8_0(ctx,
                                           table,
                                           TABLE_ROWS,
                                           LOGICAL_WIDTH,
                                           PHYSICAL_WIDTH,
                                           bad_row_ids,
                                           1u,
                                           UOCR_METAL_GET_ROWS_OUTPUT_F32,
                                           out_f32,
                                           error,
                                           sizeof(error)) == 0);
    CHECK(strstr(error, "outside table rows") != NULL);

    CHECK(uocr_metal_context_get_rows_q8_0(ctx,
                                           table,
                                           TABLE_ROWS,
                                           LOGICAL_WIDTH,
                                           33u,
                                           row_ids,
                                           OUT_ROWS,
                                           UOCR_METAL_GET_ROWS_OUTPUT_F32,
                                           out_f32,
                                           error,
                                           sizeof(error)) == 0);
    CHECK(strstr(error, "widths") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_prompt_assembly_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TABLE_ROWS = 6, HIDDEN = 8, TEXT_TOKENS = 4, IMAGE_TOKENS = 2, PROMPT_TOKENS = 6 };
    uint16_t table[TABLE_ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)TABLE_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            table[row * (uint32_t)HIDDEN + col] = (uint16_t)(0x3c00u + row * 0x40u + col);
        }
    }
    uint16_t image_features[IMAGE_TOKENS * HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
        image_features[i] = (uint16_t)(0x5000u + i);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const int32_t text_ids[TEXT_TOKENS] = {0, 3, 5, 1};
    uint16_t text_out[TEXT_TOKENS * HIDDEN];
    memset(text_out, 0, sizeof(text_out));
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 text_ids,
                                                 TEXT_TOKENS,
                                                 UINT32_MAX,
                                                 0u,
                                                 NULL,
                                                 text_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)TEXT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(text_out[token * (uint32_t)HIDDEN + col] == table[(uint32_t)text_ids[token] * (uint32_t)HIDDEN + col]);
        }
    }

    const int32_t prompt_ids[PROMPT_TOKENS] = {0, 2, 12345, 12346, 4, 1};
    uint16_t prompt_out[PROMPT_TOKENS * HIDDEN];
    memset(prompt_out, 0, sizeof(prompt_out));
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 prompt_ids,
                                                 PROMPT_TOKENS,
                                                 2u,
                                                 IMAGE_TOKENS,
                                                 image_features,
                                                 prompt_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)PROMPT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint16_t expected = 0u;
            if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
            } else {
                expected = table[(uint32_t)prompt_ids[token] * (uint32_t)HIDDEN + col];
            }
            CHECK(prompt_out[token * (uint32_t)HIDDEN + col] == expected);
        }
    }

    const int32_t bad_text_ids[1] = {TABLE_ROWS};
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 bad_text_ids,
                                                 1u,
                                                 UINT32_MAX,
                                                 0u,
                                                 NULL,
                                                 text_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "outside embedding rows") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_prompt_assembly_from_mapped_model_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROW_COUNT = 4, HIDDEN = UOCR_HIDDEN_SIZE, TEXT_TOKENS = 3, IMAGE_TOKENS = 2, PROMPT_TOKENS = 6 };
    const uint32_t rows[ROW_COUNT] = {0u, 1u, 42u, 99u};
    uint16_t row_weights[ROW_COUNT * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)(((row + 5u) * 13u + col * 7u) % 31u) - 15;
            row_weights[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.015625f);
        }
    }
    uint16_t image_features[IMAGE_TOKENS * HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
        image_features[i] = f32_to_f16_bits((float)((int)(i % 23u) - 11) * 0.03125f);
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_sparse_tok_embed_uocr_model(path, rows, row_weights, ROW_COUNT) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const int32_t text_ids[TEXT_TOKENS] = {0, 42, 1};
    uint16_t text_out[TEXT_TOKENS * HIDDEN];
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            text_ids,
                                                            TEXT_TOKENS,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            text_out,
                                                            error,
                                                            sizeof(error)) == 0);
    CHECK(strstr(error, "requires mapped model views") != NULL);

    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            text_ids,
                                                            TEXT_TOKENS,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            text_out,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)TEXT_TOKENS; ++token) {
        uint32_t row_index = UINT32_MAX;
        for (uint32_t r = 0u; r < (uint32_t)ROW_COUNT; ++r) {
            if (rows[r] == (uint32_t)text_ids[token]) {
                row_index = r;
            }
        }
        CHECK(row_index != UINT32_MAX);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(text_out[token * (uint32_t)HIDDEN + col] == row_weights[row_index * (uint32_t)HIDDEN + col]);
        }
    }

    const int32_t prompt_ids[PROMPT_TOKENS] = {0, 99, UOCR_TOKEN_IMAGE, UOCR_TOKEN_IMAGE, 42, 1};
    uint16_t prompt_out[PROMPT_TOKENS * HIDDEN];
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            prompt_ids,
                                                            PROMPT_TOKENS,
                                                            2u,
                                                            IMAGE_TOKENS,
                                                            image_features,
                                                            prompt_out,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)PROMPT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint16_t expected = 0u;
            if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
            } else {
                uint32_t row_index = UINT32_MAX;
                for (uint32_t r = 0u; r < (uint32_t)ROW_COUNT; ++r) {
                    if (rows[r] == (uint32_t)prompt_ids[token]) {
                        row_index = r;
                    }
                }
                CHECK(row_index != UINT32_MAX);
                expected = row_weights[row_index * (uint32_t)HIDDEN + col];
            }
            CHECK(prompt_out[token * (uint32_t)HIDDEN + col] == expected);
        }
    }

    CHECK(uocr_metal_context_assemble_prompt_from_model_to_arena_f16(ctx,
                                                                     text_ids,
                                                                     TEXT_TOKENS,
                                                                     UINT32_MAX,
                                                                     0u,
                                                                     NULL,
                                                                     0u,
                                                                     error,
                                                                     sizeof(error)) == 0);
    CHECK(strstr(error, "requires allocated arenas") != NULL);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 2u, PROMPT_TOKENS, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_assemble_prompt_from_model_to_arena_f16(ctx,
                                                                     text_ids,
                                                                     TEXT_TOKENS,
                                                                     UINT32_MAX,
                                                                     0u,
                                                                     NULL,
                                                                     0u,
                                                                     error,
                                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    uint16_t arena_text_out[TEXT_TOKENS * HIDDEN];
    memset(arena_text_out, 0, sizeof(arena_text_out));
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx, 0u, TEXT_TOKENS, arena_text_out, error, sizeof(error)) == 1);
    CHECK(memcmp(arena_text_out, text_out, sizeof(arena_text_out)) == 0);

    CHECK(uocr_metal_context_assemble_prompt_from_model_to_arena_f16(ctx,
                                                                     prompt_ids,
                                                                     PROMPT_TOKENS,
                                                                     2u,
                                                                     IMAGE_TOKENS,
                                                                     image_features,
                                                                     1u,
                                                                     error,
                                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    uint16_t arena_prompt_out[PROMPT_TOKENS * HIDDEN];
    memset(arena_prompt_out, 0, sizeof(arena_prompt_out));
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx, 1u, PROMPT_TOKENS, arena_prompt_out, error, sizeof(error)) == 1);
    CHECK(memcmp(arena_prompt_out, prompt_out, sizeof(arena_prompt_out)) == 0);
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx, 2u, TEXT_TOKENS, arena_text_out, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires allocated arenas") != NULL);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_rmsnorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, HIDDEN = 1280 };
    const uint16_t input_values[] = {
        0x3800u, /* 0.5 */
        0xb800u, /* -0.5 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x3400u, /* 0.25 */
        0xb400u  /* -0.25 */
    };
    const uint16_t weight_values[] = {
        0x3c00u, /* 1.0 */
        0x3800u, /* 0.5 */
        0x4000u, /* 2.0 */
        0x3e00u, /* 1.5 */
        0x3a00u  /* 0.75 */
    };
    uint16_t input[ROWS * HIDDEN];
    uint16_t weight[HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        input[i] = input_values[(i * 5u + 3u) % (sizeof(input_values) / sizeof(input_values[0]))];
    }
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        weight[i] = weight_values[(i * 7u + 1u) % (sizeof(weight_values) / sizeof(weight_values[0]))];
    }

    const float eps = 1.0e-6f;
    float expected[ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            sum += x * x;
        }
        const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + eps);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            const float w = f16_bits_to_f32(weight[col]);
            expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    float out_f32[ROWS * HIDDEN];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         eps,
                                         UOCR_METAL_RMSNORM_OUTPUT_F32,
                                         out_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-4f);
    }

    uint16_t out_f16[ROWS * HIDDEN];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         eps,
                                         UOCR_METAL_RMSNORM_OUTPUT_F16,
                                         out_f16,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 4.0e-3f);
    }

    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         0.0f,
                                         UOCR_METAL_RMSNORM_OUTPUT_F32,
                                         out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "eps must be positive") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_final_rmsnorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, HIDDEN = UOCR_HIDDEN_SIZE };
    uint16_t input[ROWS * HIDDEN];
    uint16_t weight[HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)((row * 13u + col * 5u) % 17u) - 8;
            input[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.0625f);
        }
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        weight[col] = f32_to_f16_bits(0.5f + (float)((col * 3u) % 11u) * 0.03125f);
    }

    float expected[ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            sum += x * x;
        }
        const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + UOCR_RMS_NORM_EPS);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            const float w = f16_bits_to_f32(weight[col]);
            expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
        }
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_single_final_norm_uocr_model(path, weight) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    float out_f32[ROWS * HIDDEN];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F32,
                                               out_f32,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-4f);
    }

    uint16_t out_f16[ROWS * HIDDEN];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F16,
                                               out_f16,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 4.0e-3f);
    }

    uocr_metal_context_unmap_model(ctx);
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F32,
                                               out_f32,
                                               error,
                                               sizeof(error)) == 0);
    CHECK(strstr(error, "mapped model views") != NULL);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_lm_head_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROW_COUNT = 4, HIDDEN = UOCR_HIDDEN_SIZE };
    const uint32_t rows[ROW_COUNT] = {0u, 1u, UOCR_TOKEN_IMAGE, UOCR_VOCAB_SIZE - 1u};
    uint16_t input[HIDDEN];
    uint16_t row_weights[ROW_COUNT * HIDDEN];
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const int centered = (int)((col * 7u) % 13u) - 6;
        input[col] = f32_to_f16_bits((float)centered * 0.03125f);
    }
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)(((row + 3u) * 11u + col * 5u) % 17u) - 8;
            row_weights[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.015625f);
        }
    }

    float expected[ROW_COUNT];
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            sum += f16_bits_to_f32(input[col]) * f16_bits_to_f32(row_weights[row * (uint32_t)HIDDEN + col]);
        }
        expected[row] = sum;
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_sparse_lm_head_uocr_model(path, rows, row_weights, ROW_COUNT) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(logits != NULL);
    memset(logits, 0x7f, (size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(uocr_metal_context_lm_head_f16(ctx, input, 1u, logits, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        CHECK(fabsf(logits[rows[row]] - expected[row]) < 2.0e-4f);
    }
    CHECK(fabsf(logits[UOCR_TOKEN_PAD]) < 1.0e-7f);
    CHECK(fabsf(logits[42u]) < 1.0e-7f);

    free(logits);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_argmax_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, VOCAB = UOCR_VOCAB_SIZE };
    float *logits = (float *)malloc((size_t)ROWS * (size_t)VOCAB * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = -1000.0f;
    }
    logits[0u * (uint32_t)VOCAB + 17u] = 3.25f;
    logits[0u * (uint32_t)VOCAB + 42u] = 3.25f; /* tie: lower token id wins */
    logits[1u * (uint32_t)VOCAB + UOCR_TOKEN_PAD] = -0.5f;
    logits[1u * (uint32_t)VOCAB + UOCR_TOKEN_IMAGE] = 12.0f;
    logits[1u * (uint32_t)VOCAB + (uint32_t)VOCAB - 1u] = 11.9f;
    for (uint32_t col = 0u; col < (uint32_t)VOCAB; ++col) {
        logits[2u * (uint32_t)VOCAB + col] = -INFINITY;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f, 0.0f};
    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        logits,
                                        ROWS,
                                        VOCAB,
                                        token_ids,
                                        scores,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(token_ids[0] == 17u);
    CHECK(scores[0] == 3.25f);
    CHECK(token_ids[1] == (uint32_t)UOCR_TOKEN_IMAGE);
    CHECK(scores[1] == 12.0f);
    CHECK(token_ids[2] == 0u);
    CHECK(isinf(scores[2]) && scores[2] < 0.0f);

    const float small_logits[10] = {
        -1.0f, 0.5f, 0.5f, -2.0f, 0.25f,
        -INFINITY, -4.0f, -3.0f, -2.0f, -1.0f,
    };
    uint32_t small_ids[2] = {UINT32_MAX, UINT32_MAX};
    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        small_logits,
                                        2u,
                                        5u,
                                        small_ids,
                                        NULL,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(small_ids[0] == 1u);
    CHECK(small_ids[1] == 4u);

    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        logits,
                                        ROWS,
                                        0u,
                                        token_ids,
                                        scores,
                                        error,
                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal argmax request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(logits);
    return 0;
}

static int test_metal_no_repeat_ngram_gpu_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 4, VOCAB = 16 };
    float logits[ROWS * VOCAB];
    float expected[ROWS * VOCAB];
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = (float)i * 0.125f;
        expected[i] = logits[i];
    }

    const int32_t sequence0[] = {1, 2, 3, 1, 2};       /* current prefix [1,2] bans 3 */
    const int32_t sequence1[] = {5, 6, 7, 5, 6, 8, 5, 6}; /* full window bans 7 and 8 */
    const int32_t sequence2[] = {2, 4, 2, 5};          /* ngram=1, window=2 bans 2 and 5 */
    const uocr_no_repeat_ngram_config configs[ROWS] = {
        {sequence0, 5u, 3u, 0u},
        {sequence1, 8u, 3u, 0u},
        {sequence2, 4u, 1u, 2u},
        {NULL, 0u, 0u, 0u},
    };
    CHECK(uocr_no_repeat_ngram_apply_batch(expected, ROWS, VOCAB, configs) == UOCR_OK);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_apply_no_repeat_ngram_f32(ctx,
                                                       logits,
                                                       ROWS,
                                                       VOCAB,
                                                       configs,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        if (isinf(expected[i])) {
            CHECK(isinf(logits[i]) && logits[i] < 0.0f);
        } else {
            CHECK(logits[i] == expected[i]);
        }
    }

    float unchanged[VOCAB];
    for (uint32_t i = 0u; i < (uint32_t)VOCAB; ++i) {
        unchanged[i] = (float)i;
    }
    CHECK(uocr_metal_context_apply_no_repeat_ngram_f32(ctx,
                                                       unchanged,
                                                       1u,
                                                       VOCAB,
                                                       NULL,
                                                       error,
                                                       sizeof(error)) == 1);
    for (uint32_t i = 0u; i < (uint32_t)VOCAB; ++i) {
        CHECK(unchanged[i] == (float)i);
    }

    const uocr_no_repeat_ngram_config invalid[1] = {{NULL, 5u, 3u, 0u}};
    CHECK(uocr_metal_context_apply_no_repeat_ngram_f32(ctx,
                                                       unchanged,
                                                       1u,
                                                       VOCAB,
                                                       invalid,
                                                       error,
                                                       sizeof(error)) == 0);
    CHECK(strstr(error, "failed to validate no-repeat-ngram row") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_select_greedy_with_no_repeat_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, VOCAB = UOCR_VOCAB_SIZE };
    float *logits = (float *)malloc((size_t)ROWS * (size_t)VOCAB * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = -1000.0f;
    }
    logits[0u * (uint32_t)VOCAB + 3u] = 100.0f;
    logits[0u * (uint32_t)VOCAB + 4u] = 90.0f;
    logits[1u * (uint32_t)VOCAB + 8u] = 50.0f;
    logits[1u * (uint32_t)VOCAB + 9u] = 40.0f;

    const int32_t sequence0[] = {1, 2, 3, 1, 2}; /* current prefix [1,2] bans 3 */
    const uocr_no_repeat_ngram_config no_repeat[ROWS] = {
        {sequence0, 5u, 3u, 0u},
        {NULL, 0u, 0u, 0u},
    };

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f};
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               ROWS,
                                               VOCAB,
                                               no_repeat,
                                               token_ids,
                                               scores,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(isinf(logits[3u]) && logits[3u] < 0.0f);
    CHECK(token_ids[0] == 4u);
    CHECK(scores[0] == 90.0f);
    CHECK(token_ids[1] == 8u);
    CHECK(scores[1] == 50.0f);

    logits[0u] = 1.0f;
    token_ids[0] = UINT32_MAX;
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               VOCAB,
                                               NULL,
                                               token_ids,
                                               NULL,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(token_ids[0] == 4u);

    const uocr_no_repeat_ngram_config invalid[1] = {{NULL, 5u, 3u, 0u}};
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               VOCAB,
                                               invalid,
                                               token_ids,
                                               scores,
                                               error,
                                               sizeof(error)) == 0);
    CHECK(strstr(error, "failed to apply no-repeat-ngram bans") != NULL);

    uocr_metal_context_destroy(ctx);
    free(logits);
    return 0;
}

static int test_metal_eos_stop_after_greedy_selection(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < UOCR_VOCAB_SIZE; ++i) {
        logits[i] = -1000.0f;
    }
    logits[(uint32_t)UOCR_TOKEN_EOS] = 5.0f;
    logits[42u] = 4.0f;

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t next_token = UINT32_MAX;
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               UOCR_VOCAB_SIZE,
                                               NULL,
                                               &next_token,
                                               NULL,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(next_token == (uint32_t)UOCR_TOKEN_EOS);

    const int32_t input_ids[] = {UOCR_TOKEN_BOS, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;
    request.max_new_tokens = 4u;

    uocr_sequence_state state;
    char sequence_error[128];
    CHECK(uocr_build_sequence_state(&request, &state, sequence_error, sizeof(sequence_error)) == UOCR_OK);
    int32_t generated[4] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)next_token, generated, 4u) == UOCR_OK);
    CHECK(generated[0] == UOCR_TOKEN_EOS);
    CHECK(state.generated_count == 1u);
    CHECK(state.eos == 1);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    uocr_metal_context_destroy(ctx);
    free(logits);
    return 0;
}

static int test_metal_select_next_token_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, HIDDEN = UOCR_HIDDEN_SIZE, ROW_COUNT = 3 };
    const uint32_t rows[ROW_COUNT] = {3u, 4u, (uint32_t)UOCR_TOKEN_EOS};
    uint16_t final_norm[HIDDEN];
    uint16_t hidden[ROWS * HIDDEN];
    uint16_t row_weights[ROW_COUNT * HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        final_norm[i] = 0x3c00u; /* 1.0 */
    }
    memset(hidden, 0, sizeof(hidden));
    memset(row_weights, 0, sizeof(row_weights));
    hidden[0u] = 0x3c00u;                    /* row 0 selects between token 3 and 4. */
    hidden[(uint32_t)HIDDEN + 1u] = 0x3c00u; /* row 1 selects EOS. */
    row_weights[0u * (uint32_t)HIDDEN + 0u] = 0x3800u; /* token 3: 0.5 * normed[0] */
    row_weights[1u * (uint32_t)HIDDEN + 0u] = 0x3c00u; /* token 4: 1.0 * normed[0], then banned */
    row_weights[2u * (uint32_t)HIDDEN + 1u] = 0x3c00u; /* EOS: 1.0 * normed[1] */

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_final_norm_and_sparse_lm_head_uocr_model(path,
                                                                   final_norm,
                                                                   rows,
                                                                   row_weights,
                                                                   ROW_COUNT) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)malloc((size_t)ROWS * HIDDEN * sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)ROWS * UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(normed != NULL && logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * UOCR_VOCAB_SIZE); ++i) {
        logits[i] = -12345.0f;
    }

    const int32_t sequence0[] = {1, 2, 4, 1, 2}; /* current prefix [1,2] bans token 4 */
    const uocr_no_repeat_ngram_config no_repeat[ROWS] = {
        {sequence0, 5u, 3u, 0u},
        {NULL, 0u, 0u, 0u},
    };
    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f};
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   hidden,
                                                   ROWS,
                                                   no_repeat,
                                                   normed,
                                                   logits,
                                                   token_ids,
                                                   scores,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(token_ids[0] == 3u);
    CHECK(token_ids[1] == (uint32_t)UOCR_TOKEN_EOS);
    CHECK(isinf(logits[4u]) && logits[4u] < 0.0f);
    CHECK(scores[0] == logits[3u]);
    CHECK(scores[1] == logits[1u * (uint32_t)UOCR_VOCAB_SIZE + (uint32_t)UOCR_TOKEN_EOS]);
    CHECK(scores[0] > 0.0f && scores[1] > 0.0f);

    const int32_t input_ids[] = {UOCR_TOKEN_BOS, 42};
    const uint8_t image_mask[] = {0u, 0u};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;
    request.max_new_tokens = 4u;

    uocr_sequence_state state;
    char sequence_error[128];
    CHECK(uocr_build_sequence_state(&request, &state, sequence_error, sizeof(sequence_error)) == UOCR_OK);
    int32_t generated[4] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)token_ids[1], generated, 4u) == UOCR_OK);
    CHECK(generated[0] == UOCR_TOKEN_EOS);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   hidden,
                                                   ROWS,
                                                   no_repeat,
                                                   NULL,
                                                   logits,
                                                   token_ids,
                                                   scores,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal next-token selection request") != NULL);

    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_dense_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, IN_FEATURES = 1280, OUT_FEATURES = 96 };
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint16_t weight_values[] = {
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint16_t bias_values[] = {
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t bias_value_count = (uint32_t)(sizeof(bias_values) / sizeof(bias_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)ROWS * IN_FEATURES * sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)malloc((size_t)OUT_FEATURES * IN_FEATURES * sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)malloc((size_t)OUT_FEATURES * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(uint16_t));
    CHECK(input != NULL && weight != NULL && bias != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(ROWS * IN_FEATURES); ++i) {
        input[i] = input_values[(i * 17u + i / 29u + 1u) % input_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)(OUT_FEATURES * IN_FEATURES); ++i) {
        weight[i] = weight_values[(i * 11u + i / 37u + 3u) % weight_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        bias[i] = bias_values[(i * 5u + 1u) % bias_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
            float sum = f16_bits_to_f32(bias[col]);
            for (uint32_t k = 0u; k < (uint32_t)IN_FEATURES; ++k) {
                sum += f16_bits_to_f32(input[row * (uint32_t)IN_FEATURES + k]) *
                       f16_bits_to_f32(weight[col * (uint32_t)IN_FEATURES + k]);
            }
            expected[row * (uint32_t)OUT_FEATURES + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       ROWS,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * OUT_FEATURES); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.5e-3f);
    }

    memset(out_f16, 0, (size_t)ROWS * OUT_FEATURES * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       1u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                       out_f16,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 5.0e-2f);
    }

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       NULL,
                                       1u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
        CHECK(fabsf(out_f32[col] - (expected[col] - f16_bits_to_f32(bias[col]))) < 2.5e-3f);
    }

    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       ROWS,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       (uocr_metal_dense_output_type)99,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal dense output type") != NULL);
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       0u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal dense request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_dense_q8_0(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, LOGICAL_IN_FEATURES = 1277, PHYSICAL_IN_FEATURES = 1280, OUT_FEATURES = 64 };
    enum { WEIGHT_ROW_SIZE = (PHYSICAL_IN_FEATURES / 32) * 34 };
    const uint16_t input_values[] = {
        0xb400u, /* -0.25 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t bias_values[] = {
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x2c00u  /* 0.0625 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t bias_value_count = (uint32_t)(sizeof(bias_values) / sizeof(bias_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)ROWS * LOGICAL_IN_FEATURES * sizeof(uint16_t));
    uint8_t *weight = (uint8_t *)malloc((size_t)OUT_FEATURES * WEIGHT_ROW_SIZE);
    uint16_t *bias = (uint16_t *)malloc((size_t)OUT_FEATURES * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)OUT_FEATURES * sizeof(uint16_t));
    CHECK(input != NULL && weight != NULL && bias != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(ROWS * LOGICAL_IN_FEATURES); ++i) {
        input[i] = input_values[(i * 19u + i / 31u + 2u) % input_value_count];
    }
    memset(weight, 0, (size_t)OUT_FEATURES * WEIGHT_ROW_SIZE);
    for (uint32_t out_col = 0u; out_col < (uint32_t)OUT_FEATURES; ++out_col) {
        uint8_t *row_base = weight + (size_t)out_col * WEIGHT_ROW_SIZE;
        for (uint32_t block = 0u; block < (uint32_t)(PHYSICAL_IN_FEATURES / 32); ++block) {
            uint8_t *packed = row_base + (size_t)block * 34u;
            const float scale = 0.00390625f * (float)(1u + ((out_col + block) % 4u));
            write_q8_scale_le(packed, f32_to_f16_bits(scale));
            int8_t *qs = (int8_t *)(void *)(packed + 2u);
            for (uint32_t i = 0u; i < 32u; ++i) {
                const uint32_t logical_col = block * 32u + i;
                qs[i] = logical_col < (uint32_t)LOGICAL_IN_FEATURES ?
                            (int8_t)(((int)(out_col * 13u + block * 7u + i * 5u) % 17) - 8) :
                            (int8_t)(77 + (int)i); /* physical padding must be ignored */
            }
        }
    }
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        bias[i] = bias_values[(i * 7u + 1u) % bias_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
            float sum = f16_bits_to_f32(bias[col]);
            for (uint32_t k = 0u; k < (uint32_t)LOGICAL_IN_FEATURES; ++k) {
                sum += f16_bits_to_f32(input[row * (uint32_t)LOGICAL_IN_FEATURES + k]) *
                       q8_0_expected_value(weight, WEIGHT_ROW_SIZE, col, k);
            }
            expected[row * (uint32_t)OUT_FEATURES + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_q8_0(ctx,
                                        input,
                                        weight,
                                        bias,
                                        ROWS,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * OUT_FEATURES); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 5.0e-3f);
    }

    memset(out_f16, 0, (size_t)OUT_FEATURES * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_q8_0(ctx,
                                        input,
                                        weight,
                                        bias,
                                        1u,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                        out_f16,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 7.5e-2f);
    }

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_q8_0(ctx,
                                        input,
                                        weight,
                                        NULL,
                                        1u,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
        CHECK(fabsf(out_f32[col] - (expected[col] - f16_bits_to_f32(bias[col]))) < 5.0e-3f);
    }

    CHECK(uocr_metal_context_dense_q8_0(ctx,
                                        input,
                                        weight,
                                        bias,
                                        ROWS,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        (uocr_metal_dense_output_type)99,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal Q8_0 dense output type") != NULL);
    CHECK(uocr_metal_context_dense_q8_0(ctx,
                                        input,
                                        weight,
                                        bias,
                                        ROWS,
                                        LOGICAL_IN_FEATURES,
                                        1279u,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 0);
    CHECK(strstr(error, "widths") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_dense_q4_k(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, LOGICAL_IN_FEATURES = 1277, PHYSICAL_IN_FEATURES = 1280, OUT_FEATURES = 64 };
    enum { WEIGHT_ROW_SIZE = (PHYSICAL_IN_FEATURES / UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE };
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t bias_values[] = {
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0xac00u, /* -0.0625 */
        0x3000u  /* 0.125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t bias_value_count = (uint32_t)(sizeof(bias_values) / sizeof(bias_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)ROWS * LOGICAL_IN_FEATURES * sizeof(uint16_t));
    uint8_t *weight = (uint8_t *)malloc((size_t)OUT_FEATURES * WEIGHT_ROW_SIZE);
    uint16_t *bias = (uint16_t *)malloc((size_t)OUT_FEATURES * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)OUT_FEATURES * sizeof(uint16_t));
    CHECK(input != NULL && weight != NULL && bias != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(ROWS * LOGICAL_IN_FEATURES); ++i) {
        input[i] = input_values[(i * 11u + i / 17u + 3u) % input_value_count];
    }
    init_q4_k_rows(weight, (uint32_t)OUT_FEATURES, (uint32_t)WEIGHT_ROW_SIZE, (uint32_t)PHYSICAL_IN_FEATURES, 0x2400u);
    for (uint32_t out_col = 0u; out_col < (uint32_t)OUT_FEATURES; ++out_col) {
        const uint32_t c0 = (out_col * 37u + 11u) % (uint32_t)LOGICAL_IN_FEATURES;
        uint32_t c1 = (out_col * 53u + 127u) % (uint32_t)LOGICAL_IN_FEATURES;
        uint32_t c2 = (out_col * 71u + 509u) % (uint32_t)LOGICAL_IN_FEATURES;
        if (c1 == c0) c1 = (c1 + 1u) % (uint32_t)LOGICAL_IN_FEATURES;
        if (c2 == c0 || c2 == c1) c2 = (c2 + 3u) % (uint32_t)LOGICAL_IN_FEATURES;
        set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, c0, (uint8_t)(1u + (out_col % 7u)));
        set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, c1, (uint8_t)(2u + ((out_col * 3u) % 6u)));
        set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, c2, (uint8_t)(3u + ((out_col * 5u) % 5u)));
        set_q4_k_q(weight,
                   (uint32_t)WEIGHT_ROW_SIZE,
                   out_col,
                   (uint32_t)LOGICAL_IN_FEATURES + 2u,
                   15u); /* physical padding must be ignored */
    }
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        bias[i] = bias_values[(i * 5u + 1u) % bias_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
            float sum = f16_bits_to_f32(bias[col]);
            for (uint32_t k = 0u; k < (uint32_t)LOGICAL_IN_FEATURES; ++k) {
                sum += f16_bits_to_f32(input[row * (uint32_t)LOGICAL_IN_FEATURES + k]) *
                       q4_k_expected_value(weight, (uint32_t)WEIGHT_ROW_SIZE, col, k);
            }
            expected[row * (uint32_t)OUT_FEATURES + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_q4_k(ctx,
                                        input,
                                        weight,
                                        bias,
                                        ROWS,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * OUT_FEATURES); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 5.0e-5f);
    }

    memset(out_f16, 0, (size_t)OUT_FEATURES * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_q4_k(ctx,
                                        input,
                                        weight,
                                        bias,
                                        1u,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                        out_f16,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 7.5e-4f);
    }

    memset(out_f32, 0, (size_t)OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_q4_k(ctx,
                                        input,
                                        weight,
                                        NULL,
                                        1u,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
        CHECK(fabsf(out_f32[col] - (expected[col] - f16_bits_to_f32(bias[col]))) < 5.0e-5f);
    }

    CHECK(uocr_metal_context_dense_q4_k(ctx,
                                        input,
                                        weight,
                                        bias,
                                        ROWS,
                                        LOGICAL_IN_FEATURES,
                                        1279u,
                                        OUT_FEATURES,
                                        UOCR_METAL_DENSE_OUTPUT_F32,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal Q4_K dense widths") != NULL);

    CHECK(uocr_metal_context_dense_q4_k(ctx,
                                        input,
                                        weight,
                                        bias,
                                        ROWS,
                                        LOGICAL_IN_FEATURES,
                                        PHYSICAL_IN_FEATURES,
                                        OUT_FEATURES,
                                        (uocr_metal_dense_output_type)99,
                                        out_f32,
                                        error,
                                        sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal Q4_K dense output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_quantized_dense_dot_edges(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    {
        enum { ROWS = 2, LOGICAL_IN = 33, PHYSICAL_IN = 64, OUT_FEATURES = 3 };
        enum { WEIGHT_ROW_SIZE = (PHYSICAL_IN / 32) * 34 };
        uint16_t input[ROWS * LOGICAL_IN];
        uint8_t weight[OUT_FEATURES * WEIGHT_ROW_SIZE];
        uint16_t bias[OUT_FEATURES] = {
            0x3000u, /* 0.125 */
            0xb000u, /* -0.125 */
            0x0000u  /* 0.0 */
        };
        float expected[ROWS * OUT_FEATURES];
        float out[ROWS * OUT_FEATURES];
        memset(input, 0, sizeof(input));
        memset(weight, 0, sizeof(weight));

        input[0u * (uint32_t)LOGICAL_IN + 0u] = 0x3c00u;  /* 1.0 */
        input[0u * (uint32_t)LOGICAL_IN + 31u] = 0xb800u; /* -0.5 */
        input[0u * (uint32_t)LOGICAL_IN + 32u] = 0x3400u; /* 0.25 */
        input[1u * (uint32_t)LOGICAL_IN + 0u] = 0xbc00u;  /* -1.0 */
        input[1u * (uint32_t)LOGICAL_IN + 31u] = 0x3800u; /* 0.5 */
        input[1u * (uint32_t)LOGICAL_IN + 32u] = 0x4000u; /* 2.0 */

        for (uint32_t out_col = 0u; out_col < (uint32_t)OUT_FEATURES; ++out_col) {
            uint8_t *row_base = weight + (size_t)out_col * WEIGHT_ROW_SIZE;
            write_q8_scale_le(row_base, f32_to_f16_bits(0.03125f * (float)(out_col + 1u)));
            write_q8_scale_le(row_base + 34u, f32_to_f16_bits(0.0625f * (float)(out_col + 1u)));
            set_q8_0_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 0u, (int8_t)-128);
            set_q8_0_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 31u, (int8_t)127);
            set_q8_0_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 32u, (int8_t)(-7 + (int)out_col));
            set_q8_0_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 33u, (int8_t)99); /* physical padding */
        }
        for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
            for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
                float sum = f16_bits_to_f32(bias[col]);
                for (uint32_t k = 0u; k < (uint32_t)LOGICAL_IN; ++k) {
                    sum += f16_bits_to_f32(input[row * (uint32_t)LOGICAL_IN + k]) *
                           q8_0_expected_value(weight, (uint32_t)WEIGHT_ROW_SIZE, col, k);
                }
                expected[row * (uint32_t)OUT_FEATURES + col] = sum;
            }
        }

        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_dense_q8_0(ctx,
                                            input,
                                            weight,
                                            bias,
                                            ROWS,
                                            LOGICAL_IN,
                                            PHYSICAL_IN,
                                            OUT_FEATURES,
                                            UOCR_METAL_DENSE_OUTPUT_F32,
                                            out,
                                            error,
                                            sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t i = 0u; i < (uint32_t)(ROWS * OUT_FEATURES); ++i) {
            CHECK(fabsf(out[i] - expected[i]) < 1.0e-5f);
        }
    }

    {
        enum { ROWS = 2, LOGICAL_IN = 145, PHYSICAL_IN = 256, OUT_FEATURES = 3 };
        enum { WEIGHT_ROW_SIZE = (PHYSICAL_IN / UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE };
        const uint32_t selected_cols[] = {0u, 31u, 64u, 128u, 144u};
        const uint16_t selected_values[ROWS][5] = {
            {0x3c00u, 0xb800u, 0x3400u, 0x4000u, 0xbc00u}, /* 1, -0.5, 0.25, 2, -1 */
            {0xbc00u, 0x3800u, 0xb400u, 0x3c00u, 0x3400u}  /* -1, 0.5, -0.25, 1, 0.25 */
        };
        uint16_t input[ROWS * LOGICAL_IN];
        uint8_t weight[OUT_FEATURES * WEIGHT_ROW_SIZE];
        float expected[ROWS * OUT_FEATURES];
        float out[ROWS * OUT_FEATURES];
        memset(input, 0, sizeof(input));
        init_q4_k_rows(weight, (uint32_t)OUT_FEATURES, (uint32_t)WEIGHT_ROW_SIZE, (uint32_t)PHYSICAL_IN, f32_to_f16_bits(0.03125f));

        for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
            for (uint32_t i = 0u; i < (uint32_t)(sizeof(selected_cols) / sizeof(selected_cols[0])); ++i) {
                input[row * (uint32_t)LOGICAL_IN + selected_cols[i]] = selected_values[row][i];
            }
        }
        for (uint32_t out_col = 0u; out_col < (uint32_t)OUT_FEATURES; ++out_col) {
            uint8_t *row_base = weight + (size_t)out_col * WEIGHT_ROW_SIZE;
            write_q8_scale_le(row_base + 2u, f32_to_f16_bits(0.015625f * (float)(out_col + 1u)));
            set_q4_k_scale_min(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 0u, 0u, (uint8_t)(3u + out_col), 2u);
            set_q4_k_scale_min(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 0u, 2u, (uint8_t)(17u + out_col), 5u);
            set_q4_k_scale_min(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 0u, 4u, (uint8_t)(49u + out_col), 37u);
            set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 0u, (uint8_t)(15u - out_col));
            set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 31u, 0u); /* min-only value */
            set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 64u, (uint8_t)(7u + out_col));
            set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 128u, (uint8_t)(11u - out_col));
            set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 144u, (uint8_t)(4u + out_col));
            set_q4_k_q(weight, (uint32_t)WEIGHT_ROW_SIZE, out_col, 145u, 15u); /* physical padding */
        }
        for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
            for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
                float sum = 0.0f;
                for (uint32_t k = 0u; k < (uint32_t)LOGICAL_IN; ++k) {
                    sum += f16_bits_to_f32(input[row * (uint32_t)LOGICAL_IN + k]) *
                           q4_k_expected_value(weight, (uint32_t)WEIGHT_ROW_SIZE, col, k);
                }
                expected[row * (uint32_t)OUT_FEATURES + col] = sum;
            }
        }

        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_dense_q4_k(ctx,
                                            input,
                                            weight,
                                            NULL,
                                            ROWS,
                                            LOGICAL_IN,
                                            PHYSICAL_IN,
                                            OUT_FEATURES,
                                            UOCR_METAL_DENSE_OUTPUT_F32,
                                            out,
                                            error,
                                            sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t i = 0u; i < (uint32_t)(ROWS * OUT_FEATURES); ++i) {
            CHECK(fabsf(out[i] - expected[i]) < 1.0e-5f);
        }
    }

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_attention_qkvo_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = 1280, PROJECTIONS = 4 };
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint16_t weight_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *weights[PROJECTIONS];
    float *expected[PROJECTIONS];
    float *out_f32[PROJECTIONS];
    uint16_t *out_f16[PROJECTIONS];
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        weights[p] = (uint16_t *)malloc((size_t)HIDDEN * HIDDEN * sizeof(uint16_t));
        expected[p] = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
        out_f32[p] = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
        out_f16[p] = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
        CHECK(weights[p] != NULL && expected[p] != NULL && out_f32[p] != NULL && out_f16[p] != NULL);
    }
    CHECK(input != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 19u + i / 23u + 2u) % input_value_count];
    }
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)(HIDDEN * HIDDEN); ++i) {
            weights[p][i] = weight_values[(i * (7u + p * 2u) + i / 41u + p * 3u) % weight_value_count];
        }
    }

    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                float sum = 0.0f;
                for (uint32_t k = 0u; k < (uint32_t)HIDDEN; ++k) {
                    sum += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k]) *
                           f16_bits_to_f32(weights[p][col * (uint32_t)HIDDEN + k]);
                }
                expected[p][token * (uint32_t)HIDDEN + col] = sum;
            }
        }
        memset(out_f32[p], 0, (size_t)TOKENS * HIDDEN * sizeof(float));
        memset(out_f16[p], 0, (size_t)HIDDEN * sizeof(uint16_t));
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                TOKENS,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
            CHECK(fabsf(out_f32[p][i] - expected[p][i]) < 4.0e-3f);
        }
    }

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                1u,
                                                UOCR_METAL_DENSE_OUTPUT_F16,
                                                out_f16[0],
                                                out_f16[1],
                                                out_f16[2],
                                                out_f16[3],
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
            CHECK(fabsf(f16_bits_to_f32(out_f16[p][i]) - expected[p][i]) < 1.5e-1f);
        }
    }

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                0u,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal attention projection request") != NULL);
    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                TOKENS,
                                                (uocr_metal_dense_output_type)99,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal attention projection output type") != NULL);

    uocr_metal_context_destroy(ctx);
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        free(out_f16[p]);
        free(out_f32[p]);
        free(expected[p]);
        free(weights[p]);
    }
    free(input);
    return 0;
}

static int test_metal_attention_output_residual_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = 1280 };
    const uint16_t context_values[] = {
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0xa400u, /* -0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint16_t weight_values[] = {
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0xa400u, /* -0.015625 */
        0x2800u, /* 0.03125 */
        0xa800u  /* -0.03125 */
    };
    const uint16_t residual_values[] = {
        0xb800u, /* -0.5 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint32_t context_value_count = (uint32_t)(sizeof(context_values) / sizeof(context_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *context = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)malloc((size_t)HIDDEN * HIDDEN * sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(context != NULL && weight != NULL && residual != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        context[i] = context_values[(i * 11u + i / 17u + 3u) % context_value_count];
        residual[i] = residual_values[(i * 7u + i / 19u + 1u) % residual_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)(HIDDEN * HIDDEN); ++i) {
        weight[i] = weight_values[(i * 13u + i / 23u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            float sum = f16_bits_to_f32(residual[token * (uint32_t)HIDDEN + col]);
            for (uint32_t k = 0u; k < (uint32_t)HIDDEN; ++k) {
                sum += f16_bits_to_f32(context[token * (uint32_t)HIDDEN + k]) *
                       f16_bits_to_f32(weight[col * (uint32_t)HIDDEN + k]);
            }
            expected[token * (uint32_t)HIDDEN + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           TOKENS,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.0e-4f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           1u,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           out_f16,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 8.0e-3f);
    }

    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           TOKENS,
                                                           (uocr_metal_dense_output_type)99,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal attention output type") != NULL);
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           0u,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal attention output request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(residual);
    free(weight);
    free(context);
    return 0;
}

static int test_metal_dense_swiglu_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_DENSE_LAYER0_INTERMEDIATE };
    const uint16_t input_values[] = {
        0xa800u, /* -0.03125 */
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint16_t weight_values[] = {
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0xa000u, /* -0.0078125 */
        0x2400u, /* 0.015625 */
        0xa400u  /* -0.015625 */
    };
    const uint16_t residual_values[] = {
        0xa800u, /* -0.03125 */
        0xa400u, /* -0.015625 */
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOKENS * INTERMEDIATE * sizeof(float));
    float *expected_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_no_residual = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && residual != NULL &&
          mid != NULL && expected_residual != NULL && expected_no_residual != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 11u + i / 29u + 2u) % input_value_count];
        residual[i] = residual_values[(i * 7u + i / 17u + 1u) % residual_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
        const uint32_t k0 = (row * 37u) % (uint32_t)HIDDEN;
        const uint32_t k1 = (row * 37u + 113u) % (uint32_t)HIDDEN;
        const uint32_t u0 = (row * 53u + 7u) % (uint32_t)HIDDEN;
        const uint32_t u1 = (row * 53u + 211u) % (uint32_t)HIDDEN;
        gate_weight[row * (uint32_t)HIDDEN + k0] = weight_values[(row + 1u) % weight_value_count];
        gate_weight[row * (uint32_t)HIDDEN + k1] = weight_values[(row * 3u + 2u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u0] = weight_values[(row * 5u + 3u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u1] = weight_values[(row * 7u + 4u) % weight_value_count];
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const uint32_t d0 = (col * 41u) % (uint32_t)INTERMEDIATE;
        const uint32_t d1 = (col * 41u + 977u) % (uint32_t)INTERMEDIATE;
        const uint32_t d2 = (col * 41u + 2222u) % (uint32_t)INTERMEDIATE;
        down_weight[col * (uint32_t)INTERMEDIATE + d0] = weight_values[(col + 2u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d1] = weight_values[(col * 3u + 1u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d2] = weight_values[(col * 5u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            float gate = 0.0f;
            float up = 0.0f;
            const uint32_t k0 = (row * 37u) % (uint32_t)HIDDEN;
            const uint32_t k1 = (row * 37u + 113u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (row * 53u + 7u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (row * 53u + 211u) % (uint32_t)HIDDEN;
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k0]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + k0]);
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k1]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + k1]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u0]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u0]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u1]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u1]);
            const float silu = gate / (1.0f + expf(-gate));
            mid[token * (uint32_t)INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const uint32_t d0 = (col * 41u) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (col * 41u + 977u) % (uint32_t)INTERMEDIATE;
            const uint32_t d2 = (col * 41u + 2222u) % (uint32_t)INTERMEDIATE;
            float sum = 0.0f;
            sum += mid[token * (uint32_t)INTERMEDIATE + d0] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d0]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d1] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d1]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d2] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d2]);
            if (token == 0u) {
                expected_no_residual[col] = sum;
            }
            expected_residual[token * (uint32_t)HIDDEN + col] =
                sum + f16_bits_to_f32(residual[token * (uint32_t)HIDDEN + col]);
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              TOKENS,
                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_residual[i]) < 2.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              NULL,
                                              1u,
                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                              out_f16,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_no_residual[i]) < 2.0e-5f);
    }

    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              TOKENS,
                                              (uocr_metal_dense_output_type)99,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal dense SwiGLU output type") != NULL);
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              0u,
                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal dense SwiGLU request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected_no_residual);
    free(expected_residual);
    free(mid);
    free(residual);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_shared_experts_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_SHARED_INTERMEDIATE };
    const uint16_t input_values[] = {
        0xb400u, /* -0.25 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t weight_values[] = {
        0x2800u, /* 0.03125 */
        0xa800u, /* -0.03125 */
        0x2c00u, /* 0.0625 */
        0xac00u, /* -0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOKENS * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && mid != NULL &&
          expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 13u + i / 19u + 4u) % input_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
        const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
        const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
        const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
        const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
        gate_weight[row * (uint32_t)HIDDEN + g0] = weight_values[(row + 1u) % weight_value_count];
        gate_weight[row * (uint32_t)HIDDEN + g1] = weight_values[(row * 3u + 2u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u0] = weight_values[(row * 5u + 3u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u1] = weight_values[(row * 7u + 4u) % weight_value_count];
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
        const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
        const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
        down_weight[col * (uint32_t)INTERMEDIATE + d0] = weight_values[(col + 2u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d1] = weight_values[(col * 3u + 1u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d2] = weight_values[(col * 5u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
            const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
            float gate = 0.0f;
            float up = 0.0f;
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g0]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + g0]);
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g1]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + g1]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u0]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u0]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u1]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u1]);
            const float silu = gate / (1.0f + expf(-gate));
            mid[token * (uint32_t)INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
            const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
            float sum = 0.0f;
            sum += mid[token * (uint32_t)INTERMEDIATE + d0] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d0]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d1] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d1]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d2] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d2]);
            expected[token * (uint32_t)HIDDEN + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    TOKENS,
                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    1u,
                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                    out_f16,
                                                    error,
                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 3.0e-4f);
    }

    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    TOKENS,
                                                    (uocr_metal_dense_output_type)99,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE shared experts output type") != NULL);
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    0u,
                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE shared experts request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(mid);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_shared_experts_q8_0(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_SHARED_INTERMEDIATE };
    enum { PHYSICAL_HIDDEN = HIDDEN + 32, PHYSICAL_INTERMEDIATE = INTERMEDIATE + 32 };
    enum { GATE_ROW_SIZE = (PHYSICAL_HIDDEN / 32) * 34, DOWN_ROW_SIZE = (PHYSICAL_INTERMEDIATE / 32) * 34 };
    const uint16_t input_values[] = {
        0xb400u, /* -0.25 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0x3400u  /* 0.25 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint8_t *gate_weight = (uint8_t *)calloc((size_t)INTERMEDIATE, (size_t)GATE_ROW_SIZE);
    uint8_t *up_weight = (uint8_t *)calloc((size_t)INTERMEDIATE, (size_t)GATE_ROW_SIZE);
    uint8_t *down_weight = (uint8_t *)calloc((size_t)HIDDEN, (size_t)DOWN_ROW_SIZE);
    float *mid = (float *)malloc((size_t)TOKENS * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && mid != NULL &&
          expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 17u + i / 23u + 3u) % input_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
        for (uint32_t block = 0u; block < (uint32_t)(PHYSICAL_HIDDEN / 32); ++block) {
            const uint16_t gate_scale = f32_to_f16_bits(0.03125f * (float)(1u + ((row + block) & 1u)));
            const uint16_t up_scale = f32_to_f16_bits(0.015625f * (float)(1u + ((row + block + 1u) % 3u)));
            write_q8_scale_le(gate_weight + (size_t)row * GATE_ROW_SIZE + (size_t)block * 34u, gate_scale);
            write_q8_scale_le(up_weight + (size_t)row * GATE_ROW_SIZE + (size_t)block * 34u, up_scale);
        }
        const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
        const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
        const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
        const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
        set_q8_0_q(gate_weight, GATE_ROW_SIZE, row, g0, (int8_t)((int)(row % 13u) - 6));
        set_q8_0_q(gate_weight, GATE_ROW_SIZE, row, g1, (int8_t)(5 - (int)(row % 11u)));
        set_q8_0_q(up_weight, GATE_ROW_SIZE, row, u0, (int8_t)((int)(row % 9u) - 4));
        set_q8_0_q(up_weight, GATE_ROW_SIZE, row, u1, (int8_t)(4 - (int)(row % 7u)));
        set_q8_0_q(gate_weight, GATE_ROW_SIZE, row, HIDDEN + 5u, 99); /* physical padding */
        set_q8_0_q(up_weight, GATE_ROW_SIZE, row, HIDDEN + 11u, -99); /* physical padding */
    }

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        for (uint32_t block = 0u; block < (uint32_t)(PHYSICAL_INTERMEDIATE / 32); ++block) {
            const uint16_t scale = f32_to_f16_bits(0.015625f * (float)(1u + ((col + block) % 4u)));
            write_q8_scale_le(down_weight + (size_t)col * DOWN_ROW_SIZE + (size_t)block * 34u, scale);
        }
        const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
        const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
        const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
        set_q8_0_q(down_weight, DOWN_ROW_SIZE, col, d0, (int8_t)((int)(col % 15u) - 7));
        set_q8_0_q(down_weight, DOWN_ROW_SIZE, col, d1, (int8_t)(6 - (int)(col % 13u)));
        set_q8_0_q(down_weight, DOWN_ROW_SIZE, col, d2, (int8_t)((int)(col % 11u) - 5));
        set_q8_0_q(down_weight, DOWN_ROW_SIZE, col, INTERMEDIATE + 3u, 88); /* physical padding */
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
            const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
            float gate = 0.0f;
            float up = 0.0f;
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g0]) *
                    q8_0_expected_value(gate_weight, GATE_ROW_SIZE, row, g0);
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g1]) *
                    q8_0_expected_value(gate_weight, GATE_ROW_SIZE, row, g1);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u0]) *
                  q8_0_expected_value(up_weight, GATE_ROW_SIZE, row, u0);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u1]) *
                  q8_0_expected_value(up_weight, GATE_ROW_SIZE, row, u1);
            const float silu = gate / (1.0f + expf(-gate));
            mid[token * (uint32_t)INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
            const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
            float sum = 0.0f;
            sum += mid[token * (uint32_t)INTERMEDIATE + d0] * q8_0_expected_value(down_weight, DOWN_ROW_SIZE, col, d0);
            sum += mid[token * (uint32_t)INTERMEDIATE + d1] * q8_0_expected_value(down_weight, DOWN_ROW_SIZE, col, d1);
            sum += mid[token * (uint32_t)INTERMEDIATE + d2] * q8_0_expected_value(down_weight, DOWN_ROW_SIZE, col, d2);
            expected[token * (uint32_t)HIDDEN + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_shared_experts_q8_0(ctx,
                                                     input,
                                                     gate_weight,
                                                     up_weight,
                                                     down_weight,
                                                     PHYSICAL_HIDDEN,
                                                     PHYSICAL_INTERMEDIATE,
                                                     TOKENS,
                                                     UOCR_METAL_DENSE_OUTPUT_F32,
                                                     out_f32,
                                                     error,
                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-4f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_shared_experts_q8_0(ctx,
                                                     input,
                                                     gate_weight,
                                                     up_weight,
                                                     down_weight,
                                                     PHYSICAL_HIDDEN,
                                                     PHYSICAL_INTERMEDIATE,
                                                     1u,
                                                     UOCR_METAL_DENSE_OUTPUT_F16,
                                                     out_f16,
                                                     error,
                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 3.0e-4f);
    }

    CHECK(uocr_metal_context_moe_shared_experts_q8_0(ctx,
                                                     input,
                                                     gate_weight,
                                                     up_weight,
                                                     down_weight,
                                                     PHYSICAL_HIDDEN,
                                                     PHYSICAL_INTERMEDIATE,
                                                     TOKENS,
                                                     (uocr_metal_dense_output_type)99,
                                                     out_f32,
                                                     error,
                                                     sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE shared Q8_0 output type") != NULL);
    CHECK(uocr_metal_context_moe_shared_experts_q8_0(ctx,
                                                     input,
                                                     gate_weight,
                                                     up_weight,
                                                     down_weight,
                                                     1279u,
                                                     PHYSICAL_INTERMEDIATE,
                                                     TOKENS,
                                                     UOCR_METAL_DENSE_OUTPUT_F32,
                                                     out_f32,
                                                     error,
                                                     sizeof(error)) == 0);
    CHECK(strstr(error, "widths") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(mid);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static void compute_router_topk_expected(const float *probs,
                                          uint32_t token,
                                          uint32_t experts,
                                          uint32_t top_k,
                                          uint32_t *top_ids,
                                          float *top_weights) {
    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        float best = -INFINITY;
        uint32_t best_expert = 0u;
        for (uint32_t expert = 0u; expert < experts; ++expert) {
            int already_selected = 0;
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if (top_ids[token * top_k + prev] == expert) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected) {
                continue;
            }
            const float value = probs[token * experts + expert];
            if (value > best || (value == best && expert < best_expert)) {
                best = value;
                best_expert = expert;
            }
        }
        top_ids[token * top_k + rank] = best_expert;
        top_weights[token * top_k + rank] = best;
    }
}

static int test_metal_moe_router_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = UOCR_HIDDEN_SIZE, EXPERTS = UOCR_ROUTED_EXPERTS, TOP_K = UOCR_MOE_TOP_K };
    uint16_t *input = (uint16_t *)calloc((size_t)TOKENS * HIDDEN, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc((size_t)EXPERTS * HIDDEN, sizeof(uint16_t));
    float *expected_logits = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    float *expected_probs = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    uint32_t *expected_top_ids = (uint32_t *)calloc((size_t)TOKENS * TOP_K, sizeof(uint32_t));
    float *expected_top_weights = (float *)calloc((size_t)TOKENS * TOP_K, sizeof(float));
    float *logits = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    float *probs = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    uint32_t *top_ids = (uint32_t *)malloc((size_t)TOKENS * TOP_K * sizeof(uint32_t));
    float *top_weights = (float *)malloc((size_t)TOKENS * TOP_K * sizeof(float));
    uint32_t *top_ids_optional = (uint32_t *)malloc((size_t)TOP_K * sizeof(uint32_t));
    float *top_weights_optional = (float *)malloc((size_t)TOP_K * sizeof(float));
    CHECK(input != NULL && weight != NULL && expected_logits != NULL && expected_probs != NULL &&
          expected_top_ids != NULL && expected_top_weights != NULL && logits != NULL && probs != NULL &&
          top_ids != NULL && top_weights != NULL && top_ids_optional != NULL && top_weights_optional != NULL);

    const float token_values[TOKENS][8] = {
        {0.50f, -0.25f, 0.125f, 1.00f, -0.50f, 0.25f, 0.75f, -0.125f},
        {-0.75f, 0.50f, -0.125f, 0.25f, 0.625f, -0.375f, 0.125f, 0.875f},
        {0.25f, 0.75f, 0.50f, -0.50f, 0.375f, 0.125f, -0.625f, 0.25f},
    };
    const uint32_t feature_cols[8] = {0u, 1u, 2u, 3u, 17u, 113u, 509u, 1021u};
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t i = 0u; i < 8u; ++i) {
            input[token * (uint32_t)HIDDEN + feature_cols[i]] = f32_to_f16_bits(token_values[token][i]);
        }
    }

    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        const float weights[8] = {
            ((float)expert - 31.5f) / 16.0f,
            ((float)((expert * 7u) % 17u) - 8.0f) / 32.0f,
            ((float)((expert * 11u) % 23u) - 11.0f) / 48.0f,
            ((float)((expert * 13u) % 29u) - 14.0f) / 64.0f,
            ((float)((expert * 5u) % 19u) - 9.0f) / 40.0f,
            ((float)((expert * 3u) % 31u) - 15.0f) / 80.0f,
            ((float)((expert * 17u) % 37u) - 18.0f) / 96.0f,
            ((float)((expert * 23u) % 41u) - 20.0f) / 112.0f,
        };
        for (uint32_t i = 0u; i < 8u; ++i) {
            weight[expert * (uint32_t)HIDDEN + feature_cols[i]] = f32_to_f16_bits(weights[i]);
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        float max_logit = -INFINITY;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            float sum = 0.0f;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                sum += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + col]) *
                       f16_bits_to_f32(weight[expert * (uint32_t)HIDDEN + col]);
            }
            expected_logits[token * (uint32_t)EXPERTS + expert] = sum;
            if (sum > max_logit) {
                max_logit = sum;
            }
        }
        float denom = 0.0f;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            const float value = expf(expected_logits[token * (uint32_t)EXPERTS + expert] - max_logit);
            expected_probs[token * (uint32_t)EXPERTS + expert] = value;
            denom += value;
        }
        float prob_sum = 0.0f;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            expected_probs[token * (uint32_t)EXPERTS + expert] /= denom;
            prob_sum += expected_probs[token * (uint32_t)EXPERTS + expert];
        }
        CHECK(fabsf(prob_sum - 1.0f) < 2.0e-6f);
        compute_router_topk_expected(expected_probs,
                                     token,
                                     (uint32_t)EXPERTS,
                                     (uint32_t)TOP_K,
                                     expected_top_ids,
                                     expected_top_weights);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(logits, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(probs, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(top_ids, 0xff, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * EXPERTS); ++i) {
        CHECK(fabsf(logits[i] - expected_logits[i]) < 2.0e-6f);
        CHECK(fabsf(probs[i] - expected_probs[i]) < 2.0e-6f);
    }
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * TOP_K); ++i) {
        CHECK(top_ids[i] == expected_top_ids[i]);
        CHECK(fabsf(top_weights[i] - expected_top_weights[i]) < 2.0e-6f);
    }

    memset(top_ids_optional, 0xff, (size_t)TOP_K * sizeof(uint32_t));
    memset(top_weights_optional, 0, (size_t)TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            1u,
                                            NULL,
                                            NULL,
                                            top_ids_optional,
                                            top_weights_optional,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)TOP_K; ++i) {
        CHECK(top_ids_optional[i] == expected_top_ids[i]);
        CHECK(fabsf(top_weights_optional[i] - expected_top_weights[i]) < 2.0e-6f);
    }

    /* Regression guard for the OCR routing contract.  DS4 applies
     * sqrt(softplus(logits)) plus selected-top-k renormalization/scaling in
     * its router path; Unlimited-OCR must keep raw softmax probabilities over
     * all 64 experts and pass those unrenormalized probabilities to the
     * selected experts.
     */
    memset(input, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    memset(weight, 0, (size_t)EXPERTS * HIDDEN * sizeof(uint16_t));
    input[0] = f32_to_f16_bits(1.0f);
    float max_contract_logit = -INFINITY;
    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        const float desired = ((float)expert - 31.5f) * 0.125f;
        weight[expert * (uint32_t)HIDDEN] = f32_to_f16_bits(desired);
        const float rounded = f16_bits_to_f32(weight[expert * (uint32_t)HIDDEN]);
        expected_logits[expert] = rounded;
        if (rounded > max_contract_logit) {
            max_contract_logit = rounded;
        }
    }
    float contract_denom = 0.0f;
    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        const float value = expf(expected_logits[expert] - max_contract_logit);
        expected_probs[expert] = value;
        contract_denom += value;
    }
    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        expected_probs[expert] /= contract_denom;
    }
    memset(expected_top_ids, 0, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(expected_top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    compute_router_topk_expected(expected_probs,
                                 0u,
                                 (uint32_t)EXPERTS,
                                 (uint32_t)TOP_K,
                                 expected_top_ids,
                                 expected_top_weights);
    memset(logits, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(probs, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(top_ids, 0xff, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            1u,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        CHECK(fabsf(logits[expert] - expected_logits[expert]) < 2.0e-6f);
        CHECK(fabsf(probs[expert] - expected_probs[expert]) < 2.0e-6f);
    }
    float selected_sum = 0.0f;
    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        CHECK(top_ids[rank] == expected_top_ids[rank]);
        CHECK(fabsf(top_weights[rank] - expected_top_weights[rank]) < 2.0e-6f);
        selected_sum += top_weights[rank];
    }
    CHECK(selected_sum > 0.40f && selected_sum < 0.75f);
    CHECK(fabsf(selected_sum - 1.0f) > 0.20f);
    CHECK(fabsf(selected_sum - 1.5f) > 0.50f);

    memset(weight, 0, (size_t)EXPERTS * HIDDEN * sizeof(uint16_t));
    memset(logits, 1, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(probs, 1, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(top_ids, 0xff, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * EXPERTS); ++i) {
        CHECK(fabsf(logits[i]) < 2.0e-7f);
        CHECK(fabsf(probs[i] - (1.0f / (float)EXPERTS)) < 2.0e-7f);
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            CHECK(top_ids[token * (uint32_t)TOP_K + rank] == rank);
            CHECK(fabsf(top_weights[token * (uint32_t)TOP_K + rank] - (1.0f / (float)EXPERTS)) < 2.0e-7f);
        }
    }

    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            NULL,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE router request") != NULL);
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            0u,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE router request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(top_weights_optional);
    free(top_ids_optional);
    free(top_weights);
    free(top_ids);
    free(probs);
    free(logits);
    free(expected_top_weights);
    free(expected_top_ids);
    free(expected_probs);
    free(expected_logits);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_moe_selected_experts_decode_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_EXPERT_INTERMEDIATE, TOP_K = UOCR_MOE_TOP_K };
    const uint32_t top_ids[TOP_K] = {7u, 2u, 63u, 0u, 5u, 11u};
    const float top_weights[TOP_K] = {0.375f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f};
    const uint16_t input_values[] = {
        0xb000u, /* -0.125 */
        0xa800u, /* -0.03125 */
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x3000u  /* 0.125 */
    };
    const uint16_t weight_values[] = {
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0xa000u, /* -0.0078125 */
        0xa400u, /* -0.015625 */
        0xa800u  /* -0.03125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)TOP_K * INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)TOP_K * INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)TOP_K * HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOP_K * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && mid != NULL &&
          expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        input[col] = input_values[(col * 7u + col / 23u + 3u) % input_value_count];
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            uint32_t g0 = (rank * 127u + row * 17u) % (uint32_t)HIDDEN;
            uint32_t g1 = (rank * 197u + row * 31u + 23u) % (uint32_t)HIDDEN;
            uint32_t u0 = (rank * 59u + row * 13u + 5u) % (uint32_t)HIDDEN;
            uint32_t u1 = (rank * 89u + row * 19u + 17u) % (uint32_t)HIDDEN;
            if (g1 == g0) g1 = (g1 + 1u) % (uint32_t)HIDDEN;
            if (u1 == u0) u1 = (u1 + 1u) % (uint32_t)HIDDEN;
            const size_t base = ((size_t)rank * INTERMEDIATE + row) * HIDDEN;
            gate_weight[base + g0] = weight_values[(rank + row + 1u) % weight_value_count];
            gate_weight[base + g1] = weight_values[(rank * 3u + row * 5u + 2u) % weight_value_count];
            up_weight[base + u0] = weight_values[(rank * 5u + row * 7u + 3u) % weight_value_count];
            up_weight[base + u1] = weight_values[(rank * 11u + row * 13u + 4u) % weight_value_count];
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint32_t d0 = (rank * 173u + col * 7u) % (uint32_t)INTERMEDIATE;
            uint32_t d1 = (rank * 181u + col * 11u + 37u) % (uint32_t)INTERMEDIATE;
            uint32_t d2 = (rank * 191u + col * 13u + 101u) % (uint32_t)INTERMEDIATE;
            if (d1 == d0) d1 = (d1 + 1u) % (uint32_t)INTERMEDIATE;
            if (d2 == d0 || d2 == d1) d2 = (d2 + 2u) % (uint32_t)INTERMEDIATE;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            down_weight[base + d0] = weight_values[(rank + col + 2u) % weight_value_count];
            down_weight[base + d1] = weight_values[(rank * 3u + col * 5u + 1u) % weight_value_count];
            down_weight[base + d2] = weight_values[(rank * 7u + col * 11u + 5u) % weight_value_count];
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            float gate = 0.0f;
            float up = 0.0f;
            const size_t base = ((size_t)rank * INTERMEDIATE + row) * HIDDEN;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[col]);
                gate += x * f16_bits_to_f32(gate_weight[base + col]);
                up += x * f16_bits_to_f32(up_weight[base + col]);
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid[(size_t)rank * INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        float routed_sum = 0.0f;
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            float expert_sum = 0.0f;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                expert_sum += mid[(size_t)rank * INTERMEDIATE + row] * f16_bits_to_f32(down_weight[base + row]);
            }
            routed_sum += expert_sum * top_weights[rank];
        }
        expected[col] = routed_sum;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(out_f32[col] - expected[col]) < 3.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             out_f16,
                                                             error,
                                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[col]) - expected[col]) < 3.0e-4f);
    }

    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             (uocr_metal_dense_output_type)99,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE selected-expert output type") != NULL);

    uint32_t bad_ids[TOP_K] = {7u, 2u, UOCR_ROUTED_EXPERTS, 0u, 5u, 11u};
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             bad_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected expert id") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(mid);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_selected_experts_decode_q4_k(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { HIDDEN = UOCR_HIDDEN_SIZE, PHYSICAL = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_EXPERT_INTERMEDIATE };
    enum { TOP_K = UOCR_MOE_TOP_K, ROW_SIZE = (PHYSICAL / UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE };
    enum { DOWN_PHYSICAL = 1024, DOWN_ROW_SIZE = (DOWN_PHYSICAL / UOCR_Q8_0_BLOCK_SIZE) * UOCR_Q8_0_TYPE_SIZE };
    enum { DOWN_Q4_PHYSICAL = 1024, DOWN_Q4_ROW_SIZE = (DOWN_Q4_PHYSICAL / UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE };
    const uint32_t top_ids[TOP_K] = {3u, 17u, 5u, 63u, 11u, 0u};
    const float top_weights[TOP_K] = {0.3125f, 0.25f, 0.1875f, 0.125f, 0.0625f, 0.03125f};
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint16_t down_values[] = {
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0xa400u, /* -0.015625 */
        0xa800u, /* -0.03125 */
        0x2c00u, /* 0.0625 */
        0xac00u  /* -0.0625 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t down_value_count = (uint32_t)(sizeof(down_values) / sizeof(down_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    uint8_t *gate_weight = (uint8_t *)malloc((size_t)TOP_K * INTERMEDIATE * ROW_SIZE);
    uint8_t *up_weight = (uint8_t *)malloc((size_t)TOP_K * INTERMEDIATE * ROW_SIZE);
    uint16_t *down_weight = (uint16_t *)calloc((size_t)TOP_K * HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    uint8_t *down_weight_q8 = (uint8_t *)malloc((size_t)TOP_K * HIDDEN * DOWN_ROW_SIZE);
    uint8_t *down_weight_q4 = (uint8_t *)malloc((size_t)TOP_K * HIDDEN * DOWN_Q4_ROW_SIZE);
    float *mid = (float *)malloc((size_t)TOP_K * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *expected_q8 = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *expected_q4 = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL &&
          down_weight_q8 != NULL && down_weight_q4 != NULL && mid != NULL && expected != NULL &&
          expected_q8 != NULL && expected_q4 != NULL && out_f32 != NULL && out_f16 != NULL);

    init_q4_k_rows(gate_weight, (uint32_t)(TOP_K * INTERMEDIATE), (uint32_t)ROW_SIZE, (uint32_t)PHYSICAL, 0x2c00u);
    init_q4_k_rows(up_weight, (uint32_t)(TOP_K * INTERMEDIATE), (uint32_t)ROW_SIZE, (uint32_t)PHYSICAL, 0x2c00u);
    memset(down_weight_q8, 0, (size_t)TOP_K * HIDDEN * DOWN_ROW_SIZE);
    init_q4_k_rows(down_weight_q4,
                   (uint32_t)(TOP_K * HIDDEN),
                   (uint32_t)DOWN_Q4_ROW_SIZE,
                   (uint32_t)DOWN_Q4_PHYSICAL,
                   0x2200u);
    for (uint32_t row = 0u; row < (uint32_t)(TOP_K * HIDDEN); ++row) {
        for (uint32_t block = 0u; block < (uint32_t)(DOWN_PHYSICAL / UOCR_Q8_0_BLOCK_SIZE); ++block) {
            write_q8_scale_le(down_weight_q8 + (size_t)row * DOWN_ROW_SIZE + (size_t)block * UOCR_Q8_0_TYPE_SIZE,
                              f32_to_f16_bits(0.015625f * (float)(1u + ((row + block) % 3u))));
        }
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        input[col] = input_values[(col * 5u + col / 31u + 1u) % input_value_count];
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const uint32_t gate_row = rank * (uint32_t)INTERMEDIATE + row;
            uint32_t g0 = (rank * 131u + row * 19u + 7u) % (uint32_t)HIDDEN;
            uint32_t g1 = (rank * 193u + row * 29u + 41u) % (uint32_t)HIDDEN;
            uint32_t u0 = (rank * 67u + row * 17u + 13u) % (uint32_t)HIDDEN;
            uint32_t u1 = (rank * 101u + row * 23u + 59u) % (uint32_t)HIDDEN;
            if (g1 == g0) g1 = (g1 + 1u) % (uint32_t)HIDDEN;
            if (u1 == u0) u1 = (u1 + 1u) % (uint32_t)HIDDEN;
            set_q4_k_q(gate_weight, (uint32_t)ROW_SIZE, gate_row, g0, (uint8_t)(1u + ((rank + row) % 7u)));
            set_q4_k_q(gate_weight, (uint32_t)ROW_SIZE, gate_row, g1, (uint8_t)(2u + ((rank * 3u + row) % 6u)));
            set_q4_k_q(up_weight, (uint32_t)ROW_SIZE, gate_row, u0, (uint8_t)(1u + ((rank * 5u + row) % 7u)));
            set_q4_k_q(up_weight, (uint32_t)ROW_SIZE, gate_row, u1, (uint8_t)(2u + ((rank + row * 3u) % 6u)));
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint32_t d0 = (rank * 173u + col * 7u) % (uint32_t)INTERMEDIATE;
            uint32_t d1 = (rank * 181u + col * 11u + 37u) % (uint32_t)INTERMEDIATE;
            uint32_t d2 = (rank * 191u + col * 13u + 101u) % (uint32_t)INTERMEDIATE;
            if (d1 == d0) d1 = (d1 + 1u) % (uint32_t)INTERMEDIATE;
            if (d2 == d0 || d2 == d1) d2 = (d2 + 2u) % (uint32_t)INTERMEDIATE;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            down_weight[base + d0] = down_values[(rank + col + 2u) % down_value_count];
            down_weight[base + d1] = down_values[(rank * 3u + col * 5u + 1u) % down_value_count];
            down_weight[base + d2] = down_values[(rank * 7u + col * 11u + 5u) % down_value_count];
            const uint32_t q8_row = rank * (uint32_t)HIDDEN + col;
            set_q8_0_q(down_weight_q8, (uint32_t)DOWN_ROW_SIZE, q8_row, d0, (int8_t)(1 + (int)((rank + col) % 5u)));
            set_q8_0_q(down_weight_q8, (uint32_t)DOWN_ROW_SIZE, q8_row, d1, (int8_t)(-2 - (int)((rank * 2u + col) % 4u)));
            set_q8_0_q(down_weight_q8, (uint32_t)DOWN_ROW_SIZE, q8_row, d2, (int8_t)(3 + (int)((rank * 3u + col) % 3u)));
            set_q8_0_q(down_weight_q8,
                       (uint32_t)DOWN_ROW_SIZE,
                       q8_row,
                       (uint32_t)INTERMEDIATE + 7u,
                       (int8_t)-101); /* physical padding must be ignored */
            set_q4_k_q(down_weight_q4,
                       (uint32_t)DOWN_Q4_ROW_SIZE,
                       q8_row,
                       d0,
                       (uint8_t)(1u + ((rank + col) % 8u)));
            set_q4_k_q(down_weight_q4,
                       (uint32_t)DOWN_Q4_ROW_SIZE,
                       q8_row,
                       d1,
                       (uint8_t)(2u + ((rank * 2u + col) % 7u)));
            set_q4_k_q(down_weight_q4,
                       (uint32_t)DOWN_Q4_ROW_SIZE,
                       q8_row,
                       d2,
                       (uint8_t)(3u + ((rank * 3u + col) % 6u)));
            set_q4_k_q(down_weight_q4,
                       (uint32_t)DOWN_Q4_ROW_SIZE,
                       q8_row,
                       (uint32_t)INTERMEDIATE + 13u,
                       15u); /* physical padding must be ignored */
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            float gate = 0.0f;
            float up = 0.0f;
            const uint32_t weight_row = rank * (uint32_t)INTERMEDIATE + row;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[col]);
                gate += x * q4_k_expected_value(gate_weight, (uint32_t)ROW_SIZE, weight_row, col);
                up += x * q4_k_expected_value(up_weight, (uint32_t)ROW_SIZE, weight_row, col);
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid[(size_t)rank * INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        float routed_sum = 0.0f;
        float routed_sum_q8 = 0.0f;
        float routed_sum_q4 = 0.0f;
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            float expert_sum = 0.0f;
            float expert_sum_q8 = 0.0f;
            float expert_sum_q4 = 0.0f;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            const uint32_t q8_row = rank * (uint32_t)HIDDEN + col;
            for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                expert_sum += mid[(size_t)rank * INTERMEDIATE + row] * f16_bits_to_f32(down_weight[base + row]);
                expert_sum_q8 += mid[(size_t)rank * INTERMEDIATE + row] *
                                 q8_0_expected_value(down_weight_q8, (uint32_t)DOWN_ROW_SIZE, q8_row, row);
                expert_sum_q4 += mid[(size_t)rank * INTERMEDIATE + row] *
                                 q4_k_expected_value(down_weight_q4, (uint32_t)DOWN_Q4_ROW_SIZE, q8_row, row);
            }
            routed_sum += expert_sum * top_weights[rank];
            routed_sum_q8 += expert_sum_q8 * top_weights[rank];
            routed_sum_q4 += expert_sum_q4 * top_weights[rank];
        }
        expected[col] = routed_sum;
        expected_q8[col] = routed_sum_q8;
        expected_q4[col] = routed_sum_q4;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              PHYSICAL,
                                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(out_f32[col] - expected[col]) < 5.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              PHYSICAL,
                                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                                              out_f16,
                                                              error,
                                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[col]) - expected[col]) < 7.0e-4f);
    }

    memset(out_f32, 0, (size_t)HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k_q8_0(ctx,
                                                                   input,
                                                                   top_ids,
                                                                   top_weights,
                                                                   gate_weight,
                                                                   up_weight,
                                                                   down_weight_q8,
                                                                   PHYSICAL,
                                                                   DOWN_PHYSICAL,
                                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                                   out_f32,
                                                                   error,
                                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(out_f32[col] - expected_q8[col]) < 7.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k_q8_0(ctx,
                                                                   input,
                                                                   top_ids,
                                                                   top_weights,
                                                                   gate_weight,
                                                                   up_weight,
                                                                   down_weight_q8,
                                                                   PHYSICAL,
                                                                   DOWN_PHYSICAL,
                                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                                   out_f16,
                                                                   error,
                                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[col]) - expected_q8[col]) < 8.0e-4f);
    }

    memset(out_f32, 0, (size_t)HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k_padded(ctx,
                                                                     input,
                                                                     top_ids,
                                                                     top_weights,
                                                                     gate_weight,
                                                                     up_weight,
                                                                     down_weight_q4,
                                                                     PHYSICAL,
                                                                     DOWN_Q4_PHYSICAL,
                                                                     UOCR_METAL_DENSE_OUTPUT_F32,
                                                                     out_f32,
                                                                     error,
                                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(out_f32[col] - expected_q4[col]) < 7.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k_padded(ctx,
                                                                     input,
                                                                     top_ids,
                                                                     top_weights,
                                                                     gate_weight,
                                                                     up_weight,
                                                                     down_weight_q4,
                                                                     PHYSICAL,
                                                                     DOWN_Q4_PHYSICAL,
                                                                     UOCR_METAL_DENSE_OUTPUT_F16,
                                                                     out_f16,
                                                                     error,
                                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[col]) - expected_q4[col]) < 8.0e-4f);
    }

    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k_padded(ctx,
                                                                     input,
                                                                     top_ids,
                                                                     top_weights,
                                                                     gate_weight,
                                                                     up_weight,
                                                                     down_weight_q4,
                                                                     PHYSICAL,
                                                                     900u,
                                                                     UOCR_METAL_DENSE_OUTPUT_F32,
                                                                     out_f32,
                                                                     error,
                                                                     sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected-expert Q4_K/padded-Q4_K decode down widths") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k_q8_0(ctx,
                                                                   input,
                                                                   top_ids,
                                                                   top_weights,
                                                                   gate_weight,
                                                                   up_weight,
                                                                   down_weight_q8,
                                                                   PHYSICAL,
                                                                   900u,
                                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                                   out_f32,
                                                                   error,
                                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected-expert Q4_K/Q8_0 decode down widths") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              1408u,
                                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected-expert Q4_K decode widths") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              PHYSICAL,
                                                              (uocr_metal_dense_output_type)99,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE selected-expert Q4_K output type") != NULL);

    uint32_t bad_ids[TOP_K] = {3u, 17u, UOCR_ROUTED_EXPERTS, 63u, 11u, 0u};
    CHECK(uocr_metal_context_moe_selected_experts_decode_q4_k(ctx,
                                                              input,
                                                              bad_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              PHYSICAL,
                                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected Q4_K expert id") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected_q4);
    free(expected_q8);
    free(expected);
    free(mid);
    free(down_weight_q4);
    free(down_weight_q8);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_selected_experts_prefill_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = 8, INTERMEDIATE = 5, EXPERTS = 4, TOP_K = 2 };
    uint16_t input[TOKENS * HIDDEN];
    uint32_t top_ids[TOKENS * TOP_K] = {
        0u, 2u,
        1u, 3u,
        2u, 0u,
    };
    float top_weights[TOKENS * TOP_K] = {
        0.625f, 0.25f,
        0.5f, 0.125f,
        0.375f, 0.3125f,
    };
    uint16_t gate_weight[EXPERTS * INTERMEDIATE * HIDDEN];
    uint16_t up_weight[EXPERTS * INTERMEDIATE * HIDDEN];
    uint16_t down_weight[EXPERTS * HIDDEN * INTERMEDIATE];
    float mid[TOKENS * TOP_K * INTERMEDIATE];
    float expected[TOKENS * HIDDEN];
    float out_f32[TOKENS * HIDDEN];
    uint16_t out_f16[TOKENS * HIDDEN];

    memset(gate_weight, 0, sizeof(gate_weight));
    memset(up_weight, 0, sizeof(up_weight));
    memset(down_weight, 0, sizeof(down_weight));
    const float input_values[] = {-0.5f, -0.25f, 0.0f, 0.125f, 0.25f, 0.5f, 0.75f};
    const float weight_values[] = {-0.375f, -0.125f, 0.0625f, 0.125f, 0.25f, 0.375f};
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = f32_to_f16_bits(input_values[(i * 5u + 2u) % (sizeof(input_values) / sizeof(input_values[0]))]);
    }

    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const size_t base = ((size_t)expert * INTERMEDIATE + row) * HIDDEN;
            const uint32_t g0 = (expert + row * 2u) % (uint32_t)HIDDEN;
            const uint32_t g1 = (expert * 3u + row + 1u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (expert * 5u + row + 2u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (expert + row * 3u + 4u) % (uint32_t)HIDDEN;
            gate_weight[base + g0] = f32_to_f16_bits(weight_values[(expert + row + 1u) % 6u]);
            gate_weight[base + g1] = f32_to_f16_bits(weight_values[(expert * 2u + row + 3u) % 6u]);
            up_weight[base + u0] = f32_to_f16_bits(weight_values[(expert * 3u + row + 2u) % 6u]);
            up_weight[base + u1] = f32_to_f16_bits(weight_values[(expert + row * 2u + 5u) % 6u]);
        }
    }
    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const size_t base = ((size_t)expert * HIDDEN + col) * INTERMEDIATE;
            const uint32_t d0 = (expert + col) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (expert * 2u + col + 1u) % (uint32_t)INTERMEDIATE;
            down_weight[base + d0] = f32_to_f16_bits(weight_values[(expert + col + 2u) % 6u]);
            down_weight[base + d1] = f32_to_f16_bits(weight_values[(expert * 3u + col + 4u) % 6u]);
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            const uint32_t expert = top_ids[token * (uint32_t)TOP_K + rank];
            for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                float gate = 0.0f;
                float up = 0.0f;
                const size_t weight_base = ((size_t)expert * INTERMEDIATE + row) * HIDDEN;
                const size_t input_base = (size_t)token * HIDDEN;
                for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                    const float x = f16_bits_to_f32(input[input_base + col]);
                    gate += x * f16_bits_to_f32(gate_weight[weight_base + col]);
                    up += x * f16_bits_to_f32(up_weight[weight_base + col]);
                }
                const float silu = gate / (1.0f + expf(-gate));
                mid[((size_t)token * TOP_K + rank) * INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
            }
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            float routed_sum = 0.0f;
            for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
                const uint32_t expert = top_ids[token * (uint32_t)TOP_K + rank];
                float expert_sum = 0.0f;
                const size_t mid_base = ((size_t)token * TOP_K + rank) * INTERMEDIATE;
                const size_t weight_base = ((size_t)expert * HIDDEN + col) * INTERMEDIATE;
                for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                    expert_sum += mid[mid_base + row] * f16_bits_to_f32(down_weight[weight_base + row]);
                }
                routed_sum += expert_sum * top_weights[token * (uint32_t)TOP_K + rank];
            }
            expected[token * (uint32_t)HIDDEN + col] = routed_sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.0e-5f);
    }

    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                                              out_f16,
                                                              error,
                                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 3.0e-4f);
    }

    uint32_t bad_ids[TOKENS * TOP_K];
    memcpy(bad_ids, top_ids, sizeof(bad_ids));
    bad_ids[3] = EXPERTS;
    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              bad_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE prefill expert id") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              (uocr_metal_dense_output_type)99,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE selected-expert prefill output type") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_moe_selected_experts_prefill_q4_k(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = 256, PHYSICAL = 256, INTERMEDIATE = 7, EXPERTS = 4, TOP_K = 2 };
    enum { ROW_SIZE = (PHYSICAL / UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE };
    enum { DOWN_PHYSICAL = 32, DOWN_ROW_SIZE = (DOWN_PHYSICAL / UOCR_Q8_0_BLOCK_SIZE) * UOCR_Q8_0_TYPE_SIZE };
    enum { DOWN_Q4_PHYSICAL = 256, DOWN_Q4_ROW_SIZE = (DOWN_Q4_PHYSICAL / UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE };
    uint16_t input[TOKENS * HIDDEN];
    uint32_t top_ids[TOKENS * TOP_K] = {
        0u, 2u,
        1u, 3u,
        2u, 0u,
    };
    float top_weights[TOKENS * TOP_K] = {
        0.625f, 0.25f,
        0.5f, 0.125f,
        0.375f, 0.3125f,
    };
    uint8_t gate_weight[EXPERTS * INTERMEDIATE * ROW_SIZE];
    uint8_t up_weight[EXPERTS * INTERMEDIATE * ROW_SIZE];
    uint16_t down_weight[EXPERTS * HIDDEN * INTERMEDIATE];
    uint8_t down_weight_q8[EXPERTS * HIDDEN * DOWN_ROW_SIZE];
    uint8_t down_weight_q4[EXPERTS * HIDDEN * DOWN_Q4_ROW_SIZE];
    float mid[TOKENS * TOP_K * INTERMEDIATE];
    float expected[TOKENS * HIDDEN];
    float expected_q8[TOKENS * HIDDEN];
    float expected_q4[TOKENS * HIDDEN];
    float out_f32[TOKENS * HIDDEN];
    uint16_t out_f16[TOKENS * HIDDEN];

    memset(down_weight, 0, sizeof(down_weight));
    memset(down_weight_q8, 0, sizeof(down_weight_q8));
    init_q4_k_rows(down_weight_q4,
                   (uint32_t)(EXPERTS * HIDDEN),
                   (uint32_t)DOWN_Q4_ROW_SIZE,
                   (uint32_t)DOWN_Q4_PHYSICAL,
                   0x2400u);
    init_q4_k_rows(gate_weight, (uint32_t)(EXPERTS * INTERMEDIATE), (uint32_t)ROW_SIZE, (uint32_t)PHYSICAL, 0x2800u);
    init_q4_k_rows(up_weight, (uint32_t)(EXPERTS * INTERMEDIATE), (uint32_t)ROW_SIZE, (uint32_t)PHYSICAL, 0x2800u);
    for (uint32_t row = 0u; row < (uint32_t)(EXPERTS * HIDDEN); ++row) {
        write_q8_scale_le(down_weight_q8 + (size_t)row * DOWN_ROW_SIZE,
                          f32_to_f16_bits(0.03125f * (float)(1u + (row % 2u))));
    }
    const float input_values[] = {-0.5f, -0.25f, 0.0f, 0.125f, 0.25f, 0.5f, 0.75f};
    const float down_values[] = {-0.375f, -0.125f, 0.0625f, 0.125f, 0.25f, 0.375f};
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = f32_to_f16_bits(input_values[(i * 5u + 2u) % (sizeof(input_values) / sizeof(input_values[0]))]);
    }

    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const uint32_t qrow = expert * (uint32_t)INTERMEDIATE + row;
            const uint32_t g0 = (expert * 17u + row * 11u) % (uint32_t)HIDDEN;
            const uint32_t g1 = (expert * 23u + row * 13u + 19u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (expert * 29u + row * 7u + 5u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (expert * 31u + row * 5u + 37u) % (uint32_t)HIDDEN;
            set_q4_k_q(gate_weight, (uint32_t)ROW_SIZE, qrow, g0, (uint8_t)(1u + ((expert + row) % 8u)));
            set_q4_k_q(gate_weight, (uint32_t)ROW_SIZE, qrow, g1, (uint8_t)(2u + ((expert * 3u + row) % 7u)));
            set_q4_k_q(up_weight, (uint32_t)ROW_SIZE, qrow, u0, (uint8_t)(1u + ((expert * 5u + row) % 8u)));
            set_q4_k_q(up_weight, (uint32_t)ROW_SIZE, qrow, u1, (uint8_t)(2u + ((expert + row * 3u) % 7u)));
        }
    }
    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const size_t base = ((size_t)expert * HIDDEN + col) * INTERMEDIATE;
            const uint32_t d0 = (expert + col) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (expert * 2u + col + 1u) % (uint32_t)INTERMEDIATE;
            down_weight[base + d0] = f32_to_f16_bits(down_values[(expert + col + 2u) % 6u]);
            down_weight[base + d1] = f32_to_f16_bits(down_values[(expert * 3u + col + 4u) % 6u]);
            const uint32_t q8_row = expert * (uint32_t)HIDDEN + col;
            set_q8_0_q(down_weight_q8, (uint32_t)DOWN_ROW_SIZE, q8_row, d0, (int8_t)(2 + (int)((expert + col) % 4u)));
            set_q8_0_q(down_weight_q8, (uint32_t)DOWN_ROW_SIZE, q8_row, d1, (int8_t)(-3 - (int)((expert * 2u + col) % 3u)));
            set_q8_0_q(down_weight_q8,
                       (uint32_t)DOWN_ROW_SIZE,
                       q8_row,
                       (uint32_t)INTERMEDIATE + 3u,
                       (int8_t)111); /* physical padding must be ignored */
            set_q4_k_q(down_weight_q4,
                       (uint32_t)DOWN_Q4_ROW_SIZE,
                       q8_row,
                       d0,
                       (uint8_t)(1u + ((expert + col) % 8u)));
            set_q4_k_q(down_weight_q4,
                       (uint32_t)DOWN_Q4_ROW_SIZE,
                       q8_row,
                       d1,
                       (uint8_t)(2u + ((expert * 2u + col) % 7u)));
            set_q4_k_q(down_weight_q4,
                       (uint32_t)DOWN_Q4_ROW_SIZE,
                       q8_row,
                       (uint32_t)INTERMEDIATE + 5u,
                       15u); /* physical padding must be ignored */
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            const uint32_t expert = top_ids[token * (uint32_t)TOP_K + rank];
            for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                float gate = 0.0f;
                float up = 0.0f;
                const uint32_t weight_row = expert * (uint32_t)INTERMEDIATE + row;
                const size_t input_base = (size_t)token * HIDDEN;
                for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                    const float x = f16_bits_to_f32(input[input_base + col]);
                    gate += x * q4_k_expected_value(gate_weight, (uint32_t)ROW_SIZE, weight_row, col);
                    up += x * q4_k_expected_value(up_weight, (uint32_t)ROW_SIZE, weight_row, col);
                }
                const float silu = gate / (1.0f + expf(-gate));
                mid[((size_t)token * TOP_K + rank) * INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
            }
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            float routed_sum = 0.0f;
            float routed_sum_q8 = 0.0f;
            float routed_sum_q4 = 0.0f;
            for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
                const uint32_t expert = top_ids[token * (uint32_t)TOP_K + rank];
                float expert_sum = 0.0f;
                float expert_sum_q8 = 0.0f;
                float expert_sum_q4 = 0.0f;
                const size_t mid_base = ((size_t)token * TOP_K + rank) * INTERMEDIATE;
                const size_t weight_base = ((size_t)expert * HIDDEN + col) * INTERMEDIATE;
                const uint32_t q8_row = expert * (uint32_t)HIDDEN + col;
                for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                    expert_sum += mid[mid_base + row] * f16_bits_to_f32(down_weight[weight_base + row]);
                    expert_sum_q8 += mid[mid_base + row] *
                                     q8_0_expected_value(down_weight_q8, (uint32_t)DOWN_ROW_SIZE, q8_row, row);
                    expert_sum_q4 += mid[mid_base + row] *
                                     q4_k_expected_value(down_weight_q4, (uint32_t)DOWN_Q4_ROW_SIZE, q8_row, row);
                }
                routed_sum += expert_sum * top_weights[token * (uint32_t)TOP_K + rank];
                routed_sum_q8 += expert_sum_q8 * top_weights[token * (uint32_t)TOP_K + rank];
                routed_sum_q4 += expert_sum_q4 * top_weights[token * (uint32_t)TOP_K + rank];
            }
            expected[token * (uint32_t)HIDDEN + col] = routed_sum;
            expected_q8[token * (uint32_t)HIDDEN + col] = routed_sum_q8;
            expected_q4[token * (uint32_t)HIDDEN + col] = routed_sum_q4;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k(ctx,
                                                               input,
                                                               top_ids,
                                                               top_weights,
                                                               gate_weight,
                                                               up_weight,
                                                               down_weight,
                                                               TOKENS,
                                                               HIDDEN,
                                                               PHYSICAL,
                                                               INTERMEDIATE,
                                                               EXPERTS,
                                                               TOP_K,
                                                               UOCR_METAL_DENSE_OUTPUT_F32,
                                                               out_f32,
                                                               error,
                                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 5.0e-5f);
    }

    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k(ctx,
                                                               input,
                                                               top_ids,
                                                               top_weights,
                                                               gate_weight,
                                                               up_weight,
                                                               down_weight,
                                                               TOKENS,
                                                               HIDDEN,
                                                               PHYSICAL,
                                                               INTERMEDIATE,
                                                               EXPERTS,
                                                               TOP_K,
                                                               UOCR_METAL_DENSE_OUTPUT_F16,
                                                               out_f16,
                                                               error,
                                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 6.0e-4f);
    }

    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k_q8_0(ctx,
                                                                    input,
                                                                    top_ids,
                                                                    top_weights,
                                                                    gate_weight,
                                                                    up_weight,
                                                                    down_weight_q8,
                                                                    TOKENS,
                                                                    HIDDEN,
                                                                    PHYSICAL,
                                                                    INTERMEDIATE,
                                                                    DOWN_PHYSICAL,
                                                                    EXPERTS,
                                                                    TOP_K,
                                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                                    out_f32,
                                                                    error,
                                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_q8[i]) < 5.0e-5f);
    }

    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k_q8_0(ctx,
                                                                    input,
                                                                    top_ids,
                                                                    top_weights,
                                                                    gate_weight,
                                                                    up_weight,
                                                                    down_weight_q8,
                                                                    TOKENS,
                                                                    HIDDEN,
                                                                    PHYSICAL,
                                                                    INTERMEDIATE,
                                                                    DOWN_PHYSICAL,
                                                                    EXPERTS,
                                                                    TOP_K,
                                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                                    out_f16,
                                                                    error,
                                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_q8[i]) < 6.0e-4f);
    }

    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k_padded(ctx,
                                                                      input,
                                                                      top_ids,
                                                                      top_weights,
                                                                      gate_weight,
                                                                      up_weight,
                                                                      down_weight_q4,
                                                                      TOKENS,
                                                                      HIDDEN,
                                                                      PHYSICAL,
                                                                      INTERMEDIATE,
                                                                      DOWN_Q4_PHYSICAL,
                                                                      EXPERTS,
                                                                      TOP_K,
                                                                      UOCR_METAL_DENSE_OUTPUT_F32,
                                                                      out_f32,
                                                                      error,
                                                                      sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_q4[i]) < 5.0e-5f);
    }

    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k_padded(ctx,
                                                                      input,
                                                                      top_ids,
                                                                      top_weights,
                                                                      gate_weight,
                                                                      up_weight,
                                                                      down_weight_q4,
                                                                      TOKENS,
                                                                      HIDDEN,
                                                                      PHYSICAL,
                                                                      INTERMEDIATE,
                                                                      DOWN_Q4_PHYSICAL,
                                                                      EXPERTS,
                                                                      TOP_K,
                                                                      UOCR_METAL_DENSE_OUTPUT_F16,
                                                                      out_f16,
                                                                      error,
                                                                      sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_q4[i]) < 6.0e-4f);
    }

    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k_padded(ctx,
                                                                      input,
                                                                      top_ids,
                                                                      top_weights,
                                                                      gate_weight,
                                                                      up_weight,
                                                                      down_weight_q4,
                                                                      TOKENS,
                                                                      HIDDEN,
                                                                      PHYSICAL,
                                                                      INTERMEDIATE,
                                                                      255u,
                                                                      EXPERTS,
                                                                      TOP_K,
                                                                      UOCR_METAL_DENSE_OUTPUT_F32,
                                                                      out_f32,
                                                                      error,
                                                                      sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected-expert Q4_K/padded-Q4_K prefill down widths") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k_q8_0(ctx,
                                                                    input,
                                                                    top_ids,
                                                                    top_weights,
                                                                    gate_weight,
                                                                    up_weight,
                                                                    down_weight_q8,
                                                                    TOKENS,
                                                                    HIDDEN,
                                                                    PHYSICAL,
                                                                    INTERMEDIATE,
                                                                    31u,
                                                                    EXPERTS,
                                                                    TOP_K,
                                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                                    out_f32,
                                                                    error,
                                                                    sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected-expert Q4_K/Q8_0 prefill down widths") != NULL);

    uint32_t bad_ids[TOKENS * TOP_K];
    memcpy(bad_ids, top_ids, sizeof(bad_ids));
    bad_ids[3] = EXPERTS;
    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k(ctx,
                                                               input,
                                                               bad_ids,
                                                               top_weights,
                                                               gate_weight,
                                                               up_weight,
                                                               down_weight,
                                                               TOKENS,
                                                               HIDDEN,
                                                               PHYSICAL,
                                                               INTERMEDIATE,
                                                               EXPERTS,
                                                               TOP_K,
                                                               UOCR_METAL_DENSE_OUTPUT_F32,
                                                               out_f32,
                                                               error,
                                                               sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE Q4_K prefill expert id") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k(ctx,
                                                               input,
                                                               top_ids,
                                                               top_weights,
                                                               gate_weight,
                                                               up_weight,
                                                               down_weight,
                                                               TOKENS,
                                                               HIDDEN,
                                                               384u,
                                                               INTERMEDIATE,
                                                               EXPERTS,
                                                               TOP_K,
                                                               UOCR_METAL_DENSE_OUTPUT_F32,
                                                               out_f32,
                                                               error,
                                                               sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected-expert Q4_K prefill widths") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_prefill_q4_k(ctx,
                                                               input,
                                                               top_ids,
                                                               top_weights,
                                                               gate_weight,
                                                               up_weight,
                                                               down_weight,
                                                               TOKENS,
                                                               HIDDEN,
                                                               PHYSICAL,
                                                               INTERMEDIATE,
                                                               EXPERTS,
                                                               TOP_K,
                                                               (uocr_metal_dense_output_type)99,
                                                               out_f32,
                                                               error,
                                                               sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE selected-expert Q4_K prefill output type") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_moe_combine_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = UOCR_HIDDEN_SIZE };
    const uint16_t routed_values[] = {
        0xb800u, /* -0.5 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3800u  /* 0.5 */
    };
    const uint16_t shared_values[] = {
        0xb400u, /* -0.25 */
        0xa800u, /* -0.03125 */
        0x2800u, /* 0.03125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t residual_values[] = {
        0xbc00u, /* -1.0 */
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t routed_value_count = (uint32_t)(sizeof(routed_values) / sizeof(routed_values[0]));
    const uint32_t shared_value_count = (uint32_t)(sizeof(shared_values) / sizeof(shared_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *routed = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *shared = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_no_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(routed != NULL && shared != NULL && residual != NULL && expected_residual != NULL &&
          expected_no_residual != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        routed[i] = routed_values[(i * 5u + i / 17u + 1u) % routed_value_count];
        shared[i] = shared_values[(i * 7u + i / 23u + 2u) % shared_value_count];
        residual[i] = residual_values[(i * 11u + i / 31u + 3u) % residual_value_count];
        expected_no_residual[i] = f16_bits_to_f32(routed[i]) + f16_bits_to_f32(shared[i]);
        expected_residual[i] = expected_no_residual[i] + f16_bits_to_f32(residual[i]);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             TOKENS,
                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_residual[i]) < 1.0e-7f);
    }

    memset(out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             NULL,
                                             TOKENS,
                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                             out_f16,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_no_residual[i]) < 8.0e-4f);
    }

    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             TOKENS,
                                             (uocr_metal_dense_output_type)99,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE combine output type") != NULL);
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             0u,
                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE combine request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected_no_residual);
    free(expected_residual);
    free(residual);
    free(shared);
    free(routed);
    return 0;
}

static void compute_rope_expected(const uint16_t *src,
                                  uint32_t n_tokens,
                                  uint32_t position_start,
                                  float *out) {
    enum { HEADS = 10, HEAD_DIM = 128, HALF_DIM = 64 };
    const float freq_scale = -2.0f * log2f(10000.0f) / (float)HEAD_DIM;
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        const float position = (float)(position_start + token);
        for (uint32_t head = 0u; head < (uint32_t)HEADS; ++head) {
            const uint32_t base = (token * (uint32_t)HEADS + head) * (uint32_t)HEAD_DIM;
            for (uint32_t pair = 0u; pair < (uint32_t)HALF_DIM; ++pair) {
                const uint32_t a = pair;
                const uint32_t b = pair + (uint32_t)HALF_DIM;
                const float angle = position * exp2f((float)pair * freq_scale);
                const float c = cosf(angle);
                const float s = sinf(angle);
                const float x0 = f16_bits_to_f32(src[base + a]);
                const float x1 = f16_bits_to_f32(src[base + b]);
                out[base + a] = x0 * c - x1 * s;
                out[base + b] = x0 * s + x1 * c;
            }
        }
    }
}

static int test_metal_rope_qk_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 4, HIDDEN = 1280 };
    const uint16_t values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t value_count = (uint32_t)(sizeof(values) / sizeof(values[0]));
    uint16_t *q = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected_q = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_k = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *q_out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *k_out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *q_out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k_out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && expected_q != NULL && expected_k != NULL &&
          q_out_f32 != NULL && k_out_f32 != NULL && q_out_f16 != NULL && k_out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        q[i] = values[(i * 17u + i / 31u + 1u) % value_count];
        k[i] = values[(i * 23u + i / 19u + 5u) % value_count];
    }

    const uint32_t position_start = 7u;
    compute_rope_expected(q, TOKENS, position_start, expected_q);
    compute_rope_expected(k, TOKENS, position_start, expected_k);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(q_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    memset(k_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(q_out_f32[i] - expected_q[i]) < 2.0e-4f);
        CHECK(fabsf(k_out_f32[i] - expected_k[i]) < 2.0e-4f);
    }

    memset(q_out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    memset(k_out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                         q_out_f16,
                                         k_out_f16,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(q_out_f16[i]) - expected_q[i]) < 4.0e-3f);
        CHECK(fabsf(f16_bits_to_f32(k_out_f16[i]) - expected_k[i]) < 4.0e-3f);
    }

    memset(q_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    memset(k_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         1u,
                                         0u,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < HIDDEN; ++i) {
        CHECK(fabsf(q_out_f32[i] - f16_bits_to_f32(q[i])) < 1.0e-6f);
        CHECK(fabsf(k_out_f32[i] - f16_bits_to_f32(k[i])) < 1.0e-6f);
    }

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         0u,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal RoPE request") != NULL);

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         2u,
                                         32767u,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max positions") != NULL);

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         (uocr_metal_dense_output_type)99,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal RoPE output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(k_out_f16);
    free(q_out_f16);
    free(k_out_f32);
    free(q_out_f32);
    free(expected_k);
    free(expected_q);
    free(k);
    free(q);
    return 0;
}

static int compute_prefill_attention_expected(const uint16_t *q,
                                              const uint16_t *k,
                                              const uint16_t *v,
                                              uint32_t n_tokens,
                                              float *out) {
    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    float scores[16];
    if (n_tokens > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
        return 0;
    }
    for (uint32_t query = 0u; query < n_tokens; ++query) {
        for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
            float max_score = -3.4028234663852886e38f;
            for (uint32_t key = 0u; key <= query; ++key) {
                float score = 0.0f;
                const uint32_t q_base = (query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                const uint32_t k_base = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                    score += f16_bits_to_f32(q[q_base + dim]) * f16_bits_to_f32(k[k_base + dim]);
                }
                score *= scale;
                scores[key] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float denominator = 0.0f;
            for (uint32_t key = 0u; key <= query; ++key) {
                scores[key] = expf(scores[key] - max_score);
                denominator += scores[key];
            }
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                float value = 0.0f;
                for (uint32_t key = 0u; key <= query; ++key) {
                    const uint32_t v_index = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim;
                    value += (scores[key] / denominator) * f16_bits_to_f32(v[v_index]);
                }
                out[(query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim] = value;
            }
        }
    }
    return 1;
}

static int compute_prefill_attention_varlen_expected(const uint16_t *q,
                                                     const uint16_t *k,
                                                     const uint16_t *v,
                                                     const uint32_t *cu,
                                                     uint32_t batch,
                                                     uint32_t total_tokens,
                                                     float *out) {
    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    float scores[16];
    for (uint32_t b = 0u; b < batch; ++b) {
        const uint32_t start = cu[b];
        const uint32_t end = cu[b + 1u];
        if (end <= start || end > total_tokens || end - start > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
            return 0;
        }
        for (uint32_t query = start; query < end; ++query) {
            for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
                float max_score = -3.4028234663852886e38f;
                for (uint32_t key = start; key <= query; ++key) {
                    float score = 0.0f;
                    const uint32_t q_base = (query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                    const uint32_t k_base = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                        score += f16_bits_to_f32(q[q_base + dim]) * f16_bits_to_f32(k[k_base + dim]);
                    }
                    score *= scale;
                    scores[key - start] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                float denominator = 0.0f;
                for (uint32_t key = start; key <= query; ++key) {
                    scores[key - start] = expf(scores[key - start] - max_score);
                    denominator += scores[key - start];
                }
                for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                    float value = 0.0f;
                    for (uint32_t key = start; key <= query; ++key) {
                        const uint32_t v_index = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim;
                        value += (scores[key - start] / denominator) * f16_bits_to_f32(v[v_index]);
                    }
                    out[(query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim] = value;
                }
            }
        }
    }
    return 1;
}

static int test_metal_prefill_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 5, HIDDEN = UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM };
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *v = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && v != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        q[i] = q_values[(i * 7u + i / 19u + 1u) % q_count];
        k[i] = k_values[(i * 11u + i / 23u + 3u) % k_count];
        v[i] = v_values[(i * 13u + i / 29u + 5u) % v_count];
    }
    /* Make future tokens conspicuous so query-token zero verifies the causal
     * mask cannot leak later values.
     */
    for (uint32_t token = 1u; token < (uint32_t)TOKENS; ++token) {
        v[token * (uint32_t)HIDDEN] = 0x5000u; /* 32.0 */
    }

    CHECK(compute_prefill_attention_expected(q, k, v, TOKENS, expected) == 1);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 8.0e-4f);
    }
    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
        CHECK(out_f32[dim] == f16_bits_to_f32(v[dim]));
    }

    memset(out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   out_f16,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   0u,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal prefill attention request") != NULL);
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   (uocr_metal_dense_output_type)99,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal prefill attention output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_metal_prefill_attention_varlen_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { BATCH = 2, TOTAL_TOKENS = 7, MAX_SEQLEN = 4, HIDDEN = UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM };
    const uint32_t cu[BATCH + 1] = {0u, 3u, 7u};
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *v = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && v != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        q[i] = q_values[(i * 7u + i / 19u + 2u) % q_count];
        k[i] = k_values[(i * 11u + i / 23u + 4u) % k_count];
        v[i] = v_values[(i * 13u + i / 29u + 1u) % v_count];
    }
    /* Make cross-sequence leakage obvious: first token of sequence 1 must only
     * see itself, never the large values placed at the end of sequence 0.
     */
    v[2u * (uint32_t)HIDDEN] = 0x5000u; /* 32.0 */
    v[3u * (uint32_t)HIDDEN] = 0xb800u; /* -0.5 */
    CHECK(compute_prefill_attention_varlen_expected(q, k, v, cu, BATCH, TOTAL_TOKENS, expected) == 1);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-3f);
    }
    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
        CHECK(out_f32[dim] == f16_bits_to_f32(v[dim]));
        CHECK(out_f32[3u * (uint32_t)HIDDEN + dim] == f16_bits_to_f32(v[3u * (uint32_t)HIDDEN + dim]));
    }

    memset(out_f16, 0, (size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F16,
                                                          out_f16,
                                                          error,
                                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    const uint32_t bad_cu_order[BATCH + 1] = {0u, 5u, 4u};
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          bad_cu_order,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 0);
    CHECK(strstr(error, "sequence metadata") != NULL || strstr(error, "cu_seqlens") != NULL);

    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          2u,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max_seqlen") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v);
    free(k);
    free(q);
    return 0;
}

static uint16_t kv_cache_sentinel(uint64_t index, uint32_t stream) {
    return (uint16_t)(0x1800u + ((index * 17u + (uint64_t)stream * 97u) & 0x07ffu));
}

static uint16_t kv_source_value(uint64_t index, uint32_t stream) {
    return (uint16_t)(0x3000u + ((index * 23u + (uint64_t)stream * 211u) & 0x0fffu));
}

static uint32_t kv_cache_token_for_position(uint32_t position, uint32_t prompt_length) {
    if (position < prompt_length) {
        return position;
    }
    return prompt_length + ((position - prompt_length) % UOCR_GENERATED_RING_WINDOW);
}

static uint64_t kv_cache_flat_index(uint32_t batch_slots,
                                    uint32_t cache_token_capacity,
                                    uint32_t layer,
                                    uint32_t slot,
                                    uint32_t cache_token,
                                    uint32_t head,
                                    uint32_t dim) {
    return (((uint64_t)layer * (uint64_t)batch_slots + (uint64_t)slot) *
                (uint64_t)cache_token_capacity + (uint64_t)cache_token) *
               (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM +
           (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim;
}

static int compute_decode_attention_expected(const uint16_t *q,
                                             const uint16_t *k_cache,
                                             const uint16_t *v_cache,
                                             uint32_t batch_slots,
                                             uint32_t prompt_token_capacity,
                                             uint32_t layer,
                                             uint32_t slot,
                                             uint32_t prompt_length,
                                             uint32_t generated_count,
                                             float *out) {
    uint32_t attention_length = 0u;
    if (!uocr_metal_kv_cache_attention_length(prompt_length, generated_count, &attention_length)) {
        return 0;
    }
    const uint32_t cache_token_capacity = prompt_token_capacity + UOCR_GENERATED_RING_WINDOW;
    float *scores = (float *)malloc((size_t)attention_length * sizeof(float));
    if (scores == NULL) {
        return 0;
    }

    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
        float max_score = -3.4028234663852886e38f;
        for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
            uint32_t cache_token = UINT32_MAX;
            if (!uocr_metal_kv_cache_token_for_attention_index(prompt_length,
                                                               prompt_token_capacity,
                                                               generated_count,
                                                               attention_index,
                                                               &cache_token)) {
                free(scores);
                return 0;
            }
            float score = 0.0f;
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t q_index = (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim;
                const uint64_t k_index = kv_cache_flat_index(batch_slots,
                                                             cache_token_capacity,
                                                             layer,
                                                             slot,
                                                             cache_token,
                                                             head,
                                                             dim);
                score += f16_bits_to_f32(q[q_index]) * f16_bits_to_f32(k_cache[k_index]);
            }
            score *= scale;
            scores[attention_index] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        float denominator = 0.0f;
        for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
            scores[attention_index] = expf(scores[attention_index] - max_score);
            denominator += scores[attention_index];
        }
        for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
            float value = 0.0f;
            for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
                uint32_t cache_token = UINT32_MAX;
                if (!uocr_metal_kv_cache_token_for_attention_index(prompt_length,
                                                                   prompt_token_capacity,
                                                                   generated_count,
                                                                   attention_index,
                                                                   &cache_token)) {
                    free(scores);
                    return 0;
                }
                const uint64_t v_index = kv_cache_flat_index(batch_slots,
                                                             cache_token_capacity,
                                                             layer,
                                                             slot,
                                                             cache_token,
                                                             head,
                                                             dim);
                value += (scores[attention_index] / denominator) * f16_bits_to_f32(v_cache[v_index]);
            }
            out[(uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim] = value;
        }
    }

    free(scores);
    return 1;
}

static int test_metal_kv_cache_layout_helpers(void) {
    uint32_t cache_token = UINT32_MAX;
    uint32_t attention_length = 0u;
    uint64_t offset = 0u;
    uocr_metal_decode_attention_plan decode_plan;

    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 6u, &cache_token) == 1);
    CHECK(cache_token == 6u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u + UOCR_GENERATED_RING_WINDOW - 1u, &cache_token) == 1);
    CHECK(cache_token == 7u + UOCR_GENERATED_RING_WINDOW - 1u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u + UOCR_GENERATED_RING_WINDOW, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_position(0u, 16u, 0u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_position(17u, 16u, 0u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, UOCR_MAX_POSITIONS, &cache_token) == 0);

    CHECK(uocr_metal_kv_cache_attention_length(7u, 0u, &attention_length) == 1);
    CHECK(attention_length == 7u);
    CHECK(uocr_metal_kv_cache_attention_length(7u, 5u, &attention_length) == 1);
    CHECK(attention_length == 12u);
    CHECK(uocr_metal_kv_cache_attention_length(7u, UOCR_GENERATED_RING_WINDOW, &attention_length) == 1);
    CHECK(attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_attention_length(7u, UOCR_GENERATED_RING_WINDOW + 9u, &attention_length) == 1);
    CHECK(attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS, 0u, &attention_length) == 1);
    CHECK(attention_length == UOCR_MAX_POSITIONS);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS, 1u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS - 3u, 3u, &attention_length) == 1);
    CHECK(attention_length == UOCR_MAX_POSITIONS);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS - 3u, 4u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(UINT32_MAX, 0u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(0u, 1u, &attention_length) == 0);

    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 0u, 6u, &cache_token) == 1);
    CHECK(cache_token == 6u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 7u, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 11u, &cache_token) == 1);
    CHECK(cache_token == 11u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_GENERATED_RING_WINDOW + 9u,
                                                        7u,
                                                        &cache_token) == 1);
    CHECK(cache_token == 16u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_GENERATED_RING_WINDOW + 9u,
                                                        7u + UOCR_GENERATED_RING_WINDOW - 1u,
                                                        &cache_token) == 1);
    CHECK(cache_token == 15u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 12u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_MAX_POSITIONS,
                                                        7u,
                                                        &cache_token) == 0);

    memset(&decode_plan, 0, sizeof(decode_plan));
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u,
                                                    16u,
                                                    UOCR_GENERATED_RING_WINDOW + 9u,
                                                    &decode_plan) == 1);
    CHECK(decode_plan.prompt_length == 7u);
    CHECK(decode_plan.prompt_token_capacity == 16u);
    CHECK(decode_plan.cache_token_capacity == 16u + UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.generated_count == UOCR_GENERATED_RING_WINDOW + 9u);
    CHECK(decode_plan.live_generated == UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.first_generated_index == 9u);
    CHECK(decode_plan.first_generated_position == 16u);
    CHECK(decode_plan.query_position == 7u + UOCR_GENERATED_RING_WINDOW + 8u);
    CHECK(decode_plan.attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.generated_ring_window == UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 0u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 6u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 15u) == 0);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 16u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u + UOCR_GENERATED_RING_WINDOW + 8u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u + UOCR_GENERATED_RING_WINDOW + 9u) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 7u, &cache_token) == 1);
    CHECK(cache_token == 16u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan,
                                                              7u + UOCR_GENERATED_RING_WINDOW - 1u,
                                                              &cache_token) == 1);
    CHECK(cache_token == 15u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan,
                                                              7u + UOCR_GENERATED_RING_WINDOW,
                                                              &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u, 6u, 0u, &decode_plan) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_plan(0u, 16u, 0u, &decode_plan) == 0);

    memset(&decode_plan, 0, sizeof(decode_plan));
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u, 16u, 0u, &decode_plan) == 1);
    CHECK(decode_plan.live_generated == 0u);
    CHECK(decode_plan.attention_length == 7u);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 6u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u) == 0);

    memset(&decode_plan, 0, sizeof(decode_plan));
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u,
                                                    16u,
                                                    UOCR_GENERATED_RING_WINDOW + 1u,
                                                    &decode_plan) == 1);
    CHECK(decode_plan.attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.first_generated_index == 1u);
    CHECK(decode_plan.first_generated_position == 8u);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 0u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 6u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u) == 0);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 8u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan,
                                                      7u + UOCR_GENERATED_RING_WINDOW) == 1);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 6u, &cache_token) == 1);
    CHECK(cache_token == 6u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 7u, &cache_token) == 1);
    CHECK(cache_token == 8u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan,
                                                              7u + UOCR_GENERATED_RING_WINDOW - 1u,
                                                              &cache_token) == 1);
    CHECK(cache_token == 7u);

    uocr_metal_kv_cache_layout empty_layout;
    memset(&empty_layout, 0, sizeof(empty_layout));
    CHECK(uocr_metal_kv_cache_offset(&empty_layout, 0, 0u, 0u, 0u, 0u, 0u, &offset) == 0);

    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uocr_metal_kv_cache_layout layout;
    memset(&layout, 0, sizeof(layout));
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 0);
    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 2u, 16u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 1);
    CHECK(layout.decoder_layers == UOCR_DECODER_LAYERS);
    CHECK(layout.batch_slots == 2u);
    CHECK(layout.prompt_token_capacity == 16u);
    CHECK(layout.cache_token_capacity == 16u + UOCR_GENERATED_RING_WINDOW);
    CHECK(layout.kv_heads == UOCR_KV_HEADS);
    CHECK(layout.head_dim == UOCR_HEAD_DIM);
    CHECK(layout.generated_ring_window == UOCR_GENERATED_RING_WINDOW);
    CHECK(layout.token_stride_bytes == (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM * 2u);
    CHECK(layout.slot_stride_bytes == (uint64_t)layout.cache_token_capacity * layout.token_stride_bytes);
    CHECK(layout.layer_stride_bytes == (uint64_t)layout.batch_slots * layout.slot_stride_bytes);
    CHECK(layout.tensor_bytes == (uint64_t)UOCR_DECODER_LAYERS * layout.layer_stride_bytes);
    CHECK(layout.k_offset_bytes == 0u);
    CHECK(layout.v_offset_bytes == layout.tensor_bytes);
    CHECK(layout.total_bytes == layout.tensor_bytes * 2u);
    CHECK(layout.total_bytes == uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_KV_CACHE));

    const uint32_t layer = 2u;
    const uint32_t slot = 1u;
    const uint32_t token = 9u;
    const uint32_t head = 4u;
    const uint32_t dim = 5u;
    const uint64_t expected_inner = (uint64_t)layer * layout.layer_stride_bytes +
                                    (uint64_t)slot * layout.slot_stride_bytes +
                                    (uint64_t)token * layout.token_stride_bytes +
                                    (uint64_t)head * (uint64_t)UOCR_HEAD_DIM * 2u +
                                    (uint64_t)dim * 2u;
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, head, dim, &offset) == 1);
    CHECK(offset == layout.k_offset_bytes + expected_inner);
    CHECK(uocr_metal_kv_cache_offset(&layout, 1, layer, slot, token, head, dim, &offset) == 1);
    CHECK(offset == layout.v_offset_bytes + expected_inner);
    CHECK(uocr_metal_kv_cache_offset(&layout,
                                     1,
                                     layout.decoder_layers - 1u,
                                     layout.batch_slots - 1u,
                                     layout.cache_token_capacity - 1u,
                                     layout.kv_heads - 1u,
                                     layout.head_dim - 1u,
                                     &offset) == 1);
    CHECK(offset == layout.total_bytes - 2u);
    CHECK(uocr_metal_kv_cache_offset(&layout, 2, layer, slot, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, UOCR_DECODER_LAYERS, slot, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, 2u, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, layout.cache_token_capacity, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, UOCR_KV_HEADS, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, head, UOCR_HEAD_DIM, &offset) == 0);

    uocr_metal_context_release_runtime_arenas(ctx);
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 0);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_kv_cache_write_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        BATCH_SLOTS = 2,
        PROMPT_TOKEN_CAPACITY = 8,
        PROMPT_LENGTH = 5,
        CACHE_TOKEN_CAPACITY = PROMPT_TOKEN_CAPACITY + UOCR_GENERATED_RING_WINDOW,
        LAYER = 3,
        SLOT = 1,
        N_TOKENS = PROMPT_LENGTH + UOCR_GENERATED_RING_WINDOW + 3,
        HEADS = UOCR_KV_HEADS,
        HEAD_DIM = UOCR_HEAD_DIM,
        HEAD_AREA = HEADS * HEAD_DIM
    };
    const uint64_t cache_values = (uint64_t)UOCR_DECODER_LAYERS * (uint64_t)BATCH_SLOTS *
                                  (uint64_t)CACHE_TOKEN_CAPACITY * (uint64_t)HEAD_AREA;
    const uint64_t src_values = (uint64_t)N_TOKENS * (uint64_t)HEAD_AREA;

    uint16_t *k_src = (uint16_t *)malloc((size_t)src_values * sizeof(uint16_t));
    uint16_t *v_src = (uint16_t *)malloc((size_t)src_values * sizeof(uint16_t));
    uint16_t *initial_k = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *initial_v = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *k_out = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *v_out = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint32_t *last_token_for_cache_token = (uint32_t *)malloc((size_t)CACHE_TOKEN_CAPACITY * sizeof(uint32_t));
    CHECK(k_src != NULL && v_src != NULL && initial_k != NULL && initial_v != NULL &&
          k_out != NULL && v_out != NULL && last_token_for_cache_token != NULL);

    for (uint64_t i = 0u; i < src_values; ++i) {
        k_src[i] = kv_source_value(i, 0u);
        v_src[i] = kv_source_value(i, 1u);
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        initial_k[i] = kv_cache_sentinel(i, 0u);
        initial_v[i] = kv_cache_sentinel(i, 1u);
        k_out[i] = 0u;
        v_out[i] = 0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)CACHE_TOKEN_CAPACITY; ++i) {
        last_token_for_cache_token[i] = UINT32_MAX;
    }
    for (uint32_t token = 0u; token < (uint32_t)N_TOKENS; ++token) {
        const uint32_t cache_token = kv_cache_token_for_position(token, PROMPT_LENGTH);
        CHECK(cache_token < (uint32_t)CACHE_TOKEN_CAPACITY);
        last_token_for_cache_token[cache_token] = token;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                N_TOKENS,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint64_t index = 0u; index < cache_values; ++index) {
        uint64_t rem = index;
        const uint32_t dim = (uint32_t)(rem % (uint64_t)HEAD_DIM);
        rem /= (uint64_t)HEAD_DIM;
        const uint32_t head = (uint32_t)(rem % (uint64_t)HEADS);
        rem /= (uint64_t)HEADS;
        const uint32_t cache_token = (uint32_t)(rem % (uint64_t)CACHE_TOKEN_CAPACITY);
        rem /= (uint64_t)CACHE_TOKEN_CAPACITY;
        const uint32_t slot = (uint32_t)(rem % (uint64_t)BATCH_SLOTS);
        const uint32_t layer = (uint32_t)(rem / (uint64_t)BATCH_SLOTS);

        uint16_t expected_k = initial_k[index];
        uint16_t expected_v = initial_v[index];
        if (layer == (uint32_t)LAYER && slot == (uint32_t)SLOT &&
            last_token_for_cache_token[cache_token] != UINT32_MAX) {
            const uint64_t src_index = ((uint64_t)last_token_for_cache_token[cache_token] * (uint64_t)HEADS +
                                        (uint64_t)head) * (uint64_t)HEAD_DIM + (uint64_t)dim;
            expected_k = k_src[src_index];
            expected_v = v_src[src_index];
        }
        CHECK(k_out[index] == expected_k);
        CHECK(v_out[index] == expected_v);
    }

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                NULL,
                                                NULL,
                                                1u,
                                                1u,
                                                PROMPT_TOKEN_CAPACITY,
                                                0u,
                                                0u,
                                                PROMPT_LENGTH,
                                                2u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t index = 0u; index < cache_values / (uint64_t)BATCH_SLOTS; ++index) {
        uint64_t rem = index;
        const uint32_t dim = (uint32_t)(rem % (uint64_t)HEAD_DIM);
        rem /= (uint64_t)HEAD_DIM;
        const uint32_t head = (uint32_t)(rem % (uint64_t)HEADS);
        rem /= (uint64_t)HEADS;
        const uint32_t cache_token = (uint32_t)(rem % (uint64_t)CACHE_TOKEN_CAPACITY);
        rem /= (uint64_t)CACHE_TOKEN_CAPACITY;
        const uint32_t layer = (uint32_t)rem;
        const uint64_t src_index = ((uint64_t)head * (uint64_t)HEAD_DIM) + (uint64_t)dim;
        const int should_have_value = layer == 0u && cache_token == 2u;
        CHECK(k_out[index] == (should_have_value ? k_src[src_index] : 0u));
        CHECK(v_out[index] == (should_have_value ? v_src[src_index] : 0u));
    }

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                0u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal KV cache write request") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                1u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                UOCR_DECODER_LAYERS,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "layer/slot out of range") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                1u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_TOKEN_CAPACITY + 1u,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal KV cache layout") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                2u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                UOCR_MAX_POSITIONS - 1u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max positions") != NULL);

    uocr_metal_context_destroy(ctx);
    free(last_token_for_cache_token);
    free(v_out);
    free(k_out);
    free(initial_v);
    free(initial_k);
    free(v_src);
    free(k_src);
    return 0;
}

static int test_metal_decode_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        BATCH_SLOTS = 2,
        PROMPT_TOKEN_CAPACITY = 8,
        PROMPT_LENGTH = 5,
        CACHE_TOKEN_CAPACITY = PROMPT_TOKEN_CAPACITY + UOCR_GENERATED_RING_WINDOW,
        LAYER = 4,
        SLOT = 1,
        HEAD_AREA = UOCR_KV_HEADS * UOCR_HEAD_DIM
    };
    const uint64_t cache_values = (uint64_t)UOCR_DECODER_LAYERS * (uint64_t)BATCH_SLOTS *
                                  (uint64_t)CACHE_TOKEN_CAPACITY * (uint64_t)HEAD_AREA;
    const uint64_t out_values = (uint64_t)HEAD_AREA;
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0xac00u  /* -0.0625 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)out_values * sizeof(uint16_t));
    uint16_t *k_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *v_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)out_values * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)out_values * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)out_values * sizeof(uint16_t));
    CHECK(q != NULL && k_cache != NULL && v_cache != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t no_wrap_generated = 3u;
    uint32_t attention_length = 0u;
    CHECK(uocr_metal_kv_cache_attention_length(PROMPT_LENGTH, no_wrap_generated, &attention_length) == 1);
    for (uint64_t i = 0u; i < out_values; ++i) {
        q[i] = q_values[(i * 7u + i / 11u + 3u) % q_count];
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        k_cache[i] = 0x0000u;
        v_cache[i] = 0x5000u; /* 32.0; catches accidental attention to unused ring slots. */
    }
    for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
        uint32_t cache_token = UINT32_MAX;
        CHECK(uocr_metal_kv_cache_token_for_attention_index(PROMPT_LENGTH,
                                                            PROMPT_TOKEN_CAPACITY,
                                                            no_wrap_generated,
                                                            attention_index,
                                                            &cache_token) == 1);
        for (uint32_t head = 0u; head < UOCR_KV_HEADS; ++head) {
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t serial = ((uint64_t)attention_index * (uint64_t)HEAD_AREA +
                                         (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim);
                const uint64_t index = kv_cache_flat_index(BATCH_SLOTS,
                                                           CACHE_TOKEN_CAPACITY,
                                                           LAYER,
                                                           SLOT,
                                                           cache_token,
                                                           head,
                                                           dim);
                k_cache[index] = k_values[(serial * 11u + serial / 17u + 1u) % k_count];
                v_cache[index] = v_values[(serial * 13u + serial / 19u + 4u) % v_count];
            }
        }
    }
    CHECK(compute_decode_attention_expected(q,
                                            k_cache,
                                            v_cache,
                                            BATCH_SLOTS,
                                            PROMPT_TOKEN_CAPACITY,
                                            LAYER,
                                            SLOT,
                                            PROMPT_LENGTH,
                                            no_wrap_generated,
                                            expected) == 1);

    memset(out_f32, 0, (size_t)out_values * sizeof(float));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-3f);
    }

    memset(out_f16, 0, (size_t)out_values * sizeof(uint16_t));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  out_f16,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    const uint32_t wrapped_generated = UOCR_GENERATED_RING_WINDOW + 3u;
    CHECK(uocr_metal_kv_cache_attention_length(PROMPT_LENGTH, wrapped_generated, &attention_length) == 1);
    for (uint64_t i = 0u; i < out_values; ++i) {
        q[i] = 0x0000u;
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        k_cache[i] = 0x0000u;
        v_cache[i] = 0x5000u; /* Must not be read from inactive layers/slots/tokens. */
    }
    for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
        uint32_t cache_token = UINT32_MAX;
        CHECK(uocr_metal_kv_cache_token_for_attention_index(PROMPT_LENGTH,
                                                            PROMPT_TOKEN_CAPACITY,
                                                            wrapped_generated,
                                                            attention_index,
                                                            &cache_token) == 1);
        const uint16_t value = attention_index < PROMPT_LENGTH ? 0x3c00u : 0xbc00u; /* +1 prompt, -1 generated. */
        for (uint32_t head = 0u; head < UOCR_KV_HEADS; ++head) {
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t index = kv_cache_flat_index(BATCH_SLOTS,
                                                           CACHE_TOKEN_CAPACITY,
                                                           LAYER,
                                                           SLOT,
                                                           cache_token,
                                                           head,
                                                           dim);
                k_cache[index] = 0x0000u;
                v_cache[index] = value;
            }
        }
    }
    CHECK(compute_decode_attention_expected(q,
                                            k_cache,
                                            v_cache,
                                            BATCH_SLOTS,
                                            PROMPT_TOKEN_CAPACITY,
                                            LAYER,
                                            SLOT,
                                            PROMPT_LENGTH,
                                            wrapped_generated,
                                            expected) == 1);
    memset(out_f32, 0, (size_t)out_values * sizeof(float));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  wrapped_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-5f);
    }
    CHECK(out_f32[0] > -0.98f && out_f32[0] < -0.90f);

    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  UOCR_MAX_POSITIONS,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "generated count") != NULL);

    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  (uocr_metal_dense_output_type)99,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal decode attention output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v_cache);
    free(k_cache);
    free(q);
    return 0;
}

static int test_metal_sdpa_prefill_decode_consistency_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        TOKENS = 4,
        BATCH_SLOTS = 1,
        PROMPT_TOKEN_CAPACITY = TOKENS,
        CACHE_TOKEN_CAPACITY = PROMPT_TOKEN_CAPACITY + UOCR_GENERATED_RING_WINDOW,
        LAYER = 1,
        SLOT = 0,
        HIDDEN = UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM,
        HEAD_AREA = UOCR_KV_HEADS * UOCR_HEAD_DIM
    };
    const uint64_t prompt_values = (uint64_t)TOKENS * (uint64_t)HIDDEN;
    const uint64_t cache_values = (uint64_t)UOCR_DECODER_LAYERS * (uint64_t)BATCH_SLOTS *
                                  (uint64_t)CACHE_TOKEN_CAPACITY * (uint64_t)HEAD_AREA;
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0xb000u, /* -0.125 */
        0x3000u  /* 0.125 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0xac00u, /* -0.0625 */
        0x2c00u  /* 0.0625 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)prompt_values * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)prompt_values * sizeof(uint16_t));
    uint16_t *v = (uint16_t *)malloc((size_t)prompt_values * sizeof(uint16_t));
    float *expected_prefill = (float *)malloc((size_t)prompt_values * sizeof(float));
    float *prefill_out = (float *)malloc((size_t)prompt_values * sizeof(float));
    uint16_t *k_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *v_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    float *decode_out = (float *)malloc((size_t)HIDDEN * sizeof(float));
    CHECK(q != NULL && k != NULL && v != NULL && expected_prefill != NULL && prefill_out != NULL &&
          k_cache != NULL && v_cache != NULL && decode_out != NULL);

    for (uint64_t i = 0u; i < prompt_values; ++i) {
        q[i] = q_values[(i * 7u + i / 13u + 1u) % q_count];
        k[i] = k_values[(i * 11u + i / 17u + 3u) % k_count];
        v[i] = v_values[(i * 13u + i / 19u + 5u) % v_count];
    }
    CHECK(compute_prefill_attention_expected(q, k, v, TOKENS, expected_prefill) == 1);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(prefill_out, 0, (size_t)prompt_values * sizeof(float));
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   prefill_out,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < prompt_values; ++i) {
        CHECK(fabsf(prefill_out[i] - expected_prefill[i]) < 8.0e-4f);
    }

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k,
                                                v,
                                                NULL,
                                                NULL,
                                                TOKENS,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                TOKENS,
                                                0u,
                                                k_cache,
                                                v_cache,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    memset(decode_out, 0, (size_t)HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q + ((uint32_t)TOKENS - 1u) * (uint32_t)HIDDEN,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  TOKENS,
                                                  0u,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  decode_out,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        const uint64_t prefill_index = ((uint64_t)TOKENS - 1u) * (uint64_t)HIDDEN + (uint64_t)i;
        CHECK(fabsf(decode_out[i] - expected_prefill[prefill_index]) < 1.0e-3f);
        CHECK(fabsf(decode_out[i] - prefill_out[prefill_index]) < 1.0e-3f);
    }

    uocr_metal_context_destroy(ctx);
    free(decode_out);
    free(v_cache);
    free(k_cache);
    free(prefill_out);
    free(expected_prefill);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_metal_recent_decoder_primitives_stress(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint16_t half_pool[] = {
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x4200u, /* 3.0 */
        0x5000u  /* 32.0; catches accidental fp16 RMS variance overflow */
    };
    const uint32_t half_pool_count = (uint32_t)(sizeof(half_pool) / sizeof(half_pool[0]));

    {
        enum { TABLE_ROWS = 257, HIDDEN = 1280, OUT_ROWS = 17 };
        uint16_t table[TABLE_ROWS * HIDDEN];
        uint16_t out[OUT_ROWS * HIDDEN];
        const int32_t row_ids[OUT_ROWS] = {256, 0, 128, 7, 7, 255, 1, 64, 200, 31, 16, 127, 2, 254, 3, 129, 42};
        for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * HIDDEN); ++i) {
            table[i] = half_pool[(i * 13u + i / 17u + 5u) % half_pool_count];
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_get_rows_f16(ctx,
                                              table,
                                              TABLE_ROWS,
                                              HIDDEN,
                                              row_ids,
                                              OUT_ROWS,
                                              UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                              out,
                                              error,
                                              sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                CHECK(out[row * (uint32_t)HIDDEN + col] == table[(uint32_t)row_ids[row] * (uint32_t)HIDDEN + col]);
            }
        }
    }

    {
        enum { TABLE_ROWS = 64, HIDDEN = 1280, TOKENS = 9, IMAGE_TOKENS = 3 };
        uint16_t table[TABLE_ROWS * HIDDEN];
        uint16_t image_features[IMAGE_TOKENS * HIDDEN];
        uint16_t out[TOKENS * HIDDEN];
        const int32_t ids[TOKENS] = {0, 17, 999999, -7, 123456, 63, 4, 1, 2};
        for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * HIDDEN); ++i) {
            table[i] = half_pool[(i * 7u + 3u) % half_pool_count];
        }
        for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
            image_features[i] = (uint16_t)(0x6000u + (i % 1024u));
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                uint16_t expected = 0u;
                if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                    expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
                } else {
                    expected = table[(uint32_t)ids[token] * (uint32_t)HIDDEN + col];
                }
                CHECK(out[token * (uint32_t)HIDDEN + col] == expected);
            }
        }

        const int32_t span_at_start_ids[TOKENS] = {-123, 999999, 3, 4, 5, 6, 7, 8, 9};
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     span_at_start_ids,
                                                     TOKENS,
                                                     0u,
                                                     2u,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(out[col] == image_features[col]);
            CHECK(out[(uint32_t)HIDDEN + col] == image_features[(uint32_t)HIDDEN + col]);
        }

        const int32_t span_at_end_ids[TOKENS] = {0, 1, 2, 3, 4, 5, 6, -123, 999999};
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     span_at_end_ids,
                                                     TOKENS,
                                                     7u,
                                                     2u,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(out[7u * (uint32_t)HIDDEN + col] == image_features[col]);
            CHECK(out[8u * (uint32_t)HIDDEN + col] == image_features[(uint32_t)HIDDEN + col]);
        }

        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     0u,
                                                     0u,
                                                     NULL,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "UINT32_MAX image span start") != NULL);

        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     NULL,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "requires image features") != NULL);

        const int32_t invalid_outside_span[TOKENS] = {0, TABLE_ROWS, 999999, -7, 123456, 2, 3, 4, 5};
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     invalid_outside_span,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "outside embedding rows") != NULL);
    }

    {
        enum { ROWS = 2, HIDDEN = 1280 };
        uint16_t input[ROWS * HIDDEN];
        uint16_t weight[HIDDEN];
        float out[ROWS * HIDDEN];
        float expected[ROWS * HIDDEN];
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            input[col] = 0x5000u; /* 32.0: fp16 sum of squares would overflow. */
            weight[col] = half_pool[(col * 5u + 3u) % half_pool_count];
        }
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            input[(uint32_t)HIDDEN + col] = half_pool[(col * 11u + 1u) % (half_pool_count - 1u)];
        }
        const float eps = 1.0e-6f;
        for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
            float sum = 0.0f;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
                sum += x * x;
            }
            const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + eps);
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
                const float w = f16_bits_to_f32(weight[col]);
                expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
            }
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                             input,
                                             weight,
                                             ROWS,
                                             HIDDEN,
                                             eps,
                                             UOCR_METAL_RMSNORM_OUTPUT_F32,
                                             out,
                                             error,
                                             sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
            CHECK(fabsf(out[i] - expected[i]) < 3.0e-4f);
        }
    }

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_runtime_arenas(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 16u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    uint64_t expected_kv = 0u;
    uint64_t expected_prompt = 0u;
    uint64_t expected_decoder = 0u;
    uint64_t expected_router_topk = 0u;
    uint64_t expected_moe_intermediate = 0u;
    uint64_t expected_vision = 0u;
    uint64_t expected_logits = 0u;
    CHECK(uocr_estimate_kv_cache_bytes(1u, 16u, &expected_kv) == UOCR_OK);
    CHECK(uocr_estimate_prompt_embedding_bytes(1u, 16u, &expected_prompt) == UOCR_OK);
    CHECK(uocr_estimate_decoder_scratch_bytes(1u, 16u, &expected_decoder) == UOCR_OK);
    CHECK(uocr_estimate_moe_router_topk_bytes(1u, 16u, &expected_router_topk) == UOCR_OK);
    CHECK(uocr_estimate_moe_intermediate_bytes(1u, 16u, &expected_moe_intermediate) == UOCR_OK);
    CHECK(uocr_estimate_vision_scratch_bytes_for_rows(16u,
                                                       UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
                                                       &expected_vision) == UOCR_OK);
    CHECK(uocr_estimate_logits_readback_bytes(1u, &expected_logits) == UOCR_OK);

    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_KV_CACHE) == expected_kv);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_PROMPT_EMBEDDINGS) == expected_prompt);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_HIDDEN_PINGPONG) == expected_decoder);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_ROUTER_TOPK) == expected_router_topk);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_MOE_INTERMEDIATE) == expected_moe_intermediate);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_VISION_SCRATCH) == expected_vision);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_LOGITS_READBACK) == expected_logits);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == expected_kv + expected_prompt + expected_decoder +
                                                               expected_router_topk + expected_moe_intermediate +
                                                               expected_vision + expected_logits);

    uocr_metal_context_release_runtime_arenas(ctx);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_integrated_decoder_boundary(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const int32_t input_ids[3] = {UOCR_TOKEN_BOS, 42, 77};
    const uint8_t image_mask[3] = {0u, 0u, 0u};
    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 3u;
    request.max_new_tokens = 0u;
    request.slot = 0u;
    request.image_span_start = UINT32_MAX;
    request.image_span_length = 0u;

    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires allocated runtime arenas") != NULL);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 8u, error, sizeof(error)) == 1);
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires validated fp16 decoder tensor bindings") != NULL);
    CHECK(result.generated_count == 0u);
    CHECK(result.stopped_on_eos == 0u);
    CHECK(result.last_token_id == UINT32_MAX);

    int32_t generated[1] = {0};
    request.max_new_tokens = 1u;
    result.generated_ids = generated;
    result.generated_capacity = 1u;
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires validated fp16 decoder tensor bindings") != NULL);
    CHECK(result.generated_count == 0u);
    CHECK(result.last_token_id == UINT32_MAX);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_vision_runner_requires_bindings(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t n_tokens = 1u + UOCR_GLOBAL_VISUAL_TOKENS;
    int32_t *input_ids = (int32_t *)calloc((size_t)n_tokens, sizeof(int32_t));
    uint8_t *image_mask = (uint8_t *)calloc((size_t)n_tokens, sizeof(uint8_t));
    uint16_t *pixels = (uint16_t *)calloc(3u, sizeof(uint16_t));
    uint16_t *visual = (uint16_t *)calloc((size_t)UOCR_GLOBAL_VISUAL_TOKENS * (size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    CHECK(input_ids != NULL);
    CHECK(image_mask != NULL);
    CHECK(pixels != NULL);
    CHECK(visual != NULL);

    input_ids[0] = UOCR_TOKEN_BOS;
    for (uint32_t i = 1u; i < n_tokens; ++i) {
        input_ids[i] = UOCR_TOKEN_IMAGE;
        image_mask[i] = 1u;
    }
    uocr_image_view view;
    memset(&view, 0, sizeof(view));
    view.pixels = pixels;
    view.width = UOCR_GLOBAL_VIEW_SIZE;
    view.height = UOCR_GLOBAL_VIEW_SIZE;
    view.format = UOCR_PIXEL_F16_NCHW;
    view.kind = UOCR_VIEW_GLOBAL;

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = n_tokens;
    request.views = &view;
    request.n_views = 1u;
    request.crop_grid_w = 1u;
    request.crop_grid_h = 1u;

    CHECK(uocr_metal_context_encode_visual_features_f16(ctx,
                                                        &request,
                                                        1u,
                                                        visual,
                                                        UOCR_GLOBAL_VISUAL_TOKENS,
                                                        error,
                                                        sizeof(error)) == 0);
    CHECK(strstr(error, "requires validated fp16 vision tensor bindings") != NULL);

    free(visual);
    free(pixels);
    free(image_mask);
    free(input_ids);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_decoder_binding_cache_full_model(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    if (model_path == NULL || model_path[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH is not set; skipping decoder binding cache validation\n");
        return 0;
    }

    enum {
        EXPECTED_DECODER_BINDINGS = 3u + (UOCR_DECODER_LAYERS * 6u) + 3u +
                                    ((UOCR_DECODER_LAYERS - 1u) *
                                     (1u + 3u + UOCR_ROUTED_EXPERTS * 3u)),
        EXPECTED_VISION_BINDINGS = 2u + 2u + (UOCR_SAM_BLOCKS * 14u + 11u) +
                                   (4u + UOCR_CLIP_BLOCKS * 12u)
    };
    const int32_t input_ids[3] = {UOCR_TOKEN_BOS, 42, 77};
    const uint8_t image_mask[3] = {0u, 0u, 0u};
    int32_t generated[2] = {0, 0};
    char error[1024];
    memset(error, 0, sizeof(error));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_decoder_bindings_ready(ctx) == 1);
    CHECK(uocr_metal_context_decoder_binding_count(ctx) == EXPECTED_DECODER_BINDINGS);
    CHECK(uocr_metal_context_vision_bindings_ready(ctx) == 1);
    CHECK(uocr_metal_context_vision_binding_count(ctx) == EXPECTED_VISION_BINDINGS);
    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 8u, error, sizeof(error)) == 1);

    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 3u;
    request.max_new_tokens = 0u;
    request.slot = 0u;
    request.image_span_start = UINT32_MAX;
    request.image_span_length = 0u;
    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count == 0u);
    uint16_t final_hidden[3u * UOCR_HIDDEN_SIZE];
    memset(final_hidden, 0, sizeof(final_hidden));
    CHECK(uocr_metal_context_read_decoder_final_hidden_f16(ctx,
                                                           0u,
                                                           3u,
                                                           final_hidden,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    int saw_nonzero_hidden = 0;
    for (uint32_t i = 0u; i < 3u * UOCR_HIDDEN_SIZE; ++i) {
        if (final_hidden[i] != 0u) {
            saw_nonzero_hidden = 1;
            break;
        }
    }
    CHECK(saw_nonzero_hidden == 1);

    request.max_new_tokens = 2u;
    result.generated_ids = generated;
    result.generated_capacity = 2u;
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count >= 1u);
    CHECK(result.generated_count <= 2u);
    for (uint32_t i = 0u; i < result.generated_count; ++i) {
        CHECK(generated[i] >= 0);
        CHECK((uint32_t)generated[i] < UOCR_VOCAB_SIZE);
    }
    CHECK(result.last_token_id == (uint32_t)generated[result.generated_count - 1u]);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    return 0;
}

static int test_metal_model_mapping(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_two_tensor_uocr_model(path) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_model_view_count(ctx) == 1u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 2u);
    CHECK(uocr_metal_context_decoder_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_decoder_bindings_ready(ctx) == 0);
    CHECK(uocr_metal_context_vision_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_vision_bindings_ready(ctx) == 0);
    CHECK(strstr(uocr_metal_context_vision_binding_error(ctx), "missing") != NULL);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_model_view_info view_info;
    memset(&view_info, 0, sizeof(view_info));
    CHECK(uocr_metal_context_get_model_view_info(ctx, 0u, &view_info) == 1);
    CHECK(view_info.file_offset <= model.tensors[0].payload_offset);
    CHECK(view_info.length >= UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_tensor_binding binding;
    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 1u, &binding) == 1);
    CHECK(binding.tensor_id == 1u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[0].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 16u);

    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 2u, &binding) == 1);
    CHECK(binding.tensor_id == 2u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[1].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 32u);
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 999u, &binding) == 0);

    CHECK(uocr_metal_context_warmup_model_views(ctx, 128u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_last_warmup_bytes(ctx) == 128u);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_TRANSIENT) >= sizeof(uint32_t));

    uocr_metal_context_unmap_model(ctx);
    CHECK(uocr_metal_context_model_view_count(ctx) == 0u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_decoder_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_decoder_bindings_ready(ctx) == 0);
    CHECK(uocr_metal_context_vision_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_vision_bindings_ready(ctx) == 0);
    CHECK(strstr(uocr_metal_context_vision_binding_error(ctx), "no mapped model") != NULL);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == 0u);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_public_engine_open_initializes_metal(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);

    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.recommended_working_set_bytes == uocr_metal_recommended_working_set_size());
    CHECK(report.memory_budget_bytes == uocr_metal_default_memory_budget_bytes(report.recommended_working_set_bytes));
    CHECK(report.memory_budget_bytes > 0u);
    CHECK(report.category_live_bytes[UOCR_MEMORY_KV_CACHE] == report.estimated_kv_cache_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_PROMPT_EMBEDDINGS] == report.estimated_prompt_embeddings_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_VISION_SCRATCH] == report.estimated_vision_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_DECODER_SCRATCH] == report.estimated_decoder_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_MOE_SCRATCH] == report.estimated_moe_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_LOGITS_READBACK] == report.estimated_logits_readback_bytes);
    CHECK(report.total_live_bytes == report.estimated_kv_cache_bytes +
                                     report.estimated_prompt_embeddings_bytes +
                                     report.estimated_vision_scratch_bytes +
                                     report.estimated_decoder_scratch_bytes +
                                     report.estimated_moe_scratch_bytes +
                                     report.estimated_logits_readback_bytes);

    uocr_engine_close(engine);
    return 0;
}

static int test_public_metal_text_generation_full_model(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }
    const char *model_path = getenv("UOCR_MODEL_PATH");
    if (model_path == NULL || model_path[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH is not set; skipping public Metal text generation\n");
        return 0;
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = model_path;
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1u;
    opts.max_prompt_tokens = 8u;
    opts.max_gen_tokens = 2u;
    opts.memory_budget_bytes = UINT64_MAX;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);

    const int32_t input_ids[3] = {UOCR_TOKEN_BOS, 42, 77};
    const uint8_t image_mask[3] = {0u, 0u, 0u};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 3u;
    request.crop_grid_w = 1u;
    request.crop_grid_h = 1u;
    request.max_new_tokens = 2u;

    uocr_result *result = NULL;
    CHECK(uocr_generate_prepared(engine, &request, 1u, &result) == UOCR_OK);
    CHECK(result != NULL);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);
    CHECK(uocr_result_count(result) == 1u);
    uint32_t generated_count = 0u;
    const int32_t *generated = uocr_result_tokens(result, 0u, &generated_count);
    CHECK(generated != NULL);
    CHECK(generated_count >= 1u);
    CHECK(generated_count <= 2u);
    for (uint32_t i = 0u; i < generated_count; ++i) {
        CHECK(generated[i] >= 0);
        CHECK((uint32_t)generated[i] < UOCR_VOCAB_SIZE);
    }

    uocr_result_free(result);
    uocr_engine_close(engine);
    return 0;
}

static int test_public_metal_text_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping public Metal text generated-id parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "input_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "image_mask_u8.bin") ||
        !fixture_binary_exists(dump_dir, "generated_ids_i32.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no text generated-id fixture files; skipping public Metal text generated-id parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    int32_t *expected_ids = NULL;
    uint32_t n_expected = 0u;
    CHECK(read_prepared_tokens_python_dump(dump_dir, &input_ids, &image_mask, &n_tokens, error, sizeof(error)) == 1);
    CHECK(read_text_generated_ids_python_dump(dump_dir, &expected_ids, &n_expected, error, sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(n_expected > 0u);

    for (uint32_t i = 0u; i < n_tokens; ++i) {
        if (image_mask[i] != 0u) {
            fprintf(stderr, "public Metal text generated-id parity requires a text-only fixture; image_mask[%u]=%u\n", i, (unsigned)image_mask[i]);
            free(expected_ids);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = model_path;
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1u;
    opts.max_prompt_tokens = n_tokens;
    opts.max_gen_tokens = n_expected;
    opts.memory_budget_bytes = UINT64_MAX;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = n_tokens;
    request.crop_grid_w = 1u;
    request.crop_grid_h = 1u;
    request.max_new_tokens = n_expected;

    uocr_result *result = NULL;
    const int status = uocr_generate_prepared(engine, &request, 1u, &result);
    if (status != UOCR_OK) {
        fprintf(stderr, "public Metal text generated-id parity failed: %s\n", uocr_last_error(engine));
        uocr_engine_close(engine);
        free(expected_ids);
        free(image_mask);
        free(input_ids);
        return 1;
    }
    CHECK(result != NULL);
    CHECK(uocr_result_count(result) == 1u);

    uint32_t n_actual = 0u;
    const int32_t *actual_ids = uocr_result_tokens(result, 0u, &n_actual);
    CHECK(actual_ids != NULL);
    if (n_actual != n_expected) {
        fprintf(stderr,
                "public Metal text generated-id count mismatch: actual=%u expected=%u\n",
                n_actual,
                n_expected);
        uocr_result_free(result);
        uocr_engine_close(engine);
        free(expected_ids);
        free(image_mask);
        free(input_ids);
        return 1;
    }
    for (uint32_t i = 0u; i < n_expected; ++i) {
        if (actual_ids[i] != expected_ids[i]) {
            fprintf(stderr,
                    "public Metal text generated-id mismatch at %u: actual=%d expected=%d\n",
                    i,
                    actual_ids[i],
                    expected_ids[i]);
            uocr_result_free(result);
            uocr_engine_close(engine);
            free(expected_ids);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);
    uocr_result_free(result);
    uocr_engine_close(engine);
    free(expected_ids);
    free(image_mask);
    free(input_ids);
    return 0;
}

int main(void) {
    CHECK(strcmp(uocr_metal_backend_name(), "metal") == 0);
    if (test_metal_smoke() != 0) return 1;
    if (test_metal_compile_all_kernels() != 0) return 1;
    if (test_metal_named_scratch_buffers() != 0) return 1;
    if (test_metal_sam_patch_embed_f16() != 0) return 1;
    if (test_metal_sam_abs_pos_f16() != 0) return 1;
    if (test_metal_sam_layernorm_f16() != 0) return 1;
    if (test_metal_sam_qkv_f16() != 0) return 1;
    if (test_metal_sam_neck_conv1x1_f16() != 0) return 1;
    if (test_metal_sam_neck_conv3x3_f16() != 0) return 1;
    if (test_metal_sam_layernorm2d_f16() != 0) return 1;
    if (test_metal_sam_net2_conv3x3_stride2_f16() != 0) return 1;
    if (test_metal_sam_net3_conv3x3_stride2_f16() != 0) return 1;
    if (test_metal_clip_embed_sam_f16() != 0) return 1;
    if (test_metal_clip_abs_pos_f16() != 0) return 1;
    if (test_metal_clip_pre_layernorm_f16() != 0) return 1;
    if (test_metal_clip_layernorm_f16() != 0) return 1;
    if (test_metal_clip_qkv_f16() != 0) return 1;
    if (test_metal_clip_attention_f16() != 0) return 1;
    if (test_metal_clip_output_projection_f16() != 0) return 1;
    if (test_metal_clip_quickgelu_f16() != 0) return 1;
    if (test_metal_clip_mlp_f16() != 0) return 1;
    if (test_metal_clip_residual_add_f16() != 0) return 1;
    if (test_metal_clip_transformer_f16() != 0) return 1;
    if (test_metal_clip_sam_concat_f16() != 0) return 1;
    if (test_metal_visual_projector_f16() != 0) return 1;
    if (test_metal_sam_window_partition_f16() != 0) return 1;
    if (test_metal_sam_window_attention_f16() != 0) return 1;
    if (test_metal_sam_global_attention_f16() != 0) return 1;
    if (test_metal_sam_rel_pos_attention_f16() != 0) return 1;
    if (test_metal_sam_mlp_f16() != 0) return 1;
    if (test_metal_sam_residuals_f16() != 0) return 1;
    if (test_metal_sam_transformer_block_f16() != 0) return 1;
    if (test_metal_get_rows_f16() != 0) return 1;
    if (test_metal_get_rows_q8_0() != 0) return 1;
    if (test_metal_prompt_assembly_f16() != 0) return 1;
    if (test_metal_prompt_assembly_from_mapped_model_f16() != 0) return 1;
    if (test_metal_text_prompt_embedding_full_model_parity() != 0) return 1;
    if (test_metal_text_prompt_embedding_python_dump_parity() != 0) return 1;
    if (test_metal_image_prompt_embedding_python_dump_parity() != 0) return 1;
    if (test_metal_image_prompt_decoder_smoke_python_dump() != 0) return 1;
    if (test_metal_image_decoder_layers_python_dump_parity() != 0) return 1;
    if (test_metal_image_router_topk_python_dump_parity() != 0) return 1;
    if (test_metal_image_logits_topk_python_dump_parity() != 0) return 1;
    if (test_metal_image_generated_ids_python_dump_parity() != 0) return 1;
    if (test_metal_integrated_image_embedding_python_dump_prefill() != 0) return 1;
    if (test_metal_integrated_image_embedding_generated_ids_python_dump_parity() != 0) return 1;
    if (test_metal_text_layer0_python_dump_parity() != 0) return 1;
    if (test_metal_text_layer1_python_dump_parity() != 0) return 1;
    if (test_metal_text_remaining_layers_python_dump_parity() != 0) return 1;
    if (test_metal_text_logits_topk_python_dump_parity() != 0) return 1;
    if (test_metal_text_generated_ids_python_dump_parity() != 0) return 1;
    if (test_metal_rmsnorm_f16() != 0) return 1;
    if (test_metal_final_rmsnorm_f16() != 0) return 1;
    if (test_metal_lm_head_f16() != 0) return 1;
    if (test_metal_argmax_f32() != 0) return 1;
    if (test_metal_no_repeat_ngram_gpu_f32() != 0) return 1;
    if (test_metal_select_greedy_with_no_repeat_f32() != 0) return 1;
    if (test_metal_eos_stop_after_greedy_selection() != 0) return 1;
    if (test_metal_select_next_token_f16() != 0) return 1;
    if (test_metal_dense_f16() != 0) return 1;
    if (test_metal_dense_q8_0() != 0) return 1;
    if (test_metal_dense_q4_k() != 0) return 1;
    if (test_metal_quantized_dense_dot_edges() != 0) return 1;
    if (test_metal_attention_qkvo_f16() != 0) return 1;
    if (test_metal_attention_output_residual_f16() != 0) return 1;
    if (test_metal_dense_swiglu_f16() != 0) return 1;
    if (test_metal_moe_shared_experts_f16() != 0) return 1;
    if (test_metal_moe_shared_experts_q8_0() != 0) return 1;
    if (test_metal_moe_router_f16() != 0) return 1;
    if (test_metal_moe_selected_experts_decode_f16() != 0) return 1;
    if (test_metal_moe_selected_experts_decode_q4_k() != 0) return 1;
    if (test_metal_moe_selected_experts_prefill_f16() != 0) return 1;
    if (test_metal_moe_selected_experts_prefill_q4_k() != 0) return 1;
    if (test_metal_moe_combine_f16() != 0) return 1;
    if (test_metal_rope_qk_f16() != 0) return 1;
    if (test_metal_prefill_attention_f16() != 0) return 1;
    if (test_metal_prefill_attention_varlen_f16() != 0) return 1;
    if (test_metal_kv_cache_layout_helpers() != 0) return 1;
    if (test_metal_kv_cache_write_f16() != 0) return 1;
    if (test_metal_decode_attention_f16() != 0) return 1;
    if (test_metal_sdpa_prefill_decode_consistency_f16() != 0) return 1;
    if (test_metal_recent_decoder_primitives_stress() != 0) return 1;
    if (test_metal_runtime_arenas() != 0) return 1;
    if (test_metal_integrated_decoder_boundary() != 0) return 1;
    if (test_metal_vision_runner_requires_bindings() != 0) return 1;
    if (test_metal_decoder_binding_cache_full_model() != 0) return 1;
    if (test_metal_model_mapping() != 0) return 1;
    if (test_public_engine_open_initializes_metal() != 0) return 1;
    if (test_public_metal_text_generation_full_model() != 0) return 1;
    if (test_public_metal_text_generated_ids_python_dump_parity() != 0) return 1;
    return 0;
}
