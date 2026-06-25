#ifndef UOCR_TENSOR_REGISTRY_H
#define UOCR_TENSOR_REGISTRY_H

#include <stdint.h>

#include "model/uocr_constants.h"
#include "model/uocr_format.h"

typedef uint32_t uocr_tensor_id;

enum {
    UOCR_TENSOR_ID_INVALID = 0u,

    UOCR_TENSOR_ID_TOK_EMBED = 1u,
    UOCR_TENSOR_ID_LM_HEAD = 2u,
    UOCR_TENSOR_ID_FINAL_NORM = 3u,
    UOCR_TENSOR_ID_IMAGE_NEWLINE = 4u,
    UOCR_TENSOR_ID_VIEW_SEPARATOR = 5u,
    UOCR_TENSOR_ID_PROJECTOR_WEIGHT = 6u,
    UOCR_TENSOR_ID_PROJECTOR_BIAS = 7u,

    UOCR_TENSOR_ID_LAYER_BASE = 10000u,
    UOCR_TENSOR_ID_LAYER_STRIDE = 1000u,
    UOCR_TENSOR_ID_LAYER_INPUT_NORM = 1u,
    UOCR_TENSOR_ID_LAYER_POST_ATTN_NORM = 2u,
    UOCR_TENSOR_ID_LAYER_ATTN_Q = 10u,
    UOCR_TENSOR_ID_LAYER_ATTN_K = 11u,
    UOCR_TENSOR_ID_LAYER_ATTN_V = 12u,
    UOCR_TENSOR_ID_LAYER_ATTN_O = 13u,
    UOCR_TENSOR_ID_LAYER_DENSE_GATE = 30u,
    UOCR_TENSOR_ID_LAYER_DENSE_UP = 31u,
    UOCR_TENSOR_ID_LAYER_DENSE_DOWN = 32u,
    UOCR_TENSOR_ID_LAYER_MOE_ROUTER = 40u,
    UOCR_TENSOR_ID_LAYER_MOE_SHARED_GATE = 50u,
    UOCR_TENSOR_ID_LAYER_MOE_SHARED_UP = 51u,
    UOCR_TENSOR_ID_LAYER_MOE_SHARED_DOWN = 52u,
    UOCR_TENSOR_ID_LAYER_MOE_EXPERT_BASE = 100u,
    UOCR_TENSOR_ID_LAYER_MOE_EXPERT_STRIDE = 3u,
    UOCR_TENSOR_ID_LAYER_MOE_EXPERT_GATE = 0u,
    UOCR_TENSOR_ID_LAYER_MOE_EXPERT_UP = 1u,
    UOCR_TENSOR_ID_LAYER_MOE_EXPERT_DOWN = 2u,

    UOCR_TENSOR_ID_VISION_SAM_BASE = 100000u,
    UOCR_TENSOR_ID_VISION_CLIP_BASE = 200000u,
    UOCR_TENSOR_ID_VISION_CLIP_UNUSED_PATCH_EMBED_WEIGHT = 200001u
};

static inline uocr_tensor_id uocr_tensor_id_layer(uint32_t layer, uint32_t local_id) {
    return UOCR_TENSOR_ID_LAYER_BASE + layer * UOCR_TENSOR_ID_LAYER_STRIDE + local_id;
}

static inline uocr_tensor_id uocr_tensor_id_layer_input_norm(uint32_t layer) {
    return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_INPUT_NORM);
}

static inline uocr_tensor_id uocr_tensor_id_layer_post_attn_norm(uint32_t layer) {
    return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_POST_ATTN_NORM);
}

static inline uocr_tensor_id uocr_tensor_id_layer_attn(uint32_t layer, uint32_t projection) {
    switch (projection) {
        case UOCR_TENSOR_PROJ_Q:
            return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_ATTN_Q);
        case UOCR_TENSOR_PROJ_K:
            return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_ATTN_K);
        case UOCR_TENSOR_PROJ_V:
            return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_ATTN_V);
        case UOCR_TENSOR_PROJ_O:
            return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_ATTN_O);
        default:
            return UOCR_TENSOR_ID_INVALID;
    }
}

static inline uocr_tensor_id uocr_tensor_id_layer_dense_mlp(uint32_t projection) {
    switch (projection) {
        case UOCR_TENSOR_PROJ_GATE:
            return uocr_tensor_id_layer(0u, UOCR_TENSOR_ID_LAYER_DENSE_GATE);
        case UOCR_TENSOR_PROJ_UP:
            return uocr_tensor_id_layer(0u, UOCR_TENSOR_ID_LAYER_DENSE_UP);
        case UOCR_TENSOR_PROJ_DOWN:
            return uocr_tensor_id_layer(0u, UOCR_TENSOR_ID_LAYER_DENSE_DOWN);
        default:
            return UOCR_TENSOR_ID_INVALID;
    }
}

static inline uocr_tensor_id uocr_tensor_id_moe_router(uint32_t layer) {
    return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_MOE_ROUTER);
}

static inline uocr_tensor_id uocr_tensor_id_moe_shared(uint32_t layer, uint32_t projection) {
    switch (projection) {
        case UOCR_TENSOR_PROJ_GATE:
            return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_MOE_SHARED_GATE);
        case UOCR_TENSOR_PROJ_UP:
            return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_MOE_SHARED_UP);
        case UOCR_TENSOR_PROJ_DOWN:
            return uocr_tensor_id_layer(layer, UOCR_TENSOR_ID_LAYER_MOE_SHARED_DOWN);
        default:
            return UOCR_TENSOR_ID_INVALID;
    }
}

static inline uocr_tensor_id uocr_tensor_id_moe_expert(uint32_t layer, uint32_t expert, uint32_t projection) {
    uint32_t projection_offset;
    switch (projection) {
        case UOCR_TENSOR_PROJ_GATE:
            projection_offset = UOCR_TENSOR_ID_LAYER_MOE_EXPERT_GATE;
            break;
        case UOCR_TENSOR_PROJ_UP:
            projection_offset = UOCR_TENSOR_ID_LAYER_MOE_EXPERT_UP;
            break;
        case UOCR_TENSOR_PROJ_DOWN:
            projection_offset = UOCR_TENSOR_ID_LAYER_MOE_EXPERT_DOWN;
            break;
        default:
            return UOCR_TENSOR_ID_INVALID;
    }
    return uocr_tensor_id_layer(layer,
                                UOCR_TENSOR_ID_LAYER_MOE_EXPERT_BASE +
                                    expert * UOCR_TENSOR_ID_LAYER_MOE_EXPERT_STRIDE +
                                    projection_offset);
}

#endif /* UOCR_TENSOR_REGISTRY_H */
