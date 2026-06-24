#include "model/uocr_constants.h"
#include "model/uocr_format.h"

#include <stdint.h>
#include <stdio.h>

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

static int test_language_model_constants(void) {
    CHECK(UOCR_VOCAB_SIZE == 129280u);
    CHECK(UOCR_HIDDEN_SIZE == 1280u);
    CHECK(UOCR_DECODER_LAYERS == 12u);
    CHECK(UOCR_ATTENTION_HEADS == 10u);
    CHECK(UOCR_KV_HEADS == 10u);
    CHECK(UOCR_HEAD_DIM == 128u);
    CHECK(UOCR_ROPE_THETA == 10000u);
    CHECK(UOCR_MAX_POSITIONS == 32768u);
    CHECK(UOCR_GENERATED_RING_WINDOW == 128u);
    CHECK(UOCR_DENSE_LAYER0_INTERMEDIATE == 6848u);
    CHECK(UOCR_DENSE_FIRST_LAYERS == 1u);
    CHECK(UOCR_ROUTED_EXPERTS == 64u);
    CHECK(UOCR_MOE_TOP_K == 6u);
    CHECK(UOCR_MOE_EXPERT_INTERMEDIATE == 896u);
    CHECK(UOCR_MOE_SHARED_EXPERTS == 2u);
    CHECK(UOCR_MOE_SHARED_INTERMEDIATE == 1792u);
    CHECK(UOCR_HIDDEN_ACT_SILU == 1u);
    CHECK(UOCR_RMS_NORM_EPS == 1.0e-6f);
    CHECK(UOCR_ATTENTION_BIAS == 0u);
    CHECK(UOCR_ATTENTION_DROPOUT == 0.0f);
    CHECK(UOCR_USE_MLA == 0u);
    CHECK(UOCR_MOE_SCORING_SOFTMAX == 1u);
    CHECK(UOCR_MOE_TOPK_METHOD_GREEDY == 1u);
    CHECK(UOCR_MOE_ROUTED_SCALING_FACTOR == 1.0f);
    CHECK(UOCR_MOE_NORM_TOPK_PROB == 0u);

    CHECK(UOCR_HIDDEN_SIZE == UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM);
    CHECK(UOCR_HIDDEN_SIZE == UOCR_KV_HEADS * UOCR_HEAD_DIM);
    CHECK(UOCR_MOE_SHARED_INTERMEDIATE == UOCR_MOE_SHARED_EXPERTS * UOCR_MOE_EXPERT_INTERMEDIATE);
    CHECK(UOCR_DENSE_LAYER0_INTERMEDIATE > UOCR_HIDDEN_SIZE);
    CHECK(UOCR_MOE_TOP_K <= UOCR_ROUTED_EXPERTS);
    return 0;
}

static int test_tokenizer_and_visual_constants(void) {
    CHECK(UOCR_BPE_VOCAB_SIZE == 128000u);
    CHECK(UOCR_ADDED_TOKEN_COUNT == 830u);
    CHECK(UOCR_TOKEN_BOS == 0);
    CHECK(UOCR_TOKEN_EOS == 1);
    CHECK(UOCR_TOKEN_PAD == 2);
    CHECK(UOCR_TOKEN_IMAGE == 128815);

    CHECK(UOCR_GLOBAL_VIEW_SIZE == 1024u);
    CHECK(UOCR_LOCAL_VIEW_SIZE == 640u);
    CHECK(UOCR_VISION_PATCH_SIZE == 16u);
    CHECK(UOCR_DOWNSAMPLE_RATIO == 4u);
    CHECK(UOCR_GLOBAL_GRID_QUERIES == (UOCR_GLOBAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE) / UOCR_DOWNSAMPLE_RATIO);
    CHECK(UOCR_LOCAL_GRID_QUERIES == (UOCR_LOCAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE) / UOCR_DOWNSAMPLE_RATIO);
    CHECK(UOCR_GLOBAL_ROW_NEWLINE_TOKENS == UOCR_GLOBAL_GRID_QUERIES * (UOCR_GLOBAL_GRID_QUERIES + 1u));
    CHECK(UOCR_GLOBAL_VISUAL_TOKENS == UOCR_GLOBAL_ROW_NEWLINE_TOKENS + 1u);
    CHECK(uocr_global_visual_token_count() == UOCR_GLOBAL_VISUAL_TOKENS);
    CHECK(uocr_local_visual_token_count(1u, 1u) == (UOCR_LOCAL_GRID_QUERIES + 1u) * UOCR_LOCAL_GRID_QUERIES);
    CHECK(uocr_local_visual_token_count(2u, 3u) == (UOCR_LOCAL_GRID_QUERIES * 2u + 1u) * (UOCR_LOCAL_GRID_QUERIES * 3u));

    CHECK(UOCR_SAM_HIDDEN_SIZE == 768u);
    CHECK(UOCR_SAM_ATTENTION_HEADS == 12u);
    CHECK(UOCR_SAM_HEAD_DIM == 64u);
    CHECK(UOCR_SAM_QKV_SIZE == 3u * UOCR_SAM_HIDDEN_SIZE);
    CHECK(UOCR_SAM_BLOCKS == 12u);
    CHECK(UOCR_SAM_WINDOW_SIZE == 14u);
    CHECK(UOCR_SAM_FEATURE_CHANNELS == 1024u);
    CHECK(UOCR_CLIP_HIDDEN_SIZE == 1024u);
    CHECK(UOCR_CLIP_BLOCKS == 24u);
    CHECK(UOCR_CLIP_ATTENTION_HEADS == 16u);
    CHECK(UOCR_CLIP_HEAD_DIM == 64u);
    CHECK(UOCR_CLIP_QKV_SIZE == 3u * UOCR_CLIP_HIDDEN_SIZE);
    CHECK(UOCR_CLIP_MLP_INTERMEDIATE == 4096u);
    CHECK(UOCR_PROJECTOR_IN_SIZE == 2048u);
    CHECK(UOCR_PROJECTOR_IN_SIZE == UOCR_CLIP_HIDDEN_SIZE + UOCR_SAM_FEATURE_CHANNELS);

    CHECK(uocr_sam_block_uses_global_attention(0u) == 0);
    CHECK(uocr_sam_block_uses_global_attention(2u) == 1);
    CHECK(uocr_sam_block_uses_global_attention(5u) == 1);
    CHECK(uocr_sam_block_uses_global_attention(8u) == 1);
    CHECK(uocr_sam_block_uses_global_attention(11u) == 1);
    CHECK(uocr_sam_block_uses_global_attention(12u) == 0);
    return 0;
}

static int test_default_config_record_matches_constants(void) {
    const uocr_config_record cfg = uocr_default_config_record();

    CHECK(cfg.vocab_size == UOCR_VOCAB_SIZE);
    CHECK(cfg.bpe_vocab_size == UOCR_BPE_VOCAB_SIZE);
    CHECK(cfg.added_token_count == UOCR_ADDED_TOKEN_COUNT);
    CHECK(cfg.bos_token_id == (uint32_t)UOCR_TOKEN_BOS);
    CHECK(cfg.eos_token_id == (uint32_t)UOCR_TOKEN_EOS);
    CHECK(cfg.pad_token_id == (uint32_t)UOCR_TOKEN_PAD);
    CHECK(cfg.image_token_id == (uint32_t)UOCR_TOKEN_IMAGE);
    CHECK(cfg.hidden_size == UOCR_HIDDEN_SIZE);
    CHECK(cfg.decoder_layers == UOCR_DECODER_LAYERS);
    CHECK(cfg.attention_heads == UOCR_ATTENTION_HEADS);
    CHECK(cfg.kv_heads == UOCR_KV_HEADS);
    CHECK(cfg.head_dim == UOCR_HEAD_DIM);
    CHECK(cfg.rope_theta == UOCR_ROPE_THETA);
    CHECK(cfg.max_positions == UOCR_MAX_POSITIONS);
    CHECK(cfg.generated_ring_window == UOCR_GENERATED_RING_WINDOW);
    CHECK(cfg.dense_first_layers == UOCR_DENSE_FIRST_LAYERS);
    CHECK(cfg.routed_experts == UOCR_ROUTED_EXPERTS);
    CHECK(cfg.moe_top_k == UOCR_MOE_TOP_K);
    CHECK(cfg.moe_expert_intermediate == UOCR_MOE_EXPERT_INTERMEDIATE);
    CHECK(cfg.shared_experts == UOCR_MOE_SHARED_EXPERTS);
    CHECK(cfg.dense_layer0_intermediate == UOCR_DENSE_LAYER0_INTERMEDIATE);
    CHECK(cfg.projector_in == UOCR_PROJECTOR_IN_SIZE);
    CHECK(cfg.projector_out == UOCR_HIDDEN_SIZE);
    CHECK(cfg.vision_patch_size == UOCR_VISION_PATCH_SIZE);
    CHECK(cfg.downsample_ratio == UOCR_DOWNSAMPLE_RATIO);
    CHECK(cfg.supported_global_view == UOCR_GLOBAL_VIEW_SIZE);
    CHECK(cfg.supported_local_view == UOCR_LOCAL_VIEW_SIZE);
    return 0;
}

int main(void) {
    if (test_language_model_constants() != 0) return 1;
    if (test_tokenizer_and_visual_constants() != 0) return 1;
    if (test_default_config_record_matches_constants() != 0) return 1;
    return 0;
}
