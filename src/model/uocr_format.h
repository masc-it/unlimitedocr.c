#ifndef UOCR_FORMAT_H
#define UOCR_FORMAT_H

#include <stdint.h>

#include "model/uocr_constants.h"

#define UOCR_FILE_MAGIC "UOCR"
#define UOCR_FORMAT_VERSION 1u
#define UOCR_ENDIAN_MARKER 0x01020304u
#define UOCR_FILE_ALIGNMENT 4096u

typedef enum uocr_qprofile_id {
    UOCR_QPROFILE_FP16 = 1,
    UOCR_QPROFILE_DYN_Q8 = 2,
    UOCR_QPROFILE_DYN_Q4 = 3,
    UOCR_QPROFILE_CUSTOM = 255
} uocr_qprofile_id;

typedef enum uocr_section_type {
    UOCR_SECTION_CONFIG = 1,
    UOCR_SECTION_TENSOR_DIRECTORY = 2,
    UOCR_SECTION_TOKENIZER_METADATA = 3,
    UOCR_SECTION_PROVENANCE = 4,
    UOCR_SECTION_TENSOR_DATA = 5
} uocr_section_type;

#if defined(_MSC_VER)
#pragma pack(push, 1)
#define UOCR_PACKED
#else
#define UOCR_PACKED __attribute__((packed))
#endif

typedef struct UOCR_PACKED uocr_file_header {
    char magic[4];
    uint32_t version;
    uint32_t header_size;
    uint32_t endian_marker;
    uint32_t required_alignment;
    uint32_t qprofile;
    uint32_t section_count;
    uint32_t reserved0;
    uint64_t file_size;
    uint64_t section_dir_offset;
    uint8_t model_id_hash[32];
    uint8_t reserved1[32];
} uocr_file_header;

typedef struct UOCR_PACKED uocr_section_entry {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t size;
    uint64_t alignment;
} uocr_section_entry;

typedef struct UOCR_PACKED uocr_config_record {
    uint32_t vocab_size;
    uint32_t bpe_vocab_size;
    uint32_t added_token_count;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    uint32_t image_token_id;

    uint32_t hidden_size;
    uint32_t decoder_layers;
    uint32_t attention_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t max_positions;
    uint32_t generated_ring_window;

    uint32_t dense_first_layers;
    uint32_t routed_experts;
    uint32_t moe_top_k;
    uint32_t moe_expert_intermediate;
    uint32_t shared_experts;
    uint32_t dense_layer0_intermediate;

    uint32_t projector_in;
    uint32_t projector_out;
    uint32_t vision_patch_size;
    uint32_t downsample_ratio;
    uint32_t supported_global_view;
    uint32_t supported_local_view;
} uocr_config_record;

#if defined(_MSC_VER)
#pragma pack(pop)
#else
#undef UOCR_PACKED
#endif

_Static_assert(sizeof(uocr_file_header) == 112u, "unexpected uocr_file_header size");
_Static_assert(sizeof(uocr_section_entry) == 32u, "unexpected uocr_section_entry size");
_Static_assert(sizeof(uocr_config_record) == 104u, "unexpected uocr_config_record size");

static inline uocr_config_record uocr_default_config_record(void) {
    uocr_config_record cfg;
    cfg.vocab_size = UOCR_VOCAB_SIZE;
    cfg.bpe_vocab_size = UOCR_BPE_VOCAB_SIZE;
    cfg.added_token_count = UOCR_ADDED_TOKEN_COUNT;
    cfg.bos_token_id = (uint32_t)UOCR_TOKEN_BOS;
    cfg.eos_token_id = (uint32_t)UOCR_TOKEN_EOS;
    cfg.pad_token_id = (uint32_t)UOCR_TOKEN_PAD;
    cfg.image_token_id = (uint32_t)UOCR_TOKEN_IMAGE;
    cfg.hidden_size = UOCR_HIDDEN_SIZE;
    cfg.decoder_layers = UOCR_DECODER_LAYERS;
    cfg.attention_heads = UOCR_ATTENTION_HEADS;
    cfg.kv_heads = UOCR_KV_HEADS;
    cfg.head_dim = UOCR_HEAD_DIM;
    cfg.max_positions = UOCR_MAX_POSITIONS;
    cfg.generated_ring_window = UOCR_GENERATED_RING_WINDOW;
    cfg.dense_first_layers = 1u;
    cfg.routed_experts = 64u;
    cfg.moe_top_k = 6u;
    cfg.moe_expert_intermediate = 896u;
    cfg.shared_experts = 2u;
    cfg.dense_layer0_intermediate = 6848u;
    cfg.projector_in = 2048u;
    cfg.projector_out = UOCR_HIDDEN_SIZE;
    cfg.vision_patch_size = UOCR_VISION_PATCH_SIZE;
    cfg.downsample_ratio = UOCR_DOWNSAMPLE_RATIO;
    cfg.supported_global_view = UOCR_GLOBAL_VIEW_SIZE;
    cfg.supported_local_view = UOCR_LOCAL_VIEW_SIZE;
    return cfg;
}

#endif /* UOCR_FORMAT_H */
