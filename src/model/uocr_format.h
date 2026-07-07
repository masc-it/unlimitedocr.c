#ifndef UOCR_FORMAT_H
#define UOCR_FORMAT_H

#include <stdint.h>

#include "model/uocr_constants.h"

#define UOCR_FILE_MAGIC "UOCR"
#define UOCR_FORMAT_VERSION 1u
#define UOCR_ENDIAN_MARKER 0x01020304u
#define UOCR_FILE_ALIGNMENT 4096u
#define UOCR_TENSOR_DATA_ALIGNMENT 4096u
#define UOCR_TENSOR_PAYLOAD_ALIGNMENT 16u

typedef enum uocr_qprofile_id {
    UOCR_QPROFILE_FP16 = 1,
    UOCR_QPROFILE_MIXED_Q8_0 = 2
} uocr_qprofile_id;

typedef enum uocr_section_type {
    UOCR_SECTION_CONFIG = 1,
    UOCR_SECTION_TENSOR_DIRECTORY = 2,
    UOCR_SECTION_TOKENIZER_METADATA = 3,
    UOCR_SECTION_PROVENANCE = 4,
    UOCR_SECTION_TENSOR_DATA = 5
} uocr_section_type;

typedef enum uocr_tokenizer_metadata_flags {
    UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED = 1u << 0,
    UOCR_TOKENIZER_FLAG_EMBEDDED_PAYLOAD = 1u << 1
} uocr_tokenizer_metadata_flags;

typedef enum uocr_provenance_flags {
    UOCR_PROVENANCE_FLAG_JSON_PAYLOAD = 1u << 0
} uocr_provenance_flags;

typedef enum uocr_tensor_qtype {
    UOCR_TENSOR_F16 = 1,
    UOCR_TENSOR_F32 = 2,
    UOCR_TENSOR_Q8_0 = 3
} uocr_tensor_qtype;

#define UOCR_Q8_GROUP_SIZE_DEFAULT 64u
#define UOCR_Q8_MIN (-127)
#define UOCR_Q8_MAX 127

static inline uint64_t uocr_q8_0_qweight_bytes(uint32_t rows, uint32_t cols) {
    return (uint64_t)rows * (uint64_t)cols;
}

static inline uint64_t uocr_q8_0_qscale_bytes(uint32_t rows, uint32_t cols, uint32_t group_size) {
    if (group_size == 0u || (cols % group_size) != 0u) {
        return 0u;
    }
    return (uint64_t)rows * (uint64_t)(cols / group_size) * (uint64_t)sizeof(uint16_t);
}

static inline uint64_t uocr_q8_0_total_bytes(uint32_t rows, uint32_t cols, uint32_t group_size) {
    if (group_size == 0u || (cols % group_size) != 0u) {
        return 0u;
    }
    return uocr_q8_0_qweight_bytes(rows, cols) + uocr_q8_0_qscale_bytes(rows, cols, group_size);
}

typedef enum uocr_tensor_usage {
    UOCR_TENSOR_USAGE_RUNTIME = 1,
    UOCR_TENSOR_USAGE_PRESERVED_UNUSED = 2,
    UOCR_TENSOR_USAGE_OMITTED_WITH_REASON = 3
} uocr_tensor_usage;

typedef enum uocr_tensor_flags {
    UOCR_TENSOR_FLAG_ROW_MAJOR = 1u << 0,
    UOCR_TENSOR_FLAG_TRANSPOSED = 1u << 1,
    UOCR_TENSOR_FLAG_FLATTENED_LEADING_DIM = 1u << 2
} uocr_tensor_flags;

typedef enum uocr_tensor_qtype_reason {
    UOCR_TENSOR_QTYPE_REASON_UNKNOWN = 0,
    UOCR_TENSOR_QTYPE_REASON_FP16_BASELINE = 1
} uocr_tensor_qtype_reason;

typedef enum uocr_tensor_promotion_reason {
    UOCR_TENSOR_PROMOTION_NONE = 0
} uocr_tensor_promotion_reason;

typedef enum uocr_tensor_family {
    UOCR_TENSOR_FAMILY_UNKNOWN = 0,
    UOCR_TENSOR_FAMILY_TOK_EMBED = 1,
    UOCR_TENSOR_FAMILY_LM_HEAD = 2,
    UOCR_TENSOR_FAMILY_FINAL_NORM = 3,
    UOCR_TENSOR_FAMILY_LAYER_ATTN = 4,
    UOCR_TENSOR_FAMILY_LAYER_NORM = 5,
    UOCR_TENSOR_FAMILY_LAYER_DENSE_MLP = 6,
    UOCR_TENSOR_FAMILY_MOE_ROUTER = 7,
    UOCR_TENSOR_FAMILY_MOE_EXPERT = 8,
    UOCR_TENSOR_FAMILY_MOE_SHARED = 9,
    UOCR_TENSOR_FAMILY_VISION_SAM = 10,
    UOCR_TENSOR_FAMILY_VISION_CLIP = 11,
    UOCR_TENSOR_FAMILY_PROJECTOR = 12,
    UOCR_TENSOR_FAMILY_IMAGE_NEWLINE = 13,
    UOCR_TENSOR_FAMILY_VIEW_SEPARATOR = 14
} uocr_tensor_family;

typedef enum uocr_tensor_projection {
    UOCR_TENSOR_PROJ_NONE = 0,
    UOCR_TENSOR_PROJ_WEIGHT = 1,
    UOCR_TENSOR_PROJ_BIAS = 2,
    UOCR_TENSOR_PROJ_Q = 10,
    UOCR_TENSOR_PROJ_K = 11,
    UOCR_TENSOR_PROJ_V = 12,
    UOCR_TENSOR_PROJ_O = 13,
    UOCR_TENSOR_PROJ_GATE = 20,
    UOCR_TENSOR_PROJ_UP = 21,
    UOCR_TENSOR_PROJ_DOWN = 22,
    UOCR_TENSOR_PROJ_VISION_ATTN_QKV = 30,
    UOCR_TENSOR_PROJ_VISION_ATTN_O = 31,
    UOCR_TENSOR_PROJ_VISION_MLP_FC1 = 32,
    UOCR_TENSOR_PROJ_VISION_MLP_FC2 = 33,
    UOCR_TENSOR_PROJ_VISION_PATCH_EMBED = 34,
    UOCR_TENSOR_PROJ_VISION_CONV_1X1 = 35,
    UOCR_TENSOR_PROJ_VISION_CONV_3X3 = 36
} uocr_tensor_projection;

#define UOCR_TOKENIZER_METADATA_MAGIC 0x4b4f5455u /* "UTOK" little-endian */
#define UOCR_PROVENANCE_MAGIC 0x564f5250u /* "PROV" little-endian */
#define UOCR_TENSOR_DIR_MAGIC 0x52494454u /* "TDIR" little-endian */
#define UOCR_TENSOR_MAX_DIMS 4u

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
    uint32_t rope_theta;
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

typedef struct UOCR_PACKED uocr_tokenizer_metadata_record {
    uint32_t magic;
    uint32_t version;
    uint32_t record_size;
    uint32_t flags;
    uint8_t tokenizer_hash[32];
    uint32_t vocab_size;
    uint32_t bpe_vocab_size;
    uint32_t added_token_count;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    uint32_t image_token_id;
    uint64_t tokenizer_json_offset;
    uint64_t tokenizer_json_size;
} uocr_tokenizer_metadata_record;

typedef struct UOCR_PACKED uocr_provenance_record {
    uint32_t magic;
    uint32_t version;
    uint32_t record_size;
    uint32_t flags;
    uint32_t source_tensor_count;
    uint32_t runtime_tensor_count;
    uint32_t preserved_unused_tensor_count;
    uint32_t omitted_tensor_count;
    uint32_t safetensors_file_count;
    uint32_t qprofile;
    uint32_t converter_version_major;
    uint32_t converter_version_minor;
    uint32_t converter_version_patch;
    uint32_t reserved0;
    uint8_t config_hash[32];
    uint8_t tokenizer_hash[32];
    uint8_t safetensors_index_hash[32];
    uint8_t reserved_hash[32];
    uint64_t json_offset;
    uint64_t json_size;
} uocr_provenance_record;

typedef struct UOCR_PACKED uocr_tensor_directory_header {
    uint32_t magic;
    uint32_t version;
    uint32_t entry_size;
    uint32_t tensor_count;
} uocr_tensor_directory_header;

typedef struct UOCR_PACKED uocr_tensor_entry {
    uint32_t id;
    uint32_t family;
    int32_t layer;
    int32_t expert;
    uint32_t projection;
    uint32_t usage;
    uint32_t qtype;
    uint32_t flags;
    uint32_t rank;
    uint32_t logical_shape[UOCR_TENSOR_MAX_DIMS];
    uint32_t physical_shape[UOCR_TENSOR_MAX_DIMS];
    uint64_t payload_offset;
    uint64_t payload_size;
    uint32_t block_size;
    uint32_t row_size;
    uint64_t scale_offset;
    uint64_t scale_size;
    uint64_t min_offset;
    uint64_t min_size;
    uint32_t qtype_reason;
    uint32_t promotion_reason;
    uint32_t source_name_offset;
    uint32_t source_name_size;
} uocr_tensor_entry;

#if defined(_MSC_VER)
#pragma pack(pop)
#else
#undef UOCR_PACKED
#endif

_Static_assert(sizeof(uocr_file_header) == 112u, "unexpected uocr_file_header size");
_Static_assert(sizeof(uocr_section_entry) == 32u, "unexpected uocr_section_entry size");
_Static_assert(sizeof(uocr_config_record) == 108u, "unexpected uocr_config_record size");
_Static_assert(sizeof(uocr_tokenizer_metadata_record) == 92u, "unexpected uocr_tokenizer_metadata_record size");
_Static_assert(sizeof(uocr_provenance_record) == 200u, "unexpected uocr_provenance_record size");
_Static_assert(sizeof(uocr_tensor_directory_header) == 16u, "unexpected uocr_tensor_directory_header size");
_Static_assert(sizeof(uocr_tensor_entry) == 140u, "unexpected uocr_tensor_entry size");

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
    cfg.rope_theta = UOCR_ROPE_THETA;
    cfg.max_positions = UOCR_MAX_POSITIONS;
    cfg.generated_ring_window = UOCR_GENERATED_RING_WINDOW;
    cfg.dense_first_layers = UOCR_DENSE_FIRST_LAYERS;
    cfg.routed_experts = UOCR_ROUTED_EXPERTS;
    cfg.moe_top_k = UOCR_MOE_TOP_K;
    cfg.moe_expert_intermediate = UOCR_MOE_EXPERT_INTERMEDIATE;
    cfg.shared_experts = UOCR_MOE_SHARED_EXPERTS;
    cfg.dense_layer0_intermediate = UOCR_DENSE_LAYER0_INTERMEDIATE;
    cfg.projector_in = UOCR_PROJECTOR_IN_SIZE;
    cfg.projector_out = UOCR_HIDDEN_SIZE;
    cfg.vision_patch_size = UOCR_VISION_PATCH_SIZE;
    cfg.downsample_ratio = UOCR_DOWNSAMPLE_RATIO;
    cfg.supported_global_view = UOCR_GLOBAL_VIEW_SIZE;
    cfg.supported_local_view = UOCR_LOCAL_VIEW_SIZE;
    return cfg;
}

#endif /* UOCR_FORMAT_H */
