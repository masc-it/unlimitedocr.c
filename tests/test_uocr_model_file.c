#include "model/uocr_model_file.h"
#include "model/uocr_tensor_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

typedef enum synthetic_header_corruption {
    SYNTHETIC_HEADER_OK = 0,
    SYNTHETIC_HEADER_BAD_ENDIAN = 1,
    SYNTHETIC_HEADER_ZERO_ALIGNMENT = 2,
    SYNTHETIC_HEADER_UNALIGNED_SECTION_DIR = 3
} synthetic_header_corruption;

static int write_synthetic_uocr_header_case(const char *path,
                                            int corrupt_config,
                                            synthetic_header_corruption corruption) {
    const uint64_t section_dir_offset = sizeof(uocr_file_header);
    const uint64_t written_section_dir_offset = section_dir_offset;
    const uint64_t config_offset = written_section_dir_offset + sizeof(uocr_section_entry);
    const uint64_t file_size = config_offset + sizeof(uocr_config_record);

    uocr_file_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, UOCR_FILE_MAGIC, 4u);
    header.version = UOCR_FORMAT_VERSION;
    header.header_size = (uint32_t)sizeof(uocr_file_header);
    header.endian_marker = corruption == SYNTHETIC_HEADER_BAD_ENDIAN ? 0x04030201u : UOCR_ENDIAN_MARKER;
    header.required_alignment = corruption == SYNTHETIC_HEADER_ZERO_ALIGNMENT ? 0u : 8u;
    header.qprofile = UOCR_QPROFILE_FP16;
    header.section_count = 1u;
    header.file_size = file_size;
    header.section_dir_offset = corruption == SYNTHETIC_HEADER_UNALIGNED_SECTION_DIR ? section_dir_offset + 4u
                                                                                     : section_dir_offset;

    uocr_section_entry section;
    memset(&section, 0, sizeof(section));
    section.type = UOCR_SECTION_CONFIG;
    section.offset = config_offset;
    section.size = sizeof(uocr_config_record);
    section.alignment = 8u;

    uocr_config_record config = uocr_default_config_record();
    if (corrupt_config) {
        config.vocab_size = 1u;
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        perror("fopen");
        return 1;
    }
    int failed = 0;
    failed |= fwrite(&header, 1u, sizeof(header), f) != sizeof(header);
    failed |= fwrite(&section, 1u, sizeof(section), f) != sizeof(section);
    failed |= fwrite(&config, 1u, sizeof(config), f) != sizeof(config);
    failed |= fclose(f) != 0;
    return failed ? 1 : 0;
}

static int write_synthetic_uocr(const char *path, int corrupt_config) {
    return write_synthetic_uocr_header_case(path, corrupt_config, SYNTHETIC_HEADER_OK);
}

static uint64_t align_up_u64(uint64_t value, uint64_t align) {
    const uint64_t rem = value % align;
    return rem == 0u ? value : value + (align - rem);
}

static void fill_hash(uint8_t hash[32], uint8_t seed) {
    for (uint32_t i = 0u; i < 32u; ++i) {
        hash[i] = (uint8_t)(seed + i);
    }
}

static int make_virtual_full_fp16_model(uocr_model_file *model,
                                        uocr_file_header *header,
                                        uocr_section_entry *tensor_data,
                                        uocr_provenance_record *provenance,
                                        uocr_tensor_entry **owned_tensors) {
    if (model == NULL || header == NULL || tensor_data == NULL || provenance == NULL || owned_tensors == NULL) {
        return 1;
    }
    memset(model, 0, sizeof(*model));
    memset(header, 0, sizeof(*header));
    memset(tensor_data, 0, sizeof(*tensor_data));
    memset(provenance, 0, sizeof(*provenance));
#if !defined(_WIN32)
    model->fd = -1;
#endif

    uocr_tensor_entry *tensors = (uocr_tensor_entry *)calloc(UOCR_SOURCE_TENSOR_COUNT, sizeof(*tensors));
    if (tensors == NULL) {
        return 1;
    }

    const uint64_t base_payload = UOCR_FP16_TENSOR_PAYLOAD_BYTES / UOCR_SOURCE_TENSOR_COUNT;
    const uint64_t payload_remainder = UOCR_FP16_TENSOR_PAYLOAD_BYTES % UOCR_SOURCE_TENSOR_COUNT;
    for (uint32_t i = 0u; i < UOCR_SOURCE_TENSOR_COUNT; ++i) {
        tensors[i].id = i + 1u;
        tensors[i].family = UOCR_TENSOR_FAMILY_FINAL_NORM;
        tensors[i].layer = -1;
        tensors[i].expert = -1;
        tensors[i].projection = UOCR_TENSOR_PROJ_WEIGHT;
        tensors[i].usage = UOCR_TENSOR_USAGE_RUNTIME;
        tensors[i].qtype = UOCR_TENSOR_F16;
        tensors[i].rank = 1u;
        tensors[i].logical_shape[0] = 1u;
        tensors[i].physical_shape[0] = 1u;
        tensors[i].payload_offset = i * base_payload;
        tensors[i].payload_size = base_payload;
    }
    tensors[UOCR_SOURCE_TENSOR_COUNT - 1u].id = UOCR_TENSOR_ID_VISION_CLIP_UNUSED_PATCH_EMBED_WEIGHT;
    tensors[UOCR_SOURCE_TENSOR_COUNT - 1u].family = UOCR_TENSOR_FAMILY_VISION_CLIP;
    tensors[UOCR_SOURCE_TENSOR_COUNT - 1u].usage = UOCR_TENSOR_USAGE_PRESERVED_UNUSED;
    tensors[UOCR_SOURCE_TENSOR_COUNT - 1u].payload_size = base_payload + payload_remainder;

    header->qprofile = UOCR_QPROFILE_FP16;
    header->section_count = 1u;
    tensor_data->type = UOCR_SECTION_TENSOR_DATA;
    tensor_data->size = UOCR_FP16_TENSOR_PAYLOAD_BYTES;
    provenance->source_tensor_count = UOCR_SOURCE_TENSOR_COUNT;
    provenance->runtime_tensor_count = UOCR_FP16_RUNTIME_TENSOR_COUNT;
    provenance->preserved_unused_tensor_count = UOCR_FP16_PRESERVED_UNUSED_TENSOR_COUNT;
    provenance->omitted_tensor_count = UOCR_FP16_OMITTED_TENSOR_COUNT;
    provenance->qprofile = UOCR_QPROFILE_FP16;

    model->header = header;
    model->sections = tensor_data;
    model->provenance = provenance;
    model->tensors = tensors;
    model->tensor_count = UOCR_SOURCE_TENSOR_COUNT;
    *owned_tensors = tensors;
    return 0;
}

typedef enum synthetic_tensor_corruption {
    SYNTHETIC_OK = 0,
    SYNTHETIC_CORRUPT_TENSOR_PAYLOAD = 1,
    SYNTHETIC_CORRUPT_TENSOR_ORDER = 2,
    SYNTHETIC_CORRUPT_TENSOR_DATA_ALIGNMENT = 3,
    SYNTHETIC_CORRUPT_TOKENIZER = 4,
    SYNTHETIC_CORRUPT_PROVENANCE = 5,
    SYNTHETIC_CORRUPT_TENSOR_TRANSPOSE_FLAG = 6,
    SYNTHETIC_CORRUPT_TENSOR_DATA_SECTION_TOO_SMALL = 7
} synthetic_tensor_corruption;

static int write_synthetic_uocr_with_tensors(const char *path, synthetic_tensor_corruption corruption) {
    static const char provenance_json[] = "{\"source\":\"synthetic\"}";
    const uint64_t section_dir_offset = sizeof(uocr_file_header);
    const uint32_t section_count = 5u;
    const uint64_t config_offset = section_dir_offset + section_count * sizeof(uocr_section_entry);
    const uint64_t tokenizer_offset = align_up_u64(config_offset + sizeof(uocr_config_record), 8u);
    const uint64_t tokenizer_size = sizeof(uocr_tokenizer_metadata_record);
    const uint64_t provenance_offset = align_up_u64(tokenizer_offset + tokenizer_size, 8u);
    const uint64_t provenance_size = sizeof(uocr_provenance_record) + sizeof(provenance_json) - 1u;
    const uint64_t tensor_dir_offset = align_up_u64(provenance_offset + provenance_size, 8u);
    const uint64_t tensor_dir_size = sizeof(uocr_tensor_directory_header) + 2u * sizeof(uocr_tensor_entry);
    const uint64_t tensor_data_offset = align_up_u64(tensor_dir_offset + tensor_dir_size, UOCR_TENSOR_DATA_ALIGNMENT);
    const uint64_t tensor_data_size = 32u;
    const uint64_t file_size = tensor_data_offset + tensor_data_size;

    uint8_t *buffer = (uint8_t *)calloc(1u, (size_t)file_size);
    if (buffer == NULL) {
        return 1;
    }

    uocr_file_header *header = (uocr_file_header *)(void *)buffer;
    memcpy(header->magic, UOCR_FILE_MAGIC, 4u);
    header->version = UOCR_FORMAT_VERSION;
    header->header_size = (uint32_t)sizeof(uocr_file_header);
    header->endian_marker = UOCR_ENDIAN_MARKER;
    header->required_alignment = UOCR_TENSOR_DATA_ALIGNMENT;
    header->qprofile = UOCR_QPROFILE_FP16;
    header->section_count = section_count;
    header->file_size = file_size;
    header->section_dir_offset = section_dir_offset;

    uocr_section_entry *sections = (uocr_section_entry *)(void *)(buffer + section_dir_offset);
    sections[0].type = UOCR_SECTION_CONFIG;
    sections[0].offset = config_offset;
    sections[0].size = sizeof(uocr_config_record);
    sections[0].alignment = 8u;
    sections[1].type = UOCR_SECTION_TOKENIZER_METADATA;
    sections[1].offset = tokenizer_offset;
    sections[1].size = tokenizer_size;
    sections[1].alignment = 8u;
    sections[2].type = UOCR_SECTION_PROVENANCE;
    sections[2].offset = provenance_offset;
    sections[2].size = provenance_size;
    sections[2].alignment = 8u;
    sections[3].type = UOCR_SECTION_TENSOR_DIRECTORY;
    sections[3].offset = tensor_dir_offset;
    sections[3].size = tensor_dir_size;
    sections[3].alignment = 8u;
    sections[4].type = UOCR_SECTION_TENSOR_DATA;
    sections[4].offset = tensor_data_offset;
    sections[4].size = corruption == SYNTHETIC_CORRUPT_TENSOR_DATA_SECTION_TOO_SMALL ? 16u : tensor_data_size;
    sections[4].alignment = corruption == SYNTHETIC_CORRUPT_TENSOR_DATA_ALIGNMENT ? 8u : UOCR_TENSOR_DATA_ALIGNMENT;

    uocr_config_record config = uocr_default_config_record();
    memcpy(buffer + config_offset, &config, sizeof(config));

    uocr_tokenizer_metadata_record *tokenizer = (uocr_tokenizer_metadata_record *)(void *)(buffer + tokenizer_offset);
    tokenizer->magic = UOCR_TOKENIZER_METADATA_MAGIC;
    tokenizer->version = UOCR_FORMAT_VERSION;
    tokenizer->record_size = (uint32_t)sizeof(uocr_tokenizer_metadata_record);
    tokenizer->flags = UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED;
    fill_hash(tokenizer->tokenizer_hash, 0x10u);
    tokenizer->vocab_size = UOCR_VOCAB_SIZE;
    tokenizer->bpe_vocab_size = UOCR_BPE_VOCAB_SIZE;
    tokenizer->added_token_count = UOCR_ADDED_TOKEN_COUNT;
    tokenizer->bos_token_id = (uint32_t)UOCR_TOKEN_BOS;
    tokenizer->eos_token_id = (uint32_t)UOCR_TOKEN_EOS;
    tokenizer->pad_token_id = (uint32_t)UOCR_TOKEN_PAD;
    tokenizer->image_token_id = corruption == SYNTHETIC_CORRUPT_TOKENIZER ? 7u : (uint32_t)UOCR_TOKEN_IMAGE;

    uocr_provenance_record *provenance = (uocr_provenance_record *)(void *)(buffer + provenance_offset);
    provenance->magic = UOCR_PROVENANCE_MAGIC;
    provenance->version = UOCR_FORMAT_VERSION;
    provenance->record_size = (uint32_t)sizeof(uocr_provenance_record);
    provenance->flags = UOCR_PROVENANCE_FLAG_JSON_PAYLOAD;
    provenance->source_tensor_count = 2u;
    provenance->runtime_tensor_count = corruption == SYNTHETIC_CORRUPT_PROVENANCE ? 2u : 1u;
    provenance->preserved_unused_tensor_count = 1u;
    provenance->omitted_tensor_count = 0u;
    provenance->safetensors_file_count = 1u;
    provenance->qprofile = UOCR_QPROFILE_FP16;
    provenance->converter_version_major = 0u;
    provenance->converter_version_minor = 1u;
    provenance->converter_version_patch = 0u;
    fill_hash(provenance->config_hash, 0x30u);
    fill_hash(provenance->tokenizer_hash, 0x10u);
    fill_hash(provenance->safetensors_index_hash, 0x50u);
    provenance->json_offset = sizeof(uocr_provenance_record);
    provenance->json_size = sizeof(provenance_json) - 1u;
    memcpy(buffer + provenance_offset + provenance->json_offset, provenance_json, sizeof(provenance_json) - 1u);

    uocr_tensor_directory_header *dir = (uocr_tensor_directory_header *)(void *)(buffer + tensor_dir_offset);
    dir->magic = UOCR_TENSOR_DIR_MAGIC;
    dir->version = UOCR_FORMAT_VERSION;
    dir->entry_size = (uint32_t)sizeof(uocr_tensor_entry);
    dir->tensor_count = 2u;

    uocr_tensor_entry *tensors = (uocr_tensor_entry *)(void *)(buffer + tensor_dir_offset + sizeof(*dir));
    tensors[0].id = UOCR_TENSOR_ID_TOK_EMBED;
    tensors[0].family = UOCR_TENSOR_FAMILY_TOK_EMBED;
    tensors[0].layer = -1;
    tensors[0].expert = -1;
    tensors[0].projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensors[0].usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensors[0].qtype = UOCR_TENSOR_F16;
    tensors[0].flags = corruption == SYNTHETIC_CORRUPT_TENSOR_TRANSPOSE_FLAG
                           ? UOCR_TENSOR_FLAG_TRANSPOSED
                           : UOCR_TENSOR_FLAG_ROW_MAJOR;
    tensors[0].rank = 2u;
    tensors[0].logical_shape[0] = 2u;
    tensors[0].logical_shape[1] = 4u;
    tensors[0].physical_shape[0] = 2u;
    tensors[0].physical_shape[1] = 4u;
    tensors[0].payload_offset = tensor_data_offset;
    tensors[0].payload_size = 16u;

    tensors[1].id = corruption == SYNTHETIC_CORRUPT_TENSOR_ORDER ? UOCR_TENSOR_ID_TOK_EMBED : UOCR_TENSOR_ID_VISION_CLIP_BASE;
    tensors[1].family = UOCR_TENSOR_FAMILY_VISION_CLIP;
    tensors[1].layer = -1;
    tensors[1].expert = -1;
    tensors[1].projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensors[1].usage = UOCR_TENSOR_USAGE_PRESERVED_UNUSED;
    tensors[1].qtype = UOCR_TENSOR_F16;
    tensors[1].flags = UOCR_TENSOR_FLAG_ROW_MAJOR;
    tensors[1].rank = 1u;
    tensors[1].logical_shape[0] = 8u;
    tensors[1].physical_shape[0] = 8u;
    tensors[1].payload_offset = corruption == SYNTHETIC_CORRUPT_TENSOR_PAYLOAD
                                    ? align_up_u64(file_size + 1u, UOCR_TENSOR_PAYLOAD_ALIGNMENT)
                                    : tensor_data_offset + 16u;
    tensors[1].payload_size = 16u;

    for (uint32_t i = 0u; i < tensor_data_size; ++i) {
        buffer[tensor_data_offset + i] = (uint8_t)i;
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(buffer);
        perror("fopen");
        return 1;
    }
    int failed = fwrite(buffer, 1u, (size_t)file_size, f) != (size_t)file_size;
    failed |= fclose(f) != 0;
    free(buffer);
    return failed ? 1 : 0;
}

typedef enum synthetic_q8_corruption {
    SYNTHETIC_Q8_OK = 0,
    SYNTHETIC_Q8_BAD_SCALE_ALIGNMENT = 1
} synthetic_q8_corruption;

static int write_synthetic_mixed_q8_uocr(const char *path, synthetic_q8_corruption corruption) {
    static const char provenance_json[] =
        "{\"quantization\":{\"mode\":\"mixed-q8_0\",\"group_size\":64,\"policy\":\"embeddings+decoder\"}}";
    const uint64_t section_dir_offset = sizeof(uocr_file_header);
    const uint32_t section_count = 5u;
    const uint64_t config_offset = section_dir_offset + section_count * sizeof(uocr_section_entry);
    const uint64_t tokenizer_offset = align_up_u64(config_offset + sizeof(uocr_config_record), 8u);
    const uint64_t tokenizer_size = sizeof(uocr_tokenizer_metadata_record);
    const uint64_t provenance_offset = align_up_u64(tokenizer_offset + tokenizer_size, 8u);
    const uint64_t provenance_size = sizeof(uocr_provenance_record) + sizeof(provenance_json) - 1u;
    const uint64_t tensor_dir_offset = align_up_u64(provenance_offset + provenance_size, 8u);
    const uint64_t tensor_dir_size = sizeof(uocr_tensor_directory_header) + sizeof(uocr_tensor_entry);
    const uint64_t tensor_data_offset = align_up_u64(tensor_dir_offset + tensor_dir_size, UOCR_TENSOR_DATA_ALIGNMENT);
    const uint64_t qweight_size = 2u * 64u;
    const uint64_t qscale_size = 2u * sizeof(uint16_t);
    const uint64_t qscale_offset = tensor_data_offset + qweight_size;
    const uint64_t tensor_data_size = qweight_size + qscale_size;
    const uint64_t file_size = tensor_data_offset + tensor_data_size;

    uint8_t *buffer = (uint8_t *)calloc(1u, (size_t)file_size);
    if (buffer == NULL) {
        return 1;
    }

    uocr_file_header *header = (uocr_file_header *)(void *)buffer;
    memcpy(header->magic, UOCR_FILE_MAGIC, 4u);
    header->version = UOCR_FORMAT_VERSION;
    header->header_size = (uint32_t)sizeof(uocr_file_header);
    header->endian_marker = UOCR_ENDIAN_MARKER;
    header->required_alignment = UOCR_TENSOR_DATA_ALIGNMENT;
    header->qprofile = UOCR_QPROFILE_MIXED_Q8_0;
    header->section_count = section_count;
    header->file_size = file_size;
    header->section_dir_offset = section_dir_offset;

    uocr_section_entry *sections = (uocr_section_entry *)(void *)(buffer + section_dir_offset);
    sections[0].type = UOCR_SECTION_CONFIG;
    sections[0].offset = config_offset;
    sections[0].size = sizeof(uocr_config_record);
    sections[0].alignment = 8u;
    sections[1].type = UOCR_SECTION_TOKENIZER_METADATA;
    sections[1].offset = tokenizer_offset;
    sections[1].size = tokenizer_size;
    sections[1].alignment = 8u;
    sections[2].type = UOCR_SECTION_PROVENANCE;
    sections[2].offset = provenance_offset;
    sections[2].size = provenance_size;
    sections[2].alignment = 8u;
    sections[3].type = UOCR_SECTION_TENSOR_DIRECTORY;
    sections[3].offset = tensor_dir_offset;
    sections[3].size = tensor_dir_size;
    sections[3].alignment = 8u;
    sections[4].type = UOCR_SECTION_TENSOR_DATA;
    sections[4].offset = tensor_data_offset;
    sections[4].size = tensor_data_size;
    sections[4].alignment = UOCR_TENSOR_DATA_ALIGNMENT;

    uocr_config_record config = uocr_default_config_record();
    memcpy(buffer + config_offset, &config, sizeof(config));

    uocr_tokenizer_metadata_record *tokenizer = (uocr_tokenizer_metadata_record *)(void *)(buffer + tokenizer_offset);
    tokenizer->magic = UOCR_TOKENIZER_METADATA_MAGIC;
    tokenizer->version = UOCR_FORMAT_VERSION;
    tokenizer->record_size = (uint32_t)sizeof(uocr_tokenizer_metadata_record);
    tokenizer->flags = UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED;
    fill_hash(tokenizer->tokenizer_hash, 0x10u);
    tokenizer->vocab_size = UOCR_VOCAB_SIZE;
    tokenizer->bpe_vocab_size = UOCR_BPE_VOCAB_SIZE;
    tokenizer->added_token_count = UOCR_ADDED_TOKEN_COUNT;
    tokenizer->bos_token_id = (uint32_t)UOCR_TOKEN_BOS;
    tokenizer->eos_token_id = (uint32_t)UOCR_TOKEN_EOS;
    tokenizer->pad_token_id = (uint32_t)UOCR_TOKEN_PAD;
    tokenizer->image_token_id = (uint32_t)UOCR_TOKEN_IMAGE;

    uocr_provenance_record *provenance = (uocr_provenance_record *)(void *)(buffer + provenance_offset);
    provenance->magic = UOCR_PROVENANCE_MAGIC;
    provenance->version = UOCR_FORMAT_VERSION;
    provenance->record_size = (uint32_t)sizeof(uocr_provenance_record);
    provenance->flags = UOCR_PROVENANCE_FLAG_JSON_PAYLOAD;
    provenance->source_tensor_count = 1u;
    provenance->runtime_tensor_count = 1u;
    provenance->preserved_unused_tensor_count = 0u;
    provenance->omitted_tensor_count = 0u;
    provenance->safetensors_file_count = 1u;
    provenance->qprofile = UOCR_QPROFILE_MIXED_Q8_0;
    provenance->converter_version_major = 0u;
    provenance->converter_version_minor = 2u;
    provenance->converter_version_patch = 0u;
    fill_hash(provenance->config_hash, 0x30u);
    fill_hash(provenance->tokenizer_hash, 0x10u);
    fill_hash(provenance->safetensors_index_hash, 0x50u);
    provenance->json_offset = sizeof(uocr_provenance_record);
    provenance->json_size = sizeof(provenance_json) - 1u;
    memcpy(buffer + provenance_offset + provenance->json_offset, provenance_json, sizeof(provenance_json) - 1u);

    uocr_tensor_directory_header *dir = (uocr_tensor_directory_header *)(void *)(buffer + tensor_dir_offset);
    dir->magic = UOCR_TENSOR_DIR_MAGIC;
    dir->version = UOCR_FORMAT_VERSION;
    dir->entry_size = (uint32_t)sizeof(uocr_tensor_entry);
    dir->tensor_count = 1u;

    uocr_tensor_entry *tensor = (uocr_tensor_entry *)(void *)(buffer + tensor_dir_offset + sizeof(*dir));
    tensor->id = UOCR_TENSOR_ID_TOK_EMBED;
    tensor->family = UOCR_TENSOR_FAMILY_TOK_EMBED;
    tensor->layer = -1;
    tensor->expert = -1;
    tensor->projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensor->usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensor->qtype = UOCR_TENSOR_Q8_0;
    tensor->flags = UOCR_TENSOR_FLAG_ROW_MAJOR;
    tensor->rank = 2u;
    tensor->logical_shape[0] = 2u;
    tensor->logical_shape[1] = 64u;
    tensor->physical_shape[0] = 2u;
    tensor->physical_shape[1] = 64u;
    tensor->payload_offset = tensor_data_offset;
    tensor->payload_size = qweight_size;
    tensor->block_size = UOCR_Q8_GROUP_SIZE_DEFAULT;
    tensor->row_size = 64u;
    tensor->scale_offset = corruption == SYNTHETIC_Q8_BAD_SCALE_ALIGNMENT ? qscale_offset + 2u : qscale_offset;
    tensor->scale_size = qscale_size;
    tensor->qtype_reason = UOCR_TENSOR_QTYPE_REASON_UNKNOWN;
    tensor->promotion_reason = UOCR_TENSOR_PROMOTION_NONE;

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(buffer);
        perror("fopen");
        return 1;
    }
    int failed = fwrite(buffer, 1u, (size_t)file_size, f) != (size_t)file_size;
    failed |= fclose(f) != 0;
    free(buffer);
    return failed ? 1 : 0;
}

typedef enum synthetic_q4_corruption {
    SYNTHETIC_Q4_OK = 0,
    SYNTHETIC_Q4_BAD_ROW_SIZE = 1,
    SYNTHETIC_Q4_BAD_FAMILY = 2,
    SYNTHETIC_Q4_HAS_MIN_METADATA = 3
} synthetic_q4_corruption;

static int write_synthetic_mixed_q4_uocr(const char *path, synthetic_q4_corruption corruption) {
    static const char provenance_json[] =
        "{\"quantization\":{\"mode\":\"mixed-q4\",\"group_size\":64,\"policy\":\"embeddings+decoder\"}}";
    const uint64_t section_dir_offset = sizeof(uocr_file_header);
    const uint32_t section_count = 5u;
    const uint64_t config_offset = section_dir_offset + section_count * sizeof(uocr_section_entry);
    const uint64_t tokenizer_offset = align_up_u64(config_offset + sizeof(uocr_config_record), 8u);
    const uint64_t tokenizer_size = sizeof(uocr_tokenizer_metadata_record);
    const uint64_t provenance_offset = align_up_u64(tokenizer_offset + tokenizer_size, 8u);
    const uint64_t provenance_size = sizeof(uocr_provenance_record) + sizeof(provenance_json) - 1u;
    const uint64_t tensor_dir_offset = align_up_u64(provenance_offset + provenance_size, 8u);
    const uint64_t tensor_dir_size = sizeof(uocr_tensor_directory_header) + sizeof(uocr_tensor_entry);
    const uint64_t tensor_data_offset = align_up_u64(tensor_dir_offset + tensor_dir_size, UOCR_TENSOR_DATA_ALIGNMENT);
    /* [2, 64] expert weight: 2 rows x 32 packed bytes + 2 fp16 scales. */
    const uint64_t qweight_size = 2u * 32u;
    const uint64_t qscale_size = 2u * sizeof(uint16_t);
    const uint64_t qscale_offset = tensor_data_offset + qweight_size;
    const uint64_t tensor_data_size = qweight_size + qscale_size;
    const uint64_t file_size = tensor_data_offset + tensor_data_size;

    uint8_t *buffer = (uint8_t *)calloc(1u, (size_t)file_size);
    if (buffer == NULL) {
        return 1;
    }

    uocr_file_header *header = (uocr_file_header *)(void *)buffer;
    memcpy(header->magic, UOCR_FILE_MAGIC, 4u);
    header->version = UOCR_FORMAT_VERSION;
    header->header_size = (uint32_t)sizeof(uocr_file_header);
    header->endian_marker = UOCR_ENDIAN_MARKER;
    header->required_alignment = UOCR_TENSOR_DATA_ALIGNMENT;
    header->qprofile = UOCR_QPROFILE_MIXED_Q4;
    header->section_count = section_count;
    header->file_size = file_size;
    header->section_dir_offset = section_dir_offset;

    uocr_section_entry *sections = (uocr_section_entry *)(void *)(buffer + section_dir_offset);
    sections[0].type = UOCR_SECTION_CONFIG;
    sections[0].offset = config_offset;
    sections[0].size = sizeof(uocr_config_record);
    sections[0].alignment = 8u;
    sections[1].type = UOCR_SECTION_TOKENIZER_METADATA;
    sections[1].offset = tokenizer_offset;
    sections[1].size = tokenizer_size;
    sections[1].alignment = 8u;
    sections[2].type = UOCR_SECTION_PROVENANCE;
    sections[2].offset = provenance_offset;
    sections[2].size = provenance_size;
    sections[2].alignment = 8u;
    sections[3].type = UOCR_SECTION_TENSOR_DIRECTORY;
    sections[3].offset = tensor_dir_offset;
    sections[3].size = tensor_dir_size;
    sections[3].alignment = 8u;
    sections[4].type = UOCR_SECTION_TENSOR_DATA;
    sections[4].offset = tensor_data_offset;
    sections[4].size = tensor_data_size;
    sections[4].alignment = UOCR_TENSOR_DATA_ALIGNMENT;

    uocr_config_record config = uocr_default_config_record();
    memcpy(buffer + config_offset, &config, sizeof(config));

    uocr_tokenizer_metadata_record *tokenizer = (uocr_tokenizer_metadata_record *)(void *)(buffer + tokenizer_offset);
    tokenizer->magic = UOCR_TOKENIZER_METADATA_MAGIC;
    tokenizer->version = UOCR_FORMAT_VERSION;
    tokenizer->record_size = (uint32_t)sizeof(uocr_tokenizer_metadata_record);
    tokenizer->flags = UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED;
    fill_hash(tokenizer->tokenizer_hash, 0x10u);
    tokenizer->vocab_size = UOCR_VOCAB_SIZE;
    tokenizer->bpe_vocab_size = UOCR_BPE_VOCAB_SIZE;
    tokenizer->added_token_count = UOCR_ADDED_TOKEN_COUNT;
    tokenizer->bos_token_id = (uint32_t)UOCR_TOKEN_BOS;
    tokenizer->eos_token_id = (uint32_t)UOCR_TOKEN_EOS;
    tokenizer->pad_token_id = (uint32_t)UOCR_TOKEN_PAD;
    tokenizer->image_token_id = (uint32_t)UOCR_TOKEN_IMAGE;

    uocr_provenance_record *provenance = (uocr_provenance_record *)(void *)(buffer + provenance_offset);
    provenance->magic = UOCR_PROVENANCE_MAGIC;
    provenance->version = UOCR_FORMAT_VERSION;
    provenance->record_size = (uint32_t)sizeof(uocr_provenance_record);
    provenance->flags = UOCR_PROVENANCE_FLAG_JSON_PAYLOAD;
    provenance->source_tensor_count = 1u;
    provenance->runtime_tensor_count = 1u;
    provenance->preserved_unused_tensor_count = 0u;
    provenance->omitted_tensor_count = 0u;
    provenance->safetensors_file_count = 1u;
    provenance->qprofile = UOCR_QPROFILE_MIXED_Q4;
    provenance->converter_version_major = 0u;
    provenance->converter_version_minor = 3u;
    provenance->converter_version_patch = 0u;
    fill_hash(provenance->config_hash, 0x30u);
    fill_hash(provenance->tokenizer_hash, 0x10u);
    fill_hash(provenance->safetensors_index_hash, 0x50u);
    provenance->json_offset = sizeof(uocr_provenance_record);
    provenance->json_size = sizeof(provenance_json) - 1u;
    memcpy(buffer + provenance_offset + provenance->json_offset, provenance_json, sizeof(provenance_json) - 1u);

    uocr_tensor_directory_header *dir = (uocr_tensor_directory_header *)(void *)(buffer + tensor_dir_offset);
    dir->magic = UOCR_TENSOR_DIR_MAGIC;
    dir->version = UOCR_FORMAT_VERSION;
    dir->entry_size = (uint32_t)sizeof(uocr_tensor_entry);
    dir->tensor_count = 1u;

    uocr_tensor_entry *tensor = (uocr_tensor_entry *)(void *)(buffer + tensor_dir_offset + sizeof(*dir));
    tensor->id = UOCR_TENSOR_ID_LAYER_MOE_EXPERT_BASE;
    tensor->family = corruption == SYNTHETIC_Q4_BAD_FAMILY ? UOCR_TENSOR_FAMILY_LAYER_ATTN
                                                           : UOCR_TENSOR_FAMILY_MOE_EXPERT;
    tensor->layer = 1;
    tensor->expert = 0;
    tensor->projection = UOCR_TENSOR_PROJ_GATE;
    tensor->usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensor->qtype = UOCR_TENSOR_Q4_0;
    tensor->flags = UOCR_TENSOR_FLAG_ROW_MAJOR;
    tensor->rank = 2u;
    tensor->logical_shape[0] = 2u;
    tensor->logical_shape[1] = 64u;
    tensor->physical_shape[0] = 2u;
    tensor->physical_shape[1] = 64u;
    tensor->payload_offset = tensor_data_offset;
    tensor->payload_size = qweight_size;
    tensor->block_size = UOCR_Q4_GROUP_SIZE_DEFAULT;
    tensor->row_size = corruption == SYNTHETIC_Q4_BAD_ROW_SIZE ? 64u : 32u;
    tensor->scale_offset = qscale_offset;
    tensor->scale_size = qscale_size;
    if (corruption == SYNTHETIC_Q4_HAS_MIN_METADATA) {
        tensor->min_offset = qscale_offset;
        tensor->min_size = qscale_size;
    }
    tensor->qtype_reason = UOCR_TENSOR_QTYPE_REASON_UNKNOWN;
    tensor->promotion_reason = UOCR_TENSOR_PROMOTION_NONE;

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(buffer);
        perror("fopen");
        return 1;
    }
    int failed = fwrite(buffer, 1u, (size_t)file_size, f) != (size_t)file_size;
    failed |= fclose(f) != 0;
    free(buffer);
    return failed ? 1 : 0;
}

static int make_temp_path(char *path, size_t path_size) {
    const char *template_path = "/tmp/uocr-model-file-XXXXXX";
    if (strlen(template_path) + 1u > path_size) {
        return 1;
    }
    strcpy(path, template_path);
    int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    close(fd);
    return 0;
}

static int test_rejects_bad_endian_marker(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_header_case(path, 0, SYNTHETIC_HEADER_BAD_ENDIAN) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "unsupported endian marker") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_zero_required_alignment(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_header_case(path, 0, SYNTHETIC_HEADER_ZERO_ALIGNMENT) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "required alignment must be non-zero") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_unaligned_section_directory(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_header_case(path, 0, SYNTHETIC_HEADER_UNALIGNED_SECTION_DIR) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "section directory offset must be 8-byte aligned") != NULL);
    unlink(path);
    return 0;
}

static int test_valid_synthetic_file(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr(path, 0) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);
    CHECK(model.header != NULL);
    CHECK(model.config != NULL);
    CHECK(model.header->section_count == 1u);
    CHECK(model.config->vocab_size == UOCR_VOCAB_SIZE);
    CHECK(model.config->image_token_id == (uint32_t)UOCR_TOKEN_IMAGE);
    const uocr_section_entry *config_section = uocr_model_file_find_section(&model, UOCR_SECTION_CONFIG);
    CHECK(config_section != NULL);
    CHECK(config_section->size == sizeof(uocr_config_record));
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_valid_tensor_directory(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_OK) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);
    CHECK(model.tokenizer_metadata != NULL);
    CHECK(model.tokenizer_metadata->image_token_id == (uint32_t)UOCR_TOKEN_IMAGE);
    CHECK(model.tokenizer_payload == NULL);
    CHECK(model.tokenizer_payload_size == 0u);
    CHECK(model.provenance != NULL);
    CHECK(model.provenance->source_tensor_count == 2u);
    CHECK(model.provenance->runtime_tensor_count == 1u);
    CHECK(model.provenance->preserved_unused_tensor_count == 1u);
    CHECK(model.provenance_json != NULL);
    CHECK(model.provenance_json_size > 0u);
    CHECK(model.tensor_directory != NULL);
    CHECK(model.tensors != NULL);
    CHECK(model.tensor_count == 2u);
    CHECK(model.tensors[0].id == UOCR_TENSOR_ID_TOK_EMBED);
    CHECK(model.tensors[0].family == UOCR_TENSOR_FAMILY_TOK_EMBED);
    CHECK(model.tensors[0].flags == UOCR_TENSOR_FLAG_ROW_MAJOR);
    CHECK(model.tensors[0].payload_size == 16u);
    const uocr_section_entry *tensor_data = uocr_model_file_find_section(&model, UOCR_SECTION_TENSOR_DATA);
    CHECK(tensor_data != NULL);
    CHECK(model.tensors[1].payload_offset + model.tensors[1].payload_size == tensor_data->offset + tensor_data->size);
    CHECK(uocr_model_file_find_tensor(&model, UOCR_TENSOR_ID_TOK_EMBED) == &model.tensors[0]);
    CHECK(uocr_model_file_find_tensor(&model, 999999u) == NULL);
    CHECK(strcmp(uocr_tensor_qtype_name(model.tensors[0].qtype), "f16") == 0);
    CHECK(strcmp(uocr_tensor_qtype_reason_name(UOCR_TENSOR_QTYPE_REASON_UNKNOWN), "unknown") == 0);
    CHECK(strcmp(uocr_tensor_qtype_reason_name(UOCR_TENSOR_QTYPE_REASON_FP16_BASELINE), "fp16-baseline") == 0);
    CHECK(strcmp(uocr_tensor_promotion_reason_name(UOCR_TENSOR_PROMOTION_NONE), "none") == 0);
    CHECK(strcmp(uocr_tensor_usage_name(model.tensors[1].usage), "preserved-unused") == 0);
    CHECK(strcmp(uocr_tensor_family_name(model.tensors[1].family), "VISION_CLIP") == 0);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_rejects_tensor_payload_out_of_range(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_CORRUPT_TENSOR_PAYLOAD) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "payload is out of range") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_tensor_payload_outside_tensor_data_section(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_CORRUPT_TENSOR_DATA_SECTION_TOO_SMALL) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "payload is outside tensor-data section") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_transposed_tensor_layout(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_CORRUPT_TENSOR_TRANSPOSE_FLAG) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "unsupported transposed layout") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_bad_tensor_data_alignment(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_CORRUPT_TENSOR_DATA_ALIGNMENT) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "tensor-data section alignment") != NULL);
    unlink(path);
    return 0;
}

static int test_valid_mixed_q8_tensor_directory(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_mixed_q8_uocr(path, SYNTHETIC_Q8_OK) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);
    CHECK(model.header->qprofile == UOCR_QPROFILE_MIXED_Q8_0);
    CHECK(strcmp(uocr_qprofile_name(model.header->qprofile), "mixed-q8_0") == 0);
    CHECK(model.tensor_count == 1u);
    CHECK(model.tensors[0].qtype == UOCR_TENSOR_Q8_0);
    CHECK(strcmp(uocr_tensor_qtype_name(model.tensors[0].qtype), "q8_0") == 0);
    CHECK(model.tensors[0].block_size == UOCR_Q8_GROUP_SIZE_DEFAULT);
    CHECK(model.tensors[0].payload_size == 128u);
    CHECK(model.tensors[0].scale_size == 4u);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_rejects_bad_q8_scale_alignment(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_mixed_q8_uocr(path, SYNTHETIC_Q8_BAD_SCALE_ALIGNMENT) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "scale offset") != NULL);
    unlink(path);
    return 0;
}

static int test_valid_mixed_q4_tensor_directory(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_mixed_q4_uocr(path, SYNTHETIC_Q4_OK) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);
    CHECK(model.header->qprofile == UOCR_QPROFILE_MIXED_Q4);
    CHECK(strcmp(uocr_qprofile_name(model.header->qprofile), "mixed-q4") == 0);
    CHECK(model.tensor_count == 1u);
    CHECK(model.tensors[0].qtype == UOCR_TENSOR_Q4_0);
    CHECK(strcmp(uocr_tensor_qtype_name(model.tensors[0].qtype), "q4_0") == 0);
    CHECK(model.tensors[0].block_size == UOCR_Q4_GROUP_SIZE_DEFAULT);
    CHECK(model.tensors[0].payload_size == 64u);
    CHECK(model.tensors[0].row_size == 32u);
    CHECK(model.tensors[0].scale_size == 4u);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_rejects_bad_q4_row_size(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_mixed_q4_uocr(path, SYNTHETIC_Q4_BAD_ROW_SIZE) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "row size") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_q4_outside_moe_experts(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_mixed_q4_uocr(path, SYNTHETIC_Q4_BAD_FAMILY) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "q4_0 is not supported") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_q4_min_metadata(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_mixed_q4_uocr(path, SYNTHETIC_Q4_HAS_MIN_METADATA) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "min metadata") != NULL);
    unlink(path);
    return 0;
}

static int test_full_fp16_accounting_validator_accepts_current_contract(void) {
    uocr_model_file model;
    uocr_file_header header;
    uocr_section_entry tensor_data;
    uocr_provenance_record provenance;
    uocr_tensor_entry *tensors = NULL;
    CHECK(make_virtual_full_fp16_model(&model, &header, &tensor_data, &provenance, &tensors) == 0);

    char error[512];
    CHECK(uocr_model_file_validate_full_fp16_accounting(&model, error, sizeof(error)) == 0);
    free(tensors);
    return 0;
}

static int test_full_fp16_accounting_validator_rejects_incomplete_model(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_OK) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);
    CHECK(uocr_model_file_validate_full_fp16_accounting(&model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "full fp16 source tensor count mismatch") != NULL);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_full_fp16_accounting_validator_rejects_wrong_preserved_tensor(void) {
    uocr_model_file model;
    uocr_file_header header;
    uocr_section_entry tensor_data;
    uocr_provenance_record provenance;
    uocr_tensor_entry *tensors = NULL;
    CHECK(make_virtual_full_fp16_model(&model, &header, &tensor_data, &provenance, &tensors) == 0);
    tensors[UOCR_SOURCE_TENSOR_COUNT - 1u].id = UOCR_TENSOR_ID_VISION_CLIP_BASE;

    char error[512];
    CHECK(uocr_model_file_validate_full_fp16_accounting(&model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "preserved-unused tensor mismatch") != NULL);
    free(tensors);
    return 0;
}

static int test_full_fp16_accounting_validator_rejects_payload_drift(void) {
    uocr_model_file model;
    uocr_file_header header;
    uocr_section_entry tensor_data;
    uocr_provenance_record provenance;
    uocr_tensor_entry *tensors = NULL;
    CHECK(make_virtual_full_fp16_model(&model, &header, &tensor_data, &provenance, &tensors) == 0);
    tensors[0].payload_size += 2u;

    char error[512];
    CHECK(uocr_model_file_validate_full_fp16_accounting(&model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "tensor payload byte mismatch") != NULL);
    free(tensors);
    return 0;
}

static int test_rejects_unsorted_tensor_directory(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_CORRUPT_TENSOR_ORDER) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "is not greater than previous id") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_bad_tokenizer_metadata(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_CORRUPT_TOKENIZER) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "tokenizer metadata.image_token_id mismatch") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_bad_provenance(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr_with_tensors(path, SYNTHETIC_CORRUPT_PROVENANCE) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "provenance tensor accounting mismatch") != NULL);
    unlink(path);
    return 0;
}

static int test_rejects_bad_config(void) {
    char path[128];
    CHECK(make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_synthetic_uocr(path, 1) == 0);

    char error[512];
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) != 0);
    CHECK(strstr(error, "vocab_size mismatch") != NULL);
    unlink(path);
    return 0;
}

int main(void) {
    CHECK(uocr_tensor_id_layer_attn(3u, UOCR_TENSOR_PROJ_Q) == 13010u);
    CHECK(uocr_tensor_id_moe_expert(1u, 7u, UOCR_TENSOR_PROJ_DOWN) == 11123u);
    CHECK(UOCR_TENSOR_ID_VISION_CLIP_UNUSED_PATCH_EMBED_WEIGHT == 200001u);
    CHECK(uocr_q8_0_qweight_bytes(2u, 64u) == 128u);
    CHECK(uocr_q8_0_qscale_bytes(2u, 64u, UOCR_Q8_GROUP_SIZE_DEFAULT) == 4u);
    CHECK(uocr_q8_0_total_bytes(2u, 64u, UOCR_Q8_GROUP_SIZE_DEFAULT) == 132u);
    CHECK(uocr_q4_0_qweight_bytes(2u, 64u) == 64u);
    CHECK(uocr_q4_0_qscale_bytes(2u, 64u, UOCR_Q4_GROUP_SIZE_DEFAULT) == 4u);
    CHECK(uocr_q4_0_total_bytes(2u, 64u, UOCR_Q4_GROUP_SIZE_DEFAULT) == 68u);
    CHECK(uocr_q4_0_total_bytes(2u, 63u, UOCR_Q4_GROUP_SIZE_DEFAULT) == 0u);

    if (test_rejects_bad_endian_marker() != 0) return 1;
    if (test_rejects_zero_required_alignment() != 0) return 1;
    if (test_rejects_unaligned_section_directory() != 0) return 1;
    if (test_valid_synthetic_file() != 0) return 1;
    if (test_valid_tensor_directory() != 0) return 1;
    if (test_rejects_tensor_payload_out_of_range() != 0) return 1;
    if (test_rejects_tensor_payload_outside_tensor_data_section() != 0) return 1;
    if (test_rejects_transposed_tensor_layout() != 0) return 1;
    if (test_rejects_bad_tensor_data_alignment() != 0) return 1;
    if (test_valid_mixed_q8_tensor_directory() != 0) return 1;
    if (test_rejects_bad_q8_scale_alignment() != 0) return 1;
    if (test_valid_mixed_q4_tensor_directory() != 0) return 1;
    if (test_rejects_bad_q4_row_size() != 0) return 1;
    if (test_rejects_q4_outside_moe_experts() != 0) return 1;
    if (test_rejects_q4_min_metadata() != 0) return 1;
    if (test_full_fp16_accounting_validator_accepts_current_contract() != 0) return 1;
    if (test_full_fp16_accounting_validator_rejects_incomplete_model() != 0) return 1;
    if (test_full_fp16_accounting_validator_rejects_wrong_preserved_tensor() != 0) return 1;
    if (test_full_fp16_accounting_validator_rejects_payload_drift() != 0) return 1;
    if (test_rejects_unsorted_tensor_directory() != 0) return 1;
    if (test_rejects_bad_tokenizer_metadata() != 0) return 1;
    if (test_rejects_bad_provenance() != 0) return 1;
    if (test_rejects_bad_config() != 0) return 1;
    return 0;
}
