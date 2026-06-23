#ifndef UOCR_TEST_MODEL_FILE_H
#define UOCR_TEST_MODEL_FILE_H

#include "model/uocr_format.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline uint64_t uocr_test_align_up_u64(uint64_t value, uint64_t align) {
    const uint64_t rem = value % align;
    return rem == 0u ? value : value + (align - rem);
}

static inline void uocr_test_fill_hash(uint8_t hash[32], uint8_t seed) {
    for (uint32_t i = 0u; i < 32u; ++i) {
        hash[i] = (uint8_t)(seed + i);
    }
}

static inline int uocr_test_make_temp_path(char *path, size_t path_size) {
    const char *template_path = "/tmp/uocr-test-model-XXXXXX";
    if (strlen(template_path) + 1u > path_size) {
        return 1;
    }
    strcpy(path, template_path);
    const int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    close(fd);
    return 0;
}

static inline int uocr_test_write_two_tensor_uocr_model(const char *path) {
    const uint64_t section_dir_offset = sizeof(uocr_file_header);
    const uint32_t section_count = 5u;
    const uint32_t tensor_count = 2u;
    const uint64_t config_offset = section_dir_offset + section_count * sizeof(uocr_section_entry);
    const uint64_t tokenizer_offset = uocr_test_align_up_u64(config_offset + sizeof(uocr_config_record), 8u);
    const uint64_t provenance_offset = uocr_test_align_up_u64(tokenizer_offset + sizeof(uocr_tokenizer_metadata_record), 8u);
    const uint64_t tensor_dir_offset = uocr_test_align_up_u64(provenance_offset + sizeof(uocr_provenance_record), 8u);
    const uint64_t tensor_dir_size = sizeof(uocr_tensor_directory_header) + tensor_count * sizeof(uocr_tensor_entry);
    const uint64_t tensor_data_offset = uocr_test_align_up_u64(tensor_dir_offset + tensor_dir_size, UOCR_TENSOR_DATA_ALIGNMENT);
    const uint64_t tensor0_size = 16u;
    const uint64_t tensor1_size = 32u;
    const uint64_t tensor_data_size = UOCR_TENSOR_DATA_ALIGNMENT;
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
    uocr_test_fill_hash(header->model_id_hash, 0x80u);

    uocr_section_entry *sections = (uocr_section_entry *)(void *)(buffer + section_dir_offset);
    sections[0].type = UOCR_SECTION_CONFIG;
    sections[0].offset = config_offset;
    sections[0].size = sizeof(uocr_config_record);
    sections[0].alignment = 8u;
    sections[1].type = UOCR_SECTION_TOKENIZER_METADATA;
    sections[1].offset = tokenizer_offset;
    sections[1].size = sizeof(uocr_tokenizer_metadata_record);
    sections[1].alignment = 8u;
    sections[2].type = UOCR_SECTION_PROVENANCE;
    sections[2].offset = provenance_offset;
    sections[2].size = sizeof(uocr_provenance_record);
    sections[2].alignment = 8u;
    sections[3].type = UOCR_SECTION_TENSOR_DIRECTORY;
    sections[3].offset = tensor_dir_offset;
    sections[3].size = tensor_dir_size;
    sections[3].alignment = 8u;
    sections[4].type = UOCR_SECTION_TENSOR_DATA;
    sections[4].offset = tensor_data_offset;
    sections[4].size = tensor_data_size;
    sections[4].alignment = UOCR_TENSOR_DATA_ALIGNMENT;

    const uocr_config_record config = uocr_default_config_record();
    memcpy(buffer + config_offset, &config, sizeof(config));

    uocr_tokenizer_metadata_record *tokenizer = (uocr_tokenizer_metadata_record *)(void *)(buffer + tokenizer_offset);
    tokenizer->magic = UOCR_TOKENIZER_METADATA_MAGIC;
    tokenizer->version = UOCR_FORMAT_VERSION;
    tokenizer->record_size = (uint32_t)sizeof(uocr_tokenizer_metadata_record);
    tokenizer->flags = UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED;
    uocr_test_fill_hash(tokenizer->tokenizer_hash, 0x10u);
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
    provenance->source_tensor_count = tensor_count;
    provenance->runtime_tensor_count = tensor_count;
    provenance->preserved_unused_tensor_count = 0u;
    provenance->omitted_tensor_count = 0u;
    provenance->safetensors_file_count = 1u;
    provenance->qprofile = UOCR_QPROFILE_FP16;
    uocr_test_fill_hash(provenance->config_hash, 0x30u);
    uocr_test_fill_hash(provenance->tokenizer_hash, 0x10u);
    uocr_test_fill_hash(provenance->safetensors_index_hash, 0x50u);

    uocr_tensor_directory_header *dir = (uocr_tensor_directory_header *)(void *)(buffer + tensor_dir_offset);
    dir->magic = UOCR_TENSOR_DIR_MAGIC;
    dir->version = UOCR_FORMAT_VERSION;
    dir->entry_size = (uint32_t)sizeof(uocr_tensor_entry);
    dir->tensor_count = tensor_count;

    uocr_tensor_entry *tensor = (uocr_tensor_entry *)(void *)(buffer + tensor_dir_offset + sizeof(*dir));
    tensor[0].id = 1u;
    tensor[0].family = UOCR_TENSOR_FAMILY_TOK_EMBED;
    tensor[0].layer = -1;
    tensor[0].expert = -1;
    tensor[0].projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensor[0].usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensor[0].qtype = UOCR_TENSOR_F16;
    tensor[0].rank = 1u;
    tensor[0].logical_shape[0] = 8u;
    tensor[0].physical_shape[0] = 8u;
    tensor[0].payload_offset = tensor_data_offset;
    tensor[0].payload_size = tensor0_size;

    tensor[1].id = 2u;
    tensor[1].family = UOCR_TENSOR_FAMILY_LM_HEAD;
    tensor[1].layer = -1;
    tensor[1].expert = -1;
    tensor[1].projection = UOCR_TENSOR_PROJ_WEIGHT;
    tensor[1].usage = UOCR_TENSOR_USAGE_RUNTIME;
    tensor[1].qtype = UOCR_TENSOR_F16;
    tensor[1].rank = 1u;
    tensor[1].logical_shape[0] = 16u;
    tensor[1].physical_shape[0] = 16u;
    tensor[1].payload_offset = tensor_data_offset + tensor0_size;
    tensor[1].payload_size = tensor1_size;

    for (uint32_t i = 0u; i < tensor0_size + tensor1_size; ++i) {
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

#endif /* UOCR_TEST_MODEL_FILE_H */
