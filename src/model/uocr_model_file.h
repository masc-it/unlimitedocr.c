#ifndef UOCR_MODEL_FILE_H
#define UOCR_MODEL_FILE_H

#include <stddef.h>
#include <stdint.h>

#include "model/uocr_format.h"

typedef struct uocr_model_file {
    const uint8_t *data;
    size_t size;
    int owns_mapping;
#if !defined(_WIN32)
    int fd;
#endif
    const uocr_file_header *header;
    const uocr_section_entry *sections;
    const uocr_config_record *config;
    const uocr_tokenizer_metadata_record *tokenizer_metadata;
    const uint8_t *tokenizer_payload;
    size_t tokenizer_payload_size;
    const uocr_provenance_record *provenance;
    const uint8_t *provenance_json;
    size_t provenance_json_size;
    const uocr_tensor_directory_header *tensor_directory;
    const uocr_tensor_entry *tensors;
    uint32_t tensor_count;
} uocr_model_file;

int uocr_model_file_open(const char *path, uocr_model_file *out, char *error, size_t error_size);
int uocr_model_file_validate_memory(const void *data, size_t size, uocr_model_file *out, char *error, size_t error_size);
void uocr_model_file_close(uocr_model_file *file);

const uocr_section_entry *uocr_model_file_find_section(const uocr_model_file *file, uint32_t section_type);
const uocr_tensor_entry *uocr_model_file_find_tensor(const uocr_model_file *file, uint32_t tensor_id);
const char *uocr_section_type_name(uint32_t section_type);
const char *uocr_qprofile_name(uint32_t qprofile);
const char *uocr_tensor_qtype_name(uint32_t qtype);
const char *uocr_tensor_qtype_reason_name(uint32_t reason);
const char *uocr_tensor_promotion_reason_name(uint32_t reason);
const char *uocr_tensor_usage_name(uint32_t usage);
const char *uocr_tensor_family_name(uint32_t family);

#endif /* UOCR_MODEL_FILE_H */
