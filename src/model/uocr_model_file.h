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
} uocr_model_file;

int uocr_model_file_open(const char *path, uocr_model_file *out, char *error, size_t error_size);
int uocr_model_file_validate_memory(const void *data, size_t size, uocr_model_file *out, char *error, size_t error_size);
void uocr_model_file_close(uocr_model_file *file);

const uocr_section_entry *uocr_model_file_find_section(const uocr_model_file *file, uint32_t section_type);
const char *uocr_section_type_name(uint32_t section_type);
const char *uocr_qprofile_name(uint32_t qprofile);

#endif /* UOCR_MODEL_FILE_H */
