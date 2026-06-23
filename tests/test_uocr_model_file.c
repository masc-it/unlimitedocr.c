#include "model/uocr_model_file.h"

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

static int write_synthetic_uocr(const char *path, int corrupt_config) {
    const uint64_t header_offset = 0u;
    const uint64_t section_dir_offset = sizeof(uocr_file_header);
    const uint64_t config_offset = section_dir_offset + sizeof(uocr_section_entry);
    const uint64_t file_size = config_offset + sizeof(uocr_config_record);

    (void)header_offset;
    uocr_file_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, UOCR_FILE_MAGIC, 4u);
    header.version = UOCR_FORMAT_VERSION;
    header.header_size = (uint32_t)sizeof(uocr_file_header);
    header.endian_marker = UOCR_ENDIAN_MARKER;
    header.required_alignment = 8u;
    header.qprofile = UOCR_QPROFILE_FP16;
    header.section_count = 1u;
    header.file_size = file_size;
    header.section_dir_offset = section_dir_offset;

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
    if (test_valid_synthetic_file() != 0) return 1;
    if (test_rejects_bad_config() != 0) return 1;
    return 0;
}
