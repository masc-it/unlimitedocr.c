#include "model/uocr_model_file.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static int fail(char *error, size_t error_size, const char *fmt, ...) {
    if (error != NULL && error_size > 0u) {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(error, error_size, fmt, ap);
        va_end(ap);
        error[error_size - 1u] = '\0';
    }
    return -1;
}

static int checked_range(uint64_t offset, uint64_t size, size_t file_size) {
    if (offset > (uint64_t)file_size) {
        return 0;
    }
    if (size > (uint64_t)file_size - offset) {
        return 0;
    }
    return 1;
}

const char *uocr_section_type_name(uint32_t section_type) {
    switch (section_type) {
        case UOCR_SECTION_CONFIG:
            return "config";
        case UOCR_SECTION_TENSOR_DIRECTORY:
            return "tensor-directory";
        case UOCR_SECTION_TOKENIZER_METADATA:
            return "tokenizer-metadata";
        case UOCR_SECTION_PROVENANCE:
            return "provenance";
        case UOCR_SECTION_TENSOR_DATA:
            return "tensor-data";
        default:
            return "unknown";
    }
}

const char *uocr_qprofile_name(uint32_t qprofile) {
    switch (qprofile) {
        case UOCR_QPROFILE_FP16:
            return "fp16";
        case UOCR_QPROFILE_DYN_Q8:
            return "dyn-q8";
        case UOCR_QPROFILE_DYN_Q4:
            return "dyn-q4";
        case UOCR_QPROFILE_CUSTOM:
            return "custom";
        default:
            return "unknown";
    }
}

const uocr_section_entry *uocr_model_file_find_section(const uocr_model_file *file, uint32_t section_type) {
    if (file == NULL || file->sections == NULL || file->header == NULL) {
        return NULL;
    }
    for (uint32_t i = 0u; i < file->header->section_count; ++i) {
        if (file->sections[i].type == section_type) {
            return &file->sections[i];
        }
    }
    return NULL;
}

static int validate_config(const uocr_config_record *cfg, char *error, size_t error_size) {
    const uocr_config_record expected = uocr_default_config_record();
#define CHECK_FIELD(field)                                                                 \
    do {                                                                                   \
        if (cfg->field != expected.field) {                                                \
            return fail(error, error_size, "config.%s mismatch: got %u expected %u", #field, cfg->field, expected.field); \
        }                                                                                  \
    } while (0)

    CHECK_FIELD(vocab_size);
    CHECK_FIELD(bpe_vocab_size);
    CHECK_FIELD(added_token_count);
    CHECK_FIELD(bos_token_id);
    CHECK_FIELD(eos_token_id);
    CHECK_FIELD(pad_token_id);
    CHECK_FIELD(image_token_id);
    CHECK_FIELD(hidden_size);
    CHECK_FIELD(decoder_layers);
    CHECK_FIELD(attention_heads);
    CHECK_FIELD(kv_heads);
    CHECK_FIELD(head_dim);
    CHECK_FIELD(max_positions);
    CHECK_FIELD(generated_ring_window);
    CHECK_FIELD(dense_first_layers);
    CHECK_FIELD(routed_experts);
    CHECK_FIELD(moe_top_k);
    CHECK_FIELD(moe_expert_intermediate);
    CHECK_FIELD(shared_experts);
    CHECK_FIELD(dense_layer0_intermediate);
    CHECK_FIELD(projector_in);
    CHECK_FIELD(projector_out);
    CHECK_FIELD(vision_patch_size);
    CHECK_FIELD(downsample_ratio);
    CHECK_FIELD(supported_global_view);
    CHECK_FIELD(supported_local_view);
#undef CHECK_FIELD
    return 0;
}

int uocr_model_file_validate_memory(const void *data, size_t size, uocr_model_file *out, char *error, size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
    if (out == NULL) {
        return fail(error, error_size, "output model file pointer is null");
    }
    memset(out, 0, sizeof(*out));
#if !defined(_WIN32)
    out->fd = -1;
#endif
    if (data == NULL) {
        return fail(error, error_size, "model data pointer is null");
    }
    if (size < sizeof(uocr_file_header)) {
        return fail(error, error_size, "file too small for UOCR header: %llu bytes", (unsigned long long)size);
    }

    const uint8_t *bytes = (const uint8_t *)data;
    const uocr_file_header *header = (const uocr_file_header *)bytes;
    if (memcmp(header->magic, UOCR_FILE_MAGIC, 4u) != 0) {
        return fail(error, error_size, "bad UOCR magic");
    }
    if (header->version != UOCR_FORMAT_VERSION) {
        return fail(error, error_size, "unsupported UOCR format version %u", header->version);
    }
    if (header->header_size != sizeof(uocr_file_header)) {
        return fail(error, error_size, "unexpected UOCR header size %u", header->header_size);
    }
    if (header->endian_marker != UOCR_ENDIAN_MARKER) {
        return fail(error, error_size, "unsupported endian marker 0x%08x", header->endian_marker);
    }
    if (header->file_size != (uint64_t)size) {
        return fail(error,
                    error_size,
                    "file_size mismatch: header says %llu, actual is %llu",
                    (unsigned long long)header->file_size,
                    (unsigned long long)size);
    }
    if (header->required_alignment == 0u) {
        return fail(error, error_size, "required alignment must be non-zero");
    }
    if (header->section_count == 0u) {
        return fail(error, error_size, "section_count must be non-zero");
    }

    const uint64_t section_bytes = (uint64_t)header->section_count * (uint64_t)sizeof(uocr_section_entry);
    if (!checked_range(header->section_dir_offset, section_bytes, size)) {
        return fail(error, error_size, "section directory is out of range");
    }
    if ((header->section_dir_offset % sizeof(uint64_t)) != 0u) {
        return fail(error, error_size, "section directory offset must be 8-byte aligned");
    }

    const uocr_section_entry *sections = (const uocr_section_entry *)(const void *)(bytes + header->section_dir_offset);
    for (uint32_t i = 0u; i < header->section_count; ++i) {
        const uocr_section_entry *section = &sections[i];
        if (!checked_range(section->offset, section->size, size)) {
            return fail(error,
                        error_size,
                        "section %u (%s) is out of range",
                        i,
                        uocr_section_type_name(section->type));
        }
        if (section->alignment != 0u && (section->offset % section->alignment) != 0u) {
            return fail(error,
                        error_size,
                        "section %u (%s) offset %llu is not aligned to %llu",
                        i,
                        uocr_section_type_name(section->type),
                        (unsigned long long)section->offset,
                        (unsigned long long)section->alignment);
        }
    }

    uocr_model_file view;
    memset(&view, 0, sizeof(view));
#if !defined(_WIN32)
    view.fd = -1;
#endif
    view.data = bytes;
    view.size = size;
    view.header = header;
    view.sections = sections;

    const uocr_section_entry *config_section = uocr_model_file_find_section(&view, UOCR_SECTION_CONFIG);
    if (config_section == NULL) {
        return fail(error, error_size, "missing config section");
    }
    if (config_section->size < sizeof(uocr_config_record)) {
        return fail(error, error_size, "config section too small");
    }
    view.config = (const uocr_config_record *)(const void *)(bytes + config_section->offset);
    if (validate_config(view.config, error, error_size) != 0) {
        return -1;
    }

    *out = view;
    return 0;
}

#if defined(_WIN32)
int uocr_model_file_open(const char *path, uocr_model_file *out, char *error, size_t error_size) {
    if (path == NULL) {
        return fail(error, error_size, "path is null");
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return fail(error, error_size, "failed to open '%s': %s", path, strerror(errno));
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return fail(error, error_size, "failed to seek '%s'", path);
    }
    long file_size_long = ftell(f);
    if (file_size_long < 0) {
        fclose(f);
        return fail(error, error_size, "failed to tell '%s'", path);
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return fail(error, error_size, "failed to rewind '%s'", path);
    }
    size_t file_size = (size_t)file_size_long;
    uint8_t *buffer = (uint8_t *)malloc(file_size);
    if (buffer == NULL) {
        fclose(f);
        return fail(error, error_size, "failed to allocate %llu bytes", (unsigned long long)file_size);
    }
    if (fread(buffer, 1u, file_size, f) != file_size) {
        free(buffer);
        fclose(f);
        return fail(error, error_size, "failed to read '%s'", path);
    }
    fclose(f);
    if (uocr_model_file_validate_memory(buffer, file_size, out, error, error_size) != 0) {
        free(buffer);
        return -1;
    }
    out->owns_mapping = 1;
    return 0;
}
#else
int uocr_model_file_open(const char *path, uocr_model_file *out, char *error, size_t error_size) {
    if (path == NULL) {
        return fail(error, error_size, "path is null");
    }
    if (out == NULL) {
        return fail(error, error_size, "output model file pointer is null");
    }
    memset(out, 0, sizeof(*out));
    out->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return fail(error, error_size, "failed to open '%s': %s", path, strerror(errno));
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return fail(error, error_size, "failed to stat '%s': %s", path, strerror(errno));
    }
    if (st.st_size <= 0) {
        close(fd);
        return fail(error, error_size, "file '%s' is empty", path);
    }
    const uint64_t file_size64 = (uint64_t)st.st_size;
    if ((off_t)file_size64 != st.st_size || file_size64 > (uint64_t)SIZE_MAX) {
        close(fd);
        return fail(error, error_size, "file '%s' is too large for this platform", path);
    }
    const size_t file_size = (size_t)file_size64;
    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return fail(error, error_size, "failed to mmap '%s': %s", path, strerror(errno));
    }
    if (uocr_model_file_validate_memory(mapped, file_size, out, error, error_size) != 0) {
        munmap(mapped, file_size);
        close(fd);
        return -1;
    }
    out->owns_mapping = 1;
    out->fd = fd;
    return 0;
}
#endif

void uocr_model_file_close(uocr_model_file *file) {
    if (file == NULL || file->data == NULL || !file->owns_mapping) {
        if (file != NULL) {
            memset(file, 0, sizeof(*file));
#if !defined(_WIN32)
            file->fd = -1;
#endif
        }
        return;
    }
#if defined(_WIN32)
    free((void *)file->data);
#else
    (void)munmap((void *)file->data, file->size);
    if (file->fd >= 0) {
        (void)close(file->fd);
    }
#endif
    memset(file, 0, sizeof(*file));
#if !defined(_WIN32)
    file->fd = -1;
#endif
}
