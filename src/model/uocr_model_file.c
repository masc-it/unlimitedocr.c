#include "model/uocr_model_file.h"

#include "core/uocr_alloc.h"
#include "quant/uocr_quant.h"

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

static int checked_section_relative_range(uint64_t offset, uint64_t size, uint64_t section_size) {
    if (offset > section_size) {
        return 0;
    }
    if (size > section_size - offset) {
        return 0;
    }
    return 1;
}

static int hash_is_nonzero(const uint8_t hash[32]) {
    for (size_t i = 0u; i < 32u; ++i) {
        if (hash[i] != 0u) {
            return 1;
        }
    }
    return 0;
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

const char *uocr_tensor_qtype_name(uint32_t qtype) {
    switch (qtype) {
        case UOCR_TENSOR_F16:
            return "f16";
        case UOCR_TENSOR_F32:
            return "f32";
        case UOCR_TENSOR_Q8_0:
            return "q8_0";
        case UOCR_TENSOR_Q4_K:
            return "q4_k";
        case UOCR_TENSOR_PADDED_Q4_K:
            return "padded_q4_k";
        case UOCR_TENSOR_Q2_K:
            return "q2_k";
        case UOCR_TENSOR_IQ2_XXS:
            return "iq2_xxs";
        default:
            return "unknown";
    }
}

const char *uocr_tensor_qtype_reason_name(uint32_t reason) {
    switch (reason) {
        case UOCR_TENSOR_QTYPE_REASON_UNKNOWN:
            return "unknown";
        case UOCR_TENSOR_QTYPE_REASON_FP16_BASELINE:
            return "fp16-baseline";
        case UOCR_TENSOR_QTYPE_REASON_POLICY:
            return "policy";
        case UOCR_TENSOR_QTYPE_REASON_SENSITIVE:
            return "sensitive";
        case UOCR_TENSOR_QTYPE_REASON_UNALIGNED:
            return "unaligned";
        case UOCR_TENSOR_QTYPE_REASON_CALIBRATION_DRIFT:
            return "calibration-drift";
        case UOCR_TENSOR_QTYPE_REASON_MANUAL_OVERRIDE:
            return "manual-override";
        default:
            return "unknown";
    }
}

const char *uocr_tensor_promotion_reason_name(uint32_t reason) {
    switch (reason) {
        case UOCR_TENSOR_PROMOTION_NONE:
            return "none";
        case UOCR_TENSOR_PROMOTION_SENSITIVE:
            return "sensitive";
        case UOCR_TENSOR_PROMOTION_UNALIGNED:
            return "unaligned";
        case UOCR_TENSOR_PROMOTION_CALIBRATION_DRIFT:
            return "calibration-drift";
        case UOCR_TENSOR_PROMOTION_MANUAL_OVERRIDE:
            return "manual-override";
        default:
            return "unknown";
    }
}

const char *uocr_tensor_usage_name(uint32_t usage) {
    switch (usage) {
        case UOCR_TENSOR_USAGE_RUNTIME:
            return "runtime";
        case UOCR_TENSOR_USAGE_PRESERVED_UNUSED:
            return "preserved-unused";
        case UOCR_TENSOR_USAGE_OMITTED_WITH_REASON:
            return "omitted-with-reason";
        default:
            return "unknown";
    }
}

const char *uocr_tensor_family_name(uint32_t family) {
    switch (family) {
        case UOCR_TENSOR_FAMILY_TOK_EMBED:
            return "TOK_EMBED";
        case UOCR_TENSOR_FAMILY_LM_HEAD:
            return "LM_HEAD";
        case UOCR_TENSOR_FAMILY_FINAL_NORM:
            return "FINAL_NORM";
        case UOCR_TENSOR_FAMILY_LAYER_ATTN:
            return "LAYER_ATTN";
        case UOCR_TENSOR_FAMILY_LAYER_NORM:
            return "LAYER_NORM";
        case UOCR_TENSOR_FAMILY_LAYER_DENSE_MLP:
            return "LAYER_DENSE_MLP";
        case UOCR_TENSOR_FAMILY_MOE_ROUTER:
            return "MOE_ROUTER";
        case UOCR_TENSOR_FAMILY_MOE_EXPERT:
            return "MOE_EXPERT";
        case UOCR_TENSOR_FAMILY_MOE_SHARED:
            return "MOE_SHARED";
        case UOCR_TENSOR_FAMILY_VISION_SAM:
            return "VISION_SAM";
        case UOCR_TENSOR_FAMILY_VISION_CLIP:
            return "VISION_CLIP";
        case UOCR_TENSOR_FAMILY_PROJECTOR:
            return "PROJECTOR";
        case UOCR_TENSOR_FAMILY_IMAGE_NEWLINE:
            return "IMAGE_NEWLINE";
        case UOCR_TENSOR_FAMILY_VIEW_SEPARATOR:
            return "VIEW_SEPARATOR";
        default:
            return "UNKNOWN";
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

const uocr_tensor_entry *uocr_model_file_find_tensor(const uocr_model_file *file, uint32_t tensor_id) {
    if (file == NULL || file->tensors == NULL || tensor_id == 0u) {
        return NULL;
    }
    uint32_t lo = 0u;
    uint32_t hi = file->tensor_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        const uint32_t mid_id = file->tensors[mid].id;
        if (mid_id == tensor_id) {
            return &file->tensors[mid];
        }
        if (mid_id < tensor_id) {
            lo = mid + 1u;
        } else {
            hi = mid;
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
    CHECK_FIELD(rope_theta);
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

static int validate_tokenizer_metadata(const uint8_t *bytes,
                                       const uocr_section_entry *section,
                                       uocr_model_file *view,
                                       char *error,
                                       size_t error_size) {
    if (section == NULL) {
        return 0;
    }
    if (section->size < sizeof(uocr_tokenizer_metadata_record)) {
        return fail(error, error_size, "tokenizer metadata section too small");
    }

    const uocr_tokenizer_metadata_record *meta =
        (const uocr_tokenizer_metadata_record *)(const void *)(bytes + section->offset);
    if (meta->magic != UOCR_TOKENIZER_METADATA_MAGIC) {
        return fail(error, error_size, "bad tokenizer metadata magic 0x%08x", meta->magic);
    }
    if (meta->version != UOCR_FORMAT_VERSION) {
        return fail(error, error_size, "unsupported tokenizer metadata version %u", meta->version);
    }
    if (meta->record_size != sizeof(uocr_tokenizer_metadata_record)) {
        return fail(error, error_size, "unexpected tokenizer metadata record size %u", meta->record_size);
    }
    if ((meta->flags & UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED) == 0u) {
        return fail(error, error_size, "tokenizer metadata must declare that C v1 does not require tokenizer tables");
    }
    if (!hash_is_nonzero(meta->tokenizer_hash)) {
        return fail(error, error_size, "tokenizer metadata hash is missing");
    }

#define CHECK_TOKENIZER_FIELD(field, expected_value)                                                   \
    do {                                                                                              \
        if (meta->field != (uint32_t)(expected_value)) {                                              \
            return fail(error,                                                                         \
                        error_size,                                                                   \
                        "tokenizer metadata.%s mismatch: got %u expected %u",                         \
                        #field,                                                                       \
                        meta->field,                                                                  \
                        (uint32_t)(expected_value));                                                  \
        }                                                                                             \
    } while (0)

    CHECK_TOKENIZER_FIELD(vocab_size, UOCR_VOCAB_SIZE);
    CHECK_TOKENIZER_FIELD(bpe_vocab_size, UOCR_BPE_VOCAB_SIZE);
    CHECK_TOKENIZER_FIELD(added_token_count, UOCR_ADDED_TOKEN_COUNT);
    CHECK_TOKENIZER_FIELD(bos_token_id, UOCR_TOKEN_BOS);
    CHECK_TOKENIZER_FIELD(eos_token_id, UOCR_TOKEN_EOS);
    CHECK_TOKENIZER_FIELD(pad_token_id, UOCR_TOKEN_PAD);
    CHECK_TOKENIZER_FIELD(image_token_id, UOCR_TOKEN_IMAGE);
#undef CHECK_TOKENIZER_FIELD

    const int has_payload = meta->tokenizer_json_size != 0u;
    if (has_payload && (meta->flags & UOCR_TOKENIZER_FLAG_EMBEDDED_PAYLOAD) == 0u) {
        return fail(error, error_size, "tokenizer payload present without embedded-payload flag");
    }
    if (!has_payload && meta->tokenizer_json_offset != 0u) {
        return fail(error, error_size, "tokenizer payload offset set without payload size");
    }
    if (!has_payload && (meta->flags & UOCR_TOKENIZER_FLAG_EMBEDDED_PAYLOAD) != 0u) {
        return fail(error, error_size, "embedded tokenizer flag set without payload");
    }
    if (has_payload) {
        if (!checked_section_relative_range(meta->tokenizer_json_offset, meta->tokenizer_json_size, section->size)) {
            return fail(error, error_size, "tokenizer payload is outside tokenizer metadata section");
        }
        if (meta->tokenizer_json_offset < sizeof(uocr_tokenizer_metadata_record)) {
            return fail(error, error_size, "tokenizer payload overlaps metadata record");
        }
        view->tokenizer_payload = bytes + section->offset + meta->tokenizer_json_offset;
        view->tokenizer_payload_size = (size_t)meta->tokenizer_json_size;
    }

    view->tokenizer_metadata = meta;
    return 0;
}

static int validate_provenance(const uint8_t *bytes,
                               const uocr_file_header *header,
                               const uocr_section_entry *section,
                               uocr_model_file *view,
                               char *error,
                               size_t error_size) {
    if (section == NULL) {
        return 0;
    }
    if (section->size < sizeof(uocr_provenance_record)) {
        return fail(error, error_size, "provenance section too small");
    }

    const uocr_provenance_record *prov =
        (const uocr_provenance_record *)(const void *)(bytes + section->offset);
    if (prov->magic != UOCR_PROVENANCE_MAGIC) {
        return fail(error, error_size, "bad provenance magic 0x%08x", prov->magic);
    }
    if (prov->version != UOCR_FORMAT_VERSION) {
        return fail(error, error_size, "unsupported provenance version %u", prov->version);
    }
    if (prov->record_size != sizeof(uocr_provenance_record)) {
        return fail(error, error_size, "unexpected provenance record size %u", prov->record_size);
    }
    if (prov->qprofile != header->qprofile) {
        return fail(error, error_size, "provenance qprofile mismatch: got %u expected %u", prov->qprofile, header->qprofile);
    }
    if (!hash_is_nonzero(prov->config_hash)) {
        return fail(error, error_size, "provenance config hash is missing");
    }
    if (!hash_is_nonzero(prov->tokenizer_hash)) {
        return fail(error, error_size, "provenance tokenizer hash is missing");
    }
    if (!hash_is_nonzero(prov->safetensors_index_hash)) {
        return fail(error, error_size, "provenance safetensors index hash is missing");
    }

    const uint64_t accounted = (uint64_t)prov->runtime_tensor_count +
                               (uint64_t)prov->preserved_unused_tensor_count +
                               (uint64_t)prov->omitted_tensor_count;
    if (accounted != prov->source_tensor_count) {
        return fail(error,
                    error_size,
                    "provenance tensor accounting mismatch: source=%u accounted=%llu",
                    prov->source_tensor_count,
                    (unsigned long long)accounted);
    }
    if (view->tensor_count != 0u && prov->source_tensor_count != view->tensor_count) {
        return fail(error,
                    error_size,
                    "provenance source tensor count %u does not match tensor directory count %u",
                    prov->source_tensor_count,
                    view->tensor_count);
    }
    if (prov->safetensors_file_count == 0u) {
        return fail(error, error_size, "provenance safetensors file count must be non-zero");
    }

    const int has_json = prov->json_size != 0u;
    if (has_json && (prov->flags & UOCR_PROVENANCE_FLAG_JSON_PAYLOAD) == 0u) {
        return fail(error, error_size, "provenance JSON present without JSON flag");
    }
    if (!has_json && prov->json_offset != 0u) {
        return fail(error, error_size, "provenance JSON offset set without JSON size");
    }
    if (!has_json && (prov->flags & UOCR_PROVENANCE_FLAG_JSON_PAYLOAD) != 0u) {
        return fail(error, error_size, "provenance JSON flag set without payload");
    }
    if (has_json) {
        if (!checked_section_relative_range(prov->json_offset, prov->json_size, section->size)) {
            return fail(error, error_size, "provenance JSON is outside provenance section");
        }
        if (prov->json_offset < sizeof(uocr_provenance_record)) {
            return fail(error, error_size, "provenance JSON overlaps metadata record");
        }
        view->provenance_json = bytes + section->offset + prov->json_offset;
        view->provenance_json_size = (size_t)prov->json_size;
    }

    view->provenance = prov;
    return 0;
}

static int validate_tensor_directory(const uint8_t *bytes,
                                     size_t file_size,
                                     const uocr_section_entry *directory_section,
                                     const uocr_section_entry *tensor_data_section,
                                     uocr_model_file *view,
                                     char *error,
                                     size_t error_size) {
    if (directory_section == NULL) {
        return 0;
    }
    if (directory_section->size < sizeof(uocr_tensor_directory_header)) {
        return fail(error, error_size, "tensor directory section too small");
    }

    const uocr_tensor_directory_header *dir =
        (const uocr_tensor_directory_header *)(const void *)(bytes + directory_section->offset);
    if (dir->magic != UOCR_TENSOR_DIR_MAGIC) {
        return fail(error, error_size, "bad tensor directory magic 0x%08x", dir->magic);
    }
    if (dir->version != UOCR_FORMAT_VERSION) {
        return fail(error, error_size, "unsupported tensor directory version %u", dir->version);
    }
    if (dir->entry_size != sizeof(uocr_tensor_entry)) {
        return fail(error,
                    error_size,
                    "unexpected tensor directory entry size %u",
                    dir->entry_size);
    }

    const uint64_t entries_offset = directory_section->offset + sizeof(uocr_tensor_directory_header);
    const uint64_t entries_bytes = (uint64_t)dir->tensor_count * (uint64_t)sizeof(uocr_tensor_entry);
    if (directory_section->size != sizeof(uocr_tensor_directory_header) + entries_bytes) {
        return fail(error, error_size, "tensor directory size/count mismatch");
    }
    if (!checked_range(entries_offset, entries_bytes, file_size)) {
        return fail(error, error_size, "tensor directory entries are out of range");
    }
    if (tensor_data_section != NULL) {
        if (tensor_data_section->alignment < UOCR_TENSOR_DATA_ALIGNMENT) {
            return fail(error,
                        error_size,
                        "tensor-data section alignment %llu is smaller than required %u",
                        (unsigned long long)tensor_data_section->alignment,
                        UOCR_TENSOR_DATA_ALIGNMENT);
        }
        if ((tensor_data_section->offset % UOCR_TENSOR_DATA_ALIGNMENT) != 0u) {
            return fail(error,
                        error_size,
                        "tensor-data section offset %llu is not page aligned to %u",
                        (unsigned long long)tensor_data_section->offset,
                        UOCR_TENSOR_DATA_ALIGNMENT);
        }
    }

    const uocr_tensor_entry *entries = (const uocr_tensor_entry *)(const void *)(bytes + entries_offset);
    for (uint32_t i = 0u; i < dir->tensor_count; ++i) {
        const uocr_tensor_entry *tensor = &entries[i];
        if (tensor->id == 0u) {
            return fail(error, error_size, "tensor entry %u has invalid id 0", i);
        }
        if (i > 0u && entries[i - 1u].id >= tensor->id) {
            return fail(error,
                        error_size,
                        "tensor entry %u id %u is not greater than previous id %u",
                        i,
                        tensor->id,
                        entries[i - 1u].id);
        }
        if (tensor->rank > UOCR_TENSOR_MAX_DIMS) {
            return fail(error, error_size, "tensor entry %u rank %u exceeds limit", i, tensor->rank);
        }
        const uint32_t known_tensor_flags = UOCR_TENSOR_FLAG_ROW_MAJOR | UOCR_TENSOR_FLAG_TRANSPOSED |
                                            UOCR_TENSOR_FLAG_FLATTENED_LEADING_DIM;
        if ((tensor->flags & ~known_tensor_flags) != 0u) {
            return fail(error, error_size, "tensor entry %u has unknown flags 0x%x", i, tensor->flags);
        }
        if ((tensor->flags & UOCR_TENSOR_FLAG_TRANSPOSED) != 0u) {
            return fail(error, error_size, "tensor entry %u requests unsupported transposed layout", i);
        }
        if (tensor->usage == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON) {
            if (tensor->payload_offset != 0u || tensor->payload_size != 0u) {
                return fail(error, error_size, "omitted tensor entry %u has payload bytes", i);
            }
            continue;
        }
        if (tensor->usage != UOCR_TENSOR_USAGE_RUNTIME && tensor->usage != UOCR_TENSOR_USAGE_PRESERVED_UNUSED) {
            return fail(error, error_size, "tensor entry %u has unknown usage %u", i, tensor->usage);
        }
        uocr_quant_type_info qtype_info;
        if (!uocr_quant_get_type_info(tensor->qtype, &qtype_info)) {
            return fail(error, error_size, "tensor entry %u has unknown qtype %u", i, tensor->qtype);
        }
        if (!uocr_quant_is_enabled(tensor->qtype)) {
            return fail(error, error_size, "tensor entry %u qtype %s is disabled", i, qtype_info.name);
        }
        if (tensor_data_section == NULL) {
            return fail(error, error_size, "tensor entry %u has payload but tensor-data section is missing", i);
        }
        if (tensor->payload_size == 0u) {
            return fail(error, error_size, "tensor entry %u has zero payload size", i);
        }
        if ((tensor->payload_offset % UOCR_TENSOR_PAYLOAD_ALIGNMENT) != 0u) {
            return fail(error,
                        error_size,
                        "tensor entry %u payload offset %llu is not aligned to %u",
                        i,
                        (unsigned long long)tensor->payload_offset,
                        UOCR_TENSOR_PAYLOAD_ALIGNMENT);
        }
        char quant_error[256];
        if (!uocr_quant_validate_tensor_entry(tensor, quant_error, sizeof(quant_error))) {
            return fail(error,
                        error_size,
                        "tensor entry %u quantization metadata invalid: %s",
                        i,
                        quant_error);
        }
        if (!checked_range(tensor->payload_offset, tensor->payload_size, file_size)) {
            return fail(error, error_size, "tensor entry %u payload is out of range", i);
        }
        if (tensor_data_section != NULL) {
            if (tensor->payload_offset < tensor_data_section->offset ||
                tensor->payload_offset + tensor->payload_size > tensor_data_section->offset + tensor_data_section->size) {
                return fail(error, error_size, "tensor entry %u payload is outside tensor-data section", i);
            }
        }
    }

    view->tensor_directory = dir;
    view->tensors = entries;
    view->tensor_count = dir->tensor_count;
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
        if (section->size != 0u) {
            const uint64_t section_end = section->offset + section->size;
            const uint64_t header_end = header->header_size;
            const uint64_t section_dir_end = header->section_dir_offset + section_bytes;
            if (section->offset < header_end) {
                return fail(error, error_size, "section %u (%s) overlaps file header", i, uocr_section_type_name(section->type));
            }
            if (section->offset < section_dir_end && header->section_dir_offset < section_end) {
                return fail(error, error_size, "section %u (%s) overlaps section directory", i, uocr_section_type_name(section->type));
            }
        }
        for (uint32_t j = 0u; j < i; ++j) {
            const uocr_section_entry *prev = &sections[j];
            if (prev->type == section->type) {
                return fail(error, error_size, "duplicate section type %u (%s)", section->type, uocr_section_type_name(section->type));
            }
            const uint64_t a0 = prev->offset;
            const uint64_t a1 = prev->offset + prev->size;
            const uint64_t b0 = section->offset;
            const uint64_t b1 = section->offset + section->size;
            if (prev->size != 0u && section->size != 0u && a0 < b1 && b0 < a1) {
                return fail(error,
                            error_size,
                            "section %u (%s) overlaps section %u (%s)",
                            j,
                            uocr_section_type_name(prev->type),
                            i,
                            uocr_section_type_name(section->type));
            }
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

    const uocr_section_entry *tokenizer_section = uocr_model_file_find_section(&view, UOCR_SECTION_TOKENIZER_METADATA);
    if (validate_tokenizer_metadata(bytes, tokenizer_section, &view, error, error_size) != 0) {
        return -1;
    }

    const uocr_section_entry *directory_section = uocr_model_file_find_section(&view, UOCR_SECTION_TENSOR_DIRECTORY);
    const uocr_section_entry *tensor_data_section = uocr_model_file_find_section(&view, UOCR_SECTION_TENSOR_DATA);
    if (validate_tensor_directory(bytes, size, directory_section, tensor_data_section, &view, error, error_size) != 0) {
        return -1;
    }

    const uocr_section_entry *provenance_section = uocr_model_file_find_section(&view, UOCR_SECTION_PROVENANCE);
    if (validate_provenance(bytes, header, provenance_section, &view, error, error_size) != 0) {
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
    uint8_t *buffer = (uint8_t *)uocr_malloc(file_size);
    if (buffer == NULL) {
        fclose(f);
        return fail(error, error_size, "failed to allocate %llu bytes", (unsigned long long)file_size);
    }
    if (fread(buffer, 1u, file_size, f) != file_size) {
        uocr_free(buffer);
        fclose(f);
        return fail(error, error_size, "failed to read '%s'", path);
    }
    fclose(f);
    if (uocr_model_file_validate_memory(buffer, file_size, out, error, error_size) != 0) {
        uocr_free(buffer);
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
    uocr_free((void *)file->data);
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
