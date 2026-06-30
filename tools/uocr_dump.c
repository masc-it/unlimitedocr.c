#include <stdio.h>

#include "model/uocr_model_file.h"
#include "unlimitedocr.h"

static void print_hash_prefix(const uint8_t hash[32]) {
    for (uint32_t i = 0u; i < 8u; ++i) {
        printf("%02x", hash[i]);
    }
}

static uint64_t natural_alignment(uint64_t value) {
    if (value == 0u) {
        return 0u;
    }
    return value & (~value + 1u);
}

static int print_usage(const char *argv0) {
    fprintf(stderr, "usage: %s FILE.uocr\n", argv0);
    return 2;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return print_usage(argc > 0 ? argv[0] : "uocr-dump");
    }

    const char *path = argv[1];
    char error[512];
    uocr_model_file model;
    if (uocr_model_file_open(path, &model, error, sizeof(error)) != 0) {
        fprintf(stderr, "uocr-dump: %s\n", error);
        return 1;
    }

    printf("uocr-dump (ABI 0x%06x)\n", uocr_abi_version());
    printf("path: %s\n", path);
    printf("size: %llu bytes\n", (unsigned long long)model.size);
    printf("format_version: %u\n", model.header->version);
    printf("alignment: %u\n", model.header->required_alignment);
    printf("qprofile: %s (%u)\n", uocr_qprofile_name(model.header->qprofile), model.header->qprofile);
    printf("sections: %u\n", model.header->section_count);
    for (uint32_t i = 0u; i < model.header->section_count; ++i) {
        const uocr_section_entry *section = &model.sections[i];
        printf("  [%u] %-18s offset=%llu size=%llu alignment=%llu flags=0x%x\n",
               i,
               uocr_section_type_name(section->type),
               (unsigned long long)section->offset,
               (unsigned long long)section->size,
               (unsigned long long)section->alignment,
               section->flags);
    }

    if (model.config != NULL) {
        printf("config:\n");
        printf("  vocab=%u hidden=%u layers=%u heads=%u kv_heads=%u head_dim=%u\n",
               model.config->vocab_size,
               model.config->hidden_size,
               model.config->decoder_layers,
               model.config->attention_heads,
               model.config->kv_heads,
               model.config->head_dim);
        printf("  max_positions=%u rope_theta=%u generated_ring=%u image_token=%u\n",
               model.config->max_positions,
               model.config->rope_theta,
               model.config->generated_ring_window,
               model.config->image_token_id);
        printf("  moe: routed=%u top_k=%u expert_intermediate=%u shared=%u\n",
               model.config->routed_experts,
               model.config->moe_top_k,
               model.config->moe_expert_intermediate,
               model.config->shared_experts);
        printf("  vision: global=%u local=%u patch=%u downsample=%u\n",
               model.config->supported_global_view,
               model.config->supported_local_view,
               model.config->vision_patch_size,
               model.config->downsample_ratio);
    }

    if (model.tokenizer_metadata != NULL) {
        printf("tokenizer:\n");
        printf("  vocab=%u bpe_vocab=%u added=%u bos=%u eos=%u pad=%u image=%u\n",
               model.tokenizer_metadata->vocab_size,
               model.tokenizer_metadata->bpe_vocab_size,
               model.tokenizer_metadata->added_token_count,
               model.tokenizer_metadata->bos_token_id,
               model.tokenizer_metadata->eos_token_id,
               model.tokenizer_metadata->pad_token_id,
               model.tokenizer_metadata->image_token_id);
        printf("  flags=0x%x hash=", model.tokenizer_metadata->flags);
        print_hash_prefix(model.tokenizer_metadata->tokenizer_hash);
        printf("... payload_bytes=%zu\n", model.tokenizer_payload_size);
    }

    if (model.provenance != NULL) {
        printf("provenance:\n");
        printf("  source_tensors=%u runtime=%u preserved-unused=%u omitted=%u safetensors_files=%u\n",
               model.provenance->source_tensor_count,
               model.provenance->runtime_tensor_count,
               model.provenance->preserved_unused_tensor_count,
               model.provenance->omitted_tensor_count,
               model.provenance->safetensors_file_count);
        printf("  converter=%u.%u.%u qprofile=%s (%u) json_bytes=%zu\n",
               model.provenance->converter_version_major,
               model.provenance->converter_version_minor,
               model.provenance->converter_version_patch,
               uocr_qprofile_name(model.provenance->qprofile),
               model.provenance->qprofile,
               model.provenance_json_size);
        printf("  hashes: config=");
        print_hash_prefix(model.provenance->config_hash);
        printf("... tokenizer=");
        print_hash_prefix(model.provenance->tokenizer_hash);
        printf("... safetensors_index=");
        print_hash_prefix(model.provenance->safetensors_index_hash);
        printf("...\n");
        if (model.header->qprofile == UOCR_QPROFILE_FP16) {
            char exact_error[256];
            if (uocr_model_file_validate_full_fp16_accounting(&model, exact_error, sizeof(exact_error)) == 0) {
                printf("  full_fp16_accounting: ok\n");
            } else {
                printf("  full_fp16_accounting: not-current-full-model (%s)\n", exact_error);
            }
        }
    }

    if (model.tensor_count > 0u && model.tensors != NULL) {
        enum { TOP_TENSORS = 5 };
        uint32_t qtype_counts[64] = {0};
        uint64_t qtype_bytes[64] = {0};
        uint32_t qtype_reason_counts[16] = {0};
        uint32_t promotion_reason_counts[16] = {0};
        uint32_t usage_runtime = 0u;
        uint32_t usage_preserved = 0u;
        uint32_t usage_omitted = 0u;
        uint32_t unaligned_payloads = 0u;
        uint64_t payload_bytes = 0u;
        uint64_t min_payload_offset = UINT64_MAX;
        uint64_t max_payload_end = 0u;
        uint32_t top[TOP_TENSORS];
        uint32_t top_count = 0u;
        for (uint32_t i = 0u; i < TOP_TENSORS; ++i) {
            top[i] = UINT32_MAX;
        }
        for (uint32_t i = 0u; i < model.tensor_count; ++i) {
            const uocr_tensor_entry *tensor = &model.tensors[i];
            if (tensor->qtype < 64u) {
                qtype_counts[tensor->qtype]++;
                qtype_bytes[tensor->qtype] += tensor->payload_size;
            }
            if (tensor->qtype_reason < 16u) qtype_reason_counts[tensor->qtype_reason]++;
            if (tensor->promotion_reason < 16u) promotion_reason_counts[tensor->promotion_reason]++;
            if (tensor->usage == UOCR_TENSOR_USAGE_RUNTIME) usage_runtime++;
            if (tensor->usage == UOCR_TENSOR_USAGE_PRESERVED_UNUSED) usage_preserved++;
            if (tensor->usage == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON) usage_omitted++;
            payload_bytes += tensor->payload_size;
            if (tensor->payload_size != 0u) {
                if (tensor->payload_offset < min_payload_offset) min_payload_offset = tensor->payload_offset;
                if (tensor->payload_offset + tensor->payload_size > max_payload_end) {
                    max_payload_end = tensor->payload_offset + tensor->payload_size;
                }
                if ((tensor->payload_offset % UOCR_TENSOR_PAYLOAD_ALIGNMENT) != 0u) {
                    unaligned_payloads++;
                }
            }
            uint32_t pos = top_count;
            while (pos > 0u && tensor->payload_size > model.tensors[top[pos - 1u]].payload_size) {
                if (pos < TOP_TENSORS) top[pos] = top[pos - 1u];
                pos--;
            }
            if (pos < TOP_TENSORS) {
                top[pos] = i;
                if (top_count < TOP_TENSORS) top_count++;
            }
        }

        const uocr_section_entry *tensor_data = uocr_model_file_find_section(&model, UOCR_SECTION_TENSOR_DATA);
        const uint64_t tensor_data_bytes = tensor_data != NULL ? tensor_data->size : 0u;
        const uint64_t metadata_bytes = tensor_data_bytes <= model.size ? (uint64_t)model.size - tensor_data_bytes : 0u;
        printf("tensors: %u\n", model.tensor_count);
        printf("  payload_bytes=%llu runtime=%u preserved-unused=%u omitted=%u\n",
               (unsigned long long)payload_bytes,
               usage_runtime,
               usage_preserved,
               usage_omitted);
        if (tensor_data != NULL) {
            printf("  tensor_data: offset=%llu size=%llu alignment=%llu page_aligned=%s\n",
                   (unsigned long long)tensor_data->offset,
                   (unsigned long long)tensor_data->size,
                   (unsigned long long)tensor_data->alignment,
                   (tensor_data->offset % UOCR_TENSOR_DATA_ALIGNMENT) == 0u ? "yes" : "no");
        }
        if (min_payload_offset != UINT64_MAX) {
            const uint64_t payload_span = max_payload_end - min_payload_offset;
            const uint64_t tensor_data_slack = tensor_data_bytes >= payload_bytes ? tensor_data_bytes - payload_bytes : 0u;
            printf("  payload_span: [%llu,%llu) span=%llu unaligned_payloads=%u slack_in_tensor_data=%llu\n",
                   (unsigned long long)min_payload_offset,
                   (unsigned long long)max_payload_end,
                   (unsigned long long)payload_span,
                   unaligned_payloads,
                   (unsigned long long)tensor_data_slack);
        }
        printf("  expected_memory: mmap_file=%llu tensor_data_view=%llu tensor_payload=%llu metadata=%llu\n",
               (unsigned long long)model.size,
               (unsigned long long)tensor_data_bytes,
               (unsigned long long)payload_bytes,
               (unsigned long long)metadata_bytes);
        printf("  qtypes:");
        for (uint32_t i = 0u; i < 64u; ++i) {
            if (qtype_counts[i] == 0u) continue;
            printf(" %s=%u", uocr_tensor_qtype_name(i), qtype_counts[i]);
        }
        printf("\n");
        printf("  qtype_bytes:");
        for (uint32_t i = 0u; i < 64u; ++i) {
            if (qtype_bytes[i] == 0u) continue;
            printf(" %s=%llu", uocr_tensor_qtype_name(i), (unsigned long long)qtype_bytes[i]);
        }
        printf("\n");
        printf("  qtype_reasons:");
        for (uint32_t i = 0u; i < 16u; ++i) {
            if (qtype_reason_counts[i] == 0u) continue;
            printf(" %s=%u", uocr_tensor_qtype_reason_name(i), qtype_reason_counts[i]);
        }
        printf("\n");
        printf("  promotion_reasons:");
        for (uint32_t i = 0u; i < 16u; ++i) {
            if (promotion_reason_counts[i] == 0u) continue;
            printf(" %s=%u", uocr_tensor_promotion_reason_name(i), promotion_reason_counts[i]);
        }
        printf("\n");
        printf("  largest_tensors:\n");
        for (uint32_t i = 0u; i < top_count; ++i) {
            const uocr_tensor_entry *tensor = &model.tensors[top[i]];
            printf("    [%u] id=%u family=%s qtype=%s qtype_reason=%s promotion=%s flags=0x%x offset=%llu size=%llu offset_alignment=%llu",
                   i,
                   tensor->id,
                   uocr_tensor_family_name(tensor->family),
                   uocr_tensor_qtype_name(tensor->qtype),
                   uocr_tensor_qtype_reason_name(tensor->qtype_reason),
                   uocr_tensor_promotion_reason_name(tensor->promotion_reason),
                   tensor->flags,
                   (unsigned long long)tensor->payload_offset,
                   (unsigned long long)tensor->payload_size,
                   (unsigned long long)natural_alignment(tensor->payload_offset));
            printf("\n");
        }
    }

    uocr_model_file_close(&model);
    return 0;
}
