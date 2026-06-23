#include <stdio.h>

#include "model/uocr_model_file.h"
#include "unlimitedocr.h"

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
        printf("  max_positions=%u generated_ring=%u image_token=%u\n",
               model.config->max_positions,
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

    uocr_model_file_close(&model);
    return 0;
}
