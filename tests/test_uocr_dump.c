#include "uocr_test_model_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#define CHECK(cond)                                                                        \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

#define OUTPUT_CAPACITY 32768u
#define COMMAND_CAPACITY 4096u

typedef struct dump_run_result {
    int exit_code;
    char output[OUTPUT_CAPACITY];
} dump_run_result;

static int append_text(char *dst, size_t dst_size, size_t *offset, const char *text) {
    if (dst == NULL || offset == NULL || text == NULL) {
        return 0;
    }
    while (*text != '\0') {
        if (*offset + 1u >= dst_size) {
            return 0;
        }
        dst[*offset] = *text;
        ++(*offset);
        ++text;
    }
    dst[*offset] = '\0';
    return 1;
}

static int append_shell_quoted(char *dst, size_t dst_size, size_t *offset, const char *text) {
    if (!append_text(dst, dst_size, offset, "'")) {
        return 0;
    }
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p == '\'') {
            if (!append_text(dst, dst_size, offset, "'\\''")) {
                return 0;
            }
        } else {
            char one[2] = {*p, '\0'};
            if (!append_text(dst, dst_size, offset, one)) {
                return 0;
            }
        }
    }
    return append_text(dst, dst_size, offset, "'");
}

static int build_dump_command(const char *tool_path, const char *model_path, char *command, size_t command_size) {
    size_t offset = 0u;
    command[0] = '\0';
    if (!append_shell_quoted(command, command_size, &offset, tool_path)) {
        return 0;
    }
    if (!append_text(command, command_size, &offset, " ")) {
        return 0;
    }
    if (!append_shell_quoted(command, command_size, &offset, model_path)) {
        return 0;
    }
    return append_text(command, command_size, &offset, " 2>&1");
}

static int decode_wait_status(int status) {
#if defined(_WIN32)
    return status;
#else
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 255;
#endif
}

static int run_uocr_dump(const char *tool_path, const char *model_path, dump_run_result *result) {
    if (tool_path == NULL || model_path == NULL || result == NULL) {
        return 1;
    }
    memset(result, 0, sizeof(*result));
    result->exit_code = -1;

    char command[COMMAND_CAPACITY];
    if (!build_dump_command(tool_path, model_path, command, sizeof(command))) {
        fprintf(stderr, "failed to build uocr-dump command\n");
        return 1;
    }

    FILE *pipe = popen(command, "r");
    if (pipe == NULL) {
        perror("popen");
        return 1;
    }

    size_t used = 0u;
    while (!feof(pipe)) {
        if (used + 1u >= sizeof(result->output)) {
            fprintf(stderr, "uocr-dump output exceeded %u bytes\n", OUTPUT_CAPACITY);
            (void)pclose(pipe);
            return 1;
        }
        const size_t n = fread(result->output + used, 1u, sizeof(result->output) - used - 1u, pipe);
        used += n;
        if (ferror(pipe)) {
            perror("fread");
            (void)pclose(pipe);
            return 1;
        }
    }
    result->output[used] = '\0';

    const int status = pclose(pipe);
    if (status < 0) {
        perror("pclose");
        return 1;
    }
    result->exit_code = decode_wait_status(status);
    return 0;
}

static int output_contains(const dump_run_result *result, const char *needle) {
    return result != NULL && needle != NULL && strstr(result->output, needle) != NULL;
}

static int write_truncated_file(const char *path) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        perror("fopen");
        return 1;
    }
    static const unsigned char bytes[] = {'U', 'O', 'C', 'R'};
    const int failed = fwrite(bytes, 1u, sizeof(bytes), f) != sizeof(bytes);
    const int close_failed = fclose(f) != 0;
    return failed || close_failed ? 1 : 0;
}

static int test_dump_synthetic_model_reports_core_contract(const char *tool_path) {
    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_two_tensor_uocr_model(path) == 0);

    dump_run_result result;
    CHECK(run_uocr_dump(tool_path, path, &result) == 0);
    if (result.exit_code != 0) {
        fprintf(stderr, "uocr-dump failed with exit code %d; output:\n%s\n", result.exit_code, result.output);
        (void)unlink(path);
        return 1;
    }

    CHECK(output_contains(&result, "uocr-dump (ABI 0x"));
    CHECK(output_contains(&result, "format_version: 1"));
    CHECK(output_contains(&result, "alignment: 4096"));
    CHECK(output_contains(&result, "qprofile: fp16 (1)"));
    CHECK(output_contains(&result, "sections: 5"));
    CHECK(output_contains(&result, "[0] config"));
    CHECK(output_contains(&result, "[1] tokenizer-metadata"));
    CHECK(output_contains(&result, "[2] provenance"));
    CHECK(output_contains(&result, "[3] tensor-directory"));
    CHECK(output_contains(&result, "[4] tensor-data"));

    CHECK(output_contains(&result, "config:"));
    CHECK(output_contains(&result, "vocab=129280 hidden=1280 layers=12 heads=10 kv_heads=10 head_dim=128"));
    CHECK(output_contains(&result, "max_positions=32768 rope_theta=10000 generated_ring=128 image_token=128815"));
    CHECK(output_contains(&result, "vision: global=1024 local=640 patch=16 downsample=4"));

    CHECK(output_contains(&result, "tokenizer:"));
    CHECK(output_contains(&result, "vocab=129280 bpe_vocab=128000 added=830 bos=0 eos=1 pad=2 image=128815"));
    CHECK(output_contains(&result, "flags=0x1 hash=1011121314151617... payload_bytes=0"));

    CHECK(output_contains(&result, "provenance:"));
    CHECK(output_contains(&result, "source_tensors=2 runtime=2 preserved-unused=0 omitted=0 safetensors_files=1"));
    CHECK(output_contains(&result, "converter=0.0.0 qprofile=fp16 (1) json_bytes=0"));
    CHECK(output_contains(&result, "full_fp16_accounting: not-current-full-model (full fp16 source tensor count mismatch"));

    CHECK(output_contains(&result, "tensors: 2"));
    CHECK(output_contains(&result, "payload_bytes=48 runtime=2 preserved-unused=0 omitted=0"));
    CHECK(output_contains(&result, "tensor_data:"));
    CHECK(output_contains(&result, "alignment=4096 page_aligned=yes"));
    CHECK(output_contains(&result, "payload_span:"));
    CHECK(output_contains(&result, "unaligned_payloads=0"));
    CHECK(output_contains(&result, "slack_in_tensor_data=4048"));
    CHECK(output_contains(&result, "expected_memory:"));
    CHECK(output_contains(&result, "qtypes: f16=2"));
    CHECK(output_contains(&result, "qtype_bytes: f16=48"));
    CHECK(output_contains(&result, "qtype_reasons: unknown=2"));
    CHECK(output_contains(&result, "promotion_reasons: none=2"));
    CHECK(output_contains(&result, "largest_tensors:"));
    CHECK(output_contains(&result, "id=2 family=LM_HEAD qtype=f16"));
    CHECK(output_contains(&result, "id=1 family=TOK_EMBED qtype=f16"));

    CHECK(unlink(path) == 0);
    return 0;
}

static int test_dump_rejects_invalid_model_with_loader_error(const char *tool_path) {
    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(write_truncated_file(path) == 0);

    dump_run_result result;
    CHECK(run_uocr_dump(tool_path, path, &result) == 0);
    CHECK(result.exit_code != 0);
    CHECK(output_contains(&result, "uocr-dump: file too small for UOCR header"));

    CHECK(unlink(path) == 0);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2 || argv[1] == NULL || argv[1][0] == '\0') {
        fprintf(stderr, "usage: %s /path/to/uocr-dump\n", argc > 0 ? argv[0] : "test_uocr_dump");
        return 2;
    }

    if (test_dump_synthetic_model_reports_core_contract(argv[1]) != 0) return 1;
    if (test_dump_rejects_invalid_model_with_loader_error(argv[1]) != 0) return 1;
    return 0;
}
