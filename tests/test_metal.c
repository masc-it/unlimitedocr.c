#include "backend/metal/uocr_metal.h"
#include "model/uocr_constants.h"
#include "model/uocr_model_file.h"
#include "model/uocr_tensor_registry.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_request_validation.h"
#include "runtime/uocr_sequence.h"
#include "unlimitedocr.h"

#include "uocr_test_model_file.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef UOCR_TEST_METAL_RESOURCE_PATH
#define UOCR_TEST_METAL_RESOURCE_PATH NULL
#endif

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static int test_metal_smoke(void) {
    if (!uocr_metal_is_available()) {
        printf("Metal device not available; skipping Metal smoke test\n");
        return 0;
    }

    const uint64_t recommended = uocr_metal_recommended_working_set_size();
    CHECK(recommended > 0u);
    const uint64_t default_budget = uocr_metal_default_memory_budget_bytes(recommended);
    CHECK(default_budget > 0u);
    CHECK(default_budget <= recommended);

    char error[1024];
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_smoke_test(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    return 0;
}

static size_t sam_patch_weight_index(uint32_t out_channel, uint32_t in_channel, uint32_t ky, uint32_t kx) {
    return (size_t)((((out_channel * 3u + in_channel) * UOCR_VISION_PATCH_SIZE + ky) * UOCR_VISION_PATCH_SIZE + kx));
}

static int test_metal_named_scratch_buffers(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) == 0u);
    CHECK(uocr_metal_context_total_scratch_high_watermark(ctx) == 0u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 1024u, 0, error, sizeof(error)) == 1);
    const uint64_t first_capacity = uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(first_capacity >= 1024u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, 512u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == 1024u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_DECODER, first_capacity + 1u, 0, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) >= first_capacity + 1u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    CHECK(uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_LOGITS, 4096u, 1, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS) >= 4096u);
    CHECK(uocr_metal_context_total_scratch_capacity(ctx) >=
          uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) +
              uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_LOGITS));

    uocr_metal_context_release_scratch(ctx, UOCR_METAL_SCRATCH_DECODER);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_DECODER) == 0u);
    CHECK(uocr_metal_context_scratch_high_watermark(ctx, UOCR_METAL_SCRATCH_DECODER) == first_capacity + 1u);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16u;
    uint32_t bits = 0u;
    uint32_t mantissa = (uint32_t)h & 0x03ffu;
    int exponent = (int)((h >> 10u) & 0x001fu);
    if (exponent == 0) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            mantissa &= ~0x0400u;
            bits = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((uint32_t)(exponent + 127 - 15) << 23u) | (mantissa << 13u);
    }
    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0u) {
            return (uint16_t)(sign | 0x7c00u);
        }
        return (uint16_t)(sign | 0x7e00u);
    }

    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return (uint16_t)sign;
        }
        uint32_t mant = mantissa | 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mant >> shift;
        const uint32_t round_bit = 1u << (shift - 1u);
        const uint32_t remainder = mant & (round_bit - 1u);
        if ((mant & round_bit) != 0u && (remainder != 0u || (rounded & 1u) != 0u)) {
            ++rounded;
        }
        return (uint16_t)(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13u;
    const uint32_t round_bit = 0x00001000u;
    const uint32_t remainder = mantissa & (round_bit - 1u);
    if ((mantissa & round_bit) != 0u && (remainder != 0u || (rounded_mantissa & 1u) != 0u)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0u;
            ++half_exponent;
            if (half_exponent >= 31) {
                return (uint16_t)(sign | 0x7c00u);
            }
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10u) | rounded_mantissa);
}

static int test_metal_sam_patch_embed_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    enum { WIDTH = 32u, HEIGHT = 32u, GRID = 2u, PATCH = UOCR_VISION_PATCH_SIZE };
    float *pixels_f32 = (float *)calloc(3u * WIDTH * HEIGHT, sizeof(float));
    uint16_t *weights = (uint16_t *)calloc((size_t)UOCR_SAM_HIDDEN_SIZE * 3u * PATCH * PATCH, sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)calloc(UOCR_SAM_HIDDEN_SIZE, sizeof(uint16_t));
    uint16_t *out = (uint16_t *)calloc(GRID * GRID * UOCR_SAM_HIDDEN_SIZE, sizeof(uint16_t));
    CHECK(pixels_f32 != NULL);
    CHECK(weights != NULL);
    CHECK(bias != NULL);
    CHECK(out != NULL);

    for (uint32_t c = 0u; c < 3u; ++c) {
        for (uint32_t y = 0u; y < HEIGHT; ++y) {
            for (uint32_t x = 0u; x < WIDTH; ++x) {
                pixels_f32[(c * HEIGHT + y) * WIDTH + x] = (float)(c * 100u + y + x);
            }
        }
    }
    weights[sam_patch_weight_index(0u, 0u, 0u, 0u)] = f32_to_f16_bits(1.0f);
    weights[sam_patch_weight_index(1u, 1u, 15u, 15u)] = f32_to_f16_bits(1.0f);
    weights[sam_patch_weight_index(2u, 0u, 0u, 0u)] = f32_to_f16_bits(1.0f);
    weights[sam_patch_weight_index(2u, 2u, 1u, 2u)] = f32_to_f16_bits(1.0f);
    bias[2u] = f32_to_f16_bits(3.0f);
    bias[7u] = f32_to_f16_bits(42.0f);

    uint32_t grid_w = 0u;
    uint32_t grid_h = 0u;
    CHECK(uocr_metal_context_sam_patch_embed_f16(ctx,
                                                 pixels_f32,
                                                 UOCR_PIXEL_F32_NCHW,
                                                 WIDTH,
                                                 HEIGHT,
                                                 weights,
                                                 bias,
                                                 out,
                                                 &grid_w,
                                                 &grid_h,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(grid_w == GRID);
    CHECK(grid_h == GRID);

    for (uint32_t py = 0u; py < GRID; ++py) {
        for (uint32_t px = 0u; px < GRID; ++px) {
            const uint32_t patch_index = py * GRID + px;
            const uint32_t base = patch_index * UOCR_SAM_HIDDEN_SIZE;
            const float expected0 = (float)(py * 16u + px * 16u);
            const float expected1 = (float)(100u + py * 16u + 15u + px * 16u + 15u);
            const float expected2 = expected0 + (float)(200u + py * 16u + 1u + px * 16u + 2u) + 3.0f;
            CHECK(out[base + 0u] == f32_to_f16_bits(expected0));
            CHECK(out[base + 1u] == f32_to_f16_bits(expected1));
            CHECK(out[base + 2u] == f32_to_f16_bits(expected2));
            CHECK(out[base + 7u] == f32_to_f16_bits(42.0f));
            CHECK(out[base + 9u] == f32_to_f16_bits(0.0f));
        }
    }

    uint16_t *pixels_f16 = (uint16_t *)calloc(3u * PATCH * PATCH, sizeof(uint16_t));
    CHECK(pixels_f16 != NULL);
    memset(weights, 0, (size_t)UOCR_SAM_HIDDEN_SIZE * 3u * PATCH * PATCH * sizeof(uint16_t));
    memset(out, 0, GRID * GRID * UOCR_SAM_HIDDEN_SIZE * sizeof(uint16_t));
    for (uint32_t c = 0u; c < 3u; ++c) {
        for (uint32_t y = 0u; y < PATCH; ++y) {
            for (uint32_t x = 0u; x < PATCH; ++x) {
                pixels_f16[(c * PATCH + y) * PATCH + x] = f32_to_f16_bits((float)(c * 100u + y + x));
            }
        }
    }
    weights[sam_patch_weight_index(5u, 2u, 3u, 4u)] = f32_to_f16_bits(1.0f);
    CHECK(uocr_metal_context_sam_patch_embed_f16(ctx,
                                                 pixels_f16,
                                                 UOCR_PIXEL_F16_NCHW,
                                                 PATCH,
                                                 PATCH,
                                                 weights,
                                                 NULL,
                                                 out,
                                                 &grid_w,
                                                 &grid_h,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(grid_w == 1u);
    CHECK(grid_h == 1u);
    CHECK(out[5u] == f32_to_f16_bits(207.0f));
    CHECK(out[7u] == f32_to_f16_bits(0.0f));

    CHECK(uocr_metal_context_sam_patch_embed_f16(ctx,
                                                 pixels_f32,
                                                 UOCR_PIXEL_F32_NCHW,
                                                 17u,
                                                 HEIGHT,
                                                 weights,
                                                 bias,
                                                 out,
                                                 &grid_w,
                                                 &grid_h,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "divisible") != NULL);

    free(pixels_f16);
    free(out);
    free(bias);
    free(weights);
    free(pixels_f32);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int env_flag_enabled(const char *name) {
    const char *value = getenv(name);
    return value != NULL && (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0 ||
                             strcmp(value, "yes") == 0 || strcmp(value, "YES") == 0);
}

static uint32_t env_u32_or_default(const char *name, uint32_t default_value) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return default_value;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > (unsigned long)UINT32_MAX) {
        return default_value;
    }
    return (uint32_t)parsed;
}

static uint16_t bf16_bits_to_f16_bits(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16u;
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return f32_to_f16_bits(value);
}

static int read_safetensors_u64_le(FILE *f, uint64_t *out) {
    unsigned char bytes[8];
    if (fread(bytes, 1u, sizeof(bytes), f) != sizeof(bytes)) {
        return 0;
    }
    uint64_t value = 0u;
    for (uint32_t i = 0u; i < 8u; ++i) {
        value |= ((uint64_t)bytes[i]) << (8u * i);
    }
    *out = value;
    return 1;
}

static const char *parse_u64_after(const char *cursor, uint64_t *out) {
    if (cursor == NULL || out == NULL) {
        return NULL;
    }
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
        ++cursor;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long long value = strtoull(cursor, &end, 10);
    if (errno != 0 || end == cursor) {
        return NULL;
    }
    *out = (uint64_t)value;
    return end;
}

static int find_safetensors_tensor_range(const char *header,
                                         const char *tensor_name,
                                         uint64_t *out_start,
                                         uint64_t *out_end) {
    if (header == NULL || tensor_name == NULL || out_start == NULL || out_end == NULL) {
        return 0;
    }

    char key[256];
    if (snprintf(key, sizeof(key), "\"%s\"", tensor_name) >= (int)sizeof(key)) {
        return 0;
    }
    const char *entry = strstr(header, key);
    if (entry == NULL) {
        return 0;
    }
    const char *next_entry = strstr(entry + strlen(key), "},\"");
    const char *dtype = strstr(entry, "\"dtype\":\"BF16\"");
    const char *shape = strstr(entry, "\"shape\":[129280,1280]");
    const char *offsets = strstr(entry, "\"data_offsets\":[");
    if (dtype == NULL || shape == NULL || offsets == NULL ||
        (next_entry != NULL && (dtype > next_entry || shape > next_entry || offsets > next_entry))) {
        return 0;
    }
    offsets += strlen("\"data_offsets\":[");
    uint64_t start = 0u;
    uint64_t end = 0u;
    const char *cursor = parse_u64_after(offsets, &start);
    if (cursor == NULL || *cursor != ',') {
        return 0;
    }
    cursor = parse_u64_after(cursor + 1, &end);
    if (cursor == NULL || *cursor != ']' || end <= start) {
        return 0;
    }
    *out_start = start;
    *out_end = end;
    return 1;
}

static int make_safetensors_path(const char *hf_dir, char *out, size_t out_size) {
    static const char *candidates[] = {"model-00001-of-000001.safetensors", "model.safetensors"};
    if (hf_dir == NULL || hf_dir[0] == '\0' || out == NULL || out_size == 0u) {
        return 0;
    }
    for (uint32_t i = 0u; i < (uint32_t)(sizeof(candidates) / sizeof(candidates[0])); ++i) {
        if (snprintf(out, out_size, "%s/%s", hf_dir, candidates[i]) >= (int)out_size) {
            return 0;
        }
        if (access(out, R_OK) == 0) {
            return 1;
        }
    }
    return 0;
}

static int read_hf_embed_rows_as_f16(const char *hf_dir,
                                     const int32_t *token_ids,
                                     uint32_t n_tokens,
                                     uint16_t *out_f16,
                                     char *error,
                                     size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
    if (hf_dir == NULL || token_ids == NULL || out_f16 == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid HF embedding row request");
        return 0;
    }

    char path[4096];
    if (!make_safetensors_path(hf_dir, path, sizeof(path))) {
        snprintf(error, error_size, "no safetensors payload found under %s", hf_dir);
        return 0;
    }

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(error, error_size, "failed to open %s", path);
        return 0;
    }

    uint64_t header_len = 0u;
    if (!read_safetensors_u64_le(f, &header_len) || header_len == 0u || header_len > 256ull * 1024ull * 1024ull) {
        snprintf(error, error_size, "invalid safetensors header length in %s", path);
        fclose(f);
        return 0;
    }
    char *header = (char *)malloc((size_t)header_len + 1u);
    if (header == NULL) {
        snprintf(error, error_size, "failed to allocate safetensors header buffer");
        fclose(f);
        return 0;
    }
    if (fread(header, 1u, (size_t)header_len, f) != (size_t)header_len) {
        snprintf(error, error_size, "failed to read safetensors header from %s", path);
        free(header);
        fclose(f);
        return 0;
    }
    header[header_len] = '\0';

    uint64_t tensor_start = 0u;
    uint64_t tensor_end = 0u;
    if (!find_safetensors_tensor_range(header, "model.embed_tokens.weight", &tensor_start, &tensor_end)) {
        snprintf(error, error_size, "failed to locate model.embed_tokens.weight in %s", path);
        free(header);
        fclose(f);
        return 0;
    }
    free(header);

    const uint64_t data_start = 8u + header_len;
    const uint64_t row_bytes = (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t expected_tensor_bytes = (uint64_t)UOCR_VOCAB_SIZE * row_bytes;
    if (tensor_end - tensor_start != expected_tensor_bytes) {
        snprintf(error,
                 error_size,
                 "unexpected token embedding byte size %llu in %s",
                 (unsigned long long)(tensor_end - tensor_start),
                 path);
        fclose(f);
        return 0;
    }

    unsigned char row[UOCR_HIDDEN_SIZE * 2u];
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        if (token_ids[token] < 0 || (uint32_t)token_ids[token] >= UOCR_VOCAB_SIZE) {
            snprintf(error, error_size, "token id %d out of vocab", token_ids[token]);
            fclose(f);
            return 0;
        }
        const uint64_t row_offset = data_start + tensor_start + (uint64_t)(uint32_t)token_ids[token] * row_bytes;
        if (row_offset > (uint64_t)LONG_MAX || fseek(f, (long)row_offset, SEEK_SET) != 0) {
            snprintf(error, error_size, "failed to seek token embedding row %d", token_ids[token]);
            fclose(f);
            return 0;
        }
        if (fread(row, 1u, sizeof(row), f) != sizeof(row)) {
            snprintf(error, error_size, "failed to read token embedding row %d", token_ids[token]);
            fclose(f);
            return 0;
        }
        for (uint32_t col = 0u; col < (uint32_t)UOCR_HIDDEN_SIZE; ++col) {
            const uint16_t bf16 = (uint16_t)row[2u * col] | (uint16_t)((uint16_t)row[2u * col + 1u] << 8u);
            out_f16[token * (uint32_t)UOCR_HIDDEN_SIZE + col] = bf16_bits_to_f16_bits(bf16);
        }
    }

    fclose(f);
    return 1;
}

static int make_fixture_path(const char *root, const char *name, char *out, size_t out_size) {
    if (root == NULL || root[0] == '\0' || name == NULL || out == NULL || out_size == 0u) {
        return 0;
    }
    const int written = snprintf(out, out_size, "%s/%s", root, name);
    return written >= 0 && (size_t)written < out_size;
}

static int fixture_binary_exists(const char *root, const char *name) {
    char path[4096];
    if (!make_fixture_path(root, name, path, sizeof(path))) {
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    fclose(f);
    return 1;
}

static int read_binary_file(const char *path, unsigned char **out_bytes, size_t *out_size, char *error, size_t error_size) {
    if (out_bytes == NULL || out_size == NULL) {
        snprintf(error, error_size, "invalid binary read request");
        return 0;
    }
    *out_bytes = NULL;
    *out_size = 0u;
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(error, error_size, "failed to open %s", path);
        return 0;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        snprintf(error, error_size, "failed to seek %s", path);
        fclose(f);
        return 0;
    }
    const long size_long = ftell(f);
    if (size_long <= 0) {
        snprintf(error, error_size, "invalid empty binary file %s", path);
        fclose(f);
        return 0;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        snprintf(error, error_size, "failed to rewind %s", path);
        fclose(f);
        return 0;
    }
    unsigned char *bytes = (unsigned char *)malloc((size_t)size_long);
    if (bytes == NULL) {
        snprintf(error, error_size, "failed to allocate %ld bytes for %s", size_long, path);
        fclose(f);
        return 0;
    }
    if (fread(bytes, 1u, (size_t)size_long, f) != (size_t)size_long) {
        snprintf(error, error_size, "failed to read %s", path);
        free(bytes);
        fclose(f);
        return 0;
    }
    fclose(f);
    *out_bytes = bytes;
    *out_size = (size_t)size_long;
    return 1;
}

static int read_text_file_allow_empty(const char *path, char **out_text, size_t *out_size, char *error, size_t error_size) {
    if (out_text == NULL || out_size == NULL) {
        snprintf(error, error_size, "invalid text read request");
        return 0;
    }
    *out_text = NULL;
    *out_size = 0u;
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(error, error_size, "failed to open %s", path);
        return 0;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        snprintf(error, error_size, "failed to seek %s", path);
        fclose(f);
        return 0;
    }
    const long size_long = ftell(f);
    if (size_long < 0) {
        snprintf(error, error_size, "failed to tell %s", path);
        fclose(f);
        return 0;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        snprintf(error, error_size, "failed to rewind %s", path);
        fclose(f);
        return 0;
    }
    if ((uint64_t)size_long > (uint64_t)SIZE_MAX - 1u) {
        snprintf(error, error_size, "text file %s is too large", path);
        fclose(f);
        return 0;
    }
    const size_t alloc_size = (size_t)size_long + 1u;
    char *text = (char *)malloc(alloc_size);
    if (text == NULL) {
        snprintf(error, error_size, "failed to allocate %zu bytes for %s", alloc_size, path);
        fclose(f);
        return 0;
    }
    if (size_long > 0 && fread(text, 1u, (size_t)size_long, f) != (size_t)size_long) {
        snprintf(error, error_size, "failed to read %s", path);
        free(text);
        fclose(f);
        return 0;
    }
    fclose(f);
    text[(size_t)size_long] = '\0';
    *out_text = text;
    *out_size = (size_t)size_long;
    return 1;
}

static int read_generated_text_python_dump(const char *dump_dir, char **text_out, char *error, size_t error_size) {
    if (text_out == NULL) {
        snprintf(error, error_size, "invalid generated-text dump request");
        return 0;
    }
    *text_out = NULL;
    char path[4096];
    if (!make_fixture_path(dump_dir, "generated_text.txt", path, sizeof(path))) {
        snprintf(error, error_size, "generated-text dump path is too long");
        return 0;
    }
    size_t text_size = 0u;
    return read_text_file_allow_empty(path, text_out, &text_size, error, error_size);
}

static int read_prepared_tokens_python_dump(const char *dump_dir,
                                             int32_t **input_ids_out,
                                             uint8_t **image_mask_out,
                                             uint32_t *n_tokens_out,
                                             char *error,
                                             size_t error_size) {
    if (input_ids_out == NULL || image_mask_out == NULL || n_tokens_out == NULL) {
        snprintf(error, error_size, "invalid prepared-token dump request");
        return 0;
    }
    *input_ids_out = NULL;
    *image_mask_out = NULL;
    *n_tokens_out = 0u;

    char ids_path[4096];
    char mask_path[4096];
    if (!make_fixture_path(dump_dir, "input_ids_i32.bin", ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, "image_mask_u8.bin", mask_path, sizeof(mask_path))) {
        snprintf(error, error_size, "prepared-token dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    size_t ids_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size)) {
        return 0;
    }
    if (ids_size == 0u || ids_size % sizeof(int32_t) != 0u || ids_size / sizeof(int32_t) > (size_t)UINT32_MAX) {
        snprintf(error, error_size, "invalid input_ids_i32.bin size %zu", ids_size);
        free(ids_bytes);
        return 0;
    }
    const uint32_t n_tokens = (uint32_t)(ids_size / sizeof(int32_t));

    unsigned char *mask_bytes = NULL;
    size_t mask_size = 0u;
    if (!read_binary_file(mask_path, &mask_bytes, &mask_size, error, error_size)) {
        free(ids_bytes);
        return 0;
    }
    if (mask_size != (size_t)n_tokens) {
        snprintf(error,
                 error_size,
                 "image_mask_u8.bin size mismatch: expected %u bytes, got %zu",
                 n_tokens,
                 mask_size);
        free(mask_bytes);
        free(ids_bytes);
        return 0;
    }

    int32_t *ids = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    uint8_t *mask = (uint8_t *)malloc((size_t)n_tokens * sizeof(uint8_t));
    if (ids == NULL || mask == NULL) {
        snprintf(error, error_size, "failed to allocate prepared-token dump buffers");
        free(mask);
        free(ids);
        free(mask_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < n_tokens; ++i) {
        const unsigned char *p = ids_bytes + 4u * i;
        const uint32_t raw = (uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) |
                             ((uint32_t)p[3] << 24u);
        memcpy(&ids[i], &raw, sizeof(ids[i]));
        if (ids[i] < 0 || (uint32_t)ids[i] >= UOCR_VOCAB_SIZE) {
            snprintf(error, error_size, "input_ids_i32.bin token %u is out of vocabulary: %d", i, ids[i]);
            free(mask);
            free(ids);
            free(mask_bytes);
            free(ids_bytes);
            return 0;
        }
        if (mask_bytes[i] > 1u) {
            snprintf(error, error_size, "image_mask_u8.bin has invalid value %u at token %u", mask_bytes[i], i);
            free(mask);
            free(ids);
            free(mask_bytes);
            free(ids_bytes);
            return 0;
        }
        mask[i] = (uint8_t)mask_bytes[i];
    }

    free(mask_bytes);
    free(ids_bytes);
    *input_ids_out = ids;
    *image_mask_out = mask;
    *n_tokens_out = n_tokens;
    return 1;
}

static const char *json_find_key_range(const char *start, const char *end, const char *key) {
    if (start == NULL || end == NULL || key == NULL || end < start) {
        return NULL;
    }
    char pattern[128];
    const int written = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    const size_t pattern_len = (size_t)written;
    for (const char *p = start; p + pattern_len <= end; ++p) {
        if (memcmp(p, pattern, pattern_len) == 0) {
            return p;
        }
    }
    return NULL;
}

static const char *json_find_char_range(const char *start, const char *end, char ch) {
    if (start == NULL || end == NULL || end < start) {
        return NULL;
    }
    for (const char *p = start; p < end; ++p) {
        if (*p == ch) {
            return p;
        }
    }
    return NULL;
}

static const char *json_skip_ws_range(const char *p, const char *end) {
    while (p != NULL && p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
        ++p;
    }
    return p;
}

static int json_parse_u32_key_range(const char *start, const char *end, const char *key, uint32_t *out) {
    const char *p = json_find_key_range(start, end, key);
    if (p == NULL) {
        return 0;
    }
    p = json_find_char_range(p, end, ':');
    if (p == NULL) {
        return 0;
    }
    p = json_skip_ws_range(p + 1, end);
    if (p == NULL || p >= end) {
        return 0;
    }
    errno = 0;
    char *after = NULL;
    const unsigned long value = strtoul(p, &after, 10);
    if (after == p || errno != 0 || value > (unsigned long)UINT32_MAX) {
        return 0;
    }
    *out = (uint32_t)value;
    return 1;
}

static int json_parse_string_key_range(const char *start,
                                       const char *end,
                                       const char *key,
                                       char *out,
                                       size_t out_size) {
    if (out == NULL || out_size == 0u) {
        return 0;
    }
    const char *p = json_find_key_range(start, end, key);
    if (p == NULL) {
        return 0;
    }
    p = json_find_char_range(p, end, ':');
    if (p == NULL) {
        return 0;
    }
    p = json_skip_ws_range(p + 1, end);
    if (p == NULL || p >= end || *p != '"') {
        return 0;
    }
    const char *value_start = p + 1;
    const char *value_end = json_find_char_range(value_start, end, '"');
    if (value_end == NULL || value_end < value_start) {
        return 0;
    }
    const size_t len = (size_t)(value_end - value_start);
    if (len >= out_size) {
        return 0;
    }
    memcpy(out, value_start, len);
    out[len] = '\0';
    return 1;
}

static int read_prepared_view_stubs_from_manifest(const char *dump_dir,
                                                  uocr_image_view **views_out,
                                                  uint32_t *n_views_out,
                                                  uint32_t *crop_grid_w_out,
                                                  uint32_t *crop_grid_h_out,
                                                  char *error,
                                                  size_t error_size) {
    if (views_out == NULL || n_views_out == NULL || crop_grid_w_out == NULL || crop_grid_h_out == NULL) {
        snprintf(error, error_size, "invalid fixture view-manifest request");
        return 0;
    }
    *views_out = NULL;
    *n_views_out = 0u;
    *crop_grid_w_out = 0u;
    *crop_grid_h_out = 0u;

    char manifest_path[4096];
    if (!make_fixture_path(dump_dir, "manifest.json", manifest_path, sizeof(manifest_path))) {
        snprintf(error, error_size, "fixture manifest path is too long");
        return 0;
    }
    unsigned char *manifest_bytes = NULL;
    size_t manifest_size = 0u;
    if (!read_binary_file(manifest_path, &manifest_bytes, &manifest_size, error, error_size)) {
        return 0;
    }
    char *json = (char *)malloc(manifest_size + 1u);
    if (json == NULL) {
        snprintf(error, error_size, "failed to allocate fixture manifest buffer");
        free(manifest_bytes);
        return 0;
    }
    memcpy(json, manifest_bytes, manifest_size);
    json[manifest_size] = '\0';
    free(manifest_bytes);

    const char *json_start = json;
    const char *json_end = json + manifest_size;
    uint32_t crop_grid_w = 0u;
    uint32_t crop_grid_h = 0u;
    if (!json_parse_u32_key_range(json_start, json_end, "crop_grid_w", &crop_grid_w) ||
        !json_parse_u32_key_range(json_start, json_end, "crop_grid_h", &crop_grid_h)) {
        snprintf(error, error_size, "fixture manifest is missing crop grid metadata");
        free(json);
        return 0;
    }

    const char *views_key = json_find_key_range(json_start, json_end, "views");
    if (views_key == NULL) {
        snprintf(error, error_size, "fixture manifest is missing views metadata");
        free(json);
        return 0;
    }
    const char *array_start = json_find_char_range(views_key, json_end, '[');
    if (array_start == NULL) {
        snprintf(error, error_size, "fixture manifest views metadata is invalid");
        free(json);
        return 0;
    }
    const char *array_end = NULL;
    int bracket_depth = 0;
    for (const char *p = array_start; p < json_end; ++p) {
        if (*p == '[') {
            ++bracket_depth;
        } else if (*p == ']') {
            --bracket_depth;
            if (bracket_depth == 0) {
                array_end = p;
                break;
            }
        }
    }
    if (array_end == NULL) {
        snprintf(error, error_size, "fixture manifest views array is unterminated");
        free(json);
        return 0;
    }

    uint32_t n_views = 0u;
    for (const char *p = array_start + 1; p < array_end; ++p) {
        if (*p == '{') {
            ++n_views;
            const char *object_end = json_find_char_range(p + 1, array_end, '}');
            if (object_end == NULL) {
                snprintf(error, error_size, "fixture manifest view object is unterminated");
                free(json);
                return 0;
            }
            p = object_end;
        }
    }

    uocr_image_view *views = NULL;
    if (n_views != 0u) {
        views = (uocr_image_view *)calloc((size_t)n_views, sizeof(*views));
        if (views == NULL) {
            snprintf(error, error_size, "failed to allocate fixture view stubs");
            free(json);
            return 0;
        }
    }

    static const uint16_t pixel_sentinel = 0u;
    uint32_t view_index = 0u;
    for (const char *p = array_start + 1; p < array_end;) {
        if (*p != '{') {
            ++p;
            continue;
        }
        const char *object_start = p;
        const char *object_end = json_find_char_range(object_start + 1, array_end, '}');
        if (object_end == NULL || view_index >= n_views) {
            snprintf(error, error_size, "fixture manifest view object parse failure");
            free(views);
            free(json);
            return 0;
        }

        char kind[32];
        char format[32];
        uint32_t width = 0u;
        uint32_t height = 0u;
        if (!json_parse_string_key_range(object_start, object_end, "kind", kind, sizeof(kind)) ||
            !json_parse_string_key_range(object_start, object_end, "format", format, sizeof(format)) ||
            !json_parse_u32_key_range(object_start, object_end, "width", &width) ||
            !json_parse_u32_key_range(object_start, object_end, "height", &height)) {
            snprintf(error, error_size, "fixture manifest view %u is missing kind/format/shape", view_index);
            free(views);
            free(json);
            return 0;
        }

        views[view_index].pixels = &pixel_sentinel;
        views[view_index].width = width;
        views[view_index].height = height;
        if (strcmp(kind, "global") == 0) {
            views[view_index].kind = UOCR_VIEW_GLOBAL;
        } else if (strcmp(kind, "local") == 0) {
            views[view_index].kind = UOCR_VIEW_LOCAL;
        } else {
            snprintf(error, error_size, "fixture manifest view %u has unsupported kind %s", view_index, kind);
            free(views);
            free(json);
            return 0;
        }
        if (strcmp(format, "f16_nchw") == 0) {
            views[view_index].format = UOCR_PIXEL_F16_NCHW;
        } else if (strcmp(format, "f32_nchw") == 0) {
            views[view_index].format = UOCR_PIXEL_F32_NCHW;
        } else {
            snprintf(error, error_size, "fixture manifest view %u has unsupported format %s", view_index, format);
            free(views);
            free(json);
            return 0;
        }

        ++view_index;
        p = object_end + 1;
    }
    if (view_index != n_views) {
        snprintf(error, error_size, "fixture manifest view count changed while parsing");
        free(views);
        free(json);
        return 0;
    }

    free(json);
    *views_out = views;
    *n_views_out = n_views;
    *crop_grid_w_out = crop_grid_w;
    *crop_grid_h_out = crop_grid_h;
    return 1;
}

static int read_prompt_embedding_python_dump(const char *dump_dir,
                                             int32_t **input_ids_out,
                                             uint32_t *n_tokens_out,
                                             uint16_t **embeddings_out,
                                             char *error,
                                             size_t error_size) {
    if (input_ids_out == NULL || n_tokens_out == NULL || embeddings_out == NULL) {
        snprintf(error, error_size, "invalid prompt embedding dump request");
        return 0;
    }
    *input_ids_out = NULL;
    *n_tokens_out = 0u;
    *embeddings_out = NULL;

    char ids_path[4096];
    char embeddings_path[4096];
    if (!make_fixture_path(dump_dir, "input_ids_i32.bin", ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, "prompt_embeddings_f16.bin", embeddings_path, sizeof(embeddings_path))) {
        snprintf(error, error_size, "prompt embedding dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    size_t ids_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size)) {
        return 0;
    }
    if (ids_size % sizeof(int32_t) != 0u || ids_size / sizeof(int32_t) > (size_t)UINT32_MAX) {
        snprintf(error, error_size, "invalid input_ids_i32.bin size %zu", ids_size);
        free(ids_bytes);
        return 0;
    }
    const uint32_t n_tokens = (uint32_t)(ids_size / sizeof(int32_t));
    if (n_tokens == 0u) {
        snprintf(error, error_size, "prompt embedding dump has no tokens");
        free(ids_bytes);
        return 0;
    }

    uint64_t expected_embedding_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_embedding_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "prompt embedding dump size overflow");
        free(ids_bytes);
        return 0;
    }

    unsigned char *embedding_bytes = NULL;
    size_t embedding_size = 0u;
    if (!read_binary_file(embeddings_path, &embedding_bytes, &embedding_size, error, error_size)) {
        free(ids_bytes);
        return 0;
    }
    if ((uint64_t)embedding_size != expected_embedding_bytes) {
        snprintf(error,
                 error_size,
                 "prompt_embeddings_f16.bin size mismatch: expected %llu bytes, got %zu",
                 (unsigned long long)expected_embedding_bytes,
                 embedding_size);
        free(embedding_bytes);
        free(ids_bytes);
        return 0;
    }

    int32_t *ids = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    uint16_t *embeddings = (uint16_t *)malloc((size_t)expected_embedding_bytes);
    if (ids == NULL || embeddings == NULL) {
        snprintf(error, error_size, "failed to allocate prompt embedding dump buffers");
        free(embeddings);
        free(ids);
        free(embedding_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < n_tokens; ++i) {
        const unsigned char *p = ids_bytes + 4u * i;
        const uint32_t raw = (uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
        memcpy(&ids[i], &raw, sizeof(ids[i]));
    }
    const size_t embedding_values = embedding_size / sizeof(uint16_t);
    for (size_t i = 0u; i < embedding_values; ++i) {
        embeddings[i] = (uint16_t)embedding_bytes[2u * i] | (uint16_t)((uint16_t)embedding_bytes[2u * i + 1u] << 8u);
    }

    free(embedding_bytes);
    free(ids_bytes);
    *input_ids_out = ids;
    *n_tokens_out = n_tokens;
    *embeddings_out = embeddings;
    return 1;
}

static int read_text_layer0_python_dump(const char *dump_dir,
                                        int32_t **input_ids_out,
                                        uint32_t *n_tokens_out,
                                        uint16_t **prompt_embeddings_out,
                                        uint16_t **layer0_hidden_out,
                                        char *error,
                                        size_t error_size) {
    if (layer0_hidden_out == NULL) {
        snprintf(error, error_size, "invalid text layer0 dump request");
        return 0;
    }
    *layer0_hidden_out = NULL;
    if (!read_prompt_embedding_python_dump(dump_dir,
                                           input_ids_out,
                                           n_tokens_out,
                                           prompt_embeddings_out,
                                           error,
                                           error_size)) {
        return 0;
    }

    uint64_t expected_layer_bytes = (uint64_t)(*n_tokens_out) * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_layer_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "text layer0 dump size overflow");
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }

    char layer_path[4096];
    if (!make_fixture_path(dump_dir, "layer_0_hidden_f16.bin", layer_path, sizeof(layer_path))) {
        snprintf(error, error_size, "text layer0 dump path is too long");
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }

    unsigned char *layer_bytes = NULL;
    size_t layer_size = 0u;
    if (!read_binary_file(layer_path, &layer_bytes, &layer_size, error, error_size)) {
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }
    if ((uint64_t)layer_size != expected_layer_bytes) {
        snprintf(error,
                 error_size,
                 "layer_0_hidden_f16.bin size mismatch: expected %llu bytes, got %zu",
                 (unsigned long long)expected_layer_bytes,
                 layer_size);
        free(layer_bytes);
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }

    uint16_t *layer0 = (uint16_t *)malloc((size_t)expected_layer_bytes);
    if (layer0 == NULL) {
        snprintf(error, error_size, "failed to allocate text layer0 dump buffer");
        free(layer_bytes);
        free(*prompt_embeddings_out);
        free(*input_ids_out);
        *prompt_embeddings_out = NULL;
        *input_ids_out = NULL;
        *n_tokens_out = 0u;
        return 0;
    }
    const size_t layer_values = layer_size / sizeof(uint16_t);
    for (size_t i = 0u; i < layer_values; ++i) {
        layer0[i] = (uint16_t)layer_bytes[2u * i] | (uint16_t)((uint16_t)layer_bytes[2u * i + 1u] << 8u);
    }
    free(layer_bytes);
    *layer0_hidden_out = layer0;
    return 1;
}

static int read_image_prompt_embedding_python_dump(const char *dump_dir,
                                                   int32_t **input_ids_out,
                                                   uint8_t **image_mask_out,
                                                   uint32_t *n_tokens_out,
                                                   uint16_t **visual_features_out,
                                                   uint32_t *image_span_start_out,
                                                   uint32_t *image_span_length_out,
                                                   uint16_t **prompt_embeddings_out,
                                                   char *error,
                                                   size_t error_size) {
    if (input_ids_out == NULL || image_mask_out == NULL || n_tokens_out == NULL || visual_features_out == NULL ||
        image_span_start_out == NULL || image_span_length_out == NULL || prompt_embeddings_out == NULL) {
        snprintf(error, error_size, "invalid image prompt embedding dump request");
        return 0;
    }
    *input_ids_out = NULL;
    *image_mask_out = NULL;
    *n_tokens_out = 0u;
    *visual_features_out = NULL;
    *image_span_start_out = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    *image_span_length_out = 0u;
    *prompt_embeddings_out = NULL;

    if (!read_prompt_embedding_python_dump(dump_dir,
                                           input_ids_out,
                                           n_tokens_out,
                                           prompt_embeddings_out,
                                           error,
                                           error_size)) {
        return 0;
    }

    char mask_path[4096];
    if (!make_fixture_path(dump_dir, "image_mask_u8.bin", mask_path, sizeof(mask_path))) {
        snprintf(error, error_size, "image prompt dump path is too long");
        goto fail;
    }
    unsigned char *mask_bytes = NULL;
    size_t mask_size = 0u;
    if (!read_binary_file(mask_path, &mask_bytes, &mask_size, error, error_size)) {
        goto fail;
    }
    if (mask_size != (size_t)(*n_tokens_out)) {
        snprintf(error,
                 error_size,
                 "image_mask_u8.bin size mismatch: expected %u bytes, got %zu",
                 *n_tokens_out,
                 mask_size);
        free(mask_bytes);
        goto fail;
    }
    uint32_t image_count = 0u;
    for (uint32_t i = 0u; i < *n_tokens_out; ++i) {
        if (mask_bytes[i] > 1u) {
            snprintf(error, error_size, "image_mask_u8.bin has invalid value %u at token %u", mask_bytes[i], i);
            free(mask_bytes);
            goto fail;
        }
        image_count += (uint32_t)mask_bytes[i];
    }
    if (image_count == 0u) {
        snprintf(error, error_size, "image prompt dump has no image placeholders");
        free(mask_bytes);
        goto fail;
    }

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = *input_ids_out;
    request.image_mask = mask_bytes;
    request.n_tokens = *n_tokens_out;
    request.max_new_tokens = 1u;
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    if (uocr_build_sequence_state(&request, &state, error, error_size) != UOCR_OK ||
        state.image_span_start == UOCR_SEQUENCE_NO_IMAGE_SPAN || state.image_span_length == 0u) {
        free(mask_bytes);
        goto fail;
    }
    if (state.image_span_length != image_count) {
        snprintf(error,
                 error_size,
                 "image placeholder count mismatch: span=%u count=%u",
                 state.image_span_length,
                 image_count);
        free(mask_bytes);
        goto fail;
    }

    if ((uint64_t)state.image_span_length > UINT64_MAX / (uint64_t)UOCR_HIDDEN_SIZE ||
        (uint64_t)state.image_span_length * (uint64_t)UOCR_HIDDEN_SIZE > UINT64_MAX / 2u) {
        snprintf(error, error_size, "image visual feature dump size overflow");
        free(mask_bytes);
        goto fail;
    }
    const uint64_t expected_visual_bytes =
        (uint64_t)state.image_span_length * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_visual_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "image visual feature dump size overflow");
        free(mask_bytes);
        goto fail;
    }

    char visual_path[4096];
    if (!make_fixture_path(dump_dir, "visual_features_f16.bin", visual_path, sizeof(visual_path))) {
        snprintf(error, error_size, "visual feature dump path is too long");
        free(mask_bytes);
        goto fail;
    }
    unsigned char *visual_bytes = NULL;
    size_t visual_size = 0u;
    if (!read_binary_file(visual_path, &visual_bytes, &visual_size, error, error_size)) {
        free(mask_bytes);
        goto fail;
    }
    if ((uint64_t)visual_size != expected_visual_bytes) {
        snprintf(error,
                 error_size,
                 "visual_features_f16.bin size mismatch: expected %llu bytes, got %zu",
                 (unsigned long long)expected_visual_bytes,
                 visual_size);
        free(visual_bytes);
        free(mask_bytes);
        goto fail;
    }
    uint16_t *visual = (uint16_t *)malloc((size_t)expected_visual_bytes);
    if (visual == NULL) {
        snprintf(error, error_size, "failed to allocate visual feature dump buffer");
        free(visual_bytes);
        free(mask_bytes);
        goto fail;
    }
    const size_t visual_values = visual_size / sizeof(uint16_t);
    for (size_t i = 0u; i < visual_values; ++i) {
        visual[i] = (uint16_t)visual_bytes[2u * i] | (uint16_t)((uint16_t)visual_bytes[2u * i + 1u] << 8u);
    }
    free(visual_bytes);

    *image_mask_out = mask_bytes;
    *visual_features_out = visual;
    *image_span_start_out = state.image_span_start;
    *image_span_length_out = state.image_span_length;
    return 1;

fail:
    free(*prompt_embeddings_out);
    free(*input_ids_out);
    *prompt_embeddings_out = NULL;
    *input_ids_out = NULL;
    *n_tokens_out = 0u;
    return 0;
}

static const uint16_t *model_tensor_f16_payload(const uocr_model_file *model,
                                                uint32_t tensor_id,
                                                uint64_t expected_values,
                                                char *error,
                                                size_t error_size) {
    if (model == NULL) {
        snprintf(error, error_size, "invalid model tensor lookup");
        return NULL;
    }
    const uocr_tensor_entry *tensor = uocr_model_file_find_tensor(model, tensor_id);
    if (tensor == NULL) {
        snprintf(error, error_size, "model tensor %u is missing", tensor_id);
        return NULL;
    }
    uint64_t expected_bytes = 0u;
    if (expected_values > UINT64_MAX / 2u) {
        snprintf(error, error_size, "model tensor %u byte-size overflow", tensor_id);
        return NULL;
    }
    expected_bytes = expected_values * 2u;
    if (tensor->qtype != UOCR_TENSOR_F16 || tensor->payload_size != expected_bytes ||
        tensor->payload_offset > (uint64_t)model->size || tensor->payload_size > (uint64_t)model->size - tensor->payload_offset) {
        snprintf(error,
                 error_size,
                 "model tensor %u is not the expected fp16 payload: qtype=%u bytes=%llu expected=%llu",
                 tensor_id,
                 tensor->qtype,
                 (unsigned long long)tensor->payload_size,
                 (unsigned long long)expected_bytes);
        return NULL;
    }
    return (const uint16_t *)(const void *)(model->data + tensor->payload_offset);
}

static int read_layer_hidden_python_dump(const char *dump_dir,
                                         const char *filename,
                                         uint32_t n_tokens,
                                         uint16_t **hidden_out,
                                         char *error,
                                         size_t error_size) {
    if (hidden_out == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid layer hidden dump request");
        return 0;
    }
    *hidden_out = NULL;
    uint64_t expected_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (expected_bytes > (uint64_t)SIZE_MAX) {
        snprintf(error, error_size, "layer hidden dump size overflow");
        return 0;
    }

    char path[4096];
    if (!make_fixture_path(dump_dir, filename, path, sizeof(path))) {
        snprintf(error, error_size, "layer hidden dump path is too long");
        return 0;
    }

    unsigned char *bytes = NULL;
    size_t size = 0u;
    if (!read_binary_file(path, &bytes, &size, error, error_size)) {
        return 0;
    }
    if ((uint64_t)size != expected_bytes) {
        snprintf(error,
                 error_size,
                 "%s size mismatch: expected %llu bytes, got %zu",
                 filename,
                 (unsigned long long)expected_bytes,
                 size);
        free(bytes);
        return 0;
    }

    uint16_t *hidden = (uint16_t *)malloc((size_t)expected_bytes);
    if (hidden == NULL) {
        snprintf(error, error_size, "failed to allocate layer hidden dump buffer");
        free(bytes);
        return 0;
    }
    const size_t values = size / sizeof(uint16_t);
    for (size_t i = 0u; i < values; ++i) {
        hidden[i] = (uint16_t)bytes[2u * i] | (uint16_t)((uint16_t)bytes[2u * i + 1u] << 8u);
    }
    free(bytes);
    *hidden_out = hidden;
    return 1;
}

static int run_metal_attention_block_from_model_f16(uocr_metal_context *ctx,
                                                    const uocr_model_file *model,
                                                    uint32_t layer,
                                                    const uint16_t *hidden_in,
                                                    uint32_t n_tokens,
                                                    uint16_t *attn_hidden_out,
                                                    uint16_t *mlp_input_out,
                                                    char *error,
                                                    size_t error_size) {
    if (ctx == NULL || model == NULL || hidden_in == NULL || attn_hidden_out == NULL || mlp_input_out == NULL ||
        n_tokens == 0u || layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid Metal attention-block parity request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal attention-block parity size overflow");
        return 0;
    }

    const uint16_t *input_norm_weight = model_tensor_f16_payload(model,
                                                                 uocr_tensor_id_layer_input_norm(layer),
                                                                 UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 error_size);
    const uint16_t *q_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_Q),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *k_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_K),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *v_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_V),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *o_weight = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_O),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        error_size);
    const uint16_t *post_norm_weight = model_tensor_f16_payload(model,
                                                                uocr_tensor_id_layer_post_attn_norm(layer),
                                                                UOCR_HIDDEN_SIZE,
                                                                error,
                                                                error_size);
    if (input_norm_weight == NULL || q_weight == NULL || k_weight == NULL || v_weight == NULL || o_weight == NULL ||
        post_norm_weight == NULL) {
        return 0;
    }

    uint16_t *normed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *v = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *unused_o = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *context = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    if (normed == NULL || q == NULL || k == NULL || v == NULL || unused_o == NULL || q_rope == NULL ||
        k_rope == NULL || context == NULL) {
        snprintf(error, error_size, "failed to allocate Metal attention-block parity buffers");
        free(context);
        free(k_rope);
        free(q_rope);
        free(unused_o);
        free(v);
        free(k);
        free(q);
        free(normed);
        return 0;
    }

    int ok = 0;
    if (uocr_metal_context_rmsnorm_f16(ctx,
                                       hidden_in,
                                       input_norm_weight,
                                       n_tokens,
                                       UOCR_HIDDEN_SIZE,
                                       UOCR_RMS_NORM_EPS,
                                       UOCR_METAL_RMSNORM_OUTPUT_F16,
                                       normed,
                                       error,
                                       error_size) != 1 ||
        uocr_metal_context_attention_qkvo_f16(ctx,
                                              normed,
                                              q_weight,
                                              k_weight,
                                              v_weight,
                                              o_weight,
                                              n_tokens,
                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                              q,
                                              k,
                                              v,
                                              unused_o,
                                              error,
                                              error_size) != 1 ||
        uocr_metal_context_rope_qk_f16(ctx,
                                       q,
                                       k,
                                       n_tokens,
                                       0u,
                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                       q_rope,
                                       k_rope,
                                       error,
                                       error_size) != 1 ||
        uocr_metal_context_prefill_attention_f16(ctx,
                                                 q_rope,
                                                 k_rope,
                                                 v,
                                                 n_tokens,
                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                 context,
                                                 error,
                                                 error_size) != 1 ||
        uocr_metal_context_attention_output_residual_f16(ctx,
                                                         context,
                                                         o_weight,
                                                         hidden_in,
                                                         n_tokens,
                                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                                         attn_hidden_out,
                                                         error,
                                                         error_size) != 1 ||
        uocr_metal_context_rmsnorm_f16(ctx,
                                       attn_hidden_out,
                                       post_norm_weight,
                                       n_tokens,
                                       UOCR_HIDDEN_SIZE,
                                       UOCR_RMS_NORM_EPS,
                                       UOCR_METAL_RMSNORM_OUTPUT_F16,
                                       mlp_input_out,
                                       error,
                                       error_size) != 1) {
        ok = 0;
    } else {
        ok = 1;
    }

    free(context);
    free(k_rope);
    free(q_rope);
    free(unused_o);
    free(v);
    free(k);
    free(q);
    free(normed);
    return ok;
}

static int run_metal_dense_decoder_layer0_from_model_f16(uocr_metal_context *ctx,
                                                           const uocr_model_file *model,
                                                           const uint16_t *hidden_in,
                                                           uint32_t n_tokens,
                                                           uint16_t *hidden_out,
                                                           char *error,
                                                           size_t error_size) {
    if (ctx == NULL || model == NULL || hidden_in == NULL || hidden_out == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid Metal dense decoder-layer0 request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal dense decoder-layer0 size overflow");
        return 0;
    }

    const uint16_t *gate_weight = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           error_size);
    const uint16_t *up_weight = model_tensor_f16_payload(model,
                                                         uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         error_size);
    const uint16_t *down_weight = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE,
                                                           error,
                                                           error_size);
    if (gate_weight == NULL || up_weight == NULL || down_weight == NULL) {
        return 0;
    }

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    if (attn_hidden == NULL || mlp_input == NULL) {
        snprintf(error, error_size, "failed to allocate Metal dense decoder-layer0 buffers");
        free(mlp_input);
        free(attn_hidden);
        return 0;
    }

    int ok = 0;
    if (run_metal_attention_block_from_model_f16(ctx,
                                                model,
                                                0u,
                                                hidden_in,
                                                n_tokens,
                                                attn_hidden,
                                                mlp_input,
                                                error,
                                                error_size) != 1 ||
        uocr_metal_context_dense_swiglu_f16(ctx,
                                            mlp_input,
                                            gate_weight,
                                            up_weight,
                                            down_weight,
                                            attn_hidden,
                                            n_tokens,
                                            UOCR_METAL_DENSE_OUTPUT_F16,
                                            hidden_out,
                                            error,
                                            error_size) != 1) {
        ok = 0;
    } else {
        ok = 1;
    }

    free(mlp_input);
    free(attn_hidden);
    return ok;
}

static int copy_selected_moe_expert_weights(const uocr_model_file *model,
                                            uint32_t layer,
                                            const uint32_t *top_expert_ids,
                                            uint16_t *selected_gate,
                                            uint16_t *selected_up,
                                            uint16_t *selected_down,
                                            char *error,
                                            size_t error_size) {
    if (model == NULL || top_expert_ids == NULL || selected_gate == NULL || selected_up == NULL || selected_down == NULL ||
        layer == 0u || layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid selected MoE expert copy request");
        return 0;
    }
    const uint64_t expert_values = (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * (uint64_t)UOCR_HIDDEN_SIZE;
    if (expert_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "selected MoE expert copy size overflow");
        return 0;
    }
    const size_t expert_bytes = (size_t)expert_values * sizeof(uint16_t);
    for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
        const uint32_t expert = top_expert_ids[rank];
        if (expert >= UOCR_ROUTED_EXPERTS) {
            snprintf(error, error_size, "selected MoE expert id %u out of range", expert);
            return 0;
        }
        const uint16_t *gate = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_GATE),
                                                        expert_values,
                                                        error,
                                                        error_size);
        const uint16_t *up = model_tensor_f16_payload(model,
                                                      uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_UP),
                                                      expert_values,
                                                      error,
                                                      error_size);
        const uint16_t *down = model_tensor_f16_payload(model,
                                                        uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_DOWN),
                                                        expert_values,
                                                        error,
                                                        error_size);
        if (gate == NULL || up == NULL || down == NULL) {
            return 0;
        }
        memcpy(selected_gate + (size_t)rank * (size_t)expert_values, gate, expert_bytes);
        memcpy(selected_up + (size_t)rank * (size_t)expert_values, up, expert_bytes);
        memcpy(selected_down + (size_t)rank * (size_t)expert_values, down, expert_bytes);
    }
    return 1;
}

static int read_text_logits_topk_python_dump(const char *dump_dir,
                                              int32_t **ids_out,
                                              float **scores_out,
                                              uint32_t *top_k_out,
                                              char *error,
                                              size_t error_size) {
    if (ids_out == NULL || scores_out == NULL || top_k_out == NULL) {
        snprintf(error, error_size, "invalid logits top-k dump request");
        return 0;
    }
    *ids_out = NULL;
    *scores_out = NULL;
    *top_k_out = 0u;

    char ids_path[4096];
    char scores_path[4096];
    if (!make_fixture_path(dump_dir, "logits_topk_ids_i32.bin", ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, "logits_topk_scores_f32.bin", scores_path, sizeof(scores_path))) {
        snprintf(error, error_size, "logits top-k dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    unsigned char *scores_bytes = NULL;
    size_t ids_size = 0u;
    size_t scores_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size) ||
        !read_binary_file(scores_path, &scores_bytes, &scores_size, error, error_size)) {
        free(scores_bytes);
        free(ids_bytes);
        return 0;
    }
    if (ids_size == 0u || ids_size % sizeof(int32_t) != 0u || scores_size != ids_size) {
        snprintf(error,
                 error_size,
                 "logits top-k size mismatch: ids=%zu bytes scores=%zu bytes",
                 ids_size,
                 scores_size);
        free(scores_bytes);
        free(ids_bytes);
        return 0;
    }
    const uint32_t top_k = (uint32_t)(ids_size / sizeof(int32_t));
    int32_t *ids = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    float *scores = (float *)malloc((size_t)top_k * sizeof(float));
    if (ids == NULL || scores == NULL) {
        snprintf(error, error_size, "failed to allocate logits top-k buffers");
        free(scores);
        free(ids);
        free(scores_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < top_k; ++i) {
        const uint32_t id_bits = (uint32_t)ids_bytes[4u * i] | ((uint32_t)ids_bytes[4u * i + 1u] << 8u) |
                                 ((uint32_t)ids_bytes[4u * i + 2u] << 16u) |
                                 ((uint32_t)ids_bytes[4u * i + 3u] << 24u);
        const uint32_t score_bits = (uint32_t)scores_bytes[4u * i] | ((uint32_t)scores_bytes[4u * i + 1u] << 8u) |
                                    ((uint32_t)scores_bytes[4u * i + 2u] << 16u) |
                                    ((uint32_t)scores_bytes[4u * i + 3u] << 24u);
        ids[i] = (int32_t)id_bits;
        memcpy(scores + i, &score_bits, sizeof(float));
        if (ids[i] < 0 || ids[i] >= (int32_t)UOCR_VOCAB_SIZE || !isfinite(scores[i])) {
            snprintf(error, error_size, "invalid logits top-k entry at rank %u", i);
            free(scores);
            free(ids);
            free(scores_bytes);
            free(ids_bytes);
            return 0;
        }
    }
    free(scores_bytes);
    free(ids_bytes);
    *ids_out = ids;
    *scores_out = scores;
    *top_k_out = top_k;
    return 1;
}

static int read_text_generated_ids_python_dump(const char *dump_dir,
                                               int32_t **ids_out,
                                               uint32_t *n_ids_out,
                                               char *error,
                                               size_t error_size) {
    if (ids_out == NULL || n_ids_out == NULL) {
        snprintf(error, error_size, "invalid generated-id dump request");
        return 0;
    }
    *ids_out = NULL;
    *n_ids_out = 0u;

    char path[4096];
    if (!make_fixture_path(dump_dir, "generated_ids_i32.bin", path, sizeof(path))) {
        snprintf(error, error_size, "generated-id dump path is too long");
        return 0;
    }
    unsigned char *bytes = NULL;
    size_t size = 0u;
    if (!read_binary_file(path, &bytes, &size, error, error_size)) {
        return 0;
    }
    if (size == 0u || size % sizeof(int32_t) != 0u) {
        snprintf(error, error_size, "generated-id dump size mismatch: %zu bytes", size);
        free(bytes);
        return 0;
    }
    const uint32_t n_ids = (uint32_t)(size / sizeof(int32_t));
    int32_t *ids = (int32_t *)malloc((size_t)n_ids * sizeof(int32_t));
    if (ids == NULL) {
        snprintf(error, error_size, "failed to allocate generated-id dump buffer");
        free(bytes);
        return 0;
    }
    for (uint32_t i = 0u; i < n_ids; ++i) {
        const uint32_t id_bits = (uint32_t)bytes[4u * i] | ((uint32_t)bytes[4u * i + 1u] << 8u) |
                                 ((uint32_t)bytes[4u * i + 2u] << 16u) |
                                 ((uint32_t)bytes[4u * i + 3u] << 24u);
        ids[i] = (int32_t)id_bits;
        if (ids[i] < 0 || ids[i] >= (int32_t)UOCR_VOCAB_SIZE) {
            snprintf(error, error_size, "invalid generated token id at index %u", i);
            free(ids);
            free(bytes);
            return 0;
        }
    }
    free(bytes);
    *ids_out = ids;
    *n_ids_out = n_ids;
    return 1;
}

static int read_router_topk_python_dump(const char *dump_dir,
                                         uint32_t layer,
                                         uint32_t n_tokens,
                                         uint32_t **ids_out,
                                         float **weights_out,
                                         char *error,
                                         size_t error_size) {
    if (ids_out == NULL || weights_out == NULL || n_tokens == 0u || layer == 0u || layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid router top-k dump request");
        return 0;
    }
    *ids_out = NULL;
    *weights_out = NULL;

    char ids_name[64];
    char weights_name[64];
    snprintf(ids_name, sizeof(ids_name), "layer_%u_router_top_ids_u32.bin", layer);
    snprintf(weights_name, sizeof(weights_name), "layer_%u_router_top_weights_f32.bin", layer);
    char ids_path[4096];
    char weights_path[4096];
    if (!make_fixture_path(dump_dir, ids_name, ids_path, sizeof(ids_path)) ||
        !make_fixture_path(dump_dir, weights_name, weights_path, sizeof(weights_path))) {
        snprintf(error, error_size, "router top-k dump path is too long");
        return 0;
    }

    unsigned char *ids_bytes = NULL;
    unsigned char *weights_bytes = NULL;
    size_t ids_size = 0u;
    size_t weights_size = 0u;
    if (!read_binary_file(ids_path, &ids_bytes, &ids_size, error, error_size) ||
        !read_binary_file(weights_path, &weights_bytes, &weights_size, error, error_size)) {
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }

    uint64_t expected_values = 0u;
    if ((uint64_t)n_tokens > UINT64_MAX / (uint64_t)UOCR_MOE_TOP_K ||
        (expected_values = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K) > (uint64_t)SIZE_MAX / sizeof(uint32_t)) {
        snprintf(error, error_size, "router top-k dump size overflow");
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }
    const uint64_t expected_bytes = expected_values * sizeof(uint32_t);
    if ((uint64_t)ids_size != expected_bytes || (uint64_t)weights_size != expected_bytes) {
        snprintf(error,
                 error_size,
                 "router top-k dump size mismatch for layer %u: ids=%zu weights=%zu expected=%llu",
                 layer,
                 ids_size,
                 weights_size,
                 (unsigned long long)expected_bytes);
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }

    uint32_t *ids = (uint32_t *)malloc((size_t)expected_values * sizeof(uint32_t));
    float *weights = (float *)malloc((size_t)expected_values * sizeof(float));
    if (ids == NULL || weights == NULL) {
        snprintf(error, error_size, "failed to allocate router top-k dump buffers");
        free(weights);
        free(ids);
        free(weights_bytes);
        free(ids_bytes);
        return 0;
    }
    for (uint64_t i = 0u; i < expected_values; ++i) {
        const uint32_t id_bits = (uint32_t)ids_bytes[4u * i] | ((uint32_t)ids_bytes[4u * i + 1u] << 8u) |
                                 ((uint32_t)ids_bytes[4u * i + 2u] << 16u) |
                                 ((uint32_t)ids_bytes[4u * i + 3u] << 24u);
        const uint32_t weight_bits = (uint32_t)weights_bytes[4u * i] | ((uint32_t)weights_bytes[4u * i + 1u] << 8u) |
                                     ((uint32_t)weights_bytes[4u * i + 2u] << 16u) |
                                     ((uint32_t)weights_bytes[4u * i + 3u] << 24u);
        ids[i] = id_bits;
        memcpy(weights + i, &weight_bits, sizeof(float));
        if (ids[i] >= UOCR_ROUTED_EXPERTS || !isfinite(weights[i])) {
            snprintf(error, error_size, "invalid router top-k entry at flat index %llu", (unsigned long long)i);
            free(weights);
            free(ids);
            free(weights_bytes);
            free(ids_bytes);
            return 0;
        }
    }
    free(weights_bytes);
    free(ids_bytes);
    *ids_out = ids;
    *weights_out = weights;
    return 1;
}

static void compute_logits_topk_f32(const float *logits,
                                    uint32_t vocab_size,
                                    uint32_t top_k,
                                    int32_t *ids_out,
                                    float *scores_out) {
    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        uint32_t best_id = 0u;
        float best_score = -INFINITY;
        for (uint32_t token = 0u; token < vocab_size; ++token) {
            int already_selected = 0;
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if ((uint32_t)ids_out[prev] == token) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected || isnan(logits[token])) {
                continue;
            }
            if (logits[token] > best_score || (logits[token] == best_score && token < best_id)) {
                best_score = logits[token];
                best_id = token;
            }
        }
        ids_out[rank] = (int32_t)best_id;
        scores_out[rank] = best_score;
    }
}

static int compare_hidden_f16(const char *label,
                              const uint16_t *actual,
                              const uint16_t *expected,
                              uint32_t n_tokens,
                              float max_abs_tolerance,
                              double mean_abs_tolerance) {
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    uint64_t max_index = 0u;
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        const float diff = fabsf(f16_bits_to_f32(actual[i]) - f16_bits_to_f32(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            max_index = i;
        }
        sum_abs += (double)diff;
    }
    const double mean_abs = sum_abs / (double)hidden_values;
    if (max_abs > max_abs_tolerance || mean_abs > mean_abs_tolerance) {
        fprintf(stderr,
                "%s mismatch: max_abs=%g mean_abs=%g at token %llu col %llu actual=%g expected=%g\n",
                label,
                (double)max_abs,
                mean_abs,
                (unsigned long long)(max_index / (uint64_t)UOCR_HIDDEN_SIZE),
                (unsigned long long)(max_index % (uint64_t)UOCR_HIDDEN_SIZE),
                (double)f16_bits_to_f32(actual[max_index]),
                (double)f16_bits_to_f32(expected[max_index]));
        return 0;
    }
    return 1;
}

static int copy_compact_moe_expert_weights(const uocr_model_file *model,
                                           uint32_t layer,
                                           const uint32_t *top_expert_ids,
                                           uint32_t n_tokens,
                                           uint32_t *remapped_top_expert_ids,
                                           uint32_t *expert_count_out,
                                           uint16_t **gate_out,
                                           uint16_t **up_out,
                                           uint16_t **down_out,
                                           char *error,
                                           size_t error_size) {
    if (model == NULL || top_expert_ids == NULL || remapped_top_expert_ids == NULL || expert_count_out == NULL ||
        gate_out == NULL || up_out == NULL || down_out == NULL || n_tokens == 0u || layer == 0u ||
        layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid compact MoE expert copy request");
        return 0;
    }
    *expert_count_out = 0u;
    *gate_out = NULL;
    *up_out = NULL;
    *down_out = NULL;

    uint32_t experts[UOCR_ROUTED_EXPERTS];
    uint32_t expert_count = 0u;
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
            const size_t index = (size_t)token * UOCR_MOE_TOP_K + rank;
            const uint32_t expert = top_expert_ids[index];
            if (expert >= UOCR_ROUTED_EXPERTS) {
                snprintf(error, error_size, "compact MoE expert id %u out of range", expert);
                return 0;
            }
            uint32_t remapped = UINT32_MAX;
            for (uint32_t i = 0u; i < expert_count; ++i) {
                if (experts[i] == expert) {
                    remapped = i;
                    break;
                }
            }
            if (remapped == UINT32_MAX) {
                if (expert_count >= UOCR_ROUTED_EXPERTS) {
                    snprintf(error, error_size, "compact MoE expert table overflow");
                    return 0;
                }
                remapped = expert_count;
                experts[expert_count++] = expert;
            }
            remapped_top_expert_ids[index] = remapped;
        }
    }

    const uint64_t expert_values = (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * (uint64_t)UOCR_HIDDEN_SIZE;
    uint64_t total_values = 0u;
    if (expert_count == 0u || expert_values > UINT64_MAX / (uint64_t)expert_count ||
        (total_values = expert_values * (uint64_t)expert_count) > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "compact MoE expert copy size overflow");
        return 0;
    }
    const size_t expert_bytes = (size_t)expert_values * sizeof(uint16_t);
    const size_t total_bytes = (size_t)total_values * sizeof(uint16_t);
    uint16_t *gate = (uint16_t *)malloc(total_bytes);
    uint16_t *up = (uint16_t *)malloc(total_bytes);
    uint16_t *down = (uint16_t *)malloc(total_bytes);
    if (gate == NULL || up == NULL || down == NULL) {
        snprintf(error, error_size, "failed to allocate compact MoE expert weights");
        free(down);
        free(up);
        free(gate);
        return 0;
    }

    for (uint32_t compact = 0u; compact < expert_count; ++compact) {
        const uint32_t expert = experts[compact];
        const uint16_t *src_gate = model_tensor_f16_payload(model,
                                                            uocr_tensor_id_moe_expert(layer,
                                                                                      expert,
                                                                                      UOCR_TENSOR_PROJ_GATE),
                                                            expert_values,
                                                            error,
                                                            error_size);
        const uint16_t *src_up = model_tensor_f16_payload(model,
                                                          uocr_tensor_id_moe_expert(layer,
                                                                                    expert,
                                                                                    UOCR_TENSOR_PROJ_UP),
                                                          expert_values,
                                                          error,
                                                          error_size);
        const uint16_t *src_down = model_tensor_f16_payload(model,
                                                            uocr_tensor_id_moe_expert(layer,
                                                                                      expert,
                                                                                      UOCR_TENSOR_PROJ_DOWN),
                                                            expert_values,
                                                            error,
                                                            error_size);
        if (src_gate == NULL || src_up == NULL || src_down == NULL) {
            free(down);
            free(up);
            free(gate);
            return 0;
        }
        memcpy(gate + (size_t)compact * (size_t)expert_values, src_gate, expert_bytes);
        memcpy(up + (size_t)compact * (size_t)expert_values, src_up, expert_bytes);
        memcpy(down + (size_t)compact * (size_t)expert_values, src_down, expert_bytes);
    }

    *expert_count_out = expert_count;
    *gate_out = gate;
    *up_out = up;
    *down_out = down;
    return 1;
}

static int run_metal_moe_decoder_layer_from_model_f16(uocr_metal_context *ctx,
                                                      const uocr_model_file *model,
                                                      uint32_t layer,
                                                      const uint16_t *hidden_in,
                                                      uint32_t n_tokens,
                                                      uint16_t *hidden_out,
                                                      char *error,
                                                      size_t error_size) {
    if (ctx == NULL || model == NULL || hidden_in == NULL || hidden_out == NULL || n_tokens == 0u || layer == 0u ||
        layer >= UOCR_DECODER_LAYERS) {
        snprintf(error, error_size, "invalid Metal MoE decoder-layer parity request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal MoE decoder-layer parity size overflow");
        return 0;
    }

    const uint16_t *router_weight = model_tensor_f16_payload(model,
                                                             uocr_tensor_id_moe_router(layer),
                                                             (uint64_t)UOCR_ROUTED_EXPERTS * (uint64_t)UOCR_HIDDEN_SIZE,
                                                             error,
                                                             error_size);
    const uint16_t *shared_gate = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           error_size);
    const uint16_t *shared_up = model_tensor_f16_payload(model,
                                                         uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         error_size);
    const uint16_t *shared_down = model_tensor_f16_payload(model,
                                                           uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE,
                                                           error,
                                                           error_size);
    if (router_weight == NULL || shared_gate == NULL || shared_up == NULL || shared_down == NULL) {
        return 0;
    }

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *routed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *shared = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint32_t *top_ids = (uint32_t *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(uint32_t));
    float *top_weights = (float *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(float));
    if (attn_hidden == NULL || mlp_input == NULL || routed == NULL || shared == NULL || top_ids == NULL ||
        top_weights == NULL) {
        snprintf(error, error_size, "failed to allocate Metal MoE decoder-layer parity buffers");
        free(top_weights);
        free(top_ids);
        free(shared);
        free(routed);
        free(mlp_input);
        free(attn_hidden);
        return 0;
    }

    int ok = 0;
    if (run_metal_attention_block_from_model_f16(ctx,
                                                model,
                                                layer,
                                                hidden_in,
                                                n_tokens,
                                                attn_hidden,
                                                mlp_input,
                                                error,
                                                error_size) != 1 ||
        uocr_metal_context_moe_router_f16(ctx,
                                          mlp_input,
                                          router_weight,
                                          n_tokens,
                                          NULL,
                                          NULL,
                                          top_ids,
                                          top_weights,
                                          error,
                                          error_size) != 1 ||
        uocr_metal_context_moe_shared_experts_f16(ctx,
                                                  mlp_input,
                                                  shared_gate,
                                                  shared_up,
                                                  shared_down,
                                                  n_tokens,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  shared,
                                                  error,
                                                  error_size) != 1) {
        goto cleanup;
    }

    uint32_t unique_experts[UOCR_ROUTED_EXPERTS];
    uint32_t unique_count = 0u;
    for (uint32_t i = 0u; i < n_tokens * UOCR_MOE_TOP_K; ++i) {
        uint32_t seen = 0u;
        for (uint32_t j = 0u; j < unique_count; ++j) {
            if (unique_experts[j] == top_ids[i]) {
                seen = 1u;
                break;
            }
        }
        if (!seen && unique_count < UOCR_ROUTED_EXPERTS) {
            unique_experts[unique_count++] = top_ids[i];
        }
    }
    const uint64_t compact_weight_bytes = (uint64_t)unique_count * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE *
                                          (uint64_t)UOCR_HIDDEN_SIZE * 2u * 3u;
    if (compact_weight_bytes <= 192ull * 1024ull * 1024ull) {
        uint32_t *remapped_ids = (uint32_t *)malloc((size_t)n_tokens * UOCR_MOE_TOP_K * sizeof(uint32_t));
        uint16_t *compact_gate = NULL;
        uint16_t *compact_up = NULL;
        uint16_t *compact_down = NULL;
        uint32_t compact_count = 0u;
        if (remapped_ids == NULL) {
            snprintf(error, error_size, "failed to allocate compact MoE remap buffer");
            goto cleanup;
        }
        if (copy_compact_moe_expert_weights(model,
                                            layer,
                                            top_ids,
                                            n_tokens,
                                            remapped_ids,
                                            &compact_count,
                                            &compact_gate,
                                            &compact_up,
                                            &compact_down,
                                            error,
                                            error_size) != 1 ||
            uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                                mlp_input,
                                                                remapped_ids,
                                                                top_weights,
                                                                compact_gate,
                                                                compact_up,
                                                                compact_down,
                                                                n_tokens,
                                                                UOCR_HIDDEN_SIZE,
                                                                UOCR_MOE_EXPERT_INTERMEDIATE,
                                                                compact_count,
                                                                UOCR_MOE_TOP_K,
                                                                UOCR_METAL_DENSE_OUTPUT_F16,
                                                                routed,
                                                                error,
                                                                error_size) != 1) {
            free(compact_down);
            free(compact_up);
            free(compact_gate);
            free(remapped_ids);
            goto cleanup;
        }
        free(compact_down);
        free(compact_up);
        free(compact_gate);
        free(remapped_ids);
    } else {
        const uint64_t expert_values = (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE *
                                       (uint64_t)UOCR_HIDDEN_SIZE;
        if (expert_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
            snprintf(error, error_size, "selected MoE fallback size overflow");
            goto cleanup;
        }
        uint16_t *selected_gate = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
        uint16_t *selected_up = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
        uint16_t *selected_down = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
        if (selected_gate == NULL || selected_up == NULL || selected_down == NULL) {
            snprintf(error, error_size, "failed to allocate selected MoE fallback weights");
            free(selected_down);
            free(selected_up);
            free(selected_gate);
            goto cleanup;
        }
        for (uint32_t token = 0u; token < n_tokens; ++token) {
            if (copy_selected_moe_expert_weights(model,
                                                 layer,
                                                 top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                                 selected_gate,
                                                 selected_up,
                                                 selected_down,
                                                 error,
                                                 error_size) != 1 ||
                uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                                   mlp_input + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                   top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                                                   top_weights + (size_t)token * UOCR_MOE_TOP_K,
                                                                   selected_gate,
                                                                   selected_up,
                                                                   selected_down,
                                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                                   routed + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                   error,
                                                                   error_size) != 1) {
                free(selected_down);
                free(selected_up);
                free(selected_gate);
                goto cleanup;
            }
        }
        free(selected_down);
        free(selected_up);
        free(selected_gate);
    }

    if (uocr_metal_context_moe_combine_f16(ctx,
                                           routed,
                                           shared,
                                           attn_hidden,
                                           n_tokens,
                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                           hidden_out,
                                           error,
                                           error_size) != 1) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(top_weights);
    free(top_ids);
    free(shared);
    free(routed);
    free(mlp_input);
    free(attn_hidden);
    return ok;
}

static int run_metal_decoder_prefill_from_model_f16(uocr_metal_context *ctx,
                                                     const uocr_model_file *model,
                                                     const uint16_t *prompt_embeddings_f16,
                                                     uint32_t n_tokens,
                                                     uint16_t *final_hidden_out,
                                                     char *error,
                                                     size_t error_size) {
    if (ctx == NULL || model == NULL || prompt_embeddings_f16 == NULL || final_hidden_out == NULL || n_tokens == 0u) {
        snprintf(error, error_size, "invalid Metal decoder prefill request");
        return 0;
    }
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t)) {
        snprintf(error, error_size, "Metal decoder prefill size overflow");
        return 0;
    }
    const size_t hidden_bytes = (size_t)hidden_values * sizeof(uint16_t);
    uint16_t *ping = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *pong = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    if (ping == NULL || pong == NULL) {
        snprintf(error, error_size, "failed to allocate Metal decoder prefill ping-pong buffers");
        free(pong);
        free(ping);
        return 0;
    }

    int ok = 0;
    if (run_metal_dense_decoder_layer0_from_model_f16(ctx,
                                                      model,
                                                      prompt_embeddings_f16,
                                                      n_tokens,
                                                      ping,
                                                      error,
                                                      error_size) != 1) {
        goto cleanup;
    }

    uint16_t *current = ping;
    uint16_t *next = pong;
    for (uint32_t layer = 1u; layer < UOCR_DECODER_LAYERS; ++layer) {
        if (run_metal_moe_decoder_layer_from_model_f16(ctx,
                                                       model,
                                                       layer,
                                                       current,
                                                       n_tokens,
                                                       next,
                                                       error,
                                                       error_size) != 1) {
            goto cleanup;
        }
        uint16_t *tmp = current;
        current = next;
        next = tmp;
    }
    memcpy(final_hidden_out, current, hidden_bytes);
    ok = 1;

cleanup:
    free(pong);
    free(ping);
    return ok;
}

static int test_metal_text_prompt_embedding_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_PROMPT_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_PROMPT_DUMP_DIR are not both set; skipping Python prompt dump parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *expected = NULL;
    CHECK(read_prompt_embedding_python_dump(dump_dir, &input_ids, &n_tokens, &expected, error, sizeof(error)) == 1);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint64_t output_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    CHECK(output_bytes <= (uint64_t)SIZE_MAX);
    uint16_t *actual = (uint16_t *)malloc((size_t)output_bytes);
    CHECK(actual != NULL);
    memset(actual, 0, (size_t)output_bytes);
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            actual,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    const uint64_t values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    for (uint64_t i = 0u; i < values; ++i) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "Python prompt dump mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual[i],
                    expected[i]);
            free(actual);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected);
            free(input_ids);
            return 1;
        }
    }

    free(actual);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(input_ids);
    return 0;
}

static int test_metal_image_prompt_embedding_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image prompt embedding parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no visual_features_f16.bin; skipping image prompt embedding parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    const uint64_t output_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    CHECK(output_bytes <= (uint64_t)SIZE_MAX);
    uint16_t *actual = (uint16_t *)malloc((size_t)output_bytes);
    CHECK(actual != NULL);
    memset(actual, 0, (size_t)output_bytes);

    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            image_span_start,
                                                            image_span_length,
                                                            visual_features,
                                                            actual,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    const uint64_t values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    for (uint64_t i = 0u; i < values; ++i) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "Python image prompt dump mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual[i],
                    expected[i]);
            free(actual);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    free(actual);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_prompt_decoder_smoke_python_dump(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image prompt decoder smoke\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no visual_features_f16.bin; skipping image prompt decoder smoke\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected_prompt = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected_prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image decoder smoke\n",
               n_tokens,
               n_tokens);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));
    uint16_t *prompt = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *final_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(prompt != NULL);
    CHECK(final_hidden != NULL);
    CHECK(normed != NULL);
    CHECK(logits != NULL);

    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            image_span_start,
                                                            image_span_length,
                                                            visual_features,
                                                            prompt,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(memcmp(prompt, expected_prompt, (size_t)hidden_values * sizeof(uint16_t)) == 0);

    CHECK(run_metal_decoder_prefill_from_model_f16(ctx,
                                                   &model,
                                                   prompt,
                                                   n_tokens,
                                                   final_hidden,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    uint32_t token_id = UINT32_MAX;
    float score = 0.0f;
    const uint16_t *last_hidden = final_hidden + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   last_hidden,
                                                   1u,
                                                   NULL,
                                                   normed,
                                                   logits,
                                                   &token_id,
                                                   &score,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(token_id < (uint32_t)UOCR_VOCAB_SIZE);
    CHECK(isfinite(score));

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = n_tokens;
    request.max_new_tokens = 1u;
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    int32_t generated[1] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)token_id, generated, 1u) == UOCR_OK);
    CHECK(generated[0] == (int32_t)token_id);
    CHECK(state.generated_count == 1u);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    free(logits);
    free(normed);
    free(final_hidden);
    free(prompt);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_decoder_layers_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image decoder layer parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no visual_features_f16.bin; skipping image decoder layer parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no layer_11_hidden_f16.bin; skipping image decoder layer parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected_prompt = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected_prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image decoder layer parity\n",
               n_tokens,
               n_tokens);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));
    const size_t hidden_bytes = (size_t)hidden_values * sizeof(uint16_t);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *actual_prompt = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(actual_prompt != NULL);
    CHECK(actual != NULL);

    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            input_ids,
                                                            n_tokens,
                                                            image_span_start,
                                                            image_span_length,
                                                            visual_features,
                                                            actual_prompt,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        if (actual_prompt[i] != expected_prompt[i]) {
            fprintf(stderr,
                    "image decoder prompt mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual_prompt[i],
                    expected_prompt[i]);
            free(actual);
            free(actual_prompt);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    uint16_t *expected = NULL;
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_0_hidden_f16.bin",
                                        n_tokens,
                                        &expected,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(run_metal_dense_decoder_layer0_from_model_f16(ctx,
                                                        &model,
                                                        actual_prompt,
                                                        n_tokens,
                                                        actual,
                                                        error,
                                                        sizeof(error)) == 1);
    if (!compare_hidden_f16("image layer0", actual, expected, n_tokens, 7.5e-3f, 7.5e-4)) {
        free(expected);
        free(actual);
        free(actual_prompt);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 1;
    }
    free(expected);

    for (uint32_t layer = 1u; layer < UOCR_DECODER_LAYERS; ++layer) {
        char prev_filename[64];
        char expected_filename[64];
        char label[64];
        snprintf(prev_filename, sizeof(prev_filename), "layer_%u_hidden_f16.bin", layer - 1u);
        snprintf(expected_filename, sizeof(expected_filename), "layer_%u_hidden_f16.bin", layer);
        snprintf(label, sizeof(label), "image layer%u", layer);
        uint16_t *prev_expected = NULL;
        expected = NULL;
        CHECK(read_layer_hidden_python_dump(dump_dir, prev_filename, n_tokens, &prev_expected, error, sizeof(error)) == 1);
        CHECK(read_layer_hidden_python_dump(dump_dir, expected_filename, n_tokens, &expected, error, sizeof(error)) == 1);
        memset(actual, 0, hidden_bytes);
        CHECK(run_metal_moe_decoder_layer_from_model_f16(ctx,
                                                         &model,
                                                         layer,
                                                         prev_expected,
                                                         n_tokens,
                                                         actual,
                                                         error,
                                                         sizeof(error)) == 1);
        const int close = compare_hidden_f16(label, actual, expected, n_tokens, 1.25e-2f, 1.25e-3);
        free(expected);
        free(prev_expected);
        if (!close) {
            free(actual);
            free(actual_prompt);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }
    CHECK(error[0] == '\0');

    free(actual);
    free(actual_prompt);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_router_topk_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image router top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_1_router_top_ids_u32.bin") ||
        !fixture_binary_exists(dump_dir, "layer_1_router_top_weights_f32.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no router top-k files; skipping image router top-k parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    (void)image_span_start;
    (void)image_span_length;
    (void)visual_features;
    (void)image_mask;
    (void)input_ids;

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image router top-k parity\n",
               n_tokens,
               n_tokens);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint32_t *actual_ids = (uint32_t *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(uint32_t));
    float *actual_weights = (float *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(float));
    CHECK(attn_hidden != NULL);
    CHECK(mlp_input != NULL);
    CHECK(actual_ids != NULL);
    CHECK(actual_weights != NULL);

    for (uint32_t layer = 1u; layer < UOCR_DECODER_LAYERS; ++layer) {
        char prev_filename[64];
        snprintf(prev_filename, sizeof(prev_filename), "layer_%u_hidden_f16.bin", layer - 1u);
        uint16_t *prev_hidden = NULL;
        CHECK(read_layer_hidden_python_dump(dump_dir, prev_filename, n_tokens, &prev_hidden, error, sizeof(error)) == 1);
        uint32_t *expected_ids = NULL;
        float *expected_weights = NULL;
        CHECK(read_router_topk_python_dump(dump_dir,
                                           layer,
                                           n_tokens,
                                           &expected_ids,
                                           &expected_weights,
                                           error,
                                           sizeof(error)) == 1);
        const uint16_t *router_weight = model_tensor_f16_payload(&model,
                                                                 uocr_tensor_id_moe_router(layer),
                                                                 (uint64_t)UOCR_ROUTED_EXPERTS *
                                                                     (uint64_t)UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 sizeof(error));
        CHECK(router_weight != NULL);
        memset(attn_hidden, 0, (size_t)hidden_values * sizeof(uint16_t));
        memset(mlp_input, 0, (size_t)hidden_values * sizeof(uint16_t));
        memset(actual_ids, 0, (size_t)n_tokens * UOCR_MOE_TOP_K * sizeof(uint32_t));
        memset(actual_weights, 0, (size_t)n_tokens * UOCR_MOE_TOP_K * sizeof(float));
        CHECK(run_metal_attention_block_from_model_f16(ctx,
                                                       &model,
                                                       layer,
                                                       prev_hidden,
                                                       n_tokens,
                                                       attn_hidden,
                                                       mlp_input,
                                                       error,
                                                       sizeof(error)) == 1);
        CHECK(uocr_metal_context_moe_router_f16(ctx,
                                                mlp_input,
                                                router_weight,
                                                n_tokens,
                                                NULL,
                                                NULL,
                                                actual_ids,
                                                actual_weights,
                                                error,
                                                sizeof(error)) == 1);
        const uint64_t values = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K;
        for (uint64_t i = 0u; i < values; ++i) {
            if (actual_ids[i] != expected_ids[i]) {
                fprintf(stderr,
                        "image router top-k id mismatch layer %u token %llu rank %llu: actual=%u expected=%u\n",
                        layer,
                        (unsigned long long)(i / (uint64_t)UOCR_MOE_TOP_K),
                        (unsigned long long)(i % (uint64_t)UOCR_MOE_TOP_K),
                        actual_ids[i],
                        expected_ids[i]);
                free(expected_weights);
                free(expected_ids);
                free(prev_hidden);
                free(actual_weights);
                free(actual_ids);
                free(mlp_input);
                free(attn_hidden);
                uocr_metal_context_destroy(ctx);
                uocr_model_file_close(&model);
                free(prompt);
                free(visual_features);
                free(image_mask);
                free(input_ids);
                return 1;
            }
            const float diff = fabsf(actual_weights[i] - expected_weights[i]);
            if (diff > 5.0e-5f) {
                fprintf(stderr,
                        "image router top-k weight mismatch layer %u token %llu rank %llu expert %u: actual=%g expected=%g diff=%g\n",
                        layer,
                        (unsigned long long)(i / (uint64_t)UOCR_MOE_TOP_K),
                        (unsigned long long)(i % (uint64_t)UOCR_MOE_TOP_K),
                        actual_ids[i],
                        (double)actual_weights[i],
                        (double)expected_weights[i],
                        (double)diff);
                free(expected_weights);
                free(expected_ids);
                free(prev_hidden);
                free(actual_weights);
                free(actual_ids);
                free(mlp_input);
                free(attn_hidden);
                uocr_metal_context_destroy(ctx);
                uocr_model_file_close(&model);
                free(prompt);
                free(visual_features);
                free(image_mask);
                free(input_ids);
                return 1;
            }
        }
        free(expected_weights);
        free(expected_ids);
        free(prev_hidden);
    }
    CHECK(error[0] == '\0');

    free(actual_weights);
    free(actual_ids);
    free(mlp_input);
    free(attn_hidden);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_logits_topk_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image logits top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "logits_topk_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "logits_topk_scores_f32.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no logits_topk files; skipping image logits top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no layer_11_hidden_f16.bin; skipping image logits top-k parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    float *expected_scores = NULL;
    uint32_t top_k = 0u;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    (void)image_span_start;
    (void)image_span_length;

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image logits top-k parity\n",
               n_tokens,
               n_tokens);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_logits_topk_python_dump(dump_dir,
                                            &expected_ids,
                                            &expected_scores,
                                            &top_k,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(top_k > 0u && top_k <= 128u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    int32_t *actual_ids = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    float *actual_scores = (float *)malloc((size_t)top_k * sizeof(float));
    CHECK(normed != NULL);
    CHECK(logits != NULL);
    CHECK(actual_ids != NULL);
    CHECK(actual_scores != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               last_hidden,
                                               1u,
                                               UOCR_METAL_RMSNORM_OUTPUT_F16,
                                               normed,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(uocr_metal_context_lm_head_f16(ctx, normed, 1u, logits, error, sizeof(error)) == 1);
    compute_logits_topk_f32(logits, UOCR_VOCAB_SIZE, top_k, actual_ids, actual_scores);
    CHECK(error[0] == '\0');

    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        if (actual_ids[rank] != expected_ids[rank]) {
            fprintf(stderr,
                    "image logits top-k id mismatch at rank %u: actual=%d expected=%d actual_score=%g expected_score=%g\n",
                    rank,
                    actual_ids[rank],
                    expected_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank]);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
        const float diff = fabsf(actual_scores[rank] - expected_scores[rank]);
        if (diff > 2.5e-2f) {
            fprintf(stderr,
                    "image logits top-k score mismatch at rank %u token %d: actual=%g expected=%g diff=%g\n",
                    rank,
                    actual_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank],
                    (double)diff);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    free(actual_scores);
    free(actual_ids);
    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_scores);
    free(expected_ids);
    free(layer11);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_image_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping image generated-id/text parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "generated_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "generated_text.txt")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no generated id/text files; skipping image generated-id/text parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no layer_11_hidden_f16.bin; skipping image generated-id/text parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    uint32_t n_generated = 0u;
    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    (void)image_span_start;
    (void)image_span_length;

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run image generated-id/text parity\n",
               n_tokens,
               n_tokens);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_generated_ids_python_dump(dump_dir,
                                              &expected_ids,
                                              &n_generated,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(n_generated == 1u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    uint32_t actual_token = UINT32_MAX;
    float actual_score = 0.0f;
    CHECK(normed != NULL);
    CHECK(logits != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   last_hidden,
                                                   1u,
                                                   NULL,
                                                   normed,
                                                   logits,
                                                   &actual_token,
                                                   &actual_score,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    if ((int32_t)actual_token != expected_ids[0]) {
        fprintf(stderr,
                "image generated-id mismatch: actual=%u expected=%d score=%g\n",
                actual_token,
                expected_ids[0],
                (double)actual_score);
        free(logits);
        free(normed);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_ids);
        free(layer11);
        free(prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 1;
    }

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = n_tokens;
    request.max_new_tokens = n_generated;
    uocr_sequence_state state;
    memset(&state, 0, sizeof(state));
    CHECK(uocr_build_sequence_state(&request, &state, error, sizeof(error)) == UOCR_OK);
    int32_t generated[1] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)actual_token, generated, 1u) == UOCR_OK);
    CHECK(generated[0] == expected_ids[0]);
    CHECK(state.generated_count == 1u);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_ids);
    free(layer11);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_integrated_image_embedding_python_dump_prefill(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping integrated image-embedding prefill parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin") ||
        !fixture_binary_exists(dump_dir, "prompt_embeddings_f16.bin") ||
        !fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no integrated image-embedding fixture files; skipping integrated image-embedding prefill parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *expected_prompt = NULL;
    uint16_t *expected_final_hidden = NULL;
    uocr_image_view *views = NULL;
    uint32_t n_views = 0u;
    uint32_t crop_grid_w = 0u;
    uint32_t crop_grid_h = 0u;

    CHECK(read_image_prompt_embedding_python_dump(dump_dir,
                                                  &input_ids,
                                                  &image_mask,
                                                  &n_tokens,
                                                  &visual_features,
                                                  &image_span_start,
                                                  &image_span_length,
                                                  &expected_prompt,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(read_prepared_view_stubs_from_manifest(dump_dir,
                                                 &views,
                                                 &n_views,
                                                 &crop_grid_w,
                                                 &crop_grid_h,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &expected_final_hidden,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(image_span_start != UOCR_SEQUENCE_NO_IMAGE_SPAN);
    CHECK(image_span_length > 0u);

    uocr_prepared_request prepared;
    memset(&prepared, 0, sizeof(prepared));
    prepared.input_ids = input_ids;
    prepared.image_mask = image_mask;
    prepared.n_tokens = n_tokens;
    prepared.views = views;
    prepared.n_views = n_views;
    prepared.crop_grid_w = crop_grid_w;
    prepared.crop_grid_h = crop_grid_h;
    prepared.max_new_tokens = 0u;
    uocr_request_limits limits;
    memset(&limits, 0, sizeof(limits));
    limits.max_prompt_tokens = n_tokens;
    limits.max_gen_tokens = 1u;
    CHECK(uocr_validate_prepared_request(&prepared, &limits, error, sizeof(error)) == UOCR_OK);
    CHECK(uocr_count_image_placeholders(&prepared) == image_span_length);

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run integrated image-embedding prefill parity\n",
               n_tokens,
               n_tokens);
        free(expected_final_hidden);
        free(views);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 0;
    }

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, n_tokens, error, sizeof(error)) == 1);

    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.image_features_f16 = visual_features;
    request.n_tokens = n_tokens;
    request.max_new_tokens = 0u;
    request.slot = 0u;
    request.image_span_start = image_span_start;
    request.image_span_length = image_span_length;

    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count == 0u);

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / 2u);
    uint16_t *actual_prompt = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual_final_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(actual_prompt != NULL);
    CHECK(actual_final_hidden != NULL);
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx,
                                                   0u,
                                                   n_tokens,
                                                   actual_prompt,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(uocr_metal_context_read_decoder_final_hidden_f16(ctx,
                                                           0u,
                                                           n_tokens,
                                                           actual_final_hidden,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint64_t i = 0u; i < hidden_values; ++i) {
        if (actual_prompt[i] != expected_prompt[i]) {
            fprintf(stderr,
                    "integrated image prompt mismatch at token %llu col %llu: actual=0x%04x expected=0x%04x\n",
                    (unsigned long long)(i / (uint64_t)UOCR_HIDDEN_SIZE),
                    (unsigned long long)(i % (uint64_t)UOCR_HIDDEN_SIZE),
                    actual_prompt[i],
                    expected_prompt[i]);
            free(actual_final_hidden);
            free(actual_prompt);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_final_hidden);
            free(views);
            free(expected_prompt);
            free(visual_features);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }
    if (!compare_hidden_f16("integrated image final hidden",
                            actual_final_hidden,
                            expected_final_hidden,
                            n_tokens,
                            1.25e-2f,
                            1.25e-3)) {
        free(actual_final_hidden);
        free(actual_prompt);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_final_hidden);
        free(views);
        free(expected_prompt);
        free(visual_features);
        free(image_mask);
        free(input_ids);
        return 1;
    }

    int32_t generated[2] = {0, 0};
    request.max_new_tokens = 2u;
    memset(&result, 0, sizeof(result));
    result.generated_ids = generated;
    result.generated_capacity = 2u;
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count >= 1u);
    CHECK(result.generated_count <= 2u);
    for (uint32_t i = 0u; i < result.generated_count; ++i) {
        CHECK(generated[i] >= 0);
        CHECK((uint32_t)generated[i] < UOCR_VOCAB_SIZE);
    }

    free(actual_final_hidden);
    free(actual_prompt);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_final_hidden);
    free(views);
    free(expected_prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return 0;
}

static int test_metal_integrated_image_embedding_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_IMAGE_EMBED_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_IMAGE_EMBED_DUMP_DIR are not both set; skipping integrated image-embedding generated-id/text parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "visual_features_f16.bin") ||
        !fixture_binary_exists(dump_dir, "prompt_embeddings_f16.bin") ||
        !fixture_binary_exists(dump_dir, "generated_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "generated_text.txt")) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has no integrated image-embedding generated id/text fixture files; skipping integrated image-embedding generated-id/text parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int exit_code = 1;
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *visual_features = NULL;
    uint32_t image_span_start = UOCR_SEQUENCE_NO_IMAGE_SPAN;
    uint32_t image_span_length = 0u;
    uint16_t *prompt = NULL;
    uocr_image_view *views = NULL;
    uint32_t n_views = 0u;
    uint32_t crop_grid_w = 0u;
    uint32_t crop_grid_h = 0u;
    int32_t *expected_ids = NULL;
    uint32_t n_expected = 0u;
    char *expected_text = NULL;
    int32_t *generated = NULL;
    float *scores = NULL;
    uocr_model_file model;
    memset(&model, 0, sizeof(model));
    int model_open = 0;
    uocr_metal_context *ctx = NULL;

    if (read_image_prompt_embedding_python_dump(dump_dir,
                                                &input_ids,
                                                &image_mask,
                                                &n_tokens,
                                                &visual_features,
                                                &image_span_start,
                                                &image_span_length,
                                                &prompt,
                                                error,
                                                sizeof(error)) != 1 ||
        read_prepared_view_stubs_from_manifest(dump_dir,
                                               &views,
                                               &n_views,
                                               &crop_grid_w,
                                               &crop_grid_h,
                                               error,
                                               sizeof(error)) != 1 ||
        read_text_generated_ids_python_dump(dump_dir, &expected_ids, &n_expected, error, sizeof(error)) != 1 ||
        read_generated_text_python_dump(dump_dir, &expected_text, error, sizeof(error)) != 1) {
        fprintf(stderr, "failed to read integrated image-embedding generated-id fixture: %s\n", error);
        goto cleanup;
    }
    if (n_tokens == 0u || image_span_start == UOCR_SEQUENCE_NO_IMAGE_SPAN || image_span_length == 0u ||
        n_expected == 0u) {
        fprintf(stderr,
                "invalid integrated image-embedding generated-id fixture: tokens=%u span_start=%u span_length=%u generated=%u\n",
                n_tokens,
                image_span_start,
                image_span_length,
                n_expected);
        goto cleanup;
    }

    const uint32_t max_decoder_tokens = env_u32_or_default("UOCR_IMAGE_DECODER_MAX_TOKENS", 32u);
    if (n_tokens > max_decoder_tokens) {
        printf("UOCR_IMAGE_EMBED_DUMP_DIR has %u tokens; set UOCR_IMAGE_DECODER_MAX_TOKENS=%u or higher to run integrated image-embedding generated-id/text parity\n",
               n_tokens,
               n_tokens);
        exit_code = 0;
        goto cleanup;
    }

    uocr_prepared_request prepared;
    memset(&prepared, 0, sizeof(prepared));
    prepared.input_ids = input_ids;
    prepared.image_mask = image_mask;
    prepared.n_tokens = n_tokens;
    prepared.views = views;
    prepared.n_views = n_views;
    prepared.crop_grid_w = crop_grid_w;
    prepared.crop_grid_h = crop_grid_h;
    prepared.max_new_tokens = n_expected;
    prepared.no_repeat_ngram_size = 0u;
    prepared.no_repeat_window = 0u;
    uocr_request_limits limits;
    memset(&limits, 0, sizeof(limits));
    limits.max_prompt_tokens = n_tokens;
    limits.max_gen_tokens = n_expected;
    if (uocr_validate_prepared_request(&prepared, &limits, error, sizeof(error)) != UOCR_OK) {
        fprintf(stderr, "integrated image-embedding generated-id fixture validation failed: %s\n", error);
        goto cleanup;
    }
    if (uocr_count_image_placeholders(&prepared) != image_span_length) {
        fprintf(stderr,
                "integrated image-embedding generated-id fixture image-span mismatch: placeholders=%u span=%u\n",
                uocr_count_image_placeholders(&prepared),
                image_span_length);
        goto cleanup;
    }

    if (uocr_model_file_open(model_path, &model, error, sizeof(error)) != 0) {
        fprintf(stderr, "failed to open model for integrated image-embedding generated-id parity: %s\n", error);
        goto cleanup;
    }
    model_open = 1;
    ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    if (ctx == NULL) {
        fprintf(stderr, "failed to create Metal context for integrated image-embedding generated-id parity: %s\n", error);
        goto cleanup;
    }
    if (uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) != 1 ||
        uocr_metal_context_allocate_runtime_arenas(ctx, 1u, n_tokens, error, sizeof(error)) != 1) {
        fprintf(stderr, "failed to prepare Metal context for integrated image-embedding generated-id parity: %s\n", error);
        goto cleanup;
    }

    generated = (int32_t *)calloc((size_t)n_expected, sizeof(int32_t));
    scores = (float *)calloc((size_t)n_expected, sizeof(float));
    if (generated == NULL || scores == NULL) {
        fprintf(stderr, "failed to allocate integrated image-embedding generated-id output buffers\n");
        goto cleanup;
    }

    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.image_features_f16 = visual_features;
    request.n_tokens = n_tokens;
    request.max_new_tokens = n_expected;
    request.slot = 0u;
    request.image_span_start = image_span_start;
    request.image_span_length = image_span_length;
    request.no_repeat_ngram_size = 0u;
    request.no_repeat_window = 0u;

    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    result.generated_ids = generated;
    result.generated_scores_f32_or_null = scores;
    result.generated_capacity = n_expected;
    if (uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) != 1) {
        fprintf(stderr, "integrated image-embedding generated-id generation failed: %s\n", error);
        goto cleanup;
    }
    if (result.generated_count != n_expected) {
        fprintf(stderr,
                "integrated image-embedding generated-id count mismatch: actual=%u expected=%u decoded_text='%s'\n",
                result.generated_count,
                n_expected,
                expected_text != NULL ? expected_text : "");
        goto cleanup;
    }
    for (uint32_t i = 0u; i < n_expected; ++i) {
        if (generated[i] != expected_ids[i]) {
            fprintf(stderr,
                    "integrated image-embedding generated-id mismatch at %u: actual=%d expected=%d score=%g decoded_text='%s'\n",
                    i,
                    generated[i],
                    expected_ids[i],
                    (double)scores[i],
                    expected_text != NULL ? expected_text : "");
            goto cleanup;
        }
    }

    /* C does not own tokenizer decoding in v1. The fixture's generated_text.txt is
       loaded above, and exact generated-id parity makes that decoded text parity
       deterministic for the Python-owned tokenizer/frontend path. */
    exit_code = 0;

cleanup:
    if (ctx != NULL) {
        uocr_metal_context_destroy(ctx);
    }
    if (model_open) {
        uocr_model_file_close(&model);
    }
    free(scores);
    free(generated);
    free(expected_text);
    free(expected_ids);
    free(views);
    free(prompt);
    free(visual_features);
    free(image_mask);
    free(input_ids);
    return exit_code;
}

static int test_metal_text_prompt_embedding_full_model_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *hf_dir = getenv("UOCR_HF_DIR");
    if (model_path == NULL || model_path[0] == '\0' || hf_dir == NULL || hf_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_HF_DIR are not both set; skipping full prompt parity\n");
        return 0;
    }

    enum { TOKENS = 6 };
    const int32_t token_ids[TOKENS] = {0, 1, 2, 42, 1000, 128000};
    uint16_t expected[TOKENS * UOCR_HIDDEN_SIZE];
    char error[1024];
    memset(error, 0, sizeof(error));
    CHECK(read_hf_embed_rows_as_f16(hf_dir, token_ids, TOKENS, expected, error, sizeof(error)) == 1);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t actual[TOKENS * UOCR_HIDDEN_SIZE];
    memset(actual, 0, sizeof(actual));
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            token_ids,
                                                            TOKENS,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            actual,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * UOCR_HIDDEN_SIZE); ++i) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "prompt embedding mismatch at flat index %u: actual=0x%04x expected=0x%04x\n",
                    i,
                    actual[i],
                    expected[i]);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            return 1;
        }
    }

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    return 0;
}

static int test_metal_text_layer0_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text layer0 parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *expected = NULL;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &expected, error, sizeof(error)) == 1);
    (void)input_ids;

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(n_tokens > 0u);
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / 2u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    const uint16_t *input_norm_weight = model_tensor_f16_payload(&model,
                                                                 uocr_tensor_id_layer_input_norm(0u),
                                                                 UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 sizeof(error));
    const uint16_t *q_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_Q),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *k_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_K),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *v_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_V),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *o_weight = model_tensor_f16_payload(&model,
                                                        uocr_tensor_id_layer_attn(0u, UOCR_TENSOR_PROJ_O),
                                                        (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE,
                                                        error,
                                                        sizeof(error));
    const uint16_t *post_norm_weight = model_tensor_f16_payload(&model,
                                                                uocr_tensor_id_layer_post_attn_norm(0u),
                                                                UOCR_HIDDEN_SIZE,
                                                                error,
                                                                sizeof(error));
    const uint16_t *gate_weight = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           sizeof(error));
    const uint16_t *up_weight = model_tensor_f16_payload(&model,
                                                         uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         sizeof(error));
    const uint16_t *down_weight = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE,
                                                           error,
                                                           sizeof(error));
    CHECK(input_norm_weight != NULL);
    CHECK(q_weight != NULL);
    CHECK(k_weight != NULL);
    CHECK(v_weight != NULL);
    CHECK(o_weight != NULL);
    CHECK(post_norm_weight != NULL);
    CHECK(gate_weight != NULL);
    CHECK(up_weight != NULL);
    CHECK(down_weight != NULL);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *normed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *v = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *unused_o = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *q_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *k_rope = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *context = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(normed != NULL);
    CHECK(q != NULL);
    CHECK(k != NULL);
    CHECK(v != NULL);
    CHECK(unused_o != NULL);
    CHECK(q_rope != NULL);
    CHECK(k_rope != NULL);
    CHECK(context != NULL);
    CHECK(attn_hidden != NULL);
    CHECK(mlp_input != NULL);
    CHECK(actual != NULL);

    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         prompt,
                                         input_norm_weight,
                                         n_tokens,
                                         UOCR_HIDDEN_SIZE,
                                         UOCR_RMS_NORM_EPS,
                                         UOCR_METAL_RMSNORM_OUTPUT_F16,
                                         normed,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                normed,
                                                q_weight,
                                                k_weight,
                                                v_weight,
                                                o_weight,
                                                n_tokens,
                                                UOCR_METAL_DENSE_OUTPUT_F16,
                                                q,
                                                k,
                                                v,
                                                unused_o,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         n_tokens,
                                         0u,
                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                         q_rope,
                                         k_rope,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q_rope,
                                                   k_rope,
                                                   v,
                                                   n_tokens,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   context,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           o_weight,
                                                           prompt,
                                                           n_tokens,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           attn_hidden,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         attn_hidden,
                                         post_norm_weight,
                                         n_tokens,
                                         UOCR_HIDDEN_SIZE,
                                         UOCR_RMS_NORM_EPS,
                                         UOCR_METAL_RMSNORM_OUTPUT_F16,
                                         mlp_input,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              mlp_input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              attn_hidden,
                                              n_tokens,
                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                              actual,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    float max_abs = 0.0f;
    double sum_abs = 0.0;
    uint64_t max_index = 0u;
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        const float diff = fabsf(f16_bits_to_f32(actual[i]) - f16_bits_to_f32(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            max_index = i;
        }
        sum_abs += (double)diff;
    }
    const double mean_abs = sum_abs / (double)hidden_values;
    if (max_abs > 7.5e-3f || mean_abs > 7.5e-4) {
        fprintf(stderr,
                "text layer0 mismatch: max_abs=%g mean_abs=%g at token %llu col %llu actual=%g expected=%g\n",
                (double)max_abs,
                mean_abs,
                (unsigned long long)(max_index / (uint64_t)UOCR_HIDDEN_SIZE),
                (unsigned long long)(max_index % (uint64_t)UOCR_HIDDEN_SIZE),
                (double)f16_bits_to_f32(actual[max_index]),
                (double)f16_bits_to_f32(expected[max_index]));
        free(actual);
        free(mlp_input);
        free(attn_hidden);
        free(context);
        free(k_rope);
        free(q_rope);
        free(unused_o);
        free(v);
        free(k);
        free(q);
        free(normed);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected);
        free(prompt);
        free(input_ids);
        return 1;
    }

    free(actual);
    free(mlp_input);
    free(attn_hidden);
    free(context);
    free(k_rope);
    free(q_rope);
    free(unused_o);
    free(v);
    free(k);
    free(q);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_layer1_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text layer1 parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    uint16_t *expected = NULL;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir, "layer_1_hidden_f16.bin", n_tokens, &expected, error, sizeof(error)) == 1);
    (void)input_ids;
    (void)prompt;

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    const uint64_t expert_values = (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE *
                                   (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(n_tokens > 0u);
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));
    CHECK(expert_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);

    const uint16_t *router_weight = model_tensor_f16_payload(&model,
                                                             uocr_tensor_id_moe_router(1u),
                                                             (uint64_t)UOCR_ROUTED_EXPERTS * (uint64_t)UOCR_HIDDEN_SIZE,
                                                             error,
                                                             sizeof(error));
    const uint16_t *shared_gate = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_moe_shared(1u, UOCR_TENSOR_PROJ_GATE),
                                                           (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                               (uint64_t)UOCR_HIDDEN_SIZE,
                                                           error,
                                                           sizeof(error));
    const uint16_t *shared_up = model_tensor_f16_payload(&model,
                                                         uocr_tensor_id_moe_shared(1u, UOCR_TENSOR_PROJ_UP),
                                                         (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE *
                                                             (uint64_t)UOCR_HIDDEN_SIZE,
                                                         error,
                                                         sizeof(error));
    const uint16_t *shared_down = model_tensor_f16_payload(&model,
                                                           uocr_tensor_id_moe_shared(1u, UOCR_TENSOR_PROJ_DOWN),
                                                           (uint64_t)UOCR_HIDDEN_SIZE *
                                                               (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE,
                                                           error,
                                                           sizeof(error));
    CHECK(router_weight != NULL);
    CHECK(shared_gate != NULL);
    CHECK(shared_up != NULL);
    CHECK(shared_down != NULL);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *attn_hidden = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *mlp_input = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *routed = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *shared = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    uint32_t *top_ids = (uint32_t *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(uint32_t));
    float *top_weights = (float *)calloc((size_t)n_tokens * UOCR_MOE_TOP_K, sizeof(float));
    uint16_t *selected_gate = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
    uint16_t *selected_up = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
    uint16_t *selected_down = (uint16_t *)malloc((size_t)expert_values * sizeof(uint16_t));
    CHECK(attn_hidden != NULL);
    CHECK(mlp_input != NULL);
    CHECK(routed != NULL);
    CHECK(shared != NULL);
    CHECK(actual != NULL);
    CHECK(top_ids != NULL);
    CHECK(top_weights != NULL);
    CHECK(selected_gate != NULL);
    CHECK(selected_up != NULL);
    CHECK(selected_down != NULL);

    CHECK(run_metal_attention_block_from_model_f16(ctx,
                                                   &model,
                                                   1u,
                                                   layer0,
                                                   n_tokens,
                                                   attn_hidden,
                                                   mlp_input,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            mlp_input,
                                            router_weight,
                                            n_tokens,
                                            NULL,
                                            NULL,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    mlp_input,
                                                    shared_gate,
                                                    shared_up,
                                                    shared_down,
                                                    n_tokens,
                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                    shared,
                                                    error,
                                                    sizeof(error)) == 1);
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        CHECK(copy_selected_moe_expert_weights(&model,
                                               1u,
                                               top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                               selected_gate,
                                               selected_up,
                                               selected_down,
                                               error,
                                               sizeof(error)) == 1);
        CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                                 mlp_input + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                 top_ids + (size_t)token * UOCR_MOE_TOP_K,
                                                                 top_weights + (size_t)token * UOCR_MOE_TOP_K,
                                                                 selected_gate,
                                                                 selected_up,
                                                                 selected_down,
                                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                                 routed + (size_t)token * UOCR_HIDDEN_SIZE,
                                                                 error,
                                                                 sizeof(error)) == 1);
    }
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             attn_hidden,
                                             n_tokens,
                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                             actual,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    float max_abs = 0.0f;
    double sum_abs = 0.0;
    uint64_t max_index = 0u;
    for (uint64_t i = 0u; i < hidden_values; ++i) {
        const float diff = fabsf(f16_bits_to_f32(actual[i]) - f16_bits_to_f32(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            max_index = i;
        }
        sum_abs += (double)diff;
    }
    const double mean_abs = sum_abs / (double)hidden_values;
    if (max_abs > 1.25e-2f || mean_abs > 1.25e-3) {
        fprintf(stderr,
                "text layer1 mismatch: max_abs=%g mean_abs=%g at token %llu col %llu actual=%g expected=%g\n",
                (double)max_abs,
                mean_abs,
                (unsigned long long)(max_index / (uint64_t)UOCR_HIDDEN_SIZE),
                (unsigned long long)(max_index % (uint64_t)UOCR_HIDDEN_SIZE),
                (double)f16_bits_to_f32(actual[max_index]),
                (double)f16_bits_to_f32(expected[max_index]));
        free(selected_down);
        free(selected_up);
        free(selected_gate);
        free(top_weights);
        free(top_ids);
        free(actual);
        free(shared);
        free(routed);
        free(mlp_input);
        free(attn_hidden);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected);
        free(layer0);
        free(prompt);
        free(input_ids);
        return 1;
    }

    free(selected_down);
    free(selected_up);
    free(selected_gate);
    free(top_weights);
    free(top_ids);
    free(actual);
    free(shared);
    free(routed);
    free(mlp_input);
    free(attn_hidden);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_remaining_layers_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text remaining-layer parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "layer_11_hidden_f16.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no layer_11_hidden_f16.bin; skipping text remaining-layer parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    (void)input_ids;
    (void)prompt;

    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    CHECK(n_tokens > 0u);
    CHECK(hidden_values <= (uint64_t)SIZE_MAX / sizeof(uint16_t));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t *input_hidden = (uint16_t *)malloc((size_t)hidden_values * sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)calloc((size_t)hidden_values, sizeof(uint16_t));
    CHECK(input_hidden != NULL);
    CHECK(actual != NULL);
    memcpy(input_hidden, layer0, (size_t)hidden_values * sizeof(uint16_t));

    for (uint32_t layer = 2u; layer < UOCR_DECODER_LAYERS; ++layer) {
        char prev_filename[64];
        char expected_filename[64];
        snprintf(prev_filename, sizeof(prev_filename), "layer_%u_hidden_f16.bin", layer - 1u);
        snprintf(expected_filename, sizeof(expected_filename), "layer_%u_hidden_f16.bin", layer);
        uint16_t *prev_expected = NULL;
        uint16_t *expected = NULL;
        CHECK(read_layer_hidden_python_dump(dump_dir, prev_filename, n_tokens, &prev_expected, error, sizeof(error)) == 1);
        CHECK(read_layer_hidden_python_dump(dump_dir, expected_filename, n_tokens, &expected, error, sizeof(error)) == 1);
        memcpy(input_hidden, prev_expected, (size_t)hidden_values * sizeof(uint16_t));
        memset(actual, 0, (size_t)hidden_values * sizeof(uint16_t));
        CHECK(run_metal_moe_decoder_layer_from_model_f16(ctx,
                                                         &model,
                                                         layer,
                                                         input_hidden,
                                                         n_tokens,
                                                         actual,
                                                         error,
                                                         sizeof(error)) == 1);
        char label[64];
        snprintf(label, sizeof(label), "text layer%u", layer);
        const int close = compare_hidden_f16(label, actual, expected, n_tokens, 1.25e-2f, 1.25e-3);
        free(expected);
        free(prev_expected);
        if (!close) {
            free(actual);
            free(input_hidden);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(layer0);
            free(prompt);
            free(input_ids);
            return 1;
        }
    }
    CHECK(error[0] == '\0');

    free(actual);
    free(input_hidden);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_logits_topk_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text logits top-k parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "logits_topk_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "logits_topk_scores_f32.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no logits_topk files; skipping text logits top-k parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    float *expected_scores = NULL;
    uint32_t top_k = 0u;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_logits_topk_python_dump(dump_dir,
                                            &expected_ids,
                                            &expected_scores,
                                            &top_k,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(top_k > 0u && top_k <= 128u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    int32_t *actual_ids = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    float *actual_scores = (float *)malloc((size_t)top_k * sizeof(float));
    CHECK(normed != NULL);
    CHECK(logits != NULL);
    CHECK(actual_ids != NULL);
    CHECK(actual_scores != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               last_hidden,
                                               1u,
                                               UOCR_METAL_RMSNORM_OUTPUT_F16,
                                               normed,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(uocr_metal_context_lm_head_f16(ctx, normed, 1u, logits, error, sizeof(error)) == 1);
    compute_logits_topk_f32(logits, UOCR_VOCAB_SIZE, top_k, actual_ids, actual_scores);
    CHECK(error[0] == '\0');

    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        if (actual_ids[rank] != expected_ids[rank]) {
            fprintf(stderr,
                    "text logits top-k id mismatch at rank %u: actual=%d expected=%d actual_score=%g expected_score=%g\n",
                    rank,
                    actual_ids[rank],
                    expected_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank]);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(layer0);
            free(prompt);
            free(input_ids);
            return 1;
        }
        const float diff = fabsf(actual_scores[rank] - expected_scores[rank]);
        if (diff > 2.5e-2f) {
            fprintf(stderr,
                    "text logits top-k score mismatch at rank %u token %d: actual=%g expected=%g diff=%g\n",
                    rank,
                    actual_ids[rank],
                    (double)actual_scores[rank],
                    (double)expected_scores[rank],
                    (double)diff);
            free(actual_scores);
            free(actual_ids);
            free(logits);
            free(normed);
            uocr_metal_context_destroy(ctx);
            uocr_model_file_close(&model);
            free(expected_scores);
            free(expected_ids);
            free(layer11);
            free(layer0);
            free(prompt);
            free(input_ids);
            return 1;
        }
    }

    free(actual_scores);
    free(actual_ids);
    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_scores);
    free(expected_ids);
    free(layer11);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_text_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping text generated-id parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "generated_ids_i32.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no generated_ids_i32.bin; skipping text generated-id parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint32_t n_tokens = 0u;
    uint16_t *prompt = NULL;
    uint16_t *layer0 = NULL;
    uint16_t *layer11 = NULL;
    int32_t *expected_ids = NULL;
    uint32_t n_generated = 0u;
    CHECK(read_text_layer0_python_dump(dump_dir, &input_ids, &n_tokens, &prompt, &layer0, error, sizeof(error)) == 1);
    CHECK(read_layer_hidden_python_dump(dump_dir,
                                        "layer_11_hidden_f16.bin",
                                        n_tokens,
                                        &layer11,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(read_text_generated_ids_python_dump(dump_dir,
                                              &expected_ids,
                                              &n_generated,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(n_generated == 1u);

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)calloc((size_t)UOCR_HIDDEN_SIZE, sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    uint32_t actual_token = UINT32_MAX;
    float actual_score = 0.0f;
    CHECK(normed != NULL);
    CHECK(logits != NULL);

    const uint16_t *last_hidden = layer11 + (size_t)(n_tokens - 1u) * UOCR_HIDDEN_SIZE;
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   last_hidden,
                                                   1u,
                                                   NULL,
                                                   normed,
                                                   logits,
                                                   &actual_token,
                                                   &actual_score,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    if ((int32_t)actual_token != expected_ids[0]) {
        fprintf(stderr,
                "text generated-id mismatch: actual=%u expected=%d score=%g\n",
                actual_token,
                expected_ids[0],
                (double)actual_score);
        free(logits);
        free(normed);
        uocr_metal_context_destroy(ctx);
        uocr_model_file_close(&model);
        free(expected_ids);
        free(layer11);
        free(layer0);
        free(prompt);
        free(input_ids);
        return 1;
    }

    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    free(expected_ids);
    free(layer11);
    free(layer0);
    free(prompt);
    free(input_ids);
    return 0;
}

static int test_metal_get_rows_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TABLE_ROWS = 5, ROW_WIDTH = 8, OUT_ROWS = 4 };
    const uint16_t values[] = {
        0x0000u, /* 0.0 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x3800u, /* 0.5 */
        0xb800u, /* -0.5 */
        0x4200u, /* 3.0 */
        0xc200u, /* -3.0 */
        0x4400u  /* 4.0 */
    };
    uint16_t table[TABLE_ROWS * ROW_WIDTH];
    for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * ROW_WIDTH); ++i) {
        table[i] = values[(i * 3u + 1u) % (sizeof(values) / sizeof(values[0]))];
    }
    const int32_t row_ids[OUT_ROWS] = {3, 0, 4, 1};

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint16_t out_f16[OUT_ROWS * ROW_WIDTH];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          row_ids,
                                          OUT_ROWS,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                          out_f16,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)ROW_WIDTH; ++col) {
            const uint16_t expected = table[(uint32_t)row_ids[row] * (uint32_t)ROW_WIDTH + col];
            CHECK(out_f16[row * (uint32_t)ROW_WIDTH + col] == expected);
        }
    }

    float out_f32[OUT_ROWS * ROW_WIDTH];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          row_ids,
                                          OUT_ROWS,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F32,
                                          out_f32,
                                          error,
                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)ROW_WIDTH; ++col) {
            const uint16_t expected_bits = table[(uint32_t)row_ids[row] * (uint32_t)ROW_WIDTH + col];
            CHECK(out_f32[row * (uint32_t)ROW_WIDTH + col] == f16_bits_to_f32(expected_bits));
        }
    }

    const int32_t bad_row_ids[1] = {TABLE_ROWS};
    CHECK(uocr_metal_context_get_rows_f16(ctx,
                                          table,
                                          TABLE_ROWS,
                                          ROW_WIDTH,
                                          bad_row_ids,
                                          1u,
                                          UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                          out_f16,
                                          error,
                                          sizeof(error)) == 0);
    CHECK(strstr(error, "outside table rows") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_prompt_assembly_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TABLE_ROWS = 6, HIDDEN = 8, TEXT_TOKENS = 4, IMAGE_TOKENS = 2, PROMPT_TOKENS = 6 };
    uint16_t table[TABLE_ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)TABLE_ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            table[row * (uint32_t)HIDDEN + col] = (uint16_t)(0x3c00u + row * 0x40u + col);
        }
    }
    uint16_t image_features[IMAGE_TOKENS * HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
        image_features[i] = (uint16_t)(0x5000u + i);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const int32_t text_ids[TEXT_TOKENS] = {0, 3, 5, 1};
    uint16_t text_out[TEXT_TOKENS * HIDDEN];
    memset(text_out, 0, sizeof(text_out));
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 text_ids,
                                                 TEXT_TOKENS,
                                                 UINT32_MAX,
                                                 0u,
                                                 NULL,
                                                 text_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)TEXT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(text_out[token * (uint32_t)HIDDEN + col] == table[(uint32_t)text_ids[token] * (uint32_t)HIDDEN + col]);
        }
    }

    const int32_t prompt_ids[PROMPT_TOKENS] = {0, 2, 12345, 12346, 4, 1};
    uint16_t prompt_out[PROMPT_TOKENS * HIDDEN];
    memset(prompt_out, 0, sizeof(prompt_out));
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 prompt_ids,
                                                 PROMPT_TOKENS,
                                                 2u,
                                                 IMAGE_TOKENS,
                                                 image_features,
                                                 prompt_out,
                                                 error,
                                                 sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)PROMPT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint16_t expected = 0u;
            if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
            } else {
                expected = table[(uint32_t)prompt_ids[token] * (uint32_t)HIDDEN + col];
            }
            CHECK(prompt_out[token * (uint32_t)HIDDEN + col] == expected);
        }
    }

    const int32_t bad_text_ids[1] = {TABLE_ROWS};
    CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                 table,
                                                 TABLE_ROWS,
                                                 HIDDEN,
                                                 bad_text_ids,
                                                 1u,
                                                 UINT32_MAX,
                                                 0u,
                                                 NULL,
                                                 text_out,
                                                 error,
                                                 sizeof(error)) == 0);
    CHECK(strstr(error, "outside embedding rows") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_prompt_assembly_from_mapped_model_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROW_COUNT = 4, HIDDEN = UOCR_HIDDEN_SIZE, TEXT_TOKENS = 3, IMAGE_TOKENS = 2, PROMPT_TOKENS = 6 };
    const uint32_t rows[ROW_COUNT] = {0u, 1u, 42u, 99u};
    uint16_t row_weights[ROW_COUNT * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)(((row + 5u) * 13u + col * 7u) % 31u) - 15;
            row_weights[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.015625f);
        }
    }
    uint16_t image_features[IMAGE_TOKENS * HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
        image_features[i] = f32_to_f16_bits((float)((int)(i % 23u) - 11) * 0.03125f);
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_sparse_tok_embed_uocr_model(path, rows, row_weights, ROW_COUNT) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const int32_t text_ids[TEXT_TOKENS] = {0, 42, 1};
    uint16_t text_out[TEXT_TOKENS * HIDDEN];
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            text_ids,
                                                            TEXT_TOKENS,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            text_out,
                                                            error,
                                                            sizeof(error)) == 0);
    CHECK(strstr(error, "requires mapped model views") != NULL);

    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            text_ids,
                                                            TEXT_TOKENS,
                                                            UINT32_MAX,
                                                            0u,
                                                            NULL,
                                                            text_out,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)TEXT_TOKENS; ++token) {
        uint32_t row_index = UINT32_MAX;
        for (uint32_t r = 0u; r < (uint32_t)ROW_COUNT; ++r) {
            if (rows[r] == (uint32_t)text_ids[token]) {
                row_index = r;
            }
        }
        CHECK(row_index != UINT32_MAX);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(text_out[token * (uint32_t)HIDDEN + col] == row_weights[row_index * (uint32_t)HIDDEN + col]);
        }
    }

    const int32_t prompt_ids[PROMPT_TOKENS] = {0, 99, UOCR_TOKEN_IMAGE, UOCR_TOKEN_IMAGE, 42, 1};
    uint16_t prompt_out[PROMPT_TOKENS * HIDDEN];
    CHECK(uocr_metal_context_assemble_prompt_from_model_f16(ctx,
                                                            prompt_ids,
                                                            PROMPT_TOKENS,
                                                            2u,
                                                            IMAGE_TOKENS,
                                                            image_features,
                                                            prompt_out,
                                                            error,
                                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t token = 0u; token < (uint32_t)PROMPT_TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint16_t expected = 0u;
            if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
            } else {
                uint32_t row_index = UINT32_MAX;
                for (uint32_t r = 0u; r < (uint32_t)ROW_COUNT; ++r) {
                    if (rows[r] == (uint32_t)prompt_ids[token]) {
                        row_index = r;
                    }
                }
                CHECK(row_index != UINT32_MAX);
                expected = row_weights[row_index * (uint32_t)HIDDEN + col];
            }
            CHECK(prompt_out[token * (uint32_t)HIDDEN + col] == expected);
        }
    }

    CHECK(uocr_metal_context_assemble_prompt_from_model_to_arena_f16(ctx,
                                                                     text_ids,
                                                                     TEXT_TOKENS,
                                                                     UINT32_MAX,
                                                                     0u,
                                                                     NULL,
                                                                     0u,
                                                                     error,
                                                                     sizeof(error)) == 0);
    CHECK(strstr(error, "requires allocated arenas") != NULL);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 2u, PROMPT_TOKENS, error, sizeof(error)) == 1);
    CHECK(uocr_metal_context_assemble_prompt_from_model_to_arena_f16(ctx,
                                                                     text_ids,
                                                                     TEXT_TOKENS,
                                                                     UINT32_MAX,
                                                                     0u,
                                                                     NULL,
                                                                     0u,
                                                                     error,
                                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    uint16_t arena_text_out[TEXT_TOKENS * HIDDEN];
    memset(arena_text_out, 0, sizeof(arena_text_out));
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx, 0u, TEXT_TOKENS, arena_text_out, error, sizeof(error)) == 1);
    CHECK(memcmp(arena_text_out, text_out, sizeof(arena_text_out)) == 0);

    CHECK(uocr_metal_context_assemble_prompt_from_model_to_arena_f16(ctx,
                                                                     prompt_ids,
                                                                     PROMPT_TOKENS,
                                                                     2u,
                                                                     IMAGE_TOKENS,
                                                                     image_features,
                                                                     1u,
                                                                     error,
                                                                     sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    uint16_t arena_prompt_out[PROMPT_TOKENS * HIDDEN];
    memset(arena_prompt_out, 0, sizeof(arena_prompt_out));
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx, 1u, PROMPT_TOKENS, arena_prompt_out, error, sizeof(error)) == 1);
    CHECK(memcmp(arena_prompt_out, prompt_out, sizeof(arena_prompt_out)) == 0);
    CHECK(uocr_metal_context_read_prompt_arena_f16(ctx, 2u, TEXT_TOKENS, arena_text_out, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires allocated arenas") != NULL);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_rmsnorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, HIDDEN = 1280 };
    const uint16_t input_values[] = {
        0x3800u, /* 0.5 */
        0xb800u, /* -0.5 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x3400u, /* 0.25 */
        0xb400u  /* -0.25 */
    };
    const uint16_t weight_values[] = {
        0x3c00u, /* 1.0 */
        0x3800u, /* 0.5 */
        0x4000u, /* 2.0 */
        0x3e00u, /* 1.5 */
        0x3a00u  /* 0.75 */
    };
    uint16_t input[ROWS * HIDDEN];
    uint16_t weight[HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        input[i] = input_values[(i * 5u + 3u) % (sizeof(input_values) / sizeof(input_values[0]))];
    }
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        weight[i] = weight_values[(i * 7u + 1u) % (sizeof(weight_values) / sizeof(weight_values[0]))];
    }

    const float eps = 1.0e-6f;
    float expected[ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            sum += x * x;
        }
        const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + eps);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            const float w = f16_bits_to_f32(weight[col]);
            expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    float out_f32[ROWS * HIDDEN];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         eps,
                                         UOCR_METAL_RMSNORM_OUTPUT_F32,
                                         out_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-4f);
    }

    uint16_t out_f16[ROWS * HIDDEN];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         eps,
                                         UOCR_METAL_RMSNORM_OUTPUT_F16,
                                         out_f16,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 4.0e-3f);
    }

    CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                         input,
                                         weight,
                                         ROWS,
                                         HIDDEN,
                                         0.0f,
                                         UOCR_METAL_RMSNORM_OUTPUT_F32,
                                         out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "eps must be positive") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_final_rmsnorm_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, HIDDEN = UOCR_HIDDEN_SIZE };
    uint16_t input[ROWS * HIDDEN];
    uint16_t weight[HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)((row * 13u + col * 5u) % 17u) - 8;
            input[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.0625f);
        }
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        weight[col] = f32_to_f16_bits(0.5f + (float)((col * 3u) % 11u) * 0.03125f);
    }

    float expected[ROWS * HIDDEN];
    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            sum += x * x;
        }
        const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + UOCR_RMS_NORM_EPS);
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
            const float w = f16_bits_to_f32(weight[col]);
            expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
        }
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_single_final_norm_uocr_model(path, weight) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    float out_f32[ROWS * HIDDEN];
    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F32,
                                               out_f32,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-4f);
    }

    uint16_t out_f16[ROWS * HIDDEN];
    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F16,
                                               out_f16,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 4.0e-3f);
    }

    uocr_metal_context_unmap_model(ctx);
    CHECK(uocr_metal_context_final_rmsnorm_f16(ctx,
                                               input,
                                               ROWS,
                                               UOCR_METAL_RMSNORM_OUTPUT_F32,
                                               out_f32,
                                               error,
                                               sizeof(error)) == 0);
    CHECK(strstr(error, "mapped model views") != NULL);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_lm_head_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROW_COUNT = 4, HIDDEN = UOCR_HIDDEN_SIZE };
    const uint32_t rows[ROW_COUNT] = {0u, 1u, UOCR_TOKEN_IMAGE, UOCR_VOCAB_SIZE - 1u};
    uint16_t input[HIDDEN];
    uint16_t row_weights[ROW_COUNT * HIDDEN];
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const int centered = (int)((col * 7u) % 13u) - 6;
        input[col] = f32_to_f16_bits((float)centered * 0.03125f);
    }
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const int centered = (int)(((row + 3u) * 11u + col * 5u) % 17u) - 8;
            row_weights[row * (uint32_t)HIDDEN + col] = f32_to_f16_bits((float)centered * 0.015625f);
        }
    }

    float expected[ROW_COUNT];
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            sum += f16_bits_to_f32(input[col]) * f16_bits_to_f32(row_weights[row * (uint32_t)HIDDEN + col]);
        }
        expected[row] = sum;
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_sparse_lm_head_uocr_model(path, rows, row_weights, ROW_COUNT) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(logits != NULL);
    memset(logits, 0x7f, (size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(uocr_metal_context_lm_head_f16(ctx, input, 1u, logits, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t row = 0u; row < (uint32_t)ROW_COUNT; ++row) {
        CHECK(fabsf(logits[rows[row]] - expected[row]) < 2.0e-4f);
    }
    CHECK(fabsf(logits[UOCR_TOKEN_PAD]) < 1.0e-7f);
    CHECK(fabsf(logits[42u]) < 1.0e-7f);

    free(logits);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_argmax_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, VOCAB = UOCR_VOCAB_SIZE };
    float *logits = (float *)malloc((size_t)ROWS * (size_t)VOCAB * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = -1000.0f;
    }
    logits[0u * (uint32_t)VOCAB + 17u] = 3.25f;
    logits[0u * (uint32_t)VOCAB + 42u] = 3.25f; /* tie: lower token id wins */
    logits[1u * (uint32_t)VOCAB + UOCR_TOKEN_PAD] = -0.5f;
    logits[1u * (uint32_t)VOCAB + UOCR_TOKEN_IMAGE] = 12.0f;
    logits[1u * (uint32_t)VOCAB + (uint32_t)VOCAB - 1u] = 11.9f;
    for (uint32_t col = 0u; col < (uint32_t)VOCAB; ++col) {
        logits[2u * (uint32_t)VOCAB + col] = -INFINITY;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f, 0.0f};
    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        logits,
                                        ROWS,
                                        VOCAB,
                                        token_ids,
                                        scores,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(token_ids[0] == 17u);
    CHECK(scores[0] == 3.25f);
    CHECK(token_ids[1] == (uint32_t)UOCR_TOKEN_IMAGE);
    CHECK(scores[1] == 12.0f);
    CHECK(token_ids[2] == 0u);
    CHECK(isinf(scores[2]) && scores[2] < 0.0f);

    const float small_logits[10] = {
        -1.0f, 0.5f, 0.5f, -2.0f, 0.25f,
        -INFINITY, -4.0f, -3.0f, -2.0f, -1.0f,
    };
    uint32_t small_ids[2] = {UINT32_MAX, UINT32_MAX};
    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        small_logits,
                                        2u,
                                        5u,
                                        small_ids,
                                        NULL,
                                        error,
                                        sizeof(error)) == 1);
    CHECK(small_ids[0] == 1u);
    CHECK(small_ids[1] == 4u);

    CHECK(uocr_metal_context_argmax_f32(ctx,
                                        logits,
                                        ROWS,
                                        0u,
                                        token_ids,
                                        scores,
                                        error,
                                        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal argmax request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(logits);
    return 0;
}

static int test_metal_no_repeat_ngram_gpu_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 4, VOCAB = 16 };
    float logits[ROWS * VOCAB];
    float expected[ROWS * VOCAB];
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = (float)i * 0.125f;
        expected[i] = logits[i];
    }

    const int32_t sequence0[] = {1, 2, 3, 1, 2};       /* current prefix [1,2] bans 3 */
    const int32_t sequence1[] = {5, 6, 7, 5, 6, 8, 5, 6}; /* full window bans 7 and 8 */
    const int32_t sequence2[] = {2, 4, 2, 5};          /* ngram=1, window=2 bans 2 and 5 */
    const uocr_no_repeat_ngram_config configs[ROWS] = {
        {sequence0, 5u, 3u, 0u},
        {sequence1, 8u, 3u, 0u},
        {sequence2, 4u, 1u, 2u},
        {NULL, 0u, 0u, 0u},
    };
    CHECK(uocr_no_repeat_ngram_apply_batch(expected, ROWS, VOCAB, configs) == UOCR_OK);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_apply_no_repeat_ngram_f32(ctx,
                                                       logits,
                                                       ROWS,
                                                       VOCAB,
                                                       configs,
                                                       error,
                                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        if (isinf(expected[i])) {
            CHECK(isinf(logits[i]) && logits[i] < 0.0f);
        } else {
            CHECK(logits[i] == expected[i]);
        }
    }

    float unchanged[VOCAB];
    for (uint32_t i = 0u; i < (uint32_t)VOCAB; ++i) {
        unchanged[i] = (float)i;
    }
    CHECK(uocr_metal_context_apply_no_repeat_ngram_f32(ctx,
                                                       unchanged,
                                                       1u,
                                                       VOCAB,
                                                       NULL,
                                                       error,
                                                       sizeof(error)) == 1);
    for (uint32_t i = 0u; i < (uint32_t)VOCAB; ++i) {
        CHECK(unchanged[i] == (float)i);
    }

    const uocr_no_repeat_ngram_config invalid[1] = {{NULL, 5u, 3u, 0u}};
    CHECK(uocr_metal_context_apply_no_repeat_ngram_f32(ctx,
                                                       unchanged,
                                                       1u,
                                                       VOCAB,
                                                       invalid,
                                                       error,
                                                       sizeof(error)) == 0);
    CHECK(strstr(error, "failed to validate no-repeat-ngram row") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_select_greedy_with_no_repeat_f32(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, VOCAB = UOCR_VOCAB_SIZE };
    float *logits = (float *)malloc((size_t)ROWS * (size_t)VOCAB * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * VOCAB); ++i) {
        logits[i] = -1000.0f;
    }
    logits[0u * (uint32_t)VOCAB + 3u] = 100.0f;
    logits[0u * (uint32_t)VOCAB + 4u] = 90.0f;
    logits[1u * (uint32_t)VOCAB + 8u] = 50.0f;
    logits[1u * (uint32_t)VOCAB + 9u] = 40.0f;

    const int32_t sequence0[] = {1, 2, 3, 1, 2}; /* current prefix [1,2] bans 3 */
    const uocr_no_repeat_ngram_config no_repeat[ROWS] = {
        {sequence0, 5u, 3u, 0u},
        {NULL, 0u, 0u, 0u},
    };

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f};
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               ROWS,
                                               VOCAB,
                                               no_repeat,
                                               token_ids,
                                               scores,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(isinf(logits[3u]) && logits[3u] < 0.0f);
    CHECK(token_ids[0] == 4u);
    CHECK(scores[0] == 90.0f);
    CHECK(token_ids[1] == 8u);
    CHECK(scores[1] == 50.0f);

    logits[0u] = 1.0f;
    token_ids[0] = UINT32_MAX;
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               VOCAB,
                                               NULL,
                                               token_ids,
                                               NULL,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(token_ids[0] == 4u);

    const uocr_no_repeat_ngram_config invalid[1] = {{NULL, 5u, 3u, 0u}};
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               VOCAB,
                                               invalid,
                                               token_ids,
                                               scores,
                                               error,
                                               sizeof(error)) == 0);
    CHECK(strstr(error, "failed to apply no-repeat-ngram bans") != NULL);

    uocr_metal_context_destroy(ctx);
    free(logits);
    return 0;
}

static int test_metal_eos_stop_after_greedy_selection(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    float *logits = (float *)malloc((size_t)UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(logits != NULL);
    for (uint32_t i = 0u; i < UOCR_VOCAB_SIZE; ++i) {
        logits[i] = -1000.0f;
    }
    logits[(uint32_t)UOCR_TOKEN_EOS] = 5.0f;
    logits[42u] = 4.0f;

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uint32_t next_token = UINT32_MAX;
    CHECK(uocr_metal_context_select_greedy_f32(ctx,
                                               logits,
                                               1u,
                                               UOCR_VOCAB_SIZE,
                                               NULL,
                                               &next_token,
                                               NULL,
                                               error,
                                               sizeof(error)) == 1);
    CHECK(next_token == (uint32_t)UOCR_TOKEN_EOS);

    const int32_t input_ids[] = {UOCR_TOKEN_BOS, 42};
    const uint8_t image_mask[] = {0, 0};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;
    request.max_new_tokens = 4u;

    uocr_sequence_state state;
    char sequence_error[128];
    CHECK(uocr_build_sequence_state(&request, &state, sequence_error, sizeof(sequence_error)) == UOCR_OK);
    int32_t generated[4] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)next_token, generated, 4u) == UOCR_OK);
    CHECK(generated[0] == UOCR_TOKEN_EOS);
    CHECK(state.generated_count == 1u);
    CHECK(state.eos == 1);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    uocr_metal_context_destroy(ctx);
    free(logits);
    return 0;
}

static int test_metal_select_next_token_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 2, HIDDEN = UOCR_HIDDEN_SIZE, ROW_COUNT = 3 };
    const uint32_t rows[ROW_COUNT] = {3u, 4u, (uint32_t)UOCR_TOKEN_EOS};
    uint16_t final_norm[HIDDEN];
    uint16_t hidden[ROWS * HIDDEN];
    uint16_t row_weights[ROW_COUNT * HIDDEN];
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        final_norm[i] = 0x3c00u; /* 1.0 */
    }
    memset(hidden, 0, sizeof(hidden));
    memset(row_weights, 0, sizeof(row_weights));
    hidden[0u] = 0x3c00u;                    /* row 0 selects between token 3 and 4. */
    hidden[(uint32_t)HIDDEN + 1u] = 0x3c00u; /* row 1 selects EOS. */
    row_weights[0u * (uint32_t)HIDDEN + 0u] = 0x3800u; /* token 3: 0.5 * normed[0] */
    row_weights[1u * (uint32_t)HIDDEN + 0u] = 0x3c00u; /* token 4: 1.0 * normed[0], then banned */
    row_weights[2u * (uint32_t)HIDDEN + 1u] = 0x3c00u; /* EOS: 1.0 * normed[1] */

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_final_norm_and_sparse_lm_head_uocr_model(path,
                                                                   final_norm,
                                                                   rows,
                                                                   row_weights,
                                                                   ROW_COUNT) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);

    uint16_t *normed = (uint16_t *)malloc((size_t)ROWS * HIDDEN * sizeof(uint16_t));
    float *logits = (float *)malloc((size_t)ROWS * UOCR_VOCAB_SIZE * sizeof(float));
    CHECK(normed != NULL && logits != NULL);
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * UOCR_VOCAB_SIZE); ++i) {
        logits[i] = -12345.0f;
    }

    const int32_t sequence0[] = {1, 2, 4, 1, 2}; /* current prefix [1,2] bans token 4 */
    const uocr_no_repeat_ngram_config no_repeat[ROWS] = {
        {sequence0, 5u, 3u, 0u},
        {NULL, 0u, 0u, 0u},
    };
    uint32_t token_ids[ROWS] = {UINT32_MAX, UINT32_MAX};
    float scores[ROWS] = {0.0f, 0.0f};
    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   hidden,
                                                   ROWS,
                                                   no_repeat,
                                                   normed,
                                                   logits,
                                                   token_ids,
                                                   scores,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(token_ids[0] == 3u);
    CHECK(token_ids[1] == (uint32_t)UOCR_TOKEN_EOS);
    CHECK(isinf(logits[4u]) && logits[4u] < 0.0f);
    CHECK(scores[0] == logits[3u]);
    CHECK(scores[1] == logits[1u * (uint32_t)UOCR_VOCAB_SIZE + (uint32_t)UOCR_TOKEN_EOS]);
    CHECK(scores[0] > 0.0f && scores[1] > 0.0f);

    const int32_t input_ids[] = {UOCR_TOKEN_BOS, 42};
    const uint8_t image_mask[] = {0u, 0u};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 2u;
    request.max_new_tokens = 4u;

    uocr_sequence_state state;
    char sequence_error[128];
    CHECK(uocr_build_sequence_state(&request, &state, sequence_error, sizeof(sequence_error)) == UOCR_OK);
    int32_t generated[4] = {0};
    CHECK(uocr_sequence_accept_generated_token(&state, (int32_t)token_ids[1], generated, 4u) == UOCR_OK);
    CHECK(generated[0] == UOCR_TOKEN_EOS);
    CHECK(uocr_sequence_generation_done(&state) == 1);

    CHECK(uocr_metal_context_select_next_token_f16(ctx,
                                                   hidden,
                                                   ROWS,
                                                   no_repeat,
                                                   NULL,
                                                   logits,
                                                   token_ids,
                                                   scores,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal next-token selection request") != NULL);

    free(logits);
    free(normed);
    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_metal_dense_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { ROWS = 3, IN_FEATURES = 1280, OUT_FEATURES = 96 };
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint16_t weight_values[] = {
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint16_t bias_values[] = {
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t bias_value_count = (uint32_t)(sizeof(bias_values) / sizeof(bias_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)ROWS * IN_FEATURES * sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)malloc((size_t)OUT_FEATURES * IN_FEATURES * sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)malloc((size_t)OUT_FEATURES * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)ROWS * OUT_FEATURES * sizeof(uint16_t));
    CHECK(input != NULL && weight != NULL && bias != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(ROWS * IN_FEATURES); ++i) {
        input[i] = input_values[(i * 17u + i / 29u + 1u) % input_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)(OUT_FEATURES * IN_FEATURES); ++i) {
        weight[i] = weight_values[(i * 11u + i / 37u + 3u) % weight_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        bias[i] = bias_values[(i * 5u + 1u) % bias_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
        for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
            float sum = f16_bits_to_f32(bias[col]);
            for (uint32_t k = 0u; k < (uint32_t)IN_FEATURES; ++k) {
                sum += f16_bits_to_f32(input[row * (uint32_t)IN_FEATURES + k]) *
                       f16_bits_to_f32(weight[col * (uint32_t)IN_FEATURES + k]);
            }
            expected[row * (uint32_t)OUT_FEATURES + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       ROWS,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(ROWS * OUT_FEATURES); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.5e-3f);
    }

    memset(out_f16, 0, (size_t)ROWS * OUT_FEATURES * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       1u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                       out_f16,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)OUT_FEATURES; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 5.0e-2f);
    }

    memset(out_f32, 0, (size_t)ROWS * OUT_FEATURES * sizeof(float));
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       NULL,
                                       1u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)OUT_FEATURES; ++col) {
        CHECK(fabsf(out_f32[col] - (expected[col] - f16_bits_to_f32(bias[col]))) < 2.5e-3f);
    }

    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       ROWS,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       (uocr_metal_dense_output_type)99,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal dense output type") != NULL);
    CHECK(uocr_metal_context_dense_f16(ctx,
                                       input,
                                       weight,
                                       bias,
                                       0u,
                                       IN_FEATURES,
                                       OUT_FEATURES,
                                       UOCR_METAL_DENSE_OUTPUT_F32,
                                       out_f32,
                                       error,
                                       sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal dense request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_attention_qkvo_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = 1280, PROJECTIONS = 4 };
    const uint16_t input_values[] = {
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint16_t weight_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *weights[PROJECTIONS];
    float *expected[PROJECTIONS];
    float *out_f32[PROJECTIONS];
    uint16_t *out_f16[PROJECTIONS];
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        weights[p] = (uint16_t *)malloc((size_t)HIDDEN * HIDDEN * sizeof(uint16_t));
        expected[p] = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
        out_f32[p] = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
        out_f16[p] = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
        CHECK(weights[p] != NULL && expected[p] != NULL && out_f32[p] != NULL && out_f16[p] != NULL);
    }
    CHECK(input != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 19u + i / 23u + 2u) % input_value_count];
    }
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)(HIDDEN * HIDDEN); ++i) {
            weights[p][i] = weight_values[(i * (7u + p * 2u) + i / 41u + p * 3u) % weight_value_count];
        }
    }

    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                float sum = 0.0f;
                for (uint32_t k = 0u; k < (uint32_t)HIDDEN; ++k) {
                    sum += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k]) *
                           f16_bits_to_f32(weights[p][col * (uint32_t)HIDDEN + k]);
                }
                expected[p][token * (uint32_t)HIDDEN + col] = sum;
            }
        }
        memset(out_f32[p], 0, (size_t)TOKENS * HIDDEN * sizeof(float));
        memset(out_f16[p], 0, (size_t)HIDDEN * sizeof(uint16_t));
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                TOKENS,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
            CHECK(fabsf(out_f32[p][i] - expected[p][i]) < 4.0e-3f);
        }
    }

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                1u,
                                                UOCR_METAL_DENSE_OUTPUT_F16,
                                                out_f16[0],
                                                out_f16[1],
                                                out_f16[2],
                                                out_f16[3],
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
            CHECK(fabsf(f16_bits_to_f32(out_f16[p][i]) - expected[p][i]) < 1.5e-1f);
        }
    }

    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                0u,
                                                UOCR_METAL_DENSE_OUTPUT_F32,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal attention projection request") != NULL);
    CHECK(uocr_metal_context_attention_qkvo_f16(ctx,
                                                input,
                                                weights[0],
                                                weights[1],
                                                weights[2],
                                                weights[3],
                                                TOKENS,
                                                (uocr_metal_dense_output_type)99,
                                                out_f32[0],
                                                out_f32[1],
                                                out_f32[2],
                                                out_f32[3],
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal attention projection output type") != NULL);

    uocr_metal_context_destroy(ctx);
    for (uint32_t p = 0u; p < (uint32_t)PROJECTIONS; ++p) {
        free(out_f16[p]);
        free(out_f32[p]);
        free(expected[p]);
        free(weights[p]);
    }
    free(input);
    return 0;
}

static int test_metal_attention_output_residual_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = 1280 };
    const uint16_t context_values[] = {
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0xa400u, /* -0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint16_t weight_values[] = {
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0xa400u, /* -0.015625 */
        0x2800u, /* 0.03125 */
        0xa800u  /* -0.03125 */
    };
    const uint16_t residual_values[] = {
        0xb800u, /* -0.5 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u  /* 0.5 */
    };
    const uint32_t context_value_count = (uint32_t)(sizeof(context_values) / sizeof(context_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *context = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)malloc((size_t)HIDDEN * HIDDEN * sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(context != NULL && weight != NULL && residual != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        context[i] = context_values[(i * 11u + i / 17u + 3u) % context_value_count];
        residual[i] = residual_values[(i * 7u + i / 19u + 1u) % residual_value_count];
    }
    for (uint32_t i = 0u; i < (uint32_t)(HIDDEN * HIDDEN); ++i) {
        weight[i] = weight_values[(i * 13u + i / 23u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            float sum = f16_bits_to_f32(residual[token * (uint32_t)HIDDEN + col]);
            for (uint32_t k = 0u; k < (uint32_t)HIDDEN; ++k) {
                sum += f16_bits_to_f32(context[token * (uint32_t)HIDDEN + k]) *
                       f16_bits_to_f32(weight[col * (uint32_t)HIDDEN + k]);
            }
            expected[token * (uint32_t)HIDDEN + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           TOKENS,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.0e-4f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           1u,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           out_f16,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 8.0e-3f);
    }

    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           TOKENS,
                                                           (uocr_metal_dense_output_type)99,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal attention output type") != NULL);
    CHECK(uocr_metal_context_attention_output_residual_f16(ctx,
                                                           context,
                                                           weight,
                                                           residual,
                                                           0u,
                                                           UOCR_METAL_DENSE_OUTPUT_F32,
                                                           out_f32,
                                                           error,
                                                           sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal attention output request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(residual);
    free(weight);
    free(context);
    return 0;
}

static int test_metal_dense_swiglu_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_DENSE_LAYER0_INTERMEDIATE };
    const uint16_t input_values[] = {
        0xa800u, /* -0.03125 */
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint16_t weight_values[] = {
        0x0000u, /* 0.0 */
        0x2000u, /* 0.0078125 */
        0xa000u, /* -0.0078125 */
        0x2400u, /* 0.015625 */
        0xa400u  /* -0.015625 */
    };
    const uint16_t residual_values[] = {
        0xa800u, /* -0.03125 */
        0xa400u, /* -0.015625 */
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0x2800u  /* 0.03125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOKENS * INTERMEDIATE * sizeof(float));
    float *expected_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_no_residual = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && residual != NULL &&
          mid != NULL && expected_residual != NULL && expected_no_residual != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 11u + i / 29u + 2u) % input_value_count];
        residual[i] = residual_values[(i * 7u + i / 17u + 1u) % residual_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
        const uint32_t k0 = (row * 37u) % (uint32_t)HIDDEN;
        const uint32_t k1 = (row * 37u + 113u) % (uint32_t)HIDDEN;
        const uint32_t u0 = (row * 53u + 7u) % (uint32_t)HIDDEN;
        const uint32_t u1 = (row * 53u + 211u) % (uint32_t)HIDDEN;
        gate_weight[row * (uint32_t)HIDDEN + k0] = weight_values[(row + 1u) % weight_value_count];
        gate_weight[row * (uint32_t)HIDDEN + k1] = weight_values[(row * 3u + 2u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u0] = weight_values[(row * 5u + 3u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u1] = weight_values[(row * 7u + 4u) % weight_value_count];
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const uint32_t d0 = (col * 41u) % (uint32_t)INTERMEDIATE;
        const uint32_t d1 = (col * 41u + 977u) % (uint32_t)INTERMEDIATE;
        const uint32_t d2 = (col * 41u + 2222u) % (uint32_t)INTERMEDIATE;
        down_weight[col * (uint32_t)INTERMEDIATE + d0] = weight_values[(col + 2u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d1] = weight_values[(col * 3u + 1u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d2] = weight_values[(col * 5u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            float gate = 0.0f;
            float up = 0.0f;
            const uint32_t k0 = (row * 37u) % (uint32_t)HIDDEN;
            const uint32_t k1 = (row * 37u + 113u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (row * 53u + 7u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (row * 53u + 211u) % (uint32_t)HIDDEN;
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k0]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + k0]);
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + k1]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + k1]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u0]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u0]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u1]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u1]);
            const float silu = gate / (1.0f + expf(-gate));
            mid[token * (uint32_t)INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const uint32_t d0 = (col * 41u) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (col * 41u + 977u) % (uint32_t)INTERMEDIATE;
            const uint32_t d2 = (col * 41u + 2222u) % (uint32_t)INTERMEDIATE;
            float sum = 0.0f;
            sum += mid[token * (uint32_t)INTERMEDIATE + d0] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d0]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d1] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d1]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d2] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d2]);
            if (token == 0u) {
                expected_no_residual[col] = sum;
            }
            expected_residual[token * (uint32_t)HIDDEN + col] =
                sum + f16_bits_to_f32(residual[token * (uint32_t)HIDDEN + col]);
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              TOKENS,
                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_residual[i]) < 2.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              NULL,
                                              1u,
                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                              out_f16,
                                              error,
                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_no_residual[i]) < 2.0e-5f);
    }

    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              TOKENS,
                                              (uocr_metal_dense_output_type)99,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal dense SwiGLU output type") != NULL);
    CHECK(uocr_metal_context_dense_swiglu_f16(ctx,
                                              input,
                                              gate_weight,
                                              up_weight,
                                              down_weight,
                                              residual,
                                              0u,
                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                              out_f32,
                                              error,
                                              sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal dense SwiGLU request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected_no_residual);
    free(expected_residual);
    free(mid);
    free(residual);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_shared_experts_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 2, HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_SHARED_INTERMEDIATE };
    const uint16_t input_values[] = {
        0xb400u, /* -0.25 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t weight_values[] = {
        0x2800u, /* 0.03125 */
        0xa800u, /* -0.03125 */
        0x2c00u, /* 0.0625 */
        0xac00u, /* -0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOKENS * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && mid != NULL &&
          expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = input_values[(i * 13u + i / 19u + 4u) % input_value_count];
    }

    for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
        const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
        const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
        const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
        const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
        gate_weight[row * (uint32_t)HIDDEN + g0] = weight_values[(row + 1u) % weight_value_count];
        gate_weight[row * (uint32_t)HIDDEN + g1] = weight_values[(row * 3u + 2u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u0] = weight_values[(row * 5u + 3u) % weight_value_count];
        up_weight[row * (uint32_t)HIDDEN + u1] = weight_values[(row * 7u + 4u) % weight_value_count];
    }
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
        const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
        const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
        down_weight[col * (uint32_t)INTERMEDIATE + d0] = weight_values[(col + 2u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d1] = weight_values[(col * 3u + 1u) % weight_value_count];
        down_weight[col * (uint32_t)INTERMEDIATE + d2] = weight_values[(col * 5u + 4u) % weight_value_count];
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const uint32_t g0 = (row * 29u) % (uint32_t)HIDDEN;
            const uint32_t g1 = (row * 29u + 101u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (row * 43u + 11u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (row * 43u + 307u) % (uint32_t)HIDDEN;
            float gate = 0.0f;
            float up = 0.0f;
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g0]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + g0]);
            gate += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + g1]) *
                    f16_bits_to_f32(gate_weight[row * (uint32_t)HIDDEN + g1]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u0]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u0]);
            up += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + u1]) *
                  f16_bits_to_f32(up_weight[row * (uint32_t)HIDDEN + u1]);
            const float silu = gate / (1.0f + expf(-gate));
            mid[token * (uint32_t)INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const uint32_t d0 = (col * 31u) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (col * 31u + 271u) % (uint32_t)INTERMEDIATE;
            const uint32_t d2 = (col * 31u + 1009u) % (uint32_t)INTERMEDIATE;
            float sum = 0.0f;
            sum += mid[token * (uint32_t)INTERMEDIATE + d0] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d0]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d1] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d1]);
            sum += mid[token * (uint32_t)INTERMEDIATE + d2] *
                   f16_bits_to_f32(down_weight[col * (uint32_t)INTERMEDIATE + d2]);
            expected[token * (uint32_t)HIDDEN + col] = sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    TOKENS,
                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 3.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    1u,
                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                    out_f16,
                                                    error,
                                                    sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)HIDDEN; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 3.0e-4f);
    }

    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    TOKENS,
                                                    (uocr_metal_dense_output_type)99,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE shared experts output type") != NULL);
    CHECK(uocr_metal_context_moe_shared_experts_f16(ctx,
                                                    input,
                                                    gate_weight,
                                                    up_weight,
                                                    down_weight,
                                                    0u,
                                                    UOCR_METAL_DENSE_OUTPUT_F32,
                                                    out_f32,
                                                    error,
                                                    sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE shared experts request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(mid);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static void compute_router_topk_expected(const float *probs,
                                          uint32_t token,
                                          uint32_t experts,
                                          uint32_t top_k,
                                          uint32_t *top_ids,
                                          float *top_weights) {
    for (uint32_t rank = 0u; rank < top_k; ++rank) {
        float best = -INFINITY;
        uint32_t best_expert = 0u;
        for (uint32_t expert = 0u; expert < experts; ++expert) {
            int already_selected = 0;
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if (top_ids[token * top_k + prev] == expert) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected) {
                continue;
            }
            const float value = probs[token * experts + expert];
            if (value > best || (value == best && expert < best_expert)) {
                best = value;
                best_expert = expert;
            }
        }
        top_ids[token * top_k + rank] = best_expert;
        top_weights[token * top_k + rank] = best;
    }
}

static int test_metal_moe_router_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = UOCR_HIDDEN_SIZE, EXPERTS = UOCR_ROUTED_EXPERTS, TOP_K = UOCR_MOE_TOP_K };
    uint16_t *input = (uint16_t *)calloc((size_t)TOKENS * HIDDEN, sizeof(uint16_t));
    uint16_t *weight = (uint16_t *)calloc((size_t)EXPERTS * HIDDEN, sizeof(uint16_t));
    float *expected_logits = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    float *expected_probs = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    uint32_t *expected_top_ids = (uint32_t *)calloc((size_t)TOKENS * TOP_K, sizeof(uint32_t));
    float *expected_top_weights = (float *)calloc((size_t)TOKENS * TOP_K, sizeof(float));
    float *logits = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    float *probs = (float *)malloc((size_t)TOKENS * EXPERTS * sizeof(float));
    uint32_t *top_ids = (uint32_t *)malloc((size_t)TOKENS * TOP_K * sizeof(uint32_t));
    float *top_weights = (float *)malloc((size_t)TOKENS * TOP_K * sizeof(float));
    uint32_t *top_ids_optional = (uint32_t *)malloc((size_t)TOP_K * sizeof(uint32_t));
    float *top_weights_optional = (float *)malloc((size_t)TOP_K * sizeof(float));
    CHECK(input != NULL && weight != NULL && expected_logits != NULL && expected_probs != NULL &&
          expected_top_ids != NULL && expected_top_weights != NULL && logits != NULL && probs != NULL &&
          top_ids != NULL && top_weights != NULL && top_ids_optional != NULL && top_weights_optional != NULL);

    const float token_values[TOKENS][8] = {
        {0.50f, -0.25f, 0.125f, 1.00f, -0.50f, 0.25f, 0.75f, -0.125f},
        {-0.75f, 0.50f, -0.125f, 0.25f, 0.625f, -0.375f, 0.125f, 0.875f},
        {0.25f, 0.75f, 0.50f, -0.50f, 0.375f, 0.125f, -0.625f, 0.25f},
    };
    const uint32_t feature_cols[8] = {0u, 1u, 2u, 3u, 17u, 113u, 509u, 1021u};
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t i = 0u; i < 8u; ++i) {
            input[token * (uint32_t)HIDDEN + feature_cols[i]] = f32_to_f16_bits(token_values[token][i]);
        }
    }

    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        const float weights[8] = {
            ((float)expert - 31.5f) / 16.0f,
            ((float)((expert * 7u) % 17u) - 8.0f) / 32.0f,
            ((float)((expert * 11u) % 23u) - 11.0f) / 48.0f,
            ((float)((expert * 13u) % 29u) - 14.0f) / 64.0f,
            ((float)((expert * 5u) % 19u) - 9.0f) / 40.0f,
            ((float)((expert * 3u) % 31u) - 15.0f) / 80.0f,
            ((float)((expert * 17u) % 37u) - 18.0f) / 96.0f,
            ((float)((expert * 23u) % 41u) - 20.0f) / 112.0f,
        };
        for (uint32_t i = 0u; i < 8u; ++i) {
            weight[expert * (uint32_t)HIDDEN + feature_cols[i]] = f32_to_f16_bits(weights[i]);
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        float max_logit = -INFINITY;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            float sum = 0.0f;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                sum += f16_bits_to_f32(input[token * (uint32_t)HIDDEN + col]) *
                       f16_bits_to_f32(weight[expert * (uint32_t)HIDDEN + col]);
            }
            expected_logits[token * (uint32_t)EXPERTS + expert] = sum;
            if (sum > max_logit) {
                max_logit = sum;
            }
        }
        float denom = 0.0f;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            const float value = expf(expected_logits[token * (uint32_t)EXPERTS + expert] - max_logit);
            expected_probs[token * (uint32_t)EXPERTS + expert] = value;
            denom += value;
        }
        float prob_sum = 0.0f;
        for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
            expected_probs[token * (uint32_t)EXPERTS + expert] /= denom;
            prob_sum += expected_probs[token * (uint32_t)EXPERTS + expert];
        }
        CHECK(fabsf(prob_sum - 1.0f) < 2.0e-6f);
        compute_router_topk_expected(expected_probs,
                                     token,
                                     (uint32_t)EXPERTS,
                                     (uint32_t)TOP_K,
                                     expected_top_ids,
                                     expected_top_weights);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(logits, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(probs, 0, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(top_ids, 0xff, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * EXPERTS); ++i) {
        CHECK(fabsf(logits[i] - expected_logits[i]) < 2.0e-6f);
        CHECK(fabsf(probs[i] - expected_probs[i]) < 2.0e-6f);
    }
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * TOP_K); ++i) {
        CHECK(top_ids[i] == expected_top_ids[i]);
        CHECK(fabsf(top_weights[i] - expected_top_weights[i]) < 2.0e-6f);
    }

    memset(top_ids_optional, 0xff, (size_t)TOP_K * sizeof(uint32_t));
    memset(top_weights_optional, 0, (size_t)TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            1u,
                                            NULL,
                                            NULL,
                                            top_ids_optional,
                                            top_weights_optional,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)TOP_K; ++i) {
        CHECK(top_ids_optional[i] == expected_top_ids[i]);
        CHECK(fabsf(top_weights_optional[i] - expected_top_weights[i]) < 2.0e-6f);
    }

    memset(weight, 0, (size_t)EXPERTS * HIDDEN * sizeof(uint16_t));
    memset(logits, 1, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(probs, 1, (size_t)TOKENS * EXPERTS * sizeof(float));
    memset(top_ids, 0xff, (size_t)TOKENS * TOP_K * sizeof(uint32_t));
    memset(top_weights, 0, (size_t)TOKENS * TOP_K * sizeof(float));
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * EXPERTS); ++i) {
        CHECK(fabsf(logits[i]) < 2.0e-7f);
        CHECK(fabsf(probs[i] - (1.0f / (float)EXPERTS)) < 2.0e-7f);
    }
    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            CHECK(top_ids[token * (uint32_t)TOP_K + rank] == rank);
            CHECK(fabsf(top_weights[token * (uint32_t)TOP_K + rank] - (1.0f / (float)EXPERTS)) < 2.0e-7f);
        }
    }

    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            TOKENS,
                                            logits,
                                            probs,
                                            NULL,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE router request") != NULL);
    CHECK(uocr_metal_context_moe_router_f16(ctx,
                                            input,
                                            weight,
                                            0u,
                                            logits,
                                            probs,
                                            top_ids,
                                            top_weights,
                                            error,
                                            sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE router request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(top_weights_optional);
    free(top_ids_optional);
    free(top_weights);
    free(top_ids);
    free(probs);
    free(logits);
    free(expected_top_weights);
    free(expected_top_ids);
    free(expected_probs);
    free(expected_logits);
    free(weight);
    free(input);
    return 0;
}

static int test_metal_moe_selected_experts_decode_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { HIDDEN = UOCR_HIDDEN_SIZE, INTERMEDIATE = UOCR_MOE_EXPERT_INTERMEDIATE, TOP_K = UOCR_MOE_TOP_K };
    const uint32_t top_ids[TOP_K] = {7u, 2u, 63u, 0u, 5u, 11u};
    const float top_weights[TOP_K] = {0.375f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f};
    const uint16_t input_values[] = {
        0xb000u, /* -0.125 */
        0xa800u, /* -0.03125 */
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x3000u  /* 0.125 */
    };
    const uint16_t weight_values[] = {
        0x2000u, /* 0.0078125 */
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0xa000u, /* -0.0078125 */
        0xa400u, /* -0.015625 */
        0xa800u  /* -0.03125 */
    };
    const uint32_t input_value_count = (uint32_t)(sizeof(input_values) / sizeof(input_values[0]));
    const uint32_t weight_value_count = (uint32_t)(sizeof(weight_values) / sizeof(weight_values[0]));

    uint16_t *input = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    uint16_t *gate_weight = (uint16_t *)calloc((size_t)TOP_K * INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *up_weight = (uint16_t *)calloc((size_t)TOP_K * INTERMEDIATE * HIDDEN, sizeof(uint16_t));
    uint16_t *down_weight = (uint16_t *)calloc((size_t)TOP_K * HIDDEN * INTERMEDIATE, sizeof(uint16_t));
    float *mid = (float *)malloc((size_t)TOP_K * INTERMEDIATE * sizeof(float));
    float *expected = (float *)malloc((size_t)HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)HIDDEN * sizeof(uint16_t));
    CHECK(input != NULL && gate_weight != NULL && up_weight != NULL && down_weight != NULL && mid != NULL &&
          expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        input[col] = input_values[(col * 7u + col / 23u + 3u) % input_value_count];
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            uint32_t g0 = (rank * 127u + row * 17u) % (uint32_t)HIDDEN;
            uint32_t g1 = (rank * 197u + row * 31u + 23u) % (uint32_t)HIDDEN;
            uint32_t u0 = (rank * 59u + row * 13u + 5u) % (uint32_t)HIDDEN;
            uint32_t u1 = (rank * 89u + row * 19u + 17u) % (uint32_t)HIDDEN;
            if (g1 == g0) g1 = (g1 + 1u) % (uint32_t)HIDDEN;
            if (u1 == u0) u1 = (u1 + 1u) % (uint32_t)HIDDEN;
            const size_t base = ((size_t)rank * INTERMEDIATE + row) * HIDDEN;
            gate_weight[base + g0] = weight_values[(rank + row + 1u) % weight_value_count];
            gate_weight[base + g1] = weight_values[(rank * 3u + row * 5u + 2u) % weight_value_count];
            up_weight[base + u0] = weight_values[(rank * 5u + row * 7u + 3u) % weight_value_count];
            up_weight[base + u1] = weight_values[(rank * 11u + row * 13u + 4u) % weight_value_count];
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            uint32_t d0 = (rank * 173u + col * 7u) % (uint32_t)INTERMEDIATE;
            uint32_t d1 = (rank * 181u + col * 11u + 37u) % (uint32_t)INTERMEDIATE;
            uint32_t d2 = (rank * 191u + col * 13u + 101u) % (uint32_t)INTERMEDIATE;
            if (d1 == d0) d1 = (d1 + 1u) % (uint32_t)INTERMEDIATE;
            if (d2 == d0 || d2 == d1) d2 = (d2 + 2u) % (uint32_t)INTERMEDIATE;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            down_weight[base + d0] = weight_values[(rank + col + 2u) % weight_value_count];
            down_weight[base + d1] = weight_values[(rank * 3u + col * 5u + 1u) % weight_value_count];
            down_weight[base + d2] = weight_values[(rank * 7u + col * 11u + 5u) % weight_value_count];
        }
    }

    for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            float gate = 0.0f;
            float up = 0.0f;
            const size_t base = ((size_t)rank * INTERMEDIATE + row) * HIDDEN;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[col]);
                gate += x * f16_bits_to_f32(gate_weight[base + col]);
                up += x * f16_bits_to_f32(up_weight[base + col]);
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid[(size_t)rank * INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
        }
    }

    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        float routed_sum = 0.0f;
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            float expert_sum = 0.0f;
            const size_t base = ((size_t)rank * HIDDEN + col) * INTERMEDIATE;
            for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                expert_sum += mid[(size_t)rank * INTERMEDIATE + row] * f16_bits_to_f32(down_weight[base + row]);
            }
            routed_sum += expert_sum * top_weights[rank];
        }
        expected[col] = routed_sum;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(out_f32[col] - expected[col]) < 3.0e-5f);
    }

    memset(out_f16, 0, (size_t)HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             out_f16,
                                                             error,
                                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[col]) - expected[col]) < 3.0e-4f);
    }

    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             top_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             (uocr_metal_dense_output_type)99,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE selected-expert output type") != NULL);

    uint32_t bad_ids[TOP_K] = {7u, 2u, UOCR_ROUTED_EXPERTS, 0u, 5u, 11u};
    CHECK(uocr_metal_context_moe_selected_experts_decode_f16(ctx,
                                                             input,
                                                             bad_ids,
                                                             top_weights,
                                                             gate_weight,
                                                             up_weight,
                                                             down_weight,
                                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                                             out_f32,
                                                             error,
                                                             sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE selected expert id") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(mid);
    free(down_weight);
    free(up_weight);
    free(gate_weight);
    free(input);
    return 0;
}

static int test_metal_moe_selected_experts_prefill_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = 8, INTERMEDIATE = 5, EXPERTS = 4, TOP_K = 2 };
    uint16_t input[TOKENS * HIDDEN];
    uint32_t top_ids[TOKENS * TOP_K] = {
        0u, 2u,
        1u, 3u,
        2u, 0u,
    };
    float top_weights[TOKENS * TOP_K] = {
        0.625f, 0.25f,
        0.5f, 0.125f,
        0.375f, 0.3125f,
    };
    uint16_t gate_weight[EXPERTS * INTERMEDIATE * HIDDEN];
    uint16_t up_weight[EXPERTS * INTERMEDIATE * HIDDEN];
    uint16_t down_weight[EXPERTS * HIDDEN * INTERMEDIATE];
    float mid[TOKENS * TOP_K * INTERMEDIATE];
    float expected[TOKENS * HIDDEN];
    float out_f32[TOKENS * HIDDEN];
    uint16_t out_f16[TOKENS * HIDDEN];

    memset(gate_weight, 0, sizeof(gate_weight));
    memset(up_weight, 0, sizeof(up_weight));
    memset(down_weight, 0, sizeof(down_weight));
    const float input_values[] = {-0.5f, -0.25f, 0.0f, 0.125f, 0.25f, 0.5f, 0.75f};
    const float weight_values[] = {-0.375f, -0.125f, 0.0625f, 0.125f, 0.25f, 0.375f};
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        input[i] = f32_to_f16_bits(input_values[(i * 5u + 2u) % (sizeof(input_values) / sizeof(input_values[0]))]);
    }

    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
            const size_t base = ((size_t)expert * INTERMEDIATE + row) * HIDDEN;
            const uint32_t g0 = (expert + row * 2u) % (uint32_t)HIDDEN;
            const uint32_t g1 = (expert * 3u + row + 1u) % (uint32_t)HIDDEN;
            const uint32_t u0 = (expert * 5u + row + 2u) % (uint32_t)HIDDEN;
            const uint32_t u1 = (expert + row * 3u + 4u) % (uint32_t)HIDDEN;
            gate_weight[base + g0] = f32_to_f16_bits(weight_values[(expert + row + 1u) % 6u]);
            gate_weight[base + g1] = f32_to_f16_bits(weight_values[(expert * 2u + row + 3u) % 6u]);
            up_weight[base + u0] = f32_to_f16_bits(weight_values[(expert * 3u + row + 2u) % 6u]);
            up_weight[base + u1] = f32_to_f16_bits(weight_values[(expert + row * 2u + 5u) % 6u]);
        }
    }
    for (uint32_t expert = 0u; expert < (uint32_t)EXPERTS; ++expert) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            const size_t base = ((size_t)expert * HIDDEN + col) * INTERMEDIATE;
            const uint32_t d0 = (expert + col) % (uint32_t)INTERMEDIATE;
            const uint32_t d1 = (expert * 2u + col + 1u) % (uint32_t)INTERMEDIATE;
            down_weight[base + d0] = f32_to_f16_bits(weight_values[(expert + col + 2u) % 6u]);
            down_weight[base + d1] = f32_to_f16_bits(weight_values[(expert * 3u + col + 4u) % 6u]);
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
            const uint32_t expert = top_ids[token * (uint32_t)TOP_K + rank];
            for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                float gate = 0.0f;
                float up = 0.0f;
                const size_t weight_base = ((size_t)expert * INTERMEDIATE + row) * HIDDEN;
                const size_t input_base = (size_t)token * HIDDEN;
                for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                    const float x = f16_bits_to_f32(input[input_base + col]);
                    gate += x * f16_bits_to_f32(gate_weight[weight_base + col]);
                    up += x * f16_bits_to_f32(up_weight[weight_base + col]);
                }
                const float silu = gate / (1.0f + expf(-gate));
                mid[((size_t)token * TOP_K + rank) * INTERMEDIATE + row] = f16_bits_to_f32(f32_to_f16_bits(silu * up));
            }
        }
    }

    for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            float routed_sum = 0.0f;
            for (uint32_t rank = 0u; rank < (uint32_t)TOP_K; ++rank) {
                const uint32_t expert = top_ids[token * (uint32_t)TOP_K + rank];
                float expert_sum = 0.0f;
                const size_t mid_base = ((size_t)token * TOP_K + rank) * INTERMEDIATE;
                const size_t weight_base = ((size_t)expert * HIDDEN + col) * INTERMEDIATE;
                for (uint32_t row = 0u; row < (uint32_t)INTERMEDIATE; ++row) {
                    expert_sum += mid[mid_base + row] * f16_bits_to_f32(down_weight[weight_base + row]);
                }
                routed_sum += expert_sum * top_weights[token * (uint32_t)TOP_K + rank];
            }
            expected[token * (uint32_t)HIDDEN + col] = routed_sum;
        }
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, sizeof(out_f32));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 2.0e-5f);
    }

    memset(out_f16, 0, sizeof(out_f16));
    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                                              out_f16,
                                                              error,
                                                              sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 3.0e-4f);
    }

    uint32_t bad_ids[TOKENS * TOP_K];
    memcpy(bad_ids, top_ids, sizeof(bad_ids));
    bad_ids[3] = EXPERTS;
    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              bad_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              UOCR_METAL_DENSE_OUTPUT_F32,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE prefill expert id") != NULL);

    CHECK(uocr_metal_context_moe_selected_experts_prefill_f16(ctx,
                                                              input,
                                                              top_ids,
                                                              top_weights,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              TOKENS,
                                                              HIDDEN,
                                                              INTERMEDIATE,
                                                              EXPERTS,
                                                              TOP_K,
                                                              (uocr_metal_dense_output_type)99,
                                                              out_f32,
                                                              error,
                                                              sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE selected-expert prefill output type") != NULL);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_moe_combine_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 3, HIDDEN = UOCR_HIDDEN_SIZE };
    const uint16_t routed_values[] = {
        0xb800u, /* -0.5 */
        0xb000u, /* -0.125 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3800u  /* 0.5 */
    };
    const uint16_t shared_values[] = {
        0xb400u, /* -0.25 */
        0xa800u, /* -0.03125 */
        0x2800u, /* 0.03125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t residual_values[] = {
        0xbc00u, /* -1.0 */
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t routed_value_count = (uint32_t)(sizeof(routed_values) / sizeof(routed_values[0]));
    const uint32_t shared_value_count = (uint32_t)(sizeof(shared_values) / sizeof(shared_values[0]));
    const uint32_t residual_value_count = (uint32_t)(sizeof(residual_values) / sizeof(residual_values[0]));

    uint16_t *routed = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *shared = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *residual = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_no_residual = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(routed != NULL && shared != NULL && residual != NULL && expected_residual != NULL &&
          expected_no_residual != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        routed[i] = routed_values[(i * 5u + i / 17u + 1u) % routed_value_count];
        shared[i] = shared_values[(i * 7u + i / 23u + 2u) % shared_value_count];
        residual[i] = residual_values[(i * 11u + i / 31u + 3u) % residual_value_count];
        expected_no_residual[i] = f16_bits_to_f32(routed[i]) + f16_bits_to_f32(shared[i]);
        expected_residual[i] = expected_no_residual[i] + f16_bits_to_f32(residual[i]);
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             TOKENS,
                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected_residual[i]) < 1.0e-7f);
    }

    memset(out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             NULL,
                                             TOKENS,
                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                             out_f16,
                                             error,
                                             sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected_no_residual[i]) < 8.0e-4f);
    }

    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             TOKENS,
                                             (uocr_metal_dense_output_type)99,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal MoE combine output type") != NULL);
    CHECK(uocr_metal_context_moe_combine_f16(ctx,
                                             routed,
                                             shared,
                                             residual,
                                             0u,
                                             UOCR_METAL_DENSE_OUTPUT_F32,
                                             out_f32,
                                             error,
                                             sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal MoE combine request") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected_no_residual);
    free(expected_residual);
    free(residual);
    free(shared);
    free(routed);
    return 0;
}

static void compute_rope_expected(const uint16_t *src,
                                  uint32_t n_tokens,
                                  uint32_t position_start,
                                  float *out) {
    enum { HEADS = 10, HEAD_DIM = 128, HALF_DIM = 64 };
    const float freq_scale = -2.0f * log2f(10000.0f) / (float)HEAD_DIM;
    for (uint32_t token = 0u; token < n_tokens; ++token) {
        const float position = (float)(position_start + token);
        for (uint32_t head = 0u; head < (uint32_t)HEADS; ++head) {
            const uint32_t base = (token * (uint32_t)HEADS + head) * (uint32_t)HEAD_DIM;
            for (uint32_t pair = 0u; pair < (uint32_t)HALF_DIM; ++pair) {
                const uint32_t a = pair;
                const uint32_t b = pair + (uint32_t)HALF_DIM;
                const float angle = position * exp2f((float)pair * freq_scale);
                const float c = cosf(angle);
                const float s = sinf(angle);
                const float x0 = f16_bits_to_f32(src[base + a]);
                const float x1 = f16_bits_to_f32(src[base + b]);
                out[base + a] = x0 * c - x1 * s;
                out[base + b] = x0 * s + x1 * c;
            }
        }
    }
}

static int test_metal_rope_qk_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 4, HIDDEN = 1280 };
    const uint16_t values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0xb400u, /* -0.25 */
        0x0000u, /* 0.0 */
        0x3000u, /* 0.125 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t value_count = (uint32_t)(sizeof(values) / sizeof(values[0]));
    uint16_t *q = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected_q = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *expected_k = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *q_out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *k_out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *q_out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k_out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && expected_q != NULL && expected_k != NULL &&
          q_out_f32 != NULL && k_out_f32 != NULL && q_out_f16 != NULL && k_out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        q[i] = values[(i * 17u + i / 31u + 1u) % value_count];
        k[i] = values[(i * 23u + i / 19u + 5u) % value_count];
    }

    const uint32_t position_start = 7u;
    compute_rope_expected(q, TOKENS, position_start, expected_q);
    compute_rope_expected(k, TOKENS, position_start, expected_k);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(q_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    memset(k_out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(q_out_f32[i] - expected_q[i]) < 2.0e-4f);
        CHECK(fabsf(k_out_f32[i] - expected_k[i]) < 2.0e-4f);
    }

    memset(q_out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    memset(k_out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                         q_out_f16,
                                         k_out_f16,
                                         error,
                                         sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(q_out_f16[i]) - expected_q[i]) < 4.0e-3f);
        CHECK(fabsf(f16_bits_to_f32(k_out_f16[i]) - expected_k[i]) < 4.0e-3f);
    }

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         0u,
                                         position_start,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal RoPE request") != NULL);

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         2u,
                                         32767u,
                                         UOCR_METAL_DENSE_OUTPUT_F32,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max positions") != NULL);

    CHECK(uocr_metal_context_rope_qk_f16(ctx,
                                         q,
                                         k,
                                         TOKENS,
                                         position_start,
                                         (uocr_metal_dense_output_type)99,
                                         q_out_f32,
                                         k_out_f32,
                                         error,
                                         sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal RoPE output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(k_out_f16);
    free(q_out_f16);
    free(k_out_f32);
    free(q_out_f32);
    free(expected_k);
    free(expected_q);
    free(k);
    free(q);
    return 0;
}

static int compute_prefill_attention_expected(const uint16_t *q,
                                              const uint16_t *k,
                                              const uint16_t *v,
                                              uint32_t n_tokens,
                                              float *out) {
    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    float scores[16];
    if (n_tokens > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
        return 0;
    }
    for (uint32_t query = 0u; query < n_tokens; ++query) {
        for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
            float max_score = -3.4028234663852886e38f;
            for (uint32_t key = 0u; key <= query; ++key) {
                float score = 0.0f;
                const uint32_t q_base = (query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                const uint32_t k_base = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                    score += f16_bits_to_f32(q[q_base + dim]) * f16_bits_to_f32(k[k_base + dim]);
                }
                score *= scale;
                scores[key] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float denominator = 0.0f;
            for (uint32_t key = 0u; key <= query; ++key) {
                scores[key] = expf(scores[key] - max_score);
                denominator += scores[key];
            }
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                float value = 0.0f;
                for (uint32_t key = 0u; key <= query; ++key) {
                    const uint32_t v_index = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim;
                    value += (scores[key] / denominator) * f16_bits_to_f32(v[v_index]);
                }
                out[(query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim] = value;
            }
        }
    }
    return 1;
}

static int compute_prefill_attention_varlen_expected(const uint16_t *q,
                                                     const uint16_t *k,
                                                     const uint16_t *v,
                                                     const uint32_t *cu,
                                                     uint32_t batch,
                                                     uint32_t total_tokens,
                                                     float *out) {
    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    float scores[16];
    for (uint32_t b = 0u; b < batch; ++b) {
        const uint32_t start = cu[b];
        const uint32_t end = cu[b + 1u];
        if (end <= start || end > total_tokens || end - start > (uint32_t)(sizeof(scores) / sizeof(scores[0]))) {
            return 0;
        }
        for (uint32_t query = start; query < end; ++query) {
            for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
                float max_score = -3.4028234663852886e38f;
                for (uint32_t key = start; key <= query; ++key) {
                    float score = 0.0f;
                    const uint32_t q_base = (query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                    const uint32_t k_base = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM;
                    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                        score += f16_bits_to_f32(q[q_base + dim]) * f16_bits_to_f32(k[k_base + dim]);
                    }
                    score *= scale;
                    scores[key - start] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                float denominator = 0.0f;
                for (uint32_t key = start; key <= query; ++key) {
                    scores[key - start] = expf(scores[key - start] - max_score);
                    denominator += scores[key - start];
                }
                for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                    float value = 0.0f;
                    for (uint32_t key = start; key <= query; ++key) {
                        const uint32_t v_index = (key * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim;
                        value += (scores[key - start] / denominator) * f16_bits_to_f32(v[v_index]);
                    }
                    out[(query * UOCR_ATTENTION_HEADS + head) * UOCR_HEAD_DIM + dim] = value;
                }
            }
        }
    }
    return 1;
}

static int test_metal_prefill_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { TOKENS = 5, HIDDEN = UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM };
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *v = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && v != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        q[i] = q_values[(i * 7u + i / 19u + 1u) % q_count];
        k[i] = k_values[(i * 11u + i / 23u + 3u) % k_count];
        v[i] = v_values[(i * 13u + i / 29u + 5u) % v_count];
    }
    /* Make future tokens conspicuous so query-token zero verifies the causal
     * mask cannot leak later values.
     */
    for (uint32_t token = 1u; token < (uint32_t)TOKENS; ++token) {
        v[token * (uint32_t)HIDDEN] = 0x5000u; /* 32.0 */
    }

    CHECK(compute_prefill_attention_expected(q, k, v, TOKENS, expected) == 1);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 8.0e-4f);
    }
    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
        CHECK(out_f32[dim] == f16_bits_to_f32(v[dim]));
    }

    memset(out_f16, 0, (size_t)TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   UOCR_METAL_DENSE_OUTPUT_F16,
                                                   out_f16,
                                                   error,
                                                   sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   0u,
                                                   UOCR_METAL_DENSE_OUTPUT_F32,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal prefill attention request") != NULL);
    CHECK(uocr_metal_context_prefill_attention_f16(ctx,
                                                   q,
                                                   k,
                                                   v,
                                                   TOKENS,
                                                   (uocr_metal_dense_output_type)99,
                                                   out_f32,
                                                   error,
                                                   sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal prefill attention output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v);
    free(k);
    free(q);
    return 0;
}

static int test_metal_prefill_attention_varlen_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum { BATCH = 2, TOTAL_TOKENS = 7, MAX_SEQLEN = 4, HIDDEN = UOCR_ATTENTION_HEADS * UOCR_HEAD_DIM };
    const uint32_t cu[BATCH + 1] = {0u, 3u, 7u};
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u, /* -0.125 */
        0x3400u  /* 0.25 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *k = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    uint16_t *v = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(q != NULL && k != NULL && v != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        q[i] = q_values[(i * 7u + i / 19u + 2u) % q_count];
        k[i] = k_values[(i * 11u + i / 23u + 4u) % k_count];
        v[i] = v_values[(i * 13u + i / 29u + 1u) % v_count];
    }
    /* Make cross-sequence leakage obvious: first token of sequence 1 must only
     * see itself, never the large values placed at the end of sequence 0.
     */
    v[2u * (uint32_t)HIDDEN] = 0x5000u; /* 32.0 */
    v[3u * (uint32_t)HIDDEN] = 0xb800u; /* -0.5 */
    CHECK(compute_prefill_attention_varlen_expected(q, k, v, cu, BATCH, TOTAL_TOKENS, expected) == 1);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    memset(out_f32, 0, (size_t)TOTAL_TOKENS * HIDDEN * sizeof(float));
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-3f);
    }
    for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
        CHECK(out_f32[dim] == f16_bits_to_f32(v[dim]));
        CHECK(out_f32[3u * (uint32_t)HIDDEN + dim] == f16_bits_to_f32(v[3u * (uint32_t)HIDDEN + dim]));
    }

    memset(out_f16, 0, (size_t)TOTAL_TOKENS * HIDDEN * sizeof(uint16_t));
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F16,
                                                          out_f16,
                                                          error,
                                                          sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint32_t i = 0u; i < (uint32_t)(TOTAL_TOKENS * HIDDEN); ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    const uint32_t bad_cu_order[BATCH + 1] = {0u, 5u, 4u};
    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          bad_cu_order,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          MAX_SEQLEN,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 0);
    CHECK(strstr(error, "sequence metadata") != NULL || strstr(error, "cu_seqlens") != NULL);

    CHECK(uocr_metal_context_prefill_attention_varlen_f16(ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          cu,
                                                          BATCH,
                                                          TOTAL_TOKENS,
                                                          2u,
                                                          UOCR_METAL_DENSE_OUTPUT_F32,
                                                          out_f32,
                                                          error,
                                                          sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max_seqlen") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v);
    free(k);
    free(q);
    return 0;
}

static uint16_t kv_cache_sentinel(uint64_t index, uint32_t stream) {
    return (uint16_t)(0x1800u + ((index * 17u + (uint64_t)stream * 97u) & 0x07ffu));
}

static uint16_t kv_source_value(uint64_t index, uint32_t stream) {
    return (uint16_t)(0x3000u + ((index * 23u + (uint64_t)stream * 211u) & 0x0fffu));
}

static uint32_t kv_cache_token_for_position(uint32_t position, uint32_t prompt_length) {
    if (position < prompt_length) {
        return position;
    }
    return prompt_length + ((position - prompt_length) % UOCR_GENERATED_RING_WINDOW);
}

static uint64_t kv_cache_flat_index(uint32_t batch_slots,
                                    uint32_t cache_token_capacity,
                                    uint32_t layer,
                                    uint32_t slot,
                                    uint32_t cache_token,
                                    uint32_t head,
                                    uint32_t dim) {
    return (((uint64_t)layer * (uint64_t)batch_slots + (uint64_t)slot) *
                (uint64_t)cache_token_capacity + (uint64_t)cache_token) *
               (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM +
           (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim;
}

static int compute_decode_attention_expected(const uint16_t *q,
                                             const uint16_t *k_cache,
                                             const uint16_t *v_cache,
                                             uint32_t batch_slots,
                                             uint32_t prompt_token_capacity,
                                             uint32_t layer,
                                             uint32_t slot,
                                             uint32_t prompt_length,
                                             uint32_t generated_count,
                                             float *out) {
    uint32_t attention_length = 0u;
    if (!uocr_metal_kv_cache_attention_length(prompt_length, generated_count, &attention_length)) {
        return 0;
    }
    const uint32_t cache_token_capacity = prompt_token_capacity + UOCR_GENERATED_RING_WINDOW;
    float *scores = (float *)malloc((size_t)attention_length * sizeof(float));
    if (scores == NULL) {
        return 0;
    }

    const float scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
    for (uint32_t head = 0u; head < UOCR_ATTENTION_HEADS; ++head) {
        float max_score = -3.4028234663852886e38f;
        for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
            uint32_t cache_token = UINT32_MAX;
            if (!uocr_metal_kv_cache_token_for_attention_index(prompt_length,
                                                               prompt_token_capacity,
                                                               generated_count,
                                                               attention_index,
                                                               &cache_token)) {
                free(scores);
                return 0;
            }
            float score = 0.0f;
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t q_index = (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim;
                const uint64_t k_index = kv_cache_flat_index(batch_slots,
                                                             cache_token_capacity,
                                                             layer,
                                                             slot,
                                                             cache_token,
                                                             head,
                                                             dim);
                score += f16_bits_to_f32(q[q_index]) * f16_bits_to_f32(k_cache[k_index]);
            }
            score *= scale;
            scores[attention_index] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        float denominator = 0.0f;
        for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
            scores[attention_index] = expf(scores[attention_index] - max_score);
            denominator += scores[attention_index];
        }
        for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
            float value = 0.0f;
            for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
                uint32_t cache_token = UINT32_MAX;
                if (!uocr_metal_kv_cache_token_for_attention_index(prompt_length,
                                                                   prompt_token_capacity,
                                                                   generated_count,
                                                                   attention_index,
                                                                   &cache_token)) {
                    free(scores);
                    return 0;
                }
                const uint64_t v_index = kv_cache_flat_index(batch_slots,
                                                             cache_token_capacity,
                                                             layer,
                                                             slot,
                                                             cache_token,
                                                             head,
                                                             dim);
                value += (scores[attention_index] / denominator) * f16_bits_to_f32(v_cache[v_index]);
            }
            out[(uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim] = value;
        }
    }

    free(scores);
    return 1;
}

static int test_metal_kv_cache_layout_helpers(void) {
    uint32_t cache_token = UINT32_MAX;
    uint32_t attention_length = 0u;
    uint64_t offset = 0u;
    uocr_metal_decode_attention_plan decode_plan;

    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 6u, &cache_token) == 1);
    CHECK(cache_token == 6u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u + UOCR_GENERATED_RING_WINDOW - 1u, &cache_token) == 1);
    CHECK(cache_token == 7u + UOCR_GENERATED_RING_WINDOW - 1u);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, 7u + UOCR_GENERATED_RING_WINDOW, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_position(0u, 16u, 0u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_position(17u, 16u, 0u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_position(7u, 16u, UOCR_MAX_POSITIONS, &cache_token) == 0);

    CHECK(uocr_metal_kv_cache_attention_length(7u, 0u, &attention_length) == 1);
    CHECK(attention_length == 7u);
    CHECK(uocr_metal_kv_cache_attention_length(7u, 5u, &attention_length) == 1);
    CHECK(attention_length == 12u);
    CHECK(uocr_metal_kv_cache_attention_length(7u, UOCR_GENERATED_RING_WINDOW, &attention_length) == 1);
    CHECK(attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_attention_length(7u, UOCR_GENERATED_RING_WINDOW + 9u, &attention_length) == 1);
    CHECK(attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS, 0u, &attention_length) == 1);
    CHECK(attention_length == UOCR_MAX_POSITIONS);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS, 1u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS - 3u, 3u, &attention_length) == 1);
    CHECK(attention_length == UOCR_MAX_POSITIONS);
    CHECK(uocr_metal_kv_cache_attention_length(UOCR_MAX_POSITIONS - 3u, 4u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(UINT32_MAX, 0u, &attention_length) == 0);
    CHECK(uocr_metal_kv_cache_attention_length(0u, 1u, &attention_length) == 0);

    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 0u, 6u, &cache_token) == 1);
    CHECK(cache_token == 6u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 7u, &cache_token) == 1);
    CHECK(cache_token == 7u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 11u, &cache_token) == 1);
    CHECK(cache_token == 11u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_GENERATED_RING_WINDOW + 9u,
                                                        7u,
                                                        &cache_token) == 1);
    CHECK(cache_token == 16u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_GENERATED_RING_WINDOW + 9u,
                                                        7u + UOCR_GENERATED_RING_WINDOW - 1u,
                                                        &cache_token) == 1);
    CHECK(cache_token == 15u);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u, 16u, 5u, 12u, &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_token_for_attention_index(7u,
                                                        16u,
                                                        UOCR_MAX_POSITIONS,
                                                        7u,
                                                        &cache_token) == 0);

    memset(&decode_plan, 0, sizeof(decode_plan));
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u,
                                                    16u,
                                                    UOCR_GENERATED_RING_WINDOW + 9u,
                                                    &decode_plan) == 1);
    CHECK(decode_plan.prompt_length == 7u);
    CHECK(decode_plan.prompt_token_capacity == 16u);
    CHECK(decode_plan.cache_token_capacity == 16u + UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.generated_count == UOCR_GENERATED_RING_WINDOW + 9u);
    CHECK(decode_plan.live_generated == UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.first_generated_index == 9u);
    CHECK(decode_plan.first_generated_position == 16u);
    CHECK(decode_plan.query_position == 7u + UOCR_GENERATED_RING_WINDOW + 8u);
    CHECK(decode_plan.attention_length == 7u + UOCR_GENERATED_RING_WINDOW);
    CHECK(decode_plan.generated_ring_window == UOCR_GENERATED_RING_WINDOW);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 0u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 6u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 15u) == 0);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 16u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u + UOCR_GENERATED_RING_WINDOW + 8u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u + UOCR_GENERATED_RING_WINDOW + 9u) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 0u, &cache_token) == 1);
    CHECK(cache_token == 0u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan, 7u, &cache_token) == 1);
    CHECK(cache_token == 16u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan,
                                                              7u + UOCR_GENERATED_RING_WINDOW - 1u,
                                                              &cache_token) == 1);
    CHECK(cache_token == 15u);
    CHECK(uocr_metal_kv_cache_decode_attention_index_to_token(&decode_plan,
                                                              7u + UOCR_GENERATED_RING_WINDOW,
                                                              &cache_token) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u, 6u, 0u, &decode_plan) == 0);
    CHECK(uocr_metal_kv_cache_decode_attention_plan(0u, 16u, 0u, &decode_plan) == 0);

    memset(&decode_plan, 0, sizeof(decode_plan));
    CHECK(uocr_metal_kv_cache_decode_attention_plan(7u, 16u, 0u, &decode_plan) == 1);
    CHECK(decode_plan.live_generated == 0u);
    CHECK(decode_plan.attention_length == 7u);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 6u) == 1);
    CHECK(uocr_metal_kv_cache_decode_position_allowed(&decode_plan, 7u) == 0);

    uocr_metal_kv_cache_layout empty_layout;
    memset(&empty_layout, 0, sizeof(empty_layout));
    CHECK(uocr_metal_kv_cache_offset(&empty_layout, 0, 0u, 0u, 0u, 0u, 0u, &offset) == 0);

    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    uocr_metal_kv_cache_layout layout;
    memset(&layout, 0, sizeof(layout));
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 0);
    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 2u, 16u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 1);
    CHECK(layout.decoder_layers == UOCR_DECODER_LAYERS);
    CHECK(layout.batch_slots == 2u);
    CHECK(layout.prompt_token_capacity == 16u);
    CHECK(layout.cache_token_capacity == 16u + UOCR_GENERATED_RING_WINDOW);
    CHECK(layout.kv_heads == UOCR_KV_HEADS);
    CHECK(layout.head_dim == UOCR_HEAD_DIM);
    CHECK(layout.generated_ring_window == UOCR_GENERATED_RING_WINDOW);
    CHECK(layout.token_stride_bytes == (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM * 2u);
    CHECK(layout.slot_stride_bytes == (uint64_t)layout.cache_token_capacity * layout.token_stride_bytes);
    CHECK(layout.layer_stride_bytes == (uint64_t)layout.batch_slots * layout.slot_stride_bytes);
    CHECK(layout.tensor_bytes == (uint64_t)UOCR_DECODER_LAYERS * layout.layer_stride_bytes);
    CHECK(layout.k_offset_bytes == 0u);
    CHECK(layout.v_offset_bytes == layout.tensor_bytes);
    CHECK(layout.total_bytes == layout.tensor_bytes * 2u);
    CHECK(layout.total_bytes == uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_KV_CACHE));

    const uint32_t layer = 2u;
    const uint32_t slot = 1u;
    const uint32_t token = 9u;
    const uint32_t head = 4u;
    const uint32_t dim = 5u;
    const uint64_t expected_inner = (uint64_t)layer * layout.layer_stride_bytes +
                                    (uint64_t)slot * layout.slot_stride_bytes +
                                    (uint64_t)token * layout.token_stride_bytes +
                                    (uint64_t)head * (uint64_t)UOCR_HEAD_DIM * 2u +
                                    (uint64_t)dim * 2u;
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, head, dim, &offset) == 1);
    CHECK(offset == layout.k_offset_bytes + expected_inner);
    CHECK(uocr_metal_kv_cache_offset(&layout, 1, layer, slot, token, head, dim, &offset) == 1);
    CHECK(offset == layout.v_offset_bytes + expected_inner);
    CHECK(uocr_metal_kv_cache_offset(&layout,
                                     1,
                                     layout.decoder_layers - 1u,
                                     layout.batch_slots - 1u,
                                     layout.cache_token_capacity - 1u,
                                     layout.kv_heads - 1u,
                                     layout.head_dim - 1u,
                                     &offset) == 1);
    CHECK(offset == layout.total_bytes - 2u);
    CHECK(uocr_metal_kv_cache_offset(&layout, 2, layer, slot, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, UOCR_DECODER_LAYERS, slot, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, 2u, token, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, layout.cache_token_capacity, head, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, UOCR_KV_HEADS, dim, &offset) == 0);
    CHECK(uocr_metal_kv_cache_offset(&layout, 0, layer, slot, token, head, UOCR_HEAD_DIM, &offset) == 0);

    uocr_metal_context_release_runtime_arenas(ctx);
    CHECK(uocr_metal_context_get_kv_cache_layout(ctx, &layout) == 0);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_kv_cache_write_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        BATCH_SLOTS = 2,
        PROMPT_TOKEN_CAPACITY = 8,
        PROMPT_LENGTH = 5,
        CACHE_TOKEN_CAPACITY = PROMPT_TOKEN_CAPACITY + UOCR_GENERATED_RING_WINDOW,
        LAYER = 3,
        SLOT = 1,
        N_TOKENS = PROMPT_LENGTH + UOCR_GENERATED_RING_WINDOW + 3,
        HEADS = UOCR_KV_HEADS,
        HEAD_DIM = UOCR_HEAD_DIM,
        HEAD_AREA = HEADS * HEAD_DIM
    };
    const uint64_t cache_values = (uint64_t)UOCR_DECODER_LAYERS * (uint64_t)BATCH_SLOTS *
                                  (uint64_t)CACHE_TOKEN_CAPACITY * (uint64_t)HEAD_AREA;
    const uint64_t src_values = (uint64_t)N_TOKENS * (uint64_t)HEAD_AREA;

    uint16_t *k_src = (uint16_t *)malloc((size_t)src_values * sizeof(uint16_t));
    uint16_t *v_src = (uint16_t *)malloc((size_t)src_values * sizeof(uint16_t));
    uint16_t *initial_k = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *initial_v = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *k_out = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *v_out = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint32_t *last_token_for_cache_token = (uint32_t *)malloc((size_t)CACHE_TOKEN_CAPACITY * sizeof(uint32_t));
    CHECK(k_src != NULL && v_src != NULL && initial_k != NULL && initial_v != NULL &&
          k_out != NULL && v_out != NULL && last_token_for_cache_token != NULL);

    for (uint64_t i = 0u; i < src_values; ++i) {
        k_src[i] = kv_source_value(i, 0u);
        v_src[i] = kv_source_value(i, 1u);
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        initial_k[i] = kv_cache_sentinel(i, 0u);
        initial_v[i] = kv_cache_sentinel(i, 1u);
        k_out[i] = 0u;
        v_out[i] = 0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)CACHE_TOKEN_CAPACITY; ++i) {
        last_token_for_cache_token[i] = UINT32_MAX;
    }
    for (uint32_t token = 0u; token < (uint32_t)N_TOKENS; ++token) {
        const uint32_t cache_token = kv_cache_token_for_position(token, PROMPT_LENGTH);
        CHECK(cache_token < (uint32_t)CACHE_TOKEN_CAPACITY);
        last_token_for_cache_token[cache_token] = token;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                N_TOKENS,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    for (uint64_t index = 0u; index < cache_values; ++index) {
        uint64_t rem = index;
        const uint32_t dim = (uint32_t)(rem % (uint64_t)HEAD_DIM);
        rem /= (uint64_t)HEAD_DIM;
        const uint32_t head = (uint32_t)(rem % (uint64_t)HEADS);
        rem /= (uint64_t)HEADS;
        const uint32_t cache_token = (uint32_t)(rem % (uint64_t)CACHE_TOKEN_CAPACITY);
        rem /= (uint64_t)CACHE_TOKEN_CAPACITY;
        const uint32_t slot = (uint32_t)(rem % (uint64_t)BATCH_SLOTS);
        const uint32_t layer = (uint32_t)(rem / (uint64_t)BATCH_SLOTS);

        uint16_t expected_k = initial_k[index];
        uint16_t expected_v = initial_v[index];
        if (layer == (uint32_t)LAYER && slot == (uint32_t)SLOT &&
            last_token_for_cache_token[cache_token] != UINT32_MAX) {
            const uint64_t src_index = ((uint64_t)last_token_for_cache_token[cache_token] * (uint64_t)HEADS +
                                        (uint64_t)head) * (uint64_t)HEAD_DIM + (uint64_t)dim;
            expected_k = k_src[src_index];
            expected_v = v_src[src_index];
        }
        CHECK(k_out[index] == expected_k);
        CHECK(v_out[index] == expected_v);
    }

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                NULL,
                                                NULL,
                                                1u,
                                                1u,
                                                PROMPT_TOKEN_CAPACITY,
                                                0u,
                                                0u,
                                                PROMPT_LENGTH,
                                                2u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t index = 0u; index < cache_values / (uint64_t)BATCH_SLOTS; ++index) {
        uint64_t rem = index;
        const uint32_t dim = (uint32_t)(rem % (uint64_t)HEAD_DIM);
        rem /= (uint64_t)HEAD_DIM;
        const uint32_t head = (uint32_t)(rem % (uint64_t)HEADS);
        rem /= (uint64_t)HEADS;
        const uint32_t cache_token = (uint32_t)(rem % (uint64_t)CACHE_TOKEN_CAPACITY);
        rem /= (uint64_t)CACHE_TOKEN_CAPACITY;
        const uint32_t layer = (uint32_t)rem;
        const uint64_t src_index = ((uint64_t)head * (uint64_t)HEAD_DIM) + (uint64_t)dim;
        const int should_have_value = layer == 0u && cache_token == 2u;
        CHECK(k_out[index] == (should_have_value ? k_src[src_index] : 0u));
        CHECK(v_out[index] == (should_have_value ? v_src[src_index] : 0u));
    }

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                0u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal KV cache write request") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                1u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                UOCR_DECODER_LAYERS,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "layer/slot out of range") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                1u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_TOKEN_CAPACITY + 1u,
                                                0u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "invalid Metal KV cache layout") != NULL);

    CHECK(uocr_metal_context_write_kv_cache_f16(ctx,
                                                k_src,
                                                v_src,
                                                initial_k,
                                                initial_v,
                                                2u,
                                                BATCH_SLOTS,
                                                PROMPT_TOKEN_CAPACITY,
                                                LAYER,
                                                SLOT,
                                                PROMPT_LENGTH,
                                                UOCR_MAX_POSITIONS - 1u,
                                                k_out,
                                                v_out,
                                                error,
                                                sizeof(error)) == 0);
    CHECK(strstr(error, "exceeds max positions") != NULL);

    uocr_metal_context_destroy(ctx);
    free(last_token_for_cache_token);
    free(v_out);
    free(k_out);
    free(initial_v);
    free(initial_k);
    free(v_src);
    free(k_src);
    return 0;
}

static int test_metal_decode_attention_f16(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    enum {
        BATCH_SLOTS = 2,
        PROMPT_TOKEN_CAPACITY = 8,
        PROMPT_LENGTH = 5,
        CACHE_TOKEN_CAPACITY = PROMPT_TOKEN_CAPACITY + UOCR_GENERATED_RING_WINDOW,
        LAYER = 4,
        SLOT = 1,
        HEAD_AREA = UOCR_KV_HEADS * UOCR_HEAD_DIM
    };
    const uint64_t cache_values = (uint64_t)UOCR_DECODER_LAYERS * (uint64_t)BATCH_SLOTS *
                                  (uint64_t)CACHE_TOKEN_CAPACITY * (uint64_t)HEAD_AREA;
    const uint64_t out_values = (uint64_t)HEAD_AREA;
    const uint16_t q_values[] = {
        0x0000u, /* 0.0 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0x3000u, /* 0.125 */
        0xb000u  /* -0.125 */
    };
    const uint16_t k_values[] = {
        0x0000u, /* 0.0 */
        0x2400u, /* 0.015625 */
        0x2800u, /* 0.03125 */
        0x2c00u, /* 0.0625 */
        0xac00u  /* -0.0625 */
    };
    const uint16_t v_values[] = {
        0xbc00u, /* -1.0 */
        0xb800u, /* -0.5 */
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u  /* 1.0 */
    };
    const uint32_t q_count = (uint32_t)(sizeof(q_values) / sizeof(q_values[0]));
    const uint32_t k_count = (uint32_t)(sizeof(k_values) / sizeof(k_values[0]));
    const uint32_t v_count = (uint32_t)(sizeof(v_values) / sizeof(v_values[0]));

    uint16_t *q = (uint16_t *)malloc((size_t)out_values * sizeof(uint16_t));
    uint16_t *k_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *v_cache = (uint16_t *)malloc((size_t)cache_values * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)out_values * sizeof(float));
    float *out_f32 = (float *)malloc((size_t)out_values * sizeof(float));
    uint16_t *out_f16 = (uint16_t *)malloc((size_t)out_values * sizeof(uint16_t));
    CHECK(q != NULL && k_cache != NULL && v_cache != NULL && expected != NULL && out_f32 != NULL && out_f16 != NULL);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint32_t no_wrap_generated = 3u;
    uint32_t attention_length = 0u;
    CHECK(uocr_metal_kv_cache_attention_length(PROMPT_LENGTH, no_wrap_generated, &attention_length) == 1);
    for (uint64_t i = 0u; i < out_values; ++i) {
        q[i] = q_values[(i * 7u + i / 11u + 3u) % q_count];
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        k_cache[i] = 0x0000u;
        v_cache[i] = 0x5000u; /* 32.0; catches accidental attention to unused ring slots. */
    }
    for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
        uint32_t cache_token = UINT32_MAX;
        CHECK(uocr_metal_kv_cache_token_for_attention_index(PROMPT_LENGTH,
                                                            PROMPT_TOKEN_CAPACITY,
                                                            no_wrap_generated,
                                                            attention_index,
                                                            &cache_token) == 1);
        for (uint32_t head = 0u; head < UOCR_KV_HEADS; ++head) {
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t serial = ((uint64_t)attention_index * (uint64_t)HEAD_AREA +
                                         (uint64_t)head * (uint64_t)UOCR_HEAD_DIM + (uint64_t)dim);
                const uint64_t index = kv_cache_flat_index(BATCH_SLOTS,
                                                           CACHE_TOKEN_CAPACITY,
                                                           LAYER,
                                                           SLOT,
                                                           cache_token,
                                                           head,
                                                           dim);
                k_cache[index] = k_values[(serial * 11u + serial / 17u + 1u) % k_count];
                v_cache[index] = v_values[(serial * 13u + serial / 19u + 4u) % v_count];
            }
        }
    }
    CHECK(compute_decode_attention_expected(q,
                                            k_cache,
                                            v_cache,
                                            BATCH_SLOTS,
                                            PROMPT_TOKEN_CAPACITY,
                                            LAYER,
                                            SLOT,
                                            PROMPT_LENGTH,
                                            no_wrap_generated,
                                            expected) == 1);

    memset(out_f32, 0, (size_t)out_values * sizeof(float));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-3f);
    }

    memset(out_f16, 0, (size_t)out_values * sizeof(uint16_t));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F16,
                                                  out_f16,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(f16_bits_to_f32(out_f16[i]) - expected[i]) < 1.0e-2f);
    }

    const uint32_t wrapped_generated = UOCR_GENERATED_RING_WINDOW + 3u;
    CHECK(uocr_metal_kv_cache_attention_length(PROMPT_LENGTH, wrapped_generated, &attention_length) == 1);
    for (uint64_t i = 0u; i < out_values; ++i) {
        q[i] = 0x0000u;
    }
    for (uint64_t i = 0u; i < cache_values; ++i) {
        k_cache[i] = 0x0000u;
        v_cache[i] = 0x5000u; /* Must not be read from inactive layers/slots/tokens. */
    }
    for (uint32_t attention_index = 0u; attention_index < attention_length; ++attention_index) {
        uint32_t cache_token = UINT32_MAX;
        CHECK(uocr_metal_kv_cache_token_for_attention_index(PROMPT_LENGTH,
                                                            PROMPT_TOKEN_CAPACITY,
                                                            wrapped_generated,
                                                            attention_index,
                                                            &cache_token) == 1);
        const uint16_t value = attention_index < PROMPT_LENGTH ? 0x3c00u : 0xbc00u; /* +1 prompt, -1 generated. */
        for (uint32_t head = 0u; head < UOCR_KV_HEADS; ++head) {
            for (uint32_t dim = 0u; dim < UOCR_HEAD_DIM; ++dim) {
                const uint64_t index = kv_cache_flat_index(BATCH_SLOTS,
                                                           CACHE_TOKEN_CAPACITY,
                                                           LAYER,
                                                           SLOT,
                                                           cache_token,
                                                           head,
                                                           dim);
                k_cache[index] = 0x0000u;
                v_cache[index] = value;
            }
        }
    }
    CHECK(compute_decode_attention_expected(q,
                                            k_cache,
                                            v_cache,
                                            BATCH_SLOTS,
                                            PROMPT_TOKEN_CAPACITY,
                                            LAYER,
                                            SLOT,
                                            PROMPT_LENGTH,
                                            wrapped_generated,
                                            expected) == 1);
    memset(out_f32, 0, (size_t)out_values * sizeof(float));
    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  wrapped_generated,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    for (uint64_t i = 0u; i < out_values; ++i) {
        CHECK(fabsf(out_f32[i] - expected[i]) < 1.0e-5f);
    }
    CHECK(out_f32[0] > -0.98f && out_f32[0] < -0.90f);

    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  UOCR_MAX_POSITIONS,
                                                  UOCR_METAL_DENSE_OUTPUT_F32,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "generated count") != NULL);

    CHECK(uocr_metal_context_decode_attention_f16(ctx,
                                                  q,
                                                  k_cache,
                                                  v_cache,
                                                  BATCH_SLOTS,
                                                  PROMPT_TOKEN_CAPACITY,
                                                  LAYER,
                                                  SLOT,
                                                  PROMPT_LENGTH,
                                                  no_wrap_generated,
                                                  (uocr_metal_dense_output_type)99,
                                                  out_f32,
                                                  error,
                                                  sizeof(error)) == 0);
    CHECK(strstr(error, "unsupported Metal decode attention output type") != NULL);

    uocr_metal_context_destroy(ctx);
    free(out_f16);
    free(out_f32);
    free(expected);
    free(v_cache);
    free(k_cache);
    free(q);
    return 0;
}

static int test_metal_recent_decoder_primitives_stress(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const uint16_t half_pool[] = {
        0x0000u, /* 0.0 */
        0x3400u, /* 0.25 */
        0x3800u, /* 0.5 */
        0x3c00u, /* 1.0 */
        0xbc00u, /* -1.0 */
        0x4000u, /* 2.0 */
        0xc000u, /* -2.0 */
        0x4200u, /* 3.0 */
        0x5000u  /* 32.0; catches accidental fp16 RMS variance overflow */
    };
    const uint32_t half_pool_count = (uint32_t)(sizeof(half_pool) / sizeof(half_pool[0]));

    {
        enum { TABLE_ROWS = 257, HIDDEN = 1280, OUT_ROWS = 17 };
        uint16_t table[TABLE_ROWS * HIDDEN];
        uint16_t out[OUT_ROWS * HIDDEN];
        const int32_t row_ids[OUT_ROWS] = {256, 0, 128, 7, 7, 255, 1, 64, 200, 31, 16, 127, 2, 254, 3, 129, 42};
        for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * HIDDEN); ++i) {
            table[i] = half_pool[(i * 13u + i / 17u + 5u) % half_pool_count];
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_get_rows_f16(ctx,
                                              table,
                                              TABLE_ROWS,
                                              HIDDEN,
                                              row_ids,
                                              OUT_ROWS,
                                              UOCR_METAL_GET_ROWS_OUTPUT_F16,
                                              out,
                                              error,
                                              sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t row = 0u; row < (uint32_t)OUT_ROWS; ++row) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                CHECK(out[row * (uint32_t)HIDDEN + col] == table[(uint32_t)row_ids[row] * (uint32_t)HIDDEN + col]);
            }
        }
    }

    {
        enum { TABLE_ROWS = 64, HIDDEN = 1280, TOKENS = 9, IMAGE_TOKENS = 3 };
        uint16_t table[TABLE_ROWS * HIDDEN];
        uint16_t image_features[IMAGE_TOKENS * HIDDEN];
        uint16_t out[TOKENS * HIDDEN];
        const int32_t ids[TOKENS] = {0, 17, 999999, -7, 123456, 63, 4, 1, 2};
        for (uint32_t i = 0u; i < (uint32_t)(TABLE_ROWS * HIDDEN); ++i) {
            table[i] = half_pool[(i * 7u + 3u) % half_pool_count];
        }
        for (uint32_t i = 0u; i < (uint32_t)(IMAGE_TOKENS * HIDDEN); ++i) {
            image_features[i] = (uint16_t)(0x6000u + (i % 1024u));
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t token = 0u; token < (uint32_t)TOKENS; ++token) {
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                uint16_t expected = 0u;
                if (token >= 2u && token < 2u + (uint32_t)IMAGE_TOKENS) {
                    expected = image_features[(token - 2u) * (uint32_t)HIDDEN + col];
                } else {
                    expected = table[(uint32_t)ids[token] * (uint32_t)HIDDEN + col];
                }
                CHECK(out[token * (uint32_t)HIDDEN + col] == expected);
            }
        }

        const int32_t span_at_start_ids[TOKENS] = {-123, 999999, 3, 4, 5, 6, 7, 8, 9};
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     span_at_start_ids,
                                                     TOKENS,
                                                     0u,
                                                     2u,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(out[col] == image_features[col]);
            CHECK(out[(uint32_t)HIDDEN + col] == image_features[(uint32_t)HIDDEN + col]);
        }

        const int32_t span_at_end_ids[TOKENS] = {0, 1, 2, 3, 4, 5, 6, -123, 999999};
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     span_at_end_ids,
                                                     TOKENS,
                                                     7u,
                                                     2u,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            CHECK(out[7u * (uint32_t)HIDDEN + col] == image_features[col]);
            CHECK(out[8u * (uint32_t)HIDDEN + col] == image_features[(uint32_t)HIDDEN + col]);
        }

        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     0u,
                                                     0u,
                                                     NULL,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "UINT32_MAX image span start") != NULL);

        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     ids,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     NULL,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "requires image features") != NULL);

        const int32_t invalid_outside_span[TOKENS] = {0, TABLE_ROWS, 999999, -7, 123456, 2, 3, 4, 5};
        CHECK(uocr_metal_context_assemble_prompt_f16(ctx,
                                                     table,
                                                     TABLE_ROWS,
                                                     HIDDEN,
                                                     invalid_outside_span,
                                                     TOKENS,
                                                     2u,
                                                     IMAGE_TOKENS,
                                                     image_features,
                                                     out,
                                                     error,
                                                     sizeof(error)) == 0);
        CHECK(strstr(error, "outside embedding rows") != NULL);
    }

    {
        enum { ROWS = 2, HIDDEN = 1280 };
        uint16_t input[ROWS * HIDDEN];
        uint16_t weight[HIDDEN];
        float out[ROWS * HIDDEN];
        float expected[ROWS * HIDDEN];
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            input[col] = 0x5000u; /* 32.0: fp16 sum of squares would overflow. */
            weight[col] = half_pool[(col * 5u + 3u) % half_pool_count];
        }
        for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
            input[(uint32_t)HIDDEN + col] = half_pool[(col * 11u + 1u) % (half_pool_count - 1u)];
        }
        const float eps = 1.0e-6f;
        for (uint32_t row = 0u; row < (uint32_t)ROWS; ++row) {
            float sum = 0.0f;
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
                sum += x * x;
            }
            const float scale = 1.0f / sqrtf(sum / (float)HIDDEN + eps);
            for (uint32_t col = 0u; col < (uint32_t)HIDDEN; ++col) {
                const float x = f16_bits_to_f32(input[row * (uint32_t)HIDDEN + col]);
                const float w = f16_bits_to_f32(weight[col]);
                expected[row * (uint32_t)HIDDEN + col] = x * scale * w;
            }
        }
        memset(out, 0, sizeof(out));
        CHECK(uocr_metal_context_rmsnorm_f16(ctx,
                                             input,
                                             weight,
                                             ROWS,
                                             HIDDEN,
                                             eps,
                                             UOCR_METAL_RMSNORM_OUTPUT_F32,
                                             out,
                                             error,
                                             sizeof(error)) == 1);
        CHECK(error[0] == '\0');
        for (uint32_t i = 0u; i < (uint32_t)(ROWS * HIDDEN); ++i) {
            CHECK(fabsf(out[i] - expected[i]) < 3.0e-4f);
        }
    }

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_runtime_arenas(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 16u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');

    uint64_t expected_kv = 0u;
    uint64_t expected_prompt = 0u;
    uint64_t expected_decoder = 0u;
    uint64_t expected_router_topk = 0u;
    uint64_t expected_moe_intermediate = 0u;
    uint64_t expected_vision = 0u;
    uint64_t expected_logits = 0u;
    CHECK(uocr_estimate_kv_cache_bytes(1u, 16u, &expected_kv) == UOCR_OK);
    CHECK(uocr_estimate_prompt_embedding_bytes(1u, 16u, &expected_prompt) == UOCR_OK);
    CHECK(uocr_estimate_decoder_scratch_bytes(1u, 16u, &expected_decoder) == UOCR_OK);
    CHECK(uocr_estimate_moe_router_topk_bytes(1u, 16u, &expected_router_topk) == UOCR_OK);
    CHECK(uocr_estimate_moe_intermediate_bytes(1u, 16u, &expected_moe_intermediate) == UOCR_OK);
    CHECK(uocr_estimate_vision_scratch_bytes_for_rows(16u,
                                                       UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
                                                       &expected_vision) == UOCR_OK);
    CHECK(uocr_estimate_logits_readback_bytes(1u, &expected_logits) == UOCR_OK);

    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_KV_CACHE) == expected_kv);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_PROMPT_EMBEDDINGS) == expected_prompt);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_HIDDEN_PINGPONG) == expected_decoder);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_ROUTER_TOPK) == expected_router_topk);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_MOE_INTERMEDIATE) == expected_moe_intermediate);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_VISION_SCRATCH) == expected_vision);
    CHECK(uocr_metal_context_runtime_arena_capacity(ctx, UOCR_METAL_ARENA_LOGITS_READBACK) == expected_logits);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == expected_kv + expected_prompt + expected_decoder +
                                                               expected_router_topk + expected_moe_intermediate +
                                                               expected_vision + expected_logits);

    uocr_metal_context_release_runtime_arenas(ctx);
    CHECK(uocr_metal_context_total_runtime_arena_capacity(ctx) == 0u);
    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_integrated_decoder_boundary(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);

    const int32_t input_ids[3] = {UOCR_TOKEN_BOS, 42, 77};
    const uint8_t image_mask[3] = {0u, 0u, 0u};
    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 3u;
    request.max_new_tokens = 0u;
    request.slot = 0u;
    request.image_span_start = UINT32_MAX;
    request.image_span_length = 0u;

    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires allocated runtime arenas") != NULL);

    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 8u, error, sizeof(error)) == 1);
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires validated fp16 decoder tensor bindings") != NULL);
    CHECK(result.generated_count == 0u);
    CHECK(result.stopped_on_eos == 0u);
    CHECK(result.last_token_id == UINT32_MAX);

    int32_t generated[1] = {0};
    request.max_new_tokens = 1u;
    result.generated_ids = generated;
    result.generated_capacity = 1u;
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 0);
    CHECK(strstr(error, "requires validated fp16 decoder tensor bindings") != NULL);
    CHECK(result.generated_count == 0u);
    CHECK(result.last_token_id == UINT32_MAX);

    uocr_metal_context_destroy(ctx);
    return 0;
}

static int test_metal_decoder_binding_cache_full_model(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    if (model_path == NULL || model_path[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH is not set; skipping decoder binding cache validation\n");
        return 0;
    }

    enum {
        EXPECTED_DECODER_BINDINGS = 3u + (UOCR_DECODER_LAYERS * 6u) + 3u +
                                    ((UOCR_DECODER_LAYERS - 1u) *
                                     (1u + 3u + UOCR_ROUTED_EXPERTS * 3u))
    };
    const int32_t input_ids[3] = {UOCR_TOKEN_BOS, 42, 77};
    const uint8_t image_mask[3] = {0u, 0u, 0u};
    int32_t generated[2] = {0, 0};
    char error[1024];
    memset(error, 0, sizeof(error));

    uocr_model_file model;
    CHECK(uocr_model_file_open(model_path, &model, error, sizeof(error)) == 0);
    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_decoder_bindings_ready(ctx) == 1);
    CHECK(uocr_metal_context_decoder_binding_count(ctx) == EXPECTED_DECODER_BINDINGS);
    CHECK(uocr_metal_context_allocate_runtime_arenas(ctx, 1u, 8u, error, sizeof(error)) == 1);

    uocr_metal_decoder_request_f16 request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 3u;
    request.max_new_tokens = 0u;
    request.slot = 0u;
    request.image_span_start = UINT32_MAX;
    request.image_span_length = 0u;
    uocr_metal_decoder_result_f16 result;
    memset(&result, 0, sizeof(result));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count == 0u);
    uint16_t final_hidden[3u * UOCR_HIDDEN_SIZE];
    memset(final_hidden, 0, sizeof(final_hidden));
    CHECK(uocr_metal_context_read_decoder_final_hidden_f16(ctx,
                                                           0u,
                                                           3u,
                                                           final_hidden,
                                                           error,
                                                           sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    int saw_nonzero_hidden = 0;
    for (uint32_t i = 0u; i < 3u * UOCR_HIDDEN_SIZE; ++i) {
        if (final_hidden[i] != 0u) {
            saw_nonzero_hidden = 1;
            break;
        }
    }
    CHECK(saw_nonzero_hidden == 1);

    request.max_new_tokens = 2u;
    result.generated_ids = generated;
    result.generated_capacity = 2u;
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_context_generate_f16(ctx, &request, &result, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(result.generated_count >= 1u);
    CHECK(result.generated_count <= 2u);
    for (uint32_t i = 0u; i < result.generated_count; ++i) {
        CHECK(generated[i] >= 0);
        CHECK((uint32_t)generated[i] < UOCR_VOCAB_SIZE);
    }
    CHECK(result.last_token_id == (uint32_t)generated[result.generated_count - 1u]);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    return 0;
}

static int test_metal_model_mapping(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    char path[128];
    CHECK(uocr_test_make_temp_path(path, sizeof(path)) == 0);
    CHECK(uocr_test_write_two_tensor_uocr_model(path) == 0);

    char error[1024];
    memset(error, 0, sizeof(error));
    uocr_model_file model;
    CHECK(uocr_model_file_open(path, &model, error, sizeof(error)) == 0);

    uocr_metal_context *ctx = uocr_metal_context_create(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error));
    CHECK(ctx != NULL);
    CHECK(uocr_metal_context_map_model(ctx, &model, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_model_view_count(ctx) == 1u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 2u);
    CHECK(uocr_metal_context_decoder_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_decoder_bindings_ready(ctx) == 0);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_model_view_info view_info;
    memset(&view_info, 0, sizeof(view_info));
    CHECK(uocr_metal_context_get_model_view_info(ctx, 0u, &view_info) == 1);
    CHECK(view_info.file_offset <= model.tensors[0].payload_offset);
    CHECK(view_info.length >= UOCR_TENSOR_DATA_ALIGNMENT);

    uocr_metal_tensor_binding binding;
    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 1u, &binding) == 1);
    CHECK(binding.tensor_id == 1u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[0].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 16u);

    memset(&binding, 0, sizeof(binding));
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 2u, &binding) == 1);
    CHECK(binding.tensor_id == 2u);
    CHECK(binding.view_index == 0u);
    CHECK(binding.inner_offset == model.tensors[1].payload_offset - view_info.file_offset);
    CHECK(binding.payload_size == 32u);
    CHECK(uocr_metal_context_get_tensor_binding(ctx, 999u, &binding) == 0);

    CHECK(uocr_metal_context_warmup_model_views(ctx, 128u, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    CHECK(uocr_metal_context_last_warmup_bytes(ctx) == 128u);
    CHECK(uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_TRANSIENT) >= sizeof(uint32_t));

    uocr_metal_context_unmap_model(ctx);
    CHECK(uocr_metal_context_model_view_count(ctx) == 0u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_decoder_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_decoder_bindings_ready(ctx) == 0);
    CHECK(uocr_metal_context_model_view_bytes(ctx) == 0u);

    uocr_metal_context_destroy(ctx);
    uocr_model_file_close(&model);
    unlink(path);
    return 0;
}

static int test_public_engine_open_initializes_metal(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);

    uocr_memory_report report;
    memset(&report, 0, sizeof(report));
    CHECK(uocr_engine_memory_report(engine, &report) == UOCR_OK);
    CHECK(report.recommended_working_set_bytes == uocr_metal_recommended_working_set_size());
    CHECK(report.memory_budget_bytes == uocr_metal_default_memory_budget_bytes(report.recommended_working_set_bytes));
    CHECK(report.memory_budget_bytes > 0u);
    CHECK(report.category_live_bytes[UOCR_MEMORY_KV_CACHE] == report.estimated_kv_cache_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_PROMPT_EMBEDDINGS] == report.estimated_prompt_embeddings_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_VISION_SCRATCH] == report.estimated_vision_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_DECODER_SCRATCH] == report.estimated_decoder_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_MOE_SCRATCH] == report.estimated_moe_scratch_bytes);
    CHECK(report.category_live_bytes[UOCR_MEMORY_LOGITS_READBACK] == report.estimated_logits_readback_bytes);
    CHECK(report.total_live_bytes == report.estimated_kv_cache_bytes +
                                     report.estimated_prompt_embeddings_bytes +
                                     report.estimated_vision_scratch_bytes +
                                     report.estimated_decoder_scratch_bytes +
                                     report.estimated_moe_scratch_bytes +
                                     report.estimated_logits_readback_bytes);

    uocr_engine_close(engine);
    return 0;
}

static int test_public_metal_text_generation_full_model(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }
    const char *model_path = getenv("UOCR_MODEL_PATH");
    if (model_path == NULL || model_path[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH is not set; skipping public Metal text generation\n");
        return 0;
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = model_path;
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1u;
    opts.max_prompt_tokens = 8u;
    opts.max_gen_tokens = 2u;
    opts.memory_budget_bytes = UINT64_MAX;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);

    const int32_t input_ids[3] = {UOCR_TOKEN_BOS, 42, 77};
    const uint8_t image_mask[3] = {0u, 0u, 0u};
    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = 3u;
    request.crop_grid_w = 1u;
    request.crop_grid_h = 1u;
    request.max_new_tokens = 2u;

    uocr_result *result = NULL;
    CHECK(uocr_generate_prepared(engine, &request, 1u, &result) == UOCR_OK);
    CHECK(result != NULL);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);
    CHECK(uocr_result_count(result) == 1u);
    uint32_t generated_count = 0u;
    const int32_t *generated = uocr_result_tokens(result, 0u, &generated_count);
    CHECK(generated != NULL);
    CHECK(generated_count >= 1u);
    CHECK(generated_count <= 2u);
    for (uint32_t i = 0u; i < generated_count; ++i) {
        CHECK(generated[i] >= 0);
        CHECK((uint32_t)generated[i] < UOCR_VOCAB_SIZE);
    }

    uocr_result_free(result);
    uocr_engine_close(engine);
    return 0;
}

static int test_public_metal_text_generated_ids_python_dump_parity(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }
    if (!env_flag_enabled("UOCR_RUN_LARGE_TESTS")) {
        return 0;
    }

    const char *model_path = getenv("UOCR_MODEL_PATH");
    const char *dump_dir = getenv("UOCR_LAYER_DUMP_DIR");
    if (model_path == NULL || model_path[0] == '\0' || dump_dir == NULL || dump_dir[0] == '\0') {
        printf("UOCR_RUN_LARGE_TESTS=1 but UOCR_MODEL_PATH/UOCR_LAYER_DUMP_DIR are not both set; skipping public Metal text generated-id parity\n");
        return 0;
    }
    if (!fixture_binary_exists(dump_dir, "input_ids_i32.bin") ||
        !fixture_binary_exists(dump_dir, "image_mask_u8.bin") ||
        !fixture_binary_exists(dump_dir, "generated_ids_i32.bin")) {
        printf("UOCR_LAYER_DUMP_DIR has no text generated-id fixture files; skipping public Metal text generated-id parity\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    int32_t *input_ids = NULL;
    uint8_t *image_mask = NULL;
    uint32_t n_tokens = 0u;
    int32_t *expected_ids = NULL;
    uint32_t n_expected = 0u;
    CHECK(read_prepared_tokens_python_dump(dump_dir, &input_ids, &image_mask, &n_tokens, error, sizeof(error)) == 1);
    CHECK(read_text_generated_ids_python_dump(dump_dir, &expected_ids, &n_expected, error, sizeof(error)) == 1);
    CHECK(n_tokens > 0u);
    CHECK(n_expected > 0u);

    for (uint32_t i = 0u; i < n_tokens; ++i) {
        if (image_mask[i] != 0u) {
            fprintf(stderr, "public Metal text generated-id parity requires a text-only fixture; image_mask[%u]=%u\n", i, (unsigned)image_mask[i]);
            free(expected_ids);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = model_path;
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1u;
    opts.max_prompt_tokens = n_tokens;
    opts.max_gen_tokens = n_expected;
    opts.memory_budget_bytes = UINT64_MAX;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);

    uocr_prepared_request request;
    memset(&request, 0, sizeof(request));
    request.input_ids = input_ids;
    request.image_mask = image_mask;
    request.n_tokens = n_tokens;
    request.crop_grid_w = 1u;
    request.crop_grid_h = 1u;
    request.max_new_tokens = n_expected;

    uocr_result *result = NULL;
    const int status = uocr_generate_prepared(engine, &request, 1u, &result);
    if (status != UOCR_OK) {
        fprintf(stderr, "public Metal text generated-id parity failed: %s\n", uocr_last_error(engine));
        uocr_engine_close(engine);
        free(expected_ids);
        free(image_mask);
        free(input_ids);
        return 1;
    }
    CHECK(result != NULL);
    CHECK(uocr_result_count(result) == 1u);

    uint32_t n_actual = 0u;
    const int32_t *actual_ids = uocr_result_tokens(result, 0u, &n_actual);
    CHECK(actual_ids != NULL);
    if (n_actual != n_expected) {
        fprintf(stderr,
                "public Metal text generated-id count mismatch: actual=%u expected=%u\n",
                n_actual,
                n_expected);
        uocr_result_free(result);
        uocr_engine_close(engine);
        free(expected_ids);
        free(image_mask);
        free(input_ids);
        return 1;
    }
    for (uint32_t i = 0u; i < n_expected; ++i) {
        if (actual_ids[i] != expected_ids[i]) {
            fprintf(stderr,
                    "public Metal text generated-id mismatch at %u: actual=%d expected=%d\n",
                    i,
                    actual_ids[i],
                    expected_ids[i]);
            uocr_result_free(result);
            uocr_engine_close(engine);
            free(expected_ids);
            free(image_mask);
            free(input_ids);
            return 1;
        }
    }

    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);
    uocr_result_free(result);
    uocr_engine_close(engine);
    free(expected_ids);
    free(image_mask);
    free(input_ids);
    return 0;
}

int main(void) {
    CHECK(strcmp(uocr_metal_backend_name(), "metal") == 0);
    if (test_metal_smoke() != 0) return 1;
    if (test_metal_named_scratch_buffers() != 0) return 1;
    if (test_metal_sam_patch_embed_f16() != 0) return 1;
    if (test_metal_get_rows_f16() != 0) return 1;
    if (test_metal_prompt_assembly_f16() != 0) return 1;
    if (test_metal_prompt_assembly_from_mapped_model_f16() != 0) return 1;
    if (test_metal_text_prompt_embedding_full_model_parity() != 0) return 1;
    if (test_metal_text_prompt_embedding_python_dump_parity() != 0) return 1;
    if (test_metal_image_prompt_embedding_python_dump_parity() != 0) return 1;
    if (test_metal_image_prompt_decoder_smoke_python_dump() != 0) return 1;
    if (test_metal_image_decoder_layers_python_dump_parity() != 0) return 1;
    if (test_metal_image_router_topk_python_dump_parity() != 0) return 1;
    if (test_metal_image_logits_topk_python_dump_parity() != 0) return 1;
    if (test_metal_image_generated_ids_python_dump_parity() != 0) return 1;
    if (test_metal_integrated_image_embedding_python_dump_prefill() != 0) return 1;
    if (test_metal_integrated_image_embedding_generated_ids_python_dump_parity() != 0) return 1;
    if (test_metal_text_layer0_python_dump_parity() != 0) return 1;
    if (test_metal_text_layer1_python_dump_parity() != 0) return 1;
    if (test_metal_text_remaining_layers_python_dump_parity() != 0) return 1;
    if (test_metal_text_logits_topk_python_dump_parity() != 0) return 1;
    if (test_metal_text_generated_ids_python_dump_parity() != 0) return 1;
    if (test_metal_rmsnorm_f16() != 0) return 1;
    if (test_metal_final_rmsnorm_f16() != 0) return 1;
    if (test_metal_lm_head_f16() != 0) return 1;
    if (test_metal_argmax_f32() != 0) return 1;
    if (test_metal_no_repeat_ngram_gpu_f32() != 0) return 1;
    if (test_metal_select_greedy_with_no_repeat_f32() != 0) return 1;
    if (test_metal_eos_stop_after_greedy_selection() != 0) return 1;
    if (test_metal_select_next_token_f16() != 0) return 1;
    if (test_metal_dense_f16() != 0) return 1;
    if (test_metal_attention_qkvo_f16() != 0) return 1;
    if (test_metal_attention_output_residual_f16() != 0) return 1;
    if (test_metal_dense_swiglu_f16() != 0) return 1;
    if (test_metal_moe_shared_experts_f16() != 0) return 1;
    if (test_metal_moe_router_f16() != 0) return 1;
    if (test_metal_moe_selected_experts_decode_f16() != 0) return 1;
    if (test_metal_moe_selected_experts_prefill_f16() != 0) return 1;
    if (test_metal_moe_combine_f16() != 0) return 1;
    if (test_metal_rope_qk_f16() != 0) return 1;
    if (test_metal_prefill_attention_f16() != 0) return 1;
    if (test_metal_prefill_attention_varlen_f16() != 0) return 1;
    if (test_metal_kv_cache_layout_helpers() != 0) return 1;
    if (test_metal_kv_cache_write_f16() != 0) return 1;
    if (test_metal_decode_attention_f16() != 0) return 1;
    if (test_metal_recent_decoder_primitives_stress() != 0) return 1;
    if (test_metal_runtime_arenas() != 0) return 1;
    if (test_metal_integrated_decoder_boundary() != 0) return 1;
    if (test_metal_decoder_binding_cache_full_model() != 0) return 1;
    if (test_metal_model_mapping() != 0) return 1;
    if (test_public_engine_open_initializes_metal() != 0) return 1;
    if (test_public_metal_text_generation_full_model() != 0) return 1;
    if (test_public_metal_text_generated_ids_python_dump_parity() != 0) return 1;
    return 0;
}
