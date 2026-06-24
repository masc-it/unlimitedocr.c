#include "backend/metal/uocr_metal.h"

#include "core/uocr_alloc.h"
#include "model/uocr_constants.h"
#include "model/uocr_tensor_registry.h"
#include "quant/uocr_quant.h"
#include "runtime/uocr_memory.h"
#include "runtime/uocr_vision.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifndef UOCR_METAL_RUNTIME_COMPILE
#define UOCR_METAL_RUNTIME_COMPILE 1
#endif

#ifndef UOCR_DEFAULT_RESOURCE_PATH
#define UOCR_DEFAULT_RESOURCE_PATH ""
#endif

#ifndef UOCR_DEFAULT_METALLIB_PATH
#define UOCR_DEFAULT_METALLIB_PATH ""
#endif

typedef struct uocr_metal_model_view {
    id<MTLBuffer> buffer;
    uint64_t file_offset;
    uint64_t length;
} uocr_metal_model_view;

typedef struct uocr_metal_tensor_binding_internal {
    uint32_t tensor_id;
    uint32_t view_index;
    uint64_t inner_offset;
    uint64_t payload_size;
} uocr_metal_tensor_binding_internal;

#define UOCR_METAL_DECODER_REQUIRED_TENSOR_COUNT \
    (3u + (UOCR_DECODER_LAYERS * 6u) + 3u + ((UOCR_DECODER_LAYERS - 1u) * (1u + 3u + UOCR_ROUTED_EXPERTS * 3u)))
#define UOCR_METAL_SAM_REQUIRED_TENSOR_COUNT (UOCR_SAM_BLOCKS * 14u + 11u)
#define UOCR_METAL_CLIP_REQUIRED_TENSOR_COUNT (4u + UOCR_CLIP_BLOCKS * 12u)
#define UOCR_METAL_VISION_REQUIRED_TENSOR_COUNT \
    (2u + 2u + UOCR_METAL_SAM_REQUIRED_TENSOR_COUNT + UOCR_METAL_CLIP_REQUIRED_TENSOR_COUNT)
#define UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS 5u

_Static_assert(UOCR_METAL_VISION_REQUIRED_TENSOR_COUNT == 475u, "unexpected Metal vision binding count");

typedef struct uocr_metal_decoder_binding {
    uint32_t tensor_id;
    uint32_t reserved;
    id<MTLBuffer> buffer;
    NSUInteger offset;
    uint64_t payload_size;
} uocr_metal_decoder_binding;

typedef struct uocr_metal_decoder_binding_cache {
    int valid;
    uint32_t count;
    uocr_metal_decoder_binding tensors[UOCR_METAL_DECODER_REQUIRED_TENSOR_COUNT];
} uocr_metal_decoder_binding_cache;

typedef struct uocr_metal_vision_binding {
    uint32_t tensor_id;
    uint32_t reserved;
    id<MTLBuffer> buffer;
    NSUInteger offset;
    uint64_t payload_size;
} uocr_metal_vision_binding;

typedef struct uocr_metal_vision_binding_cache {
    int valid;
    uint32_t count;
    uocr_metal_vision_binding tensors[UOCR_METAL_VISION_REQUIRED_TENSOR_COUNT];
} uocr_metal_vision_binding_cache;

typedef struct uocr_metal_payload_span {
    uint32_t tensor_index;
    uint32_t tensor_id;
    uint64_t payload_start;
    uint64_t payload_end;
    uint64_t page_start;
    uint64_t page_end;
} uocr_metal_payload_span;

typedef struct uocr_metal_scratch_buffer {
    id<MTLBuffer> buffer;
    uint64_t capacity;
    uint64_t high_watermark;
    int cpu_visible;
} uocr_metal_scratch_buffer;

typedef struct uocr_metal_runtime_arena {
    id<MTLBuffer> buffer;
    uint64_t capacity;
    int cpu_visible;
} uocr_metal_runtime_arena;

struct uocr_metal_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipeline_cache;
    NSMutableArray *transient_retains;
    uocr_metal_model_view *model_views;
    uint32_t model_view_count;
    uocr_metal_tensor_binding_internal *tensor_bindings;
    uint32_t tensor_binding_count;
    uint64_t model_view_bytes;
    uint64_t last_model_warmup_bytes;
    uocr_metal_scratch_buffer scratch[UOCR_METAL_SCRATCH_COUNT];
    uocr_metal_runtime_arena runtime_arenas[UOCR_METAL_ARENA_COUNT];
    uocr_metal_decoder_binding_cache decoder_bindings;
    char decoder_binding_error[256];
    uocr_metal_vision_binding_cache vision_bindings;
    char vision_binding_error[256];
    uocr_metal_kv_cache_layout kv_cache_layout;
    int has_kv_cache_layout;
    int has_integrated_prefill;
    uint32_t integrated_prefill_slot;
    uint32_t integrated_prefill_tokens;
    uint32_t integrated_prefill_final_segment;
};

static int metal_fail(char *error, size_t error_size, const char *fmt, ...) {
    if (error != NULL && error_size > 0u) {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(error, error_size, fmt, ap);
        va_end(ap);
        error[error_size - 1u] = '\0';
    }
    return 0;
}

static void metal_clear_error(char *error, size_t error_size) {
    if (error != NULL && error_size > 0u) {
        error[0] = '\0';
    }
}

static int checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || a > UINT64_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || (a != 0u && b > UINT64_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static uint64_t align_down_u64(uint64_t value, uint64_t align) {
    return align == 0u ? value : value - (value % align);
}

static int align_up_u64_checked(uint64_t value, uint64_t align, uint64_t *out) {
    if (out == NULL) {
        return 0;
    }
    if (align == 0u) {
        *out = value;
        return 1;
    }
    const uint64_t rem = value % align;
    if (rem == 0u) {
        *out = value;
        return 1;
    }
    return checked_add_u64(value, align - rem, out);
}

static int payload_span_compare(const void *a, const void *b) {
    const uocr_metal_payload_span *pa = (const uocr_metal_payload_span *)a;
    const uocr_metal_payload_span *pb = (const uocr_metal_payload_span *)b;
    if (pa->page_start < pb->page_start) {
        return -1;
    }
    if (pa->page_start > pb->page_start) {
        return 1;
    }
    if (pa->page_end < pb->page_end) {
        return -1;
    }
    if (pa->page_end > pb->page_end) {
        return 1;
    }
    if (pa->tensor_id < pb->tensor_id) {
        return -1;
    }
    if (pa->tensor_id > pb->tensor_id) {
        return 1;
    }
    return 0;
}

static NSString *string_from_c_or_nil(const char *s) {
    if (s == NULL || s[0] == '\0') {
        return nil;
    }
    return [NSString stringWithUTF8String:s];
}

static void add_metallib_candidates(NSMutableArray<NSString *> *paths, NSString *root) {
    if (root == nil || [root length] == 0u) {
        return;
    }
    [paths addObject:[root stringByAppendingPathComponent:@"unlimitedocr.metallib"]];
    [paths addObject:[root stringByAppendingPathComponent:@"metal/unlimitedocr.metallib"]];
    [paths addObject:[root stringByAppendingPathComponent:@"share/unlimitedocr/metal/unlimitedocr.metallib"]];
}

#if UOCR_METAL_RUNTIME_COMPILE
static void add_source_candidates(NSMutableArray<NSString *> *paths, NSString *root) {
    if (root == nil || [root length] == 0u) {
        return;
    }
    [paths addObject:[root stringByAppendingPathComponent:@"kernels/uocr_smoke.metal"]];
    [paths addObject:[root stringByAppendingPathComponent:@"uocr_smoke.metal"]];
    [paths addObject:[root stringByAppendingPathComponent:@"src/backend/metal/kernels/uocr_smoke.metal"]];
}
#endif

static id<MTLLibrary> load_precompiled_library(id<MTLDevice> device,
                                               const char *resource_path,
                                               char *diagnostic,
                                               size_t diagnostic_size) {
    if (diagnostic != NULL && diagnostic_size > 0u) {
        diagnostic[0] = '\0';
    }
    if (device == nil) {
        if (diagnostic != NULL && diagnostic_size > 0u) {
            (void)snprintf(diagnostic, diagnostic_size, "Metal device is nil");
            diagnostic[diagnostic_size - 1u] = '\0';
        }
        return nil;
    }

    NSFileManager *fm = [NSFileManager defaultManager];
    NSMutableArray<NSString *> *paths = [NSMutableArray array];

    const char *override_path = getenv("UOCR_METAL_LIBRARY_PATH");
    if (override_path != NULL && override_path[0] != '\0') {
        [paths addObject:[NSString stringWithUTF8String:override_path]];
    }

    add_metallib_candidates(paths, string_from_c_or_nil(resource_path));

    const char *env_resource_path = getenv("UOCR_METAL_RESOURCE_PATH");
    add_metallib_candidates(paths, string_from_c_or_nil(env_resource_path));

    NSString *default_metallib_path = string_from_c_or_nil(UOCR_DEFAULT_METALLIB_PATH);
    if (default_metallib_path != nil) {
        [paths addObject:default_metallib_path];
    }
    add_metallib_candidates(paths, string_from_c_or_nil(UOCR_DEFAULT_RESOURCE_PATH));
    add_metallib_candidates(paths, @".");
    add_metallib_candidates(paths, @"./build/debug");
    add_metallib_candidates(paths, @"./build/release");

    NSString *last_error_path = nil;
    NSError *last_error = nil;
    for (NSString *path in paths) {
        if (![fm fileExistsAtPath:path]) {
            continue;
        }
        NSError *load_error = nil;
        NSURL *library_url = [NSURL fileURLWithPath:path];
        id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&load_error];
        if (library != nil) {
            return library;
        }
        last_error_path = path;
        last_error = load_error;
    }

    if (diagnostic != NULL && diagnostic_size > 0u) {
        if (last_error_path != nil) {
            (void)snprintf(diagnostic,
                           diagnostic_size,
                           "failed to load precompiled Metal library %s: %s",
                           [last_error_path UTF8String],
                           last_error != nil ? [[last_error localizedDescription] UTF8String] : "unknown error");
        } else {
            (void)snprintf(diagnostic, diagnostic_size, "precompiled Metal library unlimitedocr.metallib not found");
        }
        diagnostic[diagnostic_size - 1u] = '\0';
    }
    return nil;
}

#if UOCR_METAL_RUNTIME_COMPILE
static NSString *load_runtime_source(const char *resource_path, char *error, size_t error_size) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSMutableArray<NSString *> *paths = [NSMutableArray array];

    const char *override_path = getenv("UOCR_METAL_SMOKE_SOURCE");
    if (override_path != NULL && override_path[0] != '\0') {
        [paths addObject:[NSString stringWithUTF8String:override_path]];
    }

    add_source_candidates(paths, string_from_c_or_nil(resource_path));

    const char *env_resource_path = getenv("UOCR_METAL_RESOURCE_PATH");
    add_source_candidates(paths, string_from_c_or_nil(env_resource_path));

    add_source_candidates(paths, string_from_c_or_nil(UOCR_DEFAULT_RESOURCE_PATH));
    add_source_candidates(paths, @"src/backend/metal");
    add_source_candidates(paths, @"./src/backend/metal");

    NSString *loaded_path = nil;
    NSString *loaded = nil;
    for (NSString *path in paths) {
        if (![fm fileExistsAtPath:path]) {
            continue;
        }
        NSError *read_error = nil;
        loaded = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&read_error];
        if (loaded == nil) {
            (void)metal_fail(error,
                             error_size,
                             "failed to read Metal source %s: %s",
                             [path UTF8String],
                             [[read_error localizedDescription] UTF8String]);
            return nil;
        }
        loaded_path = path;
        break;
    }

    if (loaded == nil) {
        (void)metal_fail(error,
                         error_size,
                         "Metal source kernels/uocr_smoke.metal not found; set resource_path or UOCR_METAL_RESOURCE_PATH");
        return nil;
    }

    NSMutableString *source = [NSMutableString stringWithCapacity:[loaded length] + 128u];
    [source appendFormat:@"// unlimitedocr.c runtime Metal source: %@\n", loaded_path];
    [source appendString:loaded];
    [source appendString:@"\n"];
    return source;
}
#endif

static uint32_t metal_inflight_transient_count(uocr_metal_context *ctx) {
    if (ctx == NULL || ctx->transient_retains == nil) {
        return 0u;
    }
    @synchronized(ctx->transient_retains) {
        return (uint32_t)[ctx->transient_retains count];
    }
}

typedef struct uocr_metal_touch_params {
    uint32_t n_words;
    uint32_t stride_words;
    uint32_t dst_base;
    uint32_t reserved;
} uocr_metal_touch_params;

typedef struct uocr_metal_get_rows_params {
    uint32_t table_rows;
    uint32_t row_width;
    uint32_t n_row_ids;
    uint32_t reserved;
} uocr_metal_get_rows_params;

typedef struct uocr_metal_get_rows_q8_params {
    uint32_t table_rows;
    uint32_t logical_width;
    uint32_t physical_width;
    uint32_t n_row_ids;
    uint32_t row_size;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_get_rows_q8_params;

typedef struct uocr_metal_prompt_assembly_params {
    uint32_t table_rows;
    uint32_t hidden_size;
    uint32_t n_tokens;
    uint32_t image_span_start;
    uint32_t image_span_length;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_prompt_assembly_params;

typedef struct uocr_metal_rmsnorm_params {
    uint32_t n_rows;
    uint32_t hidden_size;
    float eps;
    uint32_t reserved;
} uocr_metal_rmsnorm_params;

typedef struct uocr_metal_dense_params {
    uint32_t input_rows;
    uint32_t in_features;
    uint32_t out_features;
    uint32_t has_bias;
} uocr_metal_dense_params;

typedef struct uocr_metal_dense_q8_params {
    uint32_t input_rows;
    uint32_t logical_in_features;
    uint32_t physical_in_features;
    uint32_t out_features;
    uint32_t weight_row_size;
    uint32_t has_bias;
    uint32_t reserved0;
    uint32_t reserved1;
} uocr_metal_dense_q8_params;

typedef struct uocr_metal_dense_q4_params {
    uint32_t input_rows;
    uint32_t logical_in_features;
    uint32_t physical_in_features;
    uint32_t out_features;
    uint32_t weight_row_size;
    uint32_t has_bias;
    uint32_t reserved0;
    uint32_t reserved1;
} uocr_metal_dense_q4_params;

typedef struct uocr_metal_sam_patch_embed_params {
    uint32_t width;
    uint32_t height;
    uint32_t out_width;
    uint32_t out_height;
    uint32_t has_bias;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_sam_patch_embed_params;

typedef struct uocr_metal_sam_abs_pos_params {
    uint32_t source_grid;
    uint32_t target_width;
    uint32_t target_height;
    uint32_t channels;
} uocr_metal_sam_abs_pos_params;

typedef struct uocr_metal_sam_window_attention_params {
    uint32_t windows;
    uint32_t tokens_per_window;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_sam_window_attention_params;

typedef struct uocr_metal_sam_window_partition_params {
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t windows_per_row;
    uint32_t windows_per_col;
    uint32_t window_size;
    uint32_t hidden_size;
} uocr_metal_sam_window_partition_params;

typedef struct uocr_metal_sam_neck_conv1x1_params {
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t in_channels;
    uint32_t out_channels;
} uocr_metal_sam_neck_conv1x1_params;

typedef struct uocr_metal_sam_neck_conv3x3_params {
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t channels;
    uint32_t kernel_size;
} uocr_metal_sam_neck_conv3x3_params;

typedef struct uocr_metal_sam_conv3x3_stride2_params {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel_size;
    uint32_t stride;
} uocr_metal_sam_conv3x3_stride2_params;

typedef struct uocr_metal_clip_embed_sam_params {
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t hidden_size;
    uint32_t token_count;
} uocr_metal_clip_embed_sam_params;

typedef struct uocr_metal_clip_abs_pos_params {
    uint32_t source_grid;
    uint32_t target_width;
    uint32_t target_height;
    uint32_t hidden_size;
} uocr_metal_clip_abs_pos_params;

typedef struct uocr_metal_clip_sam_concat_params {
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t hidden_size;
    uint32_t projector_in_size;
} uocr_metal_clip_sam_concat_params;

typedef struct uocr_metal_sam_layernorm2d_params {
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t channels;
    float eps;
} uocr_metal_sam_layernorm2d_params;

typedef struct uocr_metal_sam_rel_pos_attention_params {
    uint32_t windows;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t tokens_per_window;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t rel_pos_h_length;
    uint32_t rel_pos_w_length;
    float scale;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_sam_rel_pos_attention_params;

typedef struct uocr_metal_sam_residual_params {
    uint32_t n_rows;
    uint32_t hidden_size;
    uint32_t reserved0;
    uint32_t reserved1;
} uocr_metal_sam_residual_params;

typedef struct uocr_metal_sam_mlp_params {
    uint32_t n_rows;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t reserved;
} uocr_metal_sam_mlp_params;

_Static_assert(sizeof(uocr_metal_sam_window_attention_params) == 32u,
               "uocr_metal_sam_window_attention_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_window_partition_params) == 32u,
               "uocr_metal_sam_window_partition_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_neck_conv1x1_params) == 16u,
               "uocr_metal_sam_neck_conv1x1_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_neck_conv3x3_params) == 16u,
               "uocr_metal_sam_neck_conv3x3_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_conv3x3_stride2_params) == 32u,
               "uocr_metal_sam_conv3x3_stride2_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_clip_embed_sam_params) == 16u,
               "uocr_metal_clip_embed_sam_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_clip_abs_pos_params) == 16u,
               "uocr_metal_clip_abs_pos_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_clip_sam_concat_params) == 16u,
               "uocr_metal_clip_sam_concat_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_layernorm2d_params) == 16u,
               "uocr_metal_sam_layernorm2d_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_rel_pos_attention_params) == 48u,
               "uocr_metal_sam_rel_pos_attention_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_residual_params) == 16u,
               "uocr_metal_sam_residual_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_sam_mlp_params) == 16u,
               "uocr_metal_sam_mlp_params ABI mismatch");

typedef struct uocr_metal_argmax_params {
    uint32_t rows;
    uint32_t vocab_size;
    uint32_t reserved0;
    uint32_t reserved1;
} uocr_metal_argmax_params;

typedef struct uocr_metal_no_repeat_ngram_params {
    uint32_t rows;
    uint32_t vocab_size;
    uint32_t max_candidates;
    uint32_t reserved0;
} uocr_metal_no_repeat_ngram_params;

typedef struct uocr_metal_no_repeat_ngram_pack {
    int32_t *sequences;
    uint32_t *row_offsets;
    uint32_t *sequence_lengths;
    uint32_t *ngram_sizes;
    uint32_t *windows;
    uint32_t total_sequence_tokens;
    uint32_t max_candidates;
    int has_work;
} uocr_metal_no_repeat_ngram_pack;

typedef struct uocr_metal_attention_projection_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t projection_count;
    uint32_t reserved;
} uocr_metal_attention_projection_params;

typedef struct uocr_metal_attention_output_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t reserved0;
    uint32_t reserved1;
} uocr_metal_attention_output_params;

typedef struct uocr_metal_dense_swiglu_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t has_residual;
} uocr_metal_dense_swiglu_params;

typedef struct uocr_metal_moe_router_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t experts;
    uint32_t top_k;
} uocr_metal_moe_router_params;

typedef struct uocr_metal_moe_selected_params {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t top_k;
    uint32_t reserved;
} uocr_metal_moe_selected_params;

typedef struct uocr_metal_moe_selected_q4_params {
    uint32_t hidden_size;
    uint32_t physical_hidden_size;
    uint32_t intermediate_size;
    uint32_t top_k;
    uint32_t gate_row_size;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_moe_selected_q4_params;

typedef struct uocr_metal_moe_selected_down_q8_params {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t physical_intermediate_size;
    uint32_t top_k;
    uint32_t down_row_size;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_moe_selected_down_q8_params;

typedef struct uocr_metal_moe_selected_down_q4_params {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t physical_intermediate_size;
    uint32_t top_k;
    uint32_t down_row_size;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_moe_selected_down_q4_params;

typedef struct uocr_metal_moe_prefill_selected_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_moe_prefill_selected_params;

typedef struct uocr_metal_moe_prefill_selected_q4_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t physical_hidden_size;
    uint32_t intermediate_size;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t gate_row_size;
    uint32_t reserved0;
} uocr_metal_moe_prefill_selected_q4_params;

typedef struct uocr_metal_moe_prefill_selected_down_q8_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t physical_intermediate_size;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t down_row_size;
    uint32_t reserved0;
} uocr_metal_moe_prefill_selected_down_q8_params;

typedef struct uocr_metal_moe_prefill_selected_down_q4_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t physical_intermediate_size;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t down_row_size;
    uint32_t reserved0;
} uocr_metal_moe_prefill_selected_down_q4_params;

_Static_assert(sizeof(uocr_metal_moe_selected_q4_params) == 32u,
               "uocr_metal_moe_selected_q4_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_moe_selected_down_q8_params) == 32u,
               "uocr_metal_moe_selected_down_q8_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_moe_selected_down_q4_params) == 32u,
               "uocr_metal_moe_selected_down_q4_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_moe_prefill_selected_q4_params) == 32u,
               "uocr_metal_moe_prefill_selected_q4_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_moe_prefill_selected_down_q8_params) == 32u,
               "uocr_metal_moe_prefill_selected_down_q8_params ABI mismatch");
_Static_assert(sizeof(uocr_metal_moe_prefill_selected_down_q4_params) == 32u,
               "uocr_metal_moe_prefill_selected_down_q4_params ABI mismatch");

typedef struct uocr_metal_moe_prefill_interleaved_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t expert_stride_values;
    uint32_t up_offset_values;
    uint32_t down_offset_values;
} uocr_metal_moe_prefill_interleaved_params;

typedef struct uocr_metal_moe_combine_params {
    uint32_t n_tokens;
    uint32_t hidden_size;
    uint32_t has_residual;
    uint32_t reserved;
} uocr_metal_moe_combine_params;

typedef struct uocr_metal_rope_qk_params {
    uint32_t n_tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t position_start;
    float freq_scale;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_rope_qk_params;

typedef struct uocr_metal_prefill_attention_params {
    uint32_t n_tokens;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
} uocr_metal_prefill_attention_params;

typedef struct uocr_metal_prefill_attention_varlen_params {
    uint32_t total_tokens;
    uint32_t batch;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} uocr_metal_prefill_attention_varlen_params;

typedef struct uocr_metal_decode_attention_params {
    uint32_t batch_slots;
    uint32_t cache_token_capacity;
    uint32_t layer;
    uint32_t slot;
    uint32_t prompt_length;
    uint32_t generated_count;
    uint32_t attention_length;
    uint32_t first_generated;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t ring_window;
    float scale;
} uocr_metal_decode_attention_params;

_Static_assert(sizeof(uocr_metal_decode_attention_params) == 48u,
               "uocr_metal_decode_attention_params ABI mismatch");

typedef struct uocr_metal_kv_cache_write_params {
    uint32_t n_tokens;
    uint32_t batch_slots;
    uint32_t cache_token_capacity;
    uint32_t layer;
    uint32_t slot;
    uint32_t prompt_length;
    uint32_t position_start;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t ring_window;
    uint32_t reserved0;
    uint32_t reserved1;
} uocr_metal_kv_cache_write_params;

static int metal_retain_transient_until_completed(uocr_metal_context *ctx,
                                                  id<MTLCommandBuffer> command_buffer,
                                                  id object,
                                                  char *error,
                                                  size_t error_size) {
    if (ctx == NULL || ctx->transient_retains == nil || command_buffer == nil || object == nil) {
        return metal_fail(error, error_size, "invalid Metal transient retain request");
    }

    NSMutableArray *retained_objects = [ctx->transient_retains retain];
    @synchronized(retained_objects) {
        [retained_objects addObject:object];
    }
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        (void)completed;
        @synchronized(retained_objects) {
            const NSUInteger index = [retained_objects indexOfObjectIdenticalTo:object];
            if (index != NSNotFound) {
                [retained_objects removeObjectAtIndex:index];
            }
        }
        [retained_objects release];
    }];
    return 1;
}

static id<MTLComputePipelineState> metal_get_pipeline(uocr_metal_context *ctx,
                                                       const char *function_name,
                                                       char *error,
                                                       size_t error_size) {
    if (ctx == NULL || function_name == NULL || function_name[0] == '\0') {
        (void)metal_fail(error, error_size, "invalid Metal pipeline request");
        return nil;
    }

    NSString *key = [NSString stringWithUTF8String:function_name];
    id<MTLComputePipelineState> cached = [ctx->pipeline_cache objectForKey:key];
    if (cached != nil) {
        return cached;
    }

    NSError *ns_error = nil;
    id<MTLFunction> fn = [ctx->library newFunctionWithName:key];
    if (fn == nil) {
        (void)metal_fail(error, error_size, "Metal function %s not found", function_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [ctx->device newComputePipelineStateWithFunction:fn error:&ns_error];
    [fn release];
    if (pipeline == nil) {
        (void)metal_fail(error,
                         error_size,
                         "Metal pipeline %s failed: %s",
                         function_name,
                         [[ns_error localizedDescription] UTF8String]);
        return nil;
    }

    [ctx->pipeline_cache setObject:pipeline forKey:key];
    [pipeline release];
    return [ctx->pipeline_cache objectForKey:key];
}

static NSUInteger metal_power2_threadgroup_width(NSUInteger preferred, NSUInteger max_allowed) {
    NSUInteger width = 1u;
    if (preferred == 0u || max_allowed == 0u) {
        return width;
    }
    NSUInteger limit = preferred < max_allowed ? preferred : max_allowed;
    while (width <= limit / 2u) {
        width *= 2u;
    }
    return width;
}

int uocr_metal_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device != nil) {
            [device release];
            return 1;
        }
        return 0;
    }
}

const char *uocr_metal_backend_name(void) {
    return "metal";
}

uint64_t uocr_metal_recommended_working_set_size(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return 0u;
        }
        const uint64_t size = (uint64_t)[device recommendedMaxWorkingSetSize];
        [device release];
        return size;
    }
}

uint64_t uocr_metal_default_memory_budget_bytes(uint64_t recommended_working_set_bytes) {
    if (recommended_working_set_bytes == 0u) {
        return 0u;
    }
    /* The runtime estimate already includes its own safety margin; keep most
     * of Metal's recommendation available while still leaving OS/driver room.
     */
    return (recommended_working_set_bytes / 100u) * 95u + ((recommended_working_set_bytes % 100u) * 95u) / 100u;
}

uocr_metal_context *uocr_metal_context_create(const char *resource_path, char *error, size_t error_size) {
    metal_clear_error(error, error_size);

    @autoreleasepool {
        uocr_metal_context *ctx = (uocr_metal_context *)calloc(1u, sizeof(*ctx));
        if (ctx == NULL) {
            (void)metal_fail(error, error_size, "failed to allocate Metal context");
            return NULL;
        }

        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device == nil) {
            (void)metal_fail(error, error_size, "Metal device not available");
            uocr_metal_context_destroy(ctx);
            return NULL;
        }

        ctx->queue = [ctx->device newCommandQueue];
        if (ctx->queue == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal command queue");
            uocr_metal_context_destroy(ctx);
            return NULL;
        }

        ctx->pipeline_cache = [[NSMutableDictionary alloc] init];
        if (ctx->pipeline_cache == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal pipeline cache");
            uocr_metal_context_destroy(ctx);
            return NULL;
        }

        ctx->transient_retains = [[NSMutableArray alloc] init];
        if (ctx->transient_retains == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal transient-retain array");
            uocr_metal_context_destroy(ctx);
            return NULL;
        }

        char metallib_diagnostic[512];
        memset(metallib_diagnostic, 0, sizeof(metallib_diagnostic));
        ctx->library = load_precompiled_library(ctx->device, resource_path, metallib_diagnostic, sizeof(metallib_diagnostic));

        if (ctx->library == nil) {
#if UOCR_METAL_RUNTIME_COMPILE
            NSString *source = load_runtime_source(resource_path, error, error_size);
            if (source == nil) {
                uocr_metal_context_destroy(ctx);
                return NULL;
            }

            MTLCompileOptions *options = [MTLCompileOptions new];
            NSError *compile_error = nil;
            ctx->library = [ctx->device newLibraryWithSource:source options:options error:&compile_error];
            [options release];
            if (ctx->library == nil) {
                (void)metal_fail(error,
                                 error_size,
                                 "Metal shader compilation failed after %s: %s",
                                 metallib_diagnostic[0] != '\0' ? metallib_diagnostic : "precompiled Metal library was unavailable",
                                 [[compile_error localizedDescription] UTF8String]);
                uocr_metal_context_destroy(ctx);
                return NULL;
            }
#else
            (void)metal_fail(error,
                             error_size,
                             "Metal runtime source compilation is disabled and %s",
                             metallib_diagnostic[0] != '\0' ? metallib_diagnostic : "precompiled Metal library was unavailable");
            uocr_metal_context_destroy(ctx);
            return NULL;
#endif
        }

        return ctx;
    }
}

static uint64_t metal_device_max_buffer_length(id<MTLDevice> device) {
    if (device == nil) {
        return 0u;
    }
    const NSUInteger max_length = [device maxBufferLength];
    return (uint64_t)max_length;
}

static int payload_tensor_count(const uocr_model_file *model, uint32_t *out_count) {
    if (model == NULL || out_count == NULL) {
        return 0;
    }
    uint32_t count = 0u;
    for (uint32_t i = 0u; i < model->tensor_count; ++i) {
        const uocr_tensor_entry *tensor = &model->tensors[i];
        if (tensor->usage != UOCR_TENSOR_USAGE_OMITTED_WITH_REASON && tensor->payload_size != 0u) {
            ++count;
        }
    }
    *out_count = count;
    return 1;
}

static uocr_metal_payload_span *build_payload_spans(const uocr_model_file *model,
                                                     uint64_t page_size,
                                                     uint32_t payload_count,
                                                     char *error,
                                                     size_t error_size) {
    uocr_metal_payload_span *spans = (uocr_metal_payload_span *)calloc(payload_count, sizeof(spans[0]));
    if (spans == NULL) {
        (void)metal_fail(error, error_size, "failed to allocate Metal model payload span plan");
        return NULL;
    }

    uint32_t out = 0u;
    for (uint32_t i = 0u; i < model->tensor_count; ++i) {
        const uocr_tensor_entry *tensor = &model->tensors[i];
        if (tensor->usage == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON || tensor->payload_size == 0u) {
            continue;
        }

        uint64_t payload_end = 0u;
        uint64_t page_end = 0u;
        if (!checked_add_u64(tensor->payload_offset, tensor->payload_size, &payload_end) ||
            !align_up_u64_checked(payload_end, page_size, &page_end)) {
            free(spans);
            (void)metal_fail(error,
                             error_size,
                             "tensor %u payload range overflows during Metal view planning",
                             tensor->id);
            return NULL;
        }
        if (page_end > (uint64_t)model->size) {
            page_end = (uint64_t)model->size;
        }
        if (page_end < payload_end) {
            free(spans);
            (void)metal_fail(error,
                             error_size,
                             "tensor %u payload end exceeds mapped file during Metal view planning",
                             tensor->id);
            return NULL;
        }

        spans[out].tensor_index = i;
        spans[out].tensor_id = tensor->id;
        spans[out].payload_start = tensor->payload_offset;
        spans[out].payload_end = payload_end;
        spans[out].page_start = align_down_u64(tensor->payload_offset, page_size);
        spans[out].page_end = page_end;
        ++out;
    }

    qsort(spans, payload_count, sizeof(spans[0]), payload_span_compare);
    return spans;
}

static int append_model_view_plan(uocr_metal_model_view **views,
                                  uint32_t *count,
                                  uint32_t *capacity,
                                  uint64_t start,
                                  uint64_t end,
                                  char *error,
                                  size_t error_size) {
    if (views == NULL || count == NULL || capacity == NULL || end <= start) {
        return metal_fail(error, error_size, "invalid Metal model view plan append");
    }
    if (*count == *capacity) {
        const uint32_t new_capacity = *capacity == 0u ? 4u : *capacity * 2u;
        if (new_capacity <= *capacity) {
            return metal_fail(error, error_size, "too many Metal model views");
        }
        uocr_metal_model_view *new_views = (uocr_metal_model_view *)realloc(*views, (size_t)new_capacity * sizeof(new_views[0]));
        if (new_views == NULL) {
            return metal_fail(error, error_size, "failed to grow Metal model view plan");
        }
        memset(new_views + *capacity, 0, (size_t)(new_capacity - *capacity) * sizeof(new_views[0]));
        *views = new_views;
        *capacity = new_capacity;
    }
    (*views)[*count].file_offset = start;
    (*views)[*count].length = end - start;
    (*views)[*count].buffer = nil;
    ++(*count);
    return 1;
}

static uocr_metal_model_view *build_model_view_plan(const uocr_metal_payload_span *spans,
                                                     uint32_t payload_count,
                                                     uint64_t max_buffer_length,
                                                     uint32_t *out_view_count,
                                                     char *error,
                                                     size_t error_size) {
    *out_view_count = 0u;
    if (payload_count == 0u) {
        return NULL;
    }

    uocr_metal_model_view *views = NULL;
    uint32_t view_count = 0u;
    uint32_t view_capacity = 0u;
    uint64_t current_start = 0u;
    uint64_t current_end = 0u;
    int have_current = 0;

    for (uint32_t i = 0u; i < payload_count; ++i) {
        const uocr_metal_payload_span *span = &spans[i];
        if (span->page_end <= span->page_start) {
            free(views);
            (void)metal_fail(error, error_size, "tensor %u has empty page span", span->tensor_id);
            return NULL;
        }
        if (span->page_end - span->page_start > max_buffer_length) {
            free(views);
            (void)metal_fail(error,
                             error_size,
                             "tensor %u requires %llu byte no-copy view, exceeding Metal maxBufferLength %llu",
                             span->tensor_id,
                             (unsigned long long)(span->page_end - span->page_start),
                             (unsigned long long)max_buffer_length);
            return NULL;
        }

        if (!have_current) {
            current_start = span->page_start;
            current_end = span->page_end;
            have_current = 1;
            continue;
        }

        uint64_t merged_end = current_end;
        if (span->page_end > merged_end) {
            merged_end = span->page_end;
        }
        if (merged_end - current_start <= max_buffer_length) {
            current_end = merged_end;
            continue;
        }

        if (!append_model_view_plan(&views, &view_count, &view_capacity, current_start, current_end, error, error_size)) {
            free(views);
            return NULL;
        }
        current_start = span->page_start;
        current_end = span->page_end;
    }

    if (have_current && !append_model_view_plan(&views, &view_count, &view_capacity, current_start, current_end, error, error_size)) {
        free(views);
        return NULL;
    }

    *out_view_count = view_count;
    return views;
}

static int find_view_for_payload(const uocr_metal_model_view *views,
                                 uint32_t view_count,
                                 uint64_t payload_start,
                                 uint64_t payload_size,
                                 uint32_t *out_view_index,
                                 uint64_t *out_inner_offset) {
    uint64_t payload_end = 0u;
    if (!checked_add_u64(payload_start, payload_size, &payload_end)) {
        return 0;
    }
    for (uint32_t i = 0u; i < view_count; ++i) {
        const uint64_t view_start = views[i].file_offset;
        const uint64_t view_end = view_start + views[i].length;
        if (payload_start >= view_start && payload_end <= view_end) {
            *out_view_index = i;
            *out_inner_offset = payload_start - view_start;
            return 1;
        }
    }
    return 0;
}

static void metal_invalidate_decoder_binding_cache(uocr_metal_context *ctx, const char *reason);
static int metal_refresh_decoder_binding_cache(uocr_metal_context *ctx,
                                               const uocr_model_file *model,
                                               char *error,
                                               size_t error_size);
static void metal_invalidate_vision_binding_cache(uocr_metal_context *ctx, const char *reason);
static int metal_refresh_vision_binding_cache(uocr_metal_context *ctx,
                                              const uocr_model_file *model,
                                              char *error,
                                              size_t error_size);
static int metal_sam_transformer_block_has_weights(const uocr_metal_sam_transformer_block_f16 *block);
static int metal_clip_transformer_block_has_weights(const uocr_metal_clip_transformer_block_f16 *block);
static void metal_copy_error_detail(char *detail, size_t detail_size, const char *error);

static uocr_metal_tensor_binding_internal *build_tensor_bindings(const uocr_model_file *model,
                                                                  const uocr_metal_model_view *views,
                                                                  uint32_t view_count,
                                                                  uint32_t binding_count,
                                                                  char *error,
                                                                  size_t error_size) {
    if (binding_count == 0u) {
        return NULL;
    }
    uocr_metal_tensor_binding_internal *bindings =
        (uocr_metal_tensor_binding_internal *)calloc(binding_count, sizeof(bindings[0]));
    if (bindings == NULL) {
        (void)metal_fail(error, error_size, "failed to allocate Metal tensor binding table");
        return NULL;
    }

    uint32_t out = 0u;
    for (uint32_t i = 0u; i < model->tensor_count; ++i) {
        const uocr_tensor_entry *tensor = &model->tensors[i];
        if (tensor->usage == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON || tensor->payload_size == 0u) {
            continue;
        }
        uint32_t view_index = 0u;
        uint64_t inner_offset = 0u;
        if (!find_view_for_payload(views, view_count, tensor->payload_offset, tensor->payload_size, &view_index, &inner_offset)) {
            free(bindings);
            (void)metal_fail(error,
                             error_size,
                             "tensor %u was not covered by any Metal model view",
                             tensor->id);
            return NULL;
        }
        bindings[out].tensor_id = tensor->id;
        bindings[out].view_index = view_index;
        bindings[out].inner_offset = inner_offset;
        bindings[out].payload_size = tensor->payload_size;
        ++out;
    }
    return bindings;
}

void uocr_metal_context_unmap_model(uocr_metal_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    @autoreleasepool {
        if (ctx->model_views != NULL) {
            for (uint32_t i = 0u; i < ctx->model_view_count; ++i) {
                [ctx->model_views[i].buffer release];
                ctx->model_views[i].buffer = nil;
            }
        }
        free(ctx->model_views);
        ctx->model_views = NULL;
        ctx->model_view_count = 0u;
        free(ctx->tensor_bindings);
        ctx->tensor_bindings = NULL;
        ctx->tensor_binding_count = 0u;
        metal_invalidate_decoder_binding_cache(ctx, "no mapped model");
        metal_invalidate_vision_binding_cache(ctx, "no mapped model");
        ctx->model_view_bytes = 0u;
        ctx->last_model_warmup_bytes = 0u;
    }
}

int uocr_metal_context_map_model(uocr_metal_context *ctx, const uocr_model_file *model, char *error, size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || model == NULL || model->data == NULL || model->tensors == NULL) {
        return metal_fail(error, error_size, "Metal model mapping requires a context and loaded .uocr model");
    }

    const uocr_section_entry *tensor_data = uocr_model_file_find_section(model, UOCR_SECTION_TENSOR_DATA);
    if (tensor_data == NULL || tensor_data->size == 0u) {
        return metal_fail(error, error_size, ".uocr model has no tensor-data section to map");
    }

    uint32_t payload_count = 0u;
    if (!payload_tensor_count(model, &payload_count) || payload_count == 0u) {
        return metal_fail(error, error_size, ".uocr model has no tensor payloads to map");
    }

    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        page_size_long = (long)UOCR_TENSOR_DATA_ALIGNMENT;
    }
    const uint64_t page_size = (uint64_t)page_size_long;
    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (max_buffer_length < page_size) {
        return metal_fail(error,
                          error_size,
                          "Metal maxBufferLength %llu is smaller than page size %llu",
                          (unsigned long long)max_buffer_length,
                          (unsigned long long)page_size);
    }

    uocr_metal_payload_span *spans = build_payload_spans(model, page_size, payload_count, error, error_size);
    if (spans == NULL) {
        return 0;
    }

    uint32_t view_count = 0u;
    uocr_metal_model_view *views = build_model_view_plan(spans, payload_count, max_buffer_length, &view_count, error, error_size);
    free(spans);
    if (views == NULL || view_count == 0u) {
        free(views);
        return 0;
    }

    uocr_metal_tensor_binding_internal *bindings =
        build_tensor_bindings(model, views, view_count, payload_count, error, error_size);
    if (bindings == NULL) {
        free(views);
        return 0;
    }

    @autoreleasepool {
        for (uint32_t i = 0u; i < view_count; ++i) {
            if (views[i].length > (uint64_t)SIZE_MAX) {
                free(bindings);
                free(views);
                return metal_fail(error, error_size, "Metal model view %u is too large for this platform", i);
            }
            void *view_ptr = (void *)(uintptr_t)(model->data + views[i].file_offset);
            id<MTLBuffer> buffer = [ctx->device newBufferWithBytesNoCopy:view_ptr
                                                                   length:(NSUInteger)views[i].length
                                                                  options:MTLResourceStorageModeShared
                                                              deallocator:nil];
            if (buffer == nil) {
                for (uint32_t j = 0u; j < i; ++j) {
                    [views[j].buffer release];
                    views[j].buffer = nil;
                }
                free(bindings);
                free(views);
                return metal_fail(error,
                                  error_size,
                                  "newBufferWithBytesNoCopy failed for model view %u at file offset %llu length %llu",
                                  i,
                                  (unsigned long long)views[i].file_offset,
                                  (unsigned long long)views[i].length);
            }
            buffer.label = [NSString stringWithFormat:@"uocr_model_view_%u", i];
            views[i].buffer = buffer;
        }
    }

    uocr_metal_context_unmap_model(ctx);
    ctx->model_views = views;
    ctx->model_view_count = view_count;
    ctx->tensor_bindings = bindings;
    ctx->tensor_binding_count = payload_count;
    ctx->model_view_bytes = tensor_data->size;

    char decoder_error[256];
    memset(decoder_error, 0, sizeof(decoder_error));
    if (!metal_refresh_decoder_binding_cache(ctx, model, decoder_error, sizeof(decoder_error))) {
        metal_invalidate_decoder_binding_cache(ctx,
                                               decoder_error[0] != '\0' ? decoder_error : "decoder tensor bindings are not available");
    }
    char vision_error[256];
    memset(vision_error, 0, sizeof(vision_error));
    if (!metal_refresh_vision_binding_cache(ctx, model, vision_error, sizeof(vision_error))) {
        metal_invalidate_vision_binding_cache(ctx,
                                              vision_error[0] != '\0' ? vision_error : "vision tensor bindings are not available");
    }

    metal_clear_error(error, error_size);
    return 1;
}

uint32_t uocr_metal_context_model_view_count(const uocr_metal_context *ctx) {
    return ctx != NULL ? ctx->model_view_count : 0u;
}

uint32_t uocr_metal_context_tensor_binding_count(const uocr_metal_context *ctx) {
    return ctx != NULL ? ctx->tensor_binding_count : 0u;
}

uint32_t uocr_metal_context_library_function_count(const uocr_metal_context *ctx) {
    @autoreleasepool {
        if (ctx == NULL || ctx->library == nil) {
            return 0u;
        }
        return (uint32_t)[[ctx->library functionNames] count];
    }
}

uint32_t uocr_metal_context_pipeline_cache_count(const uocr_metal_context *ctx) {
    @autoreleasepool {
        if (ctx == NULL || ctx->pipeline_cache == nil) {
            return 0u;
        }
        return (uint32_t)[ctx->pipeline_cache count];
    }
}

int uocr_metal_context_compile_all_pipelines(uocr_metal_context *ctx,
                                             uint32_t *out_pipeline_count,
                                             char *error,
                                             size_t error_size) {
    metal_clear_error(error, error_size);
    if (out_pipeline_count != NULL) {
        *out_pipeline_count = 0u;
    }
    if (ctx == NULL || ctx->library == nil) {
        return metal_fail(error, error_size, "invalid Metal context for pipeline compilation");
    }

    @autoreleasepool {
        NSArray<NSString *> *names = [[ctx->library functionNames] sortedArrayUsingSelector:@selector(compare:)];
        if ([names count] == 0u) {
            return metal_fail(error, error_size, "Metal library contains no compute functions");
        }
        for (NSString *name in names) {
            const char *function_name = [name UTF8String];
            if (function_name == NULL || function_name[0] == '\0') {
                return metal_fail(error, error_size, "Metal library contains an invalid function name");
            }
            if (metal_get_pipeline(ctx, function_name, error, error_size) == nil) {
                return 0;
            }
        }
        if (out_pipeline_count != NULL) {
            *out_pipeline_count = (uint32_t)[ctx->pipeline_cache count];
        }
    }
    metal_clear_error(error, error_size);
    return 1;
}

uint64_t uocr_metal_context_model_view_bytes(const uocr_metal_context *ctx) {
    return ctx != NULL ? ctx->model_view_bytes : 0u;
}

int uocr_metal_context_get_model_view_info(const uocr_metal_context *ctx,
                                           uint32_t view_index,
                                           uocr_metal_model_view_info *out_info) {
    if (ctx == NULL || out_info == NULL || view_index >= ctx->model_view_count) {
        return 0;
    }
    out_info->file_offset = ctx->model_views[view_index].file_offset;
    out_info->length = ctx->model_views[view_index].length;
    return 1;
}

static const uocr_metal_tensor_binding_internal *metal_find_tensor_binding(const uocr_metal_context *ctx,
                                                                            uint32_t tensor_id) {
    if (ctx == NULL || ctx->tensor_bindings == NULL || tensor_id == 0u) {
        return NULL;
    }
    uint32_t lo = 0u;
    uint32_t hi = ctx->tensor_binding_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        const uint32_t mid_id = ctx->tensor_bindings[mid].tensor_id;
        if (mid_id == tensor_id) {
            return &ctx->tensor_bindings[mid];
        }
        if (mid_id < tensor_id) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return NULL;
}

int uocr_metal_context_get_tensor_binding(const uocr_metal_context *ctx,
                                          uint32_t tensor_id,
                                          uocr_metal_tensor_binding *out_binding) {
    if (out_binding == NULL) {
        return 0;
    }
    const uocr_metal_tensor_binding_internal *binding = metal_find_tensor_binding(ctx, tensor_id);
    if (binding == NULL) {
        return 0;
    }
    out_binding->tensor_id = binding->tensor_id;
    out_binding->view_index = binding->view_index;
    out_binding->inner_offset = binding->inner_offset;
    out_binding->payload_size = binding->payload_size;
    return 1;
}

static int metal_get_mapped_tensor_buffer(const uocr_metal_context *ctx,
                                          uint32_t tensor_id,
                                          uint64_t expected_payload_size,
                                          id<MTLBuffer> *out_buffer,
                                          NSUInteger *out_offset,
                                          char *error,
                                          size_t error_size) {
    if (out_buffer == NULL || out_offset == NULL) {
        return metal_fail(error, error_size, "invalid Metal mapped tensor output request");
    }
    *out_buffer = nil;
    *out_offset = 0u;
    if (ctx == NULL || ctx->model_views == NULL || ctx->model_view_count == 0u) {
        return metal_fail(error, error_size, "Metal mapped tensor %u requires mapped model views", tensor_id);
    }
    const uocr_metal_tensor_binding_internal *binding = metal_find_tensor_binding(ctx, tensor_id);
    if (binding == NULL) {
        return metal_fail(error, error_size, "Metal mapped tensor %u is not present in mapped model views", tensor_id);
    }
    if (binding->view_index >= ctx->model_view_count || ctx->model_views[binding->view_index].buffer == nil) {
        return metal_fail(error, error_size, "Metal mapped tensor %u has an invalid model-view binding", tensor_id);
    }
    if (binding->payload_size != expected_payload_size) {
        return metal_fail(error,
                          error_size,
                          "Metal mapped tensor %u payload size mismatch: got %llu expected %llu",
                          tensor_id,
                          (unsigned long long)binding->payload_size,
                          (unsigned long long)expected_payload_size);
    }
    if (binding->inner_offset > (uint64_t)NSUIntegerMax) {
        return metal_fail(error, error_size, "Metal mapped tensor %u offset exceeds platform limit", tensor_id);
    }
    *out_buffer = ctx->model_views[binding->view_index].buffer;
    *out_offset = (NSUInteger)binding->inner_offset;
    return 1;
}

static void metal_invalidate_decoder_binding_cache(uocr_metal_context *ctx, const char *reason) {
    if (ctx == NULL) {
        return;
    }
    memset(&ctx->decoder_bindings, 0, sizeof(ctx->decoder_bindings));
    if (reason == NULL || reason[0] == '\0') {
        reason = "decoder tensor bindings are not validated";
    }
    (void)snprintf(ctx->decoder_binding_error, sizeof(ctx->decoder_binding_error), "%s", reason);
}

static int metal_tensor_expected_payload_bytes(uint32_t rank,
                                               const uint32_t dims[UOCR_TENSOR_MAX_DIMS],
                                               uint64_t *out_bytes) {
    if (dims == NULL || out_bytes == NULL || rank == 0u || rank > UOCR_TENSOR_MAX_DIMS) {
        return 0;
    }
    uint64_t values = 1u;
    for (uint32_t i = 0u; i < rank; ++i) {
        if (dims[i] == 0u || !checked_mul_u64(values, (uint64_t)dims[i], &values)) {
            return 0;
        }
    }
    return checked_mul_u64(values, 2u, out_bytes);
}

static int metal_validate_decoder_tensor_metadata(const uocr_tensor_entry *tensor,
                                                  uint32_t tensor_id,
                                                  uint32_t family,
                                                  int32_t layer,
                                                  int32_t expert,
                                                  uint32_t projection,
                                                  uint32_t rank,
                                                  const uint32_t dims[UOCR_TENSOR_MAX_DIMS],
                                                  uint64_t expected_payload_size,
                                                  char *error,
                                                  size_t error_size) {
    if (tensor == NULL) {
        return metal_fail(error, error_size, "missing decoder tensor %u", tensor_id);
    }
    if (tensor->id != tensor_id) {
        return metal_fail(error, error_size, "decoder tensor id mismatch: got %u expected %u", tensor->id, tensor_id);
    }
    if (tensor->usage != UOCR_TENSOR_USAGE_RUNTIME) {
        return metal_fail(error,
                          error_size,
                          "decoder tensor %u has unsupported usage %s",
                          tensor_id,
                          uocr_tensor_usage_name(tensor->usage));
    }
    if (tensor->qtype != UOCR_TENSOR_F16) {
        return metal_fail(error,
                          error_size,
                          "decoder tensor %u has unsupported qtype %s; integrated fp16 decoder requires f16",
                          tensor_id,
                          uocr_tensor_qtype_name(tensor->qtype));
    }
    if (tensor->family != family || tensor->layer != layer || tensor->expert != expert || tensor->projection != projection) {
        return metal_fail(error,
                          error_size,
                          "decoder tensor %u registry metadata mismatch: family=%s layer=%d expert=%d projection=%u",
                          tensor_id,
                          uocr_tensor_family_name(tensor->family),
                          tensor->layer,
                          tensor->expert,
                          tensor->projection);
    }
    if (tensor->rank != rank) {
        return metal_fail(error,
                          error_size,
                          "decoder tensor %u rank mismatch: got %u expected %u",
                          tensor_id,
                          tensor->rank,
                          rank);
    }
    for (uint32_t i = 0u; i < rank; ++i) {
        if (tensor->logical_shape[i] != dims[i] || tensor->physical_shape[i] != dims[i]) {
            return metal_fail(error,
                              error_size,
                              "decoder tensor %u shape[%u] mismatch: logical=%u physical=%u expected=%u",
                              tensor_id,
                              i,
                              tensor->logical_shape[i],
                              tensor->physical_shape[i],
                              dims[i]);
        }
    }
    for (uint32_t i = rank; i < UOCR_TENSOR_MAX_DIMS; ++i) {
        if (tensor->logical_shape[i] != 0u || tensor->physical_shape[i] != 0u) {
            return metal_fail(error,
                              error_size,
                              "decoder tensor %u has non-zero trailing shape[%u]: logical=%u physical=%u",
                              tensor_id,
                              i,
                              tensor->logical_shape[i],
                              tensor->physical_shape[i]);
        }
    }
    if (tensor->payload_size != expected_payload_size) {
        return metal_fail(error,
                          error_size,
                          "decoder tensor %u payload size mismatch: got %llu expected %llu",
                          tensor_id,
                          (unsigned long long)tensor->payload_size,
                          (unsigned long long)expected_payload_size);
    }
    return 1;
}

static int metal_append_decoder_binding(uocr_metal_context *ctx,
                                        const uocr_model_file *model,
                                        uocr_metal_decoder_binding_cache *cache,
                                        uint32_t tensor_id,
                                        uint32_t family,
                                        int32_t layer,
                                        int32_t expert,
                                        uint32_t projection,
                                        uint32_t rank,
                                        const uint32_t dims[UOCR_TENSOR_MAX_DIMS],
                                        char *error,
                                        size_t error_size) {
    if (ctx == NULL || model == NULL || cache == NULL || cache->count >= UOCR_METAL_DECODER_REQUIRED_TENSOR_COUNT) {
        return metal_fail(error, error_size, "decoder tensor binding cache overflow");
    }
    uint64_t expected_payload_size = 0u;
    if (!metal_tensor_expected_payload_bytes(rank, dims, &expected_payload_size)) {
        return metal_fail(error, error_size, "decoder tensor %u expected byte-size overflow", tensor_id);
    }
    const uocr_tensor_entry *tensor = uocr_model_file_find_tensor(model, tensor_id);
    if (!metal_validate_decoder_tensor_metadata(tensor,
                                                tensor_id,
                                                family,
                                                layer,
                                                expert,
                                                projection,
                                                rank,
                                                dims,
                                                expected_payload_size,
                                                error,
                                                error_size)) {
        return 0;
    }

    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0u;
    if (!metal_get_mapped_tensor_buffer(ctx, tensor_id, expected_payload_size, &buffer, &offset, error, error_size)) {
        return 0;
    }

    uocr_metal_decoder_binding *binding = &cache->tensors[cache->count++];
    binding->tensor_id = tensor_id;
    binding->reserved = 0u;
    binding->buffer = buffer;
    binding->offset = offset;
    binding->payload_size = expected_payload_size;
    return 1;
}

static int metal_append_decoder_binding1(uocr_metal_context *ctx,
                                         const uocr_model_file *model,
                                         uocr_metal_decoder_binding_cache *cache,
                                         uint32_t tensor_id,
                                         uint32_t family,
                                         int32_t layer,
                                         int32_t expert,
                                         uint32_t projection,
                                         uint32_t d0,
                                         char *error,
                                         size_t error_size) {
    const uint32_t dims[UOCR_TENSOR_MAX_DIMS] = {d0, 0u, 0u, 0u};
    return metal_append_decoder_binding(ctx, model, cache, tensor_id, family, layer, expert, projection, 1u, dims, error, error_size);
}

static int metal_append_decoder_binding2(uocr_metal_context *ctx,
                                         const uocr_model_file *model,
                                         uocr_metal_decoder_binding_cache *cache,
                                         uint32_t tensor_id,
                                         uint32_t family,
                                         int32_t layer,
                                         int32_t expert,
                                         uint32_t projection,
                                         uint32_t d0,
                                         uint32_t d1,
                                         char *error,
                                         size_t error_size) {
    const uint32_t dims[UOCR_TENSOR_MAX_DIMS] = {d0, d1, 0u, 0u};
    return metal_append_decoder_binding(ctx, model, cache, tensor_id, family, layer, expert, projection, 2u, dims, error, error_size);
}

static int metal_refresh_decoder_binding_cache(uocr_metal_context *ctx,
                                               const uocr_model_file *model,
                                               char *error,
                                               size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || model == NULL || ctx->model_views == NULL || ctx->tensor_bindings == NULL) {
        return metal_fail(error, error_size, "decoder tensor binding validation requires a mapped model");
    }

    uocr_metal_decoder_binding_cache cache;
    memset(&cache, 0, sizeof(cache));

#define APPEND1(tensor_id, family, layer, expert, projection, d0) \
    do { \
        if (!metal_append_decoder_binding1(ctx, model, &cache, (tensor_id), (family), (layer), (expert), (projection), (d0), error, error_size)) { \
            return 0; \
        } \
    } while (0)
#define APPEND2(tensor_id, family, layer, expert, projection, d0, d1) \
    do { \
        if (!metal_append_decoder_binding2(ctx, model, &cache, (tensor_id), (family), (layer), (expert), (projection), (d0), (d1), error, error_size)) { \
            return 0; \
        } \
    } while (0)

    APPEND2(UOCR_TENSOR_ID_TOK_EMBED,
            UOCR_TENSOR_FAMILY_TOK_EMBED,
            -1,
            -1,
            UOCR_TENSOR_PROJ_WEIGHT,
            UOCR_VOCAB_SIZE,
            UOCR_HIDDEN_SIZE);
    APPEND2(UOCR_TENSOR_ID_LM_HEAD,
            UOCR_TENSOR_FAMILY_LM_HEAD,
            -1,
            -1,
            UOCR_TENSOR_PROJ_WEIGHT,
            UOCR_VOCAB_SIZE,
            UOCR_HIDDEN_SIZE);
    APPEND1(UOCR_TENSOR_ID_FINAL_NORM,
            UOCR_TENSOR_FAMILY_FINAL_NORM,
            -1,
            -1,
            UOCR_TENSOR_PROJ_WEIGHT,
            UOCR_HIDDEN_SIZE);

    for (uint32_t layer = 0u; layer < UOCR_DECODER_LAYERS; ++layer) {
        APPEND1(uocr_tensor_id_layer_input_norm(layer),
                UOCR_TENSOR_FAMILY_LAYER_NORM,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_WEIGHT,
                UOCR_HIDDEN_SIZE);
        APPEND1(uocr_tensor_id_layer_post_attn_norm(layer),
                UOCR_TENSOR_FAMILY_LAYER_NORM,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_WEIGHT,
                UOCR_HIDDEN_SIZE);
        APPEND2(uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_Q),
                UOCR_TENSOR_FAMILY_LAYER_ATTN,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_Q,
                UOCR_HIDDEN_SIZE,
                UOCR_HIDDEN_SIZE);
        APPEND2(uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_K),
                UOCR_TENSOR_FAMILY_LAYER_ATTN,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_K,
                UOCR_HIDDEN_SIZE,
                UOCR_HIDDEN_SIZE);
        APPEND2(uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_V),
                UOCR_TENSOR_FAMILY_LAYER_ATTN,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_V,
                UOCR_HIDDEN_SIZE,
                UOCR_HIDDEN_SIZE);
        APPEND2(uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_O),
                UOCR_TENSOR_FAMILY_LAYER_ATTN,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_O,
                UOCR_HIDDEN_SIZE,
                UOCR_HIDDEN_SIZE);
    }

    APPEND2(uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_GATE),
            UOCR_TENSOR_FAMILY_LAYER_DENSE_MLP,
            0,
            -1,
            UOCR_TENSOR_PROJ_GATE,
            UOCR_DENSE_LAYER0_INTERMEDIATE,
            UOCR_HIDDEN_SIZE);
    APPEND2(uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_UP),
            UOCR_TENSOR_FAMILY_LAYER_DENSE_MLP,
            0,
            -1,
            UOCR_TENSOR_PROJ_UP,
            UOCR_DENSE_LAYER0_INTERMEDIATE,
            UOCR_HIDDEN_SIZE);
    APPEND2(uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_DOWN),
            UOCR_TENSOR_FAMILY_LAYER_DENSE_MLP,
            0,
            -1,
            UOCR_TENSOR_PROJ_DOWN,
            UOCR_HIDDEN_SIZE,
            UOCR_DENSE_LAYER0_INTERMEDIATE);

    for (uint32_t layer = 1u; layer < UOCR_DECODER_LAYERS; ++layer) {
        APPEND2(uocr_tensor_id_moe_router(layer),
                UOCR_TENSOR_FAMILY_MOE_ROUTER,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_WEIGHT,
                UOCR_ROUTED_EXPERTS,
                UOCR_HIDDEN_SIZE);
        APPEND2(uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_GATE),
                UOCR_TENSOR_FAMILY_MOE_SHARED,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_GATE,
                UOCR_MOE_SHARED_INTERMEDIATE,
                UOCR_HIDDEN_SIZE);
        APPEND2(uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_UP),
                UOCR_TENSOR_FAMILY_MOE_SHARED,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_UP,
                UOCR_MOE_SHARED_INTERMEDIATE,
                UOCR_HIDDEN_SIZE);
        APPEND2(uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_DOWN),
                UOCR_TENSOR_FAMILY_MOE_SHARED,
                (int32_t)layer,
                -1,
                UOCR_TENSOR_PROJ_DOWN,
                UOCR_HIDDEN_SIZE,
                UOCR_MOE_SHARED_INTERMEDIATE);
        for (uint32_t expert = 0u; expert < UOCR_ROUTED_EXPERTS; ++expert) {
            APPEND2(uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_GATE),
                    UOCR_TENSOR_FAMILY_MOE_EXPERT,
                    (int32_t)layer,
                    (int32_t)expert,
                    UOCR_TENSOR_PROJ_GATE,
                    UOCR_MOE_EXPERT_INTERMEDIATE,
                    UOCR_HIDDEN_SIZE);
            APPEND2(uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_UP),
                    UOCR_TENSOR_FAMILY_MOE_EXPERT,
                    (int32_t)layer,
                    (int32_t)expert,
                    UOCR_TENSOR_PROJ_UP,
                    UOCR_MOE_EXPERT_INTERMEDIATE,
                    UOCR_HIDDEN_SIZE);
            APPEND2(uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_DOWN),
                    UOCR_TENSOR_FAMILY_MOE_EXPERT,
                    (int32_t)layer,
                    (int32_t)expert,
                    UOCR_TENSOR_PROJ_DOWN,
                    UOCR_HIDDEN_SIZE,
                    UOCR_MOE_EXPERT_INTERMEDIATE);
        }
    }

#undef APPEND1
#undef APPEND2

    if (cache.count != UOCR_METAL_DECODER_REQUIRED_TENSOR_COUNT) {
        return metal_fail(error,
                          error_size,
                          "decoder tensor binding count mismatch: got %u expected %u",
                          cache.count,
                          (uint32_t)UOCR_METAL_DECODER_REQUIRED_TENSOR_COUNT);
    }
    cache.valid = 1;
    ctx->decoder_bindings = cache;
    ctx->decoder_binding_error[0] = '\0';
    return 1;
}

static const uocr_metal_decoder_binding *metal_find_decoder_binding(const uocr_metal_context *ctx,
                                                                    uint32_t tensor_id) {
    if (ctx == NULL || !ctx->decoder_bindings.valid || tensor_id == 0u) {
        return NULL;
    }
    for (uint32_t i = 0u; i < ctx->decoder_bindings.count; ++i) {
        if (ctx->decoder_bindings.tensors[i].tensor_id == tensor_id) {
            return &ctx->decoder_bindings.tensors[i];
        }
    }
    return NULL;
}

uint32_t uocr_metal_context_decoder_binding_count(const uocr_metal_context *ctx) {
    return ctx != NULL && ctx->decoder_bindings.valid ? ctx->decoder_bindings.count : 0u;
}

int uocr_metal_context_decoder_bindings_ready(const uocr_metal_context *ctx) {
    return ctx != NULL && ctx->decoder_bindings.valid &&
           ctx->decoder_bindings.count == UOCR_METAL_DECODER_REQUIRED_TENSOR_COUNT;
}

static void metal_invalidate_vision_binding_cache(uocr_metal_context *ctx, const char *reason) {
    if (ctx == NULL) {
        return;
    }
    memset(&ctx->vision_bindings, 0, sizeof(ctx->vision_bindings));
    if (reason == NULL || reason[0] == '\0') {
        reason = "vision tensor bindings are not validated";
    }
    (void)snprintf(ctx->vision_binding_error, sizeof(ctx->vision_binding_error), "%s", reason);
}

static int metal_validate_vision_tensor_metadata(const uocr_tensor_entry *tensor,
                                                 uint32_t tensor_id,
                                                 const char *role,
                                                 uint32_t family,
                                                 int32_t layer,
                                                 int32_t expert,
                                                 uint32_t projection,
                                                 uint32_t rank,
                                                 const uint32_t dims[UOCR_TENSOR_MAX_DIMS],
                                                 uint64_t expected_payload_size,
                                                 char *error,
                                                 size_t error_size) {
    if (role == NULL || role[0] == '\0') {
        role = "vision tensor";
    }
    if (tensor == NULL) {
        return metal_fail(error, error_size, "missing %s %u", role, tensor_id);
    }
    if (tensor->id != tensor_id) {
        return metal_fail(error, error_size, "%s id mismatch: got %u expected %u", role, tensor->id, tensor_id);
    }
    if (tensor->usage != UOCR_TENSOR_USAGE_RUNTIME) {
        return metal_fail(error,
                          error_size,
                          "%s %u has unsupported usage %s",
                          role,
                          tensor_id,
                          uocr_tensor_usage_name(tensor->usage));
    }
    if (tensor->qtype != UOCR_TENSOR_F16) {
        return metal_fail(error,
                          error_size,
                          "%s %u has unsupported qtype %s; production Metal vision requires f16",
                          role,
                          tensor_id,
                          uocr_tensor_qtype_name(tensor->qtype));
    }
    if (tensor->family != family || tensor->layer != layer || tensor->expert != expert || tensor->projection != projection) {
        return metal_fail(error,
                          error_size,
                          "%s %u registry metadata mismatch: family=%s layer=%d expert=%d projection=%u",
                          role,
                          tensor_id,
                          uocr_tensor_family_name(tensor->family),
                          tensor->layer,
                          tensor->expert,
                          tensor->projection);
    }
    if (tensor->rank != rank) {
        return metal_fail(error,
                          error_size,
                          "%s %u rank mismatch: got %u expected %u",
                          role,
                          tensor_id,
                          tensor->rank,
                          rank);
    }
    for (uint32_t i = 0u; i < rank; ++i) {
        if (tensor->logical_shape[i] != dims[i] || tensor->physical_shape[i] != dims[i]) {
            return metal_fail(error,
                              error_size,
                              "%s %u shape[%u] mismatch: logical=%u physical=%u expected=%u",
                              role,
                              tensor_id,
                              i,
                              tensor->logical_shape[i],
                              tensor->physical_shape[i],
                              dims[i]);
        }
    }
    for (uint32_t i = rank; i < UOCR_TENSOR_MAX_DIMS; ++i) {
        if (tensor->logical_shape[i] != 0u || tensor->physical_shape[i] != 0u) {
            return metal_fail(error,
                              error_size,
                              "%s %u has non-zero trailing shape[%u]: logical=%u physical=%u",
                              role,
                              tensor_id,
                              i,
                              tensor->logical_shape[i],
                              tensor->physical_shape[i]);
        }
    }
    if (tensor->payload_size != expected_payload_size) {
        return metal_fail(error,
                          error_size,
                          "%s %u payload size mismatch: got %llu expected %llu",
                          role,
                          tensor_id,
                          (unsigned long long)tensor->payload_size,
                          (unsigned long long)expected_payload_size);
    }
    return 1;
}

static int metal_append_vision_binding(uocr_metal_context *ctx,
                                       const uocr_model_file *model,
                                       uocr_metal_vision_binding_cache *cache,
                                       uint32_t tensor_id,
                                       const char *role,
                                       uint32_t family,
                                       int32_t layer,
                                       int32_t expert,
                                       uint32_t projection,
                                       uint32_t rank,
                                       const uint32_t dims[UOCR_TENSOR_MAX_DIMS],
                                       char *error,
                                       size_t error_size) {
    if (ctx == NULL || model == NULL || cache == NULL || cache->count >= UOCR_METAL_VISION_REQUIRED_TENSOR_COUNT) {
        return metal_fail(error, error_size, "vision tensor binding cache overflow");
    }
    uint64_t expected_payload_size = 0u;
    if (!metal_tensor_expected_payload_bytes(rank, dims, &expected_payload_size)) {
        return metal_fail(error, error_size, "vision tensor %u expected byte-size overflow", tensor_id);
    }
    const uocr_tensor_entry *tensor = uocr_model_file_find_tensor(model, tensor_id);
    if (!metal_validate_vision_tensor_metadata(tensor,
                                               tensor_id,
                                               role,
                                               family,
                                               layer,
                                               expert,
                                               projection,
                                               rank,
                                               dims,
                                               expected_payload_size,
                                               error,
                                               error_size)) {
        return 0;
    }

    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0u;
    if (!metal_get_mapped_tensor_buffer(ctx, tensor_id, expected_payload_size, &buffer, &offset, error, error_size)) {
        return 0;
    }

    uocr_metal_vision_binding *binding = &cache->tensors[cache->count++];
    binding->tensor_id = tensor_id;
    binding->reserved = 0u;
    binding->buffer = buffer;
    binding->offset = offset;
    binding->payload_size = expected_payload_size;
    return 1;
}

static int metal_append_vision_binding1(uocr_metal_context *ctx,
                                        const uocr_model_file *model,
                                        uocr_metal_vision_binding_cache *cache,
                                        uint32_t tensor_id,
                                        const char *role,
                                        uint32_t family,
                                        int32_t layer,
                                        int32_t expert,
                                        uint32_t projection,
                                        uint32_t d0,
                                        char *error,
                                        size_t error_size) {
    const uint32_t dims[UOCR_TENSOR_MAX_DIMS] = {d0, 0u, 0u, 0u};
    return metal_append_vision_binding(ctx, model, cache, tensor_id, role, family, layer, expert, projection, 1u, dims, error, error_size);
}

static int metal_append_vision_binding2(uocr_metal_context *ctx,
                                        const uocr_model_file *model,
                                        uocr_metal_vision_binding_cache *cache,
                                        uint32_t tensor_id,
                                        const char *role,
                                        uint32_t family,
                                        int32_t layer,
                                        int32_t expert,
                                        uint32_t projection,
                                        uint32_t d0,
                                        uint32_t d1,
                                        char *error,
                                        size_t error_size) {
    const uint32_t dims[UOCR_TENSOR_MAX_DIMS] = {d0, d1, 0u, 0u};
    return metal_append_vision_binding(ctx, model, cache, tensor_id, role, family, layer, expert, projection, 2u, dims, error, error_size);
}

static int metal_append_vision_binding4(uocr_metal_context *ctx,
                                        const uocr_model_file *model,
                                        uocr_metal_vision_binding_cache *cache,
                                        uint32_t tensor_id,
                                        const char *role,
                                        uint32_t family,
                                        int32_t layer,
                                        int32_t expert,
                                        uint32_t projection,
                                        uint32_t d0,
                                        uint32_t d1,
                                        uint32_t d2,
                                        uint32_t d3,
                                        char *error,
                                        size_t error_size) {
    const uint32_t dims[UOCR_TENSOR_MAX_DIMS] = {d0, d1, d2, d3};
    return metal_append_vision_binding(ctx, model, cache, tensor_id, role, family, layer, expert, projection, 4u, dims, error, error_size);
}

static int metal_refresh_vision_binding_cache(uocr_metal_context *ctx,
                                              const uocr_model_file *model,
                                              char *error,
                                              size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || model == NULL || ctx->model_views == NULL || ctx->tensor_bindings == NULL) {
        return metal_fail(error, error_size, "vision tensor binding validation requires a mapped model");
    }

    uocr_metal_vision_binding_cache cache;
    memset(&cache, 0, sizeof(cache));

#define VAPPEND1(tensor_id, role, family, layer, expert, projection, d0) \
    do { \
        if (!metal_append_vision_binding1(ctx, model, &cache, (tensor_id), (role), (family), (layer), (expert), (projection), (d0), error, error_size)) { \
            return 0; \
        } \
    } while (0)
#define VAPPEND2(tensor_id, role, family, layer, expert, projection, d0, d1) \
    do { \
        if (!metal_append_vision_binding2(ctx, model, &cache, (tensor_id), (role), (family), (layer), (expert), (projection), (d0), (d1), error, error_size)) { \
            return 0; \
        } \
    } while (0)
#define VAPPEND4(tensor_id, role, family, layer, expert, projection, d0, d1, d2, d3) \
    do { \
        if (!metal_append_vision_binding4(ctx, model, &cache, (tensor_id), (role), (family), (layer), (expert), (projection), (d0), (d1), (d2), (d3), error, error_size)) { \
            return 0; \
        } \
    } while (0)

    VAPPEND1(UOCR_TENSOR_ID_IMAGE_NEWLINE,
             "image newline tensor",
             UOCR_TENSOR_FAMILY_IMAGE_NEWLINE,
             -1,
             -1,
             UOCR_TENSOR_PROJ_NONE,
             UOCR_HIDDEN_SIZE);
    VAPPEND1(UOCR_TENSOR_ID_VIEW_SEPARATOR,
             "view separator tensor",
             UOCR_TENSOR_FAMILY_VIEW_SEPARATOR,
             -1,
             -1,
             UOCR_TENSOR_PROJ_NONE,
             UOCR_HIDDEN_SIZE);
    VAPPEND2(UOCR_TENSOR_ID_PROJECTOR_WEIGHT,
             "visual projector weight",
             UOCR_TENSOR_FAMILY_PROJECTOR,
             -1,
             -1,
             UOCR_TENSOR_PROJ_WEIGHT,
             UOCR_HIDDEN_SIZE,
             UOCR_PROJECTOR_IN_SIZE);
    VAPPEND1(UOCR_TENSOR_ID_PROJECTOR_BIAS,
             "visual projector bias",
             UOCR_TENSOR_FAMILY_PROJECTOR,
             -1,
             -1,
             UOCR_TENSOR_PROJ_BIAS,
             UOCR_HIDDEN_SIZE);

    static const uint32_t sam_block_name_order[UOCR_SAM_BLOCKS] = {0u, 1u, 10u, 11u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
    for (uint32_t sorted_block = 0u; sorted_block < UOCR_SAM_BLOCKS; ++sorted_block) {
        const uint32_t source_block = sam_block_name_order[sorted_block];
        const uint32_t base = UOCR_TENSOR_ID_VISION_SAM_BASE + sorted_block * 14u;
        const uint32_t rel_pos_length = uocr_sam_block_uses_global_attention(source_block) ? UOCR_SAM_MAX_REL_POS_SIZE : UOCR_SAM_WINDOW_REL_POS_SIZE;
        VAPPEND1(base + 0u, "SAM attention projection bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND2(base + 1u, "SAM attention projection weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND1(base + 2u, "SAM attention qkv bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_QKV_SIZE);
        VAPPEND2(base + 3u, "SAM attention qkv weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_QKV_SIZE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND2(base + 4u, "SAM relative position H", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, rel_pos_length, UOCR_SAM_HEAD_DIM);
        VAPPEND2(base + 5u, "SAM relative position W", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, rel_pos_length, UOCR_SAM_HEAD_DIM);
        VAPPEND1(base + 6u, "SAM MLP lin1 bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_MLP_INTERMEDIATE);
        VAPPEND2(base + 7u, "SAM MLP lin1 weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_MLP_INTERMEDIATE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND1(base + 8u, "SAM MLP lin2 bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND2(base + 9u, "SAM MLP lin2 weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE, UOCR_SAM_MLP_INTERMEDIATE);
        VAPPEND1(base + 10u, "SAM norm1 bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND1(base + 11u, "SAM norm1 weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND1(base + 12u, "SAM norm2 bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE);
        VAPPEND1(base + 13u, "SAM norm2 weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE);
    }

    const uint32_t sam_extra = UOCR_TENSOR_ID_VISION_SAM_BASE + UOCR_SAM_BLOCKS * 14u;
    VAPPEND4(sam_extra + 0u, "SAM neck 1x1 conv weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NECK_CHANNELS, UOCR_SAM_HIDDEN_SIZE, 1u, 1u);
    VAPPEND1(sam_extra + 1u, "SAM neck norm1 bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NECK_CHANNELS);
    VAPPEND1(sam_extra + 2u, "SAM neck norm1 weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NECK_CHANNELS);
    VAPPEND4(sam_extra + 3u, "SAM neck 3x3 conv weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NECK_CHANNELS, UOCR_SAM_NECK_CHANNELS, UOCR_SAM_NECK_KERNEL_SIZE, UOCR_SAM_NECK_KERNEL_SIZE);
    VAPPEND1(sam_extra + 4u, "SAM neck norm2 bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NECK_CHANNELS);
    VAPPEND1(sam_extra + 5u, "SAM neck norm2 weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NECK_CHANNELS);
    VAPPEND4(sam_extra + 6u, "SAM net_2 conv weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NET2_CHANNELS, UOCR_SAM_NECK_CHANNELS, UOCR_SAM_NECK_KERNEL_SIZE, UOCR_SAM_NECK_KERNEL_SIZE);
    VAPPEND4(sam_extra + 7u, "SAM net_3 conv weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_NET3_CHANNELS, UOCR_SAM_NET2_CHANNELS, UOCR_SAM_NECK_KERNEL_SIZE, UOCR_SAM_NECK_KERNEL_SIZE);
    VAPPEND1(sam_extra + 8u, "SAM patch-embed bias", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE);
    VAPPEND4(sam_extra + 9u, "SAM patch-embed weight", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_SAM_HIDDEN_SIZE, 3u, UOCR_VISION_PATCH_SIZE, UOCR_VISION_PATCH_SIZE);
    VAPPEND4(sam_extra + 10u, "SAM absolute position embedding", UOCR_TENSOR_FAMILY_VISION_SAM, -1, -1, UOCR_TENSOR_PROJ_NONE, 1u, UOCR_SAM_MAX_GRID_SIZE, UOCR_SAM_MAX_GRID_SIZE, UOCR_SAM_HIDDEN_SIZE);

    VAPPEND1(UOCR_TENSOR_ID_VISION_CLIP_BASE + 0u,
             "CLIP class embedding",
             UOCR_TENSOR_FAMILY_VISION_CLIP,
             -1,
             -1,
             UOCR_TENSOR_PROJ_NONE,
             UOCR_CLIP_HIDDEN_SIZE);
    /* model.vision_model.embeddings.patch_embedding.weight is deliberately not
     * required for normal OCR inference: CLIP consumes SAM patch embeddings.
     */
    VAPPEND2(UOCR_TENSOR_ID_VISION_CLIP_BASE + 2u,
             "CLIP position embedding",
             UOCR_TENSOR_FAMILY_VISION_CLIP,
             -1,
             -1,
             UOCR_TENSOR_PROJ_NONE,
             UOCR_CLIP_GLOBAL_TOKENS,
             UOCR_CLIP_HIDDEN_SIZE);
    VAPPEND1(UOCR_TENSOR_ID_VISION_CLIP_BASE + 3u, "CLIP pre-LayerNorm bias", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);
    VAPPEND1(UOCR_TENSOR_ID_VISION_CLIP_BASE + 4u, "CLIP pre-LayerNorm weight", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);

    for (uint32_t sorted_layer = 0u; sorted_layer < UOCR_CLIP_BLOCKS; ++sorted_layer) {
        const uint32_t base = UOCR_TENSOR_ID_VISION_CLIP_BASE + 5u + sorted_layer * 12u;
        VAPPEND1(base + 0u, "CLIP layer_norm1 bias", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND1(base + 1u, "CLIP layer_norm1 weight", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND1(base + 2u, "CLIP layer_norm2 bias", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND1(base + 3u, "CLIP layer_norm2 weight", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND1(base + 4u, "CLIP MLP fc1 bias", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_MLP_INTERMEDIATE);
        VAPPEND2(base + 5u, "CLIP MLP fc1 weight", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_MLP_INTERMEDIATE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND1(base + 6u, "CLIP MLP fc2 bias", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND2(base + 7u, "CLIP MLP fc2 weight", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE, UOCR_CLIP_MLP_INTERMEDIATE);
        VAPPEND1(base + 8u, "CLIP attention output bias", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND2(base + 9u, "CLIP attention output weight", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_HIDDEN_SIZE, UOCR_CLIP_HIDDEN_SIZE);
        VAPPEND1(base + 10u, "CLIP attention qkv bias", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_QKV_SIZE);
        VAPPEND2(base + 11u, "CLIP attention qkv weight", UOCR_TENSOR_FAMILY_VISION_CLIP, -1, -1, UOCR_TENSOR_PROJ_NONE, UOCR_CLIP_QKV_SIZE, UOCR_CLIP_HIDDEN_SIZE);
    }

#undef VAPPEND1
#undef VAPPEND2
#undef VAPPEND4

    if (cache.count != UOCR_METAL_VISION_REQUIRED_TENSOR_COUNT) {
        return metal_fail(error,
                          error_size,
                          "vision tensor binding count mismatch: got %u expected %u",
                          cache.count,
                          (uint32_t)UOCR_METAL_VISION_REQUIRED_TENSOR_COUNT);
    }
    cache.valid = 1;
    ctx->vision_bindings = cache;
    ctx->vision_binding_error[0] = '\0';
    return 1;
}

uint32_t uocr_metal_context_vision_binding_count(const uocr_metal_context *ctx) {
    return ctx != NULL && ctx->vision_bindings.valid ? ctx->vision_bindings.count : 0u;
}

int uocr_metal_context_vision_bindings_ready(const uocr_metal_context *ctx) {
    return ctx != NULL && ctx->vision_bindings.valid &&
           ctx->vision_bindings.count == UOCR_METAL_VISION_REQUIRED_TENSOR_COUNT;
}

const char *uocr_metal_context_vision_binding_error(const uocr_metal_context *ctx) {
    if (ctx == NULL || ctx->vision_binding_error[0] == '\0') {
        return "vision tensor bindings are not validated";
    }
    return ctx->vision_binding_error;
}

static const uocr_metal_vision_binding *metal_find_vision_binding(const uocr_metal_context *ctx,
                                                                  uint32_t tensor_id) {
    if (ctx == NULL || !ctx->vision_bindings.valid || tensor_id == 0u) {
        return NULL;
    }
    for (uint32_t i = 0u; i < ctx->vision_bindings.count; ++i) {
        if (ctx->vision_bindings.tensors[i].tensor_id == tensor_id) {
            return &ctx->vision_bindings.tensors[i];
        }
    }
    return NULL;
}

static const uint16_t *metal_require_vision_tensor_f16(const uocr_metal_context *ctx,
                                                       uint32_t tensor_id,
                                                       const char *role,
                                                       char *error,
                                                       size_t error_size) {
    const uocr_metal_vision_binding *binding = metal_find_vision_binding(ctx, tensor_id);
    if (binding == NULL) {
        if (role == NULL || role[0] == '\0') {
            role = "vision tensor";
        }
        (void)metal_fail(error, error_size, "missing validated %s binding for tensor %u", role, tensor_id);
        return NULL;
    }
    if ((binding->payload_size % 2u) != 0u || ((uint64_t)binding->offset % 2u) != 0u) {
        if (role == NULL || role[0] == '\0') {
            role = "vision tensor";
        }
        (void)metal_fail(error, error_size, "%s tensor %u has an unaligned fp16 payload", role, tensor_id);
        return NULL;
    }
    if (binding->buffer == nil) {
        (void)metal_fail(error, error_size, "vision tensor %u has a nil Metal buffer binding", tensor_id);
        return NULL;
    }
    const uint8_t *contents = (const uint8_t *)[binding->buffer contents];
    if (contents == NULL) {
        (void)metal_fail(error, error_size, "vision tensor %u is not CPU-visible", tensor_id);
        return NULL;
    }
    return (const uint16_t *)(const void *)(contents + binding->offset);
}

static uint32_t metal_sam_sorted_block_index_for_layer(uint32_t layer) {
    if (layer <= 1u) {
        return layer;
    }
    if (layer <= 9u) {
        return layer + 2u;
    }
    return layer - 8u;
}

static uint32_t metal_clip_sorted_layer_index_for_layer(uint32_t layer) {
    if (layer <= 1u) {
        return layer;
    }
    if (layer >= 10u && layer <= 19u) {
        return 2u + (layer - 10u);
    }
    if (layer == 2u) {
        return 12u;
    }
    if (layer >= 20u && layer <= 23u) {
        return 13u + (layer - 20u);
    }
    return 17u + (layer - 3u);
}

typedef struct uocr_metal_vision_weights_f16 {
    const uint16_t *image_newline_f16;
    const uint16_t *view_separator_f16;
    const uint16_t *projector_weight_f16;
    const uint16_t *projector_bias_f16;
    const uint16_t *sam_patch_weight_f16;
    const uint16_t *sam_patch_bias_f16;
    const uint16_t *sam_pos_embed_f16;
    const uint16_t *sam_neck_conv1x1_weight_f16;
    const uint16_t *sam_neck_norm1_weight_f16;
    const uint16_t *sam_neck_norm1_bias_f16;
    const uint16_t *sam_neck_conv3x3_weight_f16;
    const uint16_t *sam_neck_norm2_weight_f16;
    const uint16_t *sam_neck_norm2_bias_f16;
    const uint16_t *sam_net2_weight_f16;
    const uint16_t *sam_net3_weight_f16;
    const uint16_t *clip_class_embedding_f16;
    const uint16_t *clip_pos_embed_f16;
    const uint16_t *clip_pre_ln_weight_f16;
    const uint16_t *clip_pre_ln_bias_f16;
    uocr_metal_sam_transformer_block_f16 sam_blocks[UOCR_SAM_BLOCKS];
    uocr_metal_clip_transformer_block_f16 clip_blocks[UOCR_CLIP_BLOCKS];
} uocr_metal_vision_weights_f16;

static int metal_load_vision_weights_from_bindings(uocr_metal_context *ctx,
                                                   uocr_metal_vision_weights_f16 *out_weights,
                                                   char *error,
                                                   size_t error_size) {
    if (ctx == NULL || out_weights == NULL) {
        return metal_fail(error, error_size, "invalid Metal vision weight-loading request");
    }
    memset(out_weights, 0, sizeof(*out_weights));

#define VISION_TENSOR_PTR(tensor_id, role) metal_require_vision_tensor_f16(ctx, (tensor_id), (role), error, error_size)
#define ASSIGN_PTR(field, tensor_id, role)              \
    do {                                                \
        (out_weights->field) = VISION_TENSOR_PTR((tensor_id), (role)); \
        if ((out_weights->field) == NULL) {             \
            return 0;                                   \
        }                                               \
    } while (0)

    ASSIGN_PTR(image_newline_f16, UOCR_TENSOR_ID_IMAGE_NEWLINE, "image newline");
    ASSIGN_PTR(view_separator_f16, UOCR_TENSOR_ID_VIEW_SEPARATOR, "view separator");
    ASSIGN_PTR(projector_weight_f16, UOCR_TENSOR_ID_PROJECTOR_WEIGHT, "visual projector weight");
    ASSIGN_PTR(projector_bias_f16, UOCR_TENSOR_ID_PROJECTOR_BIAS, "visual projector bias");

    const uint32_t sam_extra = UOCR_TENSOR_ID_VISION_SAM_BASE + UOCR_SAM_BLOCKS * 14u;
    ASSIGN_PTR(sam_neck_conv1x1_weight_f16, sam_extra + 0u, "SAM neck 1x1 weight");
    ASSIGN_PTR(sam_neck_norm1_bias_f16, sam_extra + 1u, "SAM neck norm1 bias");
    ASSIGN_PTR(sam_neck_norm1_weight_f16, sam_extra + 2u, "SAM neck norm1 weight");
    ASSIGN_PTR(sam_neck_conv3x3_weight_f16, sam_extra + 3u, "SAM neck 3x3 weight");
    ASSIGN_PTR(sam_neck_norm2_bias_f16, sam_extra + 4u, "SAM neck norm2 bias");
    ASSIGN_PTR(sam_neck_norm2_weight_f16, sam_extra + 5u, "SAM neck norm2 weight");
    ASSIGN_PTR(sam_net2_weight_f16, sam_extra + 6u, "SAM net_2 weight");
    ASSIGN_PTR(sam_net3_weight_f16, sam_extra + 7u, "SAM net_3 weight");
    ASSIGN_PTR(sam_patch_bias_f16, sam_extra + 8u, "SAM patch bias");
    ASSIGN_PTR(sam_patch_weight_f16, sam_extra + 9u, "SAM patch weight");
    ASSIGN_PTR(sam_pos_embed_f16, sam_extra + 10u, "SAM absolute position embedding");

    ASSIGN_PTR(clip_class_embedding_f16, UOCR_TENSOR_ID_VISION_CLIP_BASE + 0u, "CLIP class embedding");
    ASSIGN_PTR(clip_pos_embed_f16, UOCR_TENSOR_ID_VISION_CLIP_BASE + 2u, "CLIP position embedding");
    ASSIGN_PTR(clip_pre_ln_bias_f16, UOCR_TENSOR_ID_VISION_CLIP_BASE + 3u, "CLIP pre-LayerNorm bias");
    ASSIGN_PTR(clip_pre_ln_weight_f16, UOCR_TENSOR_ID_VISION_CLIP_BASE + 4u, "CLIP pre-LayerNorm weight");

    for (uint32_t layer = 0u; layer < UOCR_SAM_BLOCKS; ++layer) {
        const uint32_t base = UOCR_TENSOR_ID_VISION_SAM_BASE + metal_sam_sorted_block_index_for_layer(layer) * 14u;
        uocr_metal_sam_transformer_block_f16 *block = &out_weights->sam_blocks[layer];
        block->proj_bias_f16 = VISION_TENSOR_PTR(base + 0u, "SAM attention projection bias");
        block->proj_weight_f16 = VISION_TENSOR_PTR(base + 1u, "SAM attention projection weight");
        block->qkv_bias_f16 = VISION_TENSOR_PTR(base + 2u, "SAM attention qkv bias");
        block->qkv_weight_f16 = VISION_TENSOR_PTR(base + 3u, "SAM attention qkv weight");
        block->rel_pos_h_f16 = VISION_TENSOR_PTR(base + 4u, "SAM relative position H");
        block->rel_pos_w_f16 = VISION_TENSOR_PTR(base + 5u, "SAM relative position W");
        block->mlp_lin1_bias_f16 = VISION_TENSOR_PTR(base + 6u, "SAM MLP lin1 bias");
        block->mlp_lin1_weight_f16 = VISION_TENSOR_PTR(base + 7u, "SAM MLP lin1 weight");
        block->mlp_lin2_bias_f16 = VISION_TENSOR_PTR(base + 8u, "SAM MLP lin2 bias");
        block->mlp_lin2_weight_f16 = VISION_TENSOR_PTR(base + 9u, "SAM MLP lin2 weight");
        block->norm1_bias_f16 = VISION_TENSOR_PTR(base + 10u, "SAM norm1 bias");
        block->norm1_weight_f16 = VISION_TENSOR_PTR(base + 11u, "SAM norm1 weight");
        block->norm2_bias_f16 = VISION_TENSOR_PTR(base + 12u, "SAM norm2 bias");
        block->norm2_weight_f16 = VISION_TENSOR_PTR(base + 13u, "SAM norm2 weight");
        block->rel_pos_h_length = uocr_sam_block_uses_global_attention(layer) ? UOCR_SAM_MAX_REL_POS_SIZE : UOCR_SAM_WINDOW_REL_POS_SIZE;
        block->rel_pos_w_length = block->rel_pos_h_length;
        if (!metal_sam_transformer_block_has_weights(block)) {
            return metal_fail(error, error_size, "incomplete SAM transformer block %u vision bindings", layer);
        }
    }

    for (uint32_t layer = 0u; layer < UOCR_CLIP_BLOCKS; ++layer) {
        const uint32_t base = UOCR_TENSOR_ID_VISION_CLIP_BASE + 5u + metal_clip_sorted_layer_index_for_layer(layer) * 12u;
        uocr_metal_clip_transformer_block_f16 *block = &out_weights->clip_blocks[layer];
        block->ln1_bias_f16 = VISION_TENSOR_PTR(base + 0u, "CLIP layer_norm1 bias");
        block->ln1_weight_f16 = VISION_TENSOR_PTR(base + 1u, "CLIP layer_norm1 weight");
        block->ln2_bias_f16 = VISION_TENSOR_PTR(base + 2u, "CLIP layer_norm2 bias");
        block->ln2_weight_f16 = VISION_TENSOR_PTR(base + 3u, "CLIP layer_norm2 weight");
        block->mlp_fc1_bias_f16 = VISION_TENSOR_PTR(base + 4u, "CLIP MLP fc1 bias");
        block->mlp_fc1_weight_f16 = VISION_TENSOR_PTR(base + 5u, "CLIP MLP fc1 weight");
        block->mlp_fc2_bias_f16 = VISION_TENSOR_PTR(base + 6u, "CLIP MLP fc2 bias");
        block->mlp_fc2_weight_f16 = VISION_TENSOR_PTR(base + 7u, "CLIP MLP fc2 weight");
        block->out_proj_bias_f16 = VISION_TENSOR_PTR(base + 8u, "CLIP attention output bias");
        block->out_proj_weight_f16 = VISION_TENSOR_PTR(base + 9u, "CLIP attention output weight");
        block->qkv_bias_f16 = VISION_TENSOR_PTR(base + 10u, "CLIP attention qkv bias");
        block->qkv_weight_f16 = VISION_TENSOR_PTR(base + 11u, "CLIP attention qkv weight");
        if (!metal_clip_transformer_block_has_weights(block)) {
            return metal_fail(error, error_size, "incomplete CLIP transformer block %u vision bindings", layer);
        }
    }

#undef ASSIGN_PTR
#undef VISION_TENSOR_PTR

    return 1;
}

typedef struct uocr_metal_vision_host_scratch {
    uint16_t *sam_patch_bhwc_f16;
    uint16_t *sam_pos_bhwc_f16;
    uint16_t *sam_transformer_bhwc_f16;
    uint16_t *sam_neck_a_nchw_f16;
    uint16_t *sam_neck_b_nchw_f16;
    uint16_t *sam_net2_nchw_f16;
    uint16_t *sam_net3_nchw_f16;
    uint16_t *clip_a_f16;
    uint16_t *clip_b_f16;
    uint16_t *clip_final_f16;
    uint16_t *concat_f16;
} uocr_metal_vision_host_scratch;

static void metal_vision_host_scratch_free(uocr_metal_vision_host_scratch *scratch) {
    if (scratch == NULL) {
        return;
    }
    free(scratch->concat_f16);
    free(scratch->clip_final_f16);
    free(scratch->clip_b_f16);
    free(scratch->clip_a_f16);
    free(scratch->sam_net3_nchw_f16);
    free(scratch->sam_net2_nchw_f16);
    free(scratch->sam_neck_b_nchw_f16);
    free(scratch->sam_neck_a_nchw_f16);
    free(scratch->sam_transformer_bhwc_f16);
    free(scratch->sam_pos_bhwc_f16);
    free(scratch->sam_patch_bhwc_f16);
    memset(scratch, 0, sizeof(*scratch));
}

static int metal_alloc_f16_values(uint16_t **out,
                                  uint64_t value_count,
                                  const char *label,
                                  char *error,
                                  size_t error_size) {
    if (out == NULL || value_count == 0u) {
        return metal_fail(error, error_size, "invalid Metal vision scratch allocation request");
    }
    uint64_t bytes = 0u;
    if (!checked_mul_u64(value_count, 2u, &bytes) || bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal vision scratch allocation for %s overflows", label != NULL ? label : "buffer");
    }
    *out = (uint16_t *)malloc((size_t)bytes);
    if (*out == NULL) {
        return metal_fail(error,
                          error_size,
                          "failed to allocate Metal vision scratch buffer %s (%llu bytes)",
                          label != NULL ? label : "buffer",
                          (unsigned long long)bytes);
    }
    return 1;
}

static int metal_vision_host_scratch_init(uocr_metal_vision_host_scratch *scratch,
                                          char *error,
                                          size_t error_size) {
    if (scratch == NULL) {
        return metal_fail(error, error_size, "Metal vision scratch output is null");
    }
    memset(scratch, 0, sizeof(*scratch));
    const uint64_t sam_bhwc_values = (uint64_t)UOCR_SAM_MAX_GRID_TOKENS * (uint64_t)UOCR_SAM_HIDDEN_SIZE;
    const uint64_t sam_neck_values = (uint64_t)UOCR_SAM_NECK_CHANNELS * (uint64_t)UOCR_SAM_MAX_GRID_TOKENS;
    const uint64_t max_net2_grid = (uint64_t)((UOCR_SAM_MAX_GRID_SIZE + 1u) / 2u);
    const uint64_t max_net3_grid = (max_net2_grid + 1u) / 2u;
    const uint64_t sam_net2_values = (uint64_t)UOCR_SAM_NET2_CHANNELS * max_net2_grid * max_net2_grid;
    const uint64_t sam_net3_values = (uint64_t)UOCR_SAM_NET3_CHANNELS * max_net3_grid * max_net3_grid;
    const uint64_t clip_values = (uint64_t)UOCR_CLIP_MAX_TOKENS * (uint64_t)UOCR_CLIP_HIDDEN_SIZE;
    const uint64_t concat_values = (uint64_t)UOCR_GLOBAL_GRID_QUERIES * (uint64_t)UOCR_GLOBAL_GRID_QUERIES * (uint64_t)UOCR_PROJECTOR_IN_SIZE;

#define ALLOC_SCRATCH(field, values, label)                              \
    do {                                                                 \
        if (!metal_alloc_f16_values(&(scratch->field), (values), (label), error, error_size)) { \
            metal_vision_host_scratch_free(scratch);                     \
            return 0;                                                    \
        }                                                                \
    } while (0)

    ALLOC_SCRATCH(sam_patch_bhwc_f16, sam_bhwc_values, "SAM patch BHWC");
    ALLOC_SCRATCH(sam_pos_bhwc_f16, sam_bhwc_values, "SAM positioned BHWC");
    ALLOC_SCRATCH(sam_transformer_bhwc_f16, sam_bhwc_values, "SAM transformer BHWC");
    ALLOC_SCRATCH(sam_neck_a_nchw_f16, sam_neck_values, "SAM neck A NCHW");
    ALLOC_SCRATCH(sam_neck_b_nchw_f16, sam_neck_values, "SAM neck B NCHW");
    ALLOC_SCRATCH(sam_net2_nchw_f16, sam_net2_values, "SAM net_2 NCHW");
    ALLOC_SCRATCH(sam_net3_nchw_f16, sam_net3_values, "SAM net_3 NCHW");
    ALLOC_SCRATCH(clip_a_f16, clip_values, "CLIP token A");
    ALLOC_SCRATCH(clip_b_f16, clip_values, "CLIP token B");
    ALLOC_SCRATCH(clip_final_f16, clip_values, "CLIP transformer output");
    ALLOC_SCRATCH(concat_f16, concat_values, "CLIP/SAM concat");

#undef ALLOC_SCRATCH

    return 1;
}

typedef struct uocr_metal_vision_project_context {
    uocr_metal_context *ctx;
    const uocr_prepared_request *request;
    const uocr_metal_vision_weights_f16 *weights;
    uocr_metal_vision_host_scratch *scratch;
} uocr_metal_vision_project_context;

static int metal_encode_one_view_projected_f16(uocr_metal_vision_project_context *project,
                                               const uocr_image_view *view,
                                               uint16_t *out_projected_rows_f16,
                                               char *error,
                                               size_t error_size) {
    if (project == NULL || project->ctx == NULL || project->weights == NULL || project->scratch == NULL ||
        view == NULL || out_projected_rows_f16 == NULL) {
        return metal_fail(error, error_size, "invalid Metal vision view-encoding request");
    }
    const uint32_t expected_patch_grid = view->width / UOCR_VISION_PATCH_SIZE;
    const uint32_t expected_projected_grid = view->kind == UOCR_VIEW_LOCAL ? UOCR_LOCAL_GRID_QUERIES : UOCR_GLOBAL_GRID_QUERIES;
    const uint32_t expected_clip_tokens = UOCR_CLIP_CLASS_TOKENS + expected_projected_grid * expected_projected_grid;
    const uocr_metal_vision_weights_f16 *weights = project->weights;
    uocr_metal_vision_host_scratch *scratch = project->scratch;
    uocr_metal_context *ctx = project->ctx;

#define RUN_VISION_STEP(step_name, call_expr)                                                        \
    do {                                                                                             \
        if (!(call_expr)) {                                                                          \
            char detail[512];                                                                        \
            metal_copy_error_detail(detail, sizeof(detail), error);                                   \
            return metal_fail(error, error_size, "failed to compute Metal vision %s: %s", step_name, detail); \
        }                                                                                            \
    } while (0)

    uint32_t patch_grid_w = 0u;
    uint32_t patch_grid_h = 0u;
    RUN_VISION_STEP("SAM patch embedding",
                    uocr_metal_context_sam_patch_embed_f16(ctx,
                                                            view->pixels,
                                                            view->format,
                                                            view->width,
                                                            view->height,
                                                            weights->sam_patch_weight_f16,
                                                            weights->sam_patch_bias_f16,
                                                            scratch->sam_patch_bhwc_f16,
                                                            &patch_grid_w,
                                                            &patch_grid_h,
                                                            error,
                                                            error_size));
    if (patch_grid_w != expected_patch_grid || patch_grid_h != expected_patch_grid) {
        return metal_fail(error,
                          error_size,
                          "SAM patch embedding produced grid %ux%u, expected %ux%u",
                          patch_grid_w,
                          patch_grid_h,
                          expected_patch_grid,
                          expected_patch_grid);
    }

    RUN_VISION_STEP("SAM absolute position add",
                    uocr_metal_context_sam_add_abs_pos_f16(ctx,
                                                            scratch->sam_patch_bhwc_f16,
                                                            weights->sam_pos_embed_f16,
                                                            patch_grid_w,
                                                            patch_grid_h,
                                                            scratch->sam_pos_bhwc_f16,
                                                            error,
                                                            error_size));
    RUN_VISION_STEP("SAM transformer",
                    uocr_metal_context_sam_transformer_f16(ctx,
                                                            scratch->sam_pos_bhwc_f16,
                                                            weights->sam_blocks,
                                                            UOCR_SAM_BLOCKS,
                                                            patch_grid_w,
                                                            patch_grid_h,
                                                            UOCR_METAL_DENSE_OUTPUT_F16,
                                                            scratch->sam_transformer_bhwc_f16,
                                                            error,
                                                            error_size));
    RUN_VISION_STEP("SAM neck 1x1 convolution",
                    uocr_metal_context_sam_neck_conv1x1_f16(ctx,
                                                             scratch->sam_transformer_bhwc_f16,
                                                             weights->sam_neck_conv1x1_weight_f16,
                                                             patch_grid_w,
                                                             patch_grid_h,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             scratch->sam_neck_a_nchw_f16,
                                                             error,
                                                             error_size));
    RUN_VISION_STEP("SAM neck LayerNorm2d-1",
                    uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                            scratch->sam_neck_a_nchw_f16,
                                                            weights->sam_neck_norm1_weight_f16,
                                                            weights->sam_neck_norm1_bias_f16,
                                                            patch_grid_w,
                                                            patch_grid_h,
                                                            UOCR_METAL_DENSE_OUTPUT_F16,
                                                            scratch->sam_neck_b_nchw_f16,
                                                            error,
                                                            error_size));
    RUN_VISION_STEP("SAM neck 3x3 convolution",
                    uocr_metal_context_sam_neck_conv3x3_f16(ctx,
                                                             scratch->sam_neck_b_nchw_f16,
                                                             weights->sam_neck_conv3x3_weight_f16,
                                                             patch_grid_w,
                                                             patch_grid_h,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             scratch->sam_neck_a_nchw_f16,
                                                             error,
                                                             error_size));
    RUN_VISION_STEP("SAM neck LayerNorm2d-2",
                    uocr_metal_context_sam_layernorm2d_f16(ctx,
                                                            scratch->sam_neck_a_nchw_f16,
                                                            weights->sam_neck_norm2_weight_f16,
                                                            weights->sam_neck_norm2_bias_f16,
                                                            patch_grid_w,
                                                            patch_grid_h,
                                                            UOCR_METAL_DENSE_OUTPUT_F16,
                                                            scratch->sam_neck_b_nchw_f16,
                                                            error,
                                                            error_size));
    RUN_VISION_STEP("SAM net_2 convolution",
                    uocr_metal_context_sam_net2_conv3x3_stride2_f16(ctx,
                                                                     scratch->sam_neck_b_nchw_f16,
                                                                     weights->sam_net2_weight_f16,
                                                                     patch_grid_w,
                                                                     patch_grid_h,
                                                                     UOCR_METAL_DENSE_OUTPUT_F16,
                                                                     scratch->sam_net2_nchw_f16,
                                                                     error,
                                                                     error_size));
    const uint32_t net2_grid_w = (patch_grid_w + 1u) / 2u;
    const uint32_t net2_grid_h = (patch_grid_h + 1u) / 2u;
    RUN_VISION_STEP("SAM net_3 convolution",
                    uocr_metal_context_sam_net3_conv3x3_stride2_f16(ctx,
                                                                     scratch->sam_net2_nchw_f16,
                                                                     weights->sam_net3_weight_f16,
                                                                     net2_grid_w,
                                                                     net2_grid_h,
                                                                     UOCR_METAL_DENSE_OUTPUT_F16,
                                                                     scratch->sam_net3_nchw_f16,
                                                                     error,
                                                                     error_size));
    const uint32_t sam_grid_w = (net2_grid_w + 1u) / 2u;
    const uint32_t sam_grid_h = (net2_grid_h + 1u) / 2u;
    if (sam_grid_w != expected_projected_grid || sam_grid_h != expected_projected_grid) {
        return metal_fail(error,
                          error_size,
                          "SAM net output grid %ux%u, expected %ux%u",
                          sam_grid_w,
                          sam_grid_h,
                          expected_projected_grid,
                          expected_projected_grid);
    }

    RUN_VISION_STEP("CLIP SAM embedding",
                    uocr_metal_context_clip_embed_sam_f16(ctx,
                                                           scratch->sam_net3_nchw_f16,
                                                           weights->clip_class_embedding_f16,
                                                           sam_grid_w,
                                                           sam_grid_h,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           scratch->clip_a_f16,
                                                           error,
                                                           error_size));
    RUN_VISION_STEP("CLIP absolute position add",
                    uocr_metal_context_clip_add_abs_pos_f16(ctx,
                                                             scratch->clip_a_f16,
                                                             weights->clip_pos_embed_f16,
                                                             sam_grid_w,
                                                             sam_grid_h,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             scratch->clip_b_f16,
                                                             error,
                                                             error_size));
    RUN_VISION_STEP("CLIP pre-LayerNorm",
                    uocr_metal_context_clip_pre_layernorm_f16(ctx,
                                                               scratch->clip_b_f16,
                                                               weights->clip_pre_ln_weight_f16,
                                                               weights->clip_pre_ln_bias_f16,
                                                               expected_clip_tokens,
                                                               UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                                               scratch->clip_a_f16,
                                                               error,
                                                               error_size));
    RUN_VISION_STEP("CLIP transformer",
                    uocr_metal_context_clip_transformer_f16(ctx,
                                                             scratch->clip_a_f16,
                                                             weights->clip_blocks,
                                                             UOCR_CLIP_BLOCKS,
                                                             expected_clip_tokens,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             scratch->clip_final_f16,
                                                             error,
                                                             error_size));
    RUN_VISION_STEP("CLIP/SAM concat",
                    uocr_metal_context_clip_sam_concat_f16(ctx,
                                                            scratch->clip_final_f16,
                                                            scratch->sam_net3_nchw_f16,
                                                            sam_grid_w,
                                                            sam_grid_h,
                                                            UOCR_METAL_DENSE_OUTPUT_F16,
                                                            scratch->concat_f16,
                                                            error,
                                                            error_size));
    RUN_VISION_STEP("visual projector",
                    uocr_metal_context_visual_projector_f16(ctx,
                                                             scratch->concat_f16,
                                                             weights->projector_weight_f16,
                                                             weights->projector_bias_f16,
                                                             sam_grid_w * sam_grid_h,
                                                             UOCR_METAL_DENSE_OUTPUT_F16,
                                                             out_projected_rows_f16,
                                                             error,
                                                             error_size));

#undef RUN_VISION_STEP

    return 1;
}

static int metal_project_vision_chunk_f16(const uocr_vision_chunk *chunk,
                                          uint16_t *projected_scratch_f16,
                                          uint32_t projected_scratch_rows,
                                          void *user_data,
                                          char *error,
                                          size_t error_size) {
    if (chunk == NULL || projected_scratch_f16 == NULL || user_data == NULL) {
        (void)metal_fail(error, error_size, "invalid Metal vision chunk projection request");
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    if (projected_scratch_rows < chunk->projected_token_count) {
        (void)metal_fail(error,
                         error_size,
                         "Metal vision projected scratch has %u rows, need %u",
                         projected_scratch_rows,
                         chunk->projected_token_count);
        return UOCR_ERROR_OUT_OF_MEMORY;
    }

    uocr_metal_vision_project_context *project = (uocr_metal_vision_project_context *)user_data;
    if (project->request == NULL || chunk->first_view > project->request->n_views ||
        chunk->view_count > project->request->n_views - chunk->first_view) {
        (void)metal_fail(error, error_size, "Metal vision chunk view range is invalid");
        return UOCR_ERROR_INVALID_ARGUMENT;
    }

    for (uint32_t view_index = 0u; view_index < chunk->view_count; ++view_index) {
        const uocr_image_view *view = &project->request->views[chunk->first_view + view_index];
        uint16_t *projected_out = &projected_scratch_f16[(size_t)view_index * (size_t)chunk->projected_tokens_per_view * (size_t)UOCR_HIDDEN_SIZE];
        if (!metal_encode_one_view_projected_f16(project, view, projected_out, error, error_size)) {
            return UOCR_ERROR_INTERNAL;
        }
    }

    return UOCR_OK;
}

int uocr_metal_context_encode_visual_features_f16(uocr_metal_context *ctx,
                                                  const uocr_prepared_request *request,
                                                  uint32_t max_views_per_chunk,
                                                  uint16_t *out_visual_features_f16,
                                                  uint32_t out_visual_rows,
                                                  char *error,
                                                  size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || request == NULL) {
        return metal_fail(error, error_size, "Metal vision encoding requires a context and prepared request");
    }
    if (!uocr_metal_context_vision_bindings_ready(ctx)) {
        return metal_fail(error,
                          error_size,
                          "Metal vision encoding requires validated fp16 vision tensor bindings: %s",
                          uocr_metal_context_vision_binding_error(ctx));
    }
    if (out_visual_rows != 0u && out_visual_features_f16 == NULL) {
        return metal_fail(error, error_size, "Metal vision output buffer is null for %u rows", out_visual_rows);
    }

    uocr_vision_schedule schedule;
    memset(&schedule, 0, sizeof(schedule));
    const int plan_status = uocr_plan_vision_schedule(request,
                                                      max_views_per_chunk,
                                                      NULL,
                                                      0u,
                                                      &schedule,
                                                      error,
                                                      error_size);
    if (plan_status != UOCR_OK) {
        char detail[512];
        metal_copy_error_detail(detail, sizeof(detail), error);
        return metal_fail(error, error_size, "invalid Metal vision request: %s", detail);
    }
    if (out_visual_rows != schedule.final_visual_tokens) {
        return metal_fail(error,
                          error_size,
                          "Metal vision output row count mismatch: got %u expected %u",
                          out_visual_rows,
                          schedule.final_visual_tokens);
    }
    if (schedule.final_visual_tokens == 0u) {
        metal_clear_error(error, error_size);
        return 1;
    }

    uocr_metal_vision_weights_f16 weights;
    if (!metal_load_vision_weights_from_bindings(ctx, &weights, error, error_size)) {
        return 0;
    }

    uocr_metal_vision_host_scratch scratch;
    if (!metal_vision_host_scratch_init(&scratch, error, error_size)) {
        return 0;
    }

    uint16_t *projected_scratch_f16 = NULL;
    if (!metal_alloc_f16_values(&projected_scratch_f16,
                                (uint64_t)schedule.max_chunk_projected_tokens * (uint64_t)UOCR_HIDDEN_SIZE,
                                "projected visual chunk rows",
                                error,
                                error_size)) {
        metal_vision_host_scratch_free(&scratch);
        return 0;
    }

    uocr_metal_vision_project_context project;
    memset(&project, 0, sizeof(project));
    project.ctx = ctx;
    project.request = request;
    project.weights = &weights;
    project.scratch = &scratch;

    const int process_status = uocr_process_vision_chunks_f16(request,
                                                              max_views_per_chunk,
                                                              metal_project_vision_chunk_f16,
                                                              &project,
                                                              projected_scratch_f16,
                                                              schedule.max_chunk_projected_tokens,
                                                              weights.image_newline_f16,
                                                              weights.view_separator_f16,
                                                              out_visual_features_f16,
                                                              out_visual_rows,
                                                              NULL,
                                                              error,
                                                              error_size);
    free(projected_scratch_f16);
    metal_vision_host_scratch_free(&scratch);
    if (process_status != UOCR_OK) {
        char detail[512];
        metal_copy_error_detail(detail, sizeof(detail), error);
        return metal_fail(error, error_size, "Metal vision encoding failed: %s", detail);
    }

    metal_clear_error(error, error_size);
    return 1;
}

static int scratch_slot_valid(uocr_metal_scratch_slot slot) {
    return slot >= UOCR_METAL_SCRATCH_VISION && slot < UOCR_METAL_SCRATCH_COUNT;
}

static const char *scratch_slot_name(uocr_metal_scratch_slot slot) {
    switch (slot) {
        case UOCR_METAL_SCRATCH_VISION:
            return "vision";
        case UOCR_METAL_SCRATCH_DECODER:
            return "decoder";
        case UOCR_METAL_SCRATCH_MOE:
            return "moe";
        case UOCR_METAL_SCRATCH_LOGITS:
            return "logits";
        case UOCR_METAL_SCRATCH_TRANSIENT:
            return "transient";
        default:
            return "unknown";
    }
}

static int scratch_capacity_for_request(uint64_t current_capacity, uint64_t min_length, uint64_t *out_capacity) {
    const uint64_t min_allocation = 4096u;
    uint64_t requested = 0u;
    if (!align_up_u64_checked(min_length, 256u, &requested)) {
        return 0;
    }
    if (requested < min_allocation) {
        requested = min_allocation;
    }

    uint64_t capacity = current_capacity;
    if (capacity < min_allocation) {
        capacity = min_allocation;
    }
    while (capacity < requested) {
        if (capacity > UINT64_MAX / 2u) {
            capacity = requested;
            break;
        }
        capacity *= 2u;
    }
    *out_capacity = capacity;
    return 1;
}

int uocr_metal_context_ensure_scratch(uocr_metal_context *ctx,
                                      uocr_metal_scratch_slot slot,
                                      uint64_t min_length,
                                      int cpu_visible,
                                      char *error,
                                      size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || !scratch_slot_valid(slot)) {
        return metal_fail(error, error_size, "invalid Metal scratch buffer request");
    }
    if (min_length == 0u) {
        return 1;
    }
    if (min_length > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal scratch request for %s is too large: %llu bytes",
                          scratch_slot_name(slot),
                          (unsigned long long)min_length);
    }

    uocr_metal_scratch_buffer *scratch = &ctx->scratch[slot];
    const int wants_cpu_visible = cpu_visible ? 1 : 0;
    if (scratch->buffer != nil && scratch->capacity >= min_length && scratch->cpu_visible == wants_cpu_visible) {
        if (scratch->high_watermark < min_length) {
            scratch->high_watermark = min_length;
        }
        return 1;
    }

    uint64_t new_capacity = 0u;
    if (!scratch_capacity_for_request(scratch->capacity, min_length, &new_capacity) || new_capacity > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal scratch capacity overflow for %s request of %llu bytes",
                          scratch_slot_name(slot),
                          (unsigned long long)min_length);
    }

    @autoreleasepool {
        const MTLResourceOptions storage_mode = wants_cpu_visible ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        id<MTLBuffer> buffer = [ctx->device newBufferWithLength:(NSUInteger)new_capacity options:storage_mode];
        if (buffer == nil) {
            return metal_fail(error,
                              error_size,
                              "failed to allocate %s Metal scratch buffer with %llu bytes",
                              scratch_slot_name(slot),
                              (unsigned long long)new_capacity);
        }
        buffer.label = [NSString stringWithFormat:@"uocr_scratch_%s", scratch_slot_name(slot)];
        [scratch->buffer release];
        scratch->buffer = buffer;
        scratch->capacity = new_capacity;
        scratch->cpu_visible = wants_cpu_visible;
        if (scratch->high_watermark < min_length) {
            scratch->high_watermark = min_length;
        }
    }
    return 1;
}

void uocr_metal_context_release_scratch(uocr_metal_context *ctx, uocr_metal_scratch_slot slot) {
    if (ctx == NULL || !scratch_slot_valid(slot)) {
        return;
    }
    @autoreleasepool {
        uocr_metal_scratch_buffer *scratch = &ctx->scratch[slot];
        [scratch->buffer release];
        scratch->buffer = nil;
        scratch->capacity = 0u;
        scratch->cpu_visible = 0;
    }
}

uint64_t uocr_metal_context_scratch_capacity(const uocr_metal_context *ctx, uocr_metal_scratch_slot slot) {
    if (ctx == NULL || !scratch_slot_valid(slot)) {
        return 0u;
    }
    return ctx->scratch[slot].capacity;
}

uint64_t uocr_metal_context_scratch_high_watermark(const uocr_metal_context *ctx, uocr_metal_scratch_slot slot) {
    if (ctx == NULL || !scratch_slot_valid(slot)) {
        return 0u;
    }
    return ctx->scratch[slot].high_watermark;
}

uint64_t uocr_metal_context_total_scratch_capacity(const uocr_metal_context *ctx) {
    uint64_t total = 0u;
    if (ctx == NULL) {
        return 0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)UOCR_METAL_SCRATCH_COUNT; ++i) {
        if (!checked_add_u64(total, ctx->scratch[i].capacity, &total)) {
            return UINT64_MAX;
        }
    }
    return total;
}

uint64_t uocr_metal_context_total_scratch_high_watermark(const uocr_metal_context *ctx) {
    uint64_t total = 0u;
    if (ctx == NULL) {
        return 0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)UOCR_METAL_SCRATCH_COUNT; ++i) {
        if (!checked_add_u64(total, ctx->scratch[i].high_watermark, &total)) {
            return UINT64_MAX;
        }
    }
    return total;
}

static void release_all_scratch(uocr_metal_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    for (uint32_t i = 0u; i < (uint32_t)UOCR_METAL_SCRATCH_COUNT; ++i) {
        uocr_metal_context_release_scratch(ctx, (uocr_metal_scratch_slot)i);
    }
}

int uocr_metal_context_warmup_model_views(uocr_metal_context *ctx,
                                          uint64_t max_bytes_per_view,
                                          char *error,
                                          size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL) {
        return metal_fail(error, error_size, "Metal model-view warmup requires a context");
    }
    if (ctx->model_views == NULL || ctx->model_view_count == 0u) {
        return metal_fail(error, error_size, "Metal model-view warmup requires mapped model views");
    }

    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        page_size_long = (long)UOCR_TENSOR_DATA_ALIGNMENT;
    }
    uint64_t stride_words_u64 = (uint64_t)page_size_long / sizeof(uint32_t);
    if (stride_words_u64 == 0u) {
        stride_words_u64 = 1u;
    }
    if (stride_words_u64 > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal warmup page stride is too large");
    }
    const uint32_t stride_words = (uint32_t)stride_words_u64;

    uint64_t total_probe_count = 0u;
    uint64_t total_warmup_bytes = 0u;
    for (uint32_t i = 0u; i < ctx->model_view_count; ++i) {
        uint64_t bytes = ctx->model_views[i].length;
        if (max_bytes_per_view != 0u && bytes > max_bytes_per_view) {
            bytes = max_bytes_per_view;
        }
        const uint64_t words = bytes / sizeof(uint32_t);
        if (words == 0u) {
            continue;
        }
        if (words > UINT32_MAX) {
            return metal_fail(error, error_size, "Metal model-view warmup view %u is too large for u32 probe indexing", i);
        }
        const uint64_t probes = (words + (uint64_t)stride_words - 1u) / (uint64_t)stride_words;
        if (probes > UINT32_MAX || total_probe_count > (uint64_t)UINT32_MAX - probes) {
            return metal_fail(error, error_size, "Metal model-view warmup would need too many probe outputs");
        }
        if (!checked_add_u64(total_warmup_bytes, words * sizeof(uint32_t), &total_warmup_bytes)) {
            return metal_fail(error, error_size, "Metal model-view warmup byte count overflow");
        }
        total_probe_count += probes;
    }

    if (total_probe_count == 0u) {
        ctx->last_model_warmup_bytes = 0u;
        return 1;
    }
    const uint64_t output_bytes = total_probe_count * sizeof(uint32_t);
    if (!uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_TRANSIENT, output_bytes, 1, error, error_size)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_touch_u32", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        id<MTLBuffer> dst = ctx->scratch[UOCR_METAL_SCRATCH_TRANSIENT].buffer;
        if (dst == nil || [dst contents] == NULL) {
            return metal_fail(error, error_size, "Metal model-view warmup has no CPU-visible scratch output");
        }
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create Metal warmup command buffer");
        }
        cb.label = @"uocr_model_view_warmup_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create Metal warmup command encoder");
        }
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:dst offset:0u atIndex:1u];

        uint32_t dst_base = 0u;
        for (uint32_t i = 0u; i < ctx->model_view_count; ++i) {
            uint64_t bytes = ctx->model_views[i].length;
            if (max_bytes_per_view != 0u && bytes > max_bytes_per_view) {
                bytes = max_bytes_per_view;
            }
            const uint64_t words = bytes / sizeof(uint32_t);
            if (words == 0u) {
                continue;
            }
            const uint32_t probes = (uint32_t)((words + (uint64_t)stride_words - 1u) / (uint64_t)stride_words);
            uocr_metal_touch_params params;
            params.n_words = (uint32_t)words;
            params.stride_words = stride_words;
            params.dst_base = dst_base;
            params.reserved = 0u;

            [enc setBuffer:ctx->model_views[i].buffer offset:0u atIndex:0u];
            [enc setBytes:&params length:sizeof(params) atIndex:2u];

            NSUInteger threads_per_group = pipeline.threadExecutionWidth;
            if (threads_per_group == 0u) {
                threads_per_group = 1u;
            }
            if (threads_per_group > (NSUInteger)probes) {
                threads_per_group = (NSUInteger)probes;
            }
            [enc dispatchThreads:MTLSizeMake((NSUInteger)probes, 1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
            dst_base += probes;
        }
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            return metal_fail(error, error_size, "Metal model-view warmup failed: %s", [description UTF8String]);
        }
    }

    ctx->last_model_warmup_bytes = total_warmup_bytes;
    metal_clear_error(error, error_size);
    return 1;
}

uint64_t uocr_metal_context_last_warmup_bytes(const uocr_metal_context *ctx) {
    return ctx != NULL ? ctx->last_model_warmup_bytes : 0u;
}

static int runtime_arena_slot_valid(uocr_metal_runtime_arena_slot slot) {
    return slot >= UOCR_METAL_ARENA_KV_CACHE && slot < UOCR_METAL_ARENA_COUNT;
}

static const char *runtime_arena_slot_name(uocr_metal_runtime_arena_slot slot) {
    switch (slot) {
        case UOCR_METAL_ARENA_KV_CACHE:
            return "kv-cache";
        case UOCR_METAL_ARENA_PROMPT_EMBEDDINGS:
            return "prompt-embeddings";
        case UOCR_METAL_ARENA_HIDDEN_PINGPONG:
            return "hidden-pingpong";
        case UOCR_METAL_ARENA_ROUTER_TOPK:
            return "router-topk";
        case UOCR_METAL_ARENA_MOE_INTERMEDIATE:
            return "moe-intermediate";
        case UOCR_METAL_ARENA_VISION_SCRATCH:
            return "vision-scratch";
        case UOCR_METAL_ARENA_LOGITS_READBACK:
            return "logits-readback";
        default:
            return "unknown";
    }
}

static int runtime_arena_is_cpu_visible(uocr_metal_runtime_arena_slot slot) {
    return slot == UOCR_METAL_ARENA_LOGITS_READBACK;
}

static int metal_init_kv_cache_layout(uint32_t batch_slots,
                                      uint32_t prompt_token_capacity,
                                      uocr_metal_kv_cache_layout *out_layout) {
    if (out_layout == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return 0;
    }

    uint64_t cache_token_capacity = 0u;
    uint64_t token_stride = 0u;
    uint64_t slot_stride = 0u;
    uint64_t layer_stride = 0u;
    uint64_t tensor_bytes = 0u;
    uint64_t total_bytes = 0u;
    if (!checked_add_u64((uint64_t)prompt_token_capacity,
                         (uint64_t)UOCR_GENERATED_RING_WINDOW,
                         &cache_token_capacity) ||
        cache_token_capacity > (uint64_t)UINT32_MAX ||
        !checked_mul_u64((uint64_t)UOCR_KV_HEADS, (uint64_t)UOCR_HEAD_DIM, &token_stride) ||
        !checked_mul_u64(token_stride, 2u, &token_stride) ||
        !checked_mul_u64(cache_token_capacity, token_stride, &slot_stride) ||
        !checked_mul_u64((uint64_t)batch_slots, slot_stride, &layer_stride) ||
        !checked_mul_u64((uint64_t)UOCR_DECODER_LAYERS, layer_stride, &tensor_bytes) ||
        !checked_mul_u64(tensor_bytes, 2u, &total_bytes)) {
        return 0;
    }

    memset(out_layout, 0, sizeof(*out_layout));
    out_layout->decoder_layers = UOCR_DECODER_LAYERS;
    out_layout->batch_slots = batch_slots;
    out_layout->prompt_token_capacity = prompt_token_capacity;
    out_layout->cache_token_capacity = (uint32_t)cache_token_capacity;
    out_layout->kv_heads = UOCR_KV_HEADS;
    out_layout->head_dim = UOCR_HEAD_DIM;
    out_layout->generated_ring_window = UOCR_GENERATED_RING_WINDOW;
    out_layout->token_stride_bytes = token_stride;
    out_layout->slot_stride_bytes = slot_stride;
    out_layout->layer_stride_bytes = layer_stride;
    out_layout->tensor_bytes = tensor_bytes;
    out_layout->k_offset_bytes = 0u;
    out_layout->v_offset_bytes = tensor_bytes;
    out_layout->total_bytes = total_bytes;
    return 1;
}

static int runtime_arena_capacities(uint32_t batch_slots,
                                    uint32_t prompt_token_capacity,
                                    uint64_t capacities[UOCR_METAL_ARENA_COUNT]) {
    if (capacities == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    memset(capacities, 0, (size_t)UOCR_METAL_ARENA_COUNT * sizeof(capacities[0]));

    int status = uocr_estimate_kv_cache_bytes(batch_slots, prompt_token_capacity, &capacities[UOCR_METAL_ARENA_KV_CACHE]);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_prompt_embedding_bytes(batch_slots,
                                                  prompt_token_capacity,
                                                  &capacities[UOCR_METAL_ARENA_PROMPT_EMBEDDINGS]);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_decoder_scratch_bytes(batch_slots,
                                                 prompt_token_capacity,
                                                 &capacities[UOCR_METAL_ARENA_HIDDEN_PINGPONG]);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_moe_router_topk_bytes(batch_slots,
                                                 prompt_token_capacity,
                                                 &capacities[UOCR_METAL_ARENA_ROUTER_TOPK]);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_moe_intermediate_bytes(batch_slots,
                                                  prompt_token_capacity,
                                                  &capacities[UOCR_METAL_ARENA_MOE_INTERMEDIATE]);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_vision_scratch_bytes_for_rows(prompt_token_capacity,
                                                         UOCR_GLOBAL_GRID_QUERIES * UOCR_GLOBAL_GRID_QUERIES,
                                                         &capacities[UOCR_METAL_ARENA_VISION_SCRATCH]);
    if (status != UOCR_OK) {
        return status;
    }
    status = uocr_estimate_logits_readback_bytes(batch_slots, &capacities[UOCR_METAL_ARENA_LOGITS_READBACK]);
    if (status != UOCR_OK) {
        return status;
    }
    return UOCR_OK;
}

void uocr_metal_context_release_runtime_arenas(uocr_metal_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    @autoreleasepool {
        for (uint32_t i = 0u; i < (uint32_t)UOCR_METAL_ARENA_COUNT; ++i) {
            uocr_metal_runtime_arena *arena = &ctx->runtime_arenas[i];
            [arena->buffer release];
            arena->buffer = nil;
            arena->capacity = 0u;
            arena->cpu_visible = 0;
        }
        memset(&ctx->kv_cache_layout, 0, sizeof(ctx->kv_cache_layout));
        ctx->has_kv_cache_layout = 0;
        ctx->has_integrated_prefill = 0;
        ctx->integrated_prefill_slot = 0u;
        ctx->integrated_prefill_tokens = 0u;
        ctx->integrated_prefill_final_segment = 0u;
    }
}

int uocr_metal_context_allocate_runtime_arenas(uocr_metal_context *ctx,
                                               uint32_t batch_slots,
                                               uint32_t prompt_token_capacity,
                                               char *error,
                                               size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || batch_slots == 0u || prompt_token_capacity == 0u) {
        return metal_fail(error, error_size, "invalid Metal runtime arena allocation request");
    }

    uint64_t capacities[UOCR_METAL_ARENA_COUNT];
    const int estimate_status = runtime_arena_capacities(batch_slots, prompt_token_capacity, capacities);
    if (estimate_status != UOCR_OK) {
        return metal_fail(error,
                          error_size,
                          "failed to estimate Metal runtime arena sizes: %s",
                          uocr_status_string(estimate_status));
    }

    uocr_metal_kv_cache_layout kv_layout;
    if (!metal_init_kv_cache_layout(batch_slots, prompt_token_capacity, &kv_layout) ||
        kv_layout.total_bytes != capacities[UOCR_METAL_ARENA_KV_CACHE]) {
        return metal_fail(error, error_size, "failed to build Metal KV cache layout");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (max_buffer_length == 0u) {
        return metal_fail(error, error_size, "Metal device reported maxBufferLength=0");
    }

    uocr_metal_context_release_runtime_arenas(ctx);
    @autoreleasepool {
        for (uint32_t i = 0u; i < (uint32_t)UOCR_METAL_ARENA_COUNT; ++i) {
            const uocr_metal_runtime_arena_slot slot = (uocr_metal_runtime_arena_slot)i;
            const uint64_t bytes = capacities[i];
            if (bytes == 0u) {
                continue;
            }
            if (bytes > max_buffer_length || bytes > (uint64_t)SIZE_MAX) {
                uocr_metal_context_release_runtime_arenas(ctx);
                return metal_fail(error,
                                  error_size,
                                  "Metal runtime arena %s needs %llu bytes, exceeding maxBufferLength %llu",
                                  runtime_arena_slot_name(slot),
                                  (unsigned long long)bytes,
                                  (unsigned long long)max_buffer_length);
            }
            const int cpu_visible = runtime_arena_is_cpu_visible(slot);
            const MTLResourceOptions storage_mode = cpu_visible ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
            id<MTLBuffer> buffer = [ctx->device newBufferWithLength:(NSUInteger)bytes options:storage_mode];
            if (buffer == nil) {
                uocr_metal_context_release_runtime_arenas(ctx);
                return metal_fail(error,
                                  error_size,
                                  "failed to allocate Metal runtime arena %s with %llu bytes",
                                  runtime_arena_slot_name(slot),
                                  (unsigned long long)bytes);
            }
            buffer.label = [NSString stringWithFormat:@"uocr_arena_%s", runtime_arena_slot_name(slot)];
            ctx->runtime_arenas[i].buffer = buffer;
            ctx->runtime_arenas[i].capacity = bytes;
            ctx->runtime_arenas[i].cpu_visible = cpu_visible;
        }
        ctx->kv_cache_layout = kv_layout;
        ctx->has_kv_cache_layout = 1;
    }
    return 1;
}

uint64_t uocr_metal_context_runtime_arena_capacity(const uocr_metal_context *ctx,
                                                   uocr_metal_runtime_arena_slot slot) {
    if (ctx == NULL || !runtime_arena_slot_valid(slot)) {
        return 0u;
    }
    return ctx->runtime_arenas[slot].capacity;
}

uint64_t uocr_metal_context_total_runtime_arena_capacity(const uocr_metal_context *ctx) {
    uint64_t total = 0u;
    if (ctx == NULL) {
        return 0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)UOCR_METAL_ARENA_COUNT; ++i) {
        if (!checked_add_u64(total, ctx->runtime_arenas[i].capacity, &total)) {
            return UINT64_MAX;
        }
    }
    return total;
}

int uocr_metal_context_get_kv_cache_layout(const uocr_metal_context *ctx,
                                           uocr_metal_kv_cache_layout *out_layout) {
    if (ctx == NULL || out_layout == NULL || !ctx->has_kv_cache_layout) {
        return 0;
    }
    *out_layout = ctx->kv_cache_layout;
    return 1;
}

int uocr_metal_kv_cache_offset(const uocr_metal_kv_cache_layout *layout,
                               int value_cache,
                               uint32_t layer,
                               uint32_t slot,
                               uint32_t cache_token,
                               uint32_t head,
                               uint32_t dim,
                               uint64_t *out_offset_bytes) {
    if (layout == NULL || out_offset_bytes == NULL || (value_cache != 0 && value_cache != 1) ||
        layout->total_bytes < 2u || layer >= layout->decoder_layers || slot >= layout->batch_slots ||
        cache_token >= layout->cache_token_capacity || head >= layout->kv_heads || dim >= layout->head_dim) {
        return 0;
    }

    uint64_t offset = value_cache ? layout->v_offset_bytes : layout->k_offset_bytes;
    uint64_t term = 0u;
    if (!checked_mul_u64((uint64_t)layer, layout->layer_stride_bytes, &term) ||
        !checked_add_u64(offset, term, &offset) ||
        !checked_mul_u64((uint64_t)slot, layout->slot_stride_bytes, &term) ||
        !checked_add_u64(offset, term, &offset) ||
        !checked_mul_u64((uint64_t)cache_token, layout->token_stride_bytes, &term) ||
        !checked_add_u64(offset, term, &offset) ||
        !checked_mul_u64((uint64_t)head, (uint64_t)layout->head_dim * 2u, &term) ||
        !checked_add_u64(offset, term, &offset) ||
        !checked_mul_u64((uint64_t)dim, 2u, &term) ||
        !checked_add_u64(offset, term, &offset) ||
        offset > layout->total_bytes - 2u) {
        return 0;
    }

    *out_offset_bytes = offset;
    return 1;
}

int uocr_metal_kv_cache_token_for_position(uint32_t prompt_length,
                                           uint32_t prompt_token_capacity,
                                           uint32_t position,
                                           uint32_t *out_cache_token) {
    if (out_cache_token == NULL || prompt_length == 0u || prompt_length > prompt_token_capacity ||
        position >= UOCR_MAX_POSITIONS) {
        return 0;
    }
    if (position < prompt_length) {
        *out_cache_token = position;
        return 1;
    }
    *out_cache_token = prompt_length + ((position - prompt_length) % UOCR_GENERATED_RING_WINDOW);
    return 1;
}

int uocr_metal_kv_cache_attention_length(uint32_t prompt_length,
                                         uint32_t generated_count,
                                         uint32_t *out_attention_length) {
    if (out_attention_length == NULL || prompt_length == 0u || prompt_length > UOCR_MAX_POSITIONS) {
        return 0;
    }
    const uint32_t live_generated = generated_count < UOCR_GENERATED_RING_WINDOW ? generated_count : UOCR_GENERATED_RING_WINDOW;
    if (prompt_length > UOCR_MAX_POSITIONS - live_generated) {
        return 0;
    }
    *out_attention_length = prompt_length + live_generated;
    return 1;
}

int uocr_metal_kv_cache_decode_attention_plan(uint32_t prompt_length,
                                             uint32_t prompt_token_capacity,
                                             uint32_t generated_count,
                                             uocr_metal_decode_attention_plan *out_plan) {
    if (out_plan == NULL || prompt_length == 0u || prompt_length > prompt_token_capacity ||
        prompt_length > UOCR_MAX_POSITIONS || generated_count > UOCR_MAX_POSITIONS - prompt_length) {
        return 0;
    }

    uint64_t cache_token_capacity = 0u;
    if (!checked_add_u64((uint64_t)prompt_token_capacity,
                         (uint64_t)UOCR_GENERATED_RING_WINDOW,
                         &cache_token_capacity) ||
        cache_token_capacity > (uint64_t)UINT32_MAX) {
        return 0;
    }

    uint32_t attention_length = 0u;
    if (!uocr_metal_kv_cache_attention_length(prompt_length, generated_count, &attention_length)) {
        return 0;
    }

    const uint32_t live_generated = generated_count < UOCR_GENERATED_RING_WINDOW ? generated_count : UOCR_GENERATED_RING_WINDOW;
    const uint32_t first_generated_index = generated_count - live_generated;
    const uint32_t first_generated_position = prompt_length + first_generated_index;
    const uint32_t query_position = generated_count == 0u ? prompt_length : prompt_length + generated_count - 1u;

    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->prompt_length = prompt_length;
    out_plan->prompt_token_capacity = prompt_token_capacity;
    out_plan->cache_token_capacity = (uint32_t)cache_token_capacity;
    out_plan->generated_count = generated_count;
    out_plan->live_generated = live_generated;
    out_plan->first_generated_index = first_generated_index;
    out_plan->first_generated_position = first_generated_position;
    out_plan->query_position = query_position;
    out_plan->attention_length = attention_length;
    out_plan->generated_ring_window = UOCR_GENERATED_RING_WINDOW;
    return 1;
}

int uocr_metal_kv_cache_decode_position_allowed(const uocr_metal_decode_attention_plan *plan,
                                                uint32_t key_position) {
    if (plan == NULL || plan->prompt_length == 0u || plan->generated_ring_window != UOCR_GENERATED_RING_WINDOW ||
        key_position >= UOCR_MAX_POSITIONS) {
        return 0;
    }
    if (key_position < plan->prompt_length) {
        return 1;
    }
    if (plan->live_generated == 0u) {
        return 0;
    }
    return key_position >= plan->first_generated_position && key_position <= plan->query_position;
}

int uocr_metal_kv_cache_decode_attention_index_to_token(const uocr_metal_decode_attention_plan *plan,
                                                        uint32_t attention_index,
                                                        uint32_t *out_cache_token) {
    if (plan == NULL || out_cache_token == NULL || plan->prompt_length == 0u ||
        plan->generated_ring_window != UOCR_GENERATED_RING_WINDOW || attention_index >= plan->attention_length) {
        return 0;
    }
    if (attention_index < plan->prompt_length) {
        *out_cache_token = attention_index;
        return 1;
    }

    const uint32_t generated_offset = attention_index - plan->prompt_length;
    if (generated_offset >= plan->live_generated) {
        return 0;
    }
    const uint32_t logical_generated = plan->first_generated_index + generated_offset;
    const uint32_t cache_token = plan->prompt_length + (logical_generated % plan->generated_ring_window);
    if (cache_token >= plan->cache_token_capacity) {
        return 0;
    }
    *out_cache_token = cache_token;
    return 1;
}

int uocr_metal_kv_cache_token_for_attention_index(uint32_t prompt_length,
                                                  uint32_t prompt_token_capacity,
                                                  uint32_t generated_count,
                                                  uint32_t attention_index,
                                                  uint32_t *out_cache_token) {
    uocr_metal_decode_attention_plan plan;
    if (!uocr_metal_kv_cache_decode_attention_plan(prompt_length,
                                                   prompt_token_capacity,
                                                   generated_count,
                                                   &plan)) {
        return 0;
    }
    return uocr_metal_kv_cache_decode_attention_index_to_token(&plan, attention_index, out_cache_token);
}

static void metal_decoder_result_reset(uocr_metal_decoder_result_f16 *result) {
    if (result == NULL) {
        return;
    }
    result->generated_count = 0u;
    result->stopped_on_eos = 0u;
    result->last_token_id = UINT32_MAX;
    result->last_score_f32 = 0.0f;
    result->reserved0 = 0u;
}

static int metal_decoder_runtime_arenas_ready(const uocr_metal_context *ctx) {
    if (ctx == NULL || !ctx->has_kv_cache_layout) {
        return 0;
    }
    return ctx->runtime_arenas[UOCR_METAL_ARENA_KV_CACHE].buffer != nil &&
           ctx->runtime_arenas[UOCR_METAL_ARENA_PROMPT_EMBEDDINGS].buffer != nil &&
           ctx->runtime_arenas[UOCR_METAL_ARENA_HIDDEN_PINGPONG].buffer != nil &&
           ctx->runtime_arenas[UOCR_METAL_ARENA_ROUTER_TOPK].buffer != nil &&
           ctx->runtime_arenas[UOCR_METAL_ARENA_MOE_INTERMEDIATE].buffer != nil &&
           ctx->runtime_arenas[UOCR_METAL_ARENA_LOGITS_READBACK].buffer != nil;
}

typedef struct uocr_metal_hot_path_alloc_guard {
    uocr_alloc_guard core_alloc_guard;
    uint64_t scratch_capacity;
    uint64_t runtime_arena_capacity;
    NSUInteger pipeline_cache_count;
} uocr_metal_hot_path_alloc_guard;

static uocr_metal_hot_path_alloc_guard metal_hot_path_alloc_guard_begin(const uocr_metal_context *ctx) {
    uocr_metal_hot_path_alloc_guard guard;
    guard.core_alloc_guard = uocr_alloc_guard_begin();
    guard.scratch_capacity = uocr_metal_context_total_scratch_capacity(ctx);
    guard.runtime_arena_capacity = uocr_metal_context_total_runtime_arena_capacity(ctx);
    guard.pipeline_cache_count = (ctx != NULL && ctx->pipeline_cache != nil) ? [ctx->pipeline_cache count] : 0u;
    return guard;
}

static int metal_hot_path_alloc_guard_end(const uocr_metal_context *ctx,
                                          const uocr_metal_hot_path_alloc_guard *guard,
                                          const char *label,
                                          char *error,
                                          size_t error_size) {
    if (ctx == NULL || guard == NULL) {
        return metal_fail(error, error_size, "invalid %s allocation guard", label != NULL ? label : "Metal hot path");
    }
    if (!uocr_alloc_guard_end_no_alloc(&guard->core_alloc_guard)) {
        return metal_fail(error,
                          error_size,
                          "%s allocated through the core allocator",
                          label != NULL ? label : "Metal hot path");
    }
    const uint64_t scratch_capacity = uocr_metal_context_total_scratch_capacity(ctx);
    if (scratch_capacity != guard->scratch_capacity) {
        return metal_fail(error,
                          error_size,
                          "%s grew Metal scratch capacity from %llu to %llu bytes",
                          label != NULL ? label : "Metal hot path",
                          (unsigned long long)guard->scratch_capacity,
                          (unsigned long long)scratch_capacity);
    }
    const uint64_t runtime_arena_capacity = uocr_metal_context_total_runtime_arena_capacity(ctx);
    if (runtime_arena_capacity != guard->runtime_arena_capacity) {
        return metal_fail(error,
                          error_size,
                          "%s changed runtime arena capacity from %llu to %llu bytes",
                          label != NULL ? label : "Metal hot path",
                          (unsigned long long)guard->runtime_arena_capacity,
                          (unsigned long long)runtime_arena_capacity);
    }
    const NSUInteger pipeline_cache_count = ctx->pipeline_cache != nil ? [ctx->pipeline_cache count] : 0u;
    if (pipeline_cache_count != guard->pipeline_cache_count) {
        return metal_fail(error,
                          error_size,
                          "%s grew Metal pipeline cache from %llu to %llu entries",
                          label != NULL ? label : "Metal hot path",
                          (unsigned long long)guard->pipeline_cache_count,
                          (unsigned long long)pipeline_cache_count);
    }
    return 1;
}

static int metal_prewarm_integrated_decode_pipelines(uocr_metal_context *ctx, char *error, size_t error_size) {
    static const char *const pipeline_names[] = {
        "uocr_rmsnorm_f16_to_f16",
        "uocr_dense_f16_to_f32",
        "uocr_get_rows_f16_to_f16",
        "uocr_attention_qkvo_f16_to_f16",
        "uocr_rope_qk_f16_to_f16",
        "uocr_kv_cache_write_f16",
        "uocr_decode_attention_f16_to_f16",
        "uocr_attention_output_residual_f16_to_f16",
        "uocr_dense_swiglu_gate_up_f16",
        "uocr_dense_swiglu_down_f16_to_f16",
        "uocr_moe_router_logits_f16_to_f32",
        "uocr_moe_router_softmax_topk_f32",
        "uocr_moe_prefill_interleaved_gate_up_f16",
        "uocr_moe_prefill_interleaved_down_sum_f16_to_f16",
        "uocr_moe_combine_f16_to_f16",
    };
    @autoreleasepool {
        for (uint32_t i = 0u; i < (uint32_t)(sizeof(pipeline_names) / sizeof(pipeline_names[0])); ++i) {
            if (metal_get_pipeline(ctx, pipeline_names[i], error, error_size) == nil) {
                return 0;
            }
        }
    }
    return 1;
}

static int metal_prompt_arena_slot_offset(const uocr_metal_context *ctx,
                                          uint32_t slot,
                                          uint32_t n_tokens,
                                          uint64_t *out_offset,
                                          uint64_t *out_bytes);

typedef struct uocr_metal_buffer_slice {
    id<MTLBuffer> buffer;
    NSUInteger offset;
} uocr_metal_buffer_slice;

static int metal_buffer_range_valid(id<MTLBuffer> buffer, NSUInteger offset, uint64_t bytes) {
    if (buffer == nil || bytes > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    const uint64_t length = (uint64_t)[buffer length];
    const uint64_t offset64 = (uint64_t)offset;
    uint64_t end = 0u;
    return offset64 <= length && checked_add_u64(offset64, bytes, &end) && end <= length;
}

static int metal_slice_valid(uocr_metal_buffer_slice slice, uint64_t bytes) {
    return metal_buffer_range_valid(slice.buffer, slice.offset, bytes);
}

static int metal_wait_for_command_buffer(id<MTLCommandBuffer> cb,
                                         const char *op_name,
                                         char *error,
                                         size_t error_size) {
    if (cb == nil) {
        return metal_fail(error, error_size, "failed to create %s command buffer", op_name);
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
        return metal_fail(error, error_size, "%s command failed: %s", op_name, [description UTF8String]);
    }
    return 1;
}

static int metal_hidden_arena_segment_slice(const uocr_metal_context *ctx,
                                            uint32_t slot,
                                            uint32_t segment,
                                            uint32_t n_tokens,
                                            uocr_metal_buffer_slice *out_slice,
                                            uint64_t *out_bytes) {
    if (out_slice == NULL || out_bytes == NULL || ctx == NULL || !ctx->has_kv_cache_layout ||
        ctx->runtime_arenas[UOCR_METAL_ARENA_HIDDEN_PINGPONG].buffer == nil ||
        segment >= UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS || slot >= ctx->kv_cache_layout.batch_slots ||
        n_tokens == 0u || n_tokens > ctx->kv_cache_layout.prompt_token_capacity) {
        return 0;
    }
    uint64_t segment_values = 0u;
    uint64_t segment_bytes = 0u;
    uint64_t slot_bytes = 0u;
    uint64_t segment_offset = 0u;
    uint64_t offset = 0u;
    uint64_t bytes = 0u;
    if (!checked_mul_u64((uint64_t)ctx->kv_cache_layout.prompt_token_capacity, (uint64_t)UOCR_HIDDEN_SIZE, &segment_values) ||
        !checked_mul_u64(segment_values, 2u, &segment_bytes) ||
        !checked_mul_u64(segment_bytes, (uint64_t)UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS, &slot_bytes) ||
        !checked_mul_u64((uint64_t)slot, slot_bytes, &offset) ||
        !checked_mul_u64((uint64_t)segment, segment_bytes, &segment_offset) ||
        !checked_add_u64(offset, segment_offset, &offset) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE * 2u, &bytes) ||
        offset > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    uocr_metal_buffer_slice slice;
    slice.buffer = ctx->runtime_arenas[UOCR_METAL_ARENA_HIDDEN_PINGPONG].buffer;
    slice.offset = (NSUInteger)offset;
    if (!metal_slice_valid(slice, bytes)) {
        return 0;
    }
    *out_slice = slice;
    *out_bytes = bytes;
    return 1;
}

static int metal_prompt_arena_slice(const uocr_metal_context *ctx,
                                    uint32_t slot,
                                    uint32_t n_tokens,
                                    uocr_metal_buffer_slice *out_slice,
                                    uint64_t *out_bytes) {
    if (out_slice == NULL || out_bytes == NULL || ctx == NULL) {
        return 0;
    }
    uint64_t offset = 0u;
    uint64_t bytes = 0u;
    if (!metal_prompt_arena_slot_offset(ctx, slot, n_tokens, &offset, &bytes) || offset > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    uocr_metal_buffer_slice slice;
    slice.buffer = ctx->runtime_arenas[UOCR_METAL_ARENA_PROMPT_EMBEDDINGS].buffer;
    slice.offset = (NSUInteger)offset;
    if (!metal_slice_valid(slice, bytes)) {
        return 0;
    }
    *out_slice = slice;
    *out_bytes = bytes;
    return 1;
}

static int metal_moe_intermediate_slice(const uocr_metal_context *ctx,
                                        uint32_t slot,
                                        uint64_t requested_bytes,
                                        uocr_metal_buffer_slice *out_slice) {
    if (out_slice == NULL || ctx == NULL || !ctx->has_kv_cache_layout || requested_bytes == 0u ||
        ctx->runtime_arenas[UOCR_METAL_ARENA_MOE_INTERMEDIATE].buffer == nil ||
        slot >= ctx->kv_cache_layout.batch_slots) {
        return 0;
    }
    const uint64_t per_token_values = (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE +
                                      (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE;
    uint64_t per_slot_values = 0u;
    uint64_t per_slot_bytes = 0u;
    uint64_t offset = 0u;
    if (!checked_mul_u64((uint64_t)ctx->kv_cache_layout.prompt_token_capacity, per_token_values, &per_slot_values) ||
        !checked_mul_u64(per_slot_values, 2u, &per_slot_bytes) ||
        !checked_mul_u64((uint64_t)slot, per_slot_bytes, &offset) ||
        offset > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    uocr_metal_buffer_slice slice;
    slice.buffer = ctx->runtime_arenas[UOCR_METAL_ARENA_MOE_INTERMEDIATE].buffer;
    slice.offset = (NSUInteger)offset;
    if (requested_bytes > per_slot_bytes || !metal_slice_valid(slice, requested_bytes)) {
        return 0;
    }
    *out_slice = slice;
    return 1;
}

static int metal_router_arena_slices(const uocr_metal_context *ctx,
                                     uint32_t slot,
                                     uint32_t n_tokens,
                                     uocr_metal_buffer_slice *logits,
                                     uocr_metal_buffer_slice *top_ids,
                                     uocr_metal_buffer_slice *top_weights,
                                     uint64_t *logits_bytes,
                                     uint64_t *top_ids_bytes,
                                     uint64_t *top_weights_bytes) {
    if (ctx == NULL || logits == NULL || top_ids == NULL || top_weights == NULL || logits_bytes == NULL ||
        top_ids_bytes == NULL || top_weights_bytes == NULL || !ctx->has_kv_cache_layout || n_tokens == 0u ||
        n_tokens > ctx->kv_cache_layout.prompt_token_capacity || slot >= ctx->kv_cache_layout.batch_slots ||
        ctx->runtime_arenas[UOCR_METAL_ARENA_ROUTER_TOPK].buffer == nil) {
        return 0;
    }
    uint64_t router_capacity_values = 0u;
    uint64_t top_capacity_values = 0u;
    uint64_t router_capacity_bytes = 0u;
    uint64_t top_capacity_ids_bytes = 0u;
    uint64_t top_capacity_weights_bytes = 0u;
    uint64_t slot_bytes = 0u;
    uint64_t slot_base = 0u;
    uint64_t request_router_values = 0u;
    uint64_t request_top_values = 0u;
    if (!checked_mul_u64((uint64_t)ctx->kv_cache_layout.prompt_token_capacity, (uint64_t)UOCR_ROUTED_EXPERTS, &router_capacity_values) ||
        !checked_mul_u64((uint64_t)ctx->kv_cache_layout.prompt_token_capacity, (uint64_t)UOCR_MOE_TOP_K, &top_capacity_values) ||
        !checked_mul_u64(router_capacity_values, (uint64_t)sizeof(float), &router_capacity_bytes) ||
        !checked_mul_u64(top_capacity_values, (uint64_t)sizeof(uint32_t), &top_capacity_ids_bytes) ||
        !checked_mul_u64(top_capacity_values, (uint64_t)sizeof(float), &top_capacity_weights_bytes) ||
        !checked_add_u64(router_capacity_bytes, top_capacity_ids_bytes, &slot_bytes) ||
        !checked_add_u64(slot_bytes, top_capacity_weights_bytes, &slot_bytes) ||
        !checked_mul_u64((uint64_t)slot, slot_bytes, &slot_base) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_ROUTED_EXPERTS, &request_router_values) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_MOE_TOP_K, &request_top_values) ||
        slot_base > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    *logits_bytes = request_router_values * (uint64_t)sizeof(float);
    *top_ids_bytes = request_top_values * (uint64_t)sizeof(uint32_t);
    *top_weights_bytes = request_top_values * (uint64_t)sizeof(float);
    logits->buffer = ctx->runtime_arenas[UOCR_METAL_ARENA_ROUTER_TOPK].buffer;
    logits->offset = (NSUInteger)slot_base;
    top_ids->buffer = logits->buffer;
    top_ids->offset = (NSUInteger)(slot_base + router_capacity_bytes);
    top_weights->buffer = logits->buffer;
    top_weights->offset = (NSUInteger)(slot_base + router_capacity_bytes + top_capacity_ids_bytes);
    return metal_slice_valid(*logits, *logits_bytes) && metal_slice_valid(*top_ids, *top_ids_bytes) &&
           metal_slice_valid(*top_weights, *top_weights_bytes);
}

static int metal_hidden_arena_token_slice(const uocr_metal_context *ctx,
                                          uint32_t slot,
                                          uint32_t segment,
                                          uint32_t token_index,
                                          uocr_metal_buffer_slice *out_slice) {
    if (out_slice == NULL || token_index >= UOCR_MAX_POSITIONS) {
        return 0;
    }
    uocr_metal_buffer_slice segment_slice;
    uint64_t covered_bytes = 0u;
    if (!metal_hidden_arena_segment_slice(ctx, slot, segment, token_index + 1u, &segment_slice, &covered_bytes)) {
        return 0;
    }
    (void)covered_bytes;
    const uint64_t row_bytes = (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    uint64_t row_offset = 0u;
    uint64_t offset = 0u;
    if (!checked_mul_u64((uint64_t)token_index, row_bytes, &row_offset) ||
        !checked_add_u64((uint64_t)segment_slice.offset, row_offset, &offset) ||
        offset > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    out_slice->buffer = segment_slice.buffer;
    out_slice->offset = (NSUInteger)offset;
    return metal_slice_valid(*out_slice, row_bytes);
}

static int metal_logits_arena_slices(const uocr_metal_context *ctx,
                                     uint32_t slot,
                                     uocr_metal_buffer_slice *logits,
                                     uocr_metal_buffer_slice *next_token) {
    if (ctx == NULL || logits == NULL || next_token == NULL || !ctx->has_kv_cache_layout ||
        slot >= ctx->kv_cache_layout.batch_slots ||
        ctx->runtime_arenas[UOCR_METAL_ARENA_LOGITS_READBACK].buffer == nil) {
        return 0;
    }
    const uint64_t logits_bytes_per_slot = (uint64_t)UOCR_VOCAB_SIZE * (uint64_t)sizeof(float);
    uint64_t logits_offset = 0u;
    uint64_t next_base = 0u;
    uint64_t next_offset = 0u;
    if (!checked_mul_u64((uint64_t)slot, logits_bytes_per_slot, &logits_offset) ||
        !checked_mul_u64((uint64_t)ctx->kv_cache_layout.batch_slots, logits_bytes_per_slot, &next_base) ||
        !checked_add_u64(next_base, (uint64_t)slot * (uint64_t)sizeof(int32_t), &next_offset) ||
        logits_offset > (uint64_t)NSUIntegerMax || next_offset > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    logits->buffer = ctx->runtime_arenas[UOCR_METAL_ARENA_LOGITS_READBACK].buffer;
    logits->offset = (NSUInteger)logits_offset;
    next_token->buffer = logits->buffer;
    next_token->offset = (NSUInteger)next_offset;
    return metal_slice_valid(*logits, logits_bytes_per_slot) &&
           metal_slice_valid(*next_token, (uint64_t)sizeof(int32_t));
}

static float *metal_logits_slice_cpu_ptr(uocr_metal_buffer_slice logits) {
    if (logits.buffer == nil) {
        return NULL;
    }
    void *base = [logits.buffer contents];
    if (base == NULL) {
        return NULL;
    }
    return (float *)((uint8_t *)base + (size_t)logits.offset);
}

static int32_t *metal_token_slice_cpu_ptr(uocr_metal_buffer_slice token) {
    if (token.buffer == nil) {
        return NULL;
    }
    void *base = [token.buffer contents];
    if (base == NULL) {
        return NULL;
    }
    return (int32_t *)((uint8_t *)base + (size_t)token.offset);
}

static int metal_cpu_argmax_logits(const float *logits,
                                   uint32_t vocab_size,
                                   uint32_t *token_id_out,
                                   float *score_out) {
    if (logits == NULL || token_id_out == NULL || score_out == NULL || vocab_size == 0u) {
        return 0;
    }
    uint32_t best_id = UINT32_MAX;
    float best_score = -INFINITY;
    for (uint32_t token = 0u; token < vocab_size; ++token) {
        const float score = logits[token];
        if (isnan(score)) {
            continue;
        }
        if (best_id == UINT32_MAX || score > best_score || (score == best_score && token < best_id)) {
            best_id = token;
            best_score = score;
        }
    }
    if (best_id == UINT32_MAX) {
        best_id = 0u;
        best_score = -INFINITY;
    }
    *token_id_out = best_id;
    *score_out = best_score;
    return 1;
}

static int metal_read_slice_to_host(uocr_metal_context *ctx,
                                    uocr_metal_buffer_slice src,
                                    uint64_t bytes,
                                    void *out,
                                    const char *op_name,
                                    char *error,
                                    size_t error_size) {
    if (ctx == NULL || out == NULL || bytes == 0u || bytes > (uint64_t)SIZE_MAX || !metal_slice_valid(src, bytes)) {
        return metal_fail(error, error_size, "invalid %s readback request", op_name);
    }
    @autoreleasepool {
        id<MTLBuffer> readback = [ctx->device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
        if (readback == nil) {
            return metal_fail(error, error_size, "failed to allocate %s readback buffer", op_name);
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [readback release];
            return metal_fail(error, error_size, "failed to create %s readback command buffer", op_name);
        }
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (blit == nil) {
            [readback release];
            return metal_fail(error, error_size, "failed to create %s readback blit encoder", op_name);
        }
        [blit copyFromBuffer:src.buffer sourceOffset:src.offset toBuffer:readback destinationOffset:0u size:(NSUInteger)bytes];
        [blit endEncoding];
        if (!metal_wait_for_command_buffer(cb, op_name, error, error_size)) {
            [readback release];
            return 0;
        }
        memcpy(out, [readback contents], (size_t)bytes);
        [readback release];
    }
    return 1;
}

static int metal_copy_slice(uocr_metal_context *ctx,
                            uocr_metal_buffer_slice src,
                            uocr_metal_buffer_slice dst,
                            uint64_t bytes,
                            const char *op_name,
                            char *error,
                            size_t error_size) {
    if (ctx == NULL || bytes == 0u || bytes > (uint64_t)NSUIntegerMax ||
        !metal_slice_valid(src, bytes) || !metal_slice_valid(dst, bytes)) {
        return metal_fail(error, error_size, "invalid %s copy request", op_name);
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create %s copy command buffer", op_name);
        }
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (blit == nil) {
            return metal_fail(error, error_size, "failed to create %s copy blit encoder", op_name);
        }
        [blit copyFromBuffer:src.buffer sourceOffset:src.offset toBuffer:dst.buffer destinationOffset:dst.offset size:(NSUInteger)bytes];
        [blit endEncoding];
        return metal_wait_for_command_buffer(cb, op_name, error, error_size);
    }
}

static const uocr_metal_decoder_binding *metal_require_decoder_binding(const uocr_metal_context *ctx,
                                                                       uint32_t tensor_id,
                                                                       const char *label,
                                                                       char *error,
                                                                       size_t error_size) {
    const uocr_metal_decoder_binding *binding = metal_find_decoder_binding(ctx, tensor_id);
    if (binding == NULL || binding->buffer == nil) {
        (void)metal_fail(error, error_size, "missing mapped decoder tensor for %s (id=%u)", label, tensor_id);
        return NULL;
    }
    return binding;
}

static int metal_run_token_embedding_from_token_slot_f16(uocr_metal_context *ctx,
                                                         uocr_metal_buffer_slice token_id,
                                                         uocr_metal_buffer_slice dst,
                                                         char *error,
                                                         size_t error_size) {
    if (ctx == NULL || !metal_slice_valid(token_id, sizeof(int32_t)) ||
        !metal_slice_valid(dst, (uint64_t)UOCR_HIDDEN_SIZE * 2u)) {
        return metal_fail(error, error_size, "invalid integrated token-embedding request");
    }
    const uint64_t table_bytes = (uint64_t)UOCR_VOCAB_SIZE * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uocr_metal_decoder_binding *embedding = metal_require_decoder_binding(ctx,
                                                                                UOCR_TENSOR_ID_TOK_EMBED,
                                                                                "token embedding",
                                                                                error,
                                                                                error_size);
    if (embedding == NULL || !metal_buffer_range_valid(embedding->buffer, embedding->offset, table_bytes)) {
        return 0;
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_get_rows_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated token-embedding command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated token-embedding command encoder");
        }
        uocr_metal_get_rows_params params;
        params.table_rows = UOCR_VOCAB_SIZE;
        params.row_width = UOCR_HIDDEN_SIZE;
        params.n_row_ids = 1u;
        params.reserved = 0u;
        NSUInteger threads = pipeline.threadExecutionWidth;
        if (threads == 0u) threads = 1u;
        if (threads > (NSUInteger)UOCR_HIDDEN_SIZE) threads = (NSUInteger)UOCR_HIDDEN_SIZE;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:embedding->buffer offset:embedding->offset atIndex:0u];
        [enc setBuffer:token_id.buffer offset:token_id.offset atIndex:1u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)UOCR_HIDDEN_SIZE, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated token embedding", error, error_size);
    }
}

static int metal_run_lm_head_buffer_f32(uocr_metal_context *ctx,
                                        uocr_metal_buffer_slice input,
                                        uint32_t n_rows,
                                        uocr_metal_buffer_slice logits,
                                        char *error,
                                        size_t error_size) {
    const uint64_t input_bytes = (uint64_t)n_rows * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t weight_bytes = (uint64_t)UOCR_VOCAB_SIZE * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t output_values = (uint64_t)n_rows * (uint64_t)UOCR_VOCAB_SIZE;
    const uint64_t output_bytes = output_values * (uint64_t)sizeof(float);
    if (ctx == NULL || n_rows == 0u || output_values > (uint64_t)UINT32_MAX ||
        !metal_slice_valid(input, input_bytes) || !metal_slice_valid(logits, output_bytes)) {
        return metal_fail(error, error_size, "invalid integrated LM-head request");
    }
    const uocr_metal_decoder_binding *lm_head = metal_require_decoder_binding(ctx,
                                                                              UOCR_TENSOR_ID_LM_HEAD,
                                                                              "LM head",
                                                                              error,
                                                                              error_size);
    if (lm_head == NULL || !metal_buffer_range_valid(lm_head->buffer, lm_head->offset, weight_bytes)) {
        return 0;
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_dense_f16_to_f32", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        const NSUInteger threads = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        if ((uint64_t)threads * sizeof(float) > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            return metal_fail(error, error_size, "integrated LM-head threadgroup memory exceeds device limit");
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated LM-head command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated LM-head command encoder");
        }
        uocr_metal_dense_params params;
        params.input_rows = n_rows;
        params.in_features = UOCR_HIDDEN_SIZE;
        params.out_features = UOCR_VOCAB_SIZE;
        params.has_bias = 0u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:input.buffer offset:input.offset atIndex:0u];
        [enc setBuffer:lm_head->buffer offset:lm_head->offset atIndex:1u];
        [enc setBuffer:lm_head->buffer offset:lm_head->offset atIndex:2u];
        [enc setBuffer:logits.buffer offset:logits.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated LM-head", error, error_size);
    }
}

static int metal_run_rmsnorm_buffer_f16(uocr_metal_context *ctx,
                                        uocr_metal_buffer_slice src,
                                        const uocr_metal_decoder_binding *weight,
                                        uint32_t n_rows,
                                        uocr_metal_buffer_slice dst,
                                        const char *op_name,
                                        char *error,
                                        size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)n_rows * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (ctx == NULL || weight == NULL || n_rows == 0u || !metal_slice_valid(src, hidden_bytes) ||
        !metal_slice_valid(dst, hidden_bytes) || !metal_buffer_range_valid(weight->buffer, weight->offset, UOCR_HIDDEN_SIZE * 2ull)) {
        return metal_fail(error, error_size, "invalid %s RMSNorm request", op_name);
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_rmsnorm_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        const NSUInteger threads = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        if ((uint64_t)threads * sizeof(float) > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            return metal_fail(error, error_size, "%s RMSNorm threadgroup memory exceeds device limit", op_name);
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create %s RMSNorm command buffer", op_name);
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create %s RMSNorm command encoder", op_name);
        }
        uocr_metal_rmsnorm_params params;
        params.n_rows = n_rows;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.eps = UOCR_RMS_NORM_EPS;
        params.reserved = 0u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src.buffer offset:src.offset atIndex:0u];
        [enc setBuffer:weight->buffer offset:weight->offset atIndex:1u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, op_name, error, error_size);
    }
}

static int metal_select_next_token_from_hidden_slice_f16(uocr_metal_context *ctx,
                                                        uocr_metal_buffer_slice hidden,
                                                        uocr_metal_buffer_slice norm_scratch,
                                                        uocr_metal_buffer_slice logits,
                                                        uocr_metal_buffer_slice token_slot,
                                                        const int32_t *sequence,
                                                        uint32_t sequence_len,
                                                        uint32_t no_repeat_ngram_size,
                                                        uint32_t no_repeat_window,
                                                        uint32_t *token_id_out,
                                                        float *score_out,
                                                        char *error,
                                                        size_t error_size) {
    if (ctx == NULL || sequence == NULL || token_id_out == NULL || score_out == NULL || sequence_len == 0u ||
        !metal_slice_valid(hidden, (uint64_t)UOCR_HIDDEN_SIZE * 2u) ||
        !metal_slice_valid(norm_scratch, (uint64_t)UOCR_HIDDEN_SIZE * 2u) ||
        !metal_slice_valid(logits, (uint64_t)UOCR_VOCAB_SIZE * sizeof(float)) ||
        !metal_slice_valid(token_slot, sizeof(int32_t))) {
        return metal_fail(error, error_size, "invalid integrated next-token selection request");
    }
    const uocr_metal_decoder_binding *final_norm = metal_require_decoder_binding(ctx,
                                                                                 UOCR_TENSOR_ID_FINAL_NORM,
                                                                                 "final norm",
                                                                                 error,
                                                                                 error_size);
    if (final_norm == NULL) {
        return 0;
    }
    if (!metal_run_rmsnorm_buffer_f16(ctx,
                                      hidden,
                                      final_norm,
                                      1u,
                                      norm_scratch,
                                      "integrated final RMSNorm",
                                      error,
                                      error_size) ||
        !metal_run_lm_head_buffer_f32(ctx, norm_scratch, 1u, logits, error, error_size)) {
        return 0;
    }

    float *logits_ptr = metal_logits_slice_cpu_ptr(logits);
    int32_t *token_ptr = metal_token_slice_cpu_ptr(token_slot);
    if (logits_ptr == NULL || token_ptr == NULL) {
        return metal_fail(error, error_size, "integrated next-token selection requires CPU-visible logits arena");
    }

    if (no_repeat_ngram_size != 0u) {
        const int status = uocr_no_repeat_ngram_apply(logits_ptr,
                                                      UOCR_VOCAB_SIZE,
                                                      sequence,
                                                      sequence_len,
                                                      no_repeat_ngram_size,
                                                      no_repeat_window,
                                                      NULL,
                                                      0u);
        if (status != UOCR_OK) {
            return metal_fail(error,
                              error_size,
                              "integrated no-repeat-ngram processor failed: %s",
                              uocr_status_string(status));
        }
    }

    uint32_t token_id = UINT32_MAX;
    float score = 0.0f;
    if (!metal_cpu_argmax_logits(logits_ptr, UOCR_VOCAB_SIZE, &token_id, &score) || token_id >= UOCR_VOCAB_SIZE) {
        return metal_fail(error, error_size, "integrated greedy argmax failed");
    }
    *token_ptr = (int32_t)token_id;
    *token_id_out = token_id;
    *score_out = score;
    return 1;
}

static int metal_run_attention_qkv_buffer_f16(uocr_metal_context *ctx,
                                              uocr_metal_buffer_slice src,
                                              const uocr_metal_decoder_binding *q_weight,
                                              const uocr_metal_decoder_binding *k_weight,
                                              const uocr_metal_decoder_binding *v_weight,
                                              uint32_t n_tokens,
                                              uocr_metal_buffer_slice q_dst,
                                              uocr_metal_buffer_slice k_dst,
                                              uocr_metal_buffer_slice v_dst,
                                              char *error,
                                              size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t weight_bytes = (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (ctx == NULL || q_weight == NULL || k_weight == NULL || v_weight == NULL || n_tokens == 0u ||
        !metal_slice_valid(src, hidden_bytes) || !metal_slice_valid(q_dst, hidden_bytes) ||
        !metal_slice_valid(k_dst, hidden_bytes) || !metal_slice_valid(v_dst, hidden_bytes) ||
        !metal_buffer_range_valid(q_weight->buffer, q_weight->offset, weight_bytes) ||
        !metal_buffer_range_valid(k_weight->buffer, k_weight->offset, weight_bytes) ||
        !metal_buffer_range_valid(v_weight->buffer, v_weight->offset, weight_bytes)) {
        return metal_fail(error, error_size, "invalid integrated attention QKV request");
    }
    uint64_t output_values = 0u;
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE * 3u, &output_values) ||
        output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "integrated attention QKV dispatch size overflow");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_attention_qkvo_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        const NSUInteger threads = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated attention QKV command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated attention QKV command encoder");
        }
        uocr_metal_attention_projection_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.projection_count = 3u;
        params.reserved = 0u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src.buffer offset:src.offset atIndex:0u];
        [enc setBuffer:q_weight->buffer offset:q_weight->offset atIndex:1u];
        [enc setBuffer:k_weight->buffer offset:k_weight->offset atIndex:2u];
        [enc setBuffer:v_weight->buffer offset:v_weight->offset atIndex:3u];
        [enc setBuffer:q_weight->buffer offset:q_weight->offset atIndex:4u];
        [enc setBuffer:q_dst.buffer offset:q_dst.offset atIndex:5u];
        [enc setBuffer:k_dst.buffer offset:k_dst.offset atIndex:6u];
        [enc setBuffer:v_dst.buffer offset:v_dst.offset atIndex:7u];
        [enc setBuffer:q_dst.buffer offset:q_dst.offset atIndex:8u];
        [enc setBytes:&params length:sizeof(params) atIndex:9u];
        [enc setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated attention QKV", error, error_size);
    }
}

static int metal_run_rope_qk_buffer_f16(uocr_metal_context *ctx,
                                        uocr_metal_buffer_slice q,
                                        uocr_metal_buffer_slice k,
                                        uint32_t n_tokens,
                                        uint32_t position_start,
                                        char *error,
                                        size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    uint64_t pair_threads = 0u;
    if (ctx == NULL || n_tokens == 0u || !metal_slice_valid(q, hidden_bytes) || !metal_slice_valid(k, hidden_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_ATTENTION_HEADS * (UOCR_HEAD_DIM / 2u), &pair_threads) ||
        pair_threads > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "invalid integrated RoPE request");
    }
    uint64_t position_end = 0u;
    if (!checked_add_u64((uint64_t)position_start, (uint64_t)n_tokens, &position_end) ||
        position_end > UOCR_MAX_POSITIONS) {
        return metal_fail(error, error_size, "integrated RoPE position range exceeds max positions");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_rope_qk_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        NSUInteger threads = pipeline.threadExecutionWidth;
        if (threads == 0u) threads = 1u;
        if (threads > 256u) threads = 256u;
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated RoPE command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated RoPE command encoder");
        }
        uocr_metal_rope_qk_params params;
        params.n_tokens = n_tokens;
        params.heads = UOCR_ATTENTION_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.position_start = position_start;
        params.freq_scale = -2.0f * log2f((float)UOCR_ROPE_THETA) / (float)UOCR_HEAD_DIM;
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q.buffer offset:q.offset atIndex:0u];
        [enc setBuffer:k.buffer offset:k.offset atIndex:1u];
        [enc setBuffer:q.buffer offset:q.offset atIndex:2u];
        [enc setBuffer:k.buffer offset:k.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)pair_threads, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated RoPE", error, error_size);
    }
}

static int metal_run_kv_write_buffer_f16(uocr_metal_context *ctx,
                                         uocr_metal_buffer_slice k_src,
                                         uocr_metal_buffer_slice v_src,
                                         uint32_t n_tokens,
                                         uint32_t layer,
                                         uint32_t slot,
                                         uint32_t prompt_length,
                                         uint32_t position_start,
                                         char *error,
                                         size_t error_size) {
    if (ctx == NULL || !ctx->has_kv_cache_layout || n_tokens == 0u || layer >= UOCR_DECODER_LAYERS ||
        slot >= ctx->kv_cache_layout.batch_slots || prompt_length == 0u ||
        prompt_length > ctx->kv_cache_layout.prompt_token_capacity) {
        return metal_fail(error, error_size, "invalid integrated KV write request");
    }
    const uint64_t src_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM * 2u;
    if (!metal_slice_valid(k_src, src_bytes) || !metal_slice_valid(v_src, src_bytes)) {
        return metal_fail(error, error_size, "integrated KV write source buffers are invalid");
    }
    id<MTLBuffer> kv = ctx->runtime_arenas[UOCR_METAL_ARENA_KV_CACHE].buffer;
    if (!metal_buffer_range_valid(kv, (NSUInteger)ctx->kv_cache_layout.k_offset_bytes, ctx->kv_cache_layout.tensor_bytes) ||
        !metal_buffer_range_valid(kv, (NSUInteger)ctx->kv_cache_layout.v_offset_bytes, ctx->kv_cache_layout.tensor_bytes)) {
        return metal_fail(error, error_size, "integrated KV cache arena is invalid");
    }
    uint64_t src_values = src_bytes / 2u;
    if (src_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "integrated KV write dispatch size overflow");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_kv_cache_write_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        NSUInteger threads = pipeline.threadExecutionWidth;
        if (threads == 0u) threads = 1u;
        if (threads > 256u) threads = 256u;
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated KV write command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated KV write command encoder");
        }
        uocr_metal_kv_cache_write_params params;
        params.n_tokens = n_tokens;
        params.batch_slots = ctx->kv_cache_layout.batch_slots;
        params.cache_token_capacity = ctx->kv_cache_layout.cache_token_capacity;
        params.layer = layer;
        params.slot = slot;
        params.prompt_length = prompt_length;
        params.position_start = position_start;
        params.heads = UOCR_KV_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.ring_window = UOCR_GENERATED_RING_WINDOW;
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:k_src.buffer offset:k_src.offset atIndex:0u];
        [enc setBuffer:v_src.buffer offset:v_src.offset atIndex:1u];
        [enc setBuffer:kv offset:(NSUInteger)ctx->kv_cache_layout.k_offset_bytes atIndex:2u];
        [enc setBuffer:kv offset:(NSUInteger)ctx->kv_cache_layout.v_offset_bytes atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)src_values, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated KV write", error, error_size);
    }
}

static int metal_run_decode_attention_buffer_f16(uocr_metal_context *ctx,
                                                 uocr_metal_buffer_slice q,
                                                 uint32_t layer,
                                                 uint32_t slot,
                                                 uint32_t prompt_length,
                                                 uint32_t generated_count,
                                                 uocr_metal_buffer_slice dst,
                                                 char *error,
                                                 size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (ctx == NULL || !ctx->has_kv_cache_layout || layer >= UOCR_DECODER_LAYERS ||
        slot >= ctx->kv_cache_layout.batch_slots || prompt_length == 0u ||
        prompt_length > ctx->kv_cache_layout.prompt_token_capacity ||
        generated_count > UOCR_MAX_POSITIONS - prompt_length ||
        !metal_slice_valid(q, hidden_bytes) || !metal_slice_valid(dst, hidden_bytes)) {
        return metal_fail(error, error_size, "invalid integrated decode attention request");
    }
    uocr_metal_decode_attention_plan plan;
    if (!uocr_metal_kv_cache_decode_attention_plan(prompt_length,
                                                   ctx->kv_cache_layout.prompt_token_capacity,
                                                   generated_count,
                                                   &plan)) {
        return metal_fail(error, error_size, "invalid integrated decode attention plan");
    }
    id<MTLBuffer> kv = ctx->runtime_arenas[UOCR_METAL_ARENA_KV_CACHE].buffer;
    if (!metal_buffer_range_valid(kv, (NSUInteger)ctx->kv_cache_layout.k_offset_bytes, ctx->kv_cache_layout.tensor_bytes) ||
        !metal_buffer_range_valid(kv, (NSUInteger)ctx->kv_cache_layout.v_offset_bytes, ctx->kv_cache_layout.tensor_bytes)) {
        return metal_fail(error, error_size, "integrated decode attention KV arena is invalid");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_decode_attention_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        if (pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)UOCR_HEAD_DIM) {
            return metal_fail(error, error_size, "integrated decode attention pipeline cannot run one head per group");
        }
        const NSUInteger threads = (NSUInteger)UOCR_HEAD_DIM;
        if ((uint64_t)threads * sizeof(float) > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            return metal_fail(error, error_size, "integrated decode attention threadgroup memory exceeds device limit");
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated decode attention command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated decode attention command encoder");
        }
        uocr_metal_decode_attention_params params;
        memset(&params, 0, sizeof(params));
        params.batch_slots = ctx->kv_cache_layout.batch_slots;
        params.cache_token_capacity = ctx->kv_cache_layout.cache_token_capacity;
        params.layer = layer;
        params.slot = slot;
        params.prompt_length = prompt_length;
        params.generated_count = generated_count;
        params.attention_length = plan.attention_length;
        params.first_generated = plan.first_generated_index;
        params.heads = UOCR_ATTENTION_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.ring_window = UOCR_GENERATED_RING_WINDOW;
        params.scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q.buffer offset:q.offset atIndex:0u];
        [enc setBuffer:kv offset:(NSUInteger)ctx->kv_cache_layout.k_offset_bytes atIndex:1u];
        [enc setBuffer:kv offset:(NSUInteger)ctx->kv_cache_layout.v_offset_bytes atIndex:2u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)UOCR_ATTENTION_HEADS, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated decode attention", error, error_size);
    }
}

static int metal_run_prefill_attention_buffer_f16(uocr_metal_context *ctx,
                                                  uocr_metal_buffer_slice q,
                                                  uocr_metal_buffer_slice k,
                                                  uocr_metal_buffer_slice v,
                                                  uint32_t n_tokens,
                                                  uocr_metal_buffer_slice dst,
                                                  char *error,
                                                  size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    uint64_t groups = 0u;
    if (ctx == NULL || n_tokens == 0u || n_tokens > UOCR_MAX_POSITIONS ||
        !metal_slice_valid(q, hidden_bytes) || !metal_slice_valid(k, hidden_bytes) ||
        !metal_slice_valid(v, hidden_bytes) || !metal_slice_valid(dst, hidden_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_ATTENTION_HEADS, &groups) ||
        groups > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "invalid integrated prefill attention request");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_prefill_attention_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        NSUInteger threads = metal_power2_threadgroup_width(128u, pipeline.maxTotalThreadsPerThreadgroup);
        uint64_t floats = 0u;
        uint64_t requested = 0u;
        uint64_t tg_bytes = 0u;
        if (!checked_add_u64((uint64_t)n_tokens, (uint64_t)threads, &floats) ||
            !checked_mul_u64(floats, (uint64_t)sizeof(float), &requested) ||
            !align_up_u64_checked(requested, 16u, &tg_bytes) ||
            tg_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            return metal_fail(error, error_size, "integrated prefill attention threadgroup memory overflow");
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated prefill attention command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated prefill attention command encoder");
        }
        uocr_metal_prefill_attention_params params;
        params.n_tokens = n_tokens;
        params.heads = UOCR_ATTENTION_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q.buffer offset:q.offset atIndex:0u];
        [enc setBuffer:k.buffer offset:k.offset atIndex:1u];
        [enc setBuffer:v.buffer offset:v.offset atIndex:2u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:(NSUInteger)tg_bytes atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated prefill attention", error, error_size);
    }
}

static int metal_run_attention_output_residual_buffer_f16(uocr_metal_context *ctx,
                                                          uocr_metal_buffer_slice context,
                                                          const uocr_metal_decoder_binding *o_weight,
                                                          uocr_metal_buffer_slice residual,
                                                          uint32_t n_tokens,
                                                          uocr_metal_buffer_slice dst,
                                                          char *error,
                                                          size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t weight_bytes = (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    uint64_t output_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (ctx == NULL || o_weight == NULL || n_tokens == 0u || !metal_slice_valid(context, hidden_bytes) ||
        !metal_slice_valid(residual, hidden_bytes) || !metal_slice_valid(dst, hidden_bytes) ||
        !metal_buffer_range_valid(o_weight->buffer, o_weight->offset, weight_bytes) || output_values > UINT32_MAX) {
        return metal_fail(error, error_size, "invalid integrated attention output request");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_attention_output_residual_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        const NSUInteger threads = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated attention output command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated attention output command encoder");
        }
        uocr_metal_attention_output_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:context.buffer offset:context.offset atIndex:0u];
        [enc setBuffer:o_weight->buffer offset:o_weight->offset atIndex:1u];
        [enc setBuffer:residual.buffer offset:residual.offset atIndex:2u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated attention output", error, error_size);
    }
}

static int metal_run_dense_swiglu_buffer_f16(uocr_metal_context *ctx,
                                             uocr_metal_buffer_slice input,
                                             const uocr_metal_decoder_binding *gate_weight,
                                             const uocr_metal_decoder_binding *up_weight,
                                             const uocr_metal_decoder_binding *down_weight,
                                             uocr_metal_buffer_slice residual,
                                             int has_residual,
                                             uint32_t n_tokens,
                                             uint32_t intermediate_size,
                                             uocr_metal_buffer_slice mid,
                                             uocr_metal_buffer_slice dst,
                                             const char *op_name,
                                             char *error,
                                             size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t gate_weight_bytes = (uint64_t)intermediate_size * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t down_weight_bytes = (uint64_t)UOCR_HIDDEN_SIZE * (uint64_t)intermediate_size * 2u;
    const uint64_t mid_bytes = (uint64_t)n_tokens * (uint64_t)intermediate_size * 2u;
    uint64_t gate_values = (uint64_t)n_tokens * (uint64_t)intermediate_size;
    uint64_t down_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (ctx == NULL || gate_weight == NULL || up_weight == NULL || down_weight == NULL || n_tokens == 0u ||
        intermediate_size == 0u || gate_values > UINT32_MAX || down_values > UINT32_MAX ||
        !metal_slice_valid(input, hidden_bytes) || !metal_slice_valid(mid, mid_bytes) ||
        !metal_slice_valid(dst, hidden_bytes) || (has_residual && !metal_slice_valid(residual, hidden_bytes)) ||
        !metal_buffer_range_valid(gate_weight->buffer, gate_weight->offset, gate_weight_bytes) ||
        !metal_buffer_range_valid(up_weight->buffer, up_weight->offset, gate_weight_bytes) ||
        !metal_buffer_range_valid(down_weight->buffer, down_weight->offset, down_weight_bytes)) {
        return metal_fail(error, error_size, "invalid %s SwiGLU request", op_name);
    }
    @autoreleasepool {
        id<MTLComputePipelineState> gate_pipeline = metal_get_pipeline(ctx, "uocr_dense_swiglu_gate_up_f16", error, error_size);
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, "uocr_dense_swiglu_down_f16_to_f16", error, error_size);
        if (gate_pipeline == nil || down_pipeline == nil) {
            return 0;
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create %s command buffer", op_name);
        }
        uocr_metal_dense_swiglu_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.intermediate_size = intermediate_size;
        params.has_residual = has_residual ? 1u : 0u;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create %s gate/up command encoder", op_name);
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_pipeline];
        [enc setBuffer:input.buffer offset:input.offset atIndex:0u];
        [enc setBuffer:gate_weight->buffer offset:gate_weight->offset atIndex:1u];
        [enc setBuffer:up_weight->buffer offset:up_weight->offset atIndex:2u];
        [enc setBuffer:mid.buffer offset:mid.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)gate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create %s down command encoder", op_name);
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid.buffer offset:mid.offset atIndex:0u];
        [enc setBuffer:down_weight->buffer offset:down_weight->offset atIndex:1u];
        [enc setBuffer:(has_residual ? residual.buffer : input.buffer) offset:(has_residual ? residual.offset : input.offset) atIndex:2u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)down_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, op_name, error, error_size);
    }
}

static int metal_run_moe_router_buffer_f16(uocr_metal_context *ctx,
                                           uocr_metal_buffer_slice input,
                                           const uocr_metal_decoder_binding *router_weight,
                                           uint32_t slot,
                                           uint32_t n_tokens,
                                           uocr_metal_buffer_slice *top_ids_out,
                                           uocr_metal_buffer_slice *top_weights_out,
                                           char *error,
                                           size_t error_size) {
    const uint64_t hidden_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t weight_bytes = (uint64_t)UOCR_ROUTED_EXPERTS * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    uocr_metal_buffer_slice logits;
    uocr_metal_buffer_slice top_ids;
    uocr_metal_buffer_slice top_weights;
    uint64_t logits_bytes = 0u;
    uint64_t top_ids_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t router_values = (uint64_t)n_tokens * (uint64_t)UOCR_ROUTED_EXPERTS;
    if (ctx == NULL || router_weight == NULL || top_ids_out == NULL || top_weights_out == NULL || n_tokens == 0u ||
        router_values > UINT32_MAX || !metal_slice_valid(input, hidden_bytes) ||
        !metal_buffer_range_valid(router_weight->buffer, router_weight->offset, weight_bytes) ||
        !metal_router_arena_slices(ctx, slot, n_tokens, &logits, &top_ids, &top_weights,
                                   &logits_bytes, &top_ids_bytes, &top_weights_bytes)) {
        return metal_fail(error, error_size, "invalid integrated MoE router request");
    }
    (void)logits_bytes;
    (void)top_ids_bytes;
    (void)top_weights_bytes;
    @autoreleasepool {
        id<MTLComputePipelineState> logits_pipeline = metal_get_pipeline(ctx, "uocr_moe_router_logits_f16_to_f32", error, error_size);
        id<MTLComputePipelineState> topk_pipeline = metal_get_pipeline(ctx, "uocr_moe_router_softmax_topk_f32", error, error_size);
        if (logits_pipeline == nil || topk_pipeline == nil) {
            return 0;
        }
        if (topk_pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)UOCR_ROUTED_EXPERTS) {
            return metal_fail(error, error_size, "integrated MoE router top-k pipeline does not support 64 threads");
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated MoE router command buffer");
        }
        uocr_metal_moe_router_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.experts = UOCR_ROUTED_EXPERTS;
        params.top_k = UOCR_MOE_TOP_K;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated MoE router logits encoder");
        }
        const NSUInteger logits_threads = metal_power2_threadgroup_width(256u, logits_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:logits_pipeline];
        [enc setBuffer:input.buffer offset:input.offset atIndex:0u];
        [enc setBuffer:router_weight->buffer offset:router_weight->offset atIndex:1u];
        [enc setBuffer:logits.buffer offset:logits.offset atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:logits_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)router_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(logits_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated MoE router top-k encoder");
        }
        const NSUInteger topk_threads = metal_power2_threadgroup_width(64u, topk_pipeline.maxTotalThreadsPerThreadgroup);
        const NSUInteger topk_scratch_bytes = ((NSUInteger)UOCR_ROUTED_EXPERTS + topk_threads) * sizeof(float) +
                                              (NSUInteger)UOCR_ROUTED_EXPERTS * sizeof(uint32_t);
        [enc setComputePipelineState:topk_pipeline];
        [enc setBuffer:logits.buffer offset:logits.offset atIndex:0u];
        [enc setBuffer:logits.buffer offset:logits.offset atIndex:1u];
        [enc setBuffer:top_ids.buffer offset:top_ids.offset atIndex:2u];
        [enc setBuffer:top_weights.buffer offset:top_weights.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:topk_scratch_bytes atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(topk_threads, 1u, 1u)];
        [enc endEncoding];
        if (!metal_wait_for_command_buffer(cb, "integrated MoE router", error, error_size)) {
            return 0;
        }
    }
    *top_ids_out = top_ids;
    *top_weights_out = top_weights;
    return 1;
}

static int metal_expert_interleaved_slab_for_layer(const uocr_metal_context *ctx,
                                                   uint32_t layer,
                                                   uocr_metal_buffer_slice *out_slab,
                                                   char *error,
                                                   size_t error_size) {
    if (ctx == NULL || out_slab == NULL || layer == 0u || layer >= UOCR_DECODER_LAYERS) {
        return metal_fail(error, error_size, "invalid integrated MoE expert slab request");
    }
    const uint64_t projection_values = (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * (uint64_t)UOCR_HIDDEN_SIZE;
    const uint64_t projection_bytes = projection_values * 2u;
    const uint64_t expert_stride_bytes = projection_bytes * 3u;
    const uint64_t total_bytes = expert_stride_bytes * (uint64_t)UOCR_ROUTED_EXPERTS;
    const uocr_metal_decoder_binding *gate0 = metal_require_decoder_binding(ctx,
                                                                            uocr_tensor_id_moe_expert(layer, 0u, UOCR_TENSOR_PROJ_GATE),
                                                                            "MoE expert0 gate",
                                                                            error,
                                                                            error_size);
    if (gate0 == NULL || gate0->offset > NSUIntegerMax || !metal_buffer_range_valid(gate0->buffer, gate0->offset, total_bytes)) {
        return metal_fail(error, error_size, "integrated MoE expert slab for layer %u is not contiguous in one Metal view", layer);
    }
    for (uint32_t expert = 0u; expert < UOCR_ROUTED_EXPERTS; ++expert) {
        const uint64_t base = (uint64_t)gate0->offset + (uint64_t)expert * expert_stride_bytes;
        const uocr_metal_decoder_binding *gate = metal_find_decoder_binding(ctx, uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_GATE));
        const uocr_metal_decoder_binding *up = metal_find_decoder_binding(ctx, uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_UP));
        const uocr_metal_decoder_binding *down = metal_find_decoder_binding(ctx, uocr_tensor_id_moe_expert(layer, expert, UOCR_TENSOR_PROJ_DOWN));
        if (gate == NULL || up == NULL || down == NULL || gate->buffer != gate0->buffer || up->buffer != gate0->buffer ||
            down->buffer != gate0->buffer || (uint64_t)gate->offset != base ||
            (uint64_t)up->offset != base + projection_bytes || (uint64_t)down->offset != base + 2u * projection_bytes) {
            return metal_fail(error,
                              error_size,
                              "integrated MoE expert tensors for layer %u are not interleaved-contiguous at expert %u",
                              layer,
                              expert);
        }
    }
    out_slab->buffer = gate0->buffer;
    out_slab->offset = gate0->offset;
    return 1;
}

static int metal_run_moe_interleaved_buffer_f16(uocr_metal_context *ctx,
                                                uocr_metal_buffer_slice input,
                                                uocr_metal_buffer_slice top_ids,
                                                uocr_metal_buffer_slice top_weights,
                                                uocr_metal_buffer_slice expert_slab,
                                                uint32_t slot,
                                                uint32_t n_tokens,
                                                uocr_metal_buffer_slice mid,
                                                uocr_metal_buffer_slice dst,
                                                char *error,
                                                size_t error_size) {
    (void)slot;
    const uint64_t hidden_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    const uint64_t top_ids_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K * sizeof(uint32_t);
    const uint64_t top_weights_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * 2u;
    const uint64_t projection_values = (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * (uint64_t)UOCR_HIDDEN_SIZE;
    const uint64_t slab_bytes = projection_values * 2u * 3u * (uint64_t)UOCR_ROUTED_EXPERTS;
    uint64_t gate_values = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE;
    uint64_t down_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    if (ctx == NULL || n_tokens == 0u || gate_values > UINT32_MAX || down_values > UINT32_MAX ||
        !metal_slice_valid(input, hidden_bytes) || !metal_slice_valid(top_ids, top_ids_bytes) ||
        !metal_slice_valid(top_weights, top_weights_bytes) || !metal_slice_valid(expert_slab, slab_bytes) ||
        !metal_slice_valid(mid, mid_bytes) || !metal_slice_valid(dst, hidden_bytes)) {
        return metal_fail(error, error_size, "invalid integrated routed MoE request");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> gate_pipeline = metal_get_pipeline(ctx, "uocr_moe_prefill_interleaved_gate_up_f16", error, error_size);
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, "uocr_moe_prefill_interleaved_down_sum_f16_to_f16", error, error_size);
        if (gate_pipeline == nil || down_pipeline == nil) {
            return 0;
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated routed MoE command buffer");
        }
        uocr_metal_moe_prefill_interleaved_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        params.expert_count = UOCR_ROUTED_EXPERTS;
        params.top_k = UOCR_MOE_TOP_K;
        params.expert_stride_values = (uint32_t)(projection_values * 3u);
        params.up_offset_values = (uint32_t)projection_values;
        params.down_offset_values = (uint32_t)(projection_values * 2u);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated routed MoE gate/up encoder");
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_pipeline];
        [enc setBuffer:input.buffer offset:input.offset atIndex:0u];
        [enc setBuffer:top_ids.buffer offset:top_ids.offset atIndex:1u];
        [enc setBuffer:expert_slab.buffer offset:expert_slab.offset atIndex:2u];
        [enc setBuffer:mid.buffer offset:mid.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)gate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated routed MoE down encoder");
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid.buffer offset:mid.offset atIndex:0u];
        [enc setBuffer:top_ids.buffer offset:top_ids.offset atIndex:1u];
        [enc setBuffer:top_weights.buffer offset:top_weights.offset atIndex:2u];
        [enc setBuffer:expert_slab.buffer offset:expert_slab.offset atIndex:3u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:4u];
        [enc setBytes:&params length:sizeof(params) atIndex:5u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)down_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated routed MoE", error, error_size);
    }
}

static int metal_run_moe_combine_buffer_f16(uocr_metal_context *ctx,
                                            uocr_metal_buffer_slice routed,
                                            uocr_metal_buffer_slice shared,
                                            uocr_metal_buffer_slice residual,
                                            uint32_t n_tokens,
                                            uocr_metal_buffer_slice dst,
                                            char *error,
                                            size_t error_size) {
    const uint64_t hidden_values = (uint64_t)n_tokens * (uint64_t)UOCR_HIDDEN_SIZE;
    const uint64_t hidden_bytes = hidden_values * 2u;
    if (ctx == NULL || n_tokens == 0u || hidden_values > UINT32_MAX || !metal_slice_valid(routed, hidden_bytes) ||
        !metal_slice_valid(shared, hidden_bytes) || !metal_slice_valid(residual, hidden_bytes) ||
        !metal_slice_valid(dst, hidden_bytes)) {
        return metal_fail(error, error_size, "invalid integrated MoE combine request");
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_moe_combine_f16_to_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return metal_fail(error, error_size, "failed to create integrated MoE combine command buffer");
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            return metal_fail(error, error_size, "failed to create integrated MoE combine command encoder");
        }
        uocr_metal_moe_combine_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.has_residual = 1u;
        params.reserved = 0u;
        NSUInteger threads = pipeline.threadExecutionWidth;
        if (threads == 0u) threads = 1u;
        if (threads > 256u) threads = 256u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:routed.buffer offset:routed.offset atIndex:0u];
        [enc setBuffer:shared.buffer offset:shared.offset atIndex:1u];
        [enc setBuffer:residual.buffer offset:residual.offset atIndex:2u];
        [enc setBuffer:dst.buffer offset:dst.offset atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_values, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        return metal_wait_for_command_buffer(cb, "integrated MoE combine", error, error_size);
    }
}

static int metal_run_decoder_decode_one_f16(uocr_metal_context *ctx,
                                            uint32_t slot,
                                            uint32_t prompt_length,
                                            uint32_t generated_count_after_write,
                                            char *error,
                                            size_t error_size) {
    if (ctx == NULL || !ctx->has_kv_cache_layout || slot >= ctx->kv_cache_layout.batch_slots ||
        prompt_length == 0u || prompt_length > ctx->kv_cache_layout.prompt_token_capacity ||
        generated_count_after_write == 0u || generated_count_after_write > UOCR_MAX_POSITIONS - prompt_length) {
        return metal_fail(error, error_size, "invalid integrated decoder decode-step request");
    }
    const uint32_t position = prompt_length + generated_count_after_write - 1u;
    const uint64_t hidden_bytes = (uint64_t)UOCR_HIDDEN_SIZE * 2u;

    uocr_metal_buffer_slice scratch[UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS];
    for (uint32_t i = 0u; i < UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS; ++i) {
        uint64_t bytes = 0u;
        if (!metal_hidden_arena_segment_slice(ctx, slot, i, 1u, &scratch[i], &bytes) || bytes != hidden_bytes) {
            return metal_fail(error, error_size, "integrated decode requires hidden scratch segment %u", i);
        }
    }

    uocr_metal_buffer_slice current = scratch[0];
    uint32_t current_segment = 0u;
    for (uint32_t layer = 0u; layer < UOCR_DECODER_LAYERS; ++layer) {
        uint32_t available[UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS];
        uint32_t available_count = 0u;
        for (uint32_t i = 0u; i < UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS; ++i) {
            if (i != current_segment) {
                available[available_count++] = i;
            }
        }
        if (available_count < 4u) {
            return metal_fail(error, error_size, "integrated decode scratch assignment failed at layer %u", layer);
        }
        const uint32_t norm_segment = available[0];
        const uint32_t q_segment = available[1];
        const uint32_t k_segment = available[2];
        const uint32_t v_segment = available[3];
        uocr_metal_buffer_slice norm = scratch[norm_segment];
        uocr_metal_buffer_slice q = scratch[q_segment];
        uocr_metal_buffer_slice k = scratch[k_segment];
        uocr_metal_buffer_slice v = scratch[v_segment];
        uocr_metal_buffer_slice output = k;
        const uint32_t output_segment = k_segment;

        const uocr_metal_decoder_binding *input_norm = metal_require_decoder_binding(ctx,
                                                                                     uocr_tensor_id_layer_input_norm(layer),
                                                                                     "decode input RMSNorm",
                                                                                     error,
                                                                                     error_size);
        const uocr_metal_decoder_binding *post_norm = metal_require_decoder_binding(ctx,
                                                                                    uocr_tensor_id_layer_post_attn_norm(layer),
                                                                                    "decode post-attention RMSNorm",
                                                                                    error,
                                                                                    error_size);
        const uocr_metal_decoder_binding *q_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_Q),
                                                                                   "decode attention q_proj",
                                                                                   error,
                                                                                   error_size);
        const uocr_metal_decoder_binding *k_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_K),
                                                                                   "decode attention k_proj",
                                                                                   error,
                                                                                   error_size);
        const uocr_metal_decoder_binding *v_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_V),
                                                                                   "decode attention v_proj",
                                                                                   error,
                                                                                   error_size);
        const uocr_metal_decoder_binding *o_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_O),
                                                                                   "decode attention o_proj",
                                                                                   error,
                                                                                   error_size);
        if (input_norm == NULL || post_norm == NULL || q_weight == NULL || k_weight == NULL || v_weight == NULL || o_weight == NULL) {
            return 0;
        }

        if (!metal_run_rmsnorm_buffer_f16(ctx, current, input_norm, 1u, norm, "integrated decode input RMSNorm", error, error_size) ||
            !metal_run_attention_qkv_buffer_f16(ctx, norm, q_weight, k_weight, v_weight, 1u, q, k, v, error, error_size) ||
            !metal_run_rope_qk_buffer_f16(ctx, q, k, 1u, position, error, error_size) ||
            !metal_run_kv_write_buffer_f16(ctx, k, v, 1u, layer, slot, prompt_length, position, error, error_size) ||
            !metal_run_decode_attention_buffer_f16(ctx, q, layer, slot, prompt_length, generated_count_after_write, norm, error, error_size) ||
            !metal_run_attention_output_residual_buffer_f16(ctx, norm, o_weight, current, 1u, q, error, error_size) ||
            !metal_run_rmsnorm_buffer_f16(ctx, q, post_norm, 1u, norm, "integrated decode post-attention RMSNorm", error, error_size)) {
            return 0;
        }

        if (layer == 0u) {
            const uocr_metal_decoder_binding *gate = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_GATE),
                                                                                   "decode layer0 dense gate",
                                                                                   error,
                                                                                   error_size);
            const uocr_metal_decoder_binding *up = metal_require_decoder_binding(ctx,
                                                                                 uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_UP),
                                                                                 "decode layer0 dense up",
                                                                                 error,
                                                                                 error_size);
            const uocr_metal_decoder_binding *down = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_DOWN),
                                                                                   "decode layer0 dense down",
                                                                                   error,
                                                                                   error_size);
            const uint64_t mid_bytes = (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE * 2u;
            uocr_metal_buffer_slice mid;
            if (gate == NULL || up == NULL || down == NULL ||
                !metal_moe_intermediate_slice(ctx, slot, mid_bytes, &mid) ||
                !metal_run_dense_swiglu_buffer_f16(ctx,
                                                   norm,
                                                   gate,
                                                   up,
                                                   down,
                                                   q,
                                                   1,
                                                   1u,
                                                   UOCR_DENSE_LAYER0_INTERMEDIATE,
                                                   mid,
                                                   output,
                                                   "integrated decode layer0 dense SwiGLU",
                                                   error,
                                                   error_size)) {
                return 0;
            }
        } else {
            const uocr_metal_decoder_binding *router = metal_require_decoder_binding(ctx,
                                                                                     uocr_tensor_id_moe_router(layer),
                                                                                     "decode MoE router",
                                                                                     error,
                                                                                     error_size);
            const uocr_metal_decoder_binding *shared_gate = metal_require_decoder_binding(ctx,
                                                                                          uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_GATE),
                                                                                          "decode MoE shared gate",
                                                                                          error,
                                                                                          error_size);
            const uocr_metal_decoder_binding *shared_up = metal_require_decoder_binding(ctx,
                                                                                        uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_UP),
                                                                                        "decode MoE shared up",
                                                                                        error,
                                                                                        error_size);
            const uocr_metal_decoder_binding *shared_down = metal_require_decoder_binding(ctx,
                                                                                          uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_DOWN),
                                                                                          "decode MoE shared down",
                                                                                          error,
                                                                                          error_size);
            uocr_metal_buffer_slice top_ids;
            uocr_metal_buffer_slice top_weights;
            uocr_metal_buffer_slice expert_slab;
            const uint64_t shared_mid_bytes = (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE * 2u;
            const uint64_t routed_mid_bytes = (uint64_t)UOCR_MOE_TOP_K * (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * 2u;
            uocr_metal_buffer_slice mid;
            if (router == NULL || shared_gate == NULL || shared_up == NULL || shared_down == NULL ||
                !metal_run_moe_router_buffer_f16(ctx, norm, router, slot, 1u, &top_ids, &top_weights, error, error_size) ||
                !metal_moe_intermediate_slice(ctx, slot, shared_mid_bytes, &mid) ||
                !metal_run_dense_swiglu_buffer_f16(ctx,
                                                   norm,
                                                   shared_gate,
                                                   shared_up,
                                                   shared_down,
                                                   q,
                                                   0,
                                                   1u,
                                                   UOCR_MOE_SHARED_INTERMEDIATE,
                                                   mid,
                                                   v,
                                                   "integrated decode MoE shared experts",
                                                   error,
                                                   error_size) ||
                !metal_expert_interleaved_slab_for_layer(ctx, layer, &expert_slab, error, error_size) ||
                !metal_moe_intermediate_slice(ctx, slot, routed_mid_bytes, &mid) ||
                !metal_run_moe_interleaved_buffer_f16(ctx,
                                                      norm,
                                                      top_ids,
                                                      top_weights,
                                                      expert_slab,
                                                      slot,
                                                      1u,
                                                      mid,
                                                      current,
                                                      error,
                                                      error_size) ||
                !metal_run_moe_combine_buffer_f16(ctx, current, v, q, 1u, output, error, error_size)) {
                return 0;
            }
        }

        current = output;
        current_segment = output_segment;
    }

    if (current_segment != 0u) {
        if (!metal_copy_slice(ctx, current, scratch[0], hidden_bytes, "integrated decode final-hidden copy", error, error_size)) {
            return 0;
        }
    }
    return 1;
}

static int metal_run_decoder_prefill_text_f16(uocr_metal_context *ctx,
                                              uint32_t slot,
                                              uint32_t n_tokens,
                                              char *error,
                                              size_t error_size) {
    uocr_metal_buffer_slice prompt;
    uint64_t hidden_bytes = 0u;
    if (!metal_prompt_arena_slice(ctx, slot, n_tokens, &prompt, &hidden_bytes)) {
        return metal_fail(error, error_size, "integrated prefill requires a valid prompt arena slice");
    }

    uocr_metal_buffer_slice scratch[UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS];
    for (uint32_t i = 0u; i < UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS; ++i) {
        uint64_t bytes = 0u;
        if (!metal_hidden_arena_segment_slice(ctx, slot, i, n_tokens, &scratch[i], &bytes) || bytes != hidden_bytes) {
            return metal_fail(error, error_size, "integrated prefill requires hidden scratch segment %u", i);
        }
    }

    uocr_metal_buffer_slice current = prompt;
    uint32_t current_segment = UINT32_MAX;
    for (uint32_t layer = 0u; layer < UOCR_DECODER_LAYERS; ++layer) {
        uint32_t available[UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS];
        uint32_t available_count = 0u;
        for (uint32_t i = 0u; i < UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS; ++i) {
            if (i != current_segment) {
                available[available_count++] = i;
            }
        }
        if (available_count < 4u) {
            return metal_fail(error, error_size, "integrated prefill scratch assignment failed at layer %u", layer);
        }
        const uint32_t norm_segment = available[0];
        const uint32_t q_segment = available[1];
        const uint32_t k_segment = available[2];
        const uint32_t v_segment = available[3];
        uocr_metal_buffer_slice norm = scratch[norm_segment];
        uocr_metal_buffer_slice q = scratch[q_segment];
        uocr_metal_buffer_slice k = scratch[k_segment];
        uocr_metal_buffer_slice v = scratch[v_segment];
        uocr_metal_buffer_slice output = k;
        const uint32_t output_segment = k_segment;

        char label[64];
        snprintf(label, sizeof(label), "layer%u input RMSNorm", layer);
        const uocr_metal_decoder_binding *input_norm = metal_require_decoder_binding(ctx,
                                                                                     uocr_tensor_id_layer_input_norm(layer),
                                                                                     label,
                                                                                     error,
                                                                                     error_size);
        const uocr_metal_decoder_binding *post_norm = metal_require_decoder_binding(ctx,
                                                                                    uocr_tensor_id_layer_post_attn_norm(layer),
                                                                                    "post-attention norm",
                                                                                    error,
                                                                                    error_size);
        const uocr_metal_decoder_binding *q_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_Q),
                                                                                   "attention q_proj",
                                                                                   error,
                                                                                   error_size);
        const uocr_metal_decoder_binding *k_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_K),
                                                                                   "attention k_proj",
                                                                                   error,
                                                                                   error_size);
        const uocr_metal_decoder_binding *v_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_V),
                                                                                   "attention v_proj",
                                                                                   error,
                                                                                   error_size);
        const uocr_metal_decoder_binding *o_weight = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_attn(layer, UOCR_TENSOR_PROJ_O),
                                                                                   "attention o_proj",
                                                                                   error,
                                                                                   error_size);
        if (input_norm == NULL || post_norm == NULL || q_weight == NULL || k_weight == NULL || v_weight == NULL || o_weight == NULL) {
            return 0;
        }

        if (!metal_run_rmsnorm_buffer_f16(ctx, current, input_norm, n_tokens, norm, "integrated input RMSNorm", error, error_size) ||
            !metal_run_attention_qkv_buffer_f16(ctx, norm, q_weight, k_weight, v_weight, n_tokens, q, k, v, error, error_size) ||
            !metal_run_rope_qk_buffer_f16(ctx, q, k, n_tokens, 0u, error, error_size) ||
            !metal_run_kv_write_buffer_f16(ctx, k, v, n_tokens, layer, slot, n_tokens, 0u, error, error_size) ||
            !metal_run_prefill_attention_buffer_f16(ctx, q, k, v, n_tokens, norm, error, error_size) ||
            !metal_run_attention_output_residual_buffer_f16(ctx, norm, o_weight, current, n_tokens, q, error, error_size) ||
            !metal_run_rmsnorm_buffer_f16(ctx, q, post_norm, n_tokens, norm, "integrated post-attention RMSNorm", error, error_size)) {
            return 0;
        }

        if (layer == 0u) {
            const uocr_metal_decoder_binding *gate = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_GATE),
                                                                                   "layer0 dense gate",
                                                                                   error,
                                                                                   error_size);
            const uocr_metal_decoder_binding *up = metal_require_decoder_binding(ctx,
                                                                                 uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_UP),
                                                                                 "layer0 dense up",
                                                                                 error,
                                                                                 error_size);
            const uocr_metal_decoder_binding *down = metal_require_decoder_binding(ctx,
                                                                                   uocr_tensor_id_layer_dense_mlp(UOCR_TENSOR_PROJ_DOWN),
                                                                                   "layer0 dense down",
                                                                                   error,
                                                                                   error_size);
            const uint64_t mid_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_DENSE_LAYER0_INTERMEDIATE * 2u;
            uocr_metal_buffer_slice mid;
            if (gate == NULL || up == NULL || down == NULL ||
                !metal_moe_intermediate_slice(ctx, slot, mid_bytes, &mid) ||
                !metal_run_dense_swiglu_buffer_f16(ctx,
                                                   norm,
                                                   gate,
                                                   up,
                                                   down,
                                                   q,
                                                   1,
                                                   n_tokens,
                                                   UOCR_DENSE_LAYER0_INTERMEDIATE,
                                                   mid,
                                                   output,
                                                   "integrated layer0 dense SwiGLU",
                                                   error,
                                                   error_size)) {
                return 0;
            }
        } else {
            const uocr_metal_decoder_binding *router = metal_require_decoder_binding(ctx,
                                                                                     uocr_tensor_id_moe_router(layer),
                                                                                     "MoE router",
                                                                                     error,
                                                                                     error_size);
            const uocr_metal_decoder_binding *shared_gate = metal_require_decoder_binding(ctx,
                                                                                          uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_GATE),
                                                                                          "MoE shared gate",
                                                                                          error,
                                                                                          error_size);
            const uocr_metal_decoder_binding *shared_up = metal_require_decoder_binding(ctx,
                                                                                        uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_UP),
                                                                                        "MoE shared up",
                                                                                        error,
                                                                                        error_size);
            const uocr_metal_decoder_binding *shared_down = metal_require_decoder_binding(ctx,
                                                                                          uocr_tensor_id_moe_shared(layer, UOCR_TENSOR_PROJ_DOWN),
                                                                                          "MoE shared down",
                                                                                          error,
                                                                                          error_size);
            uocr_metal_buffer_slice top_ids;
            uocr_metal_buffer_slice top_weights;
            uocr_metal_buffer_slice expert_slab;
            const uint64_t shared_mid_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE * 2u;
            const uint64_t routed_mid_bytes = (uint64_t)n_tokens * (uint64_t)UOCR_MOE_TOP_K *
                                              (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE * 2u;
            uocr_metal_buffer_slice mid;
            if (router == NULL || shared_gate == NULL || shared_up == NULL || shared_down == NULL ||
                !metal_run_moe_router_buffer_f16(ctx, norm, router, slot, n_tokens, &top_ids, &top_weights, error, error_size) ||
                !metal_moe_intermediate_slice(ctx, slot, shared_mid_bytes, &mid) ||
                !metal_run_dense_swiglu_buffer_f16(ctx,
                                                   norm,
                                                   shared_gate,
                                                   shared_up,
                                                   shared_down,
                                                   q,
                                                   0,
                                                   n_tokens,
                                                   UOCR_MOE_SHARED_INTERMEDIATE,
                                                   mid,
                                                   v,
                                                   "integrated MoE shared experts",
                                                   error,
                                                   error_size) ||
                !metal_expert_interleaved_slab_for_layer(ctx, layer, &expert_slab, error, error_size) ||
                !metal_moe_intermediate_slice(ctx, slot, routed_mid_bytes, &mid) ||
                !metal_run_moe_interleaved_buffer_f16(ctx,
                                                      norm,
                                                      top_ids,
                                                      top_weights,
                                                      expert_slab,
                                                      slot,
                                                      n_tokens,
                                                      mid,
                                                      current,
                                                      error,
                                                      error_size) ||
                !metal_run_moe_combine_buffer_f16(ctx, current, v, q, n_tokens, output, error, error_size)) {
                return 0;
            }
        }

        current = output;
        current_segment = output_segment;
    }

    if (current_segment != 0u) {
        if (!metal_copy_slice(ctx, current, scratch[0], hidden_bytes, "integrated prefill final-hidden copy", error, error_size)) {
            return 0;
        }
        current_segment = 0u;
    }
    ctx->has_integrated_prefill = 1;
    ctx->integrated_prefill_slot = slot;
    ctx->integrated_prefill_tokens = n_tokens;
    ctx->integrated_prefill_final_segment = current_segment;
    return 1;
}

int uocr_metal_context_generate_f16(uocr_metal_context *ctx,
                                    const uocr_metal_decoder_request_f16 *request,
                                    uocr_metal_decoder_result_f16 *result,
                                    char *error,
                                    size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || request == NULL || result == NULL) {
        return metal_fail(error, error_size, "invalid Metal integrated decoder request");
    }
    metal_decoder_result_reset(result);

    if (!metal_decoder_runtime_arenas_ready(ctx)) {
        return metal_fail(error, error_size, "Metal integrated decoder requires allocated runtime arenas");
    }
    if (!uocr_metal_context_decoder_bindings_ready(ctx) ||
        metal_find_decoder_binding(ctx, UOCR_TENSOR_ID_TOK_EMBED) == NULL ||
        metal_find_decoder_binding(ctx, UOCR_TENSOR_ID_FINAL_NORM) == NULL ||
        metal_find_decoder_binding(ctx, UOCR_TENSOR_ID_LM_HEAD) == NULL) {
        const char *binding_error = ctx->decoder_binding_error[0] != '\0' ?
                                        ctx->decoder_binding_error :
                                        "decoder tensor bindings are not validated";
        return metal_fail(error,
                          error_size,
                          "Metal integrated decoder requires validated fp16 decoder tensor bindings: %s",
                          binding_error);
    }
    if (request->input_ids == NULL || request->image_mask == NULL || request->n_tokens == 0u) {
        return metal_fail(error, error_size, "Metal integrated decoder requires input ids, image mask, and prompt tokens");
    }
    if (request->slot >= ctx->kv_cache_layout.batch_slots) {
        return metal_fail(error,
                          error_size,
                          "Metal integrated decoder slot %u exceeds allocated batch slots %u",
                          request->slot,
                          ctx->kv_cache_layout.batch_slots);
    }
    if (request->n_tokens > ctx->kv_cache_layout.prompt_token_capacity) {
        return metal_fail(error,
                          error_size,
                          "Metal integrated decoder prompt length %u exceeds prompt arena capacity %u",
                          request->n_tokens,
                          ctx->kv_cache_layout.prompt_token_capacity);
    }
    if (request->n_tokens > UOCR_MAX_POSITIONS || request->max_new_tokens > UOCR_MAX_POSITIONS - request->n_tokens) {
        return metal_fail(error, error_size, "Metal integrated decoder sequence length exceeds max positions");
    }
    if (request->no_repeat_ngram_size == 0u && request->no_repeat_window != 0u) {
        return metal_fail(error, error_size, "Metal integrated decoder no_repeat_window is set but no_repeat_ngram_size is zero");
    }
    if (request->no_repeat_ngram_size > UOCR_MAX_POSITIONS || request->no_repeat_window > UOCR_MAX_POSITIONS) {
        return metal_fail(error, error_size, "Metal integrated decoder no-repeat configuration exceeds max positions");
    }
    if (request->max_new_tokens > 0u &&
        (result->generated_ids == NULL || result->generated_capacity < request->max_new_tokens)) {
        return metal_fail(error,
                          error_size,
                          "Metal integrated decoder generated-token output capacity %u is smaller than requested %u",
                          result->generated_capacity,
                          request->max_new_tokens);
    }

    uint32_t image_mask_count = 0u;
    for (uint32_t i = 0u; i < request->n_tokens; ++i) {
        if (request->input_ids[i] < 0 || (uint32_t)request->input_ids[i] >= UOCR_VOCAB_SIZE) {
            return metal_fail(error,
                              error_size,
                              "Metal integrated decoder token id %d at position %u is outside vocab %u",
                              request->input_ids[i],
                              i,
                              UOCR_VOCAB_SIZE);
        }
        if (request->image_mask[i] > 1u) {
            return metal_fail(error,
                              error_size,
                              "Metal integrated decoder image mask value %u at position %u is not 0/1",
                              (unsigned)request->image_mask[i],
                              i);
        }
        image_mask_count += request->image_mask[i] != 0u ? 1u : 0u;
    }

    if (request->image_span_length == 0u) {
        if (request->image_span_start != UINT32_MAX || request->image_features_f16 != NULL || image_mask_count != 0u) {
            return metal_fail(error,
                              error_size,
                              "Metal integrated decoder text-only requests must use UINT32_MAX image span, no image features, and an empty image mask");
        }
    } else {
        if (request->image_features_f16 == NULL || request->image_span_start == UINT32_MAX ||
            request->image_span_start > request->n_tokens ||
            request->image_span_length > request->n_tokens - request->image_span_start) {
            return metal_fail(error, error_size, "Metal integrated decoder image span/features are invalid");
        }
        if (image_mask_count != request->image_span_length) {
            return metal_fail(error,
                              error_size,
                              "Metal integrated decoder image mask count %u does not match image span length %u",
                              image_mask_count,
                              request->image_span_length);
        }
        for (uint32_t i = 0u; i < request->image_span_length; ++i) {
            const uint32_t pos = request->image_span_start + i;
            if (request->image_mask[pos] == 0u) {
                return metal_fail(error,
                                  error_size,
                                  "Metal integrated decoder image span position %u is not marked in image mask",
                                  pos);
            }
        }
    }

    if (!uocr_metal_context_assemble_prompt_from_model_to_arena_f16(ctx,
                                                                    request->input_ids,
                                                                    request->n_tokens,
                                                                    request->image_span_length != 0u ? request->image_span_start : UINT32_MAX,
                                                                    request->image_span_length,
                                                                    request->image_features_f16,
                                                                    request->slot,
                                                                    error,
                                                                    error_size)) {
        char detail[512];
        (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
        return metal_fail(error, error_size, "failed to assemble integrated prompt into Metal arena: %s", detail);
    }

    if (!metal_run_decoder_prefill_text_f16(ctx, request->slot, request->n_tokens, error, error_size)) {
        char detail[512];
        (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
        ctx->has_integrated_prefill = 0;
        return metal_fail(error, error_size, "integrated Metal fp16 prompt prefill failed: %s", detail);
    }

    if (request->max_new_tokens == 0u) {
        metal_clear_error(error, error_size);
        return 1;
    }

    uint64_t sequence_len_u64 = 0u;
    uint64_t sequence_bytes = 0u;
    if (!checked_add_u64((uint64_t)request->n_tokens, (uint64_t)request->max_new_tokens, &sequence_len_u64) ||
        sequence_len_u64 > (uint64_t)UINT32_MAX ||
        !checked_mul_u64(sequence_len_u64, (uint64_t)sizeof(int32_t), &sequence_bytes) ||
        sequence_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "integrated Metal fp16 decode sequence buffer size overflow");
    }
    int32_t *sequence = (int32_t *)uocr_malloc((size_t)sequence_bytes);
    if (sequence == NULL) {
        return metal_fail(error, error_size, "failed to allocate integrated Metal fp16 decode sequence buffer");
    }
    memcpy(sequence, request->input_ids, (size_t)request->n_tokens * sizeof(int32_t));
    uint32_t sequence_len = request->n_tokens;

    uocr_metal_buffer_slice logits;
    uocr_metal_buffer_slice token_slot;
    uocr_metal_buffer_slice norm_scratch;
    uint64_t one_hidden_bytes = 0u;
    if (!metal_logits_arena_slices(ctx, request->slot, &logits, &token_slot) ||
        !metal_hidden_arena_segment_slice(ctx, request->slot, 1u, 1u, &norm_scratch, &one_hidden_bytes) ||
        one_hidden_bytes != (uint64_t)UOCR_HIDDEN_SIZE * 2u) {
        uocr_free(sequence);
        return metal_fail(error, error_size, "integrated Metal fp16 decode arena slices are invalid");
    }

    uocr_metal_buffer_slice hidden_for_selection;
    if (!metal_hidden_arena_token_slice(ctx,
                                        request->slot,
                                        ctx->integrated_prefill_final_segment,
                                        request->n_tokens - 1u,
                                        &hidden_for_selection)) {
        uocr_free(sequence);
        return metal_fail(error, error_size, "integrated Metal fp16 prefill final-token hidden slice is invalid");
    }

    if (!metal_prewarm_integrated_decode_pipelines(ctx, error, error_size)) {
        char detail[512];
        (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
        uocr_free(sequence);
        return metal_fail(error, error_size, "failed to prepare integrated Metal fp16 decode pipelines: %s", detail);
    }

    const uocr_metal_hot_path_alloc_guard decode_loop_guard =
        metal_hot_path_alloc_guard_begin(ctx);
    int decode_loop_ok = 1;
    char decode_loop_error[512];
    decode_loop_error[0] = '\0';

#define UOCR_METAL_DECODE_LOOP_FAIL(...)                    \
    do {                                                    \
        (void)snprintf(decode_loop_error,                   \
                       sizeof(decode_loop_error),           \
                       __VA_ARGS__);                        \
        decode_loop_ok = 0;                                 \
        goto decode_loop_done;                              \
    } while (0)

    while (result->generated_count < request->max_new_tokens) {
        uint32_t token_id = UINT32_MAX;
        float score = 0.0f;
        if (!metal_select_next_token_from_hidden_slice_f16(ctx,
                                                           hidden_for_selection,
                                                           norm_scratch,
                                                           logits,
                                                           token_slot,
                                                           sequence,
                                                           sequence_len,
                                                           request->no_repeat_ngram_size,
                                                           request->no_repeat_window,
                                                           &token_id,
                                                           &score,
                                                           error,
                                                           error_size)) {
            char detail[512];
            (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
            UOCR_METAL_DECODE_LOOP_FAIL("integrated Metal fp16 next-token selection failed: %s", detail);
        }
        if (token_id >= UOCR_VOCAB_SIZE) {
            UOCR_METAL_DECODE_LOOP_FAIL("integrated Metal fp16 selected invalid token id %u", token_id);
        }

        result->generated_ids[result->generated_count] = (int32_t)token_id;
        if (result->generated_scores_f32_or_null != NULL) {
            result->generated_scores_f32_or_null[result->generated_count] = score;
        }
        sequence[sequence_len++] = (int32_t)token_id;
        ++result->generated_count;
        result->last_token_id = token_id;
        result->last_score_f32 = score;
        if (token_id == (uint32_t)UOCR_TOKEN_EOS) {
            result->stopped_on_eos = 1u;
            break;
        }
        if (result->generated_count >= request->max_new_tokens) {
            break;
        }

        int32_t *token_ptr = metal_token_slice_cpu_ptr(token_slot);
        if (token_ptr == NULL) {
            UOCR_METAL_DECODE_LOOP_FAIL("integrated Metal fp16 token-id arena is not CPU-visible");
        }
        *token_ptr = (int32_t)token_id;
        uocr_metal_buffer_slice decode_input;
        uint64_t decode_input_bytes = 0u;
        if (!metal_hidden_arena_segment_slice(ctx, request->slot, 0u, 1u, &decode_input, &decode_input_bytes) ||
            decode_input_bytes != (uint64_t)UOCR_HIDDEN_SIZE * 2u ||
            !metal_run_token_embedding_from_token_slot_f16(ctx, token_slot, decode_input, error, error_size)) {
            char detail[512];
            (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
            UOCR_METAL_DECODE_LOOP_FAIL("integrated Metal fp16 generated-token embedding failed: %s", detail);
        }
        if (!metal_run_decoder_decode_one_f16(ctx,
                                              request->slot,
                                              request->n_tokens,
                                              result->generated_count,
                                              error,
                                              error_size)) {
            char detail[512];
            (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
            UOCR_METAL_DECODE_LOOP_FAIL("integrated Metal fp16 decode step failed: %s", detail);
        }
        if (!metal_hidden_arena_segment_slice(ctx, request->slot, 0u, 1u, &hidden_for_selection, &one_hidden_bytes) ||
            one_hidden_bytes != (uint64_t)UOCR_HIDDEN_SIZE * 2u) {
            UOCR_METAL_DECODE_LOOP_FAIL("integrated Metal fp16 decode hidden slice is invalid");
        }
    }

decode_loop_done:
#undef UOCR_METAL_DECODE_LOOP_FAIL

    if (!metal_hot_path_alloc_guard_end(ctx,
                                        &decode_loop_guard,
                                        "integrated Metal fp16 decode token loop",
                                        error,
                                        error_size)) {
        uocr_free(sequence);
        return 0;
    }
    uocr_free(sequence);
    if (!decode_loop_ok) {
        return metal_fail(error, error_size, "%s", decode_loop_error);
    }
    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_get_rows_f16(uocr_metal_context *ctx,
                                    const uint16_t *table_f16,
                                    uint32_t table_rows,
                                    uint32_t row_width,
                                    const int32_t *row_ids,
                                    uint32_t n_row_ids,
                                    uocr_metal_get_rows_output_type output_type,
                                    void *out,
                                    char *error,
                                    size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || table_f16 == NULL || row_ids == NULL || out == NULL ||
        table_rows == 0u || row_width == 0u || n_row_ids == 0u) {
        return metal_fail(error, error_size, "invalid Metal get-rows request");
    }
    if (output_type != UOCR_METAL_GET_ROWS_OUTPUT_F16 && output_type != UOCR_METAL_GET_ROWS_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal get-rows output type %d", (int)output_type);
    }

    for (uint32_t i = 0u; i < n_row_ids; ++i) {
        if (row_ids[i] < 0 || (uint32_t)row_ids[i] >= table_rows) {
            return metal_fail(error,
                              error_size,
                              "Metal get-rows id %d at position %u is outside table rows %u",
                              row_ids[i],
                              i,
                              table_rows);
        }
    }

    uint64_t table_values = 0u;
    uint64_t table_bytes = 0u;
    uint64_t out_values = 0u;
    uint64_t out_bytes = 0u;
    uint64_t ids_bytes = 0u;
    const uint64_t out_element_bytes = output_type == UOCR_METAL_GET_ROWS_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)table_rows, (uint64_t)row_width, &table_values) ||
        !checked_mul_u64(table_values, 2u, &table_bytes) ||
        !checked_mul_u64((uint64_t)n_row_ids, (uint64_t)row_width, &out_values) ||
        !checked_mul_u64(out_values, out_element_bytes, &out_bytes) ||
        !checked_mul_u64((uint64_t)n_row_ids, (uint64_t)sizeof(int32_t), &ids_bytes)) {
        return metal_fail(error, error_size, "Metal get-rows byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (table_bytes > max_buffer_length || ids_bytes > max_buffer_length || out_bytes > max_buffer_length ||
        table_bytes > (uint64_t)SIZE_MAX || ids_bytes > (uint64_t)SIZE_MAX || out_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal get-rows buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_GET_ROWS_OUTPUT_F16 ?
                                        "uocr_get_rows_f16_to_f16" :
                                        "uocr_get_rows_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> table = [ctx->device newBufferWithBytes:table_f16
                                                       length:(NSUInteger)table_bytes
                                                      options:MTLResourceStorageModeShared];
        if (table == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal get-rows table buffer");
        }
        table.label = @"uocr_get_rows_table_f16";

        id<MTLBuffer> ids = [ctx->device newBufferWithBytes:row_ids
                                                     length:(NSUInteger)ids_bytes
                                                    options:MTLResourceStorageModeShared];
        if (ids == nil) {
            [table release];
            return metal_fail(error, error_size, "failed to allocate Metal get-rows id buffer");
        }
        ids.label = @"uocr_get_rows_ids";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)out_bytes
                                                     options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [ids release];
            [table release];
            return metal_fail(error, error_size, "failed to allocate Metal get-rows output buffer");
        }
        dst.label = @"uocr_get_rows_output";
        memset([dst contents], 0, (size_t)out_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [ids release];
            [table release];
            return metal_fail(error, error_size, "failed to create Metal get-rows command buffer");
        }
        cb.label = @"uocr_get_rows_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [ids release];
            [table release];
            return metal_fail(error, error_size, "failed to create Metal get-rows command encoder");
        }

        uocr_metal_get_rows_params params;
        params.table_rows = table_rows;
        params.row_width = row_width;
        params.n_row_ids = n_row_ids;
        params.reserved = 0u;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:table offset:0u atIndex:0u];
        [enc setBuffer:ids offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];

        NSUInteger threads_per_group = pipeline.threadExecutionWidth;
        if (threads_per_group == 0u) {
            threads_per_group = 1u;
        }
        if (threads_per_group > (NSUInteger)row_width) {
            threads_per_group = (NSUInteger)row_width;
        }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)row_width, (NSUInteger)n_row_ids, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [ids release];
            [table release];
            return metal_fail(error, error_size, "Metal get-rows command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)out_bytes);
        [dst release];
        [ids release];
        [table release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_get_rows_q8_0(uocr_metal_context *ctx,
                                     const void *table_q8_0,
                                     uint32_t table_rows,
                                     uint32_t logical_width,
                                     uint32_t physical_width,
                                     const int32_t *row_ids,
                                     uint32_t n_row_ids,
                                     uocr_metal_get_rows_output_type output_type,
                                     void *out,
                                     char *error,
                                     size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || table_q8_0 == NULL || row_ids == NULL || out == NULL || table_rows == 0u ||
        logical_width == 0u || physical_width == 0u || n_row_ids == 0u) {
        return metal_fail(error, error_size, "invalid Metal Q8_0 get-rows request");
    }
    if (logical_width > physical_width || (physical_width % 32u) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal Q8_0 get-rows widths logical=%u physical=%u",
                          logical_width,
                          physical_width);
    }
    if (output_type != UOCR_METAL_GET_ROWS_OUTPUT_F16 && output_type != UOCR_METAL_GET_ROWS_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal Q8_0 get-rows output type %d", (int)output_type);
    }
    for (uint32_t i = 0u; i < n_row_ids; ++i) {
        if (row_ids[i] < 0 || (uint32_t)row_ids[i] >= table_rows) {
            return metal_fail(error,
                              error_size,
                              "Metal Q8_0 get-rows id %d at position %u is outside table rows %u",
                              row_ids[i],
                              i,
                              table_rows);
        }
    }

    uint64_t blocks_per_row = 0u;
    uint64_t row_size = 0u;
    uint64_t table_bytes = 0u;
    uint64_t out_values = 0u;
    uint64_t out_bytes = 0u;
    uint64_t ids_bytes = 0u;
    const uint64_t out_element_bytes = output_type == UOCR_METAL_GET_ROWS_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)(physical_width / 32u), 34u, &row_size) ||
        !checked_mul_u64((uint64_t)table_rows, row_size, &table_bytes) ||
        !checked_mul_u64((uint64_t)n_row_ids, (uint64_t)logical_width, &out_values) ||
        !checked_mul_u64(out_values, out_element_bytes, &out_bytes) ||
        !checked_mul_u64((uint64_t)n_row_ids, (uint64_t)sizeof(int32_t), &ids_bytes)) {
        return metal_fail(error, error_size, "Metal Q8_0 get-rows byte-size overflow");
    }
    blocks_per_row = (uint64_t)physical_width / 32u;
    if (blocks_per_row == 0u || row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal Q8_0 get-rows row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (table_bytes > max_buffer_length || ids_bytes > max_buffer_length || out_bytes > max_buffer_length ||
        table_bytes > (uint64_t)SIZE_MAX || ids_bytes > (uint64_t)SIZE_MAX || out_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal Q8_0 get-rows buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_GET_ROWS_OUTPUT_F16 ?
                                        "uocr_get_rows_q8_0_to_f16" :
                                        "uocr_get_rows_q8_0_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> table = [ctx->device newBufferWithBytes:table_q8_0
                                                       length:(NSUInteger)table_bytes
                                                      options:MTLResourceStorageModeShared];
        if (table == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal Q8_0 get-rows table buffer");
        }
        table.label = @"uocr_get_rows_table_q8_0";

        id<MTLBuffer> ids = [ctx->device newBufferWithBytes:row_ids
                                                     length:(NSUInteger)ids_bytes
                                                    options:MTLResourceStorageModeShared];
        if (ids == nil) {
            [table release];
            return metal_fail(error, error_size, "failed to allocate Metal Q8_0 get-rows id buffer");
        }
        ids.label = @"uocr_get_rows_q8_ids";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)out_bytes
                                                     options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [ids release];
            [table release];
            return metal_fail(error, error_size, "failed to allocate Metal Q8_0 get-rows output buffer");
        }
        dst.label = @"uocr_get_rows_q8_output";
        memset([dst contents], 0, (size_t)out_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [ids release];
            [table release];
            return metal_fail(error, error_size, "failed to create Metal Q8_0 get-rows command buffer");
        }
        cb.label = @"uocr_get_rows_q8_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [ids release];
            [table release];
            return metal_fail(error, error_size, "failed to create Metal Q8_0 get-rows command encoder");
        }

        uocr_metal_get_rows_q8_params params;
        params.table_rows = table_rows;
        params.logical_width = logical_width;
        params.physical_width = physical_width;
        params.n_row_ids = n_row_ids;
        params.row_size = (uint32_t)row_size;
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:table offset:0u atIndex:0u];
        [enc setBuffer:ids offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];

        NSUInteger threads_per_group = pipeline.threadExecutionWidth;
        if (threads_per_group == 0u) {
            threads_per_group = 1u;
        }
        if (threads_per_group > (NSUInteger)logical_width) {
            threads_per_group = (NSUInteger)logical_width;
        }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)logical_width, (NSUInteger)n_row_ids, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [ids release];
            [table release];
            return metal_fail(error, error_size, "Metal Q8_0 get-rows command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)out_bytes);
        [dst release];
        [ids release];
        [table release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

static int validate_prompt_text_token_ids(const int32_t *input_ids,
                                          uint32_t n_tokens,
                                          uint32_t table_rows,
                                          uint32_t image_span_start,
                                          uint32_t image_span_length,
                                          char *error,
                                          size_t error_size) {
    const int has_image_span = image_span_length != 0u;
    const uint64_t image_span_end = has_image_span ? (uint64_t)image_span_start + (uint64_t)image_span_length : 0u;
    for (uint32_t i = 0u; i < n_tokens; ++i) {
        if (has_image_span && i >= image_span_start && (uint64_t)i < image_span_end) {
            continue;
        }
        if (input_ids[i] < 0 || (uint32_t)input_ids[i] >= table_rows) {
            return metal_fail(error,
                              error_size,
                              "Metal prompt token id %d at position %u is outside embedding rows %u",
                              input_ids[i],
                              i,
                              table_rows);
        }
    }
    return 1;
}

static int metal_context_assemble_prompt_f16_with_table_buffer_to_buffer(uocr_metal_context *ctx,
                                                                          id<MTLBuffer> table_buffer,
                                                                          NSUInteger table_offset,
                                                                          uint32_t table_rows,
                                                                          uint32_t hidden_size,
                                                                          const int32_t *input_ids,
                                                                          uint32_t n_tokens,
                                                                          uint32_t image_span_start,
                                                                          uint32_t image_span_length,
                                                                          const uint16_t *image_features_f16,
                                                                          id<MTLBuffer> dst_buffer,
                                                                          NSUInteger dst_offset,
                                                                          const char *op_name,
                                                                          char *error,
                                                                          size_t error_size) {
    const char *op = op_name != NULL ? op_name : "Metal prompt assembly";
    if (ctx == NULL || table_buffer == nil || input_ids == NULL || dst_buffer == nil ||
        table_rows == 0u || hidden_size == 0u || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid %s request", op);
    }

    const int has_image_span = image_span_length != 0u;
    if (has_image_span) {
        uint64_t image_end = 0u;
        if (image_span_start == UINT32_MAX ||
            !checked_add_u64((uint64_t)image_span_start, (uint64_t)image_span_length, &image_end) ||
            image_span_start >= n_tokens || image_end > (uint64_t)n_tokens) {
            return metal_fail(error,
                              error_size,
                              "%s image span [%u,%llu) is outside %u tokens",
                              op,
                              image_span_start,
                              (unsigned long long)image_end,
                              n_tokens);
        }
        if (image_features_f16 == NULL) {
            return metal_fail(error, error_size, "%s image span requires image features", op);
        }
    } else if (image_span_start != UINT32_MAX) {
        return metal_fail(error, error_size, "%s text-only path should use UINT32_MAX image span start", op);
    }

    if (!validate_prompt_text_token_ids(input_ids,
                                        n_tokens,
                                        table_rows,
                                        image_span_start,
                                        image_span_length,
                                        error,
                                        error_size)) {
        return 0;
    }

    uint64_t table_values = 0u;
    uint64_t table_bytes = 0u;
    uint64_t token_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    uint64_t image_values = 0u;
    uint64_t image_bytes = 0u;
    if (!checked_mul_u64((uint64_t)table_rows, (uint64_t)hidden_size, &table_values) ||
        !checked_mul_u64(table_values, 2u, &table_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)sizeof(int32_t), &token_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &output_values) ||
        !checked_mul_u64(output_values, 2u, &output_bytes) ||
        !checked_mul_u64((uint64_t)image_span_length, (uint64_t)hidden_size, &image_values) ||
        !checked_mul_u64(image_values, 2u, &image_bytes)) {
        return metal_fail(error, error_size, "%s byte-size overflow", op);
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (token_bytes > max_buffer_length || image_bytes > max_buffer_length ||
        token_bytes > (uint64_t)SIZE_MAX || image_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "%s transient buffers exceed maxBufferLength %llu",
                          op,
                          (unsigned long long)max_buffer_length);
    }
    const uint64_t table_offset_u64 = (uint64_t)table_offset;
    const uint64_t table_length = (uint64_t)[table_buffer length];
    uint64_t table_end = 0u;
    if (!checked_add_u64(table_offset_u64, table_bytes, &table_end) || table_end > table_length) {
        return metal_fail(error,
                          error_size,
                          "%s embedding table range [%llu,%llu) exceeds Metal buffer length %llu",
                          op,
                          (unsigned long long)table_offset_u64,
                          (unsigned long long)table_end,
                          (unsigned long long)table_length);
    }
    const uint64_t dst_offset_u64 = (uint64_t)dst_offset;
    const uint64_t dst_length = (uint64_t)[dst_buffer length];
    uint64_t dst_end = 0u;
    if (!checked_add_u64(dst_offset_u64, output_bytes, &dst_end) || dst_end > dst_length) {
        return metal_fail(error,
                          error_size,
                          "%s output range [%llu,%llu) exceeds Metal buffer length %llu",
                          op,
                          (unsigned long long)dst_offset_u64,
                          (unsigned long long)dst_end,
                          (unsigned long long)dst_length);
    }

    @autoreleasepool {
        const char *function_name = has_image_span ? "uocr_assemble_prompt_with_image_f16" : "uocr_assemble_prompt_text_f16";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> tokens = [ctx->device newBufferWithBytes:input_ids
                                                        length:(NSUInteger)token_bytes
                                                       options:MTLResourceStorageModeShared];
        if (tokens == nil) {
            return metal_fail(error, error_size, "failed to allocate %s token-id buffer", op);
        }
        tokens.label = @"uocr_prompt_input_ids";

        id<MTLBuffer> image_features = nil;
        if (has_image_span) {
            image_features = [ctx->device newBufferWithBytes:image_features_f16
                                                      length:(NSUInteger)image_bytes
                                                     options:MTLResourceStorageModeShared];
            if (image_features == nil) {
                [tokens release];
                return metal_fail(error, error_size, "failed to allocate %s image-feature buffer", op);
            }
            image_features.label = @"uocr_prompt_image_features_f16";
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [image_features release];
            [tokens release];
            return metal_fail(error, error_size, "failed to create %s command buffer", op);
        }
        cb.label = @"uocr_prompt_assembly_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [image_features release];
            [tokens release];
            return metal_fail(error, error_size, "failed to create %s command encoder", op);
        }

        uocr_metal_prompt_assembly_params params;
        params.table_rows = table_rows;
        params.hidden_size = hidden_size;
        params.n_tokens = n_tokens;
        params.image_span_start = has_image_span ? image_span_start : UINT32_MAX;
        params.image_span_length = image_span_length;
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:table_buffer offset:table_offset atIndex:0u];
        [enc setBuffer:tokens offset:0u atIndex:1u];
        if (has_image_span) {
            [enc setBuffer:image_features offset:0u atIndex:2u];
            [enc setBuffer:dst_buffer offset:dst_offset atIndex:3u];
            [enc setBytes:&params length:sizeof(params) atIndex:4u];
        } else {
            [enc setBuffer:dst_buffer offset:dst_offset atIndex:2u];
            [enc setBytes:&params length:sizeof(params) atIndex:3u];
        }

        NSUInteger threads_per_group = pipeline.threadExecutionWidth;
        if (threads_per_group == 0u) {
            threads_per_group = 1u;
        }
        if (threads_per_group > (NSUInteger)hidden_size) {
            threads_per_group = (NSUInteger)hidden_size;
        }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_size, (NSUInteger)n_tokens, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [image_features release];
            [tokens release];
            return metal_fail(error, error_size, "%s command failed: %s", op, [description UTF8String]);
        }

        [image_features release];
        [tokens release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

static int metal_context_assemble_prompt_f16_with_table_buffer(uocr_metal_context *ctx,
                                                                id<MTLBuffer> table_buffer,
                                                                NSUInteger table_offset,
                                                                uint32_t table_rows,
                                                                uint32_t hidden_size,
                                                                const int32_t *input_ids,
                                                                uint32_t n_tokens,
                                                                uint32_t image_span_start,
                                                                uint32_t image_span_length,
                                                                const uint16_t *image_features_f16,
                                                                uint16_t *out_prompt_f16,
                                                                const char *op_name,
                                                                char *error,
                                                                size_t error_size) {
    const char *op = op_name != NULL ? op_name : "Metal prompt assembly";
    if (ctx == NULL || table_buffer == nil || input_ids == NULL || out_prompt_f16 == NULL ||
        table_rows == 0u || hidden_size == 0u || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid %s request", op);
    }

    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &output_values) ||
        !checked_mul_u64(output_values, 2u, &output_bytes) || output_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > metal_device_max_buffer_length(ctx->device)) {
        return metal_fail(error, error_size, "%s output byte-size overflow", op);
    }

    @autoreleasepool {
        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes
                                                     options:MTLResourceStorageModeShared];
        if (dst == nil) {
            return metal_fail(error, error_size, "failed to allocate %s output buffer", op);
        }
        dst.label = @"uocr_prompt_embeddings_f16";

        const int status = metal_context_assemble_prompt_f16_with_table_buffer_to_buffer(ctx,
                                                                                         table_buffer,
                                                                                         table_offset,
                                                                                         table_rows,
                                                                                         hidden_size,
                                                                                         input_ids,
                                                                                         n_tokens,
                                                                                         image_span_start,
                                                                                         image_span_length,
                                                                                         image_features_f16,
                                                                                         dst,
                                                                                         0u,
                                                                                         op,
                                                                                         error,
                                                                                         error_size);
        if (status == 1) {
            memcpy(out_prompt_f16, [dst contents], (size_t)output_bytes);
        }
        [dst release];
        return status;
    }
}

int uocr_metal_context_assemble_prompt_f16(uocr_metal_context *ctx,
                                           const uint16_t *embedding_table_f16,
                                           uint32_t table_rows,
                                           uint32_t hidden_size,
                                           const int32_t *input_ids,
                                           uint32_t n_tokens,
                                           uint32_t image_span_start,
                                           uint32_t image_span_length,
                                           const uint16_t *image_features_f16,
                                           uint16_t *out_prompt_f16,
                                           char *error,
                                           size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || embedding_table_f16 == NULL || input_ids == NULL || out_prompt_f16 == NULL ||
        table_rows == 0u || hidden_size == 0u || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal prompt assembly request");
    }

    uint64_t table_values = 0u;
    uint64_t table_bytes = 0u;
    if (!checked_mul_u64((uint64_t)table_rows, (uint64_t)hidden_size, &table_values) ||
        !checked_mul_u64(table_values, 2u, &table_bytes) || table_bytes > (uint64_t)SIZE_MAX ||
        table_bytes > metal_device_max_buffer_length(ctx->device)) {
        return metal_fail(error, error_size, "Metal prompt assembly table byte-size overflow");
    }

    @autoreleasepool {
        id<MTLBuffer> table = [ctx->device newBufferWithBytes:embedding_table_f16
                                                       length:(NSUInteger)table_bytes
                                                      options:MTLResourceStorageModeShared];
        if (table == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal prompt embedding table buffer");
        }
        table.label = @"uocr_prompt_embedding_table_f16";
        const int status = metal_context_assemble_prompt_f16_with_table_buffer(ctx,
                                                                               table,
                                                                               0u,
                                                                               table_rows,
                                                                               hidden_size,
                                                                               input_ids,
                                                                               n_tokens,
                                                                               image_span_start,
                                                                               image_span_length,
                                                                               image_features_f16,
                                                                               out_prompt_f16,
                                                                               "Metal prompt assembly",
                                                                               error,
                                                                               error_size);
        [table release];
        return status;
    }
}

int uocr_metal_context_assemble_prompt_from_model_f16(uocr_metal_context *ctx,
                                                      const int32_t *input_ids,
                                                      uint32_t n_tokens,
                                                      uint32_t image_span_start,
                                                      uint32_t image_span_length,
                                                      const uint16_t *image_features_f16,
                                                      uint16_t *out_prompt_f16,
                                                      char *error,
                                                      size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_ids == NULL || out_prompt_f16 == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal mapped prompt assembly request");
    }

    uint64_t table_values = 0u;
    uint64_t table_bytes = 0u;
    if (!checked_mul_u64((uint64_t)UOCR_VOCAB_SIZE, (uint64_t)UOCR_HIDDEN_SIZE, &table_values) ||
        !checked_mul_u64(table_values, 2u, &table_bytes)) {
        return metal_fail(error, error_size, "Metal mapped prompt assembly table byte-size overflow");
    }

    id<MTLBuffer> table = nil;
    NSUInteger table_offset = 0u;
    if (!metal_get_mapped_tensor_buffer(ctx,
                                        UOCR_TENSOR_ID_TOK_EMBED,
                                        table_bytes,
                                        &table,
                                        &table_offset,
                                        error,
                                        error_size)) {
        return 0;
    }

    return metal_context_assemble_prompt_f16_with_table_buffer(ctx,
                                                               table,
                                                               table_offset,
                                                               UOCR_VOCAB_SIZE,
                                                               UOCR_HIDDEN_SIZE,
                                                               input_ids,
                                                               n_tokens,
                                                               image_span_start,
                                                               image_span_length,
                                                               image_features_f16,
                                                               out_prompt_f16,
                                                               "Metal mapped prompt assembly",
                                                               error,
                                                               error_size);
}

int uocr_metal_context_sam_patch_embed_f16(uocr_metal_context *ctx,
                                           const void *pixels,
                                           uocr_pixel_format pixel_format,
                                           uint32_t width,
                                           uint32_t height,
                                           const uint16_t *patch_weight_f16,
                                           const uint16_t *patch_bias_f16_or_null,
                                           uint16_t *out_bhwc_f16,
                                           uint32_t *out_grid_w,
                                           uint32_t *out_grid_h,
                                           char *error,
                                           size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || pixels == NULL || patch_weight_f16 == NULL || out_bhwc_f16 == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM patch-embed request");
    }
    if (pixel_format != UOCR_PIXEL_F16_NCHW && pixel_format != UOCR_PIXEL_F32_NCHW) {
        return metal_fail(error, error_size, "unsupported Metal SAM patch-embed pixel format %d", (int)pixel_format);
    }
    if (width == 0u || height == 0u || (width % UOCR_VISION_PATCH_SIZE) != 0u || (height % UOCR_VISION_PATCH_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM patch-embed input must be non-empty and divisible by %u, got %ux%u",
                          UOCR_VISION_PATCH_SIZE,
                          width,
                          height);
    }

    const uint32_t grid_w = width / UOCR_VISION_PATCH_SIZE;
    const uint32_t grid_h = height / UOCR_VISION_PATCH_SIZE;
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_GLOBAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE ||
        grid_h > UOCR_GLOBAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE) {
        return metal_fail(error, error_size, "Metal SAM patch-embed grid %ux%u is out of range", grid_w, grid_h);
    }

    uint64_t pixel_values = 0u;
    uint64_t pixel_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_patches = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    const uint64_t pixel_element_size = pixel_format == UOCR_PIXEL_F16_NCHW ? 2u : 4u;
    if (!checked_mul_u64(3u, (uint64_t)width, &pixel_values) ||
        !checked_mul_u64(pixel_values, (uint64_t)height, &pixel_values) ||
        !checked_mul_u64(pixel_values, pixel_element_size, &pixel_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_HIDDEN_SIZE, 3u * UOCR_VISION_PATCH_SIZE * UOCR_VISION_PATCH_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &output_patches) ||
        !checked_mul_u64(output_patches, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &output_values) ||
        !checked_mul_u64(output_values, 2u, &output_bytes) || pixel_bytes > (uint64_t)SIZE_MAX ||
        weight_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX || pixel_bytes > max_buffer_length ||
        weight_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error, error_size, "Metal SAM patch-embed byte-size overflow");
    }

    uint16_t zero_bias[UOCR_SAM_HIDDEN_SIZE];
    const uint16_t *bias_source = patch_bias_f16_or_null;
    if (bias_source == NULL) {
        memset(zero_bias, 0, sizeof(zero_bias));
        bias_source = zero_bias;
    }

    @autoreleasepool {
        id<MTLBuffer> pixel_buffer = [ctx->device newBufferWithBytes:pixels
                                                              length:(NSUInteger)pixel_bytes
                                                             options:MTLResourceStorageModeShared];
        if (pixel_buffer == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal SAM patch-embed pixel buffer");
        }
        pixel_buffer.label = @"uocr_sam_patch_pixels";

        id<MTLBuffer> weight_buffer = [ctx->device newBufferWithBytes:patch_weight_f16
                                                               length:(NSUInteger)weight_bytes
                                                              options:MTLResourceStorageModeShared];
        if (weight_buffer == nil) {
            [pixel_buffer release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM patch-embed weight buffer");
        }
        weight_buffer.label = @"uocr_sam_patch_weight_f16";

        id<MTLBuffer> bias_buffer = [ctx->device newBufferWithBytes:bias_source
                                                             length:(NSUInteger)UOCR_SAM_HIDDEN_SIZE * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
        if (bias_buffer == nil) {
            [weight_buffer release];
            [pixel_buffer release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM patch-embed bias buffer");
        }
        bias_buffer.label = @"uocr_sam_patch_bias_f16";

        id<MTLBuffer> output_buffer = [ctx->device newBufferWithLength:(NSUInteger)output_bytes
                                                                options:MTLResourceStorageModeShared];
        if (output_buffer == nil) {
            [bias_buffer release];
            [weight_buffer release];
            [pixel_buffer release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM patch-embed output buffer");
        }
        output_buffer.label = @"uocr_sam_patch_output_bhwc_f16";

        const char *function_name = pixel_format == UOCR_PIXEL_F16_NCHW ?
                                        "uocr_sam_patch_embed_f16_input" :
                                        "uocr_sam_patch_embed_f32_input";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            [output_buffer release];
            [bias_buffer release];
            [weight_buffer release];
            [pixel_buffer release];
            return 0;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [output_buffer release];
            [bias_buffer release];
            [weight_buffer release];
            [pixel_buffer release];
            return metal_fail(error, error_size, "failed to create Metal SAM patch-embed command buffer");
        }
        cb.label = @"uocr_sam_patch_embed_command_buffer";
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [output_buffer release];
            [bias_buffer release];
            [weight_buffer release];
            [pixel_buffer release];
            return metal_fail(error, error_size, "failed to create Metal SAM patch-embed encoder");
        }

        uocr_metal_sam_patch_embed_params params;
        memset(&params, 0, sizeof(params));
        params.width = width;
        params.height = height;
        params.out_width = grid_w;
        params.out_height = grid_h;
        params.has_bias = patch_bias_f16_or_null != NULL ? 1u : 0u;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:pixel_buffer offset:0u atIndex:0u];
        [enc setBuffer:weight_buffer offset:0u atIndex:1u];
        [enc setBuffer:bias_buffer offset:0u atIndex:2u];
        [enc setBuffer:output_buffer offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];

        NSUInteger threads_x = pipeline.threadExecutionWidth;
        if (threads_x == 0u || threads_x > (NSUInteger)UOCR_SAM_HIDDEN_SIZE) {
            threads_x = 32u;
        }
        if (threads_x > pipeline.maxTotalThreadsPerThreadgroup) {
            threads_x = pipeline.maxTotalThreadsPerThreadgroup;
        }
        if (threads_x == 0u) {
            threads_x = 1u;
        }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)UOCR_SAM_HIDDEN_SIZE, (NSUInteger)output_patches, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_x, 1u, 1u)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        int ok = 1;
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            ok = metal_fail(error, error_size, "Metal SAM patch-embed command failed: %s", [description UTF8String]);
        } else {
            memcpy(out_bhwc_f16, [output_buffer contents], (size_t)output_bytes);
            if (out_grid_w != NULL) {
                *out_grid_w = grid_w;
            }
            if (out_grid_h != NULL) {
                *out_grid_h = grid_h;
            }
            metal_clear_error(error, error_size);
        }

        [output_buffer release];
        [bias_buffer release];
        [weight_buffer release];
        [pixel_buffer release];
        return ok;
    }
}

int uocr_metal_context_sam_add_abs_pos_f16(uocr_metal_context *ctx,
                                           const uint16_t *patch_bhwc_f16,
                                           const uint16_t *pos_embed_f16,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uint16_t *out_bhwc_f16,
                                           char *error,
                                           size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || patch_bhwc_f16 == NULL || pos_embed_f16 == NULL || out_bhwc_f16 == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM absolute-position request");
    }

    const uint32_t source_grid = UOCR_GLOBAL_VIEW_SIZE / UOCR_VISION_PATCH_SIZE;
    if (grid_w == 0u || grid_h == 0u || grid_w != grid_h || grid_w > source_grid || grid_h > source_grid) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM absolute-position grid must be square and in [1,%u], got %ux%u",
                          source_grid,
                          grid_w,
                          grid_h);
    }

    uint64_t target_patches = 0u;
    uint64_t target_values = 0u;
    uint64_t target_bytes = 0u;
    uint64_t source_patches = 0u;
    uint64_t source_values = 0u;
    uint64_t source_bytes = 0u;
    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &target_patches) ||
        !checked_mul_u64(target_patches, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &target_values) ||
        !checked_mul_u64(target_values, 2u, &target_bytes) ||
        !checked_mul_u64((uint64_t)source_grid, (uint64_t)source_grid, &source_patches) ||
        !checked_mul_u64(source_patches, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &source_values) ||
        !checked_mul_u64(source_values, 2u, &source_bytes) || target_bytes > (uint64_t)SIZE_MAX ||
        source_bytes > (uint64_t)SIZE_MAX || target_bytes > max_buffer_length || source_bytes > max_buffer_length) {
        return metal_fail(error, error_size, "Metal SAM absolute-position byte-size overflow");
    }

    @autoreleasepool {
        id<MTLBuffer> patch_buffer = [ctx->device newBufferWithBytes:patch_bhwc_f16
                                                             length:(NSUInteger)target_bytes
                                                            options:MTLResourceStorageModeShared];
        if (patch_buffer == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal SAM absolute-position patch buffer");
        }
        patch_buffer.label = @"uocr_sam_abs_pos_patch_bhwc_f16";

        id<MTLBuffer> pos_buffer = [ctx->device newBufferWithBytes:pos_embed_f16
                                                           length:(NSUInteger)source_bytes
                                                          options:MTLResourceStorageModeShared];
        if (pos_buffer == nil) {
            [patch_buffer release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM absolute-position table buffer");
        }
        pos_buffer.label = @"uocr_sam_abs_pos_table_f16";

        id<MTLBuffer> output_buffer = [ctx->device newBufferWithLength:(NSUInteger)target_bytes
                                                                options:MTLResourceStorageModeShared];
        if (output_buffer == nil) {
            [pos_buffer release];
            [patch_buffer release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM absolute-position output buffer");
        }
        output_buffer.label = @"uocr_sam_abs_pos_output_bhwc_f16";

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_sam_add_abs_pos_f16", error, error_size);
        if (pipeline == nil) {
            [output_buffer release];
            [pos_buffer release];
            [patch_buffer release];
            return 0;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [output_buffer release];
            [pos_buffer release];
            [patch_buffer release];
            return metal_fail(error, error_size, "failed to create Metal SAM absolute-position command buffer");
        }
        cb.label = @"uocr_sam_abs_pos_command_buffer";
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [output_buffer release];
            [pos_buffer release];
            [patch_buffer release];
            return metal_fail(error, error_size, "failed to create Metal SAM absolute-position encoder");
        }

        uocr_metal_sam_abs_pos_params params;
        params.source_grid = source_grid;
        params.target_width = grid_w;
        params.target_height = grid_h;
        params.channels = UOCR_SAM_HIDDEN_SIZE;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:patch_buffer offset:0u atIndex:0u];
        [enc setBuffer:pos_buffer offset:0u atIndex:1u];
        [enc setBuffer:output_buffer offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];

        NSUInteger threads = pipeline.threadExecutionWidth;
        if (threads == 0u) {
            threads = 64u;
        }
        if (threads > pipeline.maxTotalThreadsPerThreadgroup) {
            threads = pipeline.maxTotalThreadsPerThreadgroup;
        }
        if (threads == 0u) {
            threads = 1u;
        }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)target_values, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        int ok = 1;
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            ok = metal_fail(error, error_size, "Metal SAM absolute-position command failed: %s", [description UTF8String]);
        } else {
            memcpy(out_bhwc_f16, [output_buffer contents], (size_t)target_bytes);
            metal_clear_error(error, error_size);
        }

        [output_buffer release];
        [pos_buffer release];
        [patch_buffer release];
        return ok;
    }
}

static int metal_prompt_arena_slot_offset(const uocr_metal_context *ctx,
                                          uint32_t slot,
                                          uint32_t n_tokens,
                                          uint64_t *out_offset,
                                          uint64_t *out_bytes) {
    if (ctx == NULL || out_offset == NULL || out_bytes == NULL || !ctx->has_kv_cache_layout ||
        ctx->runtime_arenas[UOCR_METAL_ARENA_PROMPT_EMBEDDINGS].buffer == nil || n_tokens == 0u ||
        slot >= ctx->kv_cache_layout.batch_slots || n_tokens > ctx->kv_cache_layout.prompt_token_capacity) {
        return 0;
    }
    uint64_t slot_values = 0u;
    uint64_t slot_bytes = 0u;
    uint64_t offset = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    if (!checked_mul_u64((uint64_t)ctx->kv_cache_layout.prompt_token_capacity, (uint64_t)UOCR_HIDDEN_SIZE, &slot_values) ||
        !checked_mul_u64(slot_values, 2u, &slot_bytes) ||
        !checked_mul_u64((uint64_t)slot, slot_bytes, &offset) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &output_values) ||
        !checked_mul_u64(output_values, 2u, &output_bytes)) {
        return 0;
    }
    uint64_t end = 0u;
    const uint64_t arena_length = (uint64_t)[ctx->runtime_arenas[UOCR_METAL_ARENA_PROMPT_EMBEDDINGS].buffer length];
    if (!checked_add_u64(offset, output_bytes, &end) || end > arena_length) {
        return 0;
    }
    *out_offset = offset;
    *out_bytes = output_bytes;
    return 1;
}

int uocr_metal_context_assemble_prompt_from_model_to_arena_f16(uocr_metal_context *ctx,
                                                               const int32_t *input_ids,
                                                               uint32_t n_tokens,
                                                               uint32_t image_span_start,
                                                               uint32_t image_span_length,
                                                               const uint16_t *image_features_f16,
                                                               uint32_t slot,
                                                               char *error,
                                                               size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_ids == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal prompt arena assembly request");
    }

    uint64_t dst_offset = 0u;
    uint64_t output_bytes = 0u;
    if (!metal_prompt_arena_slot_offset(ctx, slot, n_tokens, &dst_offset, &output_bytes)) {
        return metal_fail(error,
                          error_size,
                          "Metal prompt arena assembly requires allocated arenas and n_tokens within prompt capacity");
    }
    if (dst_offset > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal prompt arena assembly offset exceeds platform size");
    }
    (void)output_bytes;

    uint64_t table_values = 0u;
    uint64_t table_bytes = 0u;
    if (!checked_mul_u64((uint64_t)UOCR_VOCAB_SIZE, (uint64_t)UOCR_HIDDEN_SIZE, &table_values) ||
        !checked_mul_u64(table_values, 2u, &table_bytes)) {
        return metal_fail(error, error_size, "Metal prompt arena assembly table byte-size overflow");
    }

    id<MTLBuffer> table = nil;
    NSUInteger table_offset = 0u;
    if (!metal_get_mapped_tensor_buffer(ctx,
                                        UOCR_TENSOR_ID_TOK_EMBED,
                                        table_bytes,
                                        &table,
                                        &table_offset,
                                        error,
                                        error_size)) {
        return 0;
    }

    return metal_context_assemble_prompt_f16_with_table_buffer_to_buffer(ctx,
                                                                         table,
                                                                         table_offset,
                                                                         UOCR_VOCAB_SIZE,
                                                                         UOCR_HIDDEN_SIZE,
                                                                         input_ids,
                                                                         n_tokens,
                                                                         image_span_start,
                                                                         image_span_length,
                                                                         image_features_f16,
                                                                         ctx->runtime_arenas[UOCR_METAL_ARENA_PROMPT_EMBEDDINGS].buffer,
                                                                         (NSUInteger)dst_offset,
                                                                         "Metal prompt arena assembly",
                                                                         error,
                                                                         error_size);
}

int uocr_metal_context_read_prompt_arena_f16(uocr_metal_context *ctx,
                                             uint32_t slot,
                                             uint32_t n_tokens,
                                             uint16_t *out_prompt_f16,
                                             char *error,
                                             size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || out_prompt_f16 == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal prompt arena readback request");
    }

    uint64_t src_offset = 0u;
    uint64_t readback_bytes = 0u;
    if (!metal_prompt_arena_slot_offset(ctx, slot, n_tokens, &src_offset, &readback_bytes) ||
        readback_bytes > (uint64_t)SIZE_MAX || src_offset > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal prompt arena readback requires allocated arenas and n_tokens within prompt capacity");
    }

    @autoreleasepool {
        id<MTLBuffer> readback = [ctx->device newBufferWithLength:(NSUInteger)readback_bytes
                                                          options:MTLResourceStorageModeShared];
        if (readback == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal prompt arena readback buffer");
        }
        readback.label = @"uocr_prompt_arena_readback";

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [readback release];
            return metal_fail(error, error_size, "failed to create Metal prompt arena readback command buffer");
        }
        cb.label = @"uocr_prompt_arena_readback_command_buffer";

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (blit == nil) {
            [readback release];
            return metal_fail(error, error_size, "failed to create Metal prompt arena readback blit encoder");
        }
        [blit copyFromBuffer:ctx->runtime_arenas[UOCR_METAL_ARENA_PROMPT_EMBEDDINGS].buffer
                sourceOffset:(NSUInteger)src_offset
                    toBuffer:readback
           destinationOffset:0u
                        size:(NSUInteger)readback_bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [readback release];
            return metal_fail(error, error_size, "Metal prompt arena readback failed: %s", [description UTF8String]);
        }

        memcpy(out_prompt_f16, [readback contents], (size_t)readback_bytes);
        [readback release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_read_decoder_final_hidden_f16(uocr_metal_context *ctx,
                                                     uint32_t slot,
                                                     uint32_t n_tokens,
                                                     uint16_t *out_hidden_f16,
                                                     char *error,
                                                     size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || out_hidden_f16 == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal decoder final-hidden readback request");
    }
    if (!ctx->has_integrated_prefill || ctx->integrated_prefill_slot != slot ||
        ctx->integrated_prefill_tokens != n_tokens || ctx->integrated_prefill_final_segment >= UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS) {
        return metal_fail(error,
                          error_size,
                          "Metal decoder final-hidden readback requires a completed integrated prefill for the requested slot/tokens");
    }
    uocr_metal_buffer_slice slice;
    uint64_t bytes = 0u;
    if (!metal_hidden_arena_segment_slice(ctx, slot, ctx->integrated_prefill_final_segment, n_tokens, &slice, &bytes)) {
        return metal_fail(error, error_size, "Metal decoder final-hidden arena slice is invalid");
    }
    return metal_read_slice_to_host(ctx, slice, bytes, out_hidden_f16, "Metal decoder final-hidden", error, error_size);
}

static int metal_context_rmsnorm_f16_with_weight_buffer(uocr_metal_context *ctx,
                                                         const uint16_t *input_f16,
                                                         id<MTLBuffer> weight_buffer,
                                                         NSUInteger weight_offset,
                                                         uint32_t n_rows,
                                                         uint32_t hidden_size,
                                                         float eps,
                                                         uocr_metal_rmsnorm_output_type output_type,
                                                         void *out,
                                                         const char *op_name,
                                                         char *error,
                                                         size_t error_size) {
    const char *op = op_name != NULL ? op_name : "Metal RMSNorm";
    if (ctx == NULL || input_f16 == NULL || weight_buffer == nil || out == NULL || n_rows == 0u || hidden_size == 0u) {
        return metal_fail(error, error_size, "invalid %s request", op);
    }
    if (!(eps > 0.0f)) {
        return metal_fail(error, error_size, "%s eps must be positive", op);
    }
    if (output_type != UOCR_METAL_RMSNORM_OUTPUT_F16 && output_type != UOCR_METAL_RMSNORM_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported %s output type %d", op, (int)output_type);
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_RMSNORM_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)hidden_size, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)hidden_size, 2u, &weight_bytes) ||
        !checked_mul_u64(input_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "%s byte-size overflow", op);
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "%s buffers exceed maxBufferLength %llu",
                          op,
                          (unsigned long long)max_buffer_length);
    }
    const uint64_t weight_length = (uint64_t)[weight_buffer length];
    if ((uint64_t)weight_offset > weight_length || weight_bytes > weight_length - (uint64_t)weight_offset) {
        return metal_fail(error,
                          error_size,
                          "%s weight buffer is too small: offset=%llu need=%llu length=%llu",
                          op,
                          (unsigned long long)weight_offset,
                          (unsigned long long)weight_bytes,
                          (unsigned long long)weight_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_RMSNORM_OUTPUT_F16 ?
                                        "uocr_rmsnorm_f16_to_f16" :
                                        "uocr_rmsnorm_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate %s input buffer", op);
        }
        src.label = @"uocr_rmsnorm_input_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate %s output buffer", op);
        }
        dst.label = @"uocr_rmsnorm_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "failed to create %s command buffer", op);
        }
        cb.label = @"uocr_rmsnorm_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "failed to create %s command encoder", op);
        }

        uocr_metal_rmsnorm_params params;
        params.n_rows = n_rows;
        params.hidden_size = hidden_size;
        params.eps = eps;
        params.reserved = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "%s threadgroup memory exceeds device limit", op);
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight_buffer offset:weight_offset atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [src release];
            return metal_fail(error, error_size, "%s command failed: %s", op, [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_rmsnorm_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   const uint16_t *weight_f16,
                                   uint32_t n_rows,
                                   uint32_t hidden_size,
                                   float eps,
                                   uocr_metal_rmsnorm_output_type output_type,
                                   void *out,
                                   char *error,
                                   size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_f16 == NULL || out == NULL || n_rows == 0u || hidden_size == 0u) {
        return metal_fail(error, error_size, "invalid Metal RMSNorm request");
    }

    uint64_t weight_bytes = 0u;
    if (!checked_mul_u64((uint64_t)hidden_size, 2u, &weight_bytes) || weight_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal RMSNorm byte-size overflow");
    }
    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (weight_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal RMSNorm buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:weight_f16
                                                        length:(NSUInteger)weight_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal RMSNorm weight buffer");
        }
        weight.label = @"uocr_rmsnorm_weight_f16";
        const int ok = metal_context_rmsnorm_f16_with_weight_buffer(ctx,
                                                                    input_f16,
                                                                    weight,
                                                                    0u,
                                                                    n_rows,
                                                                    hidden_size,
                                                                    eps,
                                                                    output_type,
                                                                    out,
                                                                    "Metal RMSNorm",
                                                                    error,
                                                                    error_size);
        [weight release];
        return ok;
    }
}

static int metal_context_layernorm_f16_with_parameter_buffers(uocr_metal_context *ctx,
                                                               const uint16_t *input_f16,
                                                               id<MTLBuffer> weight_buffer,
                                                               NSUInteger weight_offset,
                                                               id<MTLBuffer> bias_buffer,
                                                               NSUInteger bias_offset,
                                                               uint32_t n_rows,
                                                               uint32_t hidden_size,
                                                               float eps,
                                                               uocr_metal_layernorm_output_type output_type,
                                                               void *out,
                                                               const char *op_name,
                                                               char *error,
                                                               size_t error_size) {
    const char *op = op_name != NULL ? op_name : "Metal LayerNorm";
    if (ctx == NULL || input_f16 == NULL || weight_buffer == nil || bias_buffer == nil || out == NULL ||
        n_rows == 0u || hidden_size == 0u) {
        return metal_fail(error, error_size, "invalid %s request", op);
    }
    if (!(eps > 0.0f)) {
        return metal_fail(error, error_size, "%s eps must be positive", op);
    }
    if (output_type != UOCR_METAL_LAYERNORM_OUTPUT_F16 && output_type != UOCR_METAL_LAYERNORM_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported %s output type %d", op, (int)output_type);
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t parameter_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_LAYERNORM_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)hidden_size, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)hidden_size, 2u, &parameter_bytes) ||
        !checked_mul_u64(input_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "%s byte-size overflow", op);
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "%s buffers exceed maxBufferLength %llu",
                          op,
                          (unsigned long long)max_buffer_length);
    }
    const uint64_t weight_length = (uint64_t)[weight_buffer length];
    const uint64_t bias_length = (uint64_t)[bias_buffer length];
    if ((uint64_t)weight_offset > weight_length || parameter_bytes > weight_length - (uint64_t)weight_offset) {
        return metal_fail(error,
                          error_size,
                          "%s weight buffer is too small: offset=%llu need=%llu length=%llu",
                          op,
                          (unsigned long long)weight_offset,
                          (unsigned long long)parameter_bytes,
                          (unsigned long long)weight_length);
    }
    if ((uint64_t)bias_offset > bias_length || parameter_bytes > bias_length - (uint64_t)bias_offset) {
        return metal_fail(error,
                          error_size,
                          "%s bias buffer is too small: offset=%llu need=%llu length=%llu",
                          op,
                          (unsigned long long)bias_offset,
                          (unsigned long long)parameter_bytes,
                          (unsigned long long)bias_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_LAYERNORM_OUTPUT_F16 ?
                                        "uocr_layernorm_f16_to_f16" :
                                        "uocr_layernorm_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate %s input buffer", op);
        }
        src.label = @"uocr_layernorm_input_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate %s output buffer", op);
        }
        dst.label = @"uocr_layernorm_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "failed to create %s command buffer", op);
        }
        cb.label = @"uocr_layernorm_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "failed to create %s command encoder", op);
        }

        uocr_metal_rmsnorm_params params;
        params.n_rows = n_rows;
        params.hidden_size = hidden_size;
        params.eps = eps;
        params.reserved = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "%s threadgroup memory exceeds device limit", op);
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight_buffer offset:weight_offset atIndex:1u];
        [enc setBuffer:bias_buffer offset:bias_offset atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [src release];
            return metal_fail(error, error_size, "%s command failed: %s", op, [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_sam_layernorm_f16(uocr_metal_context *ctx,
                                         const uint16_t *input_f16,
                                         const uint16_t *weight_f16,
                                         const uint16_t *bias_f16,
                                         uint32_t n_rows,
                                         uocr_metal_layernorm_output_type output_type,
                                         void *out,
                                         char *error,
                                         size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_f16 == NULL || bias_f16 == NULL || out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM LayerNorm request");
    }

    const uint64_t parameter_bytes = (uint64_t)UOCR_SAM_HIDDEN_SIZE * 2u;
    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (parameter_bytes > (uint64_t)SIZE_MAX || parameter_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM LayerNorm buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:weight_f16
                                                        length:(NSUInteger)parameter_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal SAM LayerNorm weight buffer");
        }
        weight.label = @"uocr_sam_layernorm_weight_f16";

        id<MTLBuffer> bias = [ctx->device newBufferWithBytes:bias_f16
                                                      length:(NSUInteger)parameter_bytes
                                                     options:MTLResourceStorageModeShared];
        if (bias == nil) {
            [weight release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM LayerNorm bias buffer");
        }
        bias.label = @"uocr_sam_layernorm_bias_f16";

        const int ok = metal_context_layernorm_f16_with_parameter_buffers(ctx,
                                                                         input_f16,
                                                                         weight,
                                                                         0u,
                                                                         bias,
                                                                         0u,
                                                                         n_rows,
                                                                         UOCR_SAM_HIDDEN_SIZE,
                                                                         1.0e-6f,
                                                                         output_type,
                                                                         out,
                                                                         "Metal SAM LayerNorm",
                                                                         error,
                                                                         error_size);
        [bias release];
        [weight release];
        return ok;
    }
}

int uocr_metal_context_sam_qkv_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   const uint16_t *qkv_weight_f16,
                                   const uint16_t *qkv_bias_f16,
                                   uint32_t n_rows,
                                   uocr_metal_dense_output_type output_type,
                                   void *q_out,
                                   void *k_out,
                                   void *v_out,
                                   char *error,
                                   size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || qkv_weight_f16 == NULL || qkv_bias_f16 == NULL ||
        q_out == NULL || k_out == NULL || v_out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM QKV request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM QKV output type %d", (int)output_type);
    }
    if (UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE ||
        UOCR_SAM_QKV_SIZE != 3u * UOCR_SAM_HIDDEN_SIZE) {
        return metal_fail(error, error_size, "Metal SAM QKV constants are inconsistent");
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t bias_bytes = 0u;
    uint64_t output_values_per_projection = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_QKV_SIZE, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_QKV_SIZE, 2u, &bias_bytes) ||
        !checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &output_values_per_projection) ||
        !checked_mul_u64(output_values_per_projection, 3u, &output_values) ||
        !checked_mul_u64(output_values_per_projection, output_element_bytes, &output_bytes) ||
        input_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX ||
        bias_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal SAM QKV byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || bias_bytes > max_buffer_length ||
        output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM QKV buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_qkv_f16_to_f16" :
                                        "uocr_sam_qkv_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal SAM QKV input buffer");
        }
        src.label = @"uocr_sam_qkv_input_f16";

        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:qkv_weight_f16
                                                        length:(NSUInteger)weight_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM QKV weight buffer");
        }
        weight.label = @"uocr_sam_qkv_weight_f16";

        id<MTLBuffer> bias = [ctx->device newBufferWithBytes:qkv_bias_f16
                                                      length:(NSUInteger)bias_bytes
                                                     options:MTLResourceStorageModeShared];
        if (bias == nil) {
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM QKV bias buffer");
        }
        bias.label = @"uocr_sam_qkv_bias_f16";

        id<MTLBuffer> q_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> v_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (q_dst == nil || k_dst == nil || v_dst == nil) {
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM QKV output buffers");
        }
        q_dst.label = @"uocr_sam_q_f16";
        k_dst.label = @"uocr_sam_k_f16";
        v_dst.label = @"uocr_sam_v_f16";
        memset([q_dst contents], 0, (size_t)output_bytes);
        memset([k_dst contents], 0, (size_t)output_bytes);
        memset([v_dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal SAM QKV command buffer");
        }
        cb.label = @"uocr_sam_qkv_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal SAM QKV command encoder");
        }

        uocr_metal_dense_params params;
        params.input_rows = n_rows;
        params.in_features = UOCR_SAM_HIDDEN_SIZE;
        params.out_features = UOCR_SAM_HIDDEN_SIZE;
        params.has_bias = 1u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "Metal SAM QKV threadgroup memory exceeds device limit");
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:bias offset:0u atIndex:2u];
        [enc setBuffer:q_dst offset:0u atIndex:3u];
        [enc setBuffer:k_dst offset:0u atIndex:4u];
        [enc setBuffer:v_dst offset:0u atIndex:5u];
        [enc setBytes:&params length:sizeof(params) atIndex:6u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "Metal SAM QKV command failed: %s", [description UTF8String]);
        }

        memcpy(q_out, [q_dst contents], (size_t)output_bytes);
        memcpy(k_out, [k_dst contents], (size_t)output_bytes);
        memcpy(v_out, [v_dst contents], (size_t)output_bytes);
        [v_dst release];
        [k_dst release];
        [q_dst release];
        [bias release];
        [weight release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

static int sam_window_partition_geometry(uint32_t grid_w,
                                         uint32_t grid_h,
                                         uint32_t *out_padded_w,
                                         uint32_t *out_padded_h,
                                         uint32_t *out_windows_per_row,
                                         uint32_t *out_windows_per_col,
                                         uint32_t *out_n_windows) {
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE ||
        out_padded_w == NULL || out_padded_h == NULL || out_windows_per_row == NULL ||
        out_windows_per_col == NULL || out_n_windows == NULL) {
        return 0;
    }
    const uint32_t window_size = UOCR_SAM_WINDOW_SIZE;
    const uint32_t pad_w = (window_size - (grid_w % window_size)) % window_size;
    const uint32_t pad_h = (window_size - (grid_h % window_size)) % window_size;
    const uint32_t padded_w = grid_w + pad_w;
    const uint32_t padded_h = grid_h + pad_h;
    const uint32_t windows_per_row = padded_w / window_size;
    const uint32_t windows_per_col = padded_h / window_size;
    uint64_t n_windows = 0u;
    if (!checked_mul_u64((uint64_t)windows_per_row, (uint64_t)windows_per_col, &n_windows) ||
        n_windows == 0u || n_windows > (uint64_t)UINT32_MAX) {
        return 0;
    }
    *out_padded_w = padded_w;
    *out_padded_h = padded_h;
    *out_windows_per_row = windows_per_row;
    *out_windows_per_col = windows_per_col;
    *out_n_windows = (uint32_t)n_windows;
    return 1;
}

int uocr_metal_context_sam_window_partition_f16(uocr_metal_context *ctx,
                                                const uint16_t *input_bhwc_f16,
                                                uint32_t grid_w,
                                                uint32_t grid_h,
                                                uint16_t *out_windows_f16,
                                                uint32_t *out_n_windows,
                                                uint32_t *out_padded_w,
                                                uint32_t *out_padded_h,
                                                char *error,
                                                size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_bhwc_f16 == NULL || out_windows_f16 == NULL || out_n_windows == NULL ||
        out_padded_w == NULL || out_padded_h == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM window partition request");
    }
    if (UOCR_SAM_WINDOW_TOKENS != UOCR_SAM_WINDOW_SIZE * UOCR_SAM_WINDOW_SIZE ||
        UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE) {
        return metal_fail(error, error_size, "Metal SAM window partition constants are inconsistent");
    }

    uint32_t padded_w = 0u;
    uint32_t padded_h = 0u;
    uint32_t windows_per_row = 0u;
    uint32_t windows_per_col = 0u;
    uint32_t n_windows = 0u;
    if (!sam_window_partition_geometry(grid_w,
                                       grid_h,
                                       &padded_w,
                                       &padded_h,
                                       &windows_per_row,
                                       &windows_per_col,
                                       &n_windows)) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal SAM window partition grid %ux%u",
                          grid_w,
                          grid_h);
    }

    uint64_t input_values = 0u;
    uint64_t window_tokens = 0u;
    uint64_t window_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t window_bytes = 0u;
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &input_values) ||
        !checked_mul_u64(input_values, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64((uint64_t)n_windows, (uint64_t)UOCR_SAM_WINDOW_TOKENS, &window_tokens) ||
        !checked_mul_u64(window_tokens, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &window_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) || !checked_mul_u64(window_values, 2u, &window_bytes) ||
        input_bytes > (uint64_t)SIZE_MAX || window_bytes > (uint64_t)SIZE_MAX || window_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal SAM window partition byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || window_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM window partition buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx,
                                                                  "uocr_sam_window_partition_f16",
                                                                  error,
                                                                  error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:input_bhwc_f16
                                       length:(NSUInteger)input_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM window partition input buffer");
            goto cleanup_window_partition;
        }
        src.label = @"uocr_sam_window_partition_input_bhwc_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)window_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM window partition output buffer");
            goto cleanup_window_partition;
        }
        dst.label = @"uocr_sam_window_partition_output_f16";
        memset([dst contents], 0, (size_t)window_bytes);

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM window partition command buffer");
            goto cleanup_window_partition;
        }
        cb.label = @"uocr_sam_window_partition_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM window partition command encoder");
            goto cleanup_window_partition;
        }

        uocr_metal_sam_window_partition_params params;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.padded_width = padded_w;
        params.padded_height = padded_h;
        params.windows_per_row = windows_per_row;
        params.windows_per_col = windows_per_col;
        params.window_size = UOCR_SAM_WINDOW_SIZE;
        params.hidden_size = UOCR_SAM_HIDDEN_SIZE;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:dst offset:0u atIndex:1u];
        [enc setBytes:&params length:sizeof(params) atIndex:2u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)window_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal SAM window partition command failed: %s", [description UTF8String]);
            goto cleanup_window_partition;
        }

        memcpy(out_windows_f16, [dst contents], (size_t)window_bytes);
        *out_n_windows = n_windows;
        *out_padded_w = padded_w;
        *out_padded_h = padded_h;
        result = 1;

    cleanup_window_partition:
        [dst release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_window_unpartition_f16(uocr_metal_context *ctx,
                                                  const uint16_t *windows_f16,
                                                  uint32_t grid_w,
                                                  uint32_t grid_h,
                                                  uint16_t *out_bhwc_f16,
                                                  char *error,
                                                  size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || windows_f16 == NULL || out_bhwc_f16 == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM window unpartition request");
    }
    if (UOCR_SAM_WINDOW_TOKENS != UOCR_SAM_WINDOW_SIZE * UOCR_SAM_WINDOW_SIZE ||
        UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE) {
        return metal_fail(error, error_size, "Metal SAM window unpartition constants are inconsistent");
    }

    uint32_t padded_w = 0u;
    uint32_t padded_h = 0u;
    uint32_t windows_per_row = 0u;
    uint32_t windows_per_col = 0u;
    uint32_t n_windows = 0u;
    if (!sam_window_partition_geometry(grid_w,
                                       grid_h,
                                       &padded_w,
                                       &padded_h,
                                       &windows_per_row,
                                       &windows_per_col,
                                       &n_windows)) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal SAM window unpartition grid %ux%u",
                          grid_w,
                          grid_h);
    }

    uint64_t output_values = 0u;
    uint64_t window_tokens = 0u;
    uint64_t window_values = 0u;
    uint64_t output_bytes = 0u;
    uint64_t window_bytes = 0u;
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &output_values) ||
        !checked_mul_u64(output_values, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &output_values) ||
        !checked_mul_u64((uint64_t)n_windows, (uint64_t)UOCR_SAM_WINDOW_TOKENS, &window_tokens) ||
        !checked_mul_u64(window_tokens, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &window_values) ||
        !checked_mul_u64(output_values, 2u, &output_bytes) || !checked_mul_u64(window_values, 2u, &window_bytes) ||
        output_bytes > (uint64_t)SIZE_MAX || window_bytes > (uint64_t)SIZE_MAX || output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal SAM window unpartition byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (output_bytes > max_buffer_length || window_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM window unpartition buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx,
                                                                  "uocr_sam_window_unpartition_f16",
                                                                  error,
                                                                  error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:windows_f16
                                       length:(NSUInteger)window_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM window unpartition input buffer");
            goto cleanup_window_unpartition;
        }
        src.label = @"uocr_sam_window_unpartition_input_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM window unpartition output buffer");
            goto cleanup_window_unpartition;
        }
        dst.label = @"uocr_sam_window_unpartition_output_bhwc_f16";
        memset([dst contents], 0, (size_t)output_bytes);

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM window unpartition command buffer");
            goto cleanup_window_unpartition;
        }
        cb.label = @"uocr_sam_window_unpartition_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM window unpartition command encoder");
            goto cleanup_window_unpartition;
        }

        uocr_metal_sam_window_partition_params params;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.padded_width = padded_w;
        params.padded_height = padded_h;
        params.windows_per_row = windows_per_row;
        params.windows_per_col = windows_per_col;
        params.window_size = UOCR_SAM_WINDOW_SIZE;
        params.hidden_size = UOCR_SAM_HIDDEN_SIZE;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:dst offset:0u atIndex:1u];
        [enc setBytes:&params length:sizeof(params) atIndex:2u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error,
                                error_size,
                                "Metal SAM window unpartition command failed: %s",
                                [description UTF8String]);
            goto cleanup_window_unpartition;
        }

        memcpy(out_bhwc_f16, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_window_unpartition:
        [dst release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_neck_conv1x1_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_bhwc_f16,
                                            const uint16_t *weight_f16,
                                            uint32_t grid_w,
                                            uint32_t grid_h,
                                            uocr_metal_dense_output_type output_type,
                                            void *out_nchw,
                                            char *error,
                                            size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_bhwc_f16 == NULL || weight_f16 == NULL || out_nchw == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM neck 1x1 request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM neck 1x1 output type %d", (int)output_type);
    }
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE) {
        return metal_fail(error, error_size, "invalid Metal SAM neck 1x1 grid %ux%u", grid_w, grid_h);
    }
    if (UOCR_SAM_NECK_CHANNELS != 256u || UOCR_SAM_HIDDEN_SIZE != 768u) {
        return metal_fail(error, error_size, "Metal SAM neck 1x1 constants are inconsistent");
    }

    uint64_t spatial = 0u;
    uint64_t input_values = 0u;
    uint64_t weight_values = 0u;
    uint64_t output_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &spatial) ||
        !checked_mul_u64(spatial, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_NECK_CHANNELS, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(spatial, (uint64_t)UOCR_SAM_NECK_CHANNELS, &output_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) || !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes) || input_bytes > (uint64_t)SIZE_MAX ||
        weight_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX || output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal SAM neck 1x1 byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM neck 1x1 buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_neck_conv1x1_f16_to_f16" :
                                        "uocr_sam_neck_conv1x1_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:input_bhwc_f16
                                       length:(NSUInteger)input_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM neck 1x1 input buffer");
            goto cleanup_sam_neck_conv1x1;
        }
        src.label = @"uocr_sam_neck_conv1x1_input_bhwc_f16";

        weight = [ctx->device newBufferWithBytes:weight_f16
                                          length:(NSUInteger)weight_bytes
                                         options:MTLResourceStorageModeShared];
        if (weight == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM neck 1x1 weight buffer");
            goto cleanup_sam_neck_conv1x1;
        }
        weight.label = @"uocr_sam_neck_conv1x1_weight_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM neck 1x1 output buffer");
            goto cleanup_sam_neck_conv1x1;
        }
        dst.label = @"uocr_sam_neck_conv1x1_output_nchw";
        memset([dst contents], 0, (size_t)output_bytes);

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            result = metal_fail(error, error_size, "Metal SAM neck 1x1 threadgroup memory exceeds device limit");
            goto cleanup_sam_neck_conv1x1;
        }

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM neck 1x1 command buffer");
            goto cleanup_sam_neck_conv1x1;
        }
        cb.label = @"uocr_sam_neck_conv1x1_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM neck 1x1 command encoder");
            goto cleanup_sam_neck_conv1x1;
        }

        uocr_metal_sam_neck_conv1x1_params params;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.in_channels = UOCR_SAM_HIDDEN_SIZE;
        params.out_channels = UOCR_SAM_NECK_CHANNELS;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal SAM neck 1x1 command failed: %s", [description UTF8String]);
            goto cleanup_sam_neck_conv1x1;
        }

        memcpy(out_nchw, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_sam_neck_conv1x1:
        [dst release];
        [weight release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_neck_conv3x3_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_nchw_f16,
                                            const uint16_t *weight_f16,
                                            uint32_t grid_w,
                                            uint32_t grid_h,
                                            uocr_metal_dense_output_type output_type,
                                            void *out_nchw,
                                            char *error,
                                            size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_nchw_f16 == NULL || weight_f16 == NULL || out_nchw == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM neck 3x3 request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM neck 3x3 output type %d", (int)output_type);
    }
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE) {
        return metal_fail(error, error_size, "invalid Metal SAM neck 3x3 grid %ux%u", grid_w, grid_h);
    }
    if (UOCR_SAM_NECK_CHANNELS != 256u || UOCR_SAM_NECK_KERNEL_SIZE != 3u) {
        return metal_fail(error, error_size, "Metal SAM neck 3x3 constants are inconsistent");
    }

    uint64_t spatial = 0u;
    uint64_t value_count = 0u;
    uint64_t kernel_area = 0u;
    uint64_t weight_values = 0u;
    uint64_t tensor_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &spatial) ||
        !checked_mul_u64(spatial, (uint64_t)UOCR_SAM_NECK_CHANNELS, &value_count) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_NECK_KERNEL_SIZE, (uint64_t)UOCR_SAM_NECK_KERNEL_SIZE, &kernel_area) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_NECK_CHANNELS, (uint64_t)UOCR_SAM_NECK_CHANNELS, &weight_values) ||
        !checked_mul_u64(weight_values, kernel_area, &weight_values) ||
        !checked_mul_u64(value_count, 2u, &tensor_bytes) || !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64(value_count, output_element_bytes, &output_bytes) || value_count > (uint64_t)UINT32_MAX ||
        tensor_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal SAM neck 3x3 byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (tensor_bytes > max_buffer_length || weight_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM neck 3x3 buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_neck_conv3x3_f16_to_f16" :
                                        "uocr_sam_neck_conv3x3_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:input_nchw_f16
                                       length:(NSUInteger)tensor_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM neck 3x3 input buffer");
            goto cleanup_sam_neck_conv3x3;
        }
        src.label = @"uocr_sam_neck_conv3x3_input_nchw_f16";

        weight = [ctx->device newBufferWithBytes:weight_f16
                                          length:(NSUInteger)weight_bytes
                                         options:MTLResourceStorageModeShared];
        if (weight == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM neck 3x3 weight buffer");
            goto cleanup_sam_neck_conv3x3;
        }
        weight.label = @"uocr_sam_neck_conv3x3_weight_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM neck 3x3 output buffer");
            goto cleanup_sam_neck_conv3x3;
        }
        dst.label = @"uocr_sam_neck_conv3x3_output_nchw";
        memset([dst contents], 0, (size_t)output_bytes);

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            result = metal_fail(error, error_size, "Metal SAM neck 3x3 threadgroup memory exceeds device limit");
            goto cleanup_sam_neck_conv3x3;
        }

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM neck 3x3 command buffer");
            goto cleanup_sam_neck_conv3x3;
        }
        cb.label = @"uocr_sam_neck_conv3x3_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM neck 3x3 command encoder");
            goto cleanup_sam_neck_conv3x3;
        }

        uocr_metal_sam_neck_conv3x3_params params;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.channels = UOCR_SAM_NECK_CHANNELS;
        params.kernel_size = UOCR_SAM_NECK_KERNEL_SIZE;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)value_count, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal SAM neck 3x3 command failed: %s", [description UTF8String]);
            goto cleanup_sam_neck_conv3x3;
        }

        memcpy(out_nchw, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_sam_neck_conv3x3:
        [dst release];
        [weight release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_layernorm2d_f16(uocr_metal_context *ctx,
                                           const uint16_t *input_nchw_f16,
                                           const uint16_t *weight_f16,
                                           const uint16_t *bias_f16,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uocr_metal_dense_output_type output_type,
                                           void *out_nchw,
                                           char *error,
                                           size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_nchw_f16 == NULL || weight_f16 == NULL || bias_f16 == NULL || out_nchw == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM LayerNorm2d request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM LayerNorm2d output type %d", (int)output_type);
    }
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE) {
        return metal_fail(error, error_size, "invalid Metal SAM LayerNorm2d grid %ux%u", grid_w, grid_h);
    }
    if (UOCR_SAM_NECK_CHANNELS != 256u) {
        return metal_fail(error, error_size, "Metal SAM LayerNorm2d constants are inconsistent");
    }

    uint64_t spatial = 0u;
    uint64_t value_count = 0u;
    uint64_t tensor_bytes = 0u;
    uint64_t parameter_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &spatial) ||
        !checked_mul_u64(spatial, (uint64_t)UOCR_SAM_NECK_CHANNELS, &value_count) ||
        !checked_mul_u64(value_count, 2u, &tensor_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_NECK_CHANNELS, 2u, &parameter_bytes) ||
        !checked_mul_u64(value_count, output_element_bytes, &output_bytes) || value_count > (uint64_t)UINT32_MAX ||
        spatial > (uint64_t)UINT32_MAX || tensor_bytes > (uint64_t)SIZE_MAX ||
        parameter_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal SAM LayerNorm2d byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (tensor_bytes > max_buffer_length || parameter_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM LayerNorm2d buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_layernorm2d_f16_to_f16" :
                                        "uocr_sam_layernorm2d_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> bias = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:input_nchw_f16
                                       length:(NSUInteger)tensor_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM LayerNorm2d input buffer");
            goto cleanup_sam_layernorm2d;
        }
        src.label = @"uocr_sam_layernorm2d_input_nchw_f16";

        weight = [ctx->device newBufferWithBytes:weight_f16
                                          length:(NSUInteger)parameter_bytes
                                         options:MTLResourceStorageModeShared];
        if (weight == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM LayerNorm2d weight buffer");
            goto cleanup_sam_layernorm2d;
        }
        weight.label = @"uocr_sam_layernorm2d_weight_f16";

        bias = [ctx->device newBufferWithBytes:bias_f16
                                        length:(NSUInteger)parameter_bytes
                                       options:MTLResourceStorageModeShared];
        if (bias == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM LayerNorm2d bias buffer");
            goto cleanup_sam_layernorm2d;
        }
        bias.label = @"uocr_sam_layernorm2d_bias_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM LayerNorm2d output buffer");
            goto cleanup_sam_layernorm2d;
        }
        dst.label = @"uocr_sam_layernorm2d_output_nchw";
        memset([dst contents], 0, (size_t)output_bytes);

        const NSUInteger threads_per_group = metal_power2_threadgroup_width((NSUInteger)UOCR_SAM_NECK_CHANNELS,
                                                                            pipeline.maxTotalThreadsPerThreadgroup);
        if (threads_per_group < (NSUInteger)UOCR_SAM_NECK_CHANNELS) {
            result = metal_fail(error,
                                error_size,
                                "Metal SAM LayerNorm2d needs at least %u threads",
                                UOCR_SAM_NECK_CHANNELS);
            goto cleanup_sam_layernorm2d;
        }
        const uint64_t threadgroup_bytes = 2u * (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            result = metal_fail(error, error_size, "Metal SAM LayerNorm2d threadgroup memory exceeds device limit");
            goto cleanup_sam_layernorm2d;
        }

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM LayerNorm2d command buffer");
            goto cleanup_sam_layernorm2d;
        }
        cb.label = @"uocr_sam_layernorm2d_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM LayerNorm2d command encoder");
            goto cleanup_sam_layernorm2d;
        }

        uocr_metal_sam_layernorm2d_params params;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.channels = UOCR_SAM_NECK_CHANNELS;
        params.eps = 1.0e-6f;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:bias offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:(NSUInteger)threadgroup_bytes atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)spatial, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal SAM LayerNorm2d command failed: %s", [description UTF8String]);
            goto cleanup_sam_layernorm2d;
        }

        memcpy(out_nchw, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_sam_layernorm2d:
        [dst release];
        [bias release];
        [weight release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

static int metal_context_sam_conv3x3_stride2_nchw_f16(uocr_metal_context *ctx,
                                                       const uint16_t *input_nchw_f16,
                                                       const uint16_t *weight_f16,
                                                       uint32_t grid_w,
                                                       uint32_t grid_h,
                                                       uint32_t in_channels,
                                                       uint32_t out_channels,
                                                       uocr_metal_dense_output_type output_type,
                                                       void *out_nchw,
                                                       const char *diagnostic_name,
                                                       char *error,
                                                       size_t error_size) {
    metal_clear_error(error, error_size);
    if (diagnostic_name == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM stride-2 conv request");
    }
    if (ctx == NULL || input_nchw_f16 == NULL || weight_f16 == NULL || out_nchw == NULL ||
        in_channels == 0u || out_channels == 0u) {
        return metal_fail(error, error_size, "invalid %s request", diagnostic_name);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported %s output type %d", diagnostic_name, (int)output_type);
    }
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE) {
        return metal_fail(error, error_size, "invalid %s grid %ux%u", diagnostic_name, grid_w, grid_h);
    }
    if (UOCR_SAM_NECK_KERNEL_SIZE != 3u || UOCR_SAM_NET_STRIDE != 2u) {
        return metal_fail(error, error_size, "%s constants are inconsistent", diagnostic_name);
    }

    const uint32_t out_w = (grid_w + UOCR_SAM_NET_STRIDE - 1u) / UOCR_SAM_NET_STRIDE;
    const uint32_t out_h = (grid_h + UOCR_SAM_NET_STRIDE - 1u) / UOCR_SAM_NET_STRIDE;
    uint64_t input_spatial = 0u;
    uint64_t output_spatial = 0u;
    uint64_t input_values = 0u;
    uint64_t output_values = 0u;
    uint64_t weight_values = 0u;
    uint64_t kernel_area = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &input_spatial) ||
        !checked_mul_u64((uint64_t)out_w, (uint64_t)out_h, &output_spatial) ||
        !checked_mul_u64(input_spatial, (uint64_t)in_channels, &input_values) ||
        !checked_mul_u64(output_spatial, (uint64_t)out_channels, &output_values) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_NECK_KERNEL_SIZE, (uint64_t)UOCR_SAM_NECK_KERNEL_SIZE, &kernel_area) ||
        !checked_mul_u64((uint64_t)out_channels, (uint64_t)in_channels, &weight_values) ||
        !checked_mul_u64(weight_values, kernel_area, &weight_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) || !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes) ||
        output_values > (uint64_t)UINT32_MAX || input_bytes > (uint64_t)SIZE_MAX ||
        weight_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "%s byte-size overflow", diagnostic_name);
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "%s buffers exceed maxBufferLength %llu",
                          diagnostic_name,
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_conv3x3_stride2_f16_to_f16" :
                                        "uocr_sam_conv3x3_stride2_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:input_nchw_f16
                                       length:(NSUInteger)input_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate %s input buffer", diagnostic_name);
            goto cleanup_sam_conv3x3_stride2;
        }
        src.label = @"uocr_sam_conv3x3_stride2_input_nchw_f16";

        weight = [ctx->device newBufferWithBytes:weight_f16
                                          length:(NSUInteger)weight_bytes
                                         options:MTLResourceStorageModeShared];
        if (weight == nil) {
            result = metal_fail(error, error_size, "failed to allocate %s weight buffer", diagnostic_name);
            goto cleanup_sam_conv3x3_stride2;
        }
        weight.label = @"uocr_sam_conv3x3_stride2_weight_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate %s output buffer", diagnostic_name);
            goto cleanup_sam_conv3x3_stride2;
        }
        dst.label = @"uocr_sam_conv3x3_stride2_output_nchw";
        memset([dst contents], 0, (size_t)output_bytes);

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            result = metal_fail(error, error_size, "%s threadgroup memory exceeds device limit", diagnostic_name);
            goto cleanup_sam_conv3x3_stride2;
        }

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create %s command buffer", diagnostic_name);
            goto cleanup_sam_conv3x3_stride2;
        }
        cb.label = @"uocr_sam_conv3x3_stride2_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create %s command encoder", diagnostic_name);
            goto cleanup_sam_conv3x3_stride2;
        }

        uocr_metal_sam_conv3x3_stride2_params params;
        params.input_width = grid_w;
        params.input_height = grid_h;
        params.output_width = out_w;
        params.output_height = out_h;
        params.in_channels = in_channels;
        params.out_channels = out_channels;
        params.kernel_size = UOCR_SAM_NECK_KERNEL_SIZE;
        params.stride = UOCR_SAM_NET_STRIDE;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "%s command failed: %s", diagnostic_name, [description UTF8String]);
            goto cleanup_sam_conv3x3_stride2;
        }

        memcpy(out_nchw, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_sam_conv3x3_stride2:
        [dst release];
        [weight release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_net2_conv3x3_stride2_f16(uocr_metal_context *ctx,
                                                    const uint16_t *input_nchw_f16,
                                                    const uint16_t *weight_f16,
                                                    uint32_t grid_w,
                                                    uint32_t grid_h,
                                                    uocr_metal_dense_output_type output_type,
                                                    void *out_nchw,
                                                    char *error,
                                                    size_t error_size) {
    if (UOCR_SAM_NECK_CHANNELS != 256u || UOCR_SAM_NET2_CHANNELS != 512u ||
        UOCR_SAM_NECK_KERNEL_SIZE != 3u || UOCR_SAM_NET_STRIDE != 2u) {
        return metal_fail(error, error_size, "Metal SAM net_2 constants are inconsistent");
    }
    return metal_context_sam_conv3x3_stride2_nchw_f16(ctx,
                                                      input_nchw_f16,
                                                      weight_f16,
                                                      grid_w,
                                                      grid_h,
                                                      UOCR_SAM_NECK_CHANNELS,
                                                      UOCR_SAM_NET2_CHANNELS,
                                                      output_type,
                                                      out_nchw,
                                                      "Metal SAM net_2",
                                                      error,
                                                      error_size);
}

int uocr_metal_context_sam_net3_conv3x3_stride2_f16(uocr_metal_context *ctx,
                                                    const uint16_t *input_nchw_f16,
                                                    const uint16_t *weight_f16,
                                                    uint32_t grid_w,
                                                    uint32_t grid_h,
                                                    uocr_metal_dense_output_type output_type,
                                                    void *out_nchw,
                                                    char *error,
                                                    size_t error_size) {
    if (UOCR_SAM_NET2_CHANNELS != 512u || UOCR_SAM_NET3_CHANNELS != 1024u ||
        UOCR_SAM_FEATURE_CHANNELS != 1024u || UOCR_SAM_NECK_KERNEL_SIZE != 3u ||
        UOCR_SAM_NET_STRIDE != 2u) {
        return metal_fail(error, error_size, "Metal SAM net_3 constants are inconsistent");
    }
    return metal_context_sam_conv3x3_stride2_nchw_f16(ctx,
                                                      input_nchw_f16,
                                                      weight_f16,
                                                      grid_w,
                                                      grid_h,
                                                      UOCR_SAM_NET2_CHANNELS,
                                                      UOCR_SAM_NET3_CHANNELS,
                                                      output_type,
                                                      out_nchw,
                                                      "Metal SAM net_3",
                                                      error,
                                                      error_size);
}

int uocr_metal_context_clip_embed_sam_f16(uocr_metal_context *ctx,
                                          const uint16_t *sam_nchw_f16,
                                          const uint16_t *class_embedding_f16,
                                          uint32_t grid_w,
                                          uint32_t grid_h,
                                          uocr_metal_dense_output_type output_type,
                                          void *out_tokens,
                                          char *error,
                                          size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || sam_nchw_f16 == NULL || class_embedding_f16 == NULL || out_tokens == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP SAM embedding request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP SAM embedding output type %d", (int)output_type);
    }
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_CLIP_MAX_GRID_SIZE || grid_h > UOCR_CLIP_MAX_GRID_SIZE) {
        return metal_fail(error, error_size, "invalid Metal CLIP SAM embedding grid %ux%u", grid_w, grid_h);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_SAM_FEATURE_CHANNELS != UOCR_CLIP_HIDDEN_SIZE ||
        UOCR_CLIP_CLASS_TOKENS != 1u || UOCR_CLIP_MAX_GRID_SIZE != UOCR_GLOBAL_GRID_QUERIES) {
        return metal_fail(error, error_size, "Metal CLIP SAM embedding constants are inconsistent");
    }

    uint64_t spatial = 0u;
    uint64_t input_values = 0u;
    uint64_t output_tokens = 0u;
    uint64_t output_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t class_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &spatial) ||
        !checked_mul_u64(spatial, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &input_values) ||
        !checked_add_u64(spatial, (uint64_t)UOCR_CLIP_CLASS_TOKENS, &output_tokens) ||
        !checked_mul_u64(output_tokens, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &output_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_HIDDEN_SIZE, 2u, &class_bytes) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes) ||
        output_tokens > (uint64_t)UINT32_MAX || output_tokens > (uint64_t)UOCR_CLIP_MAX_TOKENS ||
        input_bytes > (uint64_t)SIZE_MAX || class_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP SAM embedding byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || class_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP SAM embedding buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_clip_embed_sam_f16_to_f16" :
                                        "uocr_clip_embed_sam_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:sam_nchw_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal CLIP SAM embedding input buffer");
        }
        src.label = @"uocr_clip_embed_sam_input_nchw_f16";

        id<MTLBuffer> cls = [ctx->device newBufferWithBytes:class_embedding_f16
                                                     length:(NSUInteger)class_bytes
                                                    options:MTLResourceStorageModeShared];
        if (cls == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal CLIP SAM embedding class buffer");
        }
        cls.label = @"uocr_clip_embed_sam_class_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [cls release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal CLIP SAM embedding output buffer");
        }
        dst.label = @"uocr_clip_embed_sam_output_tokens";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [cls release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal CLIP SAM embedding command buffer");
        }
        cb.label = @"uocr_clip_embed_sam_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [cls release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal CLIP SAM embedding command encoder");
        }

        uocr_metal_clip_embed_sam_params params;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.hidden_size = UOCR_CLIP_HIDDEN_SIZE;
        params.token_count = (uint32_t)output_tokens;

        NSUInteger threads_x = pipeline.threadExecutionWidth;
        if (threads_x == 0u) {
            threads_x = 1u;
        }
        if (threads_x > (NSUInteger)UOCR_CLIP_HIDDEN_SIZE) {
            threads_x = (NSUInteger)UOCR_CLIP_HIDDEN_SIZE;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:cls offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)UOCR_CLIP_HIDDEN_SIZE, (NSUInteger)output_tokens, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_x, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error,
                                error_size,
                                "Metal CLIP SAM embedding command failed: %s",
                                [description UTF8String]);
        } else {
            memcpy(out_tokens, [dst contents], (size_t)output_bytes);
            result = 1;
        }

        [dst release];
        [cls release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_clip_add_abs_pos_f16(uocr_metal_context *ctx,
                                            const uint16_t *tokens_f16,
                                            const uint16_t *pos_embed_f16,
                                            uint32_t grid_w,
                                            uint32_t grid_h,
                                            uocr_metal_dense_output_type output_type,
                                            void *out_tokens,
                                            char *error,
                                            size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || tokens_f16 == NULL || pos_embed_f16 == NULL || out_tokens == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP abs pos request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP abs pos output type %d", (int)output_type);
    }
    if (grid_w == 0u || grid_h == 0u || grid_w != grid_h ||
        (grid_w != UOCR_GLOBAL_GRID_QUERIES && grid_w != UOCR_LOCAL_GRID_QUERIES)) {
        return metal_fail(error, error_size, "unsupported Metal CLIP abs pos grid %ux%u", grid_w, grid_h);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_CLASS_TOKENS != 1u ||
        UOCR_CLIP_POSITION_GRID != UOCR_GLOBAL_GRID_QUERIES || UOCR_CLIP_GLOBAL_TOKENS != 257u ||
        UOCR_CLIP_LOCAL_TOKENS != 101u || UOCR_CLIP_MAX_TOKENS != UOCR_CLIP_GLOBAL_TOKENS) {
        return metal_fail(error, error_size, "Metal CLIP abs pos constants are inconsistent");
    }

    uint64_t spatial = 0u;
    uint64_t token_count = 0u;
    uint64_t token_values = 0u;
    uint64_t token_bytes = 0u;
    uint64_t pos_values = 0u;
    uint64_t pos_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &spatial) ||
        !checked_add_u64(spatial, (uint64_t)UOCR_CLIP_CLASS_TOKENS, &token_count) ||
        !checked_mul_u64(token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &token_values) ||
        !checked_mul_u64(token_values, 2u, &token_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_GLOBAL_TOKENS, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &pos_values) ||
        !checked_mul_u64(pos_values, 2u, &pos_bytes) ||
        !checked_mul_u64(token_values, output_element_bytes, &output_bytes) ||
        token_count > (uint64_t)UINT32_MAX || token_count > (uint64_t)UOCR_CLIP_MAX_TOKENS ||
        token_bytes > (uint64_t)SIZE_MAX || pos_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP abs pos byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (token_bytes > max_buffer_length || pos_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP abs pos buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_clip_add_abs_pos_f16_to_f16" :
                                        "uocr_clip_add_abs_pos_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> tokens = [ctx->device newBufferWithBytes:tokens_f16
                                                        length:(NSUInteger)token_bytes
                                                       options:MTLResourceStorageModeShared];
        if (tokens == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal CLIP abs pos token buffer");
        }
        tokens.label = @"uocr_clip_abs_pos_tokens_f16";

        id<MTLBuffer> pos = [ctx->device newBufferWithBytes:pos_embed_f16
                                                     length:(NSUInteger)pos_bytes
                                                    options:MTLResourceStorageModeShared];
        if (pos == nil) {
            [tokens release];
            return metal_fail(error, error_size, "failed to allocate Metal CLIP abs pos table buffer");
        }
        pos.label = @"uocr_clip_abs_pos_table_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [pos release];
            [tokens release];
            return metal_fail(error, error_size, "failed to allocate Metal CLIP abs pos output buffer");
        }
        dst.label = @"uocr_clip_abs_pos_output_tokens";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [pos release];
            [tokens release];
            return metal_fail(error, error_size, "failed to create Metal CLIP abs pos command buffer");
        }
        cb.label = @"uocr_clip_abs_pos_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [pos release];
            [tokens release];
            return metal_fail(error, error_size, "failed to create Metal CLIP abs pos command encoder");
        }

        uocr_metal_clip_abs_pos_params params;
        params.source_grid = UOCR_CLIP_POSITION_GRID;
        params.target_width = grid_w;
        params.target_height = grid_h;
        params.hidden_size = UOCR_CLIP_HIDDEN_SIZE;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:tokens offset:0u atIndex:0u];
        [enc setBuffer:pos offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)token_values, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error,
                                error_size,
                                "Metal CLIP abs pos command failed: %s",
                                [description UTF8String]);
        } else {
            memcpy(out_tokens, [dst contents], (size_t)output_bytes);
            result = 1;
        }

        [dst release];
        [pos release];
        [tokens release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

static int metal_context_clip_token_layernorm_f16(uocr_metal_context *ctx,
                                                   const uint16_t *input_f16,
                                                   const uint16_t *weight_f16,
                                                   const uint16_t *bias_f16,
                                                   uint32_t token_count,
                                                   uocr_metal_layernorm_output_type output_type,
                                                   void *out,
                                                   const char *op_name,
                                                   const char *invalid_request_message,
                                                   const char *invalid_token_count_message,
                                                   NSString *weight_label,
                                                   NSString *bias_label,
                                                   char *error,
                                                   size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_f16 == NULL || bias_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "%s", invalid_request_message);
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "%s %u", invalid_token_count_message, token_count);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_BLOCKS != 24u || UOCR_CLIP_LAYERNORM_EPS != 1.0e-5f ||
        UOCR_CLIP_PRE_LAYERNORM_EPS != UOCR_CLIP_LAYERNORM_EPS || UOCR_CLIP_GLOBAL_TOKENS != 257u ||
        UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "%s constants are inconsistent", op_name);
    }

    const uint64_t parameter_bytes = (uint64_t)UOCR_CLIP_HIDDEN_SIZE * 2u;
    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (parameter_bytes > (uint64_t)SIZE_MAX || parameter_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "%s buffers exceed maxBufferLength %llu",
                          op_name,
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:weight_f16
                                                        length:(NSUInteger)parameter_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            return metal_fail(error, error_size, "failed to allocate %s weight buffer", op_name);
        }
        weight.label = weight_label;

        id<MTLBuffer> bias = [ctx->device newBufferWithBytes:bias_f16
                                                      length:(NSUInteger)parameter_bytes
                                                     options:MTLResourceStorageModeShared];
        if (bias == nil) {
            [weight release];
            return metal_fail(error, error_size, "failed to allocate %s bias buffer", op_name);
        }
        bias.label = bias_label;

        const int ok = metal_context_layernorm_f16_with_parameter_buffers(ctx,
                                                                         input_f16,
                                                                         weight,
                                                                         0u,
                                                                         bias,
                                                                         0u,
                                                                         token_count,
                                                                         UOCR_CLIP_HIDDEN_SIZE,
                                                                         UOCR_CLIP_LAYERNORM_EPS,
                                                                         output_type,
                                                                         out,
                                                                         op_name,
                                                                         error,
                                                                         error_size);
        [bias release];
        [weight release];
        return ok;
    }
}

int uocr_metal_context_clip_pre_layernorm_f16(uocr_metal_context *ctx,
                                              const uint16_t *input_f16,
                                              const uint16_t *weight_f16,
                                              const uint16_t *bias_f16,
                                              uint32_t token_count,
                                              uocr_metal_layernorm_output_type output_type,
                                              void *out,
                                              char *error,
                                              size_t error_size) {
    return metal_context_clip_token_layernorm_f16(ctx,
                                                  input_f16,
                                                  weight_f16,
                                                  bias_f16,
                                                  token_count,
                                                  output_type,
                                                  out,
                                                  "Metal CLIP pre-LayerNorm",
                                                  "invalid Metal CLIP pre-LayerNorm request",
                                                  "invalid Metal CLIP pre-LayerNorm token count",
                                                  @"uocr_clip_pre_layernorm_weight_f16",
                                                  @"uocr_clip_pre_layernorm_bias_f16",
                                                  error,
                                                  error_size);
}

int uocr_metal_context_clip_layernorm_f16(uocr_metal_context *ctx,
                                          const uint16_t *input_f16,
                                          const uint16_t *weight_f16,
                                          const uint16_t *bias_f16,
                                          uint32_t token_count,
                                          uocr_metal_layernorm_output_type output_type,
                                          void *out,
                                          char *error,
                                          size_t error_size) {
    return metal_context_clip_token_layernorm_f16(ctx,
                                                  input_f16,
                                                  weight_f16,
                                                  bias_f16,
                                                  token_count,
                                                  output_type,
                                                  out,
                                                  "Metal CLIP LayerNorm",
                                                  "invalid Metal CLIP LayerNorm request",
                                                  "invalid Metal CLIP LayerNorm token count",
                                                  @"uocr_clip_layernorm_weight_f16",
                                                  @"uocr_clip_layernorm_bias_f16",
                                                  error,
                                                  error_size);
}

int uocr_metal_context_clip_qkv_f16(uocr_metal_context *ctx,
                                    const uint16_t *input_f16,
                                    const uint16_t *qkv_weight_f16,
                                    const uint16_t *qkv_bias_f16,
                                    uint32_t token_count,
                                    uocr_metal_dense_output_type output_type,
                                    void *q_out,
                                    void *k_out,
                                    void *v_out,
                                    char *error,
                                    size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || qkv_weight_f16 == NULL || qkv_bias_f16 == NULL ||
        q_out == NULL || k_out == NULL || v_out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP QKV request");
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP QKV token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP QKV output type %d", (int)output_type);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_QKV_SIZE != 3072u ||
        UOCR_CLIP_QKV_SIZE != 3u * UOCR_CLIP_HIDDEN_SIZE ||
        UOCR_CLIP_ATTENTION_HEADS * UOCR_CLIP_HEAD_DIM != UOCR_CLIP_HIDDEN_SIZE ||
        UOCR_CLIP_ATTENTION_HEADS != 16u || UOCR_CLIP_HEAD_DIM != 64u ||
        UOCR_CLIP_GLOBAL_TOKENS != 257u || UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP QKV constants are inconsistent");
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t bias_bytes = 0u;
    uint64_t output_values_per_projection = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_QKV_SIZE, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_QKV_SIZE, 2u, &bias_bytes) ||
        !checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &output_values_per_projection) ||
        !checked_mul_u64(output_values_per_projection, 3u, &output_values) ||
        !checked_mul_u64(output_values_per_projection, output_element_bytes, &output_bytes) ||
        input_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX ||
        bias_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal CLIP QKV byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || bias_bytes > max_buffer_length ||
        output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP QKV buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_clip_qkv_f16_to_f16" :
                                        "uocr_clip_qkv_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> bias = nil;
        id<MTLBuffer> q_dst = nil;
        id<MTLBuffer> k_dst = nil;
        id<MTLBuffer> v_dst = nil;

        src = [ctx->device newBufferWithBytes:input_f16
                                       length:(NSUInteger)input_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP QKV input buffer");
            goto cleanup_clip_qkv;
        }
        src.label = @"uocr_clip_qkv_input_f16";

        weight = [ctx->device newBufferWithBytes:qkv_weight_f16
                                          length:(NSUInteger)weight_bytes
                                         options:MTLResourceStorageModeShared];
        if (weight == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP QKV weight buffer");
            goto cleanup_clip_qkv;
        }
        weight.label = @"uocr_clip_qkv_weight_f16";

        bias = [ctx->device newBufferWithBytes:qkv_bias_f16
                                        length:(NSUInteger)bias_bytes
                                       options:MTLResourceStorageModeShared];
        if (bias == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP QKV bias buffer");
            goto cleanup_clip_qkv;
        }
        bias.label = @"uocr_clip_qkv_bias_f16";

        q_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        k_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        v_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (q_dst == nil || k_dst == nil || v_dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP QKV output buffers");
            goto cleanup_clip_qkv;
        }
        q_dst.label = @"uocr_clip_q_f16";
        k_dst.label = @"uocr_clip_k_f16";
        v_dst.label = @"uocr_clip_v_f16";
        memset([q_dst contents], 0, (size_t)output_bytes);
        memset([k_dst contents], 0, (size_t)output_bytes);
        memset([v_dst contents], 0, (size_t)output_bytes);

        uocr_metal_dense_params params;
        params.input_rows = token_count;
        params.in_features = UOCR_CLIP_HIDDEN_SIZE;
        params.out_features = UOCR_CLIP_HIDDEN_SIZE;
        params.has_bias = 1u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            result = metal_fail(error, error_size, "Metal CLIP QKV threadgroup memory exceeds device limit");
            goto cleanup_clip_qkv;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP QKV command buffer");
            goto cleanup_clip_qkv;
        }
        cb.label = @"uocr_clip_qkv_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP QKV command encoder");
            goto cleanup_clip_qkv;
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:bias offset:0u atIndex:2u];
        [enc setBuffer:q_dst offset:0u atIndex:3u];
        [enc setBuffer:k_dst offset:0u atIndex:4u];
        [enc setBuffer:v_dst offset:0u atIndex:5u];
        [enc setBytes:&params length:sizeof(params) atIndex:6u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal CLIP QKV command failed: %s", [description UTF8String]);
            goto cleanup_clip_qkv;
        }

        memcpy(q_out, [q_dst contents], (size_t)output_bytes);
        memcpy(k_out, [k_dst contents], (size_t)output_bytes);
        memcpy(v_out, [v_dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_clip_qkv:
        [v_dst release];
        [k_dst release];
        [q_dst release];
        [bias release];
        [weight release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_clip_attention_f16(uocr_metal_context *ctx,
                                          const uint16_t *q_f16,
                                          const uint16_t *k_f16,
                                          const uint16_t *v_f16,
                                          uint32_t token_count,
                                          uocr_metal_dense_output_type output_type,
                                          void *out,
                                          char *error,
                                          size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP attention request");
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP attention token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP attention output type %d", (int)output_type);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_ATTENTION_HEADS != 16u || UOCR_CLIP_HEAD_DIM != 64u ||
        UOCR_CLIP_ATTENTION_HEADS * UOCR_CLIP_HEAD_DIM != UOCR_CLIP_HIDDEN_SIZE ||
        UOCR_CLIP_GLOBAL_TOKENS != 257u || UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP attention constants are inconsistent");
    }

    uint64_t attention_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t group_count = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &attention_values) ||
        !checked_mul_u64(attention_values, 2u, &input_bytes) ||
        !checked_mul_u64(attention_values, output_element_bytes, &output_bytes) ||
        !checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_ATTENTION_HEADS, &group_count) ||
        attention_values > (uint64_t)UINT32_MAX || group_count > (uint64_t)UINT32_MAX ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP attention byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP attention buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        /* The SAM window-attention shader is an all-to-all attention primitive
         * parameterized by token count, head count, and head dim. CLIP full
         * attention is represented as one window spanning the whole token set.
         */
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_window_attention_f16_to_f16" :
                                        "uocr_sam_window_attention_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> q_src = nil;
        id<MTLBuffer> k_src = nil;
        id<MTLBuffer> v_src = nil;
        id<MTLBuffer> dst = nil;

        q_src = [ctx->device newBufferWithBytes:q_f16 length:(NSUInteger)input_bytes options:MTLResourceStorageModeShared];
        if (q_src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP attention Q buffer");
            goto cleanup_clip_attention;
        }
        q_src.label = @"uocr_clip_attention_q_f16";

        k_src = [ctx->device newBufferWithBytes:k_f16 length:(NSUInteger)input_bytes options:MTLResourceStorageModeShared];
        if (k_src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP attention K buffer");
            goto cleanup_clip_attention;
        }
        k_src.label = @"uocr_clip_attention_k_f16";

        v_src = [ctx->device newBufferWithBytes:v_f16 length:(NSUInteger)input_bytes options:MTLResourceStorageModeShared];
        if (v_src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP attention V buffer");
            goto cleanup_clip_attention;
        }
        v_src.label = @"uocr_clip_attention_v_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP attention output buffer");
            goto cleanup_clip_attention;
        }
        dst.label = @"uocr_clip_attention_output";
        memset([dst contents], 0, (size_t)output_bytes);

        uocr_metal_sam_window_attention_params params;
        params.windows = 1u;
        params.tokens_per_window = token_count;
        params.heads = UOCR_CLIP_ATTENTION_HEADS;
        params.head_dim = UOCR_CLIP_HEAD_DIM;
        params.scale = 1.0f / sqrtf((float)UOCR_CLIP_HEAD_DIM);
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width((NSUInteger)UOCR_CLIP_HEAD_DIM,
                                                                            pipeline.maxTotalThreadsPerThreadgroup);
        if (threads_per_group < (NSUInteger)UOCR_CLIP_HEAD_DIM) {
            result = metal_fail(error, error_size, "Metal CLIP attention needs at least %u threads", UOCR_CLIP_HEAD_DIM);
            goto cleanup_clip_attention;
        }
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            result = metal_fail(error, error_size, "Metal CLIP attention threadgroup memory exceeds device limit");
            goto cleanup_clip_attention;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP attention command buffer");
            goto cleanup_clip_attention;
        }
        cb.label = @"uocr_clip_attention_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP attention command encoder");
            goto cleanup_clip_attention;
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_src offset:0u atIndex:0u];
        [enc setBuffer:k_src offset:0u atIndex:1u];
        [enc setBuffer:v_src offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)group_count, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal CLIP attention command failed: %s", [description UTF8String]);
            goto cleanup_clip_attention;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_clip_attention:
        [dst release];
        [v_src release];
        [k_src release];
        [q_src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_clip_output_projection_f16(uocr_metal_context *ctx,
                                                  const uint16_t *input_f16,
                                                  const uint16_t *weight_f16,
                                                  const uint16_t *bias_f16,
                                                  uint32_t token_count,
                                                  uocr_metal_dense_output_type output_type,
                                                  void *out,
                                                  char *error,
                                                  size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_f16 == NULL || bias_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP output projection request");
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP output projection token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal CLIP output projection output type %d",
                          (int)output_type);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_GLOBAL_TOKENS != 257u || UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP output projection constants are inconsistent");
    }

    if (!uocr_metal_context_dense_f16(ctx,
                                      input_f16,
                                      weight_f16,
                                      bias_f16,
                                      token_count,
                                      UOCR_CLIP_HIDDEN_SIZE,
                                      UOCR_CLIP_HIDDEN_SIZE,
                                      output_type,
                                      out,
                                      error,
                                      error_size)) {
        char detail[512];
        (void)snprintf(detail,
                       sizeof(detail),
                       "%s",
                       (error != NULL && error[0] != '\0') ? error : "unknown error");
        return metal_fail(error, error_size, "failed to compute Metal CLIP output projection: %s", detail);
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_clip_quickgelu_f16(uocr_metal_context *ctx,
                                          const uint16_t *input_f16,
                                          uint32_t token_count,
                                          uocr_metal_dense_output_type output_type,
                                          void *out,
                                          char *error,
                                          size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP QuickGELU request");
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP QuickGELU token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP QuickGELU output type %d", (int)output_type);
    }
    if (UOCR_CLIP_MLP_INTERMEDIATE != 4096u || UOCR_CLIP_GLOBAL_TOKENS != 257u ||
        UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP QuickGELU constants are inconsistent");
    }

    uint64_t value_count_u64 = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_MLP_INTERMEDIATE, &value_count_u64) ||
        !checked_mul_u64(value_count_u64, 2u, &input_bytes) ||
        !checked_mul_u64(value_count_u64, output_element_bytes, &output_bytes) ||
        value_count_u64 > (uint64_t)UINT32_MAX || input_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP QuickGELU byte-size overflow");
    }
    const uint32_t value_count = (uint32_t)value_count_u64;

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP QuickGELU buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_clip_quickgelu_f16_to_f16" :
                                        "uocr_clip_quickgelu_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:input_f16
                                       length:(NSUInteger)input_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP QuickGELU input buffer");
            goto cleanup_clip_quickgelu;
        }
        src.label = @"uocr_clip_quickgelu_input_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP QuickGELU output buffer");
            goto cleanup_clip_quickgelu;
        }
        dst.label = @"uocr_clip_quickgelu_output";
        memset([dst contents], 0, (size_t)output_bytes);

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP QuickGELU command buffer");
            goto cleanup_clip_quickgelu;
        }
        cb.label = @"uocr_clip_quickgelu_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP QuickGELU command encoder");
            goto cleanup_clip_quickgelu;
        }

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:dst offset:0u atIndex:1u];
        [enc setBytes:&value_count length:sizeof(value_count) atIndex:2u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)value_count, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal CLIP QuickGELU command failed: %s", [description UTF8String]);
            goto cleanup_clip_quickgelu;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_clip_quickgelu:
        [dst release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_clip_mlp_f16(uocr_metal_context *ctx,
                                    const uint16_t *input_f16,
                                    const uint16_t *fc1_weight_f16,
                                    const uint16_t *fc1_bias_f16,
                                    const uint16_t *fc2_weight_f16,
                                    const uint16_t *fc2_bias_f16,
                                    uint32_t token_count,
                                    uocr_metal_dense_output_type output_type,
                                    void *out,
                                    char *error,
                                    size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || fc1_weight_f16 == NULL || fc1_bias_f16 == NULL ||
        fc2_weight_f16 == NULL || fc2_bias_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP MLP request");
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP MLP token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP MLP output type %d", (int)output_type);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_MLP_INTERMEDIATE != 4096u ||
        UOCR_CLIP_GLOBAL_TOKENS != 257u || UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP MLP constants are inconsistent");
    }

    uint64_t input_values = 0u;
    uint64_t mid_values = 0u;
    uint64_t output_values = 0u;
    uint64_t fc1_weight_values = 0u;
    uint64_t fc2_weight_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t mid_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t fc1_weight_bytes = 0u;
    uint64_t fc2_weight_bytes = 0u;
    uint64_t fc1_bias_bytes = 0u;
    uint64_t fc2_bias_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_MLP_INTERMEDIATE, &mid_values) ||
        !checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &output_values) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_MLP_INTERMEDIATE, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &fc1_weight_values) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_HIDDEN_SIZE, (uint64_t)UOCR_CLIP_MLP_INTERMEDIATE, &fc2_weight_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64(mid_values, 2u, &mid_bytes) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes) ||
        !checked_mul_u64(fc1_weight_values, 2u, &fc1_weight_bytes) ||
        !checked_mul_u64(fc2_weight_values, 2u, &fc2_weight_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_MLP_INTERMEDIATE, 2u, &fc1_bias_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_CLIP_HIDDEN_SIZE, 2u, &fc2_bias_bytes) ||
        input_bytes > (uint64_t)SIZE_MAX || mid_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        fc1_weight_bytes > (uint64_t)SIZE_MAX || fc2_weight_bytes > (uint64_t)SIZE_MAX ||
        fc1_bias_bytes > (uint64_t)SIZE_MAX || fc2_bias_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP MLP byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || mid_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        fc1_weight_bytes > max_buffer_length || fc2_weight_bytes > max_buffer_length ||
        fc1_bias_bytes > max_buffer_length || fc2_bias_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP MLP buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    uint16_t *fc1_out_f16 = (uint16_t *)malloc((size_t)mid_bytes);
    uint16_t *activated_f16 = (uint16_t *)malloc((size_t)mid_bytes);
    if (fc1_out_f16 == NULL || activated_f16 == NULL) {
        result = metal_fail(error, error_size, "failed to allocate Metal CLIP MLP intermediate buffers");
        goto cleanup_clip_mlp;
    }

    if (!uocr_metal_context_dense_f16(ctx,
                                      input_f16,
                                      fc1_weight_f16,
                                      fc1_bias_f16,
                                      token_count,
                                      UOCR_CLIP_HIDDEN_SIZE,
                                      UOCR_CLIP_MLP_INTERMEDIATE,
                                      UOCR_METAL_DENSE_OUTPUT_F16,
                                      fc1_out_f16,
                                      error,
                                      error_size)) {
        char detail[512];
        (void)snprintf(detail,
                       sizeof(detail),
                       "%s",
                       (error != NULL && error[0] != '\0') ? error : "unknown error");
        result = metal_fail(error, error_size, "failed to compute Metal CLIP MLP fc1: %s", detail);
        goto cleanup_clip_mlp;
    }

    if (!uocr_metal_context_clip_quickgelu_f16(ctx,
                                               fc1_out_f16,
                                               token_count,
                                               UOCR_METAL_DENSE_OUTPUT_F16,
                                               activated_f16,
                                               error,
                                               error_size)) {
        char detail[512];
        (void)snprintf(detail,
                       sizeof(detail),
                       "%s",
                       (error != NULL && error[0] != '\0') ? error : "unknown error");
        result = metal_fail(error, error_size, "failed to compute Metal CLIP MLP QuickGELU: %s", detail);
        goto cleanup_clip_mlp;
    }

    if (!uocr_metal_context_dense_f16(ctx,
                                      activated_f16,
                                      fc2_weight_f16,
                                      fc2_bias_f16,
                                      token_count,
                                      UOCR_CLIP_MLP_INTERMEDIATE,
                                      UOCR_CLIP_HIDDEN_SIZE,
                                      output_type,
                                      out,
                                      error,
                                      error_size)) {
        char detail[512];
        (void)snprintf(detail,
                       sizeof(detail),
                       "%s",
                       (error != NULL && error[0] != '\0') ? error : "unknown error");
        result = metal_fail(error, error_size, "failed to compute Metal CLIP MLP fc2: %s", detail);
        goto cleanup_clip_mlp;
    }

    result = 1;

cleanup_clip_mlp:
    free(activated_f16);
    free(fc1_out_f16);
    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_clip_residual_add_f16(uocr_metal_context *ctx,
                                             const uint16_t *base_f16,
                                             const uint16_t *update_f16,
                                             uint32_t token_count,
                                             uocr_metal_dense_output_type output_type,
                                             void *out,
                                             char *error,
                                             size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || base_f16 == NULL || update_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP residual request");
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP residual token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP residual output type %d", (int)output_type);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_GLOBAL_TOKENS != 257u || UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP residual constants are inconsistent");
    }

    uint64_t value_count_u64 = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &value_count_u64) ||
        !checked_mul_u64(value_count_u64, 2u, &input_bytes) ||
        !checked_mul_u64(value_count_u64, output_element_bytes, &output_bytes) ||
        value_count_u64 > (uint64_t)UINT32_MAX || input_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP residual byte-size overflow");
    }
    const uint32_t value_count = (uint32_t)value_count_u64;

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP residual buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_clip_residual_add_f16_to_f16" :
                                        "uocr_clip_residual_add_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> base = nil;
        id<MTLBuffer> update = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        base = [ctx->device newBufferWithBytes:base_f16
                                        length:(NSUInteger)input_bytes
                                       options:MTLResourceStorageModeShared];
        if (base == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP residual base buffer");
            goto cleanup_clip_residual;
        }
        base.label = @"uocr_clip_residual_base_f16";

        update = [ctx->device newBufferWithBytes:update_f16
                                          length:(NSUInteger)input_bytes
                                         options:MTLResourceStorageModeShared];
        if (update == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP residual update buffer");
            goto cleanup_clip_residual;
        }
        update.label = @"uocr_clip_residual_update_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP residual output buffer");
            goto cleanup_clip_residual;
        }
        dst.label = @"uocr_clip_residual_output";
        memset([dst contents], 0, (size_t)output_bytes);

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP residual command buffer");
            goto cleanup_clip_residual;
        }
        cb.label = @"uocr_clip_residual_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP residual command encoder");
            goto cleanup_clip_residual;
        }

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:base offset:0u atIndex:0u];
        [enc setBuffer:update offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&value_count length:sizeof(value_count) atIndex:3u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)value_count, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal CLIP residual command failed: %s", [description UTF8String]);
            goto cleanup_clip_residual;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_clip_residual:
        [dst release];
        [update release];
        [base release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

static int metal_clip_transformer_block_has_weights(const uocr_metal_clip_transformer_block_f16 *block) {
    return block != NULL && block->ln1_weight_f16 != NULL && block->ln1_bias_f16 != NULL &&
           block->qkv_weight_f16 != NULL && block->qkv_bias_f16 != NULL &&
           block->out_proj_weight_f16 != NULL && block->out_proj_bias_f16 != NULL &&
           block->ln2_weight_f16 != NULL && block->ln2_bias_f16 != NULL &&
           block->mlp_fc1_weight_f16 != NULL && block->mlp_fc1_bias_f16 != NULL &&
           block->mlp_fc2_weight_f16 != NULL && block->mlp_fc2_bias_f16 != NULL;
}

static void metal_copy_error_detail(char *detail, size_t detail_size, const char *error) {
    if (detail == NULL || detail_size == 0u) {
        return;
    }
    (void)snprintf(detail, detail_size, "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
}

static int metal_clip_transformer_activation_bytes(uint32_t token_count,
                                                   uint64_t *value_count,
                                                   uint64_t *f16_bytes,
                                                   char *error,
                                                   size_t error_size) {
    uint64_t values = 0u;
    uint64_t bytes = 0u;
    if (!checked_mul_u64((uint64_t)token_count, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &values) ||
        !checked_mul_u64(values, 2u, &bytes) || values > (uint64_t)UINT32_MAX || bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP transformer byte-size overflow");
    }
    if (value_count != NULL) {
        *value_count = values;
    }
    if (f16_bytes != NULL) {
        *f16_bytes = bytes;
    }
    return 1;
}

int uocr_metal_context_clip_transformer_block_f16(uocr_metal_context *ctx,
                                                  const uint16_t *input_f16,
                                                  const uocr_metal_clip_transformer_block_f16 *block,
                                                  uint32_t token_count,
                                                  uocr_metal_dense_output_type output_type,
                                                  void *out,
                                                  char *error,
                                                  size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || !metal_clip_transformer_block_has_weights(block) || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP transformer block request");
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP transformer block token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP transformer block output type %d", (int)output_type);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_CLIP_BLOCKS != 24u || UOCR_CLIP_ATTENTION_HEADS != 16u ||
        UOCR_CLIP_HEAD_DIM != 64u || UOCR_CLIP_QKV_SIZE != 3072u || UOCR_CLIP_MLP_INTERMEDIATE != 4096u ||
        UOCR_CLIP_GLOBAL_TOKENS != 257u || UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP transformer block constants are inconsistent");
    }

    uint64_t hidden_values = 0u;
    uint64_t hidden_bytes = 0u;
    if (!metal_clip_transformer_activation_bytes(token_count, &hidden_values, &hidden_bytes, error, error_size)) {
        return 0;
    }
    (void)hidden_values;

    int result = 0;
    uint16_t *norm1_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *q_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *k_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *v_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *attention_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *projected_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *residual1_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *norm2_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *mlp_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    if (norm1_f16 == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL || attention_f16 == NULL ||
        projected_f16 == NULL || residual1_f16 == NULL || norm2_f16 == NULL || mlp_f16 == NULL) {
        result = metal_fail(error, error_size, "failed to allocate Metal CLIP transformer block scratch buffers");
        goto cleanup_clip_transformer_block;
    }

#define RUN_CLIP_BLOCK_STEP(step_name, call_expr)                                                        \
    do {                                                                                                \
        if (!(call_expr)) {                                                                             \
            char detail[512];                                                                           \
            metal_copy_error_detail(detail, sizeof(detail), error);                                      \
            result = metal_fail(error, error_size, "failed to compute Metal CLIP transformer block %s: %s", step_name, detail); \
            goto cleanup_clip_transformer_block;                                                        \
        }                                                                                               \
    } while (0)

    RUN_CLIP_BLOCK_STEP("LayerNorm1",
                        uocr_metal_context_clip_layernorm_f16(ctx,
                                                              input_f16,
                                                              block->ln1_weight_f16,
                                                              block->ln1_bias_f16,
                                                              token_count,
                                                              UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                                              norm1_f16,
                                                              error,
                                                              error_size));
    RUN_CLIP_BLOCK_STEP("QKV",
                        uocr_metal_context_clip_qkv_f16(ctx,
                                                         norm1_f16,
                                                         block->qkv_weight_f16,
                                                         block->qkv_bias_f16,
                                                         token_count,
                                                         UOCR_METAL_DENSE_OUTPUT_F16,
                                                         q_f16,
                                                         k_f16,
                                                         v_f16,
                                                         error,
                                                         error_size));
    RUN_CLIP_BLOCK_STEP("attention",
                        uocr_metal_context_clip_attention_f16(ctx,
                                                              q_f16,
                                                              k_f16,
                                                              v_f16,
                                                              token_count,
                                                              UOCR_METAL_DENSE_OUTPUT_F16,
                                                              attention_f16,
                                                              error,
                                                              error_size));
    RUN_CLIP_BLOCK_STEP("output projection",
                        uocr_metal_context_clip_output_projection_f16(ctx,
                                                                      attention_f16,
                                                                      block->out_proj_weight_f16,
                                                                      block->out_proj_bias_f16,
                                                                      token_count,
                                                                      UOCR_METAL_DENSE_OUTPUT_F16,
                                                                      projected_f16,
                                                                      error,
                                                                      error_size));
    RUN_CLIP_BLOCK_STEP("attention residual",
                        uocr_metal_context_clip_residual_add_f16(ctx,
                                                                 input_f16,
                                                                 projected_f16,
                                                                 token_count,
                                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                                 residual1_f16,
                                                                 error,
                                                                 error_size));
    RUN_CLIP_BLOCK_STEP("LayerNorm2",
                        uocr_metal_context_clip_layernorm_f16(ctx,
                                                              residual1_f16,
                                                              block->ln2_weight_f16,
                                                              block->ln2_bias_f16,
                                                              token_count,
                                                              UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                                              norm2_f16,
                                                              error,
                                                              error_size));
    RUN_CLIP_BLOCK_STEP("MLP",
                        uocr_metal_context_clip_mlp_f16(ctx,
                                                        norm2_f16,
                                                        block->mlp_fc1_weight_f16,
                                                        block->mlp_fc1_bias_f16,
                                                        block->mlp_fc2_weight_f16,
                                                        block->mlp_fc2_bias_f16,
                                                        token_count,
                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                        mlp_f16,
                                                        error,
                                                        error_size));
    RUN_CLIP_BLOCK_STEP("MLP residual",
                        uocr_metal_context_clip_residual_add_f16(ctx,
                                                                 residual1_f16,
                                                                 mlp_f16,
                                                                 token_count,
                                                                 output_type,
                                                                 out,
                                                                 error,
                                                                 error_size));

#undef RUN_CLIP_BLOCK_STEP

    result = 1;

cleanup_clip_transformer_block:
    free(mlp_f16);
    free(norm2_f16);
    free(residual1_f16);
    free(projected_f16);
    free(attention_f16);
    free(v_f16);
    free(k_f16);
    free(q_f16);
    free(norm1_f16);
    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_clip_transformer_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_f16,
                                            const uocr_metal_clip_transformer_block_f16 *blocks,
                                            uint32_t block_count,
                                            uint32_t token_count,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || blocks == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP transformer request");
    }
    if (block_count != UOCR_CLIP_BLOCKS) {
        return metal_fail(error, error_size, "invalid Metal CLIP transformer block count %u", block_count);
    }
    if (token_count != UOCR_CLIP_GLOBAL_TOKENS && token_count != UOCR_CLIP_LOCAL_TOKENS) {
        return metal_fail(error, error_size, "invalid Metal CLIP transformer token count %u", token_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP transformer output type %d", (int)output_type);
    }
    for (uint32_t block_index = 0u; block_index < block_count; ++block_index) {
        if (!metal_clip_transformer_block_has_weights(&blocks[block_index])) {
            return metal_fail(error, error_size, "invalid Metal CLIP transformer block %u weights", block_index);
        }
    }

    uint64_t hidden_bytes = 0u;
    if (!metal_clip_transformer_activation_bytes(token_count, NULL, &hidden_bytes, error, error_size)) {
        return 0;
    }

    int result = 0;
    uint16_t *state_a_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *state_b_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    if (state_a_f16 == NULL || state_b_f16 == NULL) {
        result = metal_fail(error, error_size, "failed to allocate Metal CLIP transformer state buffers");
        goto cleanup_clip_transformer;
    }

    const uint16_t *current_f16 = input_f16;
    uint16_t *next_f16 = state_a_f16;
    for (uint32_t block_index = 0u; block_index < block_count; ++block_index) {
        const int is_last = block_index + 1u == block_count;
        const uocr_metal_dense_output_type block_output_type = is_last ? output_type : UOCR_METAL_DENSE_OUTPUT_F16;
        void *block_out = is_last ? out : (void *)next_f16;
        if (!uocr_metal_context_clip_transformer_block_f16(ctx,
                                                           current_f16,
                                                           &blocks[block_index],
                                                           token_count,
                                                           block_output_type,
                                                           block_out,
                                                           error,
                                                           error_size)) {
            char detail[512];
            metal_copy_error_detail(detail, sizeof(detail), error);
            result = metal_fail(error,
                                error_size,
                                "failed to compute Metal CLIP transformer block %u: %s",
                                block_index,
                                detail);
            goto cleanup_clip_transformer;
        }
        if (!is_last) {
            current_f16 = next_f16;
            next_f16 = next_f16 == state_a_f16 ? state_b_f16 : state_a_f16;
        }
    }

    result = 1;

cleanup_clip_transformer:
    free(state_b_f16);
    free(state_a_f16);
    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_clip_sam_concat_f16(uocr_metal_context *ctx,
                                           const uint16_t *clip_tokens_f16,
                                           const uint16_t *sam_nchw_f16,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uocr_metal_dense_output_type output_type,
                                           void *out,
                                           char *error,
                                           size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || clip_tokens_f16 == NULL || sam_nchw_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal CLIP/SAM concat request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal CLIP/SAM concat output type %d", (int)output_type);
    }
    if (!((grid_w == UOCR_GLOBAL_GRID_QUERIES && grid_h == UOCR_GLOBAL_GRID_QUERIES) ||
          (grid_w == UOCR_LOCAL_GRID_QUERIES && grid_h == UOCR_LOCAL_GRID_QUERIES))) {
        return metal_fail(error, error_size, "unsupported Metal CLIP/SAM concat grid %ux%u", grid_w, grid_h);
    }
    if (UOCR_CLIP_HIDDEN_SIZE != 1024u || UOCR_SAM_FEATURE_CHANNELS != 1024u || UOCR_PROJECTOR_IN_SIZE != 2048u ||
        UOCR_PROJECTOR_IN_SIZE != UOCR_CLIP_HIDDEN_SIZE + UOCR_SAM_FEATURE_CHANNELS ||
        UOCR_CLIP_GLOBAL_TOKENS != 257u || UOCR_CLIP_LOCAL_TOKENS != 101u) {
        return metal_fail(error, error_size, "Metal CLIP/SAM concat constants are inconsistent");
    }

    uint64_t spatial = 0u;
    uint64_t clip_tokens = 0u;
    uint64_t clip_values = 0u;
    uint64_t sam_values = 0u;
    uint64_t output_values = 0u;
    uint64_t clip_bytes = 0u;
    uint64_t sam_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &spatial) ||
        !checked_add_u64(spatial, (uint64_t)UOCR_CLIP_CLASS_TOKENS, &clip_tokens) ||
        (clip_tokens != (uint64_t)UOCR_CLIP_GLOBAL_TOKENS && clip_tokens != (uint64_t)UOCR_CLIP_LOCAL_TOKENS) ||
        !checked_mul_u64(clip_tokens, (uint64_t)UOCR_CLIP_HIDDEN_SIZE, &clip_values) ||
        !checked_mul_u64(spatial, (uint64_t)UOCR_SAM_FEATURE_CHANNELS, &sam_values) ||
        !checked_mul_u64(spatial, (uint64_t)UOCR_PROJECTOR_IN_SIZE, &output_values) ||
        !checked_mul_u64(clip_values, 2u, &clip_bytes) ||
        !checked_mul_u64(sam_values, 2u, &sam_bytes) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes) ||
        spatial > (uint64_t)UINT32_MAX || clip_bytes > (uint64_t)SIZE_MAX || sam_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal CLIP/SAM concat byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (clip_bytes > max_buffer_length || sam_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal CLIP/SAM concat buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_clip_sam_concat_f16_to_f16" :
                                        "uocr_clip_sam_concat_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> clip = nil;
        id<MTLBuffer> sam = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        clip = [ctx->device newBufferWithBytes:clip_tokens_f16
                                        length:(NSUInteger)clip_bytes
                                       options:MTLResourceStorageModeShared];
        if (clip == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP/SAM concat CLIP buffer");
            goto cleanup_clip_sam_concat;
        }
        clip.label = @"uocr_clip_sam_concat_clip_tokens_f16";

        sam = [ctx->device newBufferWithBytes:sam_nchw_f16
                                       length:(NSUInteger)sam_bytes
                                      options:MTLResourceStorageModeShared];
        if (sam == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP/SAM concat SAM buffer");
            goto cleanup_clip_sam_concat;
        }
        sam.label = @"uocr_clip_sam_concat_sam_nchw_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal CLIP/SAM concat output buffer");
            goto cleanup_clip_sam_concat;
        }
        dst.label = @"uocr_clip_sam_concat_output";
        memset([dst contents], 0, (size_t)output_bytes);

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP/SAM concat command buffer");
            goto cleanup_clip_sam_concat;
        }
        cb.label = @"uocr_clip_sam_concat_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal CLIP/SAM concat command encoder");
            goto cleanup_clip_sam_concat;
        }

        uocr_metal_clip_sam_concat_params params;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.hidden_size = UOCR_CLIP_HIDDEN_SIZE;
        params.projector_in_size = UOCR_PROJECTOR_IN_SIZE;

        NSUInteger threads_x = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        if (threads_x > (NSUInteger)UOCR_PROJECTOR_IN_SIZE) {
            threads_x = (NSUInteger)UOCR_PROJECTOR_IN_SIZE;
        }
        if (threads_x == 0u) {
            result = metal_fail(error, error_size, "Metal CLIP/SAM concat has invalid threadgroup width");
            goto cleanup_clip_sam_concat;
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:clip offset:0u atIndex:0u];
        [enc setBuffer:sam offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)UOCR_PROJECTOR_IN_SIZE, (NSUInteger)spatial, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_x, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal CLIP/SAM concat command failed: %s", [description UTF8String]);
            goto cleanup_clip_sam_concat;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_clip_sam_concat:
        [dst release];
        [sam release];
        [clip release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_visual_projector_f16(uocr_metal_context *ctx,
                                            const uint16_t *input_f16,
                                            const uint16_t *weight_f16,
                                            const uint16_t *bias_f16,
                                            uint32_t n_rows,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_f16 == NULL || bias_f16 == NULL || out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal visual projector request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal visual projector output type %d", (int)output_type);
    }
    if (UOCR_PROJECTOR_IN_SIZE != 2048u || UOCR_HIDDEN_SIZE != 1280u) {
        return metal_fail(error, error_size, "Metal visual projector constants are inconsistent");
    }

    if (!uocr_metal_context_dense_f16(ctx,
                                      input_f16,
                                      weight_f16,
                                      bias_f16,
                                      n_rows,
                                      UOCR_PROJECTOR_IN_SIZE,
                                      UOCR_HIDDEN_SIZE,
                                      output_type,
                                      out,
                                      error,
                                      error_size)) {
        char detail[512];
        metal_copy_error_detail(detail, sizeof(detail), error);
        return metal_fail(error, error_size, "failed to compute Metal visual projector: %s", detail);
    }

    metal_clear_error(error, error_size);
    return 1;
}

static int metal_sam_transformer_block_has_weights(const uocr_metal_sam_transformer_block_f16 *block) {
    return block != NULL && block->norm1_weight_f16 != NULL && block->norm1_bias_f16 != NULL &&
           block->qkv_weight_f16 != NULL && block->qkv_bias_f16 != NULL &&
           block->proj_weight_f16 != NULL && block->proj_bias_f16 != NULL &&
           block->rel_pos_h_f16 != NULL && block->rel_pos_w_f16 != NULL &&
           block->rel_pos_h_length != 0u && block->rel_pos_w_length != 0u &&
           block->norm2_weight_f16 != NULL && block->norm2_bias_f16 != NULL &&
           block->mlp_lin1_weight_f16 != NULL && block->mlp_lin1_bias_f16 != NULL &&
           block->mlp_lin2_weight_f16 != NULL && block->mlp_lin2_bias_f16 != NULL;
}

static int metal_sam_transformer_activation_bytes(uint32_t grid_w,
                                                  uint32_t grid_h,
                                                  uint64_t *row_count,
                                                  uint64_t *hidden_bytes,
                                                  char *error,
                                                  size_t error_size) {
    if (grid_w == 0u || grid_h == 0u || grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE) {
        return metal_fail(error, error_size, "invalid Metal SAM transformer grid %ux%u", grid_w, grid_h);
    }
    uint64_t rows = 0u;
    uint64_t values = 0u;
    uint64_t bytes = 0u;
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &rows) || rows == 0u ||
        rows > (uint64_t)UINT32_MAX || !checked_mul_u64(rows, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &values) ||
        !checked_mul_u64(values, 2u, &bytes) || values > (uint64_t)UINT32_MAX || bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal SAM transformer byte-size overflow");
    }
    if (row_count != NULL) {
        *row_count = rows;
    }
    if (hidden_bytes != NULL) {
        *hidden_bytes = bytes;
    }
    return 1;
}

int uocr_metal_context_sam_transformer_block_f16(uocr_metal_context *ctx,
                                                 const uint16_t *input_bhwc_f16,
                                                 const uocr_metal_sam_transformer_block_f16 *block,
                                                 uint32_t grid_w,
                                                 uint32_t grid_h,
                                                 int use_global_attention,
                                                 uocr_metal_dense_output_type output_type,
                                                 void *out_bhwc,
                                                 char *error,
                                                 size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_bhwc_f16 == NULL || !metal_sam_transformer_block_has_weights(block) || out_bhwc == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM transformer block request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM transformer block output type %d", (int)output_type);
    }
    if (UOCR_SAM_HIDDEN_SIZE != 768u || UOCR_SAM_QKV_SIZE != 2304u || UOCR_SAM_ATTENTION_HEADS != 12u ||
        UOCR_SAM_HEAD_DIM != 64u || UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE ||
        UOCR_SAM_WINDOW_SIZE != 14u || UOCR_SAM_WINDOW_TOKENS != 196u || UOCR_SAM_MLP_INTERMEDIATE != 3072u) {
        return metal_fail(error, error_size, "Metal SAM transformer block constants are inconsistent");
    }

    uint64_t row_count_u64 = 0u;
    uint64_t hidden_bytes = 0u;
    if (!metal_sam_transformer_activation_bytes(grid_w, grid_h, &row_count_u64, &hidden_bytes, error, error_size)) {
        return 0;
    }
    const uint32_t row_count = (uint32_t)row_count_u64;

    uint32_t n_windows = 1u;
    uint32_t padded_w = grid_w;
    uint32_t padded_h = grid_h;
    uint64_t attention_rows_u64 = row_count_u64;
    uint64_t attention_values = 0u;
    uint64_t attention_bytes = 0u;
    if (!use_global_attention) {
        uint32_t windows_per_row = 0u;
        uint32_t windows_per_col = 0u;
        if (!sam_window_partition_geometry(grid_w,
                                           grid_h,
                                           &padded_w,
                                           &padded_h,
                                           &windows_per_row,
                                           &windows_per_col,
                                           &n_windows)) {
            return metal_fail(error, error_size, "invalid Metal SAM transformer window grid %ux%u", grid_w, grid_h);
        }
        if (!checked_mul_u64((uint64_t)n_windows, (uint64_t)UOCR_SAM_WINDOW_TOKENS, &attention_rows_u64)) {
            return metal_fail(error, error_size, "Metal SAM transformer window row-count overflow");
        }
    }
    if (!checked_mul_u64(attention_rows_u64, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &attention_values) ||
        !checked_mul_u64(attention_values, 2u, &attention_bytes) || attention_rows_u64 > (uint64_t)UINT32_MAX ||
        attention_values > (uint64_t)UINT32_MAX || attention_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal SAM transformer attention byte-size overflow");
    }
    const uint32_t attention_rows = (uint32_t)attention_rows_u64;

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (hidden_bytes > max_buffer_length || attention_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM transformer block buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    uint16_t *norm1_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *window_tokens_f16 = (uint16_t *)malloc((size_t)attention_bytes);
    uint16_t *q_f16 = (uint16_t *)malloc((size_t)attention_bytes);
    uint16_t *k_f16 = (uint16_t *)malloc((size_t)attention_bytes);
    uint16_t *v_f16 = (uint16_t *)malloc((size_t)attention_bytes);
    uint16_t *attention_f16 = (uint16_t *)malloc((size_t)attention_bytes);
    uint16_t *projected_windows_f16 = (uint16_t *)malloc((size_t)attention_bytes);
    uint16_t *attention_bhwc_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *residual1_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *norm2_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *mlp_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    if (norm1_f16 == NULL || window_tokens_f16 == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL ||
        attention_f16 == NULL || projected_windows_f16 == NULL || attention_bhwc_f16 == NULL || residual1_f16 == NULL ||
        norm2_f16 == NULL || mlp_f16 == NULL) {
        result = metal_fail(error, error_size, "failed to allocate Metal SAM transformer block scratch buffers");
        goto cleanup_sam_transformer_block;
    }

#define RUN_SAM_BLOCK_STEP(step_name, call_expr)                                                       \
    do {                                                                                                \
        if (!(call_expr)) {                                                                             \
            char detail[512];                                                                           \
            metal_copy_error_detail(detail, sizeof(detail), error);                                      \
            result = metal_fail(error, error_size, "failed to compute Metal SAM transformer block %s: %s", step_name, detail); \
            goto cleanup_sam_transformer_block;                                                         \
        }                                                                                               \
    } while (0)

    RUN_SAM_BLOCK_STEP("LayerNorm1",
                       uocr_metal_context_sam_layernorm_f16(ctx,
                                                             input_bhwc_f16,
                                                             block->norm1_weight_f16,
                                                             block->norm1_bias_f16,
                                                             row_count,
                                                             UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                                             norm1_f16,
                                                             error,
                                                             error_size));

    if (use_global_attention) {
        RUN_SAM_BLOCK_STEP("QKV",
                           uocr_metal_context_sam_qkv_f16(ctx,
                                                           norm1_f16,
                                                           block->qkv_weight_f16,
                                                           block->qkv_bias_f16,
                                                           row_count,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           q_f16,
                                                           k_f16,
                                                           v_f16,
                                                           error,
                                                           error_size));
        RUN_SAM_BLOCK_STEP("relative-position attention",
                           uocr_metal_context_sam_rel_pos_attention_f16(ctx,
                                                                        q_f16,
                                                                        k_f16,
                                                                        v_f16,
                                                                        block->rel_pos_h_f16,
                                                                        block->rel_pos_w_f16,
                                                                        1u,
                                                                        grid_w,
                                                                        grid_h,
                                                                        block->rel_pos_h_length,
                                                                        block->rel_pos_w_length,
                                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                                        attention_f16,
                                                                        error,
                                                                        error_size));
        RUN_SAM_BLOCK_STEP("attention projection residual",
                           uocr_metal_context_sam_attention_project_residual_f16(ctx,
                                                                                 attention_f16,
                                                                                 block->proj_weight_f16,
                                                                                 block->proj_bias_f16,
                                                                                 input_bhwc_f16,
                                                                                 row_count,
                                                                                 UOCR_METAL_DENSE_OUTPUT_F16,
                                                                                 residual1_f16,
                                                                                 error,
                                                                                 error_size));
    } else {
        uint32_t actual_windows = 0u;
        uint32_t actual_padded_w = 0u;
        uint32_t actual_padded_h = 0u;
        RUN_SAM_BLOCK_STEP("window partition",
                           uocr_metal_context_sam_window_partition_f16(ctx,
                                                                       norm1_f16,
                                                                       grid_w,
                                                                       grid_h,
                                                                       window_tokens_f16,
                                                                       &actual_windows,
                                                                       &actual_padded_w,
                                                                       &actual_padded_h,
                                                                       error,
                                                                       error_size));
        if (actual_windows != n_windows || actual_padded_w != padded_w || actual_padded_h != padded_h) {
            result = metal_fail(error, error_size, "Metal SAM transformer window geometry changed during partition");
            goto cleanup_sam_transformer_block;
        }
        RUN_SAM_BLOCK_STEP("window QKV",
                           uocr_metal_context_sam_qkv_f16(ctx,
                                                           window_tokens_f16,
                                                           block->qkv_weight_f16,
                                                           block->qkv_bias_f16,
                                                           attention_rows,
                                                           UOCR_METAL_DENSE_OUTPUT_F16,
                                                           q_f16,
                                                           k_f16,
                                                           v_f16,
                                                           error,
                                                           error_size));
        RUN_SAM_BLOCK_STEP("window relative-position attention",
                           uocr_metal_context_sam_rel_pos_attention_f16(ctx,
                                                                        q_f16,
                                                                        k_f16,
                                                                        v_f16,
                                                                        block->rel_pos_h_f16,
                                                                        block->rel_pos_w_f16,
                                                                        n_windows,
                                                                        UOCR_SAM_WINDOW_SIZE,
                                                                        UOCR_SAM_WINDOW_SIZE,
                                                                        block->rel_pos_h_length,
                                                                        block->rel_pos_w_length,
                                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                                        attention_f16,
                                                                        error,
                                                                        error_size));
        RUN_SAM_BLOCK_STEP("window output projection",
                           uocr_metal_context_dense_f16(ctx,
                                                        attention_f16,
                                                        block->proj_weight_f16,
                                                        block->proj_bias_f16,
                                                        attention_rows,
                                                        UOCR_SAM_HIDDEN_SIZE,
                                                        UOCR_SAM_HIDDEN_SIZE,
                                                        UOCR_METAL_DENSE_OUTPUT_F16,
                                                        projected_windows_f16,
                                                        error,
                                                        error_size));
        RUN_SAM_BLOCK_STEP("window unpartition",
                           uocr_metal_context_sam_window_unpartition_f16(ctx,
                                                                         projected_windows_f16,
                                                                         grid_w,
                                                                         grid_h,
                                                                         attention_bhwc_f16,
                                                                         error,
                                                                         error_size));
        RUN_SAM_BLOCK_STEP("attention residual",
                           uocr_metal_context_sam_residual_add_f16(ctx,
                                                                    input_bhwc_f16,
                                                                    attention_bhwc_f16,
                                                                    row_count,
                                                                    UOCR_METAL_DENSE_OUTPUT_F16,
                                                                    residual1_f16,
                                                                    error,
                                                                    error_size));
    }

    RUN_SAM_BLOCK_STEP("LayerNorm2",
                       uocr_metal_context_sam_layernorm_f16(ctx,
                                                             residual1_f16,
                                                             block->norm2_weight_f16,
                                                             block->norm2_bias_f16,
                                                             row_count,
                                                             UOCR_METAL_LAYERNORM_OUTPUT_F16,
                                                             norm2_f16,
                                                             error,
                                                             error_size));
    RUN_SAM_BLOCK_STEP("MLP",
                       uocr_metal_context_sam_mlp_f16(ctx,
                                                       norm2_f16,
                                                       block->mlp_lin1_weight_f16,
                                                       block->mlp_lin1_bias_f16,
                                                       block->mlp_lin2_weight_f16,
                                                       block->mlp_lin2_bias_f16,
                                                       row_count,
                                                       UOCR_METAL_DENSE_OUTPUT_F16,
                                                       mlp_f16,
                                                       error,
                                                       error_size));
    RUN_SAM_BLOCK_STEP("MLP residual",
                       uocr_metal_context_sam_residual_add_f16(ctx,
                                                                residual1_f16,
                                                                mlp_f16,
                                                                row_count,
                                                                output_type,
                                                                out_bhwc,
                                                                error,
                                                                error_size));

#undef RUN_SAM_BLOCK_STEP

    result = 1;

cleanup_sam_transformer_block:
    free(mlp_f16);
    free(norm2_f16);
    free(residual1_f16);
    free(attention_bhwc_f16);
    free(projected_windows_f16);
    free(attention_f16);
    free(v_f16);
    free(k_f16);
    free(q_f16);
    free(window_tokens_f16);
    free(norm1_f16);
    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_transformer_f16(uocr_metal_context *ctx,
                                           const uint16_t *input_bhwc_f16,
                                           const uocr_metal_sam_transformer_block_f16 *blocks,
                                           uint32_t block_count,
                                           uint32_t grid_w,
                                           uint32_t grid_h,
                                           uocr_metal_dense_output_type output_type,
                                           void *out_bhwc,
                                           char *error,
                                           size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_bhwc_f16 == NULL || blocks == NULL || out_bhwc == NULL) {
        return metal_fail(error, error_size, "invalid Metal SAM transformer request");
    }
    if (block_count != UOCR_SAM_BLOCKS) {
        return metal_fail(error, error_size, "invalid Metal SAM transformer block count %u", block_count);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM transformer output type %d", (int)output_type);
    }
    for (uint32_t block_index = 0u; block_index < block_count; ++block_index) {
        if (!metal_sam_transformer_block_has_weights(&blocks[block_index])) {
            return metal_fail(error, error_size, "invalid Metal SAM transformer block %u weights", block_index);
        }
    }

    uint64_t hidden_bytes = 0u;
    if (!metal_sam_transformer_activation_bytes(grid_w, grid_h, NULL, &hidden_bytes, error, error_size)) {
        return 0;
    }

    int result = 0;
    uint16_t *state_a_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    uint16_t *state_b_f16 = (uint16_t *)malloc((size_t)hidden_bytes);
    if (state_a_f16 == NULL || state_b_f16 == NULL) {
        result = metal_fail(error, error_size, "failed to allocate Metal SAM transformer state buffers");
        goto cleanup_sam_transformer;
    }

    const uint16_t *current_f16 = input_bhwc_f16;
    uint16_t *next_f16 = state_a_f16;
    for (uint32_t block_index = 0u; block_index < block_count; ++block_index) {
        const int is_last = block_index + 1u == block_count;
        const uocr_metal_dense_output_type block_output_type = is_last ? output_type : UOCR_METAL_DENSE_OUTPUT_F16;
        void *block_out = is_last ? out_bhwc : (void *)next_f16;
        if (!uocr_metal_context_sam_transformer_block_f16(ctx,
                                                          current_f16,
                                                          &blocks[block_index],
                                                          grid_w,
                                                          grid_h,
                                                          uocr_sam_block_uses_global_attention(block_index),
                                                          block_output_type,
                                                          block_out,
                                                          error,
                                                          error_size)) {
            char detail[512];
            metal_copy_error_detail(detail, sizeof(detail), error);
            result = metal_fail(error,
                                error_size,
                                "failed to compute Metal SAM transformer block %u: %s",
                                block_index,
                                detail);
            goto cleanup_sam_transformer;
        }
        if (!is_last) {
            current_f16 = next_f16;
            next_f16 = next_f16 == state_a_f16 ? state_b_f16 : state_a_f16;
        }
    }

    result = 1;

cleanup_sam_transformer:
    free(state_b_f16);
    free(state_a_f16);
    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

static int metal_context_sam_attention_f16(uocr_metal_context *ctx,
                                           const uint16_t *q_f16,
                                           const uint16_t *k_f16,
                                           const uint16_t *v_f16,
                                           uint32_t windows,
                                           uint32_t tokens_per_window,
                                           uocr_metal_dense_output_type output_type,
                                           void *out,
                                           const char *diagnostic_name,
                                           char *error,
                                           size_t error_size) {
    uint64_t attention_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t group_count = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)windows, (uint64_t)tokens_per_window, &attention_values) ||
        !checked_mul_u64(attention_values, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &attention_values) ||
        !checked_mul_u64(attention_values, 2u, &input_bytes) ||
        !checked_mul_u64(attention_values, output_element_bytes, &output_bytes) ||
        !checked_mul_u64((uint64_t)windows, (uint64_t)tokens_per_window, &group_count) ||
        !checked_mul_u64(group_count, (uint64_t)UOCR_SAM_ATTENTION_HEADS, &group_count) ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        group_count > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "%s byte-size overflow", diagnostic_name);
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "%s buffers exceed maxBufferLength %llu",
                          diagnostic_name,
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        /* The Metal kernel is an all-to-all attention primitive parameterized
         * by tokens_per_window. Global attention is represented as one window
         * spanning the whole SAM patch grid.
         */
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_window_attention_f16_to_f16" :
                                        "uocr_sam_window_attention_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> q_src = [ctx->device newBufferWithBytes:q_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (q_src == nil) {
            return metal_fail(error, error_size, "failed to allocate %s Q buffer", diagnostic_name);
        }
        q_src.label = @"uocr_sam_attention_q_f16";

        id<MTLBuffer> k_src = [ctx->device newBufferWithBytes:k_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (k_src == nil) {
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate %s K buffer", diagnostic_name);
        }
        k_src.label = @"uocr_sam_attention_k_f16";

        id<MTLBuffer> v_src = [ctx->device newBufferWithBytes:v_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (v_src == nil) {
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate %s V buffer", diagnostic_name);
        }
        v_src.label = @"uocr_sam_attention_v_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate %s output buffer", diagnostic_name);
        }
        dst.label = @"uocr_sam_attention_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create %s command buffer", diagnostic_name);
        }
        cb.label = @"uocr_sam_attention_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create %s command encoder", diagnostic_name);
        }

        uocr_metal_sam_window_attention_params params;
        params.windows = windows;
        params.tokens_per_window = tokens_per_window;
        params.heads = UOCR_SAM_ATTENTION_HEADS;
        params.head_dim = UOCR_SAM_HEAD_DIM;
        params.scale = 1.0f / sqrtf((float)UOCR_SAM_HEAD_DIM);
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width((NSUInteger)UOCR_SAM_HEAD_DIM,
                                                                            pipeline.maxTotalThreadsPerThreadgroup);
        if (threads_per_group < (NSUInteger)UOCR_SAM_HEAD_DIM) {
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "%s needs at least %u threads", diagnostic_name, UOCR_SAM_HEAD_DIM);
        }
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "%s threadgroup memory exceeds device limit", diagnostic_name);
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_src offset:0u atIndex:0u];
        [enc setBuffer:k_src offset:0u atIndex:1u];
        [enc setBuffer:v_src offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)group_count, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "%s command failed: %s", diagnostic_name, [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [v_src release];
        [k_src release];
        [q_src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_sam_window_attention_f16(uocr_metal_context *ctx,
                                                const uint16_t *q_f16,
                                                const uint16_t *k_f16,
                                                const uint16_t *v_f16,
                                                uint32_t n_windows,
                                                uocr_metal_dense_output_type output_type,
                                                void *out,
                                                char *error,
                                                size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL || out == NULL || n_windows == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM window attention request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM window attention output type %d", (int)output_type);
    }
    if (UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE ||
        UOCR_SAM_WINDOW_TOKENS != UOCR_SAM_WINDOW_SIZE * UOCR_SAM_WINDOW_SIZE) {
        return metal_fail(error, error_size, "Metal SAM window attention constants are inconsistent");
    }

    return metal_context_sam_attention_f16(ctx,
                                           q_f16,
                                           k_f16,
                                           v_f16,
                                           n_windows,
                                           UOCR_SAM_WINDOW_TOKENS,
                                           output_type,
                                           out,
                                           "Metal SAM window attention",
                                           error,
                                           error_size);
}

int uocr_metal_context_sam_global_attention_f16(uocr_metal_context *ctx,
                                                const uint16_t *q_f16,
                                                const uint16_t *k_f16,
                                                const uint16_t *v_f16,
                                                uint32_t grid_w,
                                                uint32_t grid_h,
                                                uocr_metal_dense_output_type output_type,
                                                void *out,
                                                char *error,
                                                size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL || out == NULL ||
        grid_w == 0u || grid_h == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM global attention request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM global attention output type %d", (int)output_type);
    }
    if (UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE ||
        UOCR_SAM_MAX_GRID_TOKENS != UOCR_SAM_MAX_GRID_SIZE * UOCR_SAM_MAX_GRID_SIZE) {
        return metal_fail(error, error_size, "Metal SAM global attention constants are inconsistent");
    }
    if (grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM global attention grid %ux%u exceeds max %ux%u",
                          grid_w,
                          grid_h,
                          UOCR_SAM_MAX_GRID_SIZE,
                          UOCR_SAM_MAX_GRID_SIZE);
    }
    uint64_t tokens = 0u;
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &tokens) || tokens == 0u ||
        tokens > (uint64_t)UOCR_SAM_MAX_GRID_TOKENS || tokens > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal SAM global attention token-count overflow");
    }

    return metal_context_sam_attention_f16(ctx,
                                           q_f16,
                                           k_f16,
                                           v_f16,
                                           1u,
                                           (uint32_t)tokens,
                                           output_type,
                                           out,
                                           "Metal SAM global attention",
                                           error,
                                           error_size);
}

int uocr_metal_context_sam_rel_pos_attention_f16(uocr_metal_context *ctx,
                                                 const uint16_t *q_f16,
                                                 const uint16_t *k_f16,
                                                 const uint16_t *v_f16,
                                                 const uint16_t *rel_pos_h_f16,
                                                 const uint16_t *rel_pos_w_f16,
                                                 uint32_t n_windows,
                                                 uint32_t grid_w,
                                                 uint32_t grid_h,
                                                 uint32_t rel_pos_h_length,
                                                 uint32_t rel_pos_w_length,
                                                 uocr_metal_dense_output_type output_type,
                                                 void *out,
                                                 char *error,
                                                 size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL || rel_pos_h_f16 == NULL ||
        rel_pos_w_f16 == NULL || out == NULL || n_windows == 0u || grid_w == 0u || grid_h == 0u ||
        rel_pos_h_length == 0u || rel_pos_w_length == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM relative-position attention request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal SAM relative-position attention output type %d",
                          (int)output_type);
    }
    if (UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE ||
        UOCR_SAM_MAX_GRID_TOKENS != UOCR_SAM_MAX_GRID_SIZE * UOCR_SAM_MAX_GRID_SIZE ||
        UOCR_SAM_MAX_REL_POS_SIZE != 2u * UOCR_SAM_MAX_GRID_SIZE - 1u ||
        UOCR_SAM_WINDOW_REL_POS_SIZE != 2u * UOCR_SAM_WINDOW_SIZE - 1u) {
        return metal_fail(error, error_size, "Metal SAM relative-position attention constants are inconsistent");
    }
    if (grid_w > UOCR_SAM_MAX_GRID_SIZE || grid_h > UOCR_SAM_MAX_GRID_SIZE ||
        rel_pos_h_length > UOCR_SAM_MAX_REL_POS_SIZE || rel_pos_w_length > UOCR_SAM_MAX_REL_POS_SIZE) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM relative-position attention dimensions exceed max grid/rel-pos limits");
    }

    uint64_t tokens = 0u;
    uint64_t attention_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t rel_h_values = 0u;
    uint64_t rel_w_values = 0u;
    uint64_t rel_h_bytes = 0u;
    uint64_t rel_w_bytes = 0u;
    uint64_t group_count = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)grid_w, (uint64_t)grid_h, &tokens) || tokens == 0u ||
        tokens > (uint64_t)UOCR_SAM_MAX_GRID_TOKENS || tokens > (uint64_t)UINT32_MAX ||
        !checked_mul_u64((uint64_t)n_windows, tokens, &attention_values) ||
        !checked_mul_u64(attention_values, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &attention_values) ||
        !checked_mul_u64(attention_values, 2u, &input_bytes) ||
        !checked_mul_u64(attention_values, output_element_bytes, &output_bytes) ||
        !checked_mul_u64((uint64_t)rel_pos_h_length, (uint64_t)UOCR_SAM_HEAD_DIM, &rel_h_values) ||
        !checked_mul_u64((uint64_t)rel_pos_w_length, (uint64_t)UOCR_SAM_HEAD_DIM, &rel_w_values) ||
        !checked_mul_u64(rel_h_values, 2u, &rel_h_bytes) ||
        !checked_mul_u64(rel_w_values, 2u, &rel_w_bytes) ||
        !checked_mul_u64((uint64_t)n_windows, tokens, &group_count) ||
        !checked_mul_u64(group_count, (uint64_t)UOCR_SAM_ATTENTION_HEADS, &group_count) ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        rel_h_bytes > (uint64_t)SIZE_MAX || rel_w_bytes > (uint64_t)SIZE_MAX ||
        group_count > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal SAM relative-position attention byte-size overflow");
    }

    const uint64_t target_rel_h_length = 2u * (uint64_t)grid_h - 1u;
    const uint64_t target_rel_w_length = 2u * (uint64_t)grid_w - 1u;
    if (target_rel_h_length > (uint64_t)UOCR_SAM_MAX_REL_POS_SIZE ||
        target_rel_w_length > (uint64_t)UOCR_SAM_MAX_REL_POS_SIZE) {
        return metal_fail(error, error_size, "Metal SAM relative-position attention target rel-pos overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        rel_h_bytes > max_buffer_length || rel_w_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM relative-position attention buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_rel_pos_attention_f16_to_f16" :
                                        "uocr_sam_rel_pos_attention_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> q_src = [ctx->device newBufferWithBytes:q_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (q_src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal SAM relative-position attention Q buffer");
        }
        q_src.label = @"uocr_sam_rel_pos_attention_q_f16";

        id<MTLBuffer> k_src = [ctx->device newBufferWithBytes:k_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (k_src == nil) {
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM relative-position attention K buffer");
        }
        k_src.label = @"uocr_sam_rel_pos_attention_k_f16";

        id<MTLBuffer> v_src = [ctx->device newBufferWithBytes:v_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (v_src == nil) {
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM relative-position attention V buffer");
        }
        v_src.label = @"uocr_sam_rel_pos_attention_v_f16";

        id<MTLBuffer> rel_h = [ctx->device newBufferWithBytes:rel_pos_h_f16
                                                      length:(NSUInteger)rel_h_bytes
                                                     options:MTLResourceStorageModeShared];
        if (rel_h == nil) {
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM relative-position H buffer");
        }
        rel_h.label = @"uocr_sam_rel_pos_h_f16";

        id<MTLBuffer> rel_w = [ctx->device newBufferWithBytes:rel_pos_w_f16
                                                      length:(NSUInteger)rel_w_bytes
                                                     options:MTLResourceStorageModeShared];
        if (rel_w == nil) {
            [rel_h release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM relative-position W buffer");
        }
        rel_w.label = @"uocr_sam_rel_pos_w_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [rel_w release];
            [rel_h release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM relative-position attention output buffer");
        }
        dst.label = @"uocr_sam_rel_pos_attention_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [rel_w release];
            [rel_h release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal SAM relative-position attention command buffer");
        }
        cb.label = @"uocr_sam_rel_pos_attention_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [rel_w release];
            [rel_h release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal SAM relative-position attention command encoder");
        }

        uocr_metal_sam_rel_pos_attention_params params;
        params.windows = n_windows;
        params.grid_width = grid_w;
        params.grid_height = grid_h;
        params.tokens_per_window = (uint32_t)tokens;
        params.heads = UOCR_SAM_ATTENTION_HEADS;
        params.head_dim = UOCR_SAM_HEAD_DIM;
        params.rel_pos_h_length = rel_pos_h_length;
        params.rel_pos_w_length = rel_pos_w_length;
        params.scale = 1.0f / sqrtf((float)UOCR_SAM_HEAD_DIM);
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width((NSUInteger)UOCR_SAM_HEAD_DIM,
                                                                            pipeline.maxTotalThreadsPerThreadgroup);
        if (threads_per_group < (NSUInteger)UOCR_SAM_HEAD_DIM) {
            [dst release];
            [rel_w release];
            [rel_h release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error,
                              error_size,
                              "Metal SAM relative-position attention needs at least %u threads",
                              UOCR_SAM_HEAD_DIM);
        }
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            [dst release];
            [rel_w release];
            [rel_h release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "Metal SAM relative-position attention threadgroup memory exceeds device limit");
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_src offset:0u atIndex:0u];
        [enc setBuffer:k_src offset:0u atIndex:1u];
        [enc setBuffer:v_src offset:0u atIndex:2u];
        [enc setBuffer:rel_h offset:0u atIndex:3u];
        [enc setBuffer:rel_w offset:0u atIndex:4u];
        [enc setBuffer:dst offset:0u atIndex:5u];
        [enc setBytes:&params length:sizeof(params) atIndex:6u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)group_count, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [rel_w release];
            [rel_h release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error,
                              error_size,
                              "Metal SAM relative-position attention command failed: %s",
                              [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [rel_w release];
        [rel_h release];
        [v_src release];
        [k_src release];
        [q_src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_sam_residual_add_f16(uocr_metal_context *ctx,
                                            const uint16_t *base_f16,
                                            const uint16_t *update_f16,
                                            uint32_t n_rows,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || base_f16 == NULL || update_f16 == NULL || out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM residual request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM residual output type %d", (int)output_type);
    }

    uint64_t value_count = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &value_count) ||
        !checked_mul_u64(value_count, 2u, &input_bytes) ||
        !checked_mul_u64(value_count, output_element_bytes, &output_bytes) || value_count > (uint64_t)UINT32_MAX ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal SAM residual byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM residual buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_residual_add_f16_to_f16" :
                                        "uocr_sam_residual_add_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> base = nil;
        id<MTLBuffer> update = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        base = [ctx->device newBufferWithBytes:base_f16
                                        length:(NSUInteger)input_bytes
                                       options:MTLResourceStorageModeShared];
        if (base == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM residual base buffer");
            goto cleanup_residual_add;
        }
        base.label = @"uocr_sam_residual_base_f16";

        update = [ctx->device newBufferWithBytes:update_f16
                                          length:(NSUInteger)input_bytes
                                         options:MTLResourceStorageModeShared];
        if (update == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM residual update buffer");
            goto cleanup_residual_add;
        }
        update.label = @"uocr_sam_residual_update_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM residual output buffer");
            goto cleanup_residual_add;
        }
        dst.label = @"uocr_sam_residual_output";
        memset([dst contents], 0, (size_t)output_bytes);

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM residual command buffer");
            goto cleanup_residual_add;
        }
        cb.label = @"uocr_sam_residual_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM residual command encoder");
            goto cleanup_residual_add;
        }

        uocr_metal_sam_residual_params params;
        params.n_rows = n_rows;
        params.hidden_size = UOCR_SAM_HIDDEN_SIZE;
        params.reserved0 = 0u;
        params.reserved1 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:base offset:0u atIndex:0u];
        [enc setBuffer:update offset:0u atIndex:1u];
        [enc setBuffer:dst offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)value_count, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error, error_size, "Metal SAM residual command failed: %s", [description UTF8String]);
            goto cleanup_residual_add;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_residual_add:
        [dst release];
        [update release];
        [base release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_attention_project_residual_f16(uocr_metal_context *ctx,
                                                          const uint16_t *attention_context_f16,
                                                          const uint16_t *proj_weight_f16,
                                                          const uint16_t *proj_bias_f16,
                                                          const uint16_t *residual_f16,
                                                          uint32_t n_rows,
                                                          uocr_metal_dense_output_type output_type,
                                                          void *out,
                                                          char *error,
                                                          size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || attention_context_f16 == NULL || proj_weight_f16 == NULL || proj_bias_f16 == NULL ||
        residual_f16 == NULL || out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM attention residual request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM attention residual output type %d", (int)output_type);
    }
    if (UOCR_SAM_ATTENTION_HEADS * UOCR_SAM_HEAD_DIM != UOCR_SAM_HIDDEN_SIZE) {
        return metal_fail(error, error_size, "Metal SAM attention residual constants are inconsistent");
    }

    uint64_t activation_values = 0u;
    uint64_t activation_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t bias_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &activation_values) ||
        !checked_mul_u64(activation_values, 2u, &activation_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_HIDDEN_SIZE, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_HIDDEN_SIZE, 2u, &bias_bytes) ||
        !checked_mul_u64(activation_values, output_element_bytes, &output_bytes) ||
        activation_values > (uint64_t)UINT32_MAX || activation_bytes > (uint64_t)SIZE_MAX ||
        weight_bytes > (uint64_t)SIZE_MAX || bias_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal SAM attention residual byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (activation_bytes > max_buffer_length || weight_bytes > max_buffer_length || bias_bytes > max_buffer_length ||
        output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM attention residual buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    int result = 0;
    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_sam_attention_project_residual_f16_to_f16" :
                                        "uocr_sam_attention_project_residual_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> bias = nil;
        id<MTLBuffer> residual = nil;
        id<MTLBuffer> dst = nil;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;

        src = [ctx->device newBufferWithBytes:attention_context_f16
                                       length:(NSUInteger)activation_bytes
                                      options:MTLResourceStorageModeShared];
        if (src == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM attention residual context buffer");
            goto cleanup_attention_residual;
        }
        src.label = @"uocr_sam_attention_project_context_f16";

        weight = [ctx->device newBufferWithBytes:proj_weight_f16
                                          length:(NSUInteger)weight_bytes
                                         options:MTLResourceStorageModeShared];
        if (weight == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM attention residual weight buffer");
            goto cleanup_attention_residual;
        }
        weight.label = @"uocr_sam_attention_project_weight_f16";

        bias = [ctx->device newBufferWithBytes:proj_bias_f16
                                        length:(NSUInteger)bias_bytes
                                       options:MTLResourceStorageModeShared];
        if (bias == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM attention residual bias buffer");
            goto cleanup_attention_residual;
        }
        bias.label = @"uocr_sam_attention_project_bias_f16";

        residual = [ctx->device newBufferWithBytes:residual_f16
                                            length:(NSUInteger)activation_bytes
                                           options:MTLResourceStorageModeShared];
        if (residual == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM attention residual shortcut buffer");
            goto cleanup_attention_residual;
        }
        residual.label = @"uocr_sam_attention_project_residual_f16";

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            result = metal_fail(error, error_size, "failed to allocate Metal SAM attention residual output buffer");
            goto cleanup_attention_residual;
        }
        dst.label = @"uocr_sam_attention_project_residual_output";
        memset([dst contents], 0, (size_t)output_bytes);

        cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM attention residual command buffer");
            goto cleanup_attention_residual;
        }
        cb.label = @"uocr_sam_attention_project_residual_command_buffer";

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            result = metal_fail(error, error_size, "failed to create Metal SAM attention residual command encoder");
            goto cleanup_attention_residual;
        }

        uocr_metal_sam_residual_params params;
        params.n_rows = n_rows;
        params.hidden_size = UOCR_SAM_HIDDEN_SIZE;
        params.reserved0 = 0u;
        params.reserved1 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:bias offset:0u atIndex:2u];
        [enc setBuffer:residual offset:0u atIndex:3u];
        [enc setBuffer:dst offset:0u atIndex:4u];
        [enc setBytes:&params length:sizeof(params) atIndex:5u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)activation_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            result = metal_fail(error,
                                error_size,
                                "Metal SAM attention residual command failed: %s",
                                [description UTF8String]);
            goto cleanup_attention_residual;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        result = 1;

    cleanup_attention_residual:
        [dst release];
        [residual release];
        [bias release];
        [weight release];
        [src release];
    }

    if (result) {
        metal_clear_error(error, error_size);
    }
    return result;
}

int uocr_metal_context_sam_mlp_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   const uint16_t *lin1_weight_f16,
                                   const uint16_t *lin1_bias_f16,
                                   const uint16_t *lin2_weight_f16,
                                   const uint16_t *lin2_bias_f16,
                                   uint32_t n_rows,
                                   uocr_metal_dense_output_type output_type,
                                   void *out,
                                   char *error,
                                   size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || lin1_weight_f16 == NULL || lin1_bias_f16 == NULL ||
        lin2_weight_f16 == NULL || lin2_bias_f16 == NULL || out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal SAM MLP request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal SAM MLP output type %d", (int)output_type);
    }
    if (UOCR_SAM_MLP_RATIO != 4u || UOCR_SAM_MLP_INTERMEDIATE != UOCR_SAM_MLP_RATIO * UOCR_SAM_HIDDEN_SIZE) {
        return metal_fail(error, error_size, "Metal SAM MLP constants are inconsistent");
    }

    uint64_t input_values = 0u;
    uint64_t intermediate_values = 0u;
    uint64_t weight_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t intermediate_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t lin1_bias_bytes = 0u;
    uint64_t lin2_bias_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_SAM_MLP_INTERMEDIATE, &intermediate_values) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_MLP_INTERMEDIATE, (uint64_t)UOCR_SAM_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64(intermediate_values, 2u, &intermediate_bytes) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_MLP_INTERMEDIATE, 2u, &lin1_bias_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_SAM_HIDDEN_SIZE, 2u, &lin2_bias_bytes) ||
        !checked_mul_u64(input_values, output_element_bytes, &output_bytes) ||
        input_bytes > (uint64_t)SIZE_MAX || intermediate_bytes > (uint64_t)SIZE_MAX ||
        weight_bytes > (uint64_t)SIZE_MAX || lin1_bias_bytes > (uint64_t)SIZE_MAX ||
        lin2_bias_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        input_values > (uint64_t)UINT32_MAX || intermediate_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal SAM MLP byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || intermediate_bytes > max_buffer_length ||
        weight_bytes > max_buffer_length || lin1_bias_bytes > max_buffer_length ||
        lin2_bias_bytes > max_buffer_length || output_bytes > max_buffer_length) {
        return metal_fail(error,
                          error_size,
                          "Metal SAM MLP buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> lin1_pipeline = metal_get_pipeline(ctx,
                                                                       "uocr_sam_mlp_lin1_gelu_f16",
                                                                       error,
                                                                       error_size);
        if (lin1_pipeline == nil) {
            return 0;
        }
        const char *lin2_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_sam_mlp_lin2_f16_to_f16" :
                                             "uocr_sam_mlp_lin2_f16_to_f32";
        id<MTLComputePipelineState> lin2_pipeline = metal_get_pipeline(ctx, lin2_function_name, error, error_size);
        if (lin2_pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> input = [ctx->device newBufferWithBytes:input_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (input == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal SAM MLP input buffer");
        }
        input.label = @"uocr_sam_mlp_input_f16";

        id<MTLBuffer> lin1_weight = [ctx->device newBufferWithBytes:lin1_weight_f16
                                                             length:(NSUInteger)weight_bytes
                                                            options:MTLResourceStorageModeShared];
        if (lin1_weight == nil) {
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM MLP lin1 weight buffer");
        }
        lin1_weight.label = @"uocr_sam_mlp_lin1_weight_f16";

        id<MTLBuffer> lin1_bias = [ctx->device newBufferWithBytes:lin1_bias_f16
                                                           length:(NSUInteger)lin1_bias_bytes
                                                          options:MTLResourceStorageModeShared];
        if (lin1_bias == nil) {
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM MLP lin1 bias buffer");
        }
        lin1_bias.label = @"uocr_sam_mlp_lin1_bias_f16";

        id<MTLBuffer> lin2_weight = [ctx->device newBufferWithBytes:lin2_weight_f16
                                                             length:(NSUInteger)weight_bytes
                                                            options:MTLResourceStorageModeShared];
        if (lin2_weight == nil) {
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM MLP lin2 weight buffer");
        }
        lin2_weight.label = @"uocr_sam_mlp_lin2_weight_f16";

        id<MTLBuffer> lin2_bias = [ctx->device newBufferWithBytes:lin2_bias_f16
                                                           length:(NSUInteger)lin2_bias_bytes
                                                          options:MTLResourceStorageModeShared];
        if (lin2_bias == nil) {
            [lin2_weight release];
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM MLP lin2 bias buffer");
        }
        lin2_bias.label = @"uocr_sam_mlp_lin2_bias_f16";

        id<MTLBuffer> mid = [ctx->device newBufferWithLength:(NSUInteger)intermediate_bytes
                                                     options:MTLResourceStorageModeShared];
        if (mid == nil) {
            [lin2_bias release];
            [lin2_weight release];
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM MLP intermediate buffer");
        }
        mid.label = @"uocr_sam_mlp_mid_f16";
        memset([mid contents], 0, (size_t)intermediate_bytes);

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [mid release];
            [lin2_bias release];
            [lin2_weight release];
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal SAM MLP output buffer");
        }
        dst.label = @"uocr_sam_mlp_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [mid release];
            [lin2_bias release];
            [lin2_weight release];
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal SAM MLP command buffer");
        }
        cb.label = @"uocr_sam_mlp_command_buffer";

        uocr_metal_sam_mlp_params params;
        params.n_rows = n_rows;
        params.hidden_size = UOCR_SAM_HIDDEN_SIZE;
        params.intermediate_size = UOCR_SAM_MLP_INTERMEDIATE;
        params.reserved = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [mid release];
            [lin2_bias release];
            [lin2_weight release];
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal SAM MLP lin1 command encoder");
        }
        const NSUInteger lin1_threads = metal_power2_threadgroup_width(256u, lin1_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:lin1_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:lin1_weight offset:0u atIndex:1u];
        [enc setBuffer:lin1_bias offset:0u atIndex:2u];
        [enc setBuffer:mid offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:lin1_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(lin1_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [mid release];
            [lin2_bias release];
            [lin2_weight release];
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal SAM MLP lin2 command encoder");
        }
        const NSUInteger lin2_threads = metal_power2_threadgroup_width(256u, lin2_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:lin2_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:lin2_weight offset:0u atIndex:1u];
        [enc setBuffer:lin2_bias offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:lin2_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)input_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(lin2_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [mid release];
            [lin2_bias release];
            [lin2_weight release];
            [lin1_bias release];
            [lin1_weight release];
            [input release];
            return metal_fail(error, error_size, "Metal SAM MLP command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [mid release];
        [lin2_bias release];
        [lin2_weight release];
        [lin1_bias release];
        [lin1_weight release];
        [input release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_final_rmsnorm_f16(uocr_metal_context *ctx,
                                         const uint16_t *input_f16,
                                         uint32_t n_rows,
                                         uocr_metal_rmsnorm_output_type output_type,
                                         void *out,
                                         char *error,
                                         size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal final RMSNorm request");
    }

    id<MTLBuffer> weight = nil;
    NSUInteger weight_offset = 0u;
    const uint64_t expected_payload_size = (uint64_t)UOCR_HIDDEN_SIZE * 2u;
    if (!metal_get_mapped_tensor_buffer(ctx,
                                        UOCR_TENSOR_ID_FINAL_NORM,
                                        expected_payload_size,
                                        &weight,
                                        &weight_offset,
                                        error,
                                        error_size)) {
        return 0;
    }

    return metal_context_rmsnorm_f16_with_weight_buffer(ctx,
                                                        input_f16,
                                                        weight,
                                                        weight_offset,
                                                        n_rows,
                                                        UOCR_HIDDEN_SIZE,
                                                        UOCR_RMS_NORM_EPS,
                                                        output_type,
                                                        out,
                                                        "Metal final RMSNorm",
                                                        error,
                                                        error_size);
}

int uocr_metal_context_lm_head_f16(uocr_metal_context *ctx,
                                   const uint16_t *input_f16,
                                   uint32_t n_rows,
                                   float *logits_out_f32,
                                   char *error,
                                   size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || logits_out_f32 == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal LM-head request");
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_VOCAB_SIZE, (uint64_t)UOCR_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_VOCAB_SIZE, &output_values) ||
        !checked_mul_u64(output_values, (uint64_t)sizeof(float), &output_bytes)) {
        return metal_fail(error, error_size, "Metal LM-head byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal LM-head buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    id<MTLBuffer> lm_head_weight = nil;
    NSUInteger lm_head_offset = 0u;
    if (!metal_get_mapped_tensor_buffer(ctx,
                                        UOCR_TENSOR_ID_LM_HEAD,
                                        weight_bytes,
                                        &lm_head_weight,
                                        &lm_head_offset,
                                        error,
                                        error_size)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_dense_f16_to_f32", error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal LM-head input buffer");
        }
        src.label = @"uocr_lm_head_input_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal LM-head logits buffer");
        }
        dst.label = @"uocr_lm_head_logits_f32";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal LM-head command buffer");
        }
        cb.label = @"uocr_lm_head_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal LM-head command encoder");
        }

        uocr_metal_dense_params params;
        params.input_rows = n_rows;
        params.in_features = UOCR_HIDDEN_SIZE;
        params.out_features = UOCR_VOCAB_SIZE;
        params.has_bias = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            [dst release];
            [src release];
            return metal_fail(error, error_size, "Metal LM-head threadgroup memory exceeds device limit");
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:lm_head_weight offset:lm_head_offset atIndex:1u];
        [enc setBuffer:lm_head_weight offset:lm_head_offset atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [src release];
            return metal_fail(error, error_size, "Metal LM-head command failed: %s", [description UTF8String]);
        }

        memcpy(logits_out_f32, [dst contents], (size_t)output_bytes);
        [dst release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

static uint32_t metal_no_repeat_candidate_count(uint32_t sequence_len, uint32_t ngram_size, uint32_t window) {
    if (ngram_size == 0u || sequence_len < ngram_size) {
        return 0u;
    }
    const uint32_t effective_window = window == 0u || window > sequence_len ? sequence_len : window;
    const uint32_t search_start = sequence_len - effective_window;
    const uint32_t search_end = sequence_len - ngram_size + 1u;
    return search_end > search_start ? search_end - search_start : 0u;
}

static void metal_no_repeat_pack_free(uocr_metal_no_repeat_ngram_pack *pack) {
    if (pack == NULL) {
        return;
    }
    free(pack->sequences);
    free(pack->row_offsets);
    free(pack->sequence_lengths);
    free(pack->ngram_sizes);
    free(pack->windows);
    memset(pack, 0, sizeof(*pack));
}

static int metal_no_repeat_pack_configs(const uocr_no_repeat_ngram_config *configs,
                                        uint32_t n_rows,
                                        uint32_t vocab_size,
                                        uocr_metal_no_repeat_ngram_pack *out_pack,
                                        char *error,
                                        size_t error_size) {
    if (out_pack == NULL) {
        return metal_fail(error, error_size, "invalid Metal no-repeat-ngram pack output");
    }
    memset(out_pack, 0, sizeof(*out_pack));
    if (configs == NULL || n_rows == 0u) {
        return 1;
    }

    uint64_t total_sequence_tokens = 0u;
    uint32_t max_candidates = 0u;
    for (uint32_t row = 0u; row < n_rows; ++row) {
        const uocr_no_repeat_ngram_config *config = &configs[row];
        if (config->ngram_size == 0u) {
            continue;
        }

        uint32_t banned_count = 0u;
        const int status = uocr_no_repeat_ngram_collect_banned(config->sequence,
                                                               config->sequence_len,
                                                               vocab_size,
                                                               config->ngram_size,
                                                               config->window,
                                                               NULL,
                                                               0u,
                                                               NULL,
                                                               0u,
                                                               &banned_count);
        if (status != UOCR_OK && status != UOCR_ERROR_OUT_OF_MEMORY) {
            return metal_fail(error, error_size, "failed to validate no-repeat-ngram row %u: status %d", row, status);
        }

        const uint32_t candidates = metal_no_repeat_candidate_count(config->sequence_len, config->ngram_size, config->window);
        if (candidates == 0u) {
            continue;
        }
        if (!checked_add_u64(total_sequence_tokens, (uint64_t)config->sequence_len, &total_sequence_tokens)) {
            return metal_fail(error, error_size, "Metal no-repeat-ngram sequence metadata overflow");
        }
        if (candidates > max_candidates) {
            max_candidates = candidates;
        }
    }

    if (max_candidates == 0u) {
        return 1;
    }
    if (total_sequence_tokens > (uint64_t)UINT32_MAX) {
        return metal_fail(error, error_size, "Metal no-repeat-ngram packed sequence is too large");
    }

    uint64_t row_bytes = 0u;
    uint64_t sequence_bytes = 0u;
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)sizeof(uint32_t), &row_bytes) ||
        !checked_mul_u64(total_sequence_tokens, (uint64_t)sizeof(int32_t), &sequence_bytes) ||
        row_bytes > (uint64_t)SIZE_MAX || sequence_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error, error_size, "Metal no-repeat-ngram pack byte-size overflow");
    }

    out_pack->row_offsets = (uint32_t *)calloc((size_t)n_rows, sizeof(uint32_t));
    out_pack->sequence_lengths = (uint32_t *)calloc((size_t)n_rows, sizeof(uint32_t));
    out_pack->ngram_sizes = (uint32_t *)calloc((size_t)n_rows, sizeof(uint32_t));
    out_pack->windows = (uint32_t *)calloc((size_t)n_rows, sizeof(uint32_t));
    out_pack->sequences = (int32_t *)malloc((size_t)sequence_bytes);
    if (out_pack->row_offsets == NULL || out_pack->sequence_lengths == NULL || out_pack->ngram_sizes == NULL ||
        out_pack->windows == NULL || out_pack->sequences == NULL) {
        metal_no_repeat_pack_free(out_pack);
        return metal_fail(error, error_size, "failed to allocate Metal no-repeat-ngram pack");
    }

    uint32_t cursor = 0u;
    for (uint32_t row = 0u; row < n_rows; ++row) {
        const uocr_no_repeat_ngram_config *config = &configs[row];
        const uint32_t candidates = metal_no_repeat_candidate_count(config->sequence_len, config->ngram_size, config->window);
        if (config->ngram_size == 0u || candidates == 0u) {
            continue;
        }
        out_pack->row_offsets[row] = cursor;
        out_pack->sequence_lengths[row] = config->sequence_len;
        out_pack->ngram_sizes[row] = config->ngram_size;
        out_pack->windows[row] = config->window;
        memcpy(out_pack->sequences + cursor, config->sequence, (size_t)config->sequence_len * sizeof(int32_t));
        cursor += config->sequence_len;
    }

    out_pack->total_sequence_tokens = (uint32_t)total_sequence_tokens;
    out_pack->max_candidates = max_candidates;
    out_pack->has_work = 1;
    return 1;
}

int uocr_metal_context_apply_no_repeat_ngram_f32(uocr_metal_context *ctx,
                                                 float *logits_f32,
                                                 uint32_t n_rows,
                                                 uint32_t vocab_size,
                                                 const uocr_no_repeat_ngram_config *configs_or_null,
                                                 char *error,
                                                 size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || vocab_size == 0u) {
        return metal_fail(error, error_size, "invalid Metal no-repeat-ngram request");
    }
    if (n_rows == 0u || configs_or_null == NULL) {
        return 1;
    }
    if (logits_f32 == NULL) {
        return metal_fail(error, error_size, "invalid Metal no-repeat-ngram logits buffer");
    }

    uocr_metal_no_repeat_ngram_pack pack;
    if (!metal_no_repeat_pack_configs(configs_or_null, n_rows, vocab_size, &pack, error, error_size)) {
        return 0;
    }
    if (!pack.has_work) {
        metal_no_repeat_pack_free(&pack);
        return 1;
    }

    uint64_t logits_values = 0u;
    uint64_t logits_bytes = 0u;
    uint64_t sequence_bytes = 0u;
    uint64_t row_bytes = 0u;
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)vocab_size, &logits_values) ||
        !checked_mul_u64(logits_values, (uint64_t)sizeof(float), &logits_bytes) ||
        !checked_mul_u64((uint64_t)pack.total_sequence_tokens, (uint64_t)sizeof(int32_t), &sequence_bytes) ||
        !checked_mul_u64((uint64_t)n_rows, (uint64_t)sizeof(uint32_t), &row_bytes) ||
        logits_bytes > (uint64_t)SIZE_MAX || sequence_bytes > (uint64_t)SIZE_MAX || row_bytes > (uint64_t)SIZE_MAX) {
        metal_no_repeat_pack_free(&pack);
        return metal_fail(error, error_size, "Metal no-repeat-ngram byte-size overflow");
    }
    if (logits_bytes > (uint64_t)ctx->device.maxBufferLength || sequence_bytes > (uint64_t)ctx->device.maxBufferLength ||
        row_bytes > (uint64_t)ctx->device.maxBufferLength) {
        metal_no_repeat_pack_free(&pack);
        return metal_fail(error,
                          error_size,
                          "Metal no-repeat-ngram buffers exceed maxBufferLength %llu",
                          (unsigned long long)ctx->device.maxBufferLength);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_no_repeat_ngram_f32", error, error_size);
        if (pipeline == nil) {
            metal_no_repeat_pack_free(&pack);
            return 0;
        }

        id<MTLBuffer> logits = [ctx->device newBufferWithBytes:logits_f32
                                                        length:(NSUInteger)logits_bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> sequences = [ctx->device newBufferWithBytes:pack.sequences
                                                           length:(NSUInteger)sequence_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> row_offsets = [ctx->device newBufferWithBytes:pack.row_offsets
                                                             length:(NSUInteger)row_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sequence_lengths = [ctx->device newBufferWithBytes:pack.sequence_lengths
                                                                  length:(NSUInteger)row_bytes
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ngram_sizes = [ctx->device newBufferWithBytes:pack.ngram_sizes
                                                             length:(NSUInteger)row_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> windows = [ctx->device newBufferWithBytes:pack.windows
                                                        length:(NSUInteger)row_bytes
                                                       options:MTLResourceStorageModeShared];
        if (logits == nil || sequences == nil || row_offsets == nil || sequence_lengths == nil || ngram_sizes == nil || windows == nil) {
            [windows release];
            [ngram_sizes release];
            [sequence_lengths release];
            [row_offsets release];
            [sequences release];
            [logits release];
            metal_no_repeat_pack_free(&pack);
            return metal_fail(error, error_size, "failed to allocate Metal no-repeat-ngram buffers");
        }
        logits.label = @"uocr_no_repeat_logits_f32";
        sequences.label = @"uocr_no_repeat_sequences";
        row_offsets.label = @"uocr_no_repeat_row_offsets";
        sequence_lengths.label = @"uocr_no_repeat_sequence_lengths";
        ngram_sizes.label = @"uocr_no_repeat_ngram_sizes";
        windows.label = @"uocr_no_repeat_windows";

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [windows release];
            [ngram_sizes release];
            [sequence_lengths release];
            [row_offsets release];
            [sequences release];
            [logits release];
            metal_no_repeat_pack_free(&pack);
            return metal_fail(error, error_size, "failed to create Metal no-repeat-ngram command buffer");
        }
        cb.label = @"uocr_no_repeat_ngram_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [windows release];
            [ngram_sizes release];
            [sequence_lengths release];
            [row_offsets release];
            [sequences release];
            [logits release];
            metal_no_repeat_pack_free(&pack);
            return metal_fail(error, error_size, "failed to create Metal no-repeat-ngram command encoder");
        }

        uocr_metal_no_repeat_ngram_params params;
        params.rows = n_rows;
        params.vocab_size = vocab_size;
        params.max_candidates = pack.max_candidates;
        params.reserved0 = 0u;

        NSUInteger threads_x = (NSUInteger)pack.max_candidates;
        if (threads_x > 64u) {
            threads_x = 64u;
        }
        if (threads_x > pipeline.maxTotalThreadsPerThreadgroup) {
            threads_x = pipeline.maxTotalThreadsPerThreadgroup;
        }
        if (threads_x == 0u) {
            [enc endEncoding];
            [windows release];
            [ngram_sizes release];
            [sequence_lengths release];
            [row_offsets release];
            [sequences release];
            [logits release];
            metal_no_repeat_pack_free(&pack);
            return metal_fail(error, error_size, "Metal no-repeat-ngram pipeline has no available threads");
        }
        const NSUInteger groups_x = ((NSUInteger)pack.max_candidates + threads_x - 1u) / threads_x;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:logits offset:0u atIndex:0u];
        [enc setBuffer:sequences offset:0u atIndex:1u];
        [enc setBuffer:row_offsets offset:0u atIndex:2u];
        [enc setBuffer:sequence_lengths offset:0u atIndex:3u];
        [enc setBuffer:ngram_sizes offset:0u atIndex:4u];
        [enc setBuffer:windows offset:0u atIndex:5u];
        [enc setBytes:&params length:sizeof(params) atIndex:6u];
        [enc dispatchThreadgroups:MTLSizeMake(groups_x, (NSUInteger)n_rows, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_x, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [windows release];
            [ngram_sizes release];
            [sequence_lengths release];
            [row_offsets release];
            [sequences release];
            [logits release];
            metal_no_repeat_pack_free(&pack);
            return metal_fail(error, error_size, "Metal no-repeat-ngram command failed: %s", [description UTF8String]);
        }

        memcpy(logits_f32, [logits contents], (size_t)logits_bytes);
        [windows release];
        [ngram_sizes release];
        [sequence_lengths release];
        [row_offsets release];
        [sequences release];
        [logits release];
    }

    metal_no_repeat_pack_free(&pack);
    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_argmax_f32(uocr_metal_context *ctx,
                                  const float *logits_f32,
                                  uint32_t n_rows,
                                  uint32_t vocab_size,
                                  uint32_t *token_ids_out,
                                  float *scores_out_f32_or_null,
                                  char *error,
                                  size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || logits_f32 == NULL || token_ids_out == NULL || n_rows == 0u || vocab_size == 0u) {
        return metal_fail(error, error_size, "invalid Metal argmax request");
    }

    uint64_t logits_values = 0u;
    uint64_t logits_bytes = 0u;
    uint64_t ids_bytes = 0u;
    uint64_t scores_bytes = 0u;
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)vocab_size, &logits_values) ||
        !checked_mul_u64(logits_values, (uint64_t)sizeof(float), &logits_bytes) ||
        !checked_mul_u64((uint64_t)n_rows, (uint64_t)sizeof(uint32_t), &ids_bytes) ||
        !checked_mul_u64((uint64_t)n_rows, (uint64_t)sizeof(float), &scores_bytes)) {
        return metal_fail(error, error_size, "Metal argmax byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (logits_bytes > max_buffer_length || ids_bytes > max_buffer_length || scores_bytes > max_buffer_length ||
        logits_bytes > (uint64_t)SIZE_MAX || ids_bytes > (uint64_t)SIZE_MAX || scores_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal argmax buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_argmax_f32", error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> logits = [ctx->device newBufferWithBytes:logits_f32
                                                        length:(NSUInteger)logits_bytes
                                                       options:MTLResourceStorageModeShared];
        if (logits == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal argmax logits buffer");
        }
        logits.label = @"uocr_argmax_logits_f32";

        id<MTLBuffer> ids = [ctx->device newBufferWithLength:(NSUInteger)ids_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores = [ctx->device newBufferWithLength:(NSUInteger)scores_bytes options:MTLResourceStorageModeShared];
        if (ids == nil || scores == nil) {
            [scores release];
            [ids release];
            [logits release];
            return metal_fail(error, error_size, "failed to allocate Metal argmax output buffers");
        }
        ids.label = @"uocr_argmax_token_ids";
        scores.label = @"uocr_argmax_scores";
        memset([ids contents], 0xff, (size_t)ids_bytes);
        memset([scores contents], 0, (size_t)scores_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [scores release];
            [ids release];
            [logits release];
            return metal_fail(error, error_size, "failed to create Metal argmax command buffer");
        }
        cb.label = @"uocr_argmax_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [scores release];
            [ids release];
            [logits release];
            return metal_fail(error, error_size, "failed to create Metal argmax command encoder");
        }

        uocr_metal_argmax_params params;
        params.rows = n_rows;
        params.vocab_size = vocab_size;
        params.reserved0 = 0u;
        params.reserved1 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        uint64_t total_threadgroup_bytes = 0u;
        if (!checked_mul_u64(threadgroup_bytes, 2u, &total_threadgroup_bytes) ||
            total_threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            [scores release];
            [ids release];
            [logits release];
            return metal_fail(error, error_size, "Metal argmax threadgroup memory exceeds device limit");
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:logits offset:0u atIndex:0u];
        [enc setBuffer:ids offset:0u atIndex:1u];
        [enc setBuffer:scores offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(uint32_t) atIndex:1u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [scores release];
            [ids release];
            [logits release];
            return metal_fail(error, error_size, "Metal argmax command failed: %s", [description UTF8String]);
        }

        memcpy(token_ids_out, [ids contents], (size_t)ids_bytes);
        if (scores_out_f32_or_null != NULL) {
            memcpy(scores_out_f32_or_null, [scores contents], (size_t)scores_bytes);
        }
        [scores release];
        [ids release];
        [logits release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_select_greedy_f32(uocr_metal_context *ctx,
                                         float *logits_f32,
                                         uint32_t n_rows,
                                         uint32_t vocab_size,
                                         const uocr_no_repeat_ngram_config *no_repeat_or_null,
                                         uint32_t *token_ids_out,
                                         float *scores_out_f32_or_null,
                                         char *error,
                                         size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || logits_f32 == NULL || token_ids_out == NULL || n_rows == 0u || vocab_size == 0u) {
        return metal_fail(error, error_size, "invalid Metal greedy selection request");
    }

    if (no_repeat_or_null != NULL &&
        !uocr_metal_context_apply_no_repeat_ngram_f32(ctx,
                                                      logits_f32,
                                                      n_rows,
                                                      vocab_size,
                                                      no_repeat_or_null,
                                                      error,
                                                      error_size)) {
        char detail[512];
        (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
        return metal_fail(error, error_size, "failed to apply no-repeat-ngram bans before argmax: %s", detail);
    }

    return uocr_metal_context_argmax_f32(ctx,
                                         logits_f32,
                                         n_rows,
                                         vocab_size,
                                         token_ids_out,
                                         scores_out_f32_or_null,
                                         error,
                                         error_size);
}

int uocr_metal_context_select_next_token_f16(uocr_metal_context *ctx,
                                             const uint16_t *hidden_f16,
                                             uint32_t n_rows,
                                             const uocr_no_repeat_ngram_config *no_repeat_or_null,
                                             uint16_t *normed_scratch_f16,
                                             float *logits_scratch_f32,
                                             uint32_t *token_ids_out,
                                             float *scores_out_f32_or_null,
                                             char *error,
                                             size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || hidden_f16 == NULL || normed_scratch_f16 == NULL || logits_scratch_f32 == NULL ||
        token_ids_out == NULL || n_rows == 0u) {
        return metal_fail(error, error_size, "invalid Metal next-token selection request");
    }

    uint64_t hidden_values = 0u;
    uint64_t logits_values = 0u;
    if (!checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_HIDDEN_SIZE, &hidden_values) ||
        !checked_mul_u64((uint64_t)n_rows, (uint64_t)UOCR_VOCAB_SIZE, &logits_values) ||
        hidden_values > (uint64_t)SIZE_MAX / sizeof(uint16_t) ||
        logits_values > (uint64_t)SIZE_MAX / sizeof(float)) {
        return metal_fail(error, error_size, "Metal next-token selection scratch byte-size overflow");
    }

    if (!uocr_metal_context_final_rmsnorm_f16(ctx,
                                             hidden_f16,
                                             n_rows,
                                             UOCR_METAL_RMSNORM_OUTPUT_F16,
                                             normed_scratch_f16,
                                             error,
                                             error_size)) {
        char detail[512];
        (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
        return metal_fail(error, error_size, "failed to apply final RMSNorm before next-token selection: %s", detail);
    }

    if (!uocr_metal_context_lm_head_f16(ctx,
                                        normed_scratch_f16,
                                        n_rows,
                                        logits_scratch_f32,
                                        error,
                                        error_size)) {
        char detail[512];
        (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
        return metal_fail(error, error_size, "failed to compute LM-head logits before next-token selection: %s", detail);
    }

    if (!uocr_metal_context_select_greedy_f32(ctx,
                                             logits_scratch_f32,
                                             n_rows,
                                             UOCR_VOCAB_SIZE,
                                             no_repeat_or_null,
                                             token_ids_out,
                                             scores_out_f32_or_null,
                                             error,
                                             error_size)) {
        char detail[512];
        (void)snprintf(detail, sizeof(detail), "%s", (error != NULL && error[0] != '\0') ? error : "unknown error");
        return metal_fail(error, error_size, "failed to greedily select next token: %s", detail);
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_dense_f16(uocr_metal_context *ctx,
                                 const uint16_t *input_f16,
                                 const uint16_t *weight_f16,
                                 const uint16_t *bias_f16_or_null,
                                 uint32_t input_rows,
                                 uint32_t in_features,
                                 uint32_t out_features,
                                 uocr_metal_dense_output_type output_type,
                                 void *out,
                                 char *error,
                                 size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_f16 == NULL || out == NULL ||
        input_rows == 0u || in_features == 0u || out_features == 0u) {
        return metal_fail(error, error_size, "invalid Metal dense request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal dense output type %d", (int)output_type);
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t bias_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)input_rows, (uint64_t)in_features, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)out_features, (uint64_t)in_features, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)out_features, 2u, &bias_bytes) ||
        !checked_mul_u64((uint64_t)input_rows, (uint64_t)out_features, &output_values) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal dense byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || bias_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX ||
        bias_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal dense buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_dense_f16_to_f16" :
                                        "uocr_dense_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal dense input buffer");
        }
        src.label = @"uocr_dense_input_f16";

        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:weight_f16
                                                        length:(NSUInteger)weight_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal dense weight buffer");
        }
        weight.label = @"uocr_dense_weight_f16";

        id<MTLBuffer> bias = nil;
        id<MTLBuffer> bias_arg = weight;
        if (bias_f16_or_null != NULL) {
            bias = [ctx->device newBufferWithBytes:bias_f16_or_null
                                            length:(NSUInteger)bias_bytes
                                           options:MTLResourceStorageModeShared];
            if (bias == nil) {
                [weight release];
                [src release];
                return metal_fail(error, error_size, "failed to allocate Metal dense bias buffer");
            }
            bias.label = @"uocr_dense_bias_f16";
            bias_arg = bias;
        }

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal dense output buffer");
        }
        dst.label = @"uocr_dense_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal dense command buffer");
        }
        cb.label = @"uocr_dense_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal dense command encoder");
        }

        uocr_metal_dense_params params;
        params.input_rows = input_rows;
        params.in_features = in_features;
        params.out_features = out_features;
        params.has_bias = bias_f16_or_null != NULL ? 1u : 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:bias_arg offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "Metal dense command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [bias release];
        [weight release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_dense_q8_0(uocr_metal_context *ctx,
                                  const uint16_t *input_f16,
                                  const void *weight_q8_0,
                                  const uint16_t *bias_f16_or_null,
                                  uint32_t input_rows,
                                  uint32_t logical_in_features,
                                  uint32_t physical_in_features,
                                  uint32_t out_features,
                                  uocr_metal_dense_output_type output_type,
                                  void *out,
                                  char *error,
                                  size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_q8_0 == NULL || out == NULL || input_rows == 0u ||
        logical_in_features == 0u || physical_in_features == 0u || out_features == 0u) {
        return metal_fail(error, error_size, "invalid Metal Q8_0 dense request");
    }
    if (logical_in_features > physical_in_features || (physical_in_features % 32u) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal Q8_0 dense widths logical=%u physical=%u",
                          logical_in_features,
                          physical_in_features);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal Q8_0 dense output type %d", (int)output_type);
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_row_size = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t bias_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)input_rows, (uint64_t)logical_in_features, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)(physical_in_features / 32u), 34u, &weight_row_size) ||
        !checked_mul_u64((uint64_t)out_features, weight_row_size, &weight_bytes) ||
        !checked_mul_u64((uint64_t)out_features, 2u, &bias_bytes) ||
        !checked_mul_u64((uint64_t)input_rows, (uint64_t)out_features, &output_values) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal Q8_0 dense byte-size overflow");
    }
    if (weight_row_size == 0u || weight_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal Q8_0 dense row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || bias_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX ||
        bias_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX || output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal Q8_0 dense buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_dense_q8_0_to_f16" :
                                        "uocr_dense_q8_0_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal Q8_0 dense input buffer");
        }
        src.label = @"uocr_dense_q8_input_f16";

        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:weight_q8_0
                                                        length:(NSUInteger)weight_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal Q8_0 dense weight buffer");
        }
        weight.label = @"uocr_dense_weight_q8_0";

        id<MTLBuffer> bias = nil;
        id<MTLBuffer> bias_arg = weight;
        if (bias_f16_or_null != NULL) {
            bias = [ctx->device newBufferWithBytes:bias_f16_or_null
                                            length:(NSUInteger)bias_bytes
                                           options:MTLResourceStorageModeShared];
            if (bias == nil) {
                [weight release];
                [src release];
                return metal_fail(error, error_size, "failed to allocate Metal Q8_0 dense bias buffer");
            }
            bias.label = @"uocr_dense_q8_bias_f16";
            bias_arg = bias;
        }

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal Q8_0 dense output buffer");
        }
        dst.label = @"uocr_dense_q8_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal Q8_0 dense command buffer");
        }
        cb.label = @"uocr_dense_q8_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal Q8_0 dense command encoder");
        }

        uocr_metal_dense_q8_params params;
        params.input_rows = input_rows;
        params.logical_in_features = logical_in_features;
        params.physical_in_features = physical_in_features;
        params.out_features = out_features;
        params.weight_row_size = (uint32_t)weight_row_size;
        params.has_bias = bias_f16_or_null != NULL ? 1u : 0u;
        params.reserved0 = 0u;
        params.reserved1 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:bias_arg offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "Metal Q8_0 dense command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [bias release];
        [weight release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_dense_q4_k(uocr_metal_context *ctx,
                                  const uint16_t *input_f16,
                                  const void *weight_q4_k,
                                  const uint16_t *bias_f16_or_null,
                                  uint32_t input_rows,
                                  uint32_t logical_in_features,
                                  uint32_t physical_in_features,
                                  uint32_t out_features,
                                  uocr_metal_dense_output_type output_type,
                                  void *out,
                                  char *error,
                                  size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || weight_q4_k == NULL || out == NULL || input_rows == 0u ||
        logical_in_features == 0u || physical_in_features == 0u || out_features == 0u) {
        return metal_fail(error, error_size, "invalid Metal Q4_K dense request");
    }
    if (logical_in_features > physical_in_features ||
        (physical_in_features % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal Q4_K dense widths logical=%u physical=%u",
                          logical_in_features,
                          physical_in_features);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal Q4_K dense output type %d", (int)output_type);
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_row_size = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t bias_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)input_rows, (uint64_t)logical_in_features, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)(physical_in_features / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &weight_row_size) ||
        !checked_mul_u64((uint64_t)out_features, weight_row_size, &weight_bytes) ||
        !checked_mul_u64((uint64_t)out_features, 2u, &bias_bytes) ||
        !checked_mul_u64((uint64_t)input_rows, (uint64_t)out_features, &output_values) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal Q4_K dense byte-size overflow");
    }
    if (weight_row_size == 0u || weight_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal Q4_K dense row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || bias_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX ||
        bias_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX || output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal Q4_K dense buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_dense_q4_k_to_f16" :
                                        "uocr_dense_q4_k_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal Q4_K dense input buffer");
        }
        src.label = @"uocr_dense_q4_input_f16";

        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:weight_q4_k
                                                        length:(NSUInteger)weight_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal Q4_K dense weight buffer");
        }
        weight.label = @"uocr_dense_weight_q4_k";

        id<MTLBuffer> bias = nil;
        id<MTLBuffer> bias_arg = weight;
        if (bias_f16_or_null != NULL) {
            bias = [ctx->device newBufferWithBytes:bias_f16_or_null
                                            length:(NSUInteger)bias_bytes
                                           options:MTLResourceStorageModeShared];
            if (bias == nil) {
                [weight release];
                [src release];
                return metal_fail(error, error_size, "failed to allocate Metal Q4_K dense bias buffer");
            }
            bias.label = @"uocr_dense_q4_bias_f16";
            bias_arg = bias;
        }

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal Q4_K dense output buffer");
        }
        dst.label = @"uocr_dense_q4_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal Q4_K dense command buffer");
        }
        cb.label = @"uocr_dense_q4_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal Q4_K dense command encoder");
        }

        uocr_metal_dense_q4_params params;
        params.input_rows = input_rows;
        params.logical_in_features = logical_in_features;
        params.physical_in_features = physical_in_features;
        params.out_features = out_features;
        params.weight_row_size = (uint32_t)weight_row_size;
        params.has_bias = bias_f16_or_null != NULL ? 1u : 0u;
        params.reserved0 = 0u;
        params.reserved1 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:bias_arg offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [bias release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "Metal Q4_K dense command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [bias release];
        [weight release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_attention_qkvo_f16(uocr_metal_context *ctx,
                                          const uint16_t *input_f16,
                                          const uint16_t *q_weight_f16,
                                          const uint16_t *k_weight_f16,
                                          const uint16_t *v_weight_f16,
                                          const uint16_t *o_weight_f16,
                                          uint32_t n_tokens,
                                          uocr_metal_dense_output_type output_type,
                                          void *q_out,
                                          void *k_out,
                                          void *v_out,
                                          void *o_out,
                                          char *error,
                                          size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || q_weight_f16 == NULL || k_weight_f16 == NULL ||
        v_weight_f16 == NULL || o_weight_f16 == NULL || q_out == NULL || k_out == NULL ||
        v_out == NULL || o_out == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal attention projection request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal attention projection output type %d", (int)output_type);
    }

    uint64_t values_per_projection = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t total_projection_values = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &values_per_projection) ||
        !checked_mul_u64(values_per_projection, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, (uint64_t)UOCR_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64(values_per_projection, output_element_bytes, &output_bytes) ||
        !checked_mul_u64(values_per_projection, 4u, &total_projection_values)) {
        return metal_fail(error, error_size, "Metal attention projection byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        total_projection_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal attention projection buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_attention_qkvo_f16_to_f16" :
                                        "uocr_attention_qkvo_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:input_f16
                                                     length:(NSUInteger)input_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal attention projection input buffer");
        }
        src.label = @"uocr_attention_projection_input_f16";

        id<MTLBuffer> q_weight = [ctx->device newBufferWithBytes:q_weight_f16
                                                          length:(NSUInteger)weight_bytes
                                                         options:MTLResourceStorageModeShared];
        if (q_weight == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal Q projection weight buffer");
        }
        q_weight.label = @"uocr_attention_q_weight_f16";

        id<MTLBuffer> k_weight = [ctx->device newBufferWithBytes:k_weight_f16
                                                          length:(NSUInteger)weight_bytes
                                                         options:MTLResourceStorageModeShared];
        if (k_weight == nil) {
            [q_weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal K projection weight buffer");
        }
        k_weight.label = @"uocr_attention_k_weight_f16";

        id<MTLBuffer> v_weight = [ctx->device newBufferWithBytes:v_weight_f16
                                                          length:(NSUInteger)weight_bytes
                                                         options:MTLResourceStorageModeShared];
        if (v_weight == nil) {
            [k_weight release];
            [q_weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal V projection weight buffer");
        }
        v_weight.label = @"uocr_attention_v_weight_f16";

        id<MTLBuffer> o_weight = [ctx->device newBufferWithBytes:o_weight_f16
                                                          length:(NSUInteger)weight_bytes
                                                         options:MTLResourceStorageModeShared];
        if (o_weight == nil) {
            [v_weight release];
            [k_weight release];
            [q_weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal O projection weight buffer");
        }
        o_weight.label = @"uocr_attention_o_weight_f16";

        id<MTLBuffer> q_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> v_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> o_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (q_dst == nil || k_dst == nil || v_dst == nil || o_dst == nil) {
            [o_dst release];
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [o_weight release];
            [v_weight release];
            [k_weight release];
            [q_weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal attention projection output buffers");
        }
        q_dst.label = @"uocr_attention_q_output";
        k_dst.label = @"uocr_attention_k_output";
        v_dst.label = @"uocr_attention_v_output";
        o_dst.label = @"uocr_attention_o_output";
        memset([q_dst contents], 0, (size_t)output_bytes);
        memset([k_dst contents], 0, (size_t)output_bytes);
        memset([v_dst contents], 0, (size_t)output_bytes);
        memset([o_dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [o_dst release];
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [o_weight release];
            [v_weight release];
            [k_weight release];
            [q_weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal attention projection command buffer");
        }
        cb.label = @"uocr_attention_projection_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [o_dst release];
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [o_weight release];
            [v_weight release];
            [k_weight release];
            [q_weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal attention projection command encoder");
        }

        uocr_metal_attention_projection_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.projection_count = 4u;
        params.reserved = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:q_weight offset:0u atIndex:1u];
        [enc setBuffer:k_weight offset:0u atIndex:2u];
        [enc setBuffer:v_weight offset:0u atIndex:3u];
        [enc setBuffer:o_weight offset:0u atIndex:4u];
        [enc setBuffer:q_dst offset:0u atIndex:5u];
        [enc setBuffer:k_dst offset:0u atIndex:6u];
        [enc setBuffer:v_dst offset:0u atIndex:7u];
        [enc setBuffer:o_dst offset:0u atIndex:8u];
        [enc setBytes:&params length:sizeof(params) atIndex:9u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)total_projection_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [o_dst release];
            [v_dst release];
            [k_dst release];
            [q_dst release];
            [o_weight release];
            [v_weight release];
            [k_weight release];
            [q_weight release];
            [src release];
            return metal_fail(error, error_size, "Metal attention projection command failed: %s", [description UTF8String]);
        }

        memcpy(q_out, [q_dst contents], (size_t)output_bytes);
        memcpy(k_out, [k_dst contents], (size_t)output_bytes);
        memcpy(v_out, [v_dst contents], (size_t)output_bytes);
        memcpy(o_out, [o_dst contents], (size_t)output_bytes);
        [o_dst release];
        [v_dst release];
        [k_dst release];
        [q_dst release];
        [o_weight release];
        [v_weight release];
        [k_weight release];
        [q_weight release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_attention_output_residual_f16(uocr_metal_context *ctx,
                                                     const uint16_t *attention_context_f16,
                                                     const uint16_t *o_weight_f16,
                                                     const uint16_t *residual_f16,
                                                     uint32_t n_tokens,
                                                     uocr_metal_dense_output_type output_type,
                                                     void *out,
                                                     char *error,
                                                     size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || attention_context_f16 == NULL || o_weight_f16 == NULL || residual_f16 == NULL || out == NULL ||
        n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal attention output request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal attention output type %d", (int)output_type);
    }

    uint64_t activation_values = 0u;
    uint64_t activation_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &activation_values) ||
        !checked_mul_u64(activation_values, 2u, &activation_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, (uint64_t)UOCR_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64(activation_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal attention output byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (activation_bytes > max_buffer_length || weight_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        activation_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        activation_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal attention output buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_attention_output_residual_f16_to_f16" :
                                        "uocr_attention_output_residual_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> src = [ctx->device newBufferWithBytes:attention_context_f16
                                                     length:(NSUInteger)activation_bytes
                                                    options:MTLResourceStorageModeShared];
        if (src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal attention output input buffer");
        }
        src.label = @"uocr_attention_output_context_f16";

        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:o_weight_f16
                                                        length:(NSUInteger)weight_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal attention output weight buffer");
        }
        weight.label = @"uocr_attention_output_weight_f16";

        id<MTLBuffer> residual = [ctx->device newBufferWithBytes:residual_f16
                                                          length:(NSUInteger)activation_bytes
                                                         options:MTLResourceStorageModeShared];
        if (residual == nil) {
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal attention residual buffer");
        }
        residual.label = @"uocr_attention_output_residual_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [residual release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to allocate Metal attention output buffer");
        }
        dst.label = @"uocr_attention_output_residual_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [residual release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal attention output command buffer");
        }
        cb.label = @"uocr_attention_output_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [residual release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "failed to create Metal attention output command encoder");
        }

        uocr_metal_attention_output_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.reserved0 = 0u;
        params.reserved1 = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:src offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:residual offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)activation_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [residual release];
            [weight release];
            [src release];
            return metal_fail(error, error_size, "Metal attention output command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [residual release];
        [weight release];
        [src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

static int metal_context_swiglu_f16(uocr_metal_context *ctx,
                                    const uint16_t *input_f16,
                                    const uint16_t *gate_weight_f16,
                                    const uint16_t *up_weight_f16,
                                    const uint16_t *down_weight_f16,
                                    const uint16_t *residual_f16_or_null,
                                    uint32_t n_tokens,
                                    uint32_t intermediate_size,
                                    uocr_metal_dense_output_type output_type,
                                    void *out,
                                    const char *op_label,
                                    char *error,
                                    size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || gate_weight_f16 == NULL || up_weight_f16 == NULL ||
        down_weight_f16 == NULL || out == NULL || n_tokens == 0u || intermediate_size == 0u || op_label == NULL) {
        return metal_fail(error, error_size, "invalid Metal %s request", op_label != NULL ? op_label : "SwiGLU");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal %s output type %d", op_label, (int)output_type);
    }

    uint64_t hidden_values = 0u;
    uint64_t intermediate_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t intermediate_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &hidden_values) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)intermediate_size, &intermediate_values) ||
        !checked_mul_u64(hidden_values, 2u, &input_bytes) ||
        !checked_mul_u64(intermediate_values, 2u, &intermediate_bytes) ||
        !checked_mul_u64((uint64_t)intermediate_size, (uint64_t)UOCR_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64(hidden_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal %s byte-size overflow", op_label);
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || intermediate_bytes > max_buffer_length ||
        weight_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || intermediate_bytes > (uint64_t)SIZE_MAX ||
        weight_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        hidden_values > (uint64_t)UINT32_MAX || intermediate_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal %s buffers exceed maxBufferLength %llu",
                          op_label,
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_dense_swiglu_gate_up_f16",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_dense_swiglu_down_f16_to_f16" :
                                             "uocr_dense_swiglu_down_f16_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> input = [ctx->device newBufferWithBytes:input_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (input == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal %s input buffer", op_label);
        }
        input.label = @"uocr_dense_swiglu_input_f16";

        id<MTLBuffer> gate_weight = [ctx->device newBufferWithBytes:gate_weight_f16
                                                             length:(NSUInteger)weight_bytes
                                                            options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal %s gate weight buffer", op_label);
        }
        gate_weight.label = @"uocr_dense_swiglu_gate_weight_f16";

        id<MTLBuffer> up_weight = [ctx->device newBufferWithBytes:up_weight_f16
                                                           length:(NSUInteger)weight_bytes
                                                          options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal %s up weight buffer", op_label);
        }
        up_weight.label = @"uocr_dense_swiglu_up_weight_f16";

        id<MTLBuffer> down_weight = [ctx->device newBufferWithBytes:down_weight_f16
                                                             length:(NSUInteger)weight_bytes
                                                            options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal %s down weight buffer", op_label);
        }
        down_weight.label = @"uocr_dense_swiglu_down_weight_f16";

        id<MTLBuffer> residual = nil;
        id<MTLBuffer> residual_arg = down_weight;
        if (residual_f16_or_null != NULL) {
            residual = [ctx->device newBufferWithBytes:residual_f16_or_null
                                                length:(NSUInteger)input_bytes
                                               options:MTLResourceStorageModeShared];
            if (residual == nil) {
                [down_weight release];
                [up_weight release];
                [gate_weight release];
                [input release];
                return metal_fail(error, error_size, "failed to allocate Metal %s residual buffer", op_label);
            }
            residual.label = @"uocr_dense_swiglu_residual_f16";
            residual_arg = residual;
        }

        id<MTLBuffer> mid = [ctx->device newBufferWithLength:(NSUInteger)intermediate_bytes
                                                     options:MTLResourceStorageModeShared];
        if (mid == nil) {
            [residual release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal %s intermediate buffer", op_label);
        }
        mid.label = @"uocr_dense_swiglu_mid_f16";
        memset([mid contents], 0, (size_t)intermediate_bytes);

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [mid release];
            [residual release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal %s output buffer", op_label);
        }
        dst.label = @"uocr_dense_swiglu_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [mid release];
            [residual release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal %s command buffer", op_label);
        }
        cb.label = @"uocr_dense_swiglu_command_buffer";

        uocr_metal_dense_swiglu_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.intermediate_size = intermediate_size;
        params.has_residual = residual_f16_or_null != NULL ? 1u : 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [mid release];
            [residual release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal %s gate/up command encoder", op_label);
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:gate_weight offset:0u atIndex:1u];
        [enc setBuffer:up_weight offset:0u atIndex:2u];
        [enc setBuffer:mid offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [mid release];
            [residual release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal %s down command encoder", op_label);
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:down_weight offset:0u atIndex:1u];
        [enc setBuffer:residual_arg offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)hidden_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [mid release];
            [residual release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "Metal %s command failed: %s", op_label, [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [mid release];
        [residual release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [input release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_dense_swiglu_f16(uocr_metal_context *ctx,
                                        const uint16_t *input_f16,
                                        const uint16_t *gate_weight_f16,
                                        const uint16_t *up_weight_f16,
                                        const uint16_t *down_weight_f16,
                                        const uint16_t *residual_f16_or_null,
                                        uint32_t n_tokens,
                                        uocr_metal_dense_output_type output_type,
                                        void *out,
                                        char *error,
                                        size_t error_size) {
    return metal_context_swiglu_f16(ctx,
                                    input_f16,
                                    gate_weight_f16,
                                    up_weight_f16,
                                    down_weight_f16,
                                    residual_f16_or_null,
                                    n_tokens,
                                    UOCR_DENSE_LAYER0_INTERMEDIATE,
                                    output_type,
                                    out,
                                    "dense SwiGLU",
                                    error,
                                    error_size);
}

int uocr_metal_context_moe_shared_experts_f16(uocr_metal_context *ctx,
                                              const uint16_t *input_f16,
                                              const uint16_t *shared_gate_weight_f16,
                                              const uint16_t *shared_up_weight_f16,
                                              const uint16_t *shared_down_weight_f16,
                                              uint32_t n_tokens,
                                              uocr_metal_dense_output_type output_type,
                                              void *out,
                                              char *error,
                                              size_t error_size) {
    return metal_context_swiglu_f16(ctx,
                                    input_f16,
                                    shared_gate_weight_f16,
                                    shared_up_weight_f16,
                                    shared_down_weight_f16,
                                    NULL,
                                    n_tokens,
                                    UOCR_MOE_SHARED_INTERMEDIATE,
                                    output_type,
                                    out,
                                    "MoE shared experts",
                                    error,
                                    error_size);
}

int uocr_metal_context_moe_shared_experts_q8_0(uocr_metal_context *ctx,
                                               const uint16_t *input_f16,
                                               const void *shared_gate_weight_q8_0,
                                               const void *shared_up_weight_q8_0,
                                               const void *shared_down_weight_q8_0,
                                               uint32_t physical_hidden_features,
                                               uint32_t physical_intermediate_features,
                                               uint32_t n_tokens,
                                               uocr_metal_dense_output_type output_type,
                                               void *out,
                                               char *error,
                                               size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || shared_gate_weight_q8_0 == NULL || shared_up_weight_q8_0 == NULL ||
        shared_down_weight_q8_0 == NULL || out == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal MoE shared Q8_0 experts request");
    }
    if (physical_hidden_features < UOCR_HIDDEN_SIZE || physical_intermediate_features < UOCR_MOE_SHARED_INTERMEDIATE ||
        (physical_hidden_features % 32u) != 0u || (physical_intermediate_features % 32u) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE shared Q8_0 widths hidden=%u intermediate=%u",
                          physical_hidden_features,
                          physical_intermediate_features);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal MoE shared Q8_0 output type %d", (int)output_type);
    }

    uint64_t hidden_values = 0u;
    uint64_t intermediate_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t intermediate_bytes = 0u;
    uint64_t gate_row_size = 0u;
    uint64_t gate_weight_bytes = 0u;
    uint64_t down_row_size = 0u;
    uint64_t down_weight_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &hidden_values) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_MOE_SHARED_INTERMEDIATE, &intermediate_values) ||
        !checked_mul_u64(hidden_values, 2u, &input_bytes) ||
        !checked_mul_u64(intermediate_values, 2u, &intermediate_bytes) ||
        !checked_mul_u64((uint64_t)(physical_hidden_features / 32u), 34u, &gate_row_size) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_SHARED_INTERMEDIATE, gate_row_size, &gate_weight_bytes) ||
        !checked_mul_u64((uint64_t)(physical_intermediate_features / 32u), 34u, &down_row_size) ||
        !checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, down_row_size, &down_weight_bytes) ||
        !checked_mul_u64(hidden_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE shared Q8_0 byte-size overflow");
    }
    if (gate_row_size == 0u || down_row_size == 0u || gate_row_size > UINT32_MAX || down_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal MoE shared Q8_0 row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || intermediate_bytes > max_buffer_length ||
        gate_weight_bytes > max_buffer_length || down_weight_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX || intermediate_bytes > (uint64_t)SIZE_MAX ||
        gate_weight_bytes > (uint64_t)SIZE_MAX || down_weight_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX || hidden_values > (uint64_t)UINT32_MAX ||
        intermediate_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE shared Q8_0 buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_dense_swiglu_gate_up_q8_0",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_dense_q8_0_to_f16" :
                                             "uocr_dense_q8_0_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> input = [ctx->device newBufferWithBytes:input_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (input == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal MoE shared Q8_0 input buffer");
        }
        input.label = @"uocr_moe_shared_q8_input_f16";

        id<MTLBuffer> gate_weight = [ctx->device newBufferWithBytes:shared_gate_weight_q8_0
                                                             length:(NSUInteger)gate_weight_bytes
                                                            options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal MoE shared Q8_0 gate weight buffer");
        }
        gate_weight.label = @"uocr_moe_shared_gate_weight_q8_0";

        id<MTLBuffer> up_weight = [ctx->device newBufferWithBytes:shared_up_weight_q8_0
                                                           length:(NSUInteger)gate_weight_bytes
                                                          options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal MoE shared Q8_0 up weight buffer");
        }
        up_weight.label = @"uocr_moe_shared_up_weight_q8_0";

        id<MTLBuffer> down_weight = [ctx->device newBufferWithBytes:shared_down_weight_q8_0
                                                             length:(NSUInteger)down_weight_bytes
                                                            options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal MoE shared Q8_0 down weight buffer");
        }
        down_weight.label = @"uocr_moe_shared_down_weight_q8_0";

        id<MTLBuffer> mid = [ctx->device newBufferWithLength:(NSUInteger)intermediate_bytes
                                                     options:MTLResourceStorageModeShared];
        if (mid == nil) {
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal MoE shared Q8_0 intermediate buffer");
        }
        mid.label = @"uocr_moe_shared_q8_mid_f16";
        memset([mid contents], 0, (size_t)intermediate_bytes);

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [mid release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal MoE shared Q8_0 output buffer");
        }
        dst.label = @"uocr_moe_shared_q8_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [mid release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal MoE shared Q8_0 command buffer");
        }
        cb.label = @"uocr_moe_shared_q8_command_buffer";

        uocr_metal_dense_q8_params gate_params;
        gate_params.input_rows = n_tokens;
        gate_params.logical_in_features = UOCR_HIDDEN_SIZE;
        gate_params.physical_in_features = physical_hidden_features;
        gate_params.out_features = UOCR_MOE_SHARED_INTERMEDIATE;
        gate_params.weight_row_size = (uint32_t)gate_row_size;
        gate_params.has_bias = 0u;
        gate_params.reserved0 = 0u;
        gate_params.reserved1 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [mid release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal MoE shared Q8_0 gate/up command encoder");
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:gate_weight offset:0u atIndex:1u];
        [enc setBuffer:up_weight offset:0u atIndex:2u];
        [enc setBuffer:mid offset:0u atIndex:3u];
        [enc setBytes:&gate_params length:sizeof(gate_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        uocr_metal_dense_q8_params down_params;
        down_params.input_rows = n_tokens;
        down_params.logical_in_features = UOCR_MOE_SHARED_INTERMEDIATE;
        down_params.physical_in_features = physical_intermediate_features;
        down_params.out_features = UOCR_HIDDEN_SIZE;
        down_params.weight_row_size = (uint32_t)down_row_size;
        down_params.has_bias = 0u;
        down_params.reserved0 = 0u;
        down_params.reserved1 = 0u;

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [mid release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal MoE shared Q8_0 down command encoder");
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:down_weight offset:0u atIndex:1u];
        [enc setBuffer:down_weight offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&down_params length:sizeof(down_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)hidden_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [mid release];
            [down_weight release];
            [up_weight release];
            [gate_weight release];
            [input release];
            return metal_fail(error, error_size, "Metal MoE shared Q8_0 command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [mid release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [input release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_router_f16(uocr_metal_context *ctx,
                                      const uint16_t *input_f16,
                                      const uint16_t *router_weight_f16,
                                      uint32_t n_tokens,
                                      float *logits_out_f32_or_null,
                                      float *probs_out_f32_or_null,
                                      uint32_t *top_expert_ids_out,
                                      float *top_weights_out,
                                      char *error,
                                      size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || router_weight_f16 == NULL || top_expert_ids_out == NULL ||
        top_weights_out == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal MoE router request");
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t router_values = 0u;
    uint64_t router_bytes = 0u;
    uint64_t top_values = 0u;
    uint64_t top_ids_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_ROUTED_EXPERTS, (uint64_t)UOCR_HIDDEN_SIZE, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_ROUTED_EXPERTS, &router_values) ||
        !checked_mul_u64(router_values, (uint64_t)sizeof(float), &router_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_MOE_TOP_K, &top_values) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(uint32_t), &top_ids_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(float), &top_weights_bytes)) {
        return metal_fail(error, error_size, "Metal MoE router byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || weight_bytes > max_buffer_length || router_bytes > max_buffer_length ||
        top_ids_bytes > max_buffer_length || top_weights_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX || router_bytes > (uint64_t)SIZE_MAX ||
        top_ids_bytes > (uint64_t)SIZE_MAX || top_weights_bytes > (uint64_t)SIZE_MAX ||
        router_values > (uint64_t)UINT32_MAX || top_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE router buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> logits_pipeline = metal_get_pipeline(ctx,
                                                                         "uocr_moe_router_logits_f16_to_f32",
                                                                         error,
                                                                         error_size);
        if (logits_pipeline == nil) {
            return 0;
        }
        id<MTLComputePipelineState> topk_pipeline = metal_get_pipeline(ctx,
                                                                       "uocr_moe_router_softmax_topk_f32",
                                                                       error,
                                                                       error_size);
        if (topk_pipeline == nil) {
            return 0;
        }
        if (topk_pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)UOCR_ROUTED_EXPERTS) {
            return metal_fail(error,
                              error_size,
                              "Metal MoE router top-k pipeline supports only %llu threads; need %u",
                              (unsigned long long)topk_pipeline.maxTotalThreadsPerThreadgroup,
                              (unsigned)UOCR_ROUTED_EXPERTS);
        }

        id<MTLBuffer> input = [ctx->device newBufferWithBytes:input_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (input == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal MoE router input buffer");
        }
        input.label = @"uocr_moe_router_input_f16";

        id<MTLBuffer> weight = [ctx->device newBufferWithBytes:router_weight_f16
                                                        length:(NSUInteger)weight_bytes
                                                       options:MTLResourceStorageModeShared];
        if (weight == nil) {
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal MoE router weight buffer");
        }
        weight.label = @"uocr_moe_router_weight_f16";

        id<MTLBuffer> logits = [ctx->device newBufferWithLength:(NSUInteger)router_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> probs = [ctx->device newBufferWithLength:(NSUInteger)router_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> top_ids = [ctx->device newBufferWithLength:(NSUInteger)top_ids_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> top_weights = [ctx->device newBufferWithLength:(NSUInteger)top_weights_bytes options:MTLResourceStorageModeShared];
        if (logits == nil || probs == nil || top_ids == nil || top_weights == nil) {
            [top_weights release];
            [top_ids release];
            [probs release];
            [logits release];
            [weight release];
            [input release];
            return metal_fail(error, error_size, "failed to allocate Metal MoE router output buffers");
        }
        logits.label = @"uocr_moe_router_logits_f32";
        probs.label = @"uocr_moe_router_probs_f32";
        top_ids.label = @"uocr_moe_router_top_ids_u32";
        top_weights.label = @"uocr_moe_router_top_weights_f32";
        memset([logits contents], 0, (size_t)router_bytes);
        memset([probs contents], 0, (size_t)router_bytes);
        memset([top_ids contents], 0, (size_t)top_ids_bytes);
        memset([top_weights contents], 0, (size_t)top_weights_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [top_weights release];
            [top_ids release];
            [probs release];
            [logits release];
            [weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal MoE router command buffer");
        }
        cb.label = @"uocr_moe_router_command_buffer";

        uocr_metal_moe_router_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.experts = UOCR_ROUTED_EXPERTS;
        params.top_k = UOCR_MOE_TOP_K;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [top_weights release];
            [top_ids release];
            [probs release];
            [logits release];
            [weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal MoE router logits command encoder");
        }
        const NSUInteger logits_threads = metal_power2_threadgroup_width(256u, logits_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:logits_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:weight offset:0u atIndex:1u];
        [enc setBuffer:logits offset:0u atIndex:2u];
        [enc setBytes:&params length:sizeof(params) atIndex:3u];
        [enc setThreadgroupMemoryLength:logits_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)router_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(logits_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [top_weights release];
            [top_ids release];
            [probs release];
            [logits release];
            [weight release];
            [input release];
            return metal_fail(error, error_size, "failed to create Metal MoE router top-k command encoder");
        }
        const NSUInteger topk_threads = metal_power2_threadgroup_width(64u, topk_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:topk_pipeline];
        [enc setBuffer:logits offset:0u atIndex:0u];
        [enc setBuffer:probs offset:0u atIndex:1u];
        [enc setBuffer:top_ids offset:0u atIndex:2u];
        [enc setBuffer:top_weights offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        const NSUInteger topk_scratch_bytes = ((NSUInteger)UOCR_ROUTED_EXPERTS + topk_threads) * sizeof(float) +
                                              (NSUInteger)UOCR_ROUTED_EXPERTS * sizeof(uint32_t);
        [enc setThreadgroupMemoryLength:topk_scratch_bytes atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(topk_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [top_weights release];
            [top_ids release];
            [probs release];
            [logits release];
            [weight release];
            [input release];
            return metal_fail(error, error_size, "Metal MoE router command failed: %s", [description UTF8String]);
        }

        if (logits_out_f32_or_null != NULL) {
            memcpy(logits_out_f32_or_null, [logits contents], (size_t)router_bytes);
        }
        if (probs_out_f32_or_null != NULL) {
            memcpy(probs_out_f32_or_null, [probs contents], (size_t)router_bytes);
        }
        memcpy(top_expert_ids_out, [top_ids contents], (size_t)top_ids_bytes);
        memcpy(top_weights_out, [top_weights contents], (size_t)top_weights_bytes);

        [top_weights release];
        [top_ids release];
        [probs release];
        [logits release];
        [weight release];
        [input release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_decode_f16(uocr_metal_context *ctx,
                                                       const uint16_t *input_f16,
                                                       const uint32_t *top_expert_ids,
                                                       const float *top_weights_f32,
                                                       const uint16_t *selected_gate_weight_f16,
                                                       const uint16_t *selected_up_weight_f16,
                                                       const uint16_t *selected_down_weight_f16,
                                                       uocr_metal_dense_output_type output_type,
                                                       void *out,
                                                       char *error,
                                                       size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        selected_gate_weight_f16 == NULL || selected_up_weight_f16 == NULL || selected_down_weight_f16 == NULL ||
        out == NULL) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert decode request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert output type %d",
                          (int)output_type);
    }
    for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
        if (top_expert_ids[rank] >= UOCR_ROUTED_EXPERTS) {
            return metal_fail(error,
                              error_size,
                              "invalid Metal MoE selected expert id %u at rank %u",
                              top_expert_ids[rank],
                              rank);
        }
        for (uint32_t prev = 0u; prev < rank; ++prev) {
            if (top_expert_ids[prev] == top_expert_ids[rank]) {
                return metal_fail(error,
                                  error_size,
                                  "duplicate Metal MoE selected expert id %u",
                                  top_expert_ids[rank]);
            }
        }
        if (!isfinite(top_weights_f32[rank])) {
            return metal_fail(error, error_size, "invalid Metal MoE selected expert weight at rank %u", rank);
        }
    }

    uint64_t hidden_values = UOCR_HIDDEN_SIZE;
    uint64_t input_bytes = 0u;
    uint64_t selected_weight_values = 0u;
    uint64_t selected_weight_bytes = 0u;
    uint64_t intermediate_values = 0u;
    uint64_t intermediate_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64(hidden_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE, &intermediate_values) ||
        !checked_mul_u64(intermediate_values, (uint64_t)UOCR_HIDDEN_SIZE, &selected_weight_values) ||
        !checked_mul_u64(selected_weight_values, 2u, &selected_weight_bytes) ||
        !checked_mul_u64(intermediate_values, 2u, &intermediate_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64(hidden_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || selected_weight_bytes > max_buffer_length ||
        intermediate_bytes > max_buffer_length || top_weights_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX ||
        selected_weight_bytes > (uint64_t)SIZE_MAX || intermediate_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_selected_gate_up_f16",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_selected_down_sum_f16_to_f16" :
                                             "uocr_moe_selected_down_sum_f16_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_selected_input_f16";

        gate_weight = [ctx->device newBufferWithBytes:selected_gate_weight_f16
                                               length:(NSUInteger)selected_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_selected_gate_weight_f16";

        up_weight = [ctx->device newBufferWithBytes:selected_up_weight_f16
                                             length:(NSUInteger)selected_weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_selected_up_weight_f16";

        down_weight = [ctx->device newBufferWithBytes:selected_down_weight_f16
                                               length:(NSUInteger)selected_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_selected_down_weight_f16";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_selected_top_weights_f32";

        mid = [ctx->device newBufferWithLength:(NSUInteger)intermediate_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_selected_mid_f16";
        memset([mid contents], 0, (size_t)intermediate_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_selected_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_selected_command_buffer";

        uocr_metal_moe_selected_params params;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        params.top_k = UOCR_MOE_TOP_K;
        params.reserved = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:gate_weight offset:0u atIndex:1u];
        [enc setBuffer:up_weight offset:0u atIndex:2u];
        [enc setBuffer:mid offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:down_weight offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)UOCR_HIDDEN_SIZE, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE selected command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [top_weights release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_decode_q4_k(uocr_metal_context *ctx,
                                                        const uint16_t *input_f16,
                                                        const uint32_t *top_expert_ids,
                                                        const float *top_weights_f32,
                                                        const void *selected_gate_weight_q4_k,
                                                        const void *selected_up_weight_q4_k,
                                                        const uint16_t *selected_down_weight_f16,
                                                        uint32_t physical_hidden_features,
                                                        uocr_metal_dense_output_type output_type,
                                                        void *out,
                                                        char *error,
                                                        size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        selected_gate_weight_q4_k == NULL || selected_up_weight_q4_k == NULL ||
        selected_down_weight_f16 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert Q4_K decode request");
    }
    if (physical_hidden_features < UOCR_HIDDEN_SIZE ||
        (physical_hidden_features % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K decode widths logical=%u physical=%u",
                          UOCR_HIDDEN_SIZE,
                          physical_hidden_features);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert Q4_K output type %d",
                          (int)output_type);
    }
    for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
        if (top_expert_ids[rank] >= UOCR_ROUTED_EXPERTS) {
            return metal_fail(error,
                              error_size,
                              "invalid Metal MoE selected Q4_K expert id %u at rank %u",
                              top_expert_ids[rank],
                              rank);
        }
        for (uint32_t prev = 0u; prev < rank; ++prev) {
            if (top_expert_ids[prev] == top_expert_ids[rank]) {
                return metal_fail(error, error_size, "duplicate Metal MoE selected Q4_K expert id %u", top_expert_ids[rank]);
            }
        }
        if (!isfinite(top_weights_f32[rank])) {
            return metal_fail(error, error_size, "invalid Metal MoE selected Q4_K expert weight at rank %u", rank);
        }
    }

    uint64_t input_bytes = 0u;
    uint64_t intermediate_values = 0u;
    uint64_t gate_row_size = 0u;
    uint64_t gate_weight_bytes = 0u;
    uint64_t down_weight_values = 0u;
    uint64_t down_weight_bytes = 0u;
    uint64_t intermediate_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE, &intermediate_values) ||
        !checked_mul_u64((uint64_t)(physical_hidden_features / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &gate_row_size) ||
        !checked_mul_u64(intermediate_values, gate_row_size, &gate_weight_bytes) ||
        !checked_mul_u64(intermediate_values, 2u, &intermediate_bytes) ||
        !checked_mul_u64(intermediate_values, (uint64_t)UOCR_HIDDEN_SIZE, &down_weight_values) ||
        !checked_mul_u64(down_weight_values, 2u, &down_weight_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K decode byte-size overflow");
    }
    if (gate_row_size == 0u || gate_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K decode row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || gate_weight_bytes > max_buffer_length ||
        down_weight_bytes > max_buffer_length || intermediate_bytes > max_buffer_length ||
        top_weights_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || gate_weight_bytes > (uint64_t)SIZE_MAX ||
        down_weight_bytes > (uint64_t)SIZE_MAX || intermediate_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        intermediate_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert Q4_K decode buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_selected_gate_up_q4_k",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_selected_down_sum_f16_to_f16" :
                                             "uocr_moe_selected_down_sum_f16_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_selected_q4_input_f16";

        gate_weight = [ctx->device newBufferWithBytes:selected_gate_weight_q4_k
                                               length:(NSUInteger)gate_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_selected_gate_weight_q4_k";

        up_weight = [ctx->device newBufferWithBytes:selected_up_weight_q4_k
                                             length:(NSUInteger)gate_weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_selected_up_weight_q4_k";

        down_weight = [ctx->device newBufferWithBytes:selected_down_weight_f16
                                               length:(NSUInteger)down_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_selected_q4_down_weight_f16";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_selected_q4_top_weights_f32";

        mid = [ctx->device newBufferWithLength:(NSUInteger)intermediate_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_selected_q4_mid_f16";
        memset([mid contents], 0, (size_t)intermediate_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_selected_q4_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_selected_q4_command_buffer";

        uocr_metal_moe_selected_q4_params gate_params;
        gate_params.hidden_size = UOCR_HIDDEN_SIZE;
        gate_params.physical_hidden_size = physical_hidden_features;
        gate_params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        gate_params.top_k = UOCR_MOE_TOP_K;
        gate_params.gate_row_size = (uint32_t)gate_row_size;
        gate_params.reserved0 = 0u;
        gate_params.reserved1 = 0u;
        gate_params.reserved2 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:gate_weight offset:0u atIndex:1u];
        [enc setBuffer:up_weight offset:0u atIndex:2u];
        [enc setBuffer:mid offset:0u atIndex:3u];
        [enc setBytes:&gate_params length:sizeof(gate_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        uocr_metal_moe_selected_params down_params;
        down_params.hidden_size = UOCR_HIDDEN_SIZE;
        down_params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        down_params.top_k = UOCR_MOE_TOP_K;
        down_params.reserved = 0u;

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:down_weight offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&down_params length:sizeof(down_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)UOCR_HIDDEN_SIZE, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE selected Q4_K command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [top_weights release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_decode_q4_k_q8_0(uocr_metal_context *ctx,
                                                             const uint16_t *input_f16,
                                                             const uint32_t *top_expert_ids,
                                                             const float *top_weights_f32,
                                                             const void *selected_gate_weight_q4_k,
                                                             const void *selected_up_weight_q4_k,
                                                             const void *selected_down_weight_q8_0,
                                                             uint32_t physical_hidden_features,
                                                             uint32_t physical_intermediate_features,
                                                             uocr_metal_dense_output_type output_type,
                                                             void *out,
                                                             char *error,
                                                             size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        selected_gate_weight_q4_k == NULL || selected_up_weight_q4_k == NULL ||
        selected_down_weight_q8_0 == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert Q4_K/Q8_0 decode request");
    }
    if (physical_hidden_features < UOCR_HIDDEN_SIZE ||
        (physical_hidden_features % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/Q8_0 decode hidden widths logical=%u physical=%u",
                          UOCR_HIDDEN_SIZE,
                          physical_hidden_features);
    }
    if (physical_intermediate_features < UOCR_MOE_EXPERT_INTERMEDIATE ||
        (physical_intermediate_features % UOCR_Q8_0_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/Q8_0 decode down widths logical=%u physical=%u",
                          UOCR_MOE_EXPERT_INTERMEDIATE,
                          physical_intermediate_features);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert Q4_K/Q8_0 output type %d",
                          (int)output_type);
    }
    for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
        if (top_expert_ids[rank] >= UOCR_ROUTED_EXPERTS) {
            return metal_fail(error,
                              error_size,
                              "invalid Metal MoE selected Q4_K/Q8_0 expert id %u at rank %u",
                              top_expert_ids[rank],
                              rank);
        }
        for (uint32_t prev = 0u; prev < rank; ++prev) {
            if (top_expert_ids[prev] == top_expert_ids[rank]) {
                return metal_fail(error,
                                  error_size,
                                  "duplicate Metal MoE selected Q4_K/Q8_0 expert id %u",
                                  top_expert_ids[rank]);
            }
        }
        if (!isfinite(top_weights_f32[rank])) {
            return metal_fail(error, error_size, "invalid Metal MoE selected Q4_K/Q8_0 expert weight at rank %u", rank);
        }
    }

    uint64_t input_bytes = 0u;
    uint64_t intermediate_values = 0u;
    uint64_t gate_row_size = 0u;
    uint64_t gate_weight_bytes = 0u;
    uint64_t down_row_size = 0u;
    uint64_t down_rows = 0u;
    uint64_t down_weight_bytes = 0u;
    uint64_t intermediate_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE, &intermediate_values) ||
        !checked_mul_u64((uint64_t)(physical_hidden_features / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &gate_row_size) ||
        !checked_mul_u64(intermediate_values, gate_row_size, &gate_weight_bytes) ||
        !checked_mul_u64((uint64_t)(physical_intermediate_features / UOCR_Q8_0_BLOCK_SIZE),
                         (uint64_t)UOCR_Q8_0_TYPE_SIZE,
                         &down_row_size) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)UOCR_HIDDEN_SIZE, &down_rows) ||
        !checked_mul_u64(down_rows, down_row_size, &down_weight_bytes) ||
        !checked_mul_u64(intermediate_values, 2u, &intermediate_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/Q8_0 decode byte-size overflow");
    }
    if (gate_row_size == 0u || gate_row_size > UINT32_MAX ||
        down_row_size == 0u || down_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/Q8_0 decode row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || gate_weight_bytes > max_buffer_length ||
        down_weight_bytes > max_buffer_length || intermediate_bytes > max_buffer_length ||
        top_weights_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || gate_weight_bytes > (uint64_t)SIZE_MAX ||
        down_weight_bytes > (uint64_t)SIZE_MAX || intermediate_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        intermediate_values > (uint64_t)UINT32_MAX || down_rows > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert Q4_K/Q8_0 decode buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_selected_gate_up_q4_k",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_selected_down_sum_q8_0_to_f16" :
                                             "uocr_moe_selected_down_sum_q8_0_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/Q8_0 input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_selected_q4q8_input_f16";

        gate_weight = [ctx->device newBufferWithBytes:selected_gate_weight_q4_k
                                               length:(NSUInteger)gate_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/Q8_0 gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_selected_q4q8_gate_weight_q4_k";

        up_weight = [ctx->device newBufferWithBytes:selected_up_weight_q4_k
                                             length:(NSUInteger)gate_weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/Q8_0 up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_selected_q4q8_up_weight_q4_k";

        down_weight = [ctx->device newBufferWithBytes:selected_down_weight_q8_0
                                               length:(NSUInteger)down_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/Q8_0 down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_selected_q4q8_down_weight_q8_0";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/Q8_0 top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_selected_q4q8_top_weights_f32";

        mid = [ctx->device newBufferWithLength:(NSUInteger)intermediate_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/Q8_0 intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_selected_q4q8_mid_f16";
        memset([mid contents], 0, (size_t)intermediate_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/Q8_0 output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_selected_q4q8_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K/Q8_0 command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_selected_q4q8_command_buffer";

        uocr_metal_moe_selected_q4_params gate_params;
        gate_params.hidden_size = UOCR_HIDDEN_SIZE;
        gate_params.physical_hidden_size = physical_hidden_features;
        gate_params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        gate_params.top_k = UOCR_MOE_TOP_K;
        gate_params.gate_row_size = (uint32_t)gate_row_size;
        gate_params.reserved0 = 0u;
        gate_params.reserved1 = 0u;
        gate_params.reserved2 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K/Q8_0 gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:gate_weight offset:0u atIndex:1u];
        [enc setBuffer:up_weight offset:0u atIndex:2u];
        [enc setBuffer:mid offset:0u atIndex:3u];
        [enc setBytes:&gate_params length:sizeof(gate_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        uocr_metal_moe_selected_down_q8_params down_params;
        down_params.hidden_size = UOCR_HIDDEN_SIZE;
        down_params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        down_params.physical_intermediate_size = physical_intermediate_features;
        down_params.top_k = UOCR_MOE_TOP_K;
        down_params.down_row_size = (uint32_t)down_row_size;
        down_params.reserved0 = 0u;
        down_params.reserved1 = 0u;
        down_params.reserved2 = 0u;

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K/Q8_0 down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:down_weight offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&down_params length:sizeof(down_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)UOCR_HIDDEN_SIZE, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE selected Q4_K/Q8_0 command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [top_weights release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_decode_q4_k_padded(uocr_metal_context *ctx,
                                                               const uint16_t *input_f16,
                                                               const uint32_t *top_expert_ids,
                                                               const float *top_weights_f32,
                                                               const void *selected_gate_weight_q4_k,
                                                               const void *selected_up_weight_q4_k,
                                                               const void *selected_down_weight_padded_q4_k,
                                                               uint32_t physical_hidden_features,
                                                               uint32_t physical_intermediate_features,
                                                               uocr_metal_dense_output_type output_type,
                                                               void *out,
                                                               char *error,
                                                               size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        selected_gate_weight_q4_k == NULL || selected_up_weight_q4_k == NULL ||
        selected_down_weight_padded_q4_k == NULL || out == NULL) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert Q4_K/padded-Q4_K decode request");
    }
    if (physical_hidden_features < UOCR_HIDDEN_SIZE ||
        (physical_hidden_features % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/padded-Q4_K decode hidden widths logical=%u physical=%u",
                          UOCR_HIDDEN_SIZE,
                          physical_hidden_features);
    }
    if (physical_intermediate_features < UOCR_MOE_EXPERT_INTERMEDIATE ||
        (physical_intermediate_features % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/padded-Q4_K decode down widths logical=%u physical=%u",
                          UOCR_MOE_EXPERT_INTERMEDIATE,
                          physical_intermediate_features);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert Q4_K/padded-Q4_K output type %d",
                          (int)output_type);
    }
    for (uint32_t rank = 0u; rank < UOCR_MOE_TOP_K; ++rank) {
        if (top_expert_ids[rank] >= UOCR_ROUTED_EXPERTS) {
            return metal_fail(error,
                              error_size,
                              "invalid Metal MoE selected Q4_K/padded-Q4_K expert id %u at rank %u",
                              top_expert_ids[rank],
                              rank);
        }
        for (uint32_t prev = 0u; prev < rank; ++prev) {
            if (top_expert_ids[prev] == top_expert_ids[rank]) {
                return metal_fail(error,
                                  error_size,
                                  "duplicate Metal MoE selected Q4_K/padded-Q4_K expert id %u",
                                  top_expert_ids[rank]);
            }
        }
        if (!isfinite(top_weights_f32[rank])) {
            return metal_fail(error,
                              error_size,
                              "invalid Metal MoE selected Q4_K/padded-Q4_K expert weight at rank %u",
                              rank);
        }
    }

    uint64_t input_bytes = 0u;
    uint64_t intermediate_values = 0u;
    uint64_t gate_row_size = 0u;
    uint64_t gate_weight_bytes = 0u;
    uint64_t down_row_size = 0u;
    uint64_t down_rows = 0u;
    uint64_t down_weight_bytes = 0u;
    uint64_t intermediate_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)UOCR_MOE_EXPERT_INTERMEDIATE, &intermediate_values) ||
        !checked_mul_u64((uint64_t)(physical_hidden_features / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &gate_row_size) ||
        !checked_mul_u64(intermediate_values, gate_row_size, &gate_weight_bytes) ||
        !checked_mul_u64((uint64_t)(physical_intermediate_features / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &down_row_size) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)UOCR_HIDDEN_SIZE, &down_rows) ||
        !checked_mul_u64(down_rows, down_row_size, &down_weight_bytes) ||
        !checked_mul_u64(intermediate_values, 2u, &intermediate_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_MOE_TOP_K, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_HIDDEN_SIZE, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/padded-Q4_K decode byte-size overflow");
    }
    if (gate_row_size == 0u || gate_row_size > UINT32_MAX ||
        down_row_size == 0u || down_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/padded-Q4_K decode row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || gate_weight_bytes > max_buffer_length ||
        down_weight_bytes > max_buffer_length || intermediate_bytes > max_buffer_length ||
        top_weights_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || gate_weight_bytes > (uint64_t)SIZE_MAX ||
        down_weight_bytes > (uint64_t)SIZE_MAX || intermediate_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        intermediate_values > (uint64_t)UINT32_MAX || down_rows > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert Q4_K/padded-Q4_K decode buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_selected_gate_up_q4_k",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_selected_down_sum_q4_k_to_f16" :
                                             "uocr_moe_selected_down_sum_q4_k_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/padded-Q4_K input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_selected_q4p_input_f16";

        gate_weight = [ctx->device newBufferWithBytes:selected_gate_weight_q4_k
                                               length:(NSUInteger)gate_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/padded-Q4_K gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_selected_q4p_gate_weight_q4_k";

        up_weight = [ctx->device newBufferWithBytes:selected_up_weight_q4_k
                                             length:(NSUInteger)gate_weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/padded-Q4_K up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_selected_q4p_up_weight_q4_k";

        down_weight = [ctx->device newBufferWithBytes:selected_down_weight_padded_q4_k
                                               length:(NSUInteger)down_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/padded-Q4_K down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_selected_q4p_down_weight_q4_k";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/padded-Q4_K top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_selected_q4p_top_weights_f32";

        mid = [ctx->device newBufferWithLength:(NSUInteger)intermediate_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/padded-Q4_K intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_selected_q4p_mid_f16";
        memset([mid contents], 0, (size_t)intermediate_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE selected Q4_K/padded-Q4_K output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_selected_q4p_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K/padded-Q4_K command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_selected_q4p_command_buffer";

        uocr_metal_moe_selected_q4_params gate_params;
        gate_params.hidden_size = UOCR_HIDDEN_SIZE;
        gate_params.physical_hidden_size = physical_hidden_features;
        gate_params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        gate_params.top_k = UOCR_MOE_TOP_K;
        gate_params.gate_row_size = (uint32_t)gate_row_size;
        gate_params.reserved0 = 0u;
        gate_params.reserved1 = 0u;
        gate_params.reserved2 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K/padded-Q4_K gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:gate_weight offset:0u atIndex:1u];
        [enc setBuffer:up_weight offset:0u atIndex:2u];
        [enc setBuffer:mid offset:0u atIndex:3u];
        [enc setBytes:&gate_params length:sizeof(gate_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        uocr_metal_moe_selected_down_q4_params down_params;
        down_params.hidden_size = UOCR_HIDDEN_SIZE;
        down_params.intermediate_size = UOCR_MOE_EXPERT_INTERMEDIATE;
        down_params.physical_intermediate_size = physical_intermediate_features;
        down_params.top_k = UOCR_MOE_TOP_K;
        down_params.down_row_size = (uint32_t)down_row_size;
        down_params.reserved0 = 0u;
        down_params.reserved1 = 0u;
        down_params.reserved2 = 0u;

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE selected Q4_K/padded-Q4_K down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:down_weight offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&down_params length:sizeof(down_params) atIndex:4u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)UOCR_HIDDEN_SIZE, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE selected Q4_K/padded-Q4_K command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [top_weights release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_prefill_f16(uocr_metal_context *ctx,
                                                        const uint16_t *input_f16,
                                                        const uint32_t *top_expert_ids,
                                                        const float *top_weights_f32,
                                                        const uint16_t *expert_gate_weight_f16,
                                                        const uint16_t *expert_up_weight_f16,
                                                        const uint16_t *expert_down_weight_f16,
                                                        uint32_t n_tokens,
                                                        uint32_t hidden_size,
                                                        uint32_t intermediate_size,
                                                        uint32_t expert_count,
                                                        uint32_t top_k,
                                                        uocr_metal_dense_output_type output_type,
                                                        void *out,
                                                        char *error,
                                                        size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        expert_gate_weight_f16 == NULL || expert_up_weight_f16 == NULL || expert_down_weight_f16 == NULL ||
        out == NULL || n_tokens == 0u || hidden_size == 0u || intermediate_size == 0u || expert_count == 0u ||
        top_k == 0u) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert prefill request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert prefill output type %d",
                          (int)output_type);
    }
    if (top_k > expert_count) {
        return metal_fail(error, error_size, "Metal MoE selected-expert prefill top_k exceeds expert_count");
    }

    for (uint32_t token = 0u; token < n_tokens; ++token) {
        for (uint32_t rank = 0u; rank < top_k; ++rank) {
            const uint32_t expert = top_expert_ids[token * top_k + rank];
            if (expert >= expert_count) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE prefill expert id %u at token %u rank %u",
                                  expert,
                                  token,
                                  rank);
            }
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if (top_expert_ids[token * top_k + prev] == expert) {
                    return metal_fail(error,
                                      error_size,
                                      "duplicate Metal MoE prefill expert id %u at token %u",
                                      expert,
                                      token);
                }
            }
            if (!isfinite(top_weights_f32[token * top_k + rank])) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE prefill expert weight at token %u rank %u",
                                  token,
                                  rank);
            }
        }
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t top_values = 0u;
    uint64_t top_ids_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t weight_values = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t mid_values = 0u;
    uint64_t mid_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    uint64_t gate_groups = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)top_k, &top_values) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(uint32_t), &top_ids_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64((uint64_t)expert_count, (uint64_t)intermediate_size, &weight_values) ||
        !checked_mul_u64(weight_values, (uint64_t)hidden_size, &weight_values) ||
        !checked_mul_u64(weight_values, 2u, &weight_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)intermediate_size, &mid_values) ||
        !checked_mul_u64(mid_values, 2u, &mid_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &output_values) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)intermediate_size, &gate_groups)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert prefill byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || top_ids_bytes > max_buffer_length || top_weights_bytes > max_buffer_length ||
        weight_bytes > max_buffer_length || mid_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || top_ids_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX ||
        mid_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        gate_groups > (uint64_t)UINT32_MAX || output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert prefill buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_prefill_selected_gate_up_f16",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_prefill_selected_down_sum_f16_to_f16" :
                                             "uocr_moe_prefill_selected_down_sum_f16_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> top_ids = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_prefill_input_f16";

        top_ids = [ctx->device newBufferWithBytes:top_expert_ids
                                           length:(NSUInteger)top_ids_bytes
                                          options:MTLResourceStorageModeShared];
        if (top_ids == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill top-id buffer");
            goto cleanup;
        }
        top_ids.label = @"uocr_moe_prefill_top_ids_u32";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_prefill_top_weights_f32";

        gate_weight = [ctx->device newBufferWithBytes:expert_gate_weight_f16
                                               length:(NSUInteger)weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_prefill_gate_weight_f16";

        up_weight = [ctx->device newBufferWithBytes:expert_up_weight_f16
                                             length:(NSUInteger)weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_prefill_up_weight_f16";

        down_weight = [ctx->device newBufferWithBytes:expert_down_weight_f16
                                               length:(NSUInteger)weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_prefill_down_weight_f16";

        mid = [ctx->device newBufferWithLength:(NSUInteger)mid_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_prefill_mid_f16";
        memset([mid contents], 0, (size_t)mid_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE prefill output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_prefill_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE prefill command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_prefill_command_buffer";

        uocr_metal_moe_prefill_selected_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = hidden_size;
        params.intermediate_size = intermediate_size;
        params.expert_count = expert_count;
        params.top_k = top_k;
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE prefill gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:gate_weight offset:0u atIndex:2u];
        [enc setBuffer:up_weight offset:0u atIndex:3u];
        [enc setBuffer:mid offset:0u atIndex:4u];
        [enc setBytes:&params length:sizeof(params) atIndex:5u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)gate_groups, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE prefill down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:down_weight offset:0u atIndex:3u];
        [enc setBuffer:dst offset:0u atIndex:4u];
        [enc setBytes:&params length:sizeof(params) atIndex:5u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE prefill command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [top_weights release];
        [top_ids release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_prefill_q4_k(uocr_metal_context *ctx,
                                                         const uint16_t *input_f16,
                                                         const uint32_t *top_expert_ids,
                                                         const float *top_weights_f32,
                                                         const void *expert_gate_weight_q4_k,
                                                         const void *expert_up_weight_q4_k,
                                                         const uint16_t *expert_down_weight_f16,
                                                         uint32_t n_tokens,
                                                         uint32_t hidden_size,
                                                         uint32_t physical_hidden_size,
                                                         uint32_t intermediate_size,
                                                         uint32_t expert_count,
                                                         uint32_t top_k,
                                                         uocr_metal_dense_output_type output_type,
                                                         void *out,
                                                         char *error,
                                                         size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        expert_gate_weight_q4_k == NULL || expert_up_weight_q4_k == NULL || expert_down_weight_f16 == NULL ||
        out == NULL || n_tokens == 0u || hidden_size == 0u || physical_hidden_size == 0u ||
        intermediate_size == 0u || expert_count == 0u || top_k == 0u) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert Q4_K prefill request");
    }
    if (hidden_size > physical_hidden_size || (physical_hidden_size % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K prefill widths logical=%u physical=%u",
                          hidden_size,
                          physical_hidden_size);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert Q4_K prefill output type %d",
                          (int)output_type);
    }
    if (top_k > expert_count) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K prefill top_k exceeds expert_count");
    }

    for (uint32_t token = 0u; token < n_tokens; ++token) {
        for (uint32_t rank = 0u; rank < top_k; ++rank) {
            const uint32_t expert = top_expert_ids[token * top_k + rank];
            if (expert >= expert_count) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE Q4_K prefill expert id %u at token %u rank %u",
                                  expert,
                                  token,
                                  rank);
            }
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if (top_expert_ids[token * top_k + prev] == expert) {
                    return metal_fail(error,
                                      error_size,
                                      "duplicate Metal MoE Q4_K prefill expert id %u at token %u",
                                      expert,
                                      token);
                }
            }
            if (!isfinite(top_weights_f32[token * top_k + rank])) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE Q4_K prefill expert weight at token %u rank %u",
                                  token,
                                  rank);
            }
        }
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t top_values = 0u;
    uint64_t top_ids_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t gate_row_size = 0u;
    uint64_t q4_rows = 0u;
    uint64_t gate_weight_bytes = 0u;
    uint64_t down_weight_values = 0u;
    uint64_t down_weight_bytes = 0u;
    uint64_t mid_values = 0u;
    uint64_t mid_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)top_k, &top_values) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(uint32_t), &top_ids_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64((uint64_t)(physical_hidden_size / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &gate_row_size) ||
        !checked_mul_u64((uint64_t)expert_count, (uint64_t)intermediate_size, &q4_rows) ||
        !checked_mul_u64(q4_rows, gate_row_size, &gate_weight_bytes) ||
        !checked_mul_u64(q4_rows, (uint64_t)hidden_size, &down_weight_values) ||
        !checked_mul_u64(down_weight_values, 2u, &down_weight_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)intermediate_size, &mid_values) ||
        !checked_mul_u64(mid_values, 2u, &mid_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &output_values) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K prefill byte-size overflow");
    }
    if (gate_row_size == 0u || gate_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K prefill row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || top_ids_bytes > max_buffer_length || top_weights_bytes > max_buffer_length ||
        gate_weight_bytes > max_buffer_length || down_weight_bytes > max_buffer_length || mid_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX || top_ids_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || gate_weight_bytes > (uint64_t)SIZE_MAX ||
        down_weight_bytes > (uint64_t)SIZE_MAX || mid_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX || mid_values > (uint64_t)UINT32_MAX || output_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert Q4_K prefill buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_prefill_selected_gate_up_q4_k",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_prefill_selected_down_sum_f16_to_f16" :
                                             "uocr_moe_prefill_selected_down_sum_f16_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> top_ids = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_prefill_q4_input_f16";

        top_ids = [ctx->device newBufferWithBytes:top_expert_ids
                                           length:(NSUInteger)top_ids_bytes
                                          options:MTLResourceStorageModeShared];
        if (top_ids == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill top-id buffer");
            goto cleanup;
        }
        top_ids.label = @"uocr_moe_prefill_q4_top_ids_u32";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_prefill_q4_top_weights_f32";

        gate_weight = [ctx->device newBufferWithBytes:expert_gate_weight_q4_k
                                               length:(NSUInteger)gate_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_prefill_gate_weight_q4_k";

        up_weight = [ctx->device newBufferWithBytes:expert_up_weight_q4_k
                                             length:(NSUInteger)gate_weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_prefill_up_weight_q4_k";

        down_weight = [ctx->device newBufferWithBytes:expert_down_weight_f16
                                               length:(NSUInteger)down_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_prefill_q4_down_weight_f16";

        mid = [ctx->device newBufferWithLength:(NSUInteger)mid_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_prefill_q4_mid_f16";
        memset([mid contents], 0, (size_t)mid_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K prefill output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_prefill_q4_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K prefill command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_prefill_q4_command_buffer";

        uocr_metal_moe_prefill_selected_q4_params gate_params;
        gate_params.n_tokens = n_tokens;
        gate_params.hidden_size = hidden_size;
        gate_params.physical_hidden_size = physical_hidden_size;
        gate_params.intermediate_size = intermediate_size;
        gate_params.expert_count = expert_count;
        gate_params.top_k = top_k;
        gate_params.gate_row_size = (uint32_t)gate_row_size;
        gate_params.reserved0 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K prefill gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:gate_weight offset:0u atIndex:2u];
        [enc setBuffer:up_weight offset:0u atIndex:3u];
        [enc setBuffer:mid offset:0u atIndex:4u];
        [enc setBytes:&gate_params length:sizeof(gate_params) atIndex:5u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mid_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        uocr_metal_moe_prefill_selected_params down_params;
        down_params.n_tokens = n_tokens;
        down_params.hidden_size = hidden_size;
        down_params.intermediate_size = intermediate_size;
        down_params.expert_count = expert_count;
        down_params.top_k = top_k;
        down_params.reserved0 = 0u;
        down_params.reserved1 = 0u;
        down_params.reserved2 = 0u;

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K prefill down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:down_weight offset:0u atIndex:3u];
        [enc setBuffer:dst offset:0u atIndex:4u];
        [enc setBytes:&down_params length:sizeof(down_params) atIndex:5u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE Q4_K prefill command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [top_weights release];
        [top_ids release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_prefill_q4_k_q8_0(uocr_metal_context *ctx,
                                                              const uint16_t *input_f16,
                                                              const uint32_t *top_expert_ids,
                                                              const float *top_weights_f32,
                                                              const void *expert_gate_weight_q4_k,
                                                              const void *expert_up_weight_q4_k,
                                                              const void *expert_down_weight_q8_0,
                                                              uint32_t n_tokens,
                                                              uint32_t hidden_size,
                                                              uint32_t physical_hidden_size,
                                                              uint32_t intermediate_size,
                                                              uint32_t physical_intermediate_size,
                                                              uint32_t expert_count,
                                                              uint32_t top_k,
                                                              uocr_metal_dense_output_type output_type,
                                                              void *out,
                                                              char *error,
                                                              size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        expert_gate_weight_q4_k == NULL || expert_up_weight_q4_k == NULL || expert_down_weight_q8_0 == NULL ||
        out == NULL || n_tokens == 0u || hidden_size == 0u || physical_hidden_size == 0u ||
        intermediate_size == 0u || physical_intermediate_size == 0u || expert_count == 0u || top_k == 0u) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert Q4_K/Q8_0 prefill request");
    }
    if (hidden_size > physical_hidden_size || (physical_hidden_size % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/Q8_0 prefill hidden widths logical=%u physical=%u",
                          hidden_size,
                          physical_hidden_size);
    }
    if (intermediate_size > physical_intermediate_size ||
        (physical_intermediate_size % UOCR_Q8_0_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/Q8_0 prefill down widths logical=%u physical=%u",
                          intermediate_size,
                          physical_intermediate_size);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert Q4_K/Q8_0 prefill output type %d",
                          (int)output_type);
    }
    if (top_k > expert_count) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/Q8_0 prefill top_k exceeds expert_count");
    }

    for (uint32_t token = 0u; token < n_tokens; ++token) {
        for (uint32_t rank = 0u; rank < top_k; ++rank) {
            const uint32_t expert = top_expert_ids[token * top_k + rank];
            if (expert >= expert_count) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE Q4_K/Q8_0 prefill expert id %u at token %u rank %u",
                                  expert,
                                  token,
                                  rank);
            }
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if (top_expert_ids[token * top_k + prev] == expert) {
                    return metal_fail(error,
                                      error_size,
                                      "duplicate Metal MoE Q4_K/Q8_0 prefill expert id %u at token %u",
                                      expert,
                                      token);
                }
            }
            if (!isfinite(top_weights_f32[token * top_k + rank])) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE Q4_K/Q8_0 prefill expert weight at token %u rank %u",
                                  token,
                                  rank);
            }
        }
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t top_values = 0u;
    uint64_t top_ids_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t gate_row_size = 0u;
    uint64_t q4_rows = 0u;
    uint64_t gate_weight_bytes = 0u;
    uint64_t down_row_size = 0u;
    uint64_t down_rows = 0u;
    uint64_t down_weight_bytes = 0u;
    uint64_t mid_values = 0u;
    uint64_t mid_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)top_k, &top_values) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(uint32_t), &top_ids_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64((uint64_t)(physical_hidden_size / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &gate_row_size) ||
        !checked_mul_u64((uint64_t)expert_count, (uint64_t)intermediate_size, &q4_rows) ||
        !checked_mul_u64(q4_rows, gate_row_size, &gate_weight_bytes) ||
        !checked_mul_u64((uint64_t)(physical_intermediate_size / UOCR_Q8_0_BLOCK_SIZE),
                         (uint64_t)UOCR_Q8_0_TYPE_SIZE,
                         &down_row_size) ||
        !checked_mul_u64((uint64_t)expert_count, (uint64_t)hidden_size, &down_rows) ||
        !checked_mul_u64(down_rows, down_row_size, &down_weight_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)intermediate_size, &mid_values) ||
        !checked_mul_u64(mid_values, 2u, &mid_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &output_values) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/Q8_0 prefill byte-size overflow");
    }
    if (gate_row_size == 0u || gate_row_size > UINT32_MAX ||
        down_row_size == 0u || down_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/Q8_0 prefill row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || top_ids_bytes > max_buffer_length || top_weights_bytes > max_buffer_length ||
        gate_weight_bytes > max_buffer_length || down_weight_bytes > max_buffer_length || mid_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX || top_ids_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || gate_weight_bytes > (uint64_t)SIZE_MAX ||
        down_weight_bytes > (uint64_t)SIZE_MAX || mid_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX || mid_values > (uint64_t)UINT32_MAX ||
        output_values > (uint64_t)UINT32_MAX || down_rows > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert Q4_K/Q8_0 prefill buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_prefill_selected_gate_up_q4_k",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_prefill_selected_down_sum_q8_0_to_f16" :
                                             "uocr_moe_prefill_selected_down_sum_q8_0_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> top_ids = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_prefill_q4q8_input_f16";

        top_ids = [ctx->device newBufferWithBytes:top_expert_ids
                                           length:(NSUInteger)top_ids_bytes
                                          options:MTLResourceStorageModeShared];
        if (top_ids == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill top-id buffer");
            goto cleanup;
        }
        top_ids.label = @"uocr_moe_prefill_q4q8_top_ids_u32";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_prefill_q4q8_top_weights_f32";

        gate_weight = [ctx->device newBufferWithBytes:expert_gate_weight_q4_k
                                               length:(NSUInteger)gate_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_prefill_q4q8_gate_weight_q4_k";

        up_weight = [ctx->device newBufferWithBytes:expert_up_weight_q4_k
                                             length:(NSUInteger)gate_weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_prefill_q4q8_up_weight_q4_k";

        down_weight = [ctx->device newBufferWithBytes:expert_down_weight_q8_0
                                               length:(NSUInteger)down_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_prefill_q4q8_down_weight_q8_0";

        mid = [ctx->device newBufferWithLength:(NSUInteger)mid_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_prefill_q4q8_mid_f16";
        memset([mid contents], 0, (size_t)mid_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/Q8_0 prefill output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_prefill_q4q8_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K/Q8_0 prefill command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_prefill_q4q8_command_buffer";

        uocr_metal_moe_prefill_selected_q4_params gate_params;
        gate_params.n_tokens = n_tokens;
        gate_params.hidden_size = hidden_size;
        gate_params.physical_hidden_size = physical_hidden_size;
        gate_params.intermediate_size = intermediate_size;
        gate_params.expert_count = expert_count;
        gate_params.top_k = top_k;
        gate_params.gate_row_size = (uint32_t)gate_row_size;
        gate_params.reserved0 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K/Q8_0 prefill gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:gate_weight offset:0u atIndex:2u];
        [enc setBuffer:up_weight offset:0u atIndex:3u];
        [enc setBuffer:mid offset:0u atIndex:4u];
        [enc setBytes:&gate_params length:sizeof(gate_params) atIndex:5u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mid_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        uocr_metal_moe_prefill_selected_down_q8_params down_params;
        down_params.n_tokens = n_tokens;
        down_params.hidden_size = hidden_size;
        down_params.intermediate_size = intermediate_size;
        down_params.physical_intermediate_size = physical_intermediate_size;
        down_params.expert_count = expert_count;
        down_params.top_k = top_k;
        down_params.down_row_size = (uint32_t)down_row_size;
        down_params.reserved0 = 0u;

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K/Q8_0 prefill down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:down_weight offset:0u atIndex:3u];
        [enc setBuffer:dst offset:0u atIndex:4u];
        [enc setBytes:&down_params length:sizeof(down_params) atIndex:5u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE Q4_K/Q8_0 prefill command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [top_weights release];
        [top_ids release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_selected_experts_prefill_q4_k_padded(uocr_metal_context *ctx,
                                                                const uint16_t *input_f16,
                                                                const uint32_t *top_expert_ids,
                                                                const float *top_weights_f32,
                                                                const void *expert_gate_weight_q4_k,
                                                                const void *expert_up_weight_q4_k,
                                                                const void *expert_down_weight_padded_q4_k,
                                                                uint32_t n_tokens,
                                                                uint32_t hidden_size,
                                                                uint32_t physical_hidden_size,
                                                                uint32_t intermediate_size,
                                                                uint32_t physical_intermediate_size,
                                                                uint32_t expert_count,
                                                                uint32_t top_k,
                                                                uocr_metal_dense_output_type output_type,
                                                                void *out,
                                                                char *error,
                                                                size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || input_f16 == NULL || top_expert_ids == NULL || top_weights_f32 == NULL ||
        expert_gate_weight_q4_k == NULL || expert_up_weight_q4_k == NULL ||
        expert_down_weight_padded_q4_k == NULL || out == NULL || n_tokens == 0u || hidden_size == 0u ||
        physical_hidden_size == 0u || intermediate_size == 0u || physical_intermediate_size == 0u ||
        expert_count == 0u || top_k == 0u) {
        return metal_fail(error, error_size, "invalid Metal MoE selected-expert Q4_K/padded-Q4_K prefill request");
    }
    if (hidden_size > physical_hidden_size || (physical_hidden_size % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/padded-Q4_K prefill hidden widths logical=%u physical=%u",
                          hidden_size,
                          physical_hidden_size);
    }
    if (intermediate_size > physical_intermediate_size ||
        (physical_intermediate_size % UOCR_Q4_K_BLOCK_SIZE) != 0u) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal MoE selected-expert Q4_K/padded-Q4_K prefill down widths logical=%u physical=%u",
                          intermediate_size,
                          physical_intermediate_size);
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error,
                          error_size,
                          "unsupported Metal MoE selected-expert Q4_K/padded-Q4_K prefill output type %d",
                          (int)output_type);
    }
    if (top_k > expert_count) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert Q4_K/padded-Q4_K prefill top_k exceeds expert_count");
    }

    for (uint32_t token = 0u; token < n_tokens; ++token) {
        for (uint32_t rank = 0u; rank < top_k; ++rank) {
            const uint32_t expert = top_expert_ids[token * top_k + rank];
            if (expert >= expert_count) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE Q4_K/padded-Q4_K prefill expert id %u at token %u rank %u",
                                  expert,
                                  token,
                                  rank);
            }
            for (uint32_t prev = 0u; prev < rank; ++prev) {
                if (top_expert_ids[token * top_k + prev] == expert) {
                    return metal_fail(error,
                                      error_size,
                                      "duplicate Metal MoE Q4_K/padded-Q4_K prefill expert id %u at token %u",
                                      expert,
                                      token);
                }
            }
            if (!isfinite(top_weights_f32[token * top_k + rank])) {
                return metal_fail(error,
                                  error_size,
                                  "invalid Metal MoE Q4_K/padded-Q4_K prefill expert weight at token %u rank %u",
                                  token,
                                  rank);
            }
        }
    }

    uint64_t input_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t top_values = 0u;
    uint64_t top_ids_bytes = 0u;
    uint64_t top_weights_bytes = 0u;
    uint64_t gate_row_size = 0u;
    uint64_t q4_rows = 0u;
    uint64_t gate_weight_bytes = 0u;
    uint64_t down_row_size = 0u;
    uint64_t down_rows = 0u;
    uint64_t down_weight_bytes = 0u;
    uint64_t mid_values = 0u;
    uint64_t mid_bytes = 0u;
    uint64_t output_values = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &input_values) ||
        !checked_mul_u64(input_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)top_k, &top_values) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(uint32_t), &top_ids_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)sizeof(float), &top_weights_bytes) ||
        !checked_mul_u64((uint64_t)(physical_hidden_size / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &gate_row_size) ||
        !checked_mul_u64((uint64_t)expert_count, (uint64_t)intermediate_size, &q4_rows) ||
        !checked_mul_u64(q4_rows, gate_row_size, &gate_weight_bytes) ||
        !checked_mul_u64((uint64_t)(physical_intermediate_size / UOCR_Q4_K_BLOCK_SIZE),
                         (uint64_t)UOCR_Q4_K_TYPE_SIZE,
                         &down_row_size) ||
        !checked_mul_u64((uint64_t)expert_count, (uint64_t)hidden_size, &down_rows) ||
        !checked_mul_u64(down_rows, down_row_size, &down_weight_bytes) ||
        !checked_mul_u64(top_values, (uint64_t)intermediate_size, &mid_values) ||
        !checked_mul_u64(mid_values, 2u, &mid_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)hidden_size, &output_values) ||
        !checked_mul_u64(output_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/padded-Q4_K prefill byte-size overflow");
    }
    if (gate_row_size == 0u || gate_row_size > UINT32_MAX ||
        down_row_size == 0u || down_row_size > UINT32_MAX) {
        return metal_fail(error, error_size, "Metal MoE selected-expert Q4_K/padded-Q4_K prefill row size is invalid");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || top_ids_bytes > max_buffer_length || top_weights_bytes > max_buffer_length ||
        gate_weight_bytes > max_buffer_length || down_weight_bytes > max_buffer_length || mid_bytes > max_buffer_length ||
        output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX || top_ids_bytes > (uint64_t)SIZE_MAX ||
        top_weights_bytes > (uint64_t)SIZE_MAX || gate_weight_bytes > (uint64_t)SIZE_MAX ||
        down_weight_bytes > (uint64_t)SIZE_MAX || mid_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX || mid_values > (uint64_t)UINT32_MAX ||
        output_values > (uint64_t)UINT32_MAX || down_rows > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE selected-expert Q4_K/padded-Q4_K prefill buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> gate_up_pipeline = metal_get_pipeline(ctx,
                                                                          "uocr_moe_prefill_selected_gate_up_q4_k",
                                                                          error,
                                                                          error_size);
        if (gate_up_pipeline == nil) {
            return 0;
        }
        const char *down_function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                             "uocr_moe_prefill_selected_down_sum_q4_k_to_f16" :
                                             "uocr_moe_prefill_selected_down_sum_q4_k_to_f32";
        id<MTLComputePipelineState> down_pipeline = metal_get_pipeline(ctx, down_function_name, error, error_size);
        if (down_pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> top_ids = nil;
        id<MTLBuffer> top_weights = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> mid = nil;
        id<MTLBuffer> dst = nil;

        input = [ctx->device newBufferWithBytes:input_f16
                                         length:(NSUInteger)input_bytes
                                        options:MTLResourceStorageModeShared];
        if (input == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill input buffer");
            goto cleanup;
        }
        input.label = @"uocr_moe_prefill_q4p_input_f16";

        top_ids = [ctx->device newBufferWithBytes:top_expert_ids
                                           length:(NSUInteger)top_ids_bytes
                                          options:MTLResourceStorageModeShared];
        if (top_ids == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill top-id buffer");
            goto cleanup;
        }
        top_ids.label = @"uocr_moe_prefill_q4p_top_ids_u32";

        top_weights = [ctx->device newBufferWithBytes:top_weights_f32
                                               length:(NSUInteger)top_weights_bytes
                                              options:MTLResourceStorageModeShared];
        if (top_weights == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill top-weight buffer");
            goto cleanup;
        }
        top_weights.label = @"uocr_moe_prefill_q4p_top_weights_f32";

        gate_weight = [ctx->device newBufferWithBytes:expert_gate_weight_q4_k
                                               length:(NSUInteger)gate_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (gate_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill gate-weight buffer");
            goto cleanup;
        }
        gate_weight.label = @"uocr_moe_prefill_q4p_gate_weight_q4_k";

        up_weight = [ctx->device newBufferWithBytes:expert_up_weight_q4_k
                                             length:(NSUInteger)gate_weight_bytes
                                            options:MTLResourceStorageModeShared];
        if (up_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill up-weight buffer");
            goto cleanup;
        }
        up_weight.label = @"uocr_moe_prefill_q4p_up_weight_q4_k";

        down_weight = [ctx->device newBufferWithBytes:expert_down_weight_padded_q4_k
                                               length:(NSUInteger)down_weight_bytes
                                              options:MTLResourceStorageModeShared];
        if (down_weight == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill down-weight buffer");
            goto cleanup;
        }
        down_weight.label = @"uocr_moe_prefill_q4p_down_weight_q4_k";

        mid = [ctx->device newBufferWithLength:(NSUInteger)mid_bytes options:MTLResourceStorageModeShared];
        if (mid == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill intermediate buffer");
            goto cleanup;
        }
        mid.label = @"uocr_moe_prefill_q4p_mid_f16";
        memset([mid contents], 0, (size_t)mid_bytes);

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE Q4_K/padded-Q4_K prefill output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_prefill_q4p_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K/padded-Q4_K prefill command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_prefill_q4p_command_buffer";

        uocr_metal_moe_prefill_selected_q4_params gate_params;
        gate_params.n_tokens = n_tokens;
        gate_params.hidden_size = hidden_size;
        gate_params.physical_hidden_size = physical_hidden_size;
        gate_params.intermediate_size = intermediate_size;
        gate_params.expert_count = expert_count;
        gate_params.top_k = top_k;
        gate_params.gate_row_size = (uint32_t)gate_row_size;
        gate_params.reserved0 = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K/padded-Q4_K prefill gate/up command encoder");
            goto cleanup;
        }
        const NSUInteger gate_threads = metal_power2_threadgroup_width(256u, gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:gate_up_pipeline];
        [enc setBuffer:input offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:gate_weight offset:0u atIndex:2u];
        [enc setBuffer:up_weight offset:0u atIndex:3u];
        [enc setBuffer:mid offset:0u atIndex:4u];
        [enc setBytes:&gate_params length:sizeof(gate_params) atIndex:5u];
        [enc setThreadgroupMemoryLength:gate_threads * 2u * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mid_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(gate_threads, 1u, 1u)];
        [enc endEncoding];

        uocr_metal_moe_prefill_selected_down_q4_params down_params;
        down_params.n_tokens = n_tokens;
        down_params.hidden_size = hidden_size;
        down_params.intermediate_size = intermediate_size;
        down_params.physical_intermediate_size = physical_intermediate_size;
        down_params.expert_count = expert_count;
        down_params.top_k = top_k;
        down_params.down_row_size = (uint32_t)down_row_size;
        down_params.reserved0 = 0u;

        enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE Q4_K/padded-Q4_K prefill down command encoder");
            goto cleanup;
        }
        const NSUInteger down_threads = metal_power2_threadgroup_width(256u, down_pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:down_pipeline];
        [enc setBuffer:mid offset:0u atIndex:0u];
        [enc setBuffer:top_ids offset:0u atIndex:1u];
        [enc setBuffer:top_weights offset:0u atIndex:2u];
        [enc setBuffer:down_weight offset:0u atIndex:3u];
        [enc setBuffer:dst offset:0u atIndex:4u];
        [enc setBytes:&down_params length:sizeof(down_params) atIndex:5u];
        [enc setThreadgroupMemoryLength:down_threads * sizeof(float) atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)output_values, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE Q4_K/padded-Q4_K prefill command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [mid release];
        [down_weight release];
        [up_weight release];
        [gate_weight release];
        [top_weights release];
        [top_ids release];
        [input release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_moe_combine_f16(uocr_metal_context *ctx,
                                       const uint16_t *routed_f16,
                                       const uint16_t *shared_f16,
                                       const uint16_t *residual_f16_or_null,
                                       uint32_t n_tokens,
                                       uocr_metal_dense_output_type output_type,
                                       void *out,
                                       char *error,
                                       size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || routed_f16 == NULL || shared_f16 == NULL || out == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal MoE combine request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal MoE combine output type %d", (int)output_type);
    }

    uint64_t value_count = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &value_count) ||
        !checked_mul_u64(value_count, 2u, &input_bytes) ||
        !checked_mul_u64(value_count, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal MoE combine byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX ||
        value_count > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal MoE combine buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_moe_combine_f16_to_f16" :
                                        "uocr_moe_combine_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        int ok = 0;
        id<MTLBuffer> routed = nil;
        id<MTLBuffer> shared = nil;
        id<MTLBuffer> residual = nil;
        id<MTLBuffer> dst = nil;

        routed = [ctx->device newBufferWithBytes:routed_f16
                                          length:(NSUInteger)input_bytes
                                         options:MTLResourceStorageModeShared];
        if (routed == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE combine routed buffer");
            goto cleanup;
        }
        routed.label = @"uocr_moe_combine_routed_f16";

        shared = [ctx->device newBufferWithBytes:shared_f16
                                          length:(NSUInteger)input_bytes
                                         options:MTLResourceStorageModeShared];
        if (shared == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE combine shared buffer");
            goto cleanup;
        }
        shared.label = @"uocr_moe_combine_shared_f16";

        id<MTLBuffer> residual_arg = shared;
        if (residual_f16_or_null != NULL) {
            residual = [ctx->device newBufferWithBytes:residual_f16_or_null
                                                length:(NSUInteger)input_bytes
                                               options:MTLResourceStorageModeShared];
            if (residual == nil) {
                (void)metal_fail(error, error_size, "failed to allocate Metal MoE combine residual buffer");
                goto cleanup;
            }
            residual.label = @"uocr_moe_combine_residual_f16";
            residual_arg = residual;
        }

        dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            (void)metal_fail(error, error_size, "failed to allocate Metal MoE combine output buffer");
            goto cleanup;
        }
        dst.label = @"uocr_moe_combine_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE combine command buffer");
            goto cleanup;
        }
        cb.label = @"uocr_moe_combine_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            (void)metal_fail(error, error_size, "failed to create Metal MoE combine command encoder");
            goto cleanup;
        }

        uocr_metal_moe_combine_params params;
        params.n_tokens = n_tokens;
        params.hidden_size = UOCR_HIDDEN_SIZE;
        params.has_residual = residual_f16_or_null != NULL ? 1u : 0u;
        params.reserved = 0u;

        const NSUInteger threads_per_group = metal_power2_threadgroup_width(256u, pipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:routed offset:0u atIndex:0u];
        [enc setBuffer:shared offset:0u atIndex:1u];
        [enc setBuffer:residual_arg offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)value_count, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            (void)metal_fail(error, error_size, "Metal MoE combine command failed: %s", [description UTF8String]);
            goto cleanup;
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        ok = 1;

    cleanup:
        [dst release];
        [residual release];
        [shared release];
        [routed release];
        if (!ok) {
            return 0;
        }
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_rope_qk_f16(uocr_metal_context *ctx,
                                   const uint16_t *q_f16,
                                   const uint16_t *k_f16,
                                   uint32_t n_tokens,
                                   uint32_t position_start,
                                   uocr_metal_dense_output_type output_type,
                                   void *q_out,
                                   void *k_out,
                                   char *error,
                                   size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_f16 == NULL || q_out == NULL || k_out == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal RoPE request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal RoPE output type %d", (int)output_type);
    }
    uint64_t position_end = 0u;
    if (!checked_add_u64((uint64_t)position_start, (uint64_t)n_tokens, &position_end) ||
        position_end > (uint64_t)UOCR_MAX_POSITIONS) {
        return metal_fail(error,
                          error_size,
                          "Metal RoPE position range [%u,%llu) exceeds max positions %u",
                          position_start,
                          (unsigned long long)position_end,
                          UOCR_MAX_POSITIONS);
    }

    uint64_t tensor_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t pair_threads = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_HIDDEN_SIZE, &tensor_values) ||
        !checked_mul_u64(tensor_values, 2u, &input_bytes) ||
        !checked_mul_u64(tensor_values, output_element_bytes, &output_bytes) ||
        !checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_ATTENTION_HEADS * (UOCR_HEAD_DIM / 2u), &pair_threads)) {
        return metal_fail(error, error_size, "Metal RoPE byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX || pair_threads > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal RoPE buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_rope_qk_f16_to_f16" :
                                        "uocr_rope_qk_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> q_src = [ctx->device newBufferWithBytes:q_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (q_src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal RoPE Q input buffer");
        }
        q_src.label = @"uocr_rope_q_input_f16";

        id<MTLBuffer> k_src = [ctx->device newBufferWithBytes:k_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (k_src == nil) {
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal RoPE K input buffer");
        }
        k_src.label = @"uocr_rope_k_input_f16";

        id<MTLBuffer> q_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (q_dst == nil || k_dst == nil) {
            [k_dst release];
            [q_dst release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal RoPE output buffers");
        }
        q_dst.label = @"uocr_rope_q_output";
        k_dst.label = @"uocr_rope_k_output";
        memset([q_dst contents], 0, (size_t)output_bytes);
        memset([k_dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [k_dst release];
            [q_dst release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal RoPE command buffer");
        }
        cb.label = @"uocr_rope_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [k_dst release];
            [q_dst release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal RoPE command encoder");
        }

        uocr_metal_rope_qk_params params;
        params.n_tokens = n_tokens;
        params.heads = UOCR_ATTENTION_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.position_start = position_start;
        params.freq_scale = -2.0f * log2f((float)UOCR_ROPE_THETA) / (float)UOCR_HEAD_DIM;
        params.reserved0 = 0u;
        params.reserved1 = 0u;
        params.reserved2 = 0u;

        NSUInteger threads_per_group = pipeline.threadExecutionWidth;
        if (threads_per_group == 0u) {
            threads_per_group = 1u;
        }
        if (threads_per_group > 256u) {
            threads_per_group = 256u;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_src offset:0u atIndex:0u];
        [enc setBuffer:k_src offset:0u atIndex:1u];
        [enc setBuffer:q_dst offset:0u atIndex:2u];
        [enc setBuffer:k_dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)pair_threads, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [k_dst release];
            [q_dst release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "Metal RoPE command failed: %s", [description UTF8String]);
        }

        memcpy(q_out, [q_dst contents], (size_t)output_bytes);
        memcpy(k_out, [k_dst contents], (size_t)output_bytes);
        [k_dst release];
        [q_dst release];
        [k_src release];
        [q_src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_prefill_attention_f16(uocr_metal_context *ctx,
                                             const uint16_t *q_f16,
                                             const uint16_t *k_f16,
                                             const uint16_t *v_f16,
                                             uint32_t n_tokens,
                                             uocr_metal_dense_output_type output_type,
                                             void *out,
                                             char *error,
                                             size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL || out == NULL || n_tokens == 0u) {
        return metal_fail(error, error_size, "invalid Metal prefill attention request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal prefill attention output type %d", (int)output_type);
    }
    if (n_tokens > UOCR_MAX_POSITIONS) {
        return metal_fail(error,
                          error_size,
                          "Metal prefill attention prompt length %u exceeds max positions %u",
                          n_tokens,
                          UOCR_MAX_POSITIONS);
    }

    uint64_t tensor_values = 0u;
    uint64_t attention_groups = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_ATTENTION_HEADS, &attention_groups) ||
        !checked_mul_u64(attention_groups, (uint64_t)UOCR_HEAD_DIM, &tensor_values) ||
        !checked_mul_u64(tensor_values, 2u, &input_bytes) ||
        !checked_mul_u64(tensor_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal prefill attention byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length || input_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX || tensor_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal prefill attention buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_prefill_attention_f16_to_f16" :
                                        "uocr_prefill_attention_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        NSUInteger threads_per_group = metal_power2_threadgroup_width(128u, pipeline.maxTotalThreadsPerThreadgroup);
        if (threads_per_group == 0u) {
            threads_per_group = 1u;
        }
        uint64_t threadgroup_float_count = 0u;
        uint64_t requested_threadgroup_bytes = 0u;
        uint64_t threadgroup_bytes = 0u;
        /* Metal API validation requires threadgroup memory lengths to be
         * 16-byte aligned; the kernel only touches the requested float range.
         */
        if (!checked_add_u64((uint64_t)n_tokens, (uint64_t)threads_per_group, &threadgroup_float_count) ||
            !checked_mul_u64(threadgroup_float_count, (uint64_t)sizeof(float), &requested_threadgroup_bytes) ||
            !align_up_u64_checked(requested_threadgroup_bytes, 16u, &threadgroup_bytes) ||
            threadgroup_bytes > (uint64_t)NSUIntegerMax) {
            return metal_fail(error, error_size, "Metal prefill attention threadgroup memory overflow");
        }
        const NSUInteger max_threadgroup_memory = ctx->device.maxThreadgroupMemoryLength;
        if (threadgroup_bytes > (uint64_t)max_threadgroup_memory) {
            return metal_fail(error,
                              error_size,
                              "Metal prefill attention needs %llu bytes of threadgroup memory, exceeding limit %llu",
                              (unsigned long long)threadgroup_bytes,
                              (unsigned long long)max_threadgroup_memory);
        }

        id<MTLBuffer> q_src = [ctx->device newBufferWithBytes:q_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (q_src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal prefill attention Q buffer");
        }
        q_src.label = @"uocr_prefill_attention_q_f16";

        id<MTLBuffer> k_src = [ctx->device newBufferWithBytes:k_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (k_src == nil) {
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal prefill attention K buffer");
        }
        k_src.label = @"uocr_prefill_attention_k_f16";

        id<MTLBuffer> v_src = [ctx->device newBufferWithBytes:v_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (v_src == nil) {
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal prefill attention V buffer");
        }
        v_src.label = @"uocr_prefill_attention_v_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal prefill attention output buffer");
        }
        dst.label = @"uocr_prefill_attention_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal prefill attention command buffer");
        }
        cb.label = @"uocr_prefill_attention_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal prefill attention command encoder");
        }

        uocr_metal_prefill_attention_params params;
        params.n_tokens = n_tokens;
        params.heads = UOCR_ATTENTION_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_src offset:0u atIndex:0u];
        [enc setBuffer:k_src offset:0u atIndex:1u];
        [enc setBuffer:v_src offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:(NSUInteger)threadgroup_bytes atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)attention_groups, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "Metal prefill attention command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [v_src release];
        [k_src release];
        [q_src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_prefill_attention_varlen_f16(uocr_metal_context *ctx,
                                                    const uint16_t *q_f16,
                                                    const uint16_t *k_f16,
                                                    const uint16_t *v_f16,
                                                    const uint32_t *cu_seqlens,
                                                    uint32_t batch,
                                                    uint32_t total_tokens,
                                                    uint32_t max_seqlen,
                                                    uocr_metal_dense_output_type output_type,
                                                    void *out,
                                                    char *error,
                                                    size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_f16 == NULL || v_f16 == NULL || cu_seqlens == NULL || out == NULL ||
        batch == 0u || total_tokens == 0u || max_seqlen == 0u) {
        return metal_fail(error, error_size, "invalid Metal varlen prefill attention request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal varlen prefill attention output type %d", (int)output_type);
    }
    if (max_seqlen > UOCR_MAX_POSITIONS || cu_seqlens[0] != 0u || cu_seqlens[batch] != total_tokens) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal varlen prefill attention sequence metadata: batch=%u total_tokens=%u max_seqlen=%u",
                          batch,
                          total_tokens,
                          max_seqlen);
    }
    for (uint32_t b = 0u; b < batch; ++b) {
        const uint32_t start = cu_seqlens[b];
        const uint32_t end = cu_seqlens[b + 1u];
        if (end <= start || end > total_tokens) {
            return metal_fail(error,
                              error_size,
                              "invalid Metal varlen prefill attention cu_seqlens at batch index %u",
                              b);
        }
        const uint32_t len = end - start;
        if (len > max_seqlen || len > UOCR_MAX_POSITIONS) {
            return metal_fail(error,
                              error_size,
                              "Metal varlen prefill sequence length %u exceeds max_seqlen=%u/max_positions=%u",
                              len,
                              max_seqlen,
                              UOCR_MAX_POSITIONS);
        }
    }

    uint64_t attention_groups = 0u;
    uint64_t tensor_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t cu_bytes = 0u;
    uint64_t output_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if (!checked_mul_u64((uint64_t)total_tokens, (uint64_t)UOCR_ATTENTION_HEADS, &attention_groups) ||
        !checked_mul_u64(attention_groups, (uint64_t)UOCR_HEAD_DIM, &tensor_values) ||
        !checked_mul_u64(tensor_values, 2u, &input_bytes) ||
        !checked_mul_u64((uint64_t)batch + 1u, (uint64_t)sizeof(uint32_t), &cu_bytes) ||
        !checked_mul_u64(tensor_values, output_element_bytes, &output_bytes)) {
        return metal_fail(error, error_size, "Metal varlen prefill attention byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (input_bytes > max_buffer_length || output_bytes > max_buffer_length || cu_bytes > max_buffer_length ||
        input_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX || cu_bytes > (uint64_t)SIZE_MAX ||
        attention_groups > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal varlen prefill attention buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_prefill_attention_varlen_f16_to_f16" :
                                        "uocr_prefill_attention_varlen_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        if (pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)UOCR_HEAD_DIM) {
            return metal_fail(error,
                              error_size,
                              "Metal varlen prefill attention pipeline supports only %llu threads per group",
                              (unsigned long long)pipeline.maxTotalThreadsPerThreadgroup);
        }
        const NSUInteger threads_per_group = (NSUInteger)UOCR_HEAD_DIM;
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            return metal_fail(error, error_size, "Metal varlen prefill attention threadgroup memory exceeds device limit");
        }

        id<MTLBuffer> q_src = [ctx->device newBufferWithBytes:q_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (q_src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal varlen prefill Q buffer");
        }
        q_src.label = @"uocr_prefill_varlen_q_f16";

        id<MTLBuffer> k_src = [ctx->device newBufferWithBytes:k_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (k_src == nil) {
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal varlen prefill K buffer");
        }
        k_src.label = @"uocr_prefill_varlen_k_f16";

        id<MTLBuffer> v_src = [ctx->device newBufferWithBytes:v_f16
                                                       length:(NSUInteger)input_bytes
                                                      options:MTLResourceStorageModeShared];
        if (v_src == nil) {
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal varlen prefill V buffer");
        }
        v_src.label = @"uocr_prefill_varlen_v_f16";

        id<MTLBuffer> cu = [ctx->device newBufferWithBytes:cu_seqlens
                                                    length:(NSUInteger)cu_bytes
                                                   options:MTLResourceStorageModeShared];
        if (cu == nil) {
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal varlen prefill cu_seqlens buffer");
        }
        cu.label = @"uocr_prefill_varlen_cu_seqlens";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [cu release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal varlen prefill output buffer");
        }
        dst.label = @"uocr_prefill_varlen_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [cu release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal varlen prefill command buffer");
        }
        cb.label = @"uocr_prefill_varlen_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [cu release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal varlen prefill command encoder");
        }

        uocr_metal_prefill_attention_varlen_params params;
        memset(&params, 0, sizeof(params));
        params.total_tokens = total_tokens;
        params.batch = batch;
        params.heads = UOCR_ATTENTION_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_src offset:0u atIndex:0u];
        [enc setBuffer:k_src offset:0u atIndex:1u];
        [enc setBuffer:v_src offset:0u atIndex:2u];
        [enc setBuffer:cu offset:0u atIndex:3u];
        [enc setBuffer:dst offset:0u atIndex:4u];
        [enc setBytes:&params length:sizeof(params) atIndex:5u];
        [enc setThreadgroupMemoryLength:(NSUInteger)threadgroup_bytes atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)attention_groups, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [cu release];
            [v_src release];
            [k_src release];
            [q_src release];
            return metal_fail(error, error_size, "Metal varlen prefill attention command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [cu release];
        [v_src release];
        [k_src release];
        [q_src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_decode_attention_f16(uocr_metal_context *ctx,
                                            const uint16_t *q_f16,
                                            const uint16_t *k_cache_f16,
                                            const uint16_t *v_cache_f16,
                                            uint32_t batch_slots,
                                            uint32_t prompt_token_capacity,
                                            uint32_t layer,
                                            uint32_t slot,
                                            uint32_t prompt_length,
                                            uint32_t generated_count,
                                            uocr_metal_dense_output_type output_type,
                                            void *out,
                                            char *error,
                                            size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || q_f16 == NULL || k_cache_f16 == NULL || v_cache_f16 == NULL || out == NULL ||
        batch_slots == 0u || prompt_token_capacity == 0u) {
        return metal_fail(error, error_size, "invalid Metal decode attention request");
    }
    if (output_type != UOCR_METAL_DENSE_OUTPUT_F16 && output_type != UOCR_METAL_DENSE_OUTPUT_F32) {
        return metal_fail(error, error_size, "unsupported Metal decode attention output type %d", (int)output_type);
    }
    if (layer >= UOCR_DECODER_LAYERS || slot >= batch_slots) {
        return metal_fail(error,
                          error_size,
                          "Metal decode attention layer/slot out of range: layer=%u slot=%u batch_slots=%u",
                          layer,
                          slot,
                          batch_slots);
    }
    if (prompt_length == 0u || prompt_length > prompt_token_capacity || prompt_length > UOCR_MAX_POSITIONS) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal decode attention prompt length: prompt_length=%u prompt_token_capacity=%u",
                          prompt_length,
                          prompt_token_capacity);
    }
    if (generated_count > UOCR_MAX_POSITIONS - prompt_length) {
        return metal_fail(error,
                          error_size,
                          "Metal decode attention generated count %u exceeds remaining positions for prompt length %u",
                          generated_count,
                          prompt_length);
    }

    uocr_metal_decode_attention_plan plan;
    if (!uocr_metal_kv_cache_decode_attention_plan(prompt_length,
                                                   prompt_token_capacity,
                                                   generated_count,
                                                   &plan)) {
        return metal_fail(error, error_size, "invalid Metal decode attention window/prefix metadata");
    }

    const uint64_t cache_token_capacity_u64 = (uint64_t)plan.cache_token_capacity;
    uint64_t q_values = 0u;
    uint64_t q_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t layer_slot_tokens = 0u;
    uint64_t cache_values = 0u;
    uint64_t cache_bytes = 0u;
    const uint64_t output_element_bytes = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ? 2u : (uint64_t)sizeof(float);
    if ((uint64_t)prompt_length + (uint64_t)UOCR_GENERATED_RING_WINDOW > cache_token_capacity_u64 ||
        !checked_mul_u64((uint64_t)UOCR_ATTENTION_HEADS, (uint64_t)UOCR_HEAD_DIM, &q_values) ||
        !checked_mul_u64(q_values, 2u, &q_bytes) ||
        !checked_mul_u64(q_values, output_element_bytes, &output_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_DECODER_LAYERS, (uint64_t)batch_slots, &layer_slot_tokens) ||
        !checked_mul_u64(layer_slot_tokens, cache_token_capacity_u64, &cache_values) ||
        !checked_mul_u64(cache_values, (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM, &cache_values) ||
        !checked_mul_u64(cache_values, 2u, &cache_bytes)) {
        return metal_fail(error, error_size, "Metal decode attention byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (q_bytes > max_buffer_length || output_bytes > max_buffer_length || cache_bytes > max_buffer_length ||
        q_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX || cache_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal decode attention buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = output_type == UOCR_METAL_DENSE_OUTPUT_F16 ?
                                        "uocr_decode_attention_f16_to_f16" :
                                        "uocr_decode_attention_f16_to_f32";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }
        if (pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)UOCR_HEAD_DIM) {
            return metal_fail(error,
                              error_size,
                              "Metal decode attention pipeline supports only %llu threads per group",
                              (unsigned long long)pipeline.maxTotalThreadsPerThreadgroup);
        }
        const NSUInteger threads_per_group = (NSUInteger)UOCR_HEAD_DIM;
        const uint64_t threadgroup_bytes = (uint64_t)threads_per_group * (uint64_t)sizeof(float);
        if (threadgroup_bytes > (uint64_t)ctx->device.maxThreadgroupMemoryLength) {
            return metal_fail(error, error_size, "Metal decode attention threadgroup memory exceeds device limit");
        }

        id<MTLBuffer> q_src = [ctx->device newBufferWithBytes:q_f16
                                                       length:(NSUInteger)q_bytes
                                                      options:MTLResourceStorageModeShared];
        if (q_src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal decode attention Q buffer");
        }
        q_src.label = @"uocr_decode_attention_q_f16";

        id<MTLBuffer> k_cache = [ctx->device newBufferWithBytes:k_cache_f16
                                                         length:(NSUInteger)cache_bytes
                                                        options:MTLResourceStorageModeShared];
        if (k_cache == nil) {
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal decode attention K cache buffer");
        }
        k_cache.label = @"uocr_decode_attention_k_cache_f16";

        id<MTLBuffer> v_cache = [ctx->device newBufferWithBytes:v_cache_f16
                                                         length:(NSUInteger)cache_bytes
                                                        options:MTLResourceStorageModeShared];
        if (v_cache == nil) {
            [k_cache release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal decode attention V cache buffer");
        }
        v_cache.label = @"uocr_decode_attention_v_cache_f16";

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [v_cache release];
            [k_cache release];
            [q_src release];
            return metal_fail(error, error_size, "failed to allocate Metal decode attention output buffer");
        }
        dst.label = @"uocr_decode_attention_output";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [v_cache release];
            [k_cache release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal decode attention command buffer");
        }
        cb.label = @"uocr_decode_attention_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [v_cache release];
            [k_cache release];
            [q_src release];
            return metal_fail(error, error_size, "failed to create Metal decode attention command encoder");
        }

        uocr_metal_decode_attention_params params;
        memset(&params, 0, sizeof(params));
        params.batch_slots = batch_slots;
        params.cache_token_capacity = (uint32_t)cache_token_capacity_u64;
        params.layer = layer;
        params.slot = slot;
        params.prompt_length = prompt_length;
        params.generated_count = generated_count;
        params.attention_length = plan.attention_length;
        params.first_generated = plan.first_generated_index;
        params.heads = UOCR_ATTENTION_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.ring_window = UOCR_GENERATED_RING_WINDOW;
        params.scale = 1.0f / sqrtf((float)UOCR_HEAD_DIM);

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_src offset:0u atIndex:0u];
        [enc setBuffer:k_cache offset:0u atIndex:1u];
        [enc setBuffer:v_cache offset:0u atIndex:2u];
        [enc setBuffer:dst offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc setThreadgroupMemoryLength:(NSUInteger)threadgroup_bytes atIndex:0u];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)UOCR_ATTENTION_HEADS, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [dst release];
            [v_cache release];
            [k_cache release];
            [q_src release];
            return metal_fail(error, error_size, "Metal decode attention command failed: %s", [description UTF8String]);
        }

        memcpy(out, [dst contents], (size_t)output_bytes);
        [dst release];
        [v_cache release];
        [k_cache release];
        [q_src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

int uocr_metal_context_write_kv_cache_f16(uocr_metal_context *ctx,
                                          const uint16_t *k_f16,
                                          const uint16_t *v_f16,
                                          const uint16_t *initial_k_cache_f16_or_null,
                                          const uint16_t *initial_v_cache_f16_or_null,
                                          uint32_t n_tokens,
                                          uint32_t batch_slots,
                                          uint32_t prompt_token_capacity,
                                          uint32_t layer,
                                          uint32_t slot,
                                          uint32_t prompt_length,
                                          uint32_t position_start,
                                          uint16_t *k_cache_out_f16,
                                          uint16_t *v_cache_out_f16,
                                          char *error,
                                          size_t error_size) {
    metal_clear_error(error, error_size);
    if (ctx == NULL || k_f16 == NULL || v_f16 == NULL || k_cache_out_f16 == NULL || v_cache_out_f16 == NULL ||
        n_tokens == 0u || batch_slots == 0u || prompt_token_capacity == 0u) {
        return metal_fail(error, error_size, "invalid Metal KV cache write request");
    }
    if (layer >= UOCR_DECODER_LAYERS || slot >= batch_slots) {
        return metal_fail(error,
                          error_size,
                          "Metal KV cache layer/slot out of range: layer=%u slot=%u batch_slots=%u",
                          layer,
                          slot,
                          batch_slots);
    }
    if (prompt_length == 0u || prompt_length > prompt_token_capacity) {
        return metal_fail(error,
                          error_size,
                          "invalid Metal KV cache layout: prompt_length=%u prompt_token_capacity=%u",
                          prompt_length,
                          prompt_token_capacity);
    }

    uint64_t cache_token_capacity_u64 = 0u;
    if (!checked_add_u64((uint64_t)prompt_token_capacity,
                         (uint64_t)UOCR_GENERATED_RING_WINDOW,
                         &cache_token_capacity_u64) ||
        cache_token_capacity_u64 > (uint64_t)UINT32_MAX ||
        (uint64_t)prompt_length + (uint64_t)UOCR_GENERATED_RING_WINDOW > cache_token_capacity_u64) {
        return metal_fail(error, error_size, "Metal KV cache token capacity overflow");
    }
    const uint32_t cache_token_capacity = (uint32_t)cache_token_capacity_u64;

    uint64_t position_end = 0u;
    if (!checked_add_u64((uint64_t)position_start, (uint64_t)n_tokens, &position_end) ||
        position_end > (uint64_t)UOCR_MAX_POSITIONS) {
        return metal_fail(error,
                          error_size,
                          "Metal KV cache position range [%u,%llu) exceeds max positions %u",
                          position_start,
                          (unsigned long long)position_end,
                          UOCR_MAX_POSITIONS);
    }

    uint64_t src_values = 0u;
    uint64_t src_bytes = 0u;
    uint64_t cache_values = 0u;
    uint64_t cache_bytes = 0u;
    uint64_t layer_slot_tokens = 0u;
    if (!checked_mul_u64((uint64_t)n_tokens, (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM, &src_values) ||
        !checked_mul_u64(src_values, 2u, &src_bytes) ||
        !checked_mul_u64((uint64_t)UOCR_DECODER_LAYERS, (uint64_t)batch_slots, &layer_slot_tokens) ||
        !checked_mul_u64(layer_slot_tokens, cache_token_capacity_u64, &cache_values) ||
        !checked_mul_u64(cache_values, (uint64_t)UOCR_KV_HEADS * (uint64_t)UOCR_HEAD_DIM, &cache_values) ||
        !checked_mul_u64(cache_values, 2u, &cache_bytes)) {
        return metal_fail(error, error_size, "Metal KV cache write byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (src_bytes > max_buffer_length || cache_bytes > max_buffer_length || src_bytes > (uint64_t)SIZE_MAX ||
        cache_bytes > (uint64_t)SIZE_MAX || src_values > (uint64_t)UINT32_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal KV cache write buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_kv_cache_write_f16", error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> k_src = [ctx->device newBufferWithBytes:k_f16
                                                       length:(NSUInteger)src_bytes
                                                      options:MTLResourceStorageModeShared];
        if (k_src == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal KV K input buffer");
        }
        k_src.label = @"uocr_kv_write_k_input_f16";

        id<MTLBuffer> v_src = [ctx->device newBufferWithBytes:v_f16
                                                       length:(NSUInteger)src_bytes
                                                      options:MTLResourceStorageModeShared];
        if (v_src == nil) {
            [k_src release];
            return metal_fail(error, error_size, "failed to allocate Metal KV V input buffer");
        }
        v_src.label = @"uocr_kv_write_v_input_f16";

        id<MTLBuffer> k_cache = nil;
        if (initial_k_cache_f16_or_null != NULL) {
            k_cache = [ctx->device newBufferWithBytes:initial_k_cache_f16_or_null
                                               length:(NSUInteger)cache_bytes
                                              options:MTLResourceStorageModeShared];
        } else {
            k_cache = [ctx->device newBufferWithLength:(NSUInteger)cache_bytes options:MTLResourceStorageModeShared];
            if (k_cache != nil) {
                memset([k_cache contents], 0, (size_t)cache_bytes);
            }
        }
        if (k_cache == nil) {
            [v_src release];
            [k_src release];
            return metal_fail(error, error_size, "failed to allocate Metal KV K cache buffer");
        }
        k_cache.label = @"uocr_kv_write_k_cache_f16";

        id<MTLBuffer> v_cache = nil;
        if (initial_v_cache_f16_or_null != NULL) {
            v_cache = [ctx->device newBufferWithBytes:initial_v_cache_f16_or_null
                                               length:(NSUInteger)cache_bytes
                                              options:MTLResourceStorageModeShared];
        } else {
            v_cache = [ctx->device newBufferWithLength:(NSUInteger)cache_bytes options:MTLResourceStorageModeShared];
            if (v_cache != nil) {
                memset([v_cache contents], 0, (size_t)cache_bytes);
            }
        }
        if (v_cache == nil) {
            [k_cache release];
            [v_src release];
            [k_src release];
            return metal_fail(error, error_size, "failed to allocate Metal KV V cache buffer");
        }
        v_cache.label = @"uocr_kv_write_v_cache_f16";

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [v_cache release];
            [k_cache release];
            [v_src release];
            [k_src release];
            return metal_fail(error, error_size, "failed to create Metal KV cache write command buffer");
        }
        cb.label = @"uocr_kv_cache_write_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [v_cache release];
            [k_cache release];
            [v_src release];
            [k_src release];
            return metal_fail(error, error_size, "failed to create Metal KV cache write command encoder");
        }

        uocr_metal_kv_cache_write_params params;
        params.n_tokens = n_tokens;
        params.batch_slots = batch_slots;
        params.cache_token_capacity = cache_token_capacity;
        params.layer = layer;
        params.slot = slot;
        params.prompt_length = prompt_length;
        params.position_start = position_start;
        params.heads = UOCR_KV_HEADS;
        params.head_dim = UOCR_HEAD_DIM;
        params.ring_window = UOCR_GENERATED_RING_WINDOW;
        params.reserved0 = 0u;
        params.reserved1 = 0u;

        NSUInteger threads_per_group = pipeline.threadExecutionWidth;
        if (threads_per_group == 0u) {
            threads_per_group = 1u;
        }
        if (threads_per_group > 256u) {
            threads_per_group = 256u;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:k_src offset:0u atIndex:0u];
        [enc setBuffer:v_src offset:0u atIndex:1u];
        [enc setBuffer:k_cache offset:0u atIndex:2u];
        [enc setBuffer:v_cache offset:0u atIndex:3u];
        [enc setBytes:&params length:sizeof(params) atIndex:4u];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)src_values, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
            [v_cache release];
            [k_cache release];
            [v_src release];
            [k_src release];
            return metal_fail(error, error_size, "Metal KV cache write command failed: %s", [description UTF8String]);
        }

        memcpy(k_cache_out_f16, [k_cache contents], (size_t)cache_bytes);
        memcpy(v_cache_out_f16, [v_cache contents], (size_t)cache_bytes);
        [v_cache release];
        [k_cache release];
        [v_src release];
        [k_src release];
    }

    metal_clear_error(error, error_size);
    return 1;
}

void uocr_metal_context_destroy(uocr_metal_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    uocr_metal_context_unmap_model(ctx);
    uocr_metal_context_release_runtime_arenas(ctx);
    release_all_scratch(ctx);
    @autoreleasepool {
        [ctx->transient_retains release];
        [ctx->pipeline_cache release];
        [ctx->library release];
        [ctx->queue release];
        [ctx->device release];
        free(ctx);
    }
}

static int run_scratch_smoke(uocr_metal_context *ctx, char *error, size_t error_size) {
    if (!uocr_metal_context_ensure_scratch(ctx, UOCR_METAL_SCRATCH_TRANSIENT, 4096u, 1, error, error_size)) {
        return 0;
    }
    if (uocr_metal_context_scratch_capacity(ctx, UOCR_METAL_SCRATCH_TRANSIENT) < 4096u) {
        return metal_fail(error, error_size, "Metal scratch buffer capacity did not satisfy request");
    }
    return 1;
}

static size_t round_up_size(size_t value, size_t align) {
    if (align == 0u) {
        return value;
    }
    const size_t remainder = value % align;
    return remainder == 0u ? value : value + (align - remainder);
}

static int run_no_copy_kernel_smoke(uocr_metal_context *ctx, char *error, size_t error_size) {
    enum { N = 16 };
    const size_t bytes = (size_t)N * sizeof(uint32_t);
    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        page_size_long = 4096;
    }
    const size_t page_size = (size_t)page_size_long;
    const size_t map_size = round_up_size(bytes, page_size);

    char path[] = "/tmp/uocr-metal-smoke-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        return metal_fail(error, error_size, "mkstemp failed: %s", strerror(errno));
    }
    (void)unlink(path);
    if (ftruncate(fd, (off_t)map_size) != 0) {
        int saved_errno = errno;
        (void)close(fd);
        return metal_fail(error, error_size, "ftruncate failed: %s", strerror(saved_errno));
    }

    void *mapping = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        int saved_errno = errno;
        (void)close(fd);
        return metal_fail(error, error_size, "mmap failed: %s", strerror(saved_errno));
    }

    uint32_t *src_words = (uint32_t *)mapping;
    for (uint32_t i = 0u; i < (uint32_t)N; ++i) {
        src_words[i] = 0x1000u + i;
    }

    id<MTLBuffer> src = [ctx->device newBufferWithBytesNoCopy:mapping
                                                        length:map_size
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
    if (src == nil) {
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return metal_fail(error, error_size, "newBufferWithBytesNoCopy failed for mmap range");
    }
    src.label = @"uocr_mmap_no_copy_smoke_src";

    id<MTLBuffer> dst = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (dst == nil) {
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return metal_fail(error, error_size, "failed to allocate Metal readback buffer");
    }
    dst.label = @"uocr_smoke_dst";
    memset([dst contents], 0, bytes);

    id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_smoke_u32", error, error_size);
    if (pipeline == nil) {
        [dst release];
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return 0;
    }

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    if (cb == nil) {
        [dst release];
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return metal_fail(error, error_size, "failed to create Metal command buffer");
    }
    cb.label = @"uocr_smoke_command_buffer";

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (enc == nil) {
        [dst release];
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return metal_fail(error, error_size, "failed to create Metal command encoder");
    }
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:src offset:0u atIndex:0u];
    [enc setBuffer:dst offset:0u atIndex:1u];
    uint32_t n = (uint32_t)N;
    [enc setBytes:&n length:sizeof(n) atIndex:2u];

    NSUInteger threads_per_group = pipeline.threadExecutionWidth;
    if (threads_per_group == 0u) {
        threads_per_group = 1u;
    }
    if (threads_per_group > (NSUInteger)N) {
        threads_per_group = (NSUInteger)N;
    }
    [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1u, 1u)
   threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
    [enc endEncoding];

    const uint32_t transient_count_before = metal_inflight_transient_count(ctx);
    if (!metal_retain_transient_until_completed(ctx, cb, src, error, error_size)) {
        [dst release];
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return 0;
    }
    if (metal_inflight_transient_count(ctx) != transient_count_before + 1u) {
        [dst release];
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return metal_fail(error, error_size, "Metal transient retain tracking count did not increase");
    }

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        NSString *description = cb.error != nil ? [cb.error localizedDescription] : @"unknown command-buffer error";
        [dst release];
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return metal_fail(error, error_size, "Metal smoke command failed: %s", [description UTF8String]);
    }
    if (metal_inflight_transient_count(ctx) != transient_count_before) {
        [dst release];
        [src release];
        (void)munmap(mapping, map_size);
        (void)close(fd);
        return metal_fail(error, error_size, "Metal transient retain tracking count did not drain after command completion");
    }

    const uint32_t *dst_words = (const uint32_t *)[dst contents];
    int ok = 1;
    for (uint32_t i = 0u; i < (uint32_t)N; ++i) {
        const uint32_t expected = (0x1000u + i) ^ 0xa5a5a5a5u;
        if (dst_words[i] != expected) {
            ok = 0;
            (void)metal_fail(error,
                             error_size,
                             "Metal smoke mismatch at %u: got 0x%08x expected 0x%08x",
                             i,
                             dst_words[i],
                             expected);
            break;
        }
    }

    [dst release];
    [src release];
    (void)munmap(mapping, map_size);
    (void)close(fd);
    return ok;
}

int uocr_metal_smoke_test(const char *resource_path, char *error, size_t error_size) {
    metal_clear_error(error, error_size);
    @autoreleasepool {
        uocr_metal_context *ctx = uocr_metal_context_create(resource_path, error, error_size);
        if (ctx == NULL) {
            return 0;
        }

        int ok = run_scratch_smoke(ctx, error, error_size);
        if (ok) {
            ok = run_no_copy_kernel_smoke(ctx, error, error_size);
        }
        uocr_metal_context_destroy(ctx);
        if (ok) {
            metal_clear_error(error, error_size);
        }
        return ok;
    }
}
