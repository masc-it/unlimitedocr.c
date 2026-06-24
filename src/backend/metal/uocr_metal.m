#include "backend/metal/uocr_metal.h"

#include "model/uocr_constants.h"
#include "model/uocr_tensor_registry.h"
#include "runtime/uocr_memory.h"

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
    uocr_metal_kv_cache_layout kv_cache_layout;
    int has_kv_cache_layout;
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

#if UOCR_METAL_RUNTIME_COMPILE
static NSString *string_from_c_or_nil(const char *s) {
    if (s == NULL || s[0] == '\0') {
        return nil;
    }
    return [NSString stringWithUTF8String:s];
}

static void add_source_candidates(NSMutableArray<NSString *> *paths, NSString *root) {
    if (root == nil || [root length] == 0u) {
        return;
    }
    [paths addObject:[root stringByAppendingPathComponent:@"kernels/uocr_smoke.metal"]];
    [paths addObject:[root stringByAppendingPathComponent:@"uocr_smoke.metal"]];
    [paths addObject:[root stringByAppendingPathComponent:@"src/backend/metal/kernels/uocr_smoke.metal"]];
}
#endif

static NSString *load_runtime_source(const char *resource_path, char *error, size_t error_size) {
#if UOCR_METAL_RUNTIME_COMPILE
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
#else
    (void)resource_path;
    (void)metal_fail(error,
                     error_size,
                     "Metal runtime source compilation is disabled and precompiled metallib loading is not implemented yet");
    return nil;
#endif
}

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
                             "Metal shader compilation failed: %s",
                             [[compile_error localizedDescription] UTF8String]);
            uocr_metal_context_destroy(ctx);
            return NULL;
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
    metal_clear_error(error, error_size);
    return 1;
}

uint32_t uocr_metal_context_model_view_count(const uocr_metal_context *ctx) {
    return ctx != NULL ? ctx->model_view_count : 0u;
}

uint32_t uocr_metal_context_tensor_binding_count(const uocr_metal_context *ctx) {
    return ctx != NULL ? ctx->tensor_binding_count : 0u;
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
    status = uocr_estimate_vision_scratch_bytes(&capacities[UOCR_METAL_ARENA_VISION_SCRATCH]);
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
    if (token_bytes > max_buffer_length || output_bytes > max_buffer_length || image_bytes > max_buffer_length ||
        token_bytes > (uint64_t)SIZE_MAX || output_bytes > (uint64_t)SIZE_MAX || image_bytes > (uint64_t)SIZE_MAX) {
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

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes
                                                     options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [image_features release];
            [tokens release];
            return metal_fail(error, error_size, "failed to allocate %s output buffer", op);
        }
        dst.label = @"uocr_prompt_embeddings_f16";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [image_features release];
            [tokens release];
            return metal_fail(error, error_size, "failed to create %s command buffer", op);
        }
        cb.label = @"uocr_prompt_assembly_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
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
            [enc setBuffer:dst offset:0u atIndex:3u];
            [enc setBytes:&params length:sizeof(params) atIndex:4u];
        } else {
            [enc setBuffer:dst offset:0u atIndex:2u];
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
            [dst release];
            [image_features release];
            [tokens release];
            return metal_fail(error, error_size, "%s command failed: %s", op, [description UTF8String]);
        }

        memcpy(out_prompt_f16, [dst contents], (size_t)output_bytes);
        [dst release];
        [image_features release];
        [tokens release];
    }

    metal_clear_error(error, error_size);
    return 1;
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
