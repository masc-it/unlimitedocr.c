#include "backend/metal/uocr_metal.h"

#include "runtime/uocr_memory.h"

#include <errno.h>
#include <limits.h>
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

int uocr_metal_context_get_tensor_binding(const uocr_metal_context *ctx,
                                          uint32_t tensor_id,
                                          uocr_metal_tensor_binding *out_binding) {
    if (ctx == NULL || out_binding == NULL || tensor_id == 0u) {
        return 0;
    }
    uint32_t lo = 0u;
    uint32_t hi = ctx->tensor_binding_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        const uint32_t mid_id = ctx->tensor_bindings[mid].tensor_id;
        if (mid_id == tensor_id) {
            out_binding->tensor_id = ctx->tensor_bindings[mid].tensor_id;
            out_binding->view_index = ctx->tensor_bindings[mid].view_index;
            out_binding->inner_offset = ctx->tensor_bindings[mid].inner_offset;
            out_binding->payload_size = ctx->tensor_bindings[mid].payload_size;
            return 1;
        }
        if (mid_id < tensor_id) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return 0;
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

    const int has_image_span = image_span_length != 0u;
    if (has_image_span) {
        uint64_t image_end = 0u;
        if (image_span_start == UINT32_MAX ||
            !checked_add_u64((uint64_t)image_span_start, (uint64_t)image_span_length, &image_end) ||
            image_span_start >= n_tokens || image_end > (uint64_t)n_tokens) {
            return metal_fail(error,
                              error_size,
                              "Metal prompt image span [%u,%llu) is outside %u tokens",
                              image_span_start,
                              (unsigned long long)image_end,
                              n_tokens);
        }
        if (image_features_f16 == NULL) {
            return metal_fail(error, error_size, "Metal prompt image span requires image features");
        }
    } else if (image_span_start != UINT32_MAX) {
        return metal_fail(error, error_size, "Metal text-only prompt should use UINT32_MAX image span start");
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
        return metal_fail(error, error_size, "Metal prompt assembly byte-size overflow");
    }

    const uint64_t max_buffer_length = metal_device_max_buffer_length(ctx->device);
    if (table_bytes > max_buffer_length || token_bytes > max_buffer_length || output_bytes > max_buffer_length ||
        image_bytes > max_buffer_length || table_bytes > (uint64_t)SIZE_MAX || token_bytes > (uint64_t)SIZE_MAX ||
        output_bytes > (uint64_t)SIZE_MAX || image_bytes > (uint64_t)SIZE_MAX) {
        return metal_fail(error,
                          error_size,
                          "Metal prompt assembly buffers exceed maxBufferLength %llu",
                          (unsigned long long)max_buffer_length);
    }

    @autoreleasepool {
        const char *function_name = has_image_span ? "uocr_assemble_prompt_with_image_f16" : "uocr_assemble_prompt_text_f16";
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, function_name, error, error_size);
        if (pipeline == nil) {
            return 0;
        }

        id<MTLBuffer> table = [ctx->device newBufferWithBytes:embedding_table_f16
                                                       length:(NSUInteger)table_bytes
                                                      options:MTLResourceStorageModeShared];
        if (table == nil) {
            return metal_fail(error, error_size, "failed to allocate Metal prompt embedding table buffer");
        }
        table.label = @"uocr_prompt_embedding_table_f16";

        id<MTLBuffer> tokens = [ctx->device newBufferWithBytes:input_ids
                                                        length:(NSUInteger)token_bytes
                                                       options:MTLResourceStorageModeShared];
        if (tokens == nil) {
            [table release];
            return metal_fail(error, error_size, "failed to allocate Metal prompt token-id buffer");
        }
        tokens.label = @"uocr_prompt_input_ids";

        id<MTLBuffer> image_features = nil;
        if (has_image_span) {
            image_features = [ctx->device newBufferWithBytes:image_features_f16
                                                      length:(NSUInteger)image_bytes
                                                     options:MTLResourceStorageModeShared];
            if (image_features == nil) {
                [tokens release];
                [table release];
                return metal_fail(error, error_size, "failed to allocate Metal prompt image-feature buffer");
            }
            image_features.label = @"uocr_prompt_image_features_f16";
        }

        id<MTLBuffer> dst = [ctx->device newBufferWithLength:(NSUInteger)output_bytes
                                                     options:MTLResourceStorageModeShared];
        if (dst == nil) {
            [image_features release];
            [tokens release];
            [table release];
            return metal_fail(error, error_size, "failed to allocate Metal prompt output buffer");
        }
        dst.label = @"uocr_prompt_embeddings_f16";
        memset([dst contents], 0, (size_t)output_bytes);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            [dst release];
            [image_features release];
            [tokens release];
            [table release];
            return metal_fail(error, error_size, "failed to create Metal prompt assembly command buffer");
        }
        cb.label = @"uocr_prompt_assembly_command_buffer";

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            [dst release];
            [image_features release];
            [tokens release];
            [table release];
            return metal_fail(error, error_size, "failed to create Metal prompt assembly command encoder");
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
        [enc setBuffer:table offset:0u atIndex:0u];
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
            [table release];
            return metal_fail(error, error_size, "Metal prompt assembly command failed: %s", [description UTF8String]);
        }

        memcpy(out_prompt_f16, [dst contents], (size_t)output_bytes);
        [dst release];
        [image_features release];
        [tokens release];
        [table release];
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
