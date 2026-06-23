#include "backend/metal/uocr_metal.h"

#include <errno.h>
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

struct uocr_metal_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipeline_cache;
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

void uocr_metal_context_destroy(uocr_metal_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    @autoreleasepool {
        [ctx->pipeline_cache release];
        [ctx->library release];
        [ctx->queue release];
        [ctx->device release];
        free(ctx);
    }
}

static int run_scratch_smoke(uocr_metal_context *ctx, char *error, size_t error_size) {
    id<MTLBuffer> scratch = [ctx->device newBufferWithLength:4096u options:MTLResourceStorageModeShared];
    if (scratch == nil) {
        return metal_fail(error, error_size, "failed to allocate Metal scratch buffer");
    }
    scratch.label = @"uocr_scratch_smoke";
    memset([scratch contents], 0, 4096u);
    [scratch release];
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
