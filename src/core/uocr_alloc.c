#include "core/uocr_alloc.h"

#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define UOCR_ALLOC_MAGIC 0x554f4352414c4c43ull /* "UOCRALLC" */
#define UOCR_ALLOC_FREED_MAGIC 0x554f435246524545ull /* "UOCRFREE" */

typedef union uocr_alloc_header {
    struct {
        uint64_t magic;
        size_t size;
    } meta;
    max_align_t align;
} uocr_alloc_header;

static atomic_uint_fast64_t g_live_bytes;
static atomic_uint_fast64_t g_peak_bytes;
static atomic_uint_fast64_t g_total_allocated_bytes;
static atomic_uint_fast64_t g_total_freed_bytes;
static atomic_uint_fast64_t g_allocation_count;
static atomic_uint_fast64_t g_free_count;
static atomic_uint_fast64_t g_failed_allocation_count;

static int checked_allocation_size(size_t payload_size, size_t *total_size) {
    if (payload_size > SIZE_MAX - sizeof(uocr_alloc_header)) {
        return 0;
    }
    *total_size = sizeof(uocr_alloc_header) + payload_size;
    return 1;
}

static void record_allocation(size_t size) {
    atomic_fetch_add_explicit(&g_allocation_count, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_total_allocated_bytes, (uint64_t)size, memory_order_relaxed);
    const uint64_t live = atomic_fetch_add_explicit(&g_live_bytes, (uint64_t)size, memory_order_relaxed) + (uint64_t)size;

    uint64_t peak = atomic_load_explicit(&g_peak_bytes, memory_order_relaxed);
    while (live > peak && !atomic_compare_exchange_weak_explicit(&g_peak_bytes,
                                                                  &peak,
                                                                  live,
                                                                  memory_order_relaxed,
                                                                  memory_order_relaxed)) {
    }
}

static void record_free(size_t size) {
    atomic_fetch_add_explicit(&g_free_count, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_total_freed_bytes, (uint64_t)size, memory_order_relaxed);
    atomic_fetch_sub_explicit(&g_live_bytes, (uint64_t)size, memory_order_relaxed);
}

static void record_failure(void) {
    atomic_fetch_add_explicit(&g_failed_allocation_count, 1u, memory_order_relaxed);
}

void *uocr_malloc(size_t size) {
    size_t total_size = 0u;
    if (!checked_allocation_size(size, &total_size)) {
        record_failure();
        return NULL;
    }

    uocr_alloc_header *header = (uocr_alloc_header *)malloc(total_size == 0u ? 1u : total_size);
    if (header == NULL) {
        record_failure();
        return NULL;
    }
    header->meta.magic = UOCR_ALLOC_MAGIC;
    header->meta.size = size;
    record_allocation(size);
    return (void *)(header + 1u);
}

void *uocr_malloc_zeroed(size_t size) {
    void *ptr = uocr_malloc(size);
    if (ptr != NULL && size != 0u) {
        memset(ptr, 0, size);
    }
    return ptr;
}

void *uocr_calloc(size_t count, size_t size) {
    if (count != 0u && size > SIZE_MAX / count) {
        record_failure();
        return NULL;
    }
    return uocr_malloc_zeroed(count * size);
}

void *uocr_realloc(void *ptr, size_t size) {
    if (ptr == NULL) {
        return uocr_malloc(size);
    }
    if (size == 0u) {
        uocr_free(ptr);
        return NULL;
    }

    uocr_alloc_header *old_header = ((uocr_alloc_header *)ptr) - 1u;
    if (old_header->meta.magic != UOCR_ALLOC_MAGIC) {
        record_failure();
        return NULL;
    }

    const size_t old_size = old_header->meta.size;
    size_t total_size = 0u;
    if (!checked_allocation_size(size, &total_size)) {
        record_failure();
        return NULL;
    }

    uocr_alloc_header *new_header = (uocr_alloc_header *)realloc(old_header, total_size);
    if (new_header == NULL) {
        record_failure();
        return NULL;
    }
    new_header->meta.magic = UOCR_ALLOC_MAGIC;
    new_header->meta.size = size;

    atomic_fetch_add_explicit(&g_allocation_count, 1u, memory_order_relaxed);
    if (size >= old_size) {
        const size_t delta = size - old_size;
        atomic_fetch_add_explicit(&g_total_allocated_bytes, (uint64_t)delta, memory_order_relaxed);
        const uint64_t live = atomic_fetch_add_explicit(&g_live_bytes, (uint64_t)delta, memory_order_relaxed) + (uint64_t)delta;
        uint64_t peak = atomic_load_explicit(&g_peak_bytes, memory_order_relaxed);
        while (live > peak && !atomic_compare_exchange_weak_explicit(&g_peak_bytes,
                                                                      &peak,
                                                                      live,
                                                                      memory_order_relaxed,
                                                                      memory_order_relaxed)) {
        }
    } else {
        const size_t delta = old_size - size;
        atomic_fetch_add_explicit(&g_total_freed_bytes, (uint64_t)delta, memory_order_relaxed);
        atomic_fetch_sub_explicit(&g_live_bytes, (uint64_t)delta, memory_order_relaxed);
    }
    return (void *)(new_header + 1u);
}

void uocr_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    uocr_alloc_header *header = ((uocr_alloc_header *)ptr) - 1u;
    if (header->meta.magic != UOCR_ALLOC_MAGIC) {
        return;
    }
    const size_t size = header->meta.size;
    header->meta.magic = UOCR_ALLOC_FREED_MAGIC;
    header->meta.size = 0u;
    record_free(size);
    free(header);
}

void uocr_alloc_get_stats(uocr_alloc_stats *out_stats) {
    if (out_stats == NULL) {
        return;
    }
    out_stats->live_bytes = atomic_load_explicit(&g_live_bytes, memory_order_relaxed);
    out_stats->peak_bytes = atomic_load_explicit(&g_peak_bytes, memory_order_relaxed);
    out_stats->total_allocated_bytes = atomic_load_explicit(&g_total_allocated_bytes, memory_order_relaxed);
    out_stats->total_freed_bytes = atomic_load_explicit(&g_total_freed_bytes, memory_order_relaxed);
    out_stats->allocation_count = atomic_load_explicit(&g_allocation_count, memory_order_relaxed);
    out_stats->free_count = atomic_load_explicit(&g_free_count, memory_order_relaxed);
    out_stats->failed_allocation_count = atomic_load_explicit(&g_failed_allocation_count, memory_order_relaxed);
}

void uocr_alloc_reset_for_tests(void) {
    atomic_store_explicit(&g_live_bytes, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_peak_bytes, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_total_allocated_bytes, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_total_freed_bytes, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_allocation_count, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_free_count, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_failed_allocation_count, 0u, memory_order_relaxed);
}

void uocr_alloc_reset_peak(void) {
    const uint64_t live = atomic_load_explicit(&g_live_bytes, memory_order_relaxed);
    atomic_store_explicit(&g_peak_bytes, live, memory_order_relaxed);
}

uocr_alloc_guard uocr_alloc_guard_begin(void) {
    uocr_alloc_guard guard;
    guard.allocation_count = atomic_load_explicit(&g_allocation_count, memory_order_relaxed);
    guard.failed_allocation_count = atomic_load_explicit(&g_failed_allocation_count, memory_order_relaxed);
    return guard;
}

int uocr_alloc_guard_end_no_alloc(const uocr_alloc_guard *guard) {
    if (guard == NULL) {
        return 0;
    }
    return atomic_load_explicit(&g_allocation_count, memory_order_relaxed) == guard->allocation_count &&
           atomic_load_explicit(&g_failed_allocation_count, memory_order_relaxed) == guard->failed_allocation_count;
}
