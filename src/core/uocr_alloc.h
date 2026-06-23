#ifndef UOCR_ALLOC_H
#define UOCR_ALLOC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uocr_alloc_stats {
    uint64_t live_bytes;
    uint64_t peak_bytes;
    uint64_t total_allocated_bytes;
    uint64_t total_freed_bytes;
    uint64_t allocation_count;
    uint64_t free_count;
    uint64_t failed_allocation_count;
} uocr_alloc_stats;

typedef struct uocr_alloc_guard {
    uint64_t allocation_count;
    uint64_t failed_allocation_count;
} uocr_alloc_guard;

void *uocr_malloc(size_t size);
void *uocr_calloc(size_t count, size_t size);
void *uocr_realloc(void *ptr, size_t size);
void *uocr_malloc_zeroed(size_t size);
void uocr_free(void *ptr);

void uocr_alloc_get_stats(uocr_alloc_stats *out_stats);
void uocr_alloc_reset_for_tests(void);
void uocr_alloc_reset_peak(void);

uocr_alloc_guard uocr_alloc_guard_begin(void);
int uocr_alloc_guard_end_no_alloc(const uocr_alloc_guard *guard);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_ALLOC_H */
