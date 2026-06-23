#include "core/uocr_alloc.h"

#include <stdint.h>
#include <string.h>

#define CHECK(expr)                    \
    do {                               \
        if (!(expr)) {                 \
            return __LINE__;           \
        }                              \
    } while (0)

static int stats_are_balanced(void) {
    uocr_alloc_stats stats;
    uocr_alloc_get_stats(&stats);
    return stats.live_bytes == 0u && stats.total_allocated_bytes == stats.total_freed_bytes;
}

static int test_alloc_free_stats(void) {
    uocr_alloc_reset_for_tests();

    void *ptr = uocr_malloc(64u);
    CHECK(ptr != NULL);

    uocr_alloc_stats stats;
    uocr_alloc_get_stats(&stats);
    CHECK(stats.live_bytes == 64u);
    CHECK(stats.peak_bytes == 64u);
    CHECK(stats.total_allocated_bytes == 64u);
    CHECK(stats.allocation_count == 1u);
    CHECK(stats.free_count == 0u);

    uocr_free(ptr);
    uocr_alloc_get_stats(&stats);
    CHECK(stats.live_bytes == 0u);
    CHECK(stats.peak_bytes == 64u);
    CHECK(stats.total_allocated_bytes == 64u);
    CHECK(stats.total_freed_bytes == 64u);
    CHECK(stats.free_count == 1u);
    CHECK(stats_are_balanced());
    return 0;
}

static int test_zeroed_allocations(void) {
    uocr_alloc_reset_for_tests();

    uint8_t *ptr = (uint8_t *)uocr_malloc_zeroed(128u);
    CHECK(ptr != NULL);
    for (uint32_t i = 0u; i < 128u; ++i) {
        CHECK(ptr[i] == 0u);
        ptr[i] = (uint8_t)(i + 1u);
    }
    uocr_free(ptr);

    uint32_t *words = (uint32_t *)uocr_calloc(16u, sizeof(uint32_t));
    CHECK(words != NULL);
    for (uint32_t i = 0u; i < 16u; ++i) {
        CHECK(words[i] == 0u);
    }
    uocr_free(words);
    CHECK(stats_are_balanced());
    return 0;
}

static int test_realloc_stats_and_contents(void) {
    uocr_alloc_reset_for_tests();

    uint8_t *ptr = (uint8_t *)uocr_malloc(8u);
    CHECK(ptr != NULL);
    for (uint32_t i = 0u; i < 8u; ++i) {
        ptr[i] = (uint8_t)(0xa0u + i);
    }

    ptr = (uint8_t *)uocr_realloc(ptr, 32u);
    CHECK(ptr != NULL);
    for (uint32_t i = 0u; i < 8u; ++i) {
        CHECK(ptr[i] == (uint8_t)(0xa0u + i));
    }

    uocr_alloc_stats stats;
    uocr_alloc_get_stats(&stats);
    CHECK(stats.live_bytes == 32u);
    CHECK(stats.peak_bytes == 32u);
    CHECK(stats.total_allocated_bytes == 32u);

    ptr = (uint8_t *)uocr_realloc(ptr, 4u);
    CHECK(ptr != NULL);
    for (uint32_t i = 0u; i < 4u; ++i) {
        CHECK(ptr[i] == (uint8_t)(0xa0u + i));
    }
    uocr_alloc_get_stats(&stats);
    CHECK(stats.live_bytes == 4u);
    CHECK(stats.peak_bytes == 32u);
    CHECK(stats.total_allocated_bytes == 32u);
    CHECK(stats.total_freed_bytes == 28u);

    ptr = (uint8_t *)uocr_realloc(ptr, 0u);
    CHECK(ptr == NULL);
    CHECK(stats_are_balanced());
    return 0;
}

static int test_overflow_and_guard(void) {
    uocr_alloc_reset_for_tests();

    const uocr_alloc_guard guard = uocr_alloc_guard_begin();
    CHECK(uocr_alloc_guard_end_no_alloc(&guard));

    void *overflow = uocr_calloc((size_t)-1, 2u);
    CHECK(overflow == NULL);
    CHECK(!uocr_alloc_guard_end_no_alloc(&guard));

    uocr_alloc_stats stats;
    uocr_alloc_get_stats(&stats);
    CHECK(stats.failed_allocation_count == 1u);
    CHECK(stats.live_bytes == 0u);
    return 0;
}

int main(void) {
    int status = 0;
    if ((status = test_alloc_free_stats()) != 0) return status;
    if ((status = test_zeroed_allocations()) != 0) return status;
    if ((status = test_realloc_stats_and_contents()) != 0) return status;
    if ((status = test_overflow_and_guard()) != 0) return status;
    return 0;
}
