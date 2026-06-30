#include "runtime/uocr_profile.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static int profile_truthy_env_value(const char *value) {
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    if (strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0 ||
        strcmp(value, "off") == 0 || strcmp(value, "OFF") == 0 || strcmp(value, "no") == 0 ||
        strcmp(value, "NO") == 0) {
        return 0;
    }
    return 1;
}

int uocr_profile_env_enabled(void) {
    return profile_truthy_env_value(getenv("UOCR_PROFILE"));
}

uint64_t uocr_profile_now_ns(void) {
#if defined(_WIN32)
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    if (frequency.QuadPart <= 0) {
        return 0u;
    }
    return (uint64_t)((counter.QuadPart * 1000000000ull) / frequency.QuadPart);
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0u;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#else
    return (uint64_t)clock() * (1000000000ull / (uint64_t)CLOCKS_PER_SEC);
#endif
}

double uocr_profile_elapsed_ms(uint64_t start_ns, uint64_t end_ns) {
    if (end_ns < start_ns) {
        return 0.0;
    }
    return (double)(end_ns - start_ns) / 1000000.0;
}

void uocr_profile_state_init(uocr_profile_state *profile, int enabled) {
    if (profile == NULL) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    profile->report.enabled = enabled ? 1u : 0u;
}

void uocr_profile_reset(uocr_profile_state *profile) {
    if (profile == NULL) {
        return;
    }
    const uint32_t enabled = profile->report.enabled;
    const uint64_t next_generation_index = profile->report.generation_index;
    memset(&profile->report, 0, sizeof(profile->report));
    profile->report.enabled = enabled;
    profile->report.generation_index = next_generation_index;
}

void uocr_profile_begin_request(uocr_profile_state *profile) {
    if (profile == NULL || profile->report.enabled == 0u) {
        return;
    }
    const uint64_t next_generation_index = profile->report.generation_index + 1u;
    memset(&profile->report, 0, sizeof(profile->report));
    profile->report.enabled = 1u;
    profile->report.generation_index = next_generation_index;
}

int uocr_profile_is_enabled(const uocr_profile_state *profile) {
    return profile != NULL && profile->report.enabled != 0u;
}

static uocr_profile_event *find_or_add_event(uocr_profile_state *profile, const char *name) {
    if (!uocr_profile_is_enabled(profile) || name == NULL || name[0] == '\0') {
        return NULL;
    }
    for (uint32_t i = 0u; i < profile->report.event_count; ++i) {
        uocr_profile_event *event = &profile->report.events[i];
        if (strncmp(event->name, name, sizeof(event->name)) == 0) {
            return event;
        }
    }
    if (profile->report.event_count >= UOCR_PROFILE_MAX_EVENTS) {
        profile->report.dropped_event_count += 1u;
        return NULL;
    }
    uocr_profile_event *event = &profile->report.events[profile->report.event_count++];
    memset(event, 0, sizeof(*event));
    (void)snprintf(event->name, sizeof(event->name), "%s", name);
    event->min_ms = 0.0;
    event->max_ms = 0.0;
    return event;
}

void uocr_profile_add_event_ms(uocr_profile_state *profile, const char *name, double elapsed_ms) {
    if (elapsed_ms < 0.0) {
        elapsed_ms = 0.0;
    }
    uocr_profile_event *event = find_or_add_event(profile, name);
    if (event == NULL) {
        return;
    }
    event->calls += 1u;
    event->total_ms += elapsed_ms;
    if (event->calls == 1u || elapsed_ms < event->min_ms) {
        event->min_ms = elapsed_ms;
    }
    if (elapsed_ms > event->max_ms) {
        event->max_ms = elapsed_ms;
    }
}

void uocr_profile_add_event_ns(uocr_profile_state *profile, const char *name, uint64_t start_ns, uint64_t end_ns) {
    uocr_profile_add_event_ms(profile, name, uocr_profile_elapsed_ms(start_ns, end_ns));
}

void uocr_profile_add_event_now(uocr_profile_state *profile, const char *name, uint64_t start_ns) {
    if (!uocr_profile_is_enabled(profile)) {
        return;
    }
    uocr_profile_add_event_ns(profile, name, start_ns, uocr_profile_now_ns());
}

void uocr_profile_add_event_ms_f(uocr_profile_state *profile, double elapsed_ms, const char *fmt, ...) {
    if (!uocr_profile_is_enabled(profile) || fmt == NULL) {
        return;
    }
    char name[UOCR_PROFILE_EVENT_NAME_SIZE];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    name[sizeof(name) - 1u] = '\0';
    uocr_profile_add_event_ms(profile, name, elapsed_ms);
}

void uocr_profile_add_event_now_f(uocr_profile_state *profile, uint64_t start_ns, const char *fmt, ...) {
    if (!uocr_profile_is_enabled(profile) || fmt == NULL) {
        return;
    }
    char name[UOCR_PROFILE_EVENT_NAME_SIZE];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    name[sizeof(name) - 1u] = '\0';
    uocr_profile_add_event_ms(profile, name, uocr_profile_elapsed_ms(start_ns, uocr_profile_now_ns()));
}

void uocr_profile_add_metal_buffer_allocation(uocr_profile_state *profile, uint64_t bytes) {
    if (!uocr_profile_is_enabled(profile)) {
        return;
    }
    profile->report.metal_buffer_allocation_count += 1u;
    profile->report.metal_buffer_allocation_bytes += bytes;
}

void uocr_profile_add_metal_command_buffer(uocr_profile_state *profile) {
    if (uocr_profile_is_enabled(profile)) {
        profile->report.metal_command_buffer_count += 1u;
    }
}

void uocr_profile_add_metal_command_encoder(uocr_profile_state *profile) {
    if (uocr_profile_is_enabled(profile)) {
        profile->report.metal_command_encoder_count += 1u;
    }
}

void uocr_profile_add_metal_command_buffer_wait(uocr_profile_state *profile) {
    if (uocr_profile_is_enabled(profile)) {
        profile->report.metal_command_buffer_wait_count += 1u;
    }
}

void uocr_profile_add_metal_mps_descriptor(uocr_profile_state *profile) {
    if (uocr_profile_is_enabled(profile)) {
        profile->report.metal_mps_descriptor_count += 1u;
    }
}

void uocr_profile_add_metal_mps_ndarray(uocr_profile_state *profile) {
    if (uocr_profile_is_enabled(profile)) {
        profile->report.metal_mps_ndarray_count += 1u;
    }
}

void uocr_profile_add_metal_nsarray(uocr_profile_state *profile) {
    if (uocr_profile_is_enabled(profile)) {
        profile->report.metal_nsarray_count += 1u;
    }
}

void uocr_profile_add_metal_transient_retain_object(uocr_profile_state *profile) {
    if (uocr_profile_is_enabled(profile)) {
        profile->report.metal_transient_retain_object_count += 1u;
    }
}
