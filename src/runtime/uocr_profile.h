#ifndef UOCR_RUNTIME_PROFILE_H
#define UOCR_RUNTIME_PROFILE_H

#include <stdint.h>
#include <string.h>

#include "unlimitedocr.h"

#ifndef UOCR_ENABLE_PROFILING
#define UOCR_ENABLE_PROFILING 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uocr_profile_state {
    uocr_profile_report report;
} uocr_profile_state;

#if UOCR_ENABLE_PROFILING

int uocr_profile_env_enabled(void);
uint64_t uocr_profile_now_ns(void);
double uocr_profile_elapsed_ms(uint64_t start_ns, uint64_t end_ns);

void uocr_profile_state_init(uocr_profile_state *profile, int enabled);
void uocr_profile_begin_request(uocr_profile_state *profile);
void uocr_profile_reset(uocr_profile_state *profile);
int uocr_profile_is_enabled(const uocr_profile_state *profile);

void uocr_profile_add_event_ms(uocr_profile_state *profile, const char *name, double elapsed_ms);
void uocr_profile_add_event_ns(uocr_profile_state *profile, const char *name, uint64_t start_ns, uint64_t end_ns);
void uocr_profile_add_event_now(uocr_profile_state *profile, const char *name, uint64_t start_ns);
void uocr_profile_add_event_ms_f(uocr_profile_state *profile, double elapsed_ms, const char *fmt, ...);
void uocr_profile_add_event_now_f(uocr_profile_state *profile, uint64_t start_ns, const char *fmt, ...);

void uocr_profile_add_metal_buffer_allocation(uocr_profile_state *profile, uint64_t bytes);
void uocr_profile_add_metal_command_buffer(uocr_profile_state *profile);
void uocr_profile_add_metal_command_encoder(uocr_profile_state *profile);
void uocr_profile_add_metal_command_buffer_wait(uocr_profile_state *profile);
void uocr_profile_add_metal_mps_descriptor(uocr_profile_state *profile);
void uocr_profile_add_metal_mps_ndarray(uocr_profile_state *profile);
void uocr_profile_add_metal_nsarray(uocr_profile_state *profile);
void uocr_profile_add_metal_transient_retain_object(uocr_profile_state *profile);

#else /* !UOCR_ENABLE_PROFILING */

static inline int uocr_profile_env_enabled(void) { return 0; }
static inline uint64_t uocr_profile_now_ns(void) { return 0; }
static inline double uocr_profile_elapsed_ms(uint64_t start_ns, uint64_t end_ns) {
    (void)start_ns; (void)end_ns; return 0.0;
}

static inline void uocr_profile_reset(uocr_profile_state *profile) {
    if (profile != NULL) {
        const uint64_t next_gen = profile->report.generation_index;
        memset(&profile->report, 0, sizeof(profile->report));
        profile->report.generation_index = next_gen;
        profile->report.enabled = 0;
    }
}
static inline void uocr_profile_state_init(uocr_profile_state *profile, int enabled) {
    if (profile != NULL) { uocr_profile_reset(profile); }
    (void)enabled;
}
static inline void uocr_profile_begin_request(uocr_profile_state *profile) {
    if (profile != NULL) { uocr_profile_reset(profile); }
}
static inline int uocr_profile_is_enabled(const uocr_profile_state *profile) {
    (void)profile; return 0;
}

static inline void uocr_profile_add_event_ms(uocr_profile_state *profile, const char *name, double elapsed_ms) {
    (void)profile; (void)name; (void)elapsed_ms;
}
static inline void uocr_profile_add_event_ns(uocr_profile_state *profile, const char *name, uint64_t start_ns, uint64_t end_ns) {
    (void)profile; (void)name; (void)start_ns; (void)end_ns;
}
static inline void uocr_profile_add_event_now(uocr_profile_state *profile, const char *name, uint64_t start_ns) {
    (void)profile; (void)name; (void)start_ns;
}
static inline void uocr_profile_add_event_ms_f(uocr_profile_state *profile, double elapsed_ms, const char *fmt, ...) {
    (void)profile; (void)elapsed_ms; (void)fmt;
}
static inline void uocr_profile_add_event_now_f(uocr_profile_state *profile, uint64_t start_ns, const char *fmt, ...) {
    (void)profile; (void)start_ns; (void)fmt;
}

static inline void uocr_profile_add_metal_buffer_allocation(uocr_profile_state *profile, uint64_t bytes) {
    (void)profile; (void)bytes;
}
static inline void uocr_profile_add_metal_command_buffer(uocr_profile_state *profile) {
    (void)profile;
}
static inline void uocr_profile_add_metal_command_encoder(uocr_profile_state *profile) {
    (void)profile;
}
static inline void uocr_profile_add_metal_command_buffer_wait(uocr_profile_state *profile) {
    (void)profile;
}
static inline void uocr_profile_add_metal_mps_descriptor(uocr_profile_state *profile) {
    (void)profile;
}
static inline void uocr_profile_add_metal_mps_ndarray(uocr_profile_state *profile) {
    (void)profile;
}
static inline void uocr_profile_add_metal_nsarray(uocr_profile_state *profile) {
    (void)profile;
}
static inline void uocr_profile_add_metal_transient_retain_object(uocr_profile_state *profile) {
    (void)profile;
}

#endif /* UOCR_ENABLE_PROFILING */

#ifdef __cplusplus
}
#endif

#endif /* UOCR_RUNTIME_PROFILE_H */
