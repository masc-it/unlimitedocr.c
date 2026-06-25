#ifndef UOCR_RUNTIME_PROFILE_H
#define UOCR_RUNTIME_PROFILE_H

#include <stdint.h>

#include "unlimitedocr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uocr_profile_state {
    uocr_profile_report report;
} uocr_profile_state;

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

#ifdef __cplusplus
}
#endif

#endif /* UOCR_RUNTIME_PROFILE_H */
