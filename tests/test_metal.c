#include "backend/metal/uocr_metal.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

int main(void) {
    if (!uocr_metal_is_available()) {
        fprintf(stderr, "skipping Metal smoke test: Metal is unavailable\n");
        return 0;
    }

    char error[512];
    memset(error, 0, sizeof(error));
    uocr_metal_context *ctx = uocr_metal_context_create(NULL, error, sizeof(error));
    if (ctx == NULL) {
        fprintf(stderr, "skipping Metal smoke test: failed to create context: %s\n", error[0] != '\0' ? error : "unknown error");
        return 0;
    }

    CHECK(uocr_metal_backend_name() != NULL);
    CHECK(uocr_metal_context_model_view_count(ctx) == 0u);
    CHECK(uocr_metal_context_tensor_binding_count(ctx) == 0u);
    CHECK(uocr_metal_context_decoder_binding_count(ctx) == 0u);
    CHECK(!uocr_metal_context_decoder_bindings_ready(ctx));
    CHECK(uocr_metal_context_vision_binding_count(ctx) == 0u);
    CHECK(!uocr_metal_context_vision_bindings_ready(ctx));

    uocr_metal_context_destroy(ctx);
    return 0;
}
