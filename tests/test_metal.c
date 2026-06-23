#include "backend/metal/uocr_metal.h"
#include "unlimitedocr.h"

#include <stdio.h>
#include <string.h>

#ifndef UOCR_TEST_METAL_RESOURCE_PATH
#define UOCR_TEST_METAL_RESOURCE_PATH NULL
#endif

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static int test_metal_smoke(void) {
    if (!uocr_metal_is_available()) {
        printf("Metal device not available; skipping Metal smoke test\n");
        return 0;
    }

    char error[1024];
    memset(error, 0, sizeof(error));
    CHECK(uocr_metal_smoke_test(UOCR_TEST_METAL_RESOURCE_PATH, error, sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    return 0;
}

static int test_public_engine_open_initializes_metal(void) {
    if (!uocr_metal_is_available()) {
        return 0;
    }

    uocr_engine_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend = "metal";
    opts.resource_path = UOCR_TEST_METAL_RESOURCE_PATH;
    opts.max_batch = 1;
    opts.max_prompt_tokens = 16;
    opts.max_gen_tokens = 4;

    uocr_engine *engine = uocr_engine_open(&opts);
    CHECK(engine != NULL);
    CHECK(strcmp(uocr_engine_backend(engine), "metal") == 0);
    CHECK(strcmp(uocr_last_error(engine), "OK") == 0);
    uocr_engine_close(engine);
    return 0;
}

int main(void) {
    CHECK(strcmp(uocr_metal_backend_name(), "metal") == 0);
    if (test_metal_smoke() != 0) return 1;
    if (test_public_engine_open_initializes_metal() != 0) return 1;
    return 0;
}
