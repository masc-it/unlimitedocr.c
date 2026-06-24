#ifndef UOCR_RESULT_INTERNAL_H
#define UOCR_RESULT_INTERNAL_H

#include <stdint.h>

#include "unlimitedocr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate an empty result with one zero-length generated-token list per
 * sequence. Used by ABI smoke paths before inference kernels produce tokens.
 */
int uocr_result_create_empty(uint32_t n_sequences, uocr_result **out_result);

/* Allocate a result by copying caller-owned generated token buffers.  The
 * generated_tokens array and each non-empty per-sequence pointer are consumed
 * only for the duration of the call; the returned uocr_result owns its copies.
 */
int uocr_result_create_from_generated(uint32_t n_sequences,
                                      const int32_t *const *generated_tokens,
                                      const uint32_t *generated_counts,
                                      uocr_result **out_result);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_RESULT_INTERNAL_H */
