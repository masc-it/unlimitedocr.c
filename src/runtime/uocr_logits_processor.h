#ifndef UOCR_LOGITS_PROCESSOR_H
#define UOCR_LOGITS_PROCESSOR_H

#include <stddef.h>
#include <stdint.h>

#include "unlimitedocr.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Collect token ids that would repeat an n-gram if emitted next.
 *
 * Semantics intentionally mirror data/context/modeling_unlimitedocr.py:
 * SlidingWindowNoRepeatNgramProcessor.  The current prefix is the last
 * ngram_size-1 tokens of `sequence`; previous n-grams are searched from
 * max(0, len-window) up to len-ngram_size inclusive.  window==0 means the
 * entire sequence is searched, matching HF's non-windowed no-repeat mode.
 *
 * The function performs no allocation.  If out_capacity is too small it returns
 * UOCR_ERROR_OUT_OF_MEMORY and still writes the required unique banned-token
 * count to out_count.
 */
int uocr_no_repeat_ngram_collect_banned(const int32_t *sequence,
                                        uint32_t sequence_len,
                                        uint32_t vocab_size,
                                        uint32_t ngram_size,
                                        uint32_t window,
                                        const int32_t *whitelist_token_ids,
                                        uint32_t whitelist_count,
                                        int32_t *out_banned,
                                        uint32_t out_capacity,
                                        uint32_t *out_count);

/* Apply the same processor in-place by writing -inf to banned logits. */
int uocr_no_repeat_ngram_apply(float *logits,
                               uint32_t vocab_size,
                               const int32_t *sequence,
                               uint32_t sequence_len,
                               uint32_t ngram_size,
                               uint32_t window,
                               const int32_t *whitelist_token_ids,
                               uint32_t whitelist_count);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_LOGITS_PROCESSOR_H */
