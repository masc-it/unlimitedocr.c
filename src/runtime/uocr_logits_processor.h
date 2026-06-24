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

typedef struct uocr_no_repeat_ngram_config {
    const int32_t *sequence;
    uint32_t sequence_len;
    uint32_t ngram_size;
    uint32_t window;
} uocr_no_repeat_ngram_config;

/* Apply the same processor in-place by writing -inf to banned logits. */
int uocr_no_repeat_ngram_apply(float *logits,
                               uint32_t vocab_size,
                               const int32_t *sequence,
                               uint32_t sequence_len,
                               uint32_t ngram_size,
                               uint32_t window,
                               const int32_t *whitelist_token_ids,
                               uint32_t whitelist_count);

/* Apply per-row no-repeat configs to a row-major [n_rows, vocab_size] logits
 * buffer. configs_or_null==NULL is a no-op, and per-row ngram_size==0 disables
 * the processor for that row. The helper performs no allocation and is the CPU
 * readback path used before greedy argmax until a GPU logits-processor kernel
 * is added.
 */
int uocr_no_repeat_ngram_apply_batch(float *logits,
                                     uint32_t n_rows,
                                     uint32_t vocab_size,
                                     const uocr_no_repeat_ngram_config *configs_or_null);

#ifdef __cplusplus
}
#endif

#endif /* UOCR_LOGITS_PROCESSOR_H */
