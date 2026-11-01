# V2 multi-page execution plan

## Scope and working rules

Process page images supplied in document order as one document, using base mode
and primarily mixed Q4. Preserve v1, single-page generation, model caches and
existing quantization policy. Multi-page gundam and PDF rendering are out of scope.

Follow the continuation behavior of the upstream version we inspected, described
in [the research notes](unlimitedocr2_research.md#verified-upstream-scheduler-contract).

Run each activity's behavioral E2E probe only after its implementation is complete.
Mark an activity complete only after its probe passes. Do not run separate tests
for individual helpers or kernels, or add permanent tests. Keep temporary probes
under `data.tmp/probes/v2-multi-page/`, record actual results and limitations here, then
remove the probes created for this work. Preserve the existing synthetic fixtures
and unrelated notebook edits.

## Activities

[x] **multi-page-window**: Process several pages in one window.
- Allow the native engine to accept several pages and map each page's visual
  embeddings to its corresponding image tokens in the prompt. Preserve the
  `<PAGE>` tokens between groups of image tokens. Track positions separately for
  each page and prevent generation from ending before all pages are processed.
- Support as many pages as the configured limits allow. Return a clear error
  for unsupported layouts or requests that exceed those limits. Start with a
  two-page fixture.
- **End-of-activity E2E probe:** prepare a real two-page document in Python,
  generate its OCR output with the native Q4 engine, then decode and parse it.
  Verify both pages' content, order and markers, and check that generation
  reaches EOS rather than the token limit. Check that a request exceeding the
  limits fails, then that the same engine can process a valid request.

- **Evidence (2026-11-01):** Release build and the end-of-activity E2E probe
  passed on M1 Pro / macOS 26.0.1 with the cached v2 Q4 model and Metal API
  validation. Two 1024px warehouse-report pages produced 303 tokens from a
  553-token prompt, with both pages' seven expected text lines and natural EOS.
  Reordering the pages around a blank page produced three pages and 304 tokens
  from an 827-token prompt. The two-page output was identical across three runs.
- Requests exceeding prompt/output limits, missing a page marker, or assigning
  272/274 image tokens instead of 273 per page were rejected. The same engine
  remained usable after those errors and an eight-token truncated request.
  V2 live allocations returned to 1,846,112,256 bytes after requests; tracked
  two-/three-page peaks were 2,445,670,080 / 2,717,135,040 bytes. These are not RSS
  measurements or a long-document memory guarantee.
- V1 Q4 base and v2 Q4 base/gundam token IDs matched the frozen `93ed7a3` library
  and metallib. GPU shader validation passed the multi-page/recovery cases but
  then aborted in the unchanged v1 baseline's Q4 prefill kernel: 33,280 bytes of
  threadgroup memory exceeded 32,768. The full probe passed with shader
  instrumentation off and API validation on. No upstream runtime comparison,
  multi-page Q8/FP16 checks or unit-test runs were performed. Probe files were
  removed after recording these results.

[ ] **document-streaming**: Stream a document across window boundaries.
- Add a native session that holds state for one document. Provide the Python
  APIs `stream_pages(images)` and `generate_pages(images)`; the latter collects
  the chunks into one string.
- Read pages from the supplied iterable only when needed. Keep the pages being
  transcribed, plus the next page for context. When moving to the next window,
  keep the previous page's marker, its last 128-159 image tokens (aligned to
  32-token cache blocks), up to 128 trailing generated text tokens, and the
  cached next page. Follow the linked retention rules for token eligibility.
- Carry the cached attention data (main K/V and router K) into the next window.
  Rebuild routing summaries and page information to match the new layout. Keep
  odd/even page positions consistent across the document, without applying
  positional rotation again to copied keys.
- Include each page once in the output. Do not repeat text carried forward as
  context. Ending a window must not end the document while pages remain. Stop
  with an error if the preceding window is incomplete or the next window skips
  pages. Hold the engine lock for the session; release it and the session's
  buffers when processing finishes, the caller closes the stream, or an error
  occurs.
- **End-of-activity E2E probe:** stream a real Q4 document through at least three
  small windows, with content continuing across a boundary. Verify two
  consecutive transitions and that input is read no more than one page beyond
  the current window. Check that every page appears once, in order, and that
  collecting the streamed chunks produces the same text as `generate_pages()`.
  Compare how windows end and generation resumes with the pinned upstream
  SGLang version. If upstream cannot be run, record that comparison as blocked;
  do not claim the behavior has been verified against it.
- Upstream reuses cached states after rearranging their context, which is an
  approximation. Use its windowed implementation as the reference; Hugging Face inference retaining the full
  history can produce different results.

[ ] **memory-and-recovery**: Verify memory use, limits and recovery.
- Before starting a window, check that memory can hold its buffers, the retained
  context, image processing, and both copies of data being moved. Let callers
  configure pages per window, output tokens per page and total document output
  tokens. Document how those limits work. Return a clear error if there is
  insufficient memory for even the smallest supported window.
- Discard token history and Python objects from completed windows, except for
  the context needed by the next window. Release images and output buffers
  when no longer needed. Document that the memory bound excludes the complete
  output string and images the caller has already loaded.
- Ensure the engine remains usable after cancellation, an error while reading
  pages, or reaching a resource limit.
- **End-of-activity E2E probe:** repeat the document to increase its length 10x
  with fixed window settings, consuming and discarding streamed chunks. Measure
  current and peak native memory use and process memory. Check that memory
  stops growing after warmup, even as more pages are processed. Close a stream
  early, make the input iterator fail, and request more memory than the budget
  allows; after each case, check that the engine can process another document.
  Include checks that v1 and single-page generation still work, plus a v2 Q8
  multi-page run.
