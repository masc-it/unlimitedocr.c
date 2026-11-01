# Unlimited-OCR-2 implementation plan

Status: `todo`

## Target

Ship v2 OCR with bounded-memory long-document parsing, without changing v1 behavior. Reuse existing vision, MoE, projection, and quantization kernels. Single-window inference is an intermediate checkpoint, not the finished feature.

Model selection is implemented: `UnlimitedOCR(model="v2", quant="q4")`; default remains v1. Mixed Q4 is the primary v2 quantization profile. Experimental single-page v2 generation works in base/gundam modes (mixed-Q4 and mixed-Q8 verified). Source builds also support base-mode multi-page windows and document streaming (mixed-Q4 E2E verified); upstream runtime parity and the long-document memory plateau check remain pending. `generate(image)` is unchanged. `stream_pages(images)` accepts an iterable of ordered base-mode pages and yields complete-window text chunks; `generate_pages(images)` joins those chunks for convenience. Only the joining API intentionally retains the full output.

Remaining multi-page work follows the [activity-based execution plan](plan_v2_multi_page.md), with behavioral E2E probes only at activity boundaries.

Detailed tensor inventory, math, source links, and pitfalls: [research notes](unlimitedocr2_research.md).
Pinned HF revision: `c14071b549f4fcab10909f3c4acc6a2edb366947`.

## Scope decisions

- Implement this checkpoint, not a generic model framework. Use a small v1/v2 descriptor and explicit dispatch; avoid unrelated refactoring.
- Use mixed Q4 as the primary profile for remaining multi-page and document-window checkpoints; retain Q8 for comparison. Keep new router weights FP16 and router reductions/entmax FP32. Reuse existing FP16/Q8/Q4 handling for other weights. Do not change the existing constructor defaults.
- Start with retained v2 K/V for one active page window, bounded by its token limit and the existing memory budget. Keep the v1 ring unchanged. Do not allocate for the entire document or the full 131072-token maximum by default.
- Support standard page-marked OCR first. Reject v2 text-only/markerless requests initially instead of implementing unused 1D YaRN fallback.
- Page-window scheduling and KV-tail reuse are required. Follow upstream continuation semantics, not independent OCR calls stitched together. Defer only multi-page tiling, router quantization, and unrelated performance tuning.
- No new C reference backend or test framework. Use upstream inference and small Python math fixtures; add deeper instrumentation only to diagnose a mismatch.

Before implementing the cache:

- [done] Inspect the pinned customized SGLang wheel and record its contract in the research notes: contiguous owned pages + one lookahead; retain last-page marker/image tail, text tail, and lookahead K/V; rebuild router summaries/page state; preserve global page parity; stop before transcribing lookahead. Copy-splice is approximate continuation. Use this contract for the cache/session interface; scheduler runtime parity is still unverified.

## Phase 1: Convert and load v2

Outcome: the existing conversion/load tools accept the actual v2 checkpoint and still load v1 files.

Files: `src/unlimitedocr_c/{api,convert,tensor_registry,frontend}.py`, `src/model/`.

- [done] Add explicit model selection, pinned source/tokenizer resolution, and separate model/revision/quant caches. Keep existing v1 cache lookup working.
- [done] Merge both shard headers and stream tensors from their own shard offsets, including FP16/Q8/Q4 and filtered tensor comparison.
- [done] Validate the v2 config/tokenizer without requiring its absent `processor_config.json`.
- [done] Register DASH tensors with stable IDs. Preserve existing vision IDs; validate/omit deterministic I64 `position_ids` and explicitly classify unused indexer weights.
- [done] Add variant metadata to `.uocr` and native loading. V2 uses file format 2 with a 60-byte config extension; v1 files retain their layout. Reject unknown configs. Public v2 generation now requires Metal, complete bindings, and a validated single-page image request.

QA checkpoint:

- [done] Convert the real checkpoint to mixed-Q8, load it natively, and compare FP16/Q8 tensors from each shard with their source. Verify every source tensor is accounted for.
- [done] Run a disposable two-shard conversion probe (FP16/Q8/Q4, cached headers, missing shard, wrong index assignment) and existing converter/API checks.
- [done] Run native model-file checks after adding v2 metadata.

QA evidence:

- `uv run python -m pytest tests/test_convert.py tests/test_unlimitedocr_api.py -q`: 26 passed, 1 opt-in large-model test skipped.
- Rebuilt native loader/API/Metal/dump; `ctest --test-dir build -R '^uocr_(model_file|constants|dump|api)$' --output-on-failure`: 4 passed.
- Disposable probes first reproduced missing sharding/v2 support, then passed. Actual v2 headers/config/tokenizer validate for FP16/Q8/Q4: 2819 tensors, 2769 runtime, 49 preserved-unused, 1 omitted; all v1 IDs unchanged. FP16 planned payload is 6,687,956,480 bytes.
- Small synthetic v2 tensors were converted, compared against source, and loaded natively, including filtered conversion. Invalid position buffers, unsupported configs, malformed file versions, and quantized router records are rejected. At that conversion-only checkpoint, CPU/Metal engine opening reported v2 inference unavailable. Probes removed after verification.
- Rewriting a previously validated config exposed a path-only JSON cache bug; cache keys now include file identity. Repeatable check: build a plan, change `dash_stage0` to `indexer` at the same path, build again in the same process; the second build must reject it.
- Model selection/frontend checks: `uv run python -m pytest tests/test_unlimitedocr_api.py tests/test_frontend.py tests/test_frontend_boundary.py tests/test_convert.py -q`: 52 passed, 1 skipped. A disposable offline probe verified repo/revision/quant cache isolation, pinned v2 tokenizer resolution, mismatched variants/tokenizers, failed reconversion preserving the cache, and the constructor-to-preparation-to-decoding path with a fake engine. Downloads/conversion were stubbed for this probe; tokenizers were real cached HF files. Probe removed.
- Real pinned payload download and mixed-Q8 conversion completed through `resolve_model_path(model='v2', quant='q8')`. Cached file: `~/Library/Caches/unlimitedocr/v2/7e04dada5a6c1229/mixed-q8_0/unlimitedocr-q8.uocr`, 3,468,861,440 bytes. Conversion reported zero source NaN/Inf values. `build/uocr-dump` loaded the complete file and confirmed format 2, 2819 source tensors, 2769 runtime, 49 preserved-unused, and 1 omitted.
- Source-byte checks matched layer-0 attention Q and router Q-norm from shard 1, plus layer-10 expert-53 down-projection and input norm from shard 2. Both Q8 weights/scales and FP16 payloads were checked. Real-file verification exposed a comparison-tool bug: filtered plans relocate scale slabs, so absolute scale offsets cannot be compared with the full target file. Comparison now follows target offsets and verifies their bytes, retaining scale-presence/size checks. Wrong scale pointers still fail payload comparison. Disposable regression probe removed; converter/API checks again passed (26 passed, 1 opt-in test skipped).
- Phase-1 conversion/loading is verified for the real mixed-Q8 model. Full FP16 conversion remains unverified; mixed-Q4 conversion and Q4/Q8 OCR evidence follows in phase 2.

## Phase 2: Generate a single page

Outcome: `UnlimitedOCR(model="v2", quant="q4").generate(image)` works in base and gundam modes.

Files: `src/unlimitedocr_c/{frontend,api,ocr}.py`, `src/runtime/`, `src/core/uocr_api.c`, `src/backend/metal/uocr_metal.m`, Metal RoPE/attention kernels.

- [done] Load v2's 834 added-token entries; build `<PAGE><image>Multi page parsing.` and cap gundam crops at 16. Preserve page markers in output. Reject markerless and text-only v2 inputs.
- [done] Implement standalone request-owned page/token state and Metal page2d RoPE (63 resampled token pairs plus one page pair), supporting both 10-head main attention and 1-head routing. Prefix/instruction positions remain page 0; routing page IDs remain window-local.
- [done] Wire page state and persistent GPU positions into the mapped-model v2 decoder forward path. Coordinates advance once per input token and are shared across all layers for chunked prefill and decode, including the public single-page generation loop.
- [done] Implement standalone Metal DASH math: 128-dimensional Gemma RMSNorm (`1 + weight` in FP32), learned `head_cls` summaries over 32-token blocks, page-aware candidates/fallback, top-16 entmax-1.5, up to 12 selected blocks, centered log-prior bias, and local 128-token attention. Local entries win on overlap. GPU scratch is linear in window/query capacity; attention never constructs a dense score matrix.
- [done] Add a request-owned 12-layer retained attention cache with chunked prefill/decode, one page-state advance per input token, cached block summaries, and tail-only refresh. A host bridge accepts synthetic projected Q/K/V and exercises Gemma norm, page RoPE, routing, and attention together without a model.
- [done] Map and validate the 60 DASH weights, connect actual main/router projections to retained attention, and reuse the existing 12-layer decoder loop including residuals, dense FFN, and MoE. The diagnostic decoder-forward boundary accepts pre-embedded rows; public generation consumes GPU-resident visual/token embeddings.
- [done] Reuse existing projection/norm/FFN/MoE and FP16/Q8/Q4 final-head selection operations. Add a diagnostic raw argmax for the latest v2 hidden state; it intentionally does not implement page/EOS guards.
- [done] Allocate retained main/router K/V, summaries, positions, and reusable attention scratch from explicit window/query/page capacities. Enforce the attention-cache budget before allocation; account GPU payload and CPU state. Reject over-capacity appends, incomplete/out-of-order layers, and invalid page transitions. Reset/release supports cancellation; GPU errors poison the request until reset.
- [done] Add decoder-only admission for retained attention, hidden/FFN/MoE workspaces, token staging, and rounded selection/logit/transient scratch. Skip the v1 ring arena for v2. Keep v1's existing ring and generation path unchanged.
- [done] Combine decoder admission with model/vision costs and safety overhead in the full engine request budget. Public single-page generation encodes vision first, releases chunk scratch, then allocates the retained decoder window and assembles visual/token embeddings on GPU. Success/failure releases request-owned state; mapped weights/pipelines remain reusable.
- [done] Implement upstream page-count guard state and fused pre-argmax token filtering for FP16/Q8/Q4 heads, including the 20000-token per-page cap. V1 passes an unrestricted filter.
- [done] Connect page-count guarding to public v2 generation. Optional repetition processing remains off, as upstream does. Enforce one page-marked image, one global view, and at most 16 local crops; reject text-only/multi-page requests until phase 3.

QA checkpoint:

- [done] Check page-state coordinates against pinned SGLang pure helpers and Metal page RoPE against an FP16 numerical reference, including offsets beyond 32768 and both head counts.
- [done] Check standalone router summaries/selections/biases and attention output across partial blocks, the 128-token boundary, and empty-candidate fallback, using a NumPy transcription of the pinned HF inference math. Full upstream/model execution remains a separate check.
- [todo] Compare real base and gundam OCR with pinned upstream `infer`, including a page producing more than 128 tokens. Check prompt IDs, final text, and page markers; inspect intermediate logits only if results diverge. Account for BF16/FP16 rounding rather than requiring bitwise identity.
- [done] Confirm a too-small request memory budget rejects before workspace allocation, and real v1 image output matches a baseline build exactly.
- [done] Read real rendered text in v2 base/gundam modes, preserve markers, reach EOS, and generate a page with more than 128 output tokens. This does not replace the pending upstream comparison.
- [done] Convert the real checkpoint to mixed Q4, verify source-byte conversions across both shards and the FP16 DASH keep-list, compare Q4/Q8 OCR on the same fixtures, and verify budget rejection, truncated-request cleanup and engine reuse.

QA evidence:

- Frontend probes passed for single/two-page markers, 834 added-token entries, a 16-crop gundam image, and page-marker decoding.
- Disposable page-state probe matched `replay_coords` and `replay_page_isolation` extracted from the pinned SGLang source. Covered 273/701-token image spans, more than 128 decode tokens, extra page markers, page bases 0/1/(2^40+1), and invalid transitions/counter overflow without state mutation.
- On local Apple Metal, freshly compiled FAST and SAFE libraries produced bit-identical FP16 RoPE outputs to the NumPy FP16 reference for both head counts, including offsets 32768 and 131071. Shader-side frequency evaluation initially caused 0.00659 max absolute error at a long offset; a host-computed FP32 frequency table reduced the tested error to zero. This is FP16 primitive parity, not upstream BF16 model parity. Temporary probe and libraries removed.
- `cmake --build build --target unlimitedocr test_sequence test_metal test_api test_request_validation -j 4` succeeded. `ctest --test-dir build -R '^uocr_(api|sequence|request_validation|metal)$' --output-on-failure`: 4 passed.
- Disposable DASH probes passed on freshly compiled FAST and SAFE Metal libraries: 1/769-token prefill, 7 queries against 1283 retained keys, single-token decode, mixed-page blocks, empty image candidates, tied/uniform scores, masked scores outranking very negative candidates, and nonfinite router activations. Selected IDs matched the stable NumPy reference; summary tolerance 5e-6, bias tolerance 5e-4, attention tolerance 2.5e-4 plus 0.2% relative. Gemma norm maximum absolute error was 0.0009765625. Probes transcribed pinned HF math; Torch/upstream model execution was not used. A failing tie-order probe exposed and fixed top-M insertion displacing earlier equal-score blocks. Temporary probes removed.
- Rebuilt `unlimitedocr`, `test_metal`, and `test_api`; `ctest --test-dir build -R '^uocr_(api|metal)$' --output-on-failure`: 2 passed. V1 attention/norm kernels remain unchanged.
- Disposable retained-cache probe passed on actual Metal: two prefill chunks (65/64 tokens), then 131 one-token decode appends through all 12 layers. An extra answer page forced remote-text fallback, exercising generated K/V older than 128 tokens. Layer 0/11 outputs at selected boundaries matched a freshly recomputed full-retained-key primitive. Synthetic projections were used; no model weights or FFN/MoE were executed.
- For capacities 260 tokens / 129 query rows / 3 page buckets, tracked cache/workspace memory stayed at 17,551,664 bytes and returned to zero on release. This is a within-window allocation check, not the long-document memory proof. Exact-budget allocation, insufficient budget, capacity exhaustion, invalid-token rollback, layer ordering, cancellation/reset, and destruction passed. An intentionally incomplete Metal library verified that an encode failure leaves output untouched and poisons the request until reset. Temporary probes removed.
- Rebuilt `unlimitedocr`, `test_api`, `test_metal`, `test_memory`, and `test_sequence`; `ctest --test-dir build -R '^uocr_(api|metal|memory|sequence)$' --output-on-failure`: 4 passed.
- Real mapped mixed-Q8 decoder probe passed: 2294 decoder bindings, a 769-token synthetic-embedding prompt, and two decode steps through all 12 layers. Whole prefill and 385/384-token chunks produced bit-identical hidden states and the same raw next token (11297). This exercised more than 16 remote candidates. The learned weights were real, but inputs were synthetic embeddings, not vision output or a pinned upstream numerical reference.
- Decoder admission matched tracked live allocations at 95,548,012 bytes for 771 retained tokens / 769 query rows / 3 page buckets. One-byte-short admission rejected before allocation; missing DASH bindings rejected mapping; reset invalidated stale selection; destruction returned tracked allocations to zero. No v1 ring buffer was allocated. Public generation was still guarded at that decoder-only checkpoint.
- The new check exposed a pre-existing generic Metal pipeline issue: optional function constants still require specialization before pipeline creation. A tiny failing shader probe reproduced the validation abort; specialization with empty values restored runtime fallbacks and preserved pipeline caching. Actual mapped final RMSNorm also matched a scalar reference. Fix committed separately as `b1bd388`.
- An actual v1 mixed-Q8 text-decoder run produced identical eight-token IDs and scores before/after the decoder changes, using a baseline built from `b1bd388`. V1 image golden QA remains todo. Rebuilt both `build/` and `build/release/`: CTest 13 passed; `uv run python -m pytest tests -q` with large/perf opt-ins disabled: 105 passed, 7 skipped, 3 expected invalid-limit warmup warnings. An initial unscoped pytest run collected the temporary baseline's duplicate test files; removing that archive and targeting `tests/` resolved collection. All disposable probes removed.
- Disposable page-guard probe matched the extracted pinned HF helper using a NumPy score-array adapter: premature EOS, missing/extra markers, page-cap boundaries including 20000, and disabled guard/cap. Actual Metal FP16/Q8/Q4 argmax probes verified unrestricted selection, exclusion of a dominant EOS logit, and forced tokens before reduction. Probes first failed, then passed; removed after verification.
- Public mixed-Q8 OCR passed on a rendered invoice: base (279 prompt tokens, 1 view, 68 output tokens) and gundam (1519 prompt tokens, 12 local crops plus global, 71 output tokens) both preserved `<PAGE>`, reached EOS, and read invoice 12345 / USD 42.00. `UnlimitedOCR(model='v2', quant='q8', download=False).generate(image)` also passed. V2 skips the legacy markerless synthetic warmup and compiles pipelines on the first real request.
- A ten-item invoice produced 271 tokens with all ten rows correct and EOS. Measured tracked live/peak/estimated bytes were 3,468,464,128 / 4,956,287,488 / 5,204,104,012. After each request, live bytes returned to model-only usage. Markerless input rejected; a budget between the former underestimate and actual peak rejected before workspace allocation.
- That memory probe exposed a shared estimator drift: ten SAM global-attention/relative-logit/packing and convolution im2col/temp slices were missing. `2b0bd5b` adds them for both variants. The predicted single-global workspace now matches Metal exactly at 1,487,122,432 bytes (previously 144,945,152). The failing CPU/real-image probes passed after the fix.
- Real v1 mixed-Q8 image OCR matched token IDs and decoded text exactly against a baseline built from `2b0bd5b`. A library with its DASH norm kernel deliberately removed failed after vision encoding; two attempts both released all request allocations and retained only model usage. Final checks: Python 105 passed / 7 skipped (large/perf opt-ins disabled, 3 expected invalid-limit warmup warnings); CTest 13 passed. Temporary images, probes, and baseline build removed after verification.
- Mixed-Q4 qualification used the existing conversion/runtime paths without kernel or runtime changes. `HF_HUB_OFFLINE=1 uv run python -c 'from unlimitedocr_c import resolve_model_path; print(resolve_model_path(model="v2", quant="q4"))'` converted the pinned cached checkpoint to `~/Library/Caches/unlimitedocr/v2/7e04dada5a6c1229/mixed-q4/unlimitedocr-q4.uocr`. File size: 1,846,509,568 bytes; tensor-data/model-view size: 1,846,112,256 bytes. Source NaN/Inf count was zero. Native `uocr-dump` confirmed format 2 and 2295 Q4 / 48 Q8 / 476 FP16 records (including the omitted position-ID record).
- Source conversion bytes matched all 60 FP16 DASH tensors, sampled Q8 attention projections, and Q4 embeddings, LM head, dense MLP, shared/routed experts, projector, CLIP and SAM matrices. Samples span both source shards. No precision policy changed: decoder attention stays Q8 and DASH stays FP16.
- Q4/Q8 comparison used two Arial-36 rendered fixtures: a 640x896 invoice (12345 / USD 42.00) and a 640x1024 ten-item invoice (Item 01: USD 11.00 through Item 10: USD 20.00). Both quantizations read all expected content in base/gundam modes, preserved one `<PAGE>` and reached EOS. Q4 produced 68 tokens for the short fixture and 271 for the long fixture in both modes. Gundam used 13 and 16 views respectively. Text blocks matched Q8 exactly in all four comparisons, with all ten rows ordered. Coordinates and some heading labels differed; token equality is not required. The public `UnlimitedOCR(model="v2", quant="q4", download=False)` path also passed.
- Q4 tracked live bytes returned to model-only 1,846,112,256 after each request. Base peak/estimate: 3,333,935,616 / 3,500,634,547 bytes. Long gundam peak/estimate: 4,140,166,656 / 4,347,177,139 bytes. A model-size-plus-512-MiB budget rejected before request allocation. Markerless input rejected; a one-token truncated request released state, and the same engine then completed the invoice. Final checks: `uv run python -m pytest tests -q` with large/perf opt-ins disabled: 105 passed, 7 skipped, 3 expected invalid-limit warmup warnings; `ctest --test-dir build --output-on-failure`: 13 passed. Disposable probes/images removed; the converted model remains cached.
- Native single-page OCR is exercised end-to-end. Upstream BF16/model parity, broader OCR quality evaluation and v2 FP16 image inference remain unverified. HF top-M routing and the wheel's optimized full-candidate entmax path differ; see research notes before interpreting scheduler output parity.

## Phase 3: Multi-page and quantization

Outcome: ordered base-mode pages work through the public API; existing Q8/Q4 paths work with v2.

Files: frontend/API, `src/runtime/uocr_{request_validation,sequence,vision}.*`, `src/core/uocr_api.c`, Metal prompt assembly/embedding paths; FFI/header only if necessary; `README.md`.

- [done] Replace the v2 single-image-span assumption with mask-driven visual-row copies on GPU. Keep `<PAGE>` embeddings between spans, size page counters from the request, and include that capacity in memory estimates. V1 keeps contiguous-image validation; the public C ABI and attention kernels are unchanged.
- [done] Add `generate_pages(images)` using one 1024 global view per page and the phase-2 attention implementation. It joins the document stream rather than preparing the entire document at once.
- [done] Enable existing Q8/Q4 policies for unchanged tensor families; keep DASH weights FP16. Single-page Q4 qualification and primary-profile documentation completed before multi-page work.

QA checkpoint:

- [todo] One two-page fixture verifies prompt placement, page coordinates/transitions, output ordering, and EOS guarding against upstream. Reuse this document for FP16/Q8/Q4 smoke checks; compare OCR quality, not exact quantized token equality.
- [done] Confirm switching v1/v2 cannot reuse the wrong tokenizer/model cache using the offline probe. No new broad benchmark matrix or tests of unchanged vision/MoE math.

QA evidence: the first activity in [the execution plan](plan_v2_multi_page.md) passed real Q4 two-/three-page OCR, page reordering with a blank page, invalid-request rejection and engine reuse. Single-page v1 base and v2 base/gundam outputs matched the frozen baseline. The complete E2E probe passed Metal API validation; GPU shader validation stopped in an unchanged v1 baseline kernel after the v2 checks passed. Upstream runtime comparison and multi-page Q8/FP16 checks remain pending. The high-level joining API now uses the document session described in phase 4.

## Phase 4: Long documents with bounded working memory

Outcome: `stream_pages` processes documents spanning many windows without retaining all images, K/V, router history, or generated text. The feature is complete only after this checkpoint.

Files: Python API/frontend/FFI; native session lifecycle, sequence state, memory planner, vision scheduling, and Metal cache handling.

- [done] Add a native document session and hold both Python engine/instance locks until the stream closes. Reentrant calls are rejected; completion, errors and early close release session state.
- [done] Consume pages lazily with one lookahead and expose `window_pages`. Reuse the lookahead K/V and its final visual embedding, including a cached-only final window. Add conservative admission for the retained state and rewritten prompt; the final accounting/budget audit remains in the next activity.
- [done] Implement the inspected copy-splice contract: retain the previous marker, aligned image tail, text tail and lookahead; copy main K/V and router K, rebuild summaries/page state, preserve global page parity, and force the next page marker without duplicate output. Incomplete or noncontiguous windows fail explicitly. Upstream runtime equivalence is not established.
- [done] Yield complete-window text chunks and release completed input/output buffers. Only the joining API intentionally retains all output. The retained record is fixed-size; long-document native/RSS measurements remain pending.
- [done] Document window/page/document token limits, approximate continuation, same-thread stream ownership, early closure and failure behavior in the README. Only the active window is subject to the model context cap.

QA checkpoint:

- [done] Run real Q4 documents across three windows, including text continuing onto the next page, reordered window sizes, blank pages and cached-only final windows. Verify page order, no duplicated output and streamed/joined equality.
- [todo] Compare continuation with the pinned SGLang scheduler using the same settings. Blocked locally: its CUDA runtime is unavailable. Full-cache HF output is not the scheduler oracle.
- [todo] Repeat the page input to increase document length by 10x, consume/discard streamed chunks, and measure live/peak native allocations and process memory. With fixed window settings, active memory must plateau rather than grow with completed pages. No broad benchmark suite.
- [done] Close a stream mid-document, then reuse the engine. The E2E probe also covered a waiting caller, input failures, skipped windows, exact/short output budgets and ordinary single-window baseline parity. Per the activity-based plan, no micro-tests or unit-test runs were added.

QA evidence: see the document-streaming activity in [the execution plan](plan_v2_multi_page.md). Native E2E passed with Metal API and GPU shader validation. One failed probe exposed a GPU-private vision read; copying through the shared prompt buffer fixed it, and the full probe passed afterward. Upstream runtime comparison is blocked, and the 10x document-length/native/RSS memory check remains pending. Do not mark bounded-memory support complete until that check passes.
