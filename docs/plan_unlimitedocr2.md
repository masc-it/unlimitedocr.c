# Unlimited-OCR-2 implementation plan

Status: `todo`

## Target

Ship v2 OCR with bounded-memory long-document parsing, without changing v1 behavior. Reuse existing vision, MoE, projection, and quantization kernels. Single-window inference is an intermediate checkpoint, not the finished feature.

Model selection is implemented: `UnlimitedOCR(model="v2", quant="q8")`; default remains v1. Native v2 inference is still blocked pending phase 2. Keep `generate(image)`. Add `stream_pages(images)` accepting an iterable of ordered base-mode pages and yielding text chunks; `generate_pages(images)` joins those chunks for convenience. Only the joining API intentionally retains the full output.

Detailed tensor inventory, math, source links, and pitfalls: [research notes](unlimitedocr2_research.md).
Pinned HF revision: `c14071b549f4fcab10909f3c4acc6a2edb366947`.

## Scope decisions

- Implement this checkpoint, not a generic model framework. Use a small v1/v2 descriptor and explicit dispatch; avoid unrelated refactoring.
- Keep new router weights FP16 and router reductions/entmax FP32. Reuse existing FP16/Q8/Q4 handling for other weights.
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
- [done] Add variant metadata to `.uocr` and native loading. V2 uses file format 2 with a 60-byte config extension; v1 files retain their layout. Reject unknown configs and block v2 engine/Metal inference until sparse attention is implemented.

QA checkpoint:

- [todo] Convert the real checkpoint, load it, and compare one tensor from each shard with its source. Verify every source tensor is accounted for.
- [done] Run a disposable two-shard conversion probe (FP16/Q8/Q4, cached headers, missing shard, wrong index assignment) and existing converter/API checks.
- [done] Run native model-file checks after adding v2 metadata.

QA evidence:

- `uv run python -m pytest tests/test_convert.py tests/test_unlimitedocr_api.py -q`: 26 passed, 1 opt-in large-model test skipped.
- Rebuilt native loader/API/Metal/dump; `ctest --test-dir build -R '^uocr_(model_file|constants|dump|api)$' --output-on-failure`: 4 passed.
- Disposable probes first reproduced missing sharding/v2 support, then passed. Actual v2 headers/config/tokenizer validate for FP16/Q8/Q4: 2819 tensors, 2769 runtime, 49 preserved-unused, 1 omitted; all v1 IDs unchanged. FP16 planned payload is 6,687,956,480 bytes.
- Small synthetic v2 tensors were converted, compared against source, and loaded natively, including filtered conversion. Invalid position buffers, unsupported configs, malformed file versions, and quantized router records are rejected. CPU/Metal engine opening reports v2 inference unavailable. Probes removed after verification.
- Rewriting a previously validated config exposed a path-only JSON cache bug; cache keys now include file identity. Repeatable check: build a plan, change `dash_stage0` to `indexer` at the same path, build again in the same process; the second build must reject it.
- Model selection/frontend checks: `uv run python -m pytest tests/test_unlimitedocr_api.py tests/test_frontend.py tests/test_frontend_boundary.py tests/test_convert.py -q`: 52 passed, 1 skipped. A disposable offline probe verified repo/revision/quant cache isolation, pinned v2 tokenizer resolution, mismatched variants/tokenizers, failed reconversion preserving the cache, and the constructor-to-preparation-to-decoding path with a fake engine. Downloads/conversion were stubbed for this probe; tokenizers were real cached HF files. Probe removed.
- Phase-1 code is implemented. Full checkpoint payload conversion/loading and numerical OCR remain unverified; no full v2 weight download or inference has run.

## Phase 2: Generate a single page

Outcome: `UnlimitedOCR(model="v2").generate(image)` works in base and gundam modes.

Files: `src/unlimitedocr_c/{frontend,api,ocr}.py`, `src/runtime/`, `src/core/uocr_api.c`, `src/backend/metal/uocr_metal.m`, Metal RoPE/attention kernels.

- [done] Load v2's 834 added-token entries; build `<PAGE><image>Multi page parsing.` and cap gundam crops at 16. Preserve page markers in output. Reject markerless and text-only v2 inputs.
- [done] Implement standalone request-owned page/token state and Metal page2d RoPE (63 resampled token pairs plus one page pair), supporting both 10-head main attention and 1-head routing. Prefix/instruction positions remain page 0; routing page IDs remain window-local.
- [todo] Wire the page state and persistent GPU position buffer into v2 prefill/decode. Advance state once per input token and share coordinates across layers; the standalone operation is not yet connected to generation.
- [todo] Implement DASH in both prefill and decode: 128-dimensional Q/K, Gemma RMSNorm (`1 + weight`), learned `head_cls` summaries over 32-token blocks, page-aware candidates/fallback, top-16 entmax-1.5, up to 12 selected blocks, centered log-prior bias, and local 128-token attention. Local entries win on overlap.
- [todo] Reuse matrix/norm primitives where mathematically equivalent. Add only the missing router and sparse-gather attention operations; avoid a dense attention implementation that must later be replaced.
- [todo] Allocate v2 K/V and router state from actual request capacity; enforce the memory budget. State stays per engine/request, never process-global.
- [todo] Apply upstream page-count/EOS guard before argmax, with the 20000-token per-page cap. Leave optional repetition processing off by default, as upstream does.

QA checkpoint:

- [done] Check page-state coordinates against pinned SGLang pure helpers and Metal page RoPE against an FP16 numerical reference, including offsets beyond 32768 and both head counts.
- [todo] Extend the compact math probe to router selections/biases and attention output across a partial block, the 128-token boundary, and empty-candidate fallback. Use pinned upstream helpers on small tensors; no full-model dependency.
- [todo] Compare real base and gundam OCR with pinned upstream `infer`, including a page producing more than 128 tokens. Check prompt IDs, final text, and page markers; inspect intermediate logits only if results diverge. Account for BF16/FP16 rounding rather than requiring bitwise identity.
- [todo] Confirm a too-small memory budget fails cleanly and one existing v1 golden page still works.

QA evidence:

- Frontend probes passed for single/two-page markers, 834 added-token entries, a 16-crop gundam image, and page-marker decoding.
- Disposable page-state probe matched `replay_coords` and `replay_page_isolation` extracted from the pinned SGLang source. Covered 273/701-token image spans, more than 128 decode tokens, extra page markers, page bases 0/1/(2^40+1), and invalid transitions/counter overflow without state mutation.
- On local Apple Metal, freshly compiled FAST and SAFE libraries produced bit-identical FP16 RoPE outputs to the NumPy FP16 reference for both head counts, including offsets 32768 and 131071. Shader-side frequency evaluation initially caused 0.00659 max absolute error at a long offset; a host-computed FP32 frequency table reduced the tested error to zero. This is FP16 primitive parity, not upstream BF16 model parity. Temporary probe and libraries removed.
- `cmake --build build --target unlimitedocr test_sequence test_metal test_api test_request_validation -j 4` succeeded. `ctest --test-dir build -R '^uocr_(api|sequence|request_validation|metal)$' --output-on-failure`: 4 passed.
- Native/reference model inference has not run; it still requires DASH attention, decoder/cache integration, weights, and a suitable reference environment. The v2 inference guard remains enabled.

## Phase 3: Multi-page and quantization

Outcome: ordered base-mode pages work through the public API; existing Q8/Q4 paths work with v2.

Files: frontend/API, `src/runtime/uocr_{request_validation,sequence,vision}.*`, `src/core/uocr_api.c`, Metal prompt assembly/embedding paths; FFI/header only if necessary; `README.md`.

- [todo] Replace the single-image-span assumption with a mask-driven visual-row mapping. Keep `<PAGE>` embeddings between image spans; reuse the existing request image mask and avoid a public ABI change unless needed.
- [todo] Add `generate_pages(images)` using one 1024 global view per page, correct page transitions, and the same attention implementation from phase 2.
- [todo] Enable existing Q8/Q4 policies for unchanged tensor families; keep DASH weights FP16. Update model-selection and supported-limit documentation.

QA checkpoint:

- [todo] One two-page fixture verifies prompt placement, page coordinates/transitions, output ordering, and EOS guarding against upstream. Reuse this document for FP16/Q8/Q4 smoke checks; compare OCR quality, not exact quantized token equality.
- [done] Confirm switching v1/v2 cannot reuse the wrong tokenizer/model cache using the offline probe. No new broad benchmark matrix or tests of unchanged vision/MoE math.

QA evidence: not run.

## Phase 4: Long documents with bounded working memory

Outcome: `stream_pages` processes documents spanning many windows without retaining all images, K/V, router history, or generated text. The feature is complete only after this checkpoint.

Files: Python API/frontend/FFI; native session lifecycle, sequence state, memory planner, vision scheduling, and Metal cache handling.

- [todo] Add a request-owned native document session that can prefill the next window while retaining the exact upstream continuation state. Keep the existing engine lock for the session lifetime; cancellation/error/iterator close releases session buffers and the lock.
- [todo] Consume pages lazily: one owned-page window plus one lookahead. Reuse the lookahead encoding/KV in the next window. Size buffers from window/page-token limits and the memory budget, including tail-copy headroom. Expose a page-window setting; fail clearly if the minimum cannot fit.
- [todo] Implement the verified copy-splice contract: retain last-page marker, aligned image tail (128-159 tokens), up to 128 text tokens, and lookahead; copy main/router K/V, rebuild summaries/page state, preserve global page parity, and force the next page marker without duplicating output. Keep window EOS separate from document completion. Fail explicitly on incomplete/noncontiguous continuation.
- [todo] Yield output incrementally. Keep native token history and Python bookkeeping bounded too; release processed image views and output buffers. The convenience joining API may grow with output size, and caller-owned preloaded images are outside the working-memory guarantee.
- [todo] Document supported window/token limits and cross-window context behavior. Total document length must not be limited by a single-window positional cap.

QA checkpoint:

- [todo] Use one document spanning at least three small windows, with content continuing across a boundary. Compare with the pinned SGLang scheduler using the same settings: every page appears once, in order, with continuation and window stopping behavior preserved. Full-cache HF output is not the scheduler oracle.
- [todo] Repeat the page input to increase document length by 10x, consume/discard streamed chunks, and measure live/peak native allocations and process memory. With fixed window settings, active memory must plateau rather than grow with completed pages. No broad benchmark suite.
- [todo] Close a stream mid-document, then reuse the engine successfully. Run the existing fast C/Python suite once before finishing.

QA evidence: source inspection complete; runtime checks not run. SGLang windowed inference is the continuation oracle, not full-cache HF. Do not mark bounded-memory support complete until the memory check passes.
