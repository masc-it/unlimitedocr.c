# Unlimited-OCR C/Metal optimization implementation plan

This plan tracks the next performance and memory work for the current fp16
Metal image-to-text engine. It intentionally focuses on decode throughput,
vision memory correctness, vision/MPS object reuse, and high-cost vision
kernels.

Status legend:

- [x] Finding is identified and accepted as current behavior.
- [ ] Implementation work remains.

## Highest-impact next work

1. [ ] GPU-resident decode token selection plus fused/optimized LM-head top-k.
2. [ ] Fix vision arena/workspace accounting and add liveness-based private scratch aliasing.
3. [ ] Convert vision weights to direct Metal slices and cache MPS descriptors/weight arrays.
4. [ ] Fuse dense epilogues to reduce kernel launches and memory passes.
5. [ ] Replace scalar patch/conv kernels with tiled high-performance kernels.

## 0. Measurement and regression gates

Impact: every optimization below should be measured against a stable baseline so
we know whether it improves real OCR latency, decode tokens/sec, peak memory,
and open/request overhead instead of only moving cost between stages.

Implementation status:

- [ ] Add an opt-in profiling mode for the public path (`profile=True` or `UOCR_PROFILE=1`).
- [ ] Record wall timings for preprocessing, engine open, vision encode, prompt assembly, prefill, LM head/token selection, first token, decode loop, and Python decode.
- [ ] Record Metal-stage timings for SAM patch/neck, SAM blocks, CLIP blocks, projector, formatter, decoder prefill, final norm, LM head, argmax/top-k, and no-repeat handling.
- [ ] Record memory by category: model views, runtime arenas, KV cache, visual-feature storage, private vision workspace, transient buffers, logits/readback, and total live/peak bytes.
- [ ] Produce baseline profiles for `docs/test.png` with `preset="base"`, `preset="gundam"`, `max_length=4096`, and the default OCR settings.
- [ ] Require before/after profile notes in PRs that change kernels, scratch sizing, MPS object lifetime, or decode token selection.

## 1. GPU-resident decode token selection and optimized LM-head top-k

Current finding:

- [x] `metal_run_lm_head_buffer_f32` uses `uocr_dense_f16_to_f32` for the LM head.
- [x] The decode path does not use MPS/GEMV or a fused LM-head/top-k kernel.
- [x] Current cost is one threadgroup per vocab output and a full f32 logits write.
- [x] Relevant files: `src/backend/metal/uocr_metal.m:5972`, `src/backend/metal/kernels/uocr_smoke.metal:577`.

Impact: this is the most important decode-loop optimization. The current path
writes all vocab logits and then performs token selection separately, creating
large memory traffic and avoidable launch/readback overhead per generated token.
A GPU-resident top-1/top-k path should improve first-token latency and decode
tokens/sec, and it is a prerequisite for keeping greedy/no-repeat selection off
the CPU.

Implementation status:

- [ ] Add detailed timing around final norm, LM head, logits write, no-repeat masking, argmax/top-k, and token readback.
- [ ] Add an intermediate GPU argmax/top-k kernel over the existing logits buffer to remove CPU-side logits scanning first.
- [ ] Keep only the selected token id/probability or compact top-k results visible to CPU unless debug profiling requests full logits.
- [ ] Implement a fused LM-head + top-1/top-k kernel that accumulates `hidden x lm_head` and reduces directly to candidates without writing full logits.
- [ ] Compare custom fused LM-head against an MPS/GEMV + GPU argmax path; keep the faster and simpler path for decode.
- [ ] Integrate no-repeat-ngram banning into the GPU selection path or apply it as a compact GPU mask before final selection.
- [ ] Validate token parity against the current full-logits greedy path on fixed prompts and `docs/test.png`.
- [ ] Update memory accounting to reduce or eliminate full-size f32 logits/readback buffers when the fused path is active.

## 2. Vision memory accounting is wrong/stale

Current finding:

- [x] Runtime arena `VISION_SCRATCH` is sized by `uocr_estimate_vision_scratch_bytes_for_rows(prompt_token_capacity, 256, ...)`.
- [x] The current GPU path uses that arena only as final visual-feature storage.
- [x] Default 4096-token engines reserve about 24.6 MiB for “vision scratch”, while base-image final features need only about 0.67 MiB.
- [x] Real private vision workspace is allocated separately and is not represented correctly in the runtime estimate.
- [x] Relevant files: `src/backend/metal/uocr_metal.m:3636`, `src/backend/metal/uocr_metal.m:3672`, `src/backend/metal/uocr_metal.m:4755`, estimator `src/runtime/uocr_memory.c:150`.

Impact: public memory reports and admission decisions are misleading. The engine
over-reserves a stale runtime arena while under-reporting the actual private
vision workspace. This can reject valid requests, admit requests that later OOM,
and hide the true source of peak memory.

Implementation status:

- [ ] Rename or reclassify `VISION_SCRATCH` as final visual-feature storage if that is its only runtime use.
- [ ] Size final visual-feature storage from actual visual rows and feature width, not prompt token capacity.
- [ ] Add a separate estimated category for private vision workspace.
- [ ] Include private vision workspace in `uocr_engine_memory_report` and memory admission.
- [ ] Report requested rows/views and computed final-feature bytes in OOM diagnostics.
- [ ] Add estimator cases for base global view, gundam local+global views, and multi-page base requests.
- [ ] Remove or update stale `uocr_estimate_vision_scratch_bytes_for_rows(..., 256, ...)` assumptions.

## 3. Vision private scratch is oversized and not liveness-optimized

Current finding:

- [x] `metal_vision_device_scratch_init` allocates many always-live private slices.
- [x] Current allocations include 10 SAM hidden buffers, 6 SAM attention buffers, 14 CLIP buffers, and other stage buffers.
- [x] Current computed private scratch size is about 143 MiB after the first vision request.
- [x] Many buffers are mutually exclusive by phase and do not need unique always-live storage.
- [x] Relevant file: `src/backend/metal/uocr_metal.m:12530`.

Impact: the current vision path permanently raises GPU memory after the first
vision request. Liveness aliasing can significantly reduce private scratch peak,
make default engines fit on more machines, and leave more budget for KV cache,
longer outputs, or quantized model variants.

Implementation status:

- [ ] Inventory every private scratch slice with size, storage mode, producer stage, consumer stage, and last use.
- [ ] Split vision execution into explicit phases: SAM patch/pos, SAM blocks, SAM neck, CLIP embedding, CLIP blocks, projector, formatter.
- [ ] Build a liveness table and assign mutually exclusive buffers to alias groups.
- [ ] Size scratch by actual local/global view shape instead of a single worst-case layout.
- [ ] Replace always-live fixed slices with an aliasing allocator over one or more private scratch buffers.
- [ ] Track high-water private workspace bytes at runtime and expose them in memory reports.
- [ ] Validate aliasing with Metal debug validation enabled and with repeated base/gundam requests on a reusable engine.

## 4. Vision weights are cached as host pointers, then rematerialized

Current finding:

- [x] `uocr_metal_vision_weights_f16` stores `const uint16_t *` pointers.
- [x] Each op calls `metal_make_buffer_slice_from_bytes` to rematerialize a Metal slice.
- [x] This causes repeated model-backed pointer scans, retain/release traffic, possible buffer creation, and prevents clean MPS weight-array caching.
- [x] Relevant files: `src/backend/metal/uocr_metal.m:2352`, `src/backend/metal/uocr_metal.m:2434`, `src/backend/metal/uocr_metal.m:4157`, repeated around `src/backend/metal/uocr_metal.m:12612+`.

Impact: this adds CPU overhead on every vision op and blocks longer-lived MPS
object caches. Converting vision weights to direct Metal buffer slices should
reduce ObjC/Metal churn, simplify binding lifetime, and align vision with the
already-cached decoder binding model.

Implementation status:

- [ ] Change vision binding structs to store `uocr_metal_buffer_slice` directly, matching decoder bindings.
- [ ] Build all vision weight slices once during engine/model binding validation.
- [ ] Remove hot-path `metal_make_buffer_slice_from_bytes` calls for vision weights.
- [ ] Update SAM, CLIP, projector, newline, and view-separator bindings to consume cached slices.
- [ ] Ensure cached slices retain the underlying no-copy model buffers for the engine lifetime.
- [ ] Add binding error messages that name the missing tensor or invalid slice.
- [ ] Re-run repeated image requests and confirm no per-op model-slice rematerialization remains in profiles.

## 5. MPS object churn is high

Current finding:

- [x] Every MPS matmul creates descriptors, NDArrays, NSArrays, and transient retain entries.
- [x] The current hot-path guard does not catch Objective-C/MPS allocations.
- [x] Relevant files: `src/backend/metal/uocr_metal.m:4451`, `src/backend/metal/uocr_metal.m:4504`, `src/backend/metal/uocr_metal.m:4627`.

Impact: per-layer ObjC/MPS allocation churn increases CPU overhead, makes
profiling noisy, and can dominate smaller batched matmuls. Caching descriptors
and long-lived weight NDArrays should reduce request latency and make the vision
and prefill paths closer to allocation-free.

Implementation status:

- [ ] Audit all MPS helper paths for per-call descriptor, NDArray, NSArray, and retain-entry creation.
- [ ] Define cache keys by dtype, shape, strides/layout, transpose flags, batch count, and operation type.
- [ ] Cache stable MPS descriptors for repeated SAM/CLIP/projector/decoder shapes.
- [ ] Cache long-lived weight NDArrays for model-backed weights.
- [ ] Reuse input/output NDArray wrappers where buffer/shape lifetimes allow.
- [ ] Extend hot-path diagnostics to expose ObjC/MPS allocation counts or add scoped counters around MPS wrappers.
- [ ] Verify cached objects are invalidated only when engine/model/backend resources are destroyed.

## 6. Too many separate epilogue kernels

Current finding:

- [x] MPS matmul epilogues are split into separate bias, GELU/QuickGELU, and residual kernels.
- [x] Relevant files: `src/backend/metal/uocr_metal.m:5349`, `src/backend/metal/uocr_metal.m:12780`, `src/backend/metal/uocr_metal.m:12785`, `src/backend/metal/uocr_metal.m:12914`, `src/backend/metal/uocr_metal.m:12995`.

Impact: every separate epilogue kernel adds launch overhead and an extra memory
read/write pass over activations. Fusing common epilogues should improve SAM,
CLIP, projector, and decoder prefill latency, especially when many small layers
run sequentially.

Implementation status:

- [ ] Add fused epilogue kernels for `bias + GELU`.
- [ ] Add fused epilogue kernels for `bias + QuickGELU`.
- [ ] Add fused epilogue kernels for `bias + residual`.
- [ ] Fold QKV bias into attention load or use a fused projection-major bias kernel.
- [ ] Replace SAM MLP, CLIP MLP, attention projection, projector, and decoder dense epilogue call sites where shapes match.
- [ ] Validate fp16/fp32 accumulation and activation parity against the current split-kernel path.
- [ ] Measure kernel-launch count and activation memory traffic before and after fusion.

## 7. Patch/conv kernels are scalar, not high-performance tiled kernels

Current finding:

- [x] SAM patch embed loops 768 MACs per output element serially.
- [x] SAM neck/stride convs use one-output/threadgroup reductions.
- [x] Relevant files: `src/backend/metal/kernels/uocr_smoke.metal:237`, `src/backend/metal/kernels/uocr_smoke.metal:1176`, `src/backend/metal/kernels/uocr_smoke.metal:1251`, `src/backend/metal/kernels/uocr_smoke.metal:1349`.

Impact: scalar vision kernels can dominate image-encoding latency and waste GPU
parallelism. Tiled/simdgroup kernels or GEMM/MPS formulations should materially
improve first-token latency for every image request, including base mode.

Implementation status:

- [ ] Benchmark SAM patch embed and SAM neck/stride convs separately to rank kernel work.
- [ ] Replace patch embed with a tiled/simdgroup kernel or transform it into an MPS/custom GEMM.
- [ ] Replace SAM neck and stride convs with tiled convolution kernels that compute multiple outputs per threadgroup.
- [ ] Use vectorized loads and shared/threadgroup memory where beneficial for input tiles and weights.
- [ ] Validate numeric parity for base and gundam views.
- [ ] Measure occupancy, launch count, and stage latency after each kernel replacement.

## 8. Views are processed serially

Current finding:

- [x] The arena path loops over views one at a time using one scratch/projected buffer.
- [x] `max_views_per_chunk` is effectively not used for GPU batching.
- [x] This is especially inefficient for crop-mode and multi-page requests.
- [x] Relevant files: `src/backend/metal/uocr_metal.m:13115`, `src/backend/metal/uocr_metal.m:13136`, `src/backend/metal/uocr_metal.m:13137`.

Impact: serial view processing underutilizes the GPU and repeats dispatch/setup
costs for local crops and pages. Batched chunks should improve throughput for
crop-heavy `gundam` requests and multi-page OCR while preserving final visual
feature order.

Implementation status:

- [ ] Make `max_views_per_chunk` drive actual GPU batching.
- [ ] Batch views with matching shapes: local `640x640` together, global `1024x1024` together, and pages where possible.
- [ ] Allocate chunk scratch from the liveness-aware private workspace planner.
- [ ] Use batched MPS matmuls or batched custom kernels for CLIP/SAM transformer stages where shapes match.
- [ ] Preserve final visual-token order for local rows, global rows/newlines, and separators.
- [ ] Add profile cases for multiple local crops and multi-page base requests.

## 9. Engine allocates worst-case arenas at open

Current finding:

- [x] Defaults allocate arenas for 4096 prompt tokens even if a request is about 277 tokens.
- [x] Default runtime arenas excluding model are about 390 MiB.
- [x] A 277-token request would need far smaller prompt/KV/runtime allocations.
- [x] Relevant files: `src/core/uocr_api.c:488`, `src/core/uocr_api.c:625`.

Impact: worst-case allocation at engine open increases baseline memory, slows
startup, and prevents small requests from running on tighter memory budgets.
Lazy/grow-on-demand arenas sized to actual requests should reduce memory
footprint for common OCR calls while preserving configurable maximum caps.

Implementation status:

- [ ] Size owned helper engines from the prepared request instead of defaulting to 4096 prompt tokens.
- [ ] Convert runtime arenas to lazy/grow-on-demand buffers with explicit maximum caps.
- [ ] Keep reusable engines configurable: initial size, growth policy, and hard max prompt/gen tokens.
- [ ] Ensure growth happens before command encoding and never inside the per-token hot loop.
- [ ] Add clear oversize-request errors that include requested and configured limits.
- [ ] Expose current capacity and high-water usage in memory reports.
- [ ] Re-profile `docs/test.png` base request to confirm non-model memory drops substantially.

## 10. Metal path is fp16-only end-to-end

Current finding:

- [x] Public Metal generation rejects non-fp16 qprofiles.
- [x] Model tensor data is about 6.36 GiB fp16.
- [x] Relevant files: `src/core/uocr_api.c:182`, `src/core/uocr_api.c:261`.

Impact: fp16-only weights dominate memory and load footprint, limiting devices
that can run the engine. Integrated q8/q4 decoder support can reduce model
memory substantially. Vision can remain fp16 initially while decoder weights and
matmuls become quantized.

Implementation status:

- [ ] Define supported q8/q4 `.uocr` tensor layouts and metadata for decoder weights.
- [ ] Add q8 decoder matmul/dequant kernels or MPS-compatible dequantized weight caching.
- [ ] Support q4 decoder matmul/dequant after q8 is correct and profiled.
- [ ] Keep vision fp16 as an explicit supported mixed-profile mode unless/until vision quantization is implemented.
- [ ] Update model binding validation and public qprofile errors for mixed fp16-vision/q8-or-q4-decoder profiles.
- [ ] Update memory estimates for quantized model views, dequant scratch, and cached dequantized weights if used.
- [ ] Validate text output parity against fp16 on fixed prompts and `docs/test.png`.

## 11. Visual formatting uses many tiny blit copies

Current finding:

- [x] Global formatting emits many row-level blit copies.
- [x] This is lower priority than LM head, scratch accounting, MPS churn, epilogue fusion, and scalar conv/patch kernels.
- [x] Relevant file: `src/backend/metal/uocr_metal.m:13067`.

Impact: tiny blit copies add command overhead and can cause avoidable
synchronization/encoder fragmentation. The expected win is smaller than the
other items, but it is straightforward cleanup once projector/formatter ownership
is clearer.

Implementation status:

- [ ] Replace row-level blits with one formatting kernel that writes newlines and separators directly.
- [ ] Consider having the projector write directly into the final visual layout for global/base cases.
- [ ] Preserve crop-mode stitched local row order and multi-page order exactly.
- [ ] Validate final visual-feature bytes against the current formatter on base, gundam, and multi-page requests.
- [ ] Measure command-buffer/encoder count and formatting latency before and after the change.

## 12. Recommended execution order

Impact: this order targets the largest latency and memory risks first while
reducing dependencies between changes.

Implementation status:

- [ ] Land profiling and memory-report fixes needed to prove improvements.
- [ ] Implement GPU argmax/top-k after the current LM-head logits path.
- [ ] Implement fused LM-head + top-k or select MPS/GEMV + GPU argmax if faster.
- [ ] Correct vision final-feature arena accounting and add private workspace accounting.
- [ ] Implement liveness-based private vision scratch aliasing.
- [ ] Convert vision weight bindings from host pointers to cached Metal slices.
- [ ] Cache MPS descriptors and weight NDArrays.
- [ ] Fuse common dense epilogues.
- [ ] Replace scalar SAM patch/conv kernels with tiled kernels or GEMM formulations.
- [ ] Add batched view chunks for local crops and multi-page OCR.
- [ ] Add lazy/grow-on-demand runtime arenas for request-sized allocation.
- [ ] Add q8/q4 decoder support after fp16 memory/runtime behavior is stable.
- [ ] Clean up visual formatting blits.
