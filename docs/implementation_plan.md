# Unlimited-OCR C/Metal fastest-engine implementation plan

Goal: make this repository the fastest practical engine for running the
`baidu/Unlimited-OCR` model on Metal. This plan is based on a fresh code read of
`src/backend/metal/uocr_metal.m`, `src/backend/metal/kernels/uocr_smoke.metal`,
`src/core/uocr_api.c`, and `src/runtime/uocr_memory.c`.

Status legend:

- [x] Finding is confirmed in the current codebase.
- [ ] Implementation work remains.

## Non-negotiable optimization rules

Impact: speed work is not complete without timings. Every optimization must be
accepted or rejected from comparable measurements on the same machine, model,
request, resource path, and build type. The repository should keep only the
confirmed optimized production implementation.

Implementation status:

- [ ] Capture baseline timings before changing an optimized area.
- [ ] Capture after-change timings on the same request/model/machine.
- [ ] Record stage timings, not only total wall time, so the improved stage is visible.
- [ ] Treat an optimization as incomplete until correctness and timing improvement are both recorded.
- [ ] Do not land fallback or dual production paths.
- [ ] Allow temporary comparison code only during local profiling; delete it before merge.
- [ ] When an optimized path is confirmed, remove the old naive path, runtime toggle, unused kernels, and unused helpers from production code.
- [ ] If a candidate is not faster or is less correct, revert/delete it instead of preserving it as a fallback.

## Code-read corrections to the previous plan

Impact: some prior optimization notes described a newer/intended GPU vision path,
but the current public image path is more host-resident and therefore has bigger
opportunities than scratch aliasing alone.

Implementation status:

- [x] Public image generation now calls `uocr_metal_context_generate_image_f16()`, which encodes formatted visual rows into a reusable Metal vision workspace and passes that slice into prompt assembly without a host visual-feature allocation.
- [x] The production vision path uses reusable top-level Metal workspace slices for SAM/CLIP/concat/projector/final rows; many per-block host buffers still remain inside transformer helpers (`src/backend/metal/uocr_metal.m`).
- [x] Many vision helpers allocate `newBufferWithBytes` / `newBufferWithLength`, commit, wait, and `memcpy` back to host for each stage.
- [x] `UOCR_METAL_ARENA_VISION_SCRATCH` is no longer allocated from stale prompt-token estimates at engine open; production vision workspace now grows through the request-shaped Metal vision scratch buffer.
- [x] Integrated decode already uses a fused GPU LM-head argmax kernel (`uocr_lm_head_argmax_f16`) for the public `uocr_metal_context_generate_f16()` path (`src/backend/metal/uocr_metal.m:5222`, `src/backend/metal/kernels/uocr_smoke.metal:2024`).
- [x] The old full-logits LM-head path still exists in helper/diagnostic code (`uocr_metal_context_lm_head_f16`, `src/backend/metal/uocr_metal.m:13084`) and should not be treated as the public decode path.

## Highest-impact next work

1. [x] Add mandatory stage timing and memory profiling.
2. [ ] Make vision GPU-resident end-to-end: no per-stage host scratch, readback, or rematerialized weight buffers.
3. [ ] Optimize decoder token kernels beyond LM head: attention projections, dense MLP, routed/shared MoE, and binding lookup overhead.
4. [ ] Correct runtime memory accounting and switch arenas to request-sized lazy/grow-on-demand allocation.
5. [ ] Cache direct binding structures, MPS descriptors/NDArrays, and expert slab metadata.
6. [ ] Replace scalar patch/conv/dense reductions with tiled/simdgroup kernels or confirmed faster MPS/GEMM forms.
7. [ ] Add measured q8/q4 decoder support only if it improves the target speed/memory tradeoff.

## 0. Measurement and regression gates

Impact: without timings, we cannot know whether an optimization improves OCR
latency or simply moves cost between CPU allocation, command encoding, GPU
compute, synchronization, and memory traffic.

Implementation status:

- [x] Add an opt-in profiling mode for public OCR (`profile=True` or `UOCR_PROFILE=1`).
- [x] Record wall timings for Python preprocessing, engine open, request validation, vision, prompt assembly, prefill, first token, decode loop, and Python decode.
- [x] Record Metal timings for SAM patch/neck, SAM blocks, CLIP blocks, projector, formatter, prompt assembly, each decoder layer group, final norm, LM-head selection, no-repeat processing, and command-buffer waits.
- [x] Record allocation/object counts for Metal buffers, command buffers, command encoders, MPS descriptors, MPS NDArrays, NSArrays, and transient-retain arrays.
- [x] Record memory by category: model views, runtime arenas, KV cache, host vision scratch, private/shared vision workspace, transient buffers, logits/selection scratch, and total live/peak bytes.
- [x] Produce baseline profiles for `docs/test.png` with `preset="base"`, `preset="gundam"`, `max_length=4096`, and default OCR settings.
- [ ] Reject optimization PRs that do not include comparable before/after timing evidence.
- [ ] Reject optimization PRs that keep fallback, duplicate, or unselected production paths after the optimized path is confirmed.

## 1. Make vision GPU-resident end-to-end

Current finding:

- [x] `uocr_metal_context_encode_visual_features_f16()` now reuses a Metal vision workspace for top-level SAM/CLIP/concat/projected/final rows; the diagnostic host-output API copies the final slice out only for callers that request it.
- [x] CLIP/SAM transformer block functions still allocate temporary host buffers inside every block (`src/backend/metal/uocr_metal.m`).
- [x] Vision stage helpers still repeatedly create Metal buffers from host pointers and wait for completion before copying outputs back.
- [x] Public image generation now splices the final Metal visual-feature slice into the prompt arena without creating a temporary host visual-feature upload.

Impact: this is likely the largest first-token and image-encoding bottleneck in
the current public path. The engine is paying CPU allocation cost, shared-buffer
copy cost, GPU/CPU synchronization, and command-buffer setup for many small
stages. A fastest engine should keep pixels/features on GPU from patch embed
through prompt arena assembly.

Implementation status:

- [x] Replace host vision scratch with reusable Metal slices for SAM, CLIP, concat, projector, and final formatted rows.
- [ ] Encode the complete vision graph into Metal buffers without per-stage readback.
- [x] Write final formatted visual rows directly into a Metal visual-feature slice or directly into the prompt arena.
- [ ] Keep only one production vision path after timing confirms the GPU-resident implementation; delete the host-scratch production path.
- [ ] Preserve diagnostic parity entrypoints only if they are clearly out of the production path and do not keep slow fallback code alive.
- [ ] Capture before/after timings for total vision, per SAM/CLIP/projector stage, command-buffer wait time, and CPU allocation count.

## 2. Correct vision memory accounting and workspace sizing

Current finding:

- [x] `runtime_arena_capacities()` now leaves `UOCR_METAL_ARENA_VISION_SCRATCH` empty; request-shaped vision workspace allocation is handled by `UOCR_METAL_SCRATCH_VISION`.
- [x] The current production vision path does not use that runtime arena.
- [x] The estimator now mirrors the reusable Metal vision workspace high-water for SAM, CLIP, concat, projected chunk, and final visual rows, while still grouping those bytes under one “vision scratch” category (`src/runtime/uocr_memory.c`).
- [x] A 4096-token default engine no longer reserves a stale prompt-sized vision arena at open; final visual/workspace estimates are now attached to accepted image requests.

Impact: memory reports and admission decisions are wrong. They over-reserve an
unused arena, under-describe actual host/transient vision allocations, and make
it harder to run small requests on constrained machines.

Implementation status:

- [x] Remove unused `UOCR_METAL_ARENA_VISION_SCRATCH` from production allocation or repurpose it as the confirmed GPU-resident visual workspace.
- [ ] Separate memory categories for final visual features, GPU vision workspace, host staging, and transient buffers.
- [x] Size final visual features from actual visual rows and hidden size, not prompt token capacity.
- [x] Size vision workspace from actual view shape and chunk size.
- [x] Include vision workspace high-water marks in `uocr_engine_memory_report()`.
- [ ] Update OOM diagnostics to show requested views, visual rows, workspace bytes, and configured budget.
- [ ] Verify memory reports against measured allocations before and after the GPU-resident vision rewrite.

## 3. Optimize decoder token compute beyond LM head

Current finding:

- [x] Public integrated decode already uses fused LM-head argmax, but the rest of per-token decode still runs many custom reductions and kernels per layer.
- [x] Attention Q/K/V/O, dense layer-0, shared experts, routed experts, router, norms, RoPE, KV write, attention, and combine are separate kernels (`src/backend/metal/uocr_metal.m:6282`).
- [x] Many custom matvec kernels use one threadgroup per output element with serial dot-product reductions (`src/backend/metal/kernels/uocr_smoke.metal:2193`, `:2433`, `:2567`, `:3513`, `:3614`).
- [x] MoE compute is larger than LM head over a full decode token: routed/shared experts across 11 MoE layers dominate total decoder FLOPs.

Impact: focusing only on LM head will not make decode fastest. Per-token latency
is dominated by the full decoder layer stack: QKV/O projections, layer-0 MLP,
router/top-k, routed experts, shared experts, and combine. Optimizing these
kernels and reducing launches is required for best tokens/sec.

Implementation status:

- [ ] Add per-token timing buckets for final norm/LM selection, attention projections, attention, dense layer-0 MLP, MoE router, shared experts, routed experts, combine, and command encoding.
- [ ] Replace scalar one-output/threadgroup matvec reductions with tiled/simdgroup kernels using vectorized loads and better data reuse.
- [ ] Benchmark custom decode matvec kernels against MPS/GEMV for `n_tokens=1`; keep only the confirmed faster production path per operation shape.
- [ ] Fuse decode-layer epilogues where possible: norm+projection setup, projection+bias, output+residual, MoE combine+residual.
- [ ] Consider packed QKV/O and packed gate/up forms where tensor layout or conversion can make them contiguous.
- [ ] Validate generated-token parity on fixed prompts and `docs/test.png` before deleting the old kernels.
- [ ] Delete slower/unselected production kernels and runtime toggles after timing selects the implementation.

## 4. LM-head selection: keep fused path, improve it only with timings

Current finding:

- [x] Integrated public generation uses `metal_select_next_token_from_hidden_slice_f16()` and `uocr_lm_head_argmax_f16`, avoiding a full f32 logits write for the public decode loop (`src/backend/metal/uocr_metal.m:5385`, `src/backend/metal/kernels/uocr_smoke.metal:2024`).
- [x] A naive full-logits helper path still exists for non-integrated helpers (`src/backend/metal/uocr_metal.m:13084`, `:13713`).
- [x] The current fused path is top-1 only and still uses a custom dot/reduction scheme.

Impact: LM head is still a major decode cost, but the public path is no longer
the old full-logits path. Work here should optimize the existing fused selector,
not add fallback paths.

Implementation status:

- [ ] Time final RMSNorm, no-repeat ban collection, fused LM-head partial reduction, final argmax, token readback, and CPU sequence update separately.
- [ ] Tune `uocr_lm_head_argmax_f16` tile size, lanes per token, threadgroup memory usage, and weight access pattern.
- [ ] Add top-k only if a measured feature needs it; otherwise keep top-1 greedy minimal.
- [ ] Move no-repeat history to a reusable GPU sequence buffer to avoid copying the CPU sequence into scratch every token.
- [ ] Remove the CPU rewrite of `token_slot` after argmax if timing/correctness confirms the GPU-written slot can be reused directly.
- [ ] Delete or quarantine the full-logits LM-head helper from production code after parity tests no longer need it.

## 5. Cache decoder binding lookup and MoE expert slab metadata

Current finding:

- [x] Decoder bindings are stored in a flat array and searched linearly (`metal_find_decoder_binding`, `src/backend/metal/uocr_metal.m:1985`).
- [x] Hot decode code calls `metal_require_decoder_binding()` repeatedly per layer/token (`src/backend/metal/uocr_metal.m:5065`, `:6282`).
- [x] `metal_expert_interleaved_slab_for_layer()` revalidates all 64 experts for a layer and is called from prefill/decode (`src/backend/metal/uocr_metal.m:6110`, `:6452`, `:6656`).

Impact: this is avoidable CPU overhead in the decode loop. It also adds branchy
validation work to the hottest path and can hide GPU improvements behind command
encoding overhead.

Implementation status:

- [ ] Build direct decoder binding structs at model map time: per-layer norms, Q/K/V/O, layer-0 MLP, each MoE router/shared, and each routed expert slab.
- [ ] Validate MoE expert contiguity once during binding, cache one slab slice per MoE layer, and remove per-token expert-slab scans.
- [ ] Replace linear tensor-id lookup in hot decode/prefill with direct cached pointers/slices.
- [ ] Add profiling counters for binding lookup and expert-slab validation before removal.
- [ ] Delete the old hot-path lookup path after cached bindings are validated.

## 6. Convert vision weights to direct Metal slices

Current finding:

- [x] Validated vision bindings already contain Metal buffer/offset metadata, but `uocr_metal_vision_weights_f16` converts them back to `const uint16_t *` host pointers (`src/backend/metal/uocr_metal.m:2460`, `:2482`).
- [x] Vision ops then create fresh Metal buffers from those host pointers for weights, biases, and activations.

Impact: this causes repeated model-backed host access, buffer creation,
retain/release traffic, and prevents long-lived MPS weight objects. It also
blocks a clean GPU-resident vision graph.

Implementation status:

- [ ] Change vision weight structs to store `uocr_metal_buffer_slice` directly.
- [ ] Build all vision slices once during model binding validation.
- [ ] Update SAM, CLIP, projector, newline, and separator code to consume slices, not host pointers.
- [ ] Remove hot-path weight `newBufferWithBytes` creation.
- [ ] Cache any required packed/reordered vision weights at engine open only if timing proves the packed form is faster.
- [ ] Delete the host-pointer production path after GPU-resident vision is confirmed.

## 7. MPS object churn and matmul path selection

Current finding:

- [x] `metal_run_mps_matmul_nt_f16()` creates descriptors, NDArrays, NSArrays, transient-retain arrays, and a destination fill encoder per call (`src/backend/metal/uocr_metal.m:4397`, `:4443`).
- [x] Runtime env toggles and thresholds choose MPS vs custom kernels (`UOCR_METAL_DISABLE_MPS`, `UOCR_METAL_MPS_MATMUL_MIN_FLOPS`).
- [x] The hot-path allocation guard does not catch Objective-C/MPS allocations.

Impact: MPS can be fast for large prefill/vision matmuls, but per-call object
churn and zero-fill blits add CPU and command overhead. The fastest engine needs
confirmed per-shape implementations, not runtime fallback paths.

Implementation status:

- [ ] Add MPS allocation/object counters and per-call timings.
- [ ] Cache stable descriptors and long-lived weight NDArrays by shape/layout/transpose.
- [ ] Reuse input/output NDArray wrappers where buffer lifetimes allow.
- [ ] Measure whether destination zero-fill blits are required; remove them if confirmed unnecessary.
- [ ] Benchmark MPS vs custom kernels per operation shape: vision block matmuls, decoder prefill matmuls, decode matvecs.
- [ ] Keep only the confirmed production implementation per shape/stage; delete unselected runtime toggles and fallback code.

## 8. Fuse dense epilogues and packed projections

Current finding:

- [x] Many paths split matmul/projection from bias, GELU/QuickGELU, and residual.
- [x] SAM/CLIP transformer blocks perform LayerNorm, QKV, attention, projection, residual, MLP, and residual as separate host-orchestrated calls.
- [x] QKV is often three projections rather than one packed projection in MPS paths.

Impact: separate epilogues increase launch count and activation memory traffic.
Packed projections and fused epilogues reduce both GPU work and CPU command
encoding overhead.

Implementation status:

- [ ] Add fused `bias + GELU`, `bias + QuickGELU`, and `bias + residual` kernels.
- [ ] Fold QKV bias into attention load or use a fused projection-major bias kernel.
- [ ] Benchmark packed QKV matmul plus split/attention against three separate Q/K/V matmuls.
- [ ] Apply fusions only where before/after timings improve the target stage.
- [ ] Remove split epilogue production code after fused kernels are confirmed.

## 9. Replace scalar patch/conv kernels with tiled kernels

Current finding:

- [x] SAM patch embed loops 768 MACs per output element serially (`src/backend/metal/kernels/uocr_smoke.metal:237`).
- [x] SAM neck and stride convs use one-output/threadgroup reductions (`src/backend/metal/kernels/uocr_smoke.metal:1176`, `:1251`, `:1349`).

Impact: these kernels waste GPU parallelism and are on every image request.
Tiled/simdgroup kernels or GEMM formulations should improve vision latency once
vision is GPU-resident.

Implementation status:

- [ ] Benchmark patch embed, neck 1x1, neck 3x3, net_2, and net_3 separately.
- [ ] Replace patch embed with a tiled/simdgroup kernel or confirmed faster GEMM/im2col formulation.
- [ ] Replace 1x1 and 3x3 convs with tiled kernels computing multiple outputs per threadgroup.
- [ ] Validate base/global and local crop parity.
- [ ] Delete scalar production kernels after tiled kernels are confirmed faster.

## 10. Batch views and pages where shapes match

Current finding:

- [x] `metal_project_vision_chunk_f16()` loops over views inside a chunk and processes each view serially (`src/backend/metal/uocr_metal.m:3008`).
- [x] `max_views_per_chunk` affects host scheduling but does not produce true GPU batched SAM/CLIP execution.

Impact: crop-mode and multi-page OCR repeat setup and underuse the GPU. Batched
chunks should improve throughput when multiple local crops or pages have the
same shape.

Implementation status:

- [ ] Batch local `640x640` views together and global `1024x1024` views together.
- [ ] Use batched MPS matmuls or batched custom kernels for matching transformer shapes.
- [ ] Size workspace by chunk shape and liveness.
- [ ] Preserve final local/global/page visual order exactly.
- [ ] Add timing cases for many local crops and multi-page base OCR.
- [ ] Keep only the batched production path if timing confirms it beats serial processing.

## 11. Lazy/grow-on-demand runtime arenas

Current finding:

- [x] Engine open allocates runtime arenas for configured maxima (`src/core/uocr_api.c:651`).
- [x] Default settings allocate for 4096 prompt tokens even when a base request is around 277 tokens.
- [x] For 4096 prompt tokens, non-model runtime arenas excluding stale vision/safety are roughly 365 MiB; for 277 tokens they are roughly 32 MiB.

Impact: worst-case allocation at open raises baseline memory, slows startup, and
makes small requests harder to run. Request-sized arenas also make memory timing
more interpretable.

Implementation status:

- [ ] Allocate arenas lazily from the first request or grow before command encoding.
- [ ] Keep hard max caps for reusable engines, but size current capacity to actual request high-water marks.
- [ ] Never grow inside the per-token hot loop.
- [ ] Expose current capacity and high-water usage in memory reports.
- [ ] Update admission errors with requested and configured limits.
- [ ] Delete eager worst-case allocation from the production open path after lazy growth is validated.

## 12. Quantized decoder support, measured as speed/memory tradeoff

Current finding:

- [x] Public Metal generation rejects non-fp16 qprofiles (`src/core/uocr_api.c:182`, `:261`).
- [x] The fp16 model tensor payload is about 6.36 GiB.
- [x] q8/q4 kernels and converter pieces exist, but they are not integrated into public generation.

Impact: fp16 model size dominates memory. q8/q4 can expand device coverage and
may improve bandwidth-bound stages, but quantized kernels must be timed because
some dequant paths can be slower than fp16 on high-memory GPUs.

Implementation status:

- [ ] Define the exact supported mixed profile: likely fp16 vision plus q8/q4 decoder first.
- [ ] Integrate q8 decoder bindings and kernels with direct cached slices.
- [ ] Integrate q4 routed-expert kernels after q8 correctness and timings.
- [ ] Keep router/norm/sensitive tensors fp16 unless calibration and timings prove otherwise.
- [ ] Compare fp16 vs q8/q4 on first-token latency, decode tokens/sec, and peak memory.
- [ ] Keep only confirmed production qprofile paths; do not preserve slow quant fallbacks.

## 13. GPU visual formatting and prompt splice

Current finding:

- [x] Visual formatting is still host-side in `uocr_process_vision_chunks_f16()` (`src/runtime/uocr_vision.c:249`), but public image generation writes those rows into a reusable Metal visual-feature slice.
- [x] Public prompt assembly now binds the Metal visual-feature slice directly; diagnostic host-pointer prompt assembly still supports temporary uploads.

Impact: formatting and prompt splice are smaller than SAM/CLIP/decoder compute,
but they currently force host residency. Once vision is GPU-resident, keeping
formatting and prompt assembly on GPU avoids a final round-trip.

Implementation status:

- [ ] Add a Metal formatter that writes newline/separator rows and crop/global order directly.
- [ ] Prefer projector output directly into final visual layout when shape/order allows.
- [x] Splice GPU visual rows into prompt arena without host copy.
- [ ] Validate byte-for-byte final visual rows against the current formatter.
- [ ] Remove host formatting from the production path after the GPU formatter is confirmed.

## 14. Production-path cleanup: no fallback code

Current finding:

- [x] The file contains many helper/diagnostic CPU-pointer APIs that allocate buffers, copy to GPU, wait, and copy back.
- [x] Some helpers are useful for parity tests but are too slow to remain selectable production implementations.
- [x] Runtime toggles such as disabling MPS or changing MPS thresholds can preserve slow paths indefinitely.

Impact: fastest-engine work will regress if old slow paths remain easy to call
or accidentally become production fallbacks. The implementation should converge
toward one measured path per stage/shape.

Implementation status:

- [ ] Mark diagnostic-only APIs clearly and keep them out of public OCR dispatch.
- [ ] Delete unused slow production helpers as optimized replacements land.
- [ ] Remove runtime fallback toggles after measurements select the implementation.
- [ ] Keep tests on the confirmed path; if parity needs old behavior, use offline fixtures rather than production fallback code.
- [ ] Add CI/static checks or grep-based tests for disallowed production fallbacks where practical.

## Recommended execution order

Impact: this sequence first makes performance visible, then removes the largest
host-residency issue, then attacks decode throughput and memory pressure.

Implementation status:

- [x] Land profiling/timing/memory instrumentation and capture baselines.
- [ ] Rewrite vision to be GPU-resident through prompt splice; remove host-scratch production path.
- [ ] Fix vision memory accounting to match the new GPU-resident workspace.
- [ ] Cache decoder bindings and MoE expert slabs to remove hot-loop CPU lookup/validation.
- [ ] Optimize decode matvec/MoE kernels, including LM-head tuning, with before/after timings.
- [ ] Cache MPS descriptors/NDArrays and select one measured matmul path per shape.
- [ ] Fuse dense epilogues and packed projections where timings improve.
- [ ] Replace scalar patch/conv kernels.
- [ ] Add real view batching for crop/multi-page.
- [ ] Implement lazy/grow-on-demand arenas.
- [ ] Add measured q8/q4 decoder support.
- [ ] Delete remaining unselected fallback/duplicate production code.
