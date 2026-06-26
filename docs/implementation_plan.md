# Unlimited-OCR C/Metal fastest-engine implementation plan

Goal: make this repository the fastest practical engine for `baidu/Unlimited-OCR`
on Metal. The path to 10x is fixed in this document: one production execution
plan, one primitive choice per stage, and no open two-way decision points.

Status legend:

- [x] Confirmed item is present in the current codebase/profile.
- [ ] Implementation work remains.

## 0. Current measured baseline

Source baseline: `docs/baseline_profiles/docs-test-metal-profile.json`, captured
for `docs/test.png` with `preset="base"` and `preset="gundam"`, `max_length=4096`,
`max_gen_tokens=512`, default no-repeat settings, fp16 `.uocr`, Metal backend.

Key findings:

- [x] Base request takes **18.26s** total; `native.generate_prepared` is **17.71s**.
- [x] Base vision is **11.03s**; SAM transformer blocks are **10.01s**.
- [x] Base prompt assembly is **4.04s**, prefill is **1.48s**, decode loop is **1.15s**.
- [x] Base records **488** command-buffer waits totaling **17.01s**.
- [x] Gundam/crop request takes **126.53s** total; `native.generate_prepared` is **125.96s**.
- [x] Gundam vision is **101.04s**; SAM transformer blocks are **88.93s**.
- [x] Gundam prompt assembly is **4.44s**, prefill is **14.63s**, decode loop is **5.85s**.
- [x] Gundam records **11,110** command-buffer waits totaling **119.64s**.
- [x] The largest bottleneck is Metal vision, especially SAM; Python is secondary.
- [x] Local probing shows tokenizer load, request preparation, and token decode each cost roughly **210ms**.

10x target:

- [ ] Base `docs/test.png` target: **18.26s -> <=1.83s** on the same machine/build/model.
- [ ] Gundam `docs/test.png` target: **126.53s -> <=12.65s** on the same machine/build/model.
- [ ] Report stage timings before/after, not just total wall time.
- [ ] Keep generated ids/text parity acceptable for fixed golden requests.

## 1. Fixed technical decisions for the 10x path

These are the chosen production decisions. Do not add runtime selection between
alternate implementations in the public OCR path.

Implementation status:

- [ ] Use **same-shape batched vision command graphs**: one command buffer for the full local `640x640` batch and one command buffer for the global `1024x1024` batch.
- [ ] Use **GPU-private vision workspace** for all intermediate SAM, CLIP, concat, projector, and formatted visual rows.
- [ ] Use **MPSNDArray matrix multiplication** for large dense GEMMs in SAM, CLIP, projector, and decoder prefill.
- [ ] Use **custom tiled Metal kernels** for SAM attention, SAM patch/neck/stride convolutions, visual formatting, decode single-token matvecs, routed/shared MoE, and LM-head argmax.
- [x] Use **persistent GPU token-id buffers** and fold prompt assembly into the prefill command graph.
- [x] Use **request-sized lazy arena growth before command encoding**; never grow inside a hot loop.
- [ ] Use **cached Python tokenizers** keyed by resolved tokenizer path.
- [ ] Keep the 10x milestone on **fp16 weights**; q8 decoder is a follow-up compression milestone after fp16 speed is fixed.
- [ ] Remove production runtime toggles that preserve slow paths after the chosen path lands.

## 2. Non-negotiable optimization rules

Implementation status:

- [x] Profiling exists for Python preprocessing, engine open, request validation, vision, prompt assembly, prefill, first token, decode loop, Python decode, Metal waits, allocation/object counts, and memory categories.
- [x] Baseline profiles exist for `docs/test.png` base and gundam.
- [ ] Capture before/after timings on the same machine, model, request, resource path, and build type for every optimized area.
- [ ] Keep default tests and CI additions under **30 seconds** each; move longer perf runs behind explicit manual commands.
- [ ] Treat an optimization as incomplete until correctness and timing improvement are both recorded.
- [ ] Do not land permanent runtime toggles that keep slow paths alive.
- [ ] Temporary comparison code is allowed only during local profiling; delete it before merge.
- [ ] When the chosen optimized path is confirmed, remove the old naive path, duplicate kernels, fallback selection, and unused helpers from production dispatch.
- [ ] Revert a candidate that fails the combined speed and correctness gates.

## 3. Top edit #1: same-shape batched vision command graphs

Impact: this is the largest gundam/crop win. Current gundam spends **101s** in
vision and records **11,110** waits. The chosen fix is one batched GPU-resident
vision graph per shape group.

Current finding:

- [x] Public image generation enters `uocr_metal_context_generate_image_f16()`.
- [x] `src/core/uocr_api.c` now computes a same-shape chunk limit for both schedule planning and Metal dispatch.
- [x] `metal_project_vision_chunk_f16()` loops over views and calls `metal_encode_one_view_projected_f16()` serially.
- [x] `max_views_per_chunk` is currently scheduling metadata only; it does not create true batched SAM/CLIP execution.
- [x] Production vision has reusable top-level workspace slices, but helpers still retain pointer-oriented inputs/outputs and wait per operation.
- [x] Command-buffer wait time nearly equals total native time in both base and gundam profiles.

Implementation status:

- [x] Change public image generation to pass the full same-shape batch size into `uocr_plan_vision_schedule()` and `uocr_metal_context_generate_image_f16()`.
- [x] Partition each request into exactly two shape groups: all local `640x640` views, then all global `1024x1024` views.
- [ ] Add a batch dimension to SAM patch, SAM transformer, SAM neck, CLIP embedding, CLIP transformer, concat, projector, and formatter dispatch.
- [x] Encode the entire local shape group in one command buffer.
- [x] Encode the entire global shape group in one command buffer.
- [x] Make production vision helpers accept `uocr_metal_buffer_slice` inputs and outputs, not host pointers.
- [x] Route production per-view projected-row output through Metal workspace slices so chunk formatting consumes the reusable workspace slice directly.
- [ ] Allocate vision intermediates from GPU-private workspace slices.
- [ ] Keep only input pixels and final generated-token readback CPU-visible.
- [ ] Preserve exact final feature order: local crops first, global rows, newline rows, separator rows, and multi-page order.
- [ ] Add profiles for base, max-crop gundam, and multi-page base after batching lands.
- [ ] Delete the serial one-view production path after batched dispatch passes parity and timing gates.

Acceptance target:

- [ ] Gundam vision drops from **101.04s** to **<=12s** before SAM kernel rewrites.
- [ ] Gundam command-buffer waits drop from **11,110** to **<=100**.

## 4. Top edit #2: custom tiled SAM attention and vision kernels

Impact: base mode still spends **10.01s** of **11.03s** vision time in SAM blocks.
The chosen fix is custom tiled Metal for attention and convolutions, plus MPSNDArray
for large dense GEMMs.

Current finding:

- [x] SAM transformer dominates base and gundam vision profiles.
- [x] `uocr_sam_window_attention_flash_impl()` and `uocr_sam_rel_pos_attention_flash_impl()` loop over keys per query and recompute relative-position contributions in the hot path.
- [x] Dense kernels such as `uocr_dense_dot_f16()` use one threadgroup per output element with serial dot-product reduction.
- [x] SAM patch, neck, and stride conv kernels use one-output/threadgroup reduction patterns.
- [x] Global-attention SAM blocks 02/05/08/11 are ~1.43s each in base.

Implementation status:

- [ ] Add microbenchmarks for SAM patch embed, SAM window block, SAM global block, SAM neck 1x1, SAM neck 3x3, net_2, net_3, CLIP block, and projector.
- [ ] Rewrite SAM global attention with a tiled FlashAttention-style kernel processing multiple queries per threadgroup and reusing K/V tiles.
- [ ] Rewrite SAM window attention with the same tiled attention kernel specialized for `14x14` windows.
- [ ] Precompute SAM relative-position bias tables during conversion for `64x64`, `40x40`, and `14x14` attention shapes.
- [ ] Precompute SAM absolute position tables during conversion for `64x64` and `40x40` patch grids.
- [ ] Precompute CLIP absolute position tables during conversion for `16x16` and `10x10` patch grids.
- [x] Use MPSNDArray matrix multiplication for SAM QKV, SAM output projection, SAM MLP, CLIP QKV, CLIP output projection, CLIP MLP, and projector dense GEMMs.
- [x] Cache all MPS descriptors and weight NDArrays at model map time.
- [x] Replace SAM patch embed with a custom tiled patch kernel.
- [x] Replace SAM neck 1x1, neck 3x3, net_2, and net_3 with custom tiled convolution kernels.
- [ ] Fuse vision epilogues into fixed kernels: bias, GELU, QuickGELU, residual add, and layout conversion.
- [ ] Validate base/global and local-crop parity against current fixture/golden outputs.
- [ ] Delete scalar production kernels after tiled kernels and MPS GEMMs pass parity and timing gates.

Acceptance target:

- [ ] Base vision drops from **11.03s** to **<=1.1s**.
- [ ] Base SAM blocks drop from **10.01s** to **<=0.8s**.

## 5. Top edit #3: prompt assembly fused into prefill, then custom decode kernels

Impact: once vision is fixed, prompt assembly, prefill, and decode become the
ceiling. The chosen fix is persistent token buffers, prompt assembly inside the
prefill command graph, MPSNDArray for prefill GEMMs, and custom tiled kernels for
single-token decode.

Current finding:

- [x] Prompt assembly uses `metal_context_assemble_prompt_f16_with_table_buffer_to_buffer()` and dispatches a full per-token/per-hidden kernel.
- [x] Prompt assembly uploads a new token-id buffer and commits/waits before prefill.
- [x] Integrated prefill batches decoder layer commands into one command buffer, but measured wall time is much larger than per-layer profile buckets.
- [x] Public decode already uses fused LM-head argmax instead of full f32 logits.
- [x] Decode matvec/MoE kernels still include scalar one-output/threadgroup reductions.

Implementation status:

- [x] Allocate one persistent GPU token-id buffer per engine slot.
- [x] Copy request input ids into the persistent token-id buffer before command encoding.
- [x] Replace standalone prompt assembly with one prefill-graph step: text-token gather plus image-feature blit into the prompt arena.
- [x] Encode prompt assembly and all prefill layers into one command buffer.
- [ ] Use MPSNDArray matrix multiplication for decoder prefill QKV, O projection, dense layer-0 MLP, shared experts, routed experts, and LM head prefill-size GEMMs.
- [x] Add f32 MPSNDArray matmul support for decoder prefill router logits and row-batched LM-head logits.
- [ ] Use custom tiled single-token kernels for decode QKV, O projection, dense layer-0 MLP, router, shared experts, routed experts, combine, and LM-head argmax.
- [x] Store no-repeat history in a reusable GPU sequence buffer.
- [x] Use the GPU-written token slot directly for generated-token embedding.
- [ ] Fuse decode epilogues into fixed kernels: projection bias, residual add, MoE combine, and final hidden copy.
- [x] Add detailed timing for token gather, image-span blit, prefill command encoding, prefill wait, decode layer kernels, and token readback.
- [ ] Validate fixed-prompt generated ids and `docs/test.png` text before deleting old kernels.
- [ ] Delete slow/unselected prompt, prefill, and decode production paths after the fused path passes parity and timing gates.

Acceptance target:

- [ ] Base prompt assembly drops from **4.04s** to **<=0.10s**.
- [ ] Base prefill drops from **1.48s** to **<=0.30s**.
- [ ] Base decode loop drops from **1.15s** to **<=0.50s** for the baseline generated length.
- [ ] Gundam prefill drops from **14.63s** to **<=2.0s**.

## 6. Python frontend cache cleanup

Impact: Python is not the 10x bottleneck, but fixed caching removes 200ms-scale
costs that matter after native latency drops.

Current finding:

- [x] `load_tokenizer()` constructs and validates a tokenizer on every prepare/decode call.
- [x] Local probing shows `load_tokenizer`, `prepare_image(base)`, and `decode_generated_ids()` each spend roughly 210ms mostly from tokenizer file load/validation.
- [x] Python preprocessing is **242ms** base and **339ms** gundam in the baseline profile.
- [x] Python decode is **~226ms** in both baseline cases.

Implementation status:

- [ ] Add an unbounded process-local tokenizer cache keyed by resolved tokenizer path.
- [ ] Cache tokenizer validation results by path and file identity.
- [ ] Cache decoded EOS text by tokenizer path.
- [ ] Use the cached tokenizer in `prepare_image()`, `prepare_pages()`, `prepare_text()`, and `decode_generated_ids()`.
- [ ] Add tests that verify cache reuse and deterministic output.

Acceptance target:

- [ ] Python preprocessing plus Python decode overhead for cached-tokenizer calls is **<=100ms** on `docs/test.png`.

## 7. MPS object policy for chosen large GEMMs

Impact: MPS is the chosen primitive for large dense GEMMs. The production path
must make MPS object creation cold-start-only.

Current finding:

- [x] `metal_run_mps_matmul_nt_f16()` creates and retains MPS descriptors, NDArrays, NSArrays, and transient-retain objects per call.
- [x] Some MPS descriptor/NDArray caching exists, but profiling still shows thousands of MPS objects in gundam.
- [x] Runtime env toggles such as `UOCR_METAL_DISABLE_MPS` preserve non-chosen production paths.

Implementation status:

- [x] Add MPS allocation/object counters and per-call timings.
- [x] Cache stable descriptors and long-lived weight NDArrays by shape/layout/transpose.
- [x] Cache workspace input/output NDArrays for every reusable vision and decoder prefill slice.
- [x] Prebuild all MPS descriptors and weight NDArrays during model map/binding validation.
- [x] Use stack/static small Objective-C arrays for fixed two-input matmul encoding where supported by the API boundary.
- [x] Remove destination zero-fill blits from MPS matmul calls after parity confirms all destinations are fully overwritten.
- [x] Make public Metal OCR require the chosen MPS GEMM path for large GEMMs.
- [x] Delete production MPS-disable toggles and threshold selection after the fixed MPS path passes timing gates.

Acceptance target:

- [ ] MPS descriptor and NDArray allocation counts are near zero inside repeated public `generate_prepared()` calls on a reused engine.

## 8. Request-sized lazy arenas

Impact: arenas should match the actual request high-water mark. This lowers
startup cost and makes memory reports meaningful.

Current finding:

- [x] Engine open allocates runtime arenas for configured maxima.
- [x] Default engines allocate for 4096 prompt tokens even when a base request is ~277 tokens.
- [x] Request-shaped vision workspace allocation already exists through `UOCR_METAL_SCRATCH_VISION`.

Implementation status:

- [x] Remove decoder arena allocation from `uocr_engine_open()`.
- [x] Grow decoder arenas once during request admission using the accepted request shape.
- [x] Keep engine hard caps for max batch, max prompt tokens, and max gen tokens.
- [x] Reject requests that exceed hard caps before allocating.
- [x] Record current capacity and high-water usage in memory reports.
- [x] Update admission errors with requested limits, configured limits, and current arena capacity.
- [x] Prove no arena growth occurs across prefill, decode, vision, and token-selection hot loops.

Acceptance target:

- [x] Base request non-model arena capacity is within 25% of the request estimate.

## 9. Quantization policy after fp16 10x

Impact: quantization is a compression milestone, not the primary 10x speed lever.
The chosen first quantized public path is q8 decoder with fp16 vision.

Current finding:

- [x] Public Metal generation currently accepts only fp16 `.uocr` models.
- [x] q8 and q4 converter/kernels exist, but are not integrated into public generation.
- [x] MoE routed experts dominate model bytes.

Implementation status:

- [ ] Keep fp16 as the required profile for the 10x latency milestone.
- [ ] Implement the first quantized public profile as fp16 vision plus q8 decoder.
- [ ] Keep router, norms, biases, position tables, image newline, and view separator fp16 in the q8 decoder profile.
- [ ] Add q8 decoder bindings using direct cached slices.
- [ ] Add q8 kernels for token embedding, attention projections, layer-0 MLP, shared experts, routed experts, and LM-head argmax.
- [ ] Measure q8 first-token latency, decode tokens/sec, and peak memory against the optimized fp16 engine.
- [ ] Start q4 routed experts only after q8 decoder passes parity and timing gates.

## 10. Production-path cleanup

Impact: fastest-engine work will regress if old slow paths remain selectable in
public dispatch.

Current finding:

- [x] The Metal file contains many diagnostic CPU-pointer APIs that allocate buffers, copy to GPU, wait, and copy back.
- [x] Some helper paths are necessary for parity tests but too slow for production dispatch.
- [x] Runtime toggles and thresholds preserve non-chosen paths.

Implementation status:

- [ ] Mark diagnostic-only APIs with `diagnostic` in the function name and comment header.
- [ ] Keep diagnostic APIs out of `uocr_generate_prepared()` and public OCR dispatch.
- [ ] Delete unused slow production helpers as optimized replacements land.
- [ ] Remove runtime fallback toggles after the chosen implementation passes timing gates.
- [ ] Keep tests on the confirmed path; parity-only behavior must use offline fixtures.
- [ ] Add grep-based tests that fail when public dispatch calls diagnostic helpers.

## 11. Execution order

Implementation status:

- [x] Land profiling/timing/memory instrumentation and capture baseline profiles.
- [ ] Implement same-shape batched vision command graphs.
- [ ] Convert production vision helpers to buffer-slice APIs and GPU-private workspace.
- [ ] Rewrite SAM attention and convolution kernels.
- [ ] Cache MPS descriptors/NDArrays and lock large GEMMs to the fixed MPS path.
- [x] Fuse prompt assembly into the prefill command graph.
- [ ] Replace decode single-token matvec/MoE kernels with custom tiled kernels.
- [ ] Add Python tokenizer/decode caching.
- [x] Implement request-sized lazy arenas.
- [ ] Delete unselected fallback/duplicate production code after each stage passes gates.
- [ ] Start q8 decoder compression work after the fp16 10x milestone.

## 12. Acceptance checklist for the 10x milestone

Implementation status:

- [ ] Base `docs/test.png` total latency is <=10% of baseline on the same machine/build/model.
- [ ] Gundam `docs/test.png` total latency is <=10% of baseline on the same machine/build/model.
- [ ] Base vision is **<=1.1s**.
- [ ] Gundam vision is **<=12s**.
- [ ] Base prompt assembly is **<=0.10s**.
- [ ] Base prefill is **<=0.30s**.
- [ ] Base decode loop is **<=0.50s** for the baseline generated length.
- [ ] Gundam prefill is **<=2.0s**.
- [ ] Generated ids/text match accepted parity thresholds for fixed fixtures.
- [ ] Metal command-buffer wait count and allocation/object counters are materially reduced.
- [ ] Memory reports remain accurate and include workspace high-water marks.
- [ ] Old slow production paths and toggles are removed from public dispatch.
