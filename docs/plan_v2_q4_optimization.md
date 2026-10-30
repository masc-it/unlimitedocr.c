# V2 mixed-Q4 optimization plan

Status: `todo` (not done)

Checkboxes: `- [x] done` means completed and verified; `- [ ] todo` means not
done. Research and measurements are complete for the initial investigation;
none of the proposed kernel or scheduling changes have been implemented.

## Goal and scope

Improve speed and/or peak memory for native Unlimited-OCR-2 inference, with
**mixed Q4 as the primary quantization profile**. Measure first-token latency,
steady decode cost, complete-request latency, and memory separately.

The initial target is the shipped single-page base/gundam path on Apple Silicon.
Multi-page inference and bounded-memory continuation remain required follow-up
work in [the v2 implementation plan](plan_unlimitedocr2.md). This plan does not
claim those features are implemented.

Constraints:

- Preserve v1 behavior, existing model files and caches, and the current mixed-Q4
  precision policy. Do not introduce persistent dequantized weight copies.
- Preserve page-aware RoPE, top-16 entmax routing, causal/local overlap rules,
  and page/EOS guards. Kernel optimization must not silently change the model.
- Keep memory admission, failure cleanup, and engine reuse correct. Allocation
  or compilation failures must remain explicit.
- Keep the macOS 15+ baseline. M5-only acceleration is not a solution for the
  measured M1 Pro target.
- Each phase must leave a working public OCR path. Land measured improvements;
  record and reject candidates that regress the intended workload.

## Current state and completed investigation

- [x] done: Audit existing Q4, vision, DASH, and generation implementations.
- [x] done: Review earlier tuning results in `howto_quant.md` and `plan_q4.md`.
- [x] done: Run warmed v2 mixed-Q4 base/gundam measurements and diagnostic
  checkpoint profiles on an Apple M1 Pro with 16 GB RAM.
- [x] done: Compare relevant Metal/MLX implementations and Apple guidance online.
- [x] done: Check that repeated baseline and detailed-profile runs produce the
  same 65-token outputs on the invoice fixture.
- [x] done: Verify the four-local-view chunk configuration on that fixture.
- [x] done: Remove disposable investigation probes without modifying production
  code or the persistent synthetic stress dataset.

Already implemented: vectorized group-half-split Q4 unpacking, fused
quantization/compute, fused activation/residual/combine paths, expert-bucketed
prefill, simdgroup-MMA GEMMs, fused partial argmax, GPU-resident embeddings,
and incremental refresh of DASH's mutable block summaries. Do not propose
these again as new work.

### Recorded baseline

Code audited: release `v0.5.0`, commit `03eda77`.

Input: `data.tmp/synthetic-ocr/q4-stress/images/invoice.png`.
Model: pinned v2 mixed-Q4 cache, approximately 1.846 GB of model views.
Execution: Release profiling-enabled native build, prepared requests through
`Engine.generate_prepared`, model already loaded, one warmup request, then two
repeats each at output caps 1 and 65. Image preparation and model loading were
excluded. The normal runs used profiling detail level 0.

| Measurement | Base | Gundam |
|---|---:|---:|
| Views | 1 | 13 |
| Prompt tokens | 279 | 1,519 |
| Vision stage | 1.12 s | 5.33 s |
| Request capped at one output token | 1.36 s | 6.13 s |
| Request capped at 65 output tokens | 2.03 s | 6.93 s |
| Incremental decode cost, `(T65 - T1) / 64` | 10.45 ms/token | 12.56 ms/token |
| Tracked peak memory | 3.33 GB | 4.14 GB |

GB means decimal bytes. These are token-capped measurements on one fixture,
not complete-document throughput or a general accuracy qualification.

Normal profiles showed three additional command buffers/waits and 206 command
encoders per additional generated token. V2 currently uses separate submissions
for embedding, decoder forward, and argmax; v1 already has an eight-token GPU
chunk path.

Profiling limits:

- Ordinary per-operation timers often measure CPU encoding rather than GPU
  execution. V2 also labels decoded-token forwards as `metal.prefill.*` because
  it reuses the prefill orchestration function.
- Detail level 3 inserts synchronization. Base's 65-token request rose from
  about 2.03 s to 3.21 s. Detailed gundam profiles also dropped events at the
  report capacity. Do not treat those reports as precise GPU time shares.
- Tracked memory is not total process memory. Per-layer SAM relative-logit
  buffers are profile-counted but bypass the usual native memory reservation
  path; MPS/driver allocations also need separate observation.
- Raw investigation profiles were disposable and removed. Recreate a baseline
  before implementation; the table above is the retained evidence.

### Measured configuration opportunity

With `UOCR_VISION_LOCAL_MAX_VIEWS_PER_CHUNK=4`, the gundam fixture's tracked peak
fell from **4.14 GB to 3.34 GB**. The two 65-token runs remained around **6.93 s**,
with identical token IDs to the default chunk-size runs.

```bash
UOCR_VISION_LOCAL_MAX_VIEWS_PER_CHUNK=4 uv run your_script.py
```

This setting was exercised only in probe processes. The default was not changed,
and broader qualification remains `todo` in phase 2.

## Priorities and relevant files

| Phase | Target | Main benefit | Status |
|---|---|---|---|
| 1 | V2 submission/encoder scheduling | Decode speed | todo |
| 2 | Vision scratch lifetimes and chunk sizing | Memory | todo |
| 3 | SAM global matrix-tiled attention | Memory and first-token latency | todo |
| 4 | SAM window matrix-tiled attention | First-token latency | todo |
| 5 | DASH attention parallelism | Decode speed; prefill opportunity | todo |
| 6 | Occupancy-aware routed-MoE prefill | Prefill speed | todo |
| 7 | Q4 vision GEMM staging | Vision speed | todo |
| 8 | Q4 LM-head and decode GEMVs | Decode speed | todo |
| 9 | DASH router/projection specialization | Decode and future longer windows | todo |

Files:

- `src/backend/metal/uocr_metal.m`: command ownership, dispatch, workspace
  lifetimes, MPS projections, and model integration.
- `src/backend/metal/kernels/sam_attention.metal`: SAM global/window attention.
- `src/backend/metal/kernels/dash.metal`: summaries, routing, and sparse attention.
- `src/backend/metal/kernels/gemm_q4.metal`: Q4 staging, vision and expert GEMMs.
- `src/backend/metal/kernels/gemm_q8.metal`: shared MMA/bucketing infrastructure.
- `src/backend/metal/kernels/decode_gemv_q4.metal`: Q4 decode projections.
- `src/backend/metal/kernels/lm_head_q4.metal`: fused Q4 partial argmax.
- `src/runtime/uocr_memory.c`: admission estimates and memory accounting.
- `src/runtime/uocr_vision.c`: local-view chunk scheduling.
- `src/runtime/uocr_page_state.*` and `uocr_logits_processor.*`: page state and
  generation guards that GPU chunking must preserve.
- `tools/gen_metal.py` and `CMakeLists.txt`: kernel/build integration as needed.

## Shared measurement and acceptance gates

- [ ] todo: Recreate the normal baseline on the same model and fixture; record
  commit, hardware, build flags, view counts, prompt/output counts, and warmup.
- [ ] todo: Separate prefill from decode attribution. Use GPU timestamps,
  counter sampling, or isolated GPU probes for kernel measurements; use normal
  non-checkpoint runs for end-to-end acceptance.
- [ ] todo: Compare candidates with the same inputs and output lengths, using
  repeated/interleaved runs. Include natural EOS runs as a separate quality
  check so changed output lengths cannot masquerade as speedups.
- [ ] todo: Record tracked allocations, Metal allocation usage where available,
  and process memory. Do not sum independent category high-water marks or claim
  tracked peak is RSS.
- [ ] todo: Validate numerical behavior against the current native path and
  small reference calculations, then exercise real v2 Q4 base/gundam OCR,
  strict parsing, v1 regression, and v2 Q8 dispatch compatibility.
- [ ] todo: Exercise budget rejection and reuse after failure/truncation for
  changes affecting allocation or generation state.
- [ ] todo: Verify runtime-compiled and precompiled Metal paths when shaders
  change. Run the relevant existing checks at each implementation checkpoint.

Use disposable probes under `data.tmp/probes/v2-q4-<phase>/`; record findings
and commands here before removing them. Do not add permanent tests unless
explicitly requested. Keep unrun checks marked `todo` and state blockers.

## Phase 1: V2 decode with fewer submissions

Outcome: public v2 OCR uses one submission per token before attempting larger
GPU-resident decode chunks.

- [ ] todo: Move command-buffer ownership to generation orchestration so
  embedding, forward, and argmax can be encoded in one batch per token.
- [ ] todo: Consolidate compatible compute encoders with correct dependencies.
- [ ] todo: Keep CPU page-state updates at safe boundaries and preserve explicit
  abort/cleanup behavior if encoding or GPU execution fails.
- [ ] todo: After the single-token checkpoint passes, adapt v1-style GPU chunks
  to v2, including GPU page positions, page/EOS guards, output limits, and
  correct treatment of work after EOS. Do not copy v1 cache semantics.

QA checkpoint:

- [ ] todo: Confirm the measured three-waits-per-token pattern is removed;
  report encoder counts and normal-run decode latency before/after.
- [ ] todo: Check first PAGE, EOS, short token caps, page boundaries in low-level
  probes, and engine reuse after a failed request.
- [ ] todo: Confirm chunked output/state matches the unchunked path, including
  stopping inside a chunk. Keep the one-token path working throughout rollout.

QA evidence: not run. Reference: [3].

## Phase 2: Reusable vision scratch and qualified chunk sizing

Outcome: sequential SAM layers reuse temporary relative-position storage;
chunk-size choices have measured speed/memory behavior.

- [ ] todo: Replace per-layer H/W relative-logit allocations retained until
  batch completion with a reusable, capacity-checked workspace pair.
- [ ] todo: Account for this storage and audit other untracked dynamic vision
  allocations. Extend the existing planner instead of adding a new allocator
  unless a clear need is demonstrated.
- [ ] todo: Alias additional workspace regions only where GPU lifetimes do not
  overlap; preserve required ordering/fences.
- [ ] todo: Qualify four-view local chunks across representative crop counts
  and document the tradeoff. Change the default only if evidence supports it.

QA checkpoint:

- [ ] todo: Compare actual allocation counts/peaks as well as the native report;
  verify that relative-logit storage no longer grows with encoded layer count.
- [ ] todo: Compare chunk sizes 4 and 8 on base and several gundam shapes,
  including a maximum-crop case, with complete OCR and parsing checks.
- [ ] todo: Verify budget rejection, allocation failure cleanup, and repeat
  requests without retained-memory growth.

QA evidence: the initial chunk-size result is recorded above; workspace reuse
and broader chunk qualification have not run. Reference: [7].

## Phase 3: SAM global attention without dense score matrices

Outcome: the global SAM attention path uses tiled QK/softmax/PV and the native
memory estimate reflects the new working set.

Current FP32 logits plus FP16 probabilities for one 1024-pixel global view cost
`12 * 4096 * 4096 * (4 + 2) = 1,207,959,552` bytes. This is removable buffer
capacity, not a promised net memory saving before replacement scratch is sized.

- [ ] todo: Implement matrix-tiled attention for SAM head dimension 64 using
  `simdgroup_matrix`, FP32 accumulation, and online softmax statistics.
- [ ] todo: Preserve query-dependent decomposed H/W relative-position terms;
  use smaller precomputed relative logits or tile-local evaluation as measured.
- [ ] todo: Integrate the replacement into global-view and local-crop global
  attention, then remove obsolete dense logits/probabilities from the planner.

QA checkpoint:

- [ ] todo: Compare attention output to the current MPS path on real tensors
  and adversarial softmax inputs, with explicit numerical tolerances.
- [ ] todo: Measure GPU latency and total memory for relevant 1024/640 view
  shapes; confirm removal of the quadratic buffers in allocation evidence.
- [ ] todo: Pass real base/gundam OCR and existing v1 checks before switching
  the production dispatch.

QA evidence: not run. References: [1], [2].

## Phase 4: Matrix-tiled SAM window attention

Outcome: SAM window attention retains current semantics while using matrix MMA
for QK and PV instead of serial per-key SIMD dot products.

Window attention already stages K/V tiles and uses online softmax. The new work
is matrix-core utilization and reuse across multiple queries.

- [ ] todo: Specialize matrix tiles for head dimension 64 and 14x14 windows,
  preserving relative-position terms and existing image-padding semantics.
- [ ] todo: Reuse the global attention implementation where useful without
  forcing both workloads into the same tile configuration.
- [ ] todo: Evaluate reuse for CLIP attention only after the SAM checkpoint;
  keep it only if the different sequence lengths benefit.

QA checkpoint:

- [ ] todo: Verify padded/non-padded window behavior and tile-edge masking.
- [ ] todo: Compare window kernel time and full vision latency in base/gundam;
  confirm improvements survive with detailed checkpoints disabled.
- [ ] todo: Verify OCR, memory accounting, and both shader build paths.

QA evidence: not run. References: [1], [2].

## Phase 5: Parallel DASH sparse attention

Outcome: decode attention has enough parallel work while retaining sparse
memory usage and the exact selection/bias contract.

The current kernel assigns one SIMD group per query/head: only 10 SIMD groups
for decode, each visiting up to 512 selected/local keys serially.

- [ ] todo: Benchmark multiple SIMD groups per head and small split-K variants.
  Merge partial maxima, denominators, and weighted outputs stably.
- [ ] todo: Preserve remote biases, causal bounds, empty partitions, and
  single inclusion of overlapping local tokens.
- [ ] todo: Separately investigate matrix-tiled sparse prefill where selections
  allow useful reuse; do not materialize a dense attention matrix.
- [ ] todo: Size/account for any reduction scratch and keep decode/prefill
  dispatch based on measured workload shape.

QA checkpoint:

- [ ] todo: Compare against current attention with fixed selections, overlap,
  empty remote sets, partial blocks, and extreme logits.
- [ ] todo: Measure kernel occupancy/latency and full decode cost. Reject split
  counts whose reduction overhead outweighs additional parallelism.
- [ ] todo: Confirm unchanged page/EOS behavior and full OCR quality checks.

QA evidence: not run. References: [2], [4].

## Phase 6: Occupancy-aware routed-MoE prefill

Outcome: small expert buckets avoid unnecessary 64-row padding while larger
buckets retain efficient tiled execution.

The base fixture averages `279 * 6 / 64 = 26.2` selected rows per expert;
actual routing distribution must be measured before choosing a dispatch policy.
Expert sorting/bucketing and fused Q4 MMA are already implemented.

- [ ] todo: Measure per-expert row-count distributions on representative prompts.
- [ ] todo: Benchmark 16/32-row variants alongside the existing 64-row path.
- [ ] todo: Choose bucket/tile scheduling on GPU without routing readback.
- [ ] todo: Investigate compact dispatch only if empty/partial tile overhead
  is material; preserve fused gate/up and deterministic expert combination.

QA checkpoint:

- [ ] todo: Exercise skewed routing, empty experts, and partial tiles against
  the current implementation using the same routes and quantized weights.
- [ ] todo: Measure complete MoE prefill including bucket/combine overhead,
  then verify end-to-end base/gundam latency and output quality.

QA evidence: not run. References: [5], [6]. External performance figures are
from different models/hardware and are not expected speedups for this engine.

## Phase 7: Q4-specific vision GEMM staging

Outcome: faster vision GEMMs with unchanged group-64 half-split storage and no
persistent dequantized copies.

- [ ] todo: Benchmark vectorized packed loads, cheaper nibble conversion, and
  packed-byte/scale reuse across the two halves of a quantization group.
- [ ] todo: Evaluate register prefetching or bounded software pipelining while
  measuring register and threadgroup-memory pressure.
- [ ] todo: Tune large vision GEMMs separately from small expert buckets.

Prior evidence in `howto_quant.md`: Q8 `BN=64` caused occupancy collapse and
`BK=64` was slightly slower. Revisit larger tiles only with a concrete
Q4-specific rationale; bigger tiles are not an optimization by themselves.

QA checkpoint:

- [ ] todo: Compare numerical results on actual quantized tensors, preserving
  SAM erf-GELU and CLIP QuickGELU behavior and all fused epilogues.
- [ ] todo: Report shape-specific GPU throughput and normal-run vision latency;
  reject variants that merely shift cost into staging or extra allocations.
- [ ] todo: Verify base/gundam OCR and runtime/precompiled shader parity.

QA evidence: not run. Reference: [5].

## Phase 8: Q4 LM-head and decode GEMV tuning

Outcome: measured decode improvements without full-logit or dequantized-weight
buffers and without changing model-file layout.

The LM head reads approximately 88 MB of weights/scales per token and currently
uses 1,024-thread groups. Existing unpacking and argmax designs were already
benchmarked during the earlier Q4 work.

- [ ] todo: Benchmark smaller groups with multiple output rows per SIMD group
  and activation/preprocessing reuse across rows.
- [ ] todo: Investigate cheaper quantized dot-product instruction sequences
  while retaining the current packing and scale semantics.
- [ ] todo: Preserve fused partial argmax and update partial-buffer sizing
  coherently if the logical reduction geometry changes.
- [ ] todo: Apply improvements to expert/shared/dense decode GEMVs only where
  their actual dimensions show a benefit.

QA checkpoint:

- [ ] todo: Verify tail dimensions, tied logits, invalid-token masks, and page/EOS
  filtering before argmax; compare against the existing quantized path.
- [ ] todo: Measure effective bandwidth and instruction/occupancy limits, then
  confirm an end-to-end decode improvement rather than a microbenchmark-only win.

QA evidence: not run. Reference: [5].

## Phase 9: DASH router/projection specialization

Outcome: lower router overhead on decode and a qualified path for larger block
counts, without changing the sparse routing algorithm.

- [ ] todo: Specialize the two small per-layer router Q/K projections for
  single-token decode; evaluate normalization/RoPE fusion with preserved rounding.
- [ ] todo: Maintain incremental tail-summary accumulator state rather than
  recomputing the partial block, with explicit ownership/reset at boundaries.
- [ ] todo: For larger contexts, benchmark parallel block scoring and top-k
  merging against the current serial candidate scan and lane-zero insertion.
- [ ] todo: Preserve top-16 entmax, finite masked sentinels, stable tie handling,
  FP32 norm semantics, sanitization order, and page-bucket eligibility. Do not
  substitute the upstream full-candidate routing approximation.

QA checkpoint:

- [ ] todo: Compare router projections, summaries, selected blocks, and biases
  against the current native path and established reference calculations.
- [ ] todo: Exercise partial/full 32-token blocks, masked candidates, ties,
  extreme values, and reset/reuse. Keep untested multi-page behavior explicit.
- [ ] todo: Measure current single-page benefit separately from synthetic
  longer-context results; do not advertise unimplemented document scheduling.

QA evidence: not run. See [v2 research notes](unlimitedocr2_research.md) for the
routing oracle and continuation distinctions.

## Deferred work and guardrails

- [ ] todo: Reassess KV quantization after vision memory and continuation work.
  It changes activation precision and requires separate quality qualification.
- [ ] todo: Reassess long-window cache allocation after the document scheduler
  exists. Sparse attention reads do not permit arbitrary eviction of older K/V;
  the 128-token local span is not a retention limit.

Q2, speculative decoding, M5-only paths, and a new allocator/framework are outside
this initial plan. Upstream full-model numerical parity remains unverified;
local optimization QA must not be presented as having established it.

## Online references

Reviewed during the initial investigation; external code on `main` may change.
Check hardware assumptions, precision/layout differences, and licenses before
adapting an implementation.

1. [Metal FlashAttention implementation and architecture notes](https://github.com/philipturner/metal-flash-attention).
2. [MLX Steel matrix attention](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h).
3. [Apple command-buffer best practices](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html).
4. [MLX vector and split-K attention](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h).
5. [MLX quantized GEMV/GEMM and grouped kernels](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h).
6. [MLX small-group MoE performance discussion](https://github.com/ml-explore/mlx/issues/4246).
7. [Apple resource reuse, aliasing, and synchronization](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/ResourceHeaps/ResourceHeaps.html).
