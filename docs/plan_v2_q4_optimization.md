# V2 mixed-Q4 optimization plan

Status: `todo` (not done)

Checkboxes: `- [x] done` means completed and verified; `- [ ] todo` means not
done. Initial research is complete. Phase 1's single-token submission and
encoder-consolidation checkpoint is implemented and verified. GPU decode
chunking is deferred after an unfavorable prototype tradeoff. Phase 2's shared
SAM relative-logit storage, shared local/global backing storage, and batch-abort
ownership fixes are implemented. Four-crop batches are qualified on the measured
M1 Pro. Phase 3's bounded-query MPS checkpoint is implemented and verified;
custom fused global online attention is deferred after slower prototypes.
Phase 4's v2 SAM window MMA kernel and phase 5's order-preserving DASH decode
kernel are implemented and verified. Phase 6's v2 Q4 M16 routed-prefill
checkpoint is also implemented and verified. Phase 7's vision-GEMM prototypes
were measured; production rollout is deferred because complete-request gains
were small relative to variation and added dispatch complexity. Phase 8's
Q4 LM-head scheduling/reduction and paired gate/up checkpoint is implemented
and verified. Phase 9's projection, fused router norm/RoPE, and SIMD routing
checkpoint is implemented and verified, with the Q8 caveat below. Incremental
tail summaries, CLIP reuse, matrix-tiled sparse prefill, and remaining
staging/cached-buffer accounting are not done.

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
encoders per additional generated token. That baseline used separate submissions
for embedding, decoder forward, and argmax; v1 already had an eight-token GPU
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
| 1 | V2 submission/encoder scheduling | Decode speed | single-token checkpoint done; GPU chunking deferred |
| 2 | Vision scratch lifetimes and chunk sizing | Memory | lifetimes/chunk qualification done; staging accounting remains todo |
| 3 | Bounded SAM global attention | Memory and first-token latency | bounded MPS checkpoint done; fused online kernel deferred |
| 4 | SAM window matrix-tiled attention | First-token latency | v2 SAM checkpoint done; CLIP reuse remains todo |
| 5 | DASH attention parallelism | Decode speed; prefill opportunity | decode checkpoint done; matrix-tiled sparse prefill remains todo |
| 6 | Occupancy-aware routed-MoE prefill | Prefill speed | v2 Q4 M16 checkpoint done; mixed-width/compact dispatch not justified |
| 7 | Q4 vision GEMM staging | Vision speed | investigated; production deferred after small whole-request gains |
| 8 | Q4 LM-head and decode GEMVs | Decode speed | LM-head/paired gate-up checkpoint done; down GEMVs unchanged |
| 9 | DASH router/projection specialization | Decode and future longer windows | Projection/norm/routing checkpoint done; incremental summaries todo |

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

- [x] done: Recreate the normal baseline on the same model and fixture; record
  commit, hardware, build flags, view counts, prompt/output counts, and warmup.
- [ ] todo: Separate prefill from decode attribution. Use GPU timestamps,
  counter sampling, or isolated GPU probes for kernel measurements; use normal
  non-checkpoint runs for end-to-end acceptance.
- [x] done: Compare candidates with the same inputs and output lengths, using
  repeated/interleaved runs. Include natural EOS runs as a separate quality
  check so changed output lengths cannot masquerade as speedups.
- [x] done: Record tracked allocations, Metal allocation usage where available,
  and process memory. Do not sum independent category high-water marks or claim
  tracked peak is RSS.
- [ ] todo: Validate numerical behavior against the current native path and
  small reference calculations, then exercise real v2 Q4 base/gundam OCR,
  strict parsing, v1 regression, and v2 Q8 dispatch compatibility.
- [x] done: Exercise budget rejection and reuse after failure/truncation for
  changes affecting allocation or generation state.
- [ ] todo: Verify runtime-compiled and precompiled Metal paths when shaders
  change. Run the relevant existing checks at each implementation checkpoint.

Use disposable probes under `data.tmp/probes/v2-q4-<phase>/`; record findings
and commands here before removing them. Do not add permanent tests unless
explicitly requested. Keep unrun checks marked `todo` and state blockers.

## Phase 1: V2 decode with fewer submissions

Outcome: public v2 OCR uses one submission per token before attempting larger
GPU-resident decode chunks.

- [x] done: Move command-buffer ownership to generation orchestration so
  embedding, forward, and argmax can be encoded in one batch per token.
- [x] done: Consolidate compatible compute encoders with correct dependencies.
- [x] done: Keep CPU page-state updates at safe boundaries and preserve explicit
  abort/cleanup behavior if encoding or GPU execution fails.
- [ ] todo: After the single-token checkpoint passes, adapt v1-style GPU chunks
  to v2, including GPU page positions, page/EOS guards, output limits, and
  correct treatment of work after EOS. Do not copy v1 cache semantics.
  **Deferred:** the oracle prototype below found a small long-output benefit and
  a short-output regression. No production GPU chunking has been implemented.

QA checkpoint:

- [x] done: Confirm the measured three-waits-per-token pattern is removed;
  report encoder counts and normal-run decode latency before/after.
- [x] done: Check first PAGE, EOS, short token caps, page boundaries in low-level
  probes, and engine reuse after a failed request.
- [ ] todo: Confirm chunked output/state matches the unchunked path, including
  stopping inside a chunk. Keep the one-token path working throughout rollout.

### Single-token checkpoint evidence (2026-10-30)

Implementation in `src/backend/metal/uocr_metal.m`:

- Decoder forward is encoder-only. V1 and the diagnostic host-forward API keep
  their explicit synchronous submission boundaries.
- V2 generation prepares CPU page positions while idle, then batches embedding,
  forward, and filtered argmax. Host token readback follows successful completion.
- Compute-encoder reuse is opt-in during v2 generation only. Former encoder
  boundaries insert buffer memory barriers. Blits, submission, and abort close
  the shared encoder; both success and failure clear the opt-in state.
- V2 selection reads the recorded final hidden segment directly, removing the
  unnecessary segment-zero blit. V1 retains that copy for its chunk decoder.
- Normal combined waits have separate `metal.v2.prefill.command_wait` and
  `metal.v2.decode.command_wait` events. They are host wait durations, not isolated
  GPU kernel timings; legacy per-operation encoding labels remain unchanged.

The initial submission-only candidate preserved outputs and reduced waits but
showed no reliable speed improvement. Encoder consolidation supplied the measured
win; the submission-only candidate was not landed separately.

Normal-mode incremental counts per decoded token:

| Counter | Baseline | Checkpoint |
|---|---:|---:|
| Command buffers / waits | 3 / 3 | 1 / 1 |
| Command encoders | 206 | 1 |

Same M1 Pro, pinned mixed-Q4 model and invoice as the initial investigation.
The baseline was saved from the Release build at `697ff33`, before code changes.
A disposable A/B probe loaded that saved library and the candidate explicitly,
warmed both engines, and interleaved old/new runs. Three pairs per version used
output caps 1 and 257, with identical 257-token outputs. Mean incremental decode
cost fell from **10.85 to 7.44 ms/token**, about **31% lower latency**. Mean total
257-token request time fell from **4.16 to 3.32 s**. These measurements used
profiling-enabled Release builds with detailed checkpoints disabled.

Separate EOS-complete comparisons, with identical token IDs and strict parsing:

| Path | Output tokens | Baseline | Checkpoint |
|---|---:|---:|---:|
| V2 Q4 base | 385 | 5.74 s | 4.36 s |
| V2 Q4 gundam | 385 | 11.00 s | 9.84 s |
| V2 Q8 base | 385 | 6.64 s | 4.57 s |

These complete-request figures are single comparisons, not broad throughput
claims. V1 Q4 base also retained all 381 output IDs and unchanged submission/
encoder counts. Repeated caps 1/65 checked truncation and subsequent reuse.
Tracked complete-request peaks were unchanged: **3,333,935,616 bytes** base and
**4,139,373,056 bytes** gundam. V2 live tracked bytes returned to model-only after
each request. Process RSS and driver/MPS allocation peaks were not measured.

A low-level probe exercised a two-page host-embedding forward, generated page
transitions, and raw argmax; this is not multi-page image generation. A budget
of required decoder bytes minus one rejected before allocation. Three injected
readback-workspace failures occurred after forward encoding with the shared
encoder open. Each aborted with no active batch/encoder, cleared transient
retains and the reuse flag, then successfully reused the same context after
request release. Tracked request bytes returned to zero. The probe also passed
with `MTL_DEBUG_LAYER=1`. Actual GPU execution faults were not injected.

Public `UnlimitedOCR(model="v2", quant="q4", download=False)` matched the baseline
text and parsed under Metal API validation. A separate Release build with
profiling OFF and runtime Metal compilation OFF produced the same 385 tokens
using the precompiled metallib (4.37 s). No shaders or model formats changed;
therefore runtime shader compilation was not separately requalified. Decoder
detail level 3 also completed after encoder consolidation and retained the
baseline first-token output; its added checkpoint waits remain intentional.

Commands/checks used:

```bash
cmake --build build/release --parallel 4
UOCR_RUN_LARGE_TESTS=0 UOCR_RUN_PERF_TESTS=0 HF_HUB_OFFLINE=1 \
  uv run --extra dev python -m pytest tests -q
ctest --test-dir build/release --output-on-failure
```

Results: **105 Python checks passed, 7 opt-in checks skipped**, with three expected
invalid-limit warmup warnings; **13 native checks passed**. Disposable scripts,
saved libraries, and profiles lived under `data.tmp/probes/v2-q4-submit/` and were
removed after recording this evidence. The dedicated profiling-OFF build was
also removed. Reference: [3].

### GPU chunking experiment and deferral (2026-10-30)

- [x] done: Measure an optimistic eight-token chunk prototype before adding
  production GPU page/EOS-state machinery.
- [x] done: Compare valid output IDs and include a short natural-EOS case.
- [x] done: Record the tradeoff and retain single-token production decoding.

A disposable native probe used real image prefill and mixed-Q4 decoder weights.
It compared one-token submissions with eight-token submissions in the same
process. The chunk path chained token embeddings from GPU selection results,
but supplied future page coordinates and selection filters from a verified
reference trace. This removes the cost of the required GPU state kernels and
is therefore an optimistic scheduling experiment, not a shippable decoder.

One warmup per variant was discarded, followed by three interleaved measurements
per variant. Times below cover decode only; model load, image preparation and
vision/prefill are excluded. GPU durations were also collected from completed
command buffers' `GPUStartTime`/`GPUEndTime`.

| Fixture | Useful decode steps | One-token wall time | Eight-token wall time | Change |
|---|---:|---:|---:|---:|
| Invoice base | 256 | 1,959.54 ms | 1,890.88 ms | 3.5% lower |
| Invoice gundam | 256 | 2,395.38 ms | 2,327.32 ms | 2.8% lower |
| Blank base, natural EOS | 18 | 125.32 ms | 158.44 ms | 26.4% higher |

For the invoice, waits/encoders fell from 256 to 32. Most time remained GPU work:
roughly 1.76-1.79 s base and 2.20-2.24 s gundam. The blank page emits 19 tokens
including the initial prefill selection; chunking executes 24 decode steps for
18 useful steps. Dead-step selection was forced to EOS, but decoder forwards
still ran. Valid output IDs matched. Repeated gundam prefill produced small
score differences, so the probe checked scores within 0.01 rather than claiming
bitwise logit equality.

Decision: **defer production chunking**. The modest long-output gain does not
yet justify GPU page-state/guard machinery, exact retained-state handling after
EOS, and short-output overrun. This does not rule out a later EOS-predicated or
pipelined implementation. Revisit after larger compute wins or a concrete way
to avoid dead forwards. The prototype did not implement retained-state rollback,
and must not be treated as passing the chunked-state QA gate above.

The next practical checkpoint was Phase 2's shared relative-logit storage.
The oracle source/library and traces were disposable and removed after recording
these results; no chunking code or scheduling default was changed.

## Phase 2: Reusable vision scratch and qualified chunk sizing

Outcome: sequential SAM layers reuse temporary relative-position storage;
chunk-size choices have measured speed/memory behavior.

- [x] done: Replace per-layer H/W relative-logit allocations retained until
  batch completion with a reusable, capacity-checked workspace pair.
- [x] done: Cover relative-logit storage with the existing accounted workspace;
  no additional allocation or larger admission estimate is needed.
- [x] done: Trace intermittent Metal allocation peaks and audit explicit
  allocations in the public vision path.
- [x] done: Reuse one backing buffer for local/global workspace layouts after
  chunk completion; invalidate cached MPS views before changing offsets.
- [x] done: Fix batch-abort ownership for C/ctypes callers without an outer
  autorelease pool.
- [x] done: Qualify four-view local chunks across representative crop counts
  on the M1 Pro and document the tradeoff. Keep the shared default unchanged.
- [ ] todo: Reuse/account pixel-upload and patch-extraction buffers and account
  cached position-conversion buffers. Keep MPS/driver usage separately visible;
  the native report is still not a complete device-memory cap.

QA checkpoint:

- [x] done: Compare actual allocation counts/peaks as well as the native report;
  verify that relative-logit storage no longer grows with encoded layer count.
- [x] done: Compare chunk sizes 4 and 8 on base and several gundam shapes,
  including a maximum-crop case, with complete OCR and parsing checks.
- [x] done: Verify budget rejection, allocation failure cleanup, and repeat
  requests without retained-memory growth.

### Shared relative-logit checkpoint evidence (2026-10-30)

`metal_context_sam_rel_pos_attention_batch_f16_to_slice()` now borrows the
existing `sam_global_rel_h_f32` / `sam_global_rel_w_f32` workspace slices. Global
and window attention run sequentially. For the supported patch grids G=40/64,
the global capacity per axis (`B * 12 * G^3 * 4` bytes) exceeds the padded-window
requirement (`B * 12 * ceil(G/14)^2 * 14^3 * 4`). Both slice capacities are checked
before encoding. A buffer memory barrier separates relative-logit production
from consumption. No per-layer allocation, retain, or release remains here.

Before the change, a disposable allocation-count probe failed its target of no
per-layer relative-logit buffers. Afterward, complete requests retained identical
output IDs and strict parse results: v2 Q4 base, 13-view gundam, maximum 17-view
gundam, v2 Q8 base, and v1 Q4 base. The maximum-crop fixture placed the invoice
on a 2048x2048 white canvas. Each case was warmed and repeated twice.

| Request | Native buffer allocations before | After |
|---|---:|---:|
| V2 Q4/Q8 base | 30 | 14 |
| V2 Q4 gundam, 13 or 17 views | 67 | 19 |
| V1 Q4 base | 28 | 12 |

This removes 16 allocations in base and 48 across the three gundam chunks. The
old buffers retained across eight SAM window layers totaled 52,684,800 bytes
for a global-view batch and 151,732,224 bytes for an eight-local-view batch.
Those bytes now reuse storage that was already allocated and admitted.

A 1 ms sampler read `MTLDevice.currentAllocatedSize`; process peak RSS was also
recorded in separate before/after processes. Representative repeated Metal
samples were:

| V2 Q4 request | Before | After |
|---|---:|---:|
| Base | 3,415,621,632 B | 3,362,930,688 B |
| Gundam, 13 views | 4,346,888,192 B | 4,195,106,816 B |
| Gundam, 17 views | 4,347,904,000 B | 4,196,122,624 B |

The approximately 52.69/151.78 MB differences include allocation alignment.
**These are not guaranteed overall peak reductions:** one before-change
13-view run sampled 5,718,933,504 bytes, and one after-change 17-view run sampled
5,650,219,008 bytes. The cause of these larger transient peaks is not yet
established; broader resource-lifetime accounting remains open. A 1 ms sampler
can also miss short peaks. Process peak RSS stayed roughly 2.13-2.27 GB and did
not track the private GPU allocation savings consistently. No RSS improvement
or end-to-end speedup is claimed from these memory-probe runs.

Native tracked workspace estimates and peaks remain unchanged because the
reused slots were already counted. A direct probe validated 1024/one-view and
640/eight-view shapes with zero Q/K, constant V=2, and poisoned relative-logit
scratch. Window/global/window operations in one command buffer returned exactly
2, allocated zero new buffers, and performed one wait. An undersized H slice
rejected explicitly before dispatch; restoring it allowed reuse. Request bytes
returned to zero after workspace release. This passed with `MTL_DEBUG_LAYER=1`.

Build/checks: `cmake --build build/release --parallel 4`; the existing Python
suite passed **105 checks**, skipped **7 opt-in checks**, and emitted the three
expected invalid-limit warmup warnings. `ctest --test-dir build/release
--output-on-failure` passed **13 checks**. No shaders changed. Probes, profiles,
and the generated square image under `data.tmp/probes/v2-q4-chunk/` were removed.
The persistent synthetic dataset and user notebook edits were left untouched.

Broader allocation/failure auditing and four-versus-eight-view chunk
qualification remain `todo`; the default remains eight local views. Reference: [7].

### Backing-storage lifetime and chunk-size checkpoint (2026-10-30)

Baseline: `4c6ee1f`, same M1 Pro, model caches, Release/profiling build and invoice.
A disposable instrumented backend sampled `currentAllocatedSize` at workspace
allocation/release boundaries, in addition to the 1 ms sampler. The boundary
trace reproduced the overlap on every maximum-crop request, even when polling
missed it:

- Local workspace: **2,289,385,472 bytes**. After the local wait and MPS cache
  invalidation, one extra reference remained and releasing our reference did
  not free the buffer.
- Allocating the **1,487,122,432-byte** global workspace raised Metal usage to
  **5,663,588,352 bytes** on warmed runs.
- A disposable deallocation marker captured the old buffer's eventual release
  on the driver's completion thread through `MTLResourceList`,
  `IOGPUMetalCommandBufferStorageDealloc`, and `didCompleteWithStartTime`.
  The application's completed-handler retain list was already empty.

The CPU wait establishes GPU completion but does not guarantee that the driver's
resource-list teardown has finished. A tighter autorelease pool around command
buffer creation alone did not remove this overlap; no timing sleeps or extra
GPU submissions were adopted.

The public vision path now uses the existing chunk-shape estimator to reserve
`max(local_workspace, global_workspace)` once. Each shape receives its existing
slice layout within that buffer. Chunk waits remain unchanged, and MPS views are
invalidated before relayout. This also handles four-crop batches, where the
global layout is larger than the local layout. Final visual features remain in
their separate buffer, and the shared workspace is released before decoder
allocation. Native admission estimates do not increase.

The allocation-count regression probe failed before the change (two vision
backing allocations per request), then passed afterward (one). Warmed gundam
allocation counts fell **19 -> 18** at eight crops per chunk. Four maximum-crop
instrumented requests and four ordinary one-token requests showed no shape
transition spike; ordinary warmed Metal peaks were **4,196,122,624 bytes** and
end-of-request usage stayed **1,862,516,736 bytes**. This removes the observed
roughly **1.47 GB** overlap; it is not a hard cap on every driver allocation.

A separate C-caller probe exposed an abort leak: an autoreleased, unsubmitted
batch kept a **32 MiB** GPU buffer alive even after native tracked bytes reached
zero. `metal_command_batch_begin()` now drains the creation autorelease within
its own pool while retaining the context's owned reference. Abort can then
release the batch and its resources without requiring a caller-side pool.
Three repeated aborts returned both tracked bytes and Metal allocation delta
to zero, including transient retention. The probe passed Metal API validation.

#### Four-versus-eight-crop qualification

Four v2 Q4 shapes used complete natural-EOS output, not token-capped throughput:
base (one view), an 800x800 padded invoice (five views), the original invoice
(13 views), and a 2048x2048 padded invoice (17 views, the v2 crop maximum).
Each variant had a warmup, then three interleaved runs in alternating order.
Polling was disabled for latency measurements; model load and image preparation
were excluded. All **385 output IDs** matched between chunk sizes and against
the saved baseline; strict parsing passed.

| Views | Eight-crop request mean | Four-crop request mean | Tracked peak, eight | Tracked peak, four |
|---|---:|---:|---:|---:|
| 1 | 4.320 s | 4.337 s | 3.334 GB | 3.334 GB |
| 5 | 6.286 s | 6.288 s | 3.335 GB | 3.335 GB |
| 13 | 9.636 s | 9.644 s | 4.139 GB | 3.337 GB |
| 17 | 11.282 s | 11.329 s | 4.140 GB | 3.338 GB |

Separate one-token, 1 ms sampled maximum-crop runs measured **4,196,122,624 ->
3,374,202,880 bytes** with eight versus four crops: approximately **822 MB less**
Metal allocation. Four crops cost about 0.4% on the 17-view complete request in
this small sample. Use `UOCR_VISION_LOCAL_MAX_VIEWS_PER_CHUNK=4` for this measured
low-memory configuration. The shared default remains eight: v1 scheduling and
other hardware have not been qualified for a default change. Revisit the
tradeoff after the dense SAM attention buffers are removed.

#### Compatibility, failures, and remaining limits

- Saved-baseline/candidate complete outputs matched for v2 Q8 base (385 IDs),
  v1 Q4 base and gundam (381 IDs), including repeated engine reuse. One initial
  baseline Q8 run differed between cold and warm selections; a fresh baseline
  and candidate repeat matched both cold and warmed outputs. This does not
  establish universal bitwise determinism or upstream numerical parity.
- Under `MTL_DEBUG_LAYER=1`, injected failures at workspace, final-feature,
  pixel-upload and patch-extraction allocation, plus failure after encoding an
  entire local chunk before commit, all cleaned up and allowed reuse. Tracked
  bytes returned to model-only and Metal usage returned within 1 MiB of the
  warmed baseline after each failure. Actual GPU execution faults were not
  injected.
- A 3.6 GB budget rejected the 17-view/eight-crop request before allocation
  (estimate 4,347,414,766 bytes), then accepted a base request on the same engine.
- Native tracked live bytes returned to **1,846,112,256 bytes** after v2 requests.
  Completed repeated requests showed no retained-resource growth. Process RSS
  remains a separate measure; no RSS reduction is claimed.
- The audit found remaining unaccounted pixel-upload and patch-extraction
  allocations: each is 19,660,800 bytes for eight FP16 640-pixel crops. Cached
  position-conversion buffers and framework/driver allocations also remain
  outside complete native accounting. The large backing-buffer overlap is
  fixed, but full accounting remains `todo`.

Checks: Release build passed; **105 Python checks passed, 7 opt-in checks
skipped**, with the same three expected warmup warnings; **13 native checks
passed**. No shader, model-file, precision-policy, or scheduling-default changes.
Disposable instrumented libraries, allocation-failure/abort probes, samples and
padded images lived under `data.tmp/probes/v2-q4-lifetimes/` and were removed after
recording these results. The user notebook and persistent stress data were left
untouched.

## Phase 3: Bounded SAM global attention

Outcome: SAM global attention uses one bounded score/probability tile rather
than materializing every head/view's matrices simultaneously. Native admission
and allocation agree on the smaller working set. A fused online kernel remains
a separate, unfinished optimization.

- [x] done: Prototype `simdgroup_matrix` QK/PV with FP32 accumulation and online
  softmax; compare multiple tile/storage variants against MPS.
- [ ] todo: Ship a fused online kernel only after it beats the existing path.
  **Deferred:** tested variants were slower on the M1 Pro. They are not in
  production, and this checkpoint must not be called fused FlashAttention.
- [x] done: Preserve query-dependent decomposed H/W relative terms and the
  existing FP16 normalized-probability rounding in the accepted MPS path.
- [x] done: Process heads and bounded query blocks sequentially, reusing one
  score/probability pair. Update the shared planner and admission estimator.

QA checkpoint for the bounded-MPS implementation:

- [x] done: Compare full attention outputs to the prior MPS implementation on
  real tensors and adversarial inputs; check a sampled FP64 reference.
- [x] done: Measure GPU latency, normal complete-request latency and allocation
  peaks for 1024/640 views, including maximum crop count.
- [x] done: Pass complete v2 Q4 base/gundam OCR, v2 Q8 base and v1 Q4 regression.
- [x] done: Exercise tile tails, undersized slices, budget rejection, failures
  after encoding, engine reuse, and runtime/precompiled Metal paths.

### Bounded-query checkpoint evidence (2026-10-30)

Baseline: `5c1dfaa`; same M1 Pro, 16 GB, pinned mixed-Q4 model and invoice.
All custom shader experiments stayed under `data.tmp/probes/v2-q4-sam-global/`.
No shader change is shipped in this checkpoint.

#### Rejected fused candidates and selected implementation

Disposable matrix-MMA prototypes used online softmax, FP32 accumulators and
precomputed relative logits. Query/key tiles included 16x32, 32x64, 32x32 and
64x32, with shared-memory and register epilogues. They passed synthetic numerical
checks but lost to MPS. The best tested 64x32 variant took about **78.18 ms versus
49.54 ms** for a 64x64 grid/one view, and **101.84 versus 79.80 ms** for a 40x40
grid/eight views. These are completed command-buffer GPU times for the attention
operation, not full OCR latency. No claim of an exhaustive kernel search is made.

The accepted alternative keeps the current MPS products and softmax shader.
For each head, it processes at most **2,048 query rows**, consumes the resulting
probabilities in PV, and then reuses the same scratch for the next tile. Q/K/VT
packing, relative-logit production and output unpacking remain unchanged. The
softmax sees one head with relative-logit buffer offsets shifted to the tile's
query rows. Existing batch ordering protects producer/consumer lifetimes; no
additional CPU waits are introduced in public generation.

`UOCR_SAM_GLOBAL_MPS_QUERY_TILE` in `src/runtime/uocr_memory.h` is the shared
layout constant for the Metal allocator, dispatch and admission estimator.
The score/probability pair now holds `min(N, 2048) * N * (4 + 2)` bytes, independent
of head and view counts. It is still a dense tile; the fused online work remains
unfinished.

| Shape | Old score/probability pair | New pair | Full vision backing, old -> new |
|---|---:|---:|---:|
| 1024 pixels, one view | 1,207,959,552 B | 50,331,648 B | 1,487,122,432 -> 329,494,528 B |
| 640 pixels, eight views | 1,474,560,000 B | 15,360,000 B | 2,289,385,472 -> 830,185,472 B |
| 640 pixels, four views | 737,280,000 B | 15,360,000 B | 1,145,921,536 -> 424,001,536 B |

A 512-query MPS tile was materially slower. A full 4,096-query tile removed less
base memory. The 2,048-row choice preserved tested numerical results and offered
a useful memory/latency tradeoff. Isolated GPU means, one warmup then three runs,
were **49.53 -> 52.62 ms** for the global view and **79.36 -> 82.75 ms** for eight
local views. The accepted attention operation itself is slightly slower; the
complete-request checks below are the rollout gate.

#### Correctness and public behavior

- Synthetic grids 7/17/40/63/64 exercised one, two and eight views, normal inputs,
  uniform attention, large finite logits and a partial query tile (63x63).
  The accepted path's full FP16 outputs matched the prior MPS path bitwise in
  these checks. An independent FP64 oracle checked sampled queries/all heads
  with `atol=0.003, rtol=0.01`; maximum sampled absolute error was approximately
  0.001454 in the extreme-logit case, including the baseline's FP16 probability
  rounding. No nonfinite outputs occurred.
- A diagnostic backend captured both implementations' complete outputs on real
  Q4 vision tensors. All **16 comparisons** across SAM global layers, global
  views and two eight-crop chunks matched bitwise. These synchronization-heavy
  runs were used only for numerical QA, not performance claims.
- Complete natural-EOS v2 Q4 base, 13-view and 17-view gundam outputs retained
  all **385 IDs** and strict parse results. Four-crop maximum-view output also
  matched. V2 Q8 base retained 385 IDs; v1 Q4 base/gundam retained **381 IDs**.
  These checks establish local regression parity, not upstream full-model parity.

#### Memory and normal latency

Separate-process 1 ms sampling of `MTLDevice.currentAllocatedSize` measured:

| V2 Q4 request | Tracked peak, old -> new | Sampled Metal peak, old -> new |
|---|---:|---:|
| Base | 3,333,935,616 -> 2,176,307,712 B | 3,362,930,688 -> 2,205,302,784 B |
| Gundam, 13 views | 4,139,373,056 -> 2,680,173,056 B | 4,195,106,816 -> 2,735,915,008 B |
| Gundam, 17 views | 4,140,397,056 -> 2,681,197,056 B | 4,196,122,624 -> 2,736,930,816 B |

Four-crop maximum-view gundam measured **2,275,013,120 tracked bytes** and
**2,311,077,888 Metal bytes**. The default remains eight crops. End-of-request
v2 tracked bytes returned to **1,846,112,256**. RSS did not consistently follow
GPU allocation changes, so no RSS improvement is claimed. Sampling can miss
short peaks, and the remaining accounting limits from Phase 2 still apply.

A separate A/B probe loaded both saved-baseline and candidate libraries, warmed
both engines, disabled polling/checkpoint instrumentation, and alternated order
across three pairs of complete requests. Loading/preparation were excluded:

| Request | Baseline mean | Candidate mean |
|---|---:|---:|
| V2 Q4 base | 4.408 s | 4.316 s |
| V2 Q4 gundam, 13 views | 9.690 s | 9.577 s |
| V1 Q4 base | 5.092 s | 5.059 s |

The primary result is lower memory, with no complete-request regression in this
small A/B sample. Do not extrapolate a general throughput improvement from one
fixture. Profiled encoder counts rose **851 -> 1,039** for base and **2,740 ->
3,496** for 13-view gundam; waits stayed **386** and **388**, respectively.

#### Failure, build and cleanup checks

- The disposable workspace-bound probe failed on the old estimator, then
  passed with actual workspace sizes matching the new estimates. An undersized
  score slice rejected before dispatch and could be restored/reused.
- Three injected failures after tiled QK or after tiled PV cleaned up, returned
  tracked bytes to model-only and Metal usage within 1 MiB of its warmed baseline,
  and allowed reuse. A complete 385-token request after those failures matched
  the baseline. Actual GPU execution faults were not injected.
- A **2.5 GB** budget rejected the eight-crop maximum-view request (estimate
  **2,815,254,766 bytes**), then accepted base and four-crop maximum-view requests
  on the same engine.
- Numerical and failure probes passed `MTL_DEBUG_LAYER=1`; complete maximum-crop
  production generation also passed validation. A runtime-only probe library
  ran from an empty resource directory with the generated shader source and
  matched precompiled base output. Production shader sources are unchanged.
- Release build passed; **105 Python checks passed, 7 opt-in checks skipped**,
  with the three expected invalid-limit warmup warnings; **13 native checks
  passed**. Disposable prototypes, libraries, logs and padded images were removed.
  The user notebook and persistent synthetic stress data were untouched.

References: [1], [2] remain relevant to a future faster fused implementation.

## Phase 4: Matrix-tiled SAM window attention

Outcome: v2's canonical SAM windows use matrix MMA for QK/PV with FP32 online
softmax. V1 retains its original reduction order. Small v2 floating-point
rounding differences are explicitly qualified below.

Window attention already stages K/V tiles and uses online softmax. The new work
is matrix-core utilization and reuse across multiple queries.

- [x] done: Specialize matrix tiles for head dimension 64 and 14x14 windows,
  preserving relative-position terms and existing image-padding semantics.
- [x] done: Adapt the earlier matrix-tile prototype to short windows without
  forcing the same dispatch onto global attention. Global attention remains MPS.
- [ ] todo: Evaluate reuse for CLIP attention only after the SAM checkpoint;
  keep it only if the different sequence lengths benefit.

QA checkpoint:

- [x] done: Verify padded/non-padded window behavior and tile-edge masking.
- [x] done: Compare window kernel time and full vision latency in base/gundam;
  confirm improvements survive with detailed checkpoints disabled.
- [x] done: Verify OCR, memory accounting, and both shader build paths.

### V2 window-MMA checkpoint evidence (2026-10-30)

Baseline: `809c72b`; same M1 Pro, 16 GB, pinned models and Release/profiling build.
The kernel is `uocr_sam_window_attention_mma_f16` in `sam_attention.metal`;
`uocr_smoke.metal` is regenerated from the fragments.

- Each 128-thread group computes **16 queries x 32 keys** with four SIMD groups.
  QK uses FP16 input matrices and FP32 accumulation. PV uses **FP32
  probabilities**, FP16 V and FP32 accumulation; no additional half-precision
  probability rounding or persistent dequantized weights are introduced.
- Online maxima/sums are updated per key tile. Shared K storage is reused for V,
  and score storage is reused for output rescaling after barriers. Static
  threadgroup memory is **12,480 bytes**; no new device buffers are allocated.
- All 196 tokens participate, including image-padding tokens already present in
  a 14x14 window. Only query/key positions beyond 196 are masked as tile tails.
- Dispatch selects MMA only for **v2, 14x14 windows, head dimension 64**, with
  matching relative tables. V1 and other shapes retain their previous kernels;
  nonmatching-table interpolation is unchanged. A missing MMA pipeline fails
  explicitly rather than silently changing dispatch.
- An exploratory v1 run moved a box coordinate by one unit on its 0-999 grid.
  Production therefore leaves v1 arithmetic unchanged to preserve compatibility.
  This checkpoint does not enable the new kernel for v1.

#### Numerical and OCR qualification

Synthetic checks exercised 1/25 windows, 9-window batches of 4/8 views, uniform
attention, large finite logits, biased image-padding tokens, and the partial
query/key tiles at token 196. The tolerance was `atol=0.003, rtol=0.003` against
both the prior implementation and an independent sampled FP64 oracle. Maximum
absolute difference from the scalar path was **0.0004883** on the normal cases
and **0.0019532** on the extreme-logit case; maximum sampled FP64 error was
approximately **0.0009664**. No nonfinite outputs occurred.

A diagnostic backend compared complete outputs on real Q4 tensors across
**32 window-attention calls**: eight layers in base plus eight layers for each
of the maximum-crop request's three chunks. Maximum absolute difference was
**0.0009765625**, with per-call RMS error no greater than **3.583e-6**. These
results are numerically close, not bitwise identical. Diagnostic waits/readbacks
were excluded from performance measurements.

Complete natural-EOS and strict parsing checks:

| Path / fixture | Tokens | Comparison with baseline |
|---|---:|---|
| V2 Q4 invoice, base / 13-view / 17-view gundam | 385 | Exact IDs |
| V2 Q4 blank, base | 19 | Exact IDs |
| V2 Q4 table, base | 448 | Exact IDs |
| V2 Q4 columns, base | 577 | Exact IDs |
| V2 Q4 edge text, gundam | 300 | Exact IDs |
| V2 Q4 degraded scan, gundam | 385 | Exact IDs |
| V1 Q4 invoice, base / gundam | 381 | Exact IDs; original kernel |
| V2 Q8 invoice, base | 385 | Same text, block order/types and metadata; four box coordinates moved by one unit |

**Q8 is not token-identical:** four coordinates across three blocks moved by
**0.001** on the normalized v2 page grid. The comparison required exact text and
non-geometric structure, and no box coordinate change beyond one native integer
unit. The drift was repeatable in the checked cold/warm runs. This is a measured
rounding difference, not a repair applied by the parser or a guarantee that all
inputs have the same bound. V2 FP16 end-to-end OCR and universal output parity
remain unverified.

#### Performance and memory

Isolated attention probes used one warmup and four measured runs (median GPU
command-buffer duration), including unchanged relative-logit production:

| Shape | Scalar | Matrix |
|---|---:|---:|
| 25 windows, one global view | 42.14 ms | 9.21 ms |
| 9 windows, eight local views | 121.20 ms | 26.65 ms |
| 9 windows, four local views | 60.63 ms | 13.34 ms |

This is approximately **4.6x faster** for the window operation. A separate
complete-request A/B probe warmed both libraries and alternated order across
three pairs, with detailed checkpoints and polling disabled:

| V2 Q4 invoice | Total, before -> after | Vision, before -> after |
|---|---:|---:|
| Base | 4.311 -> 4.046 s | 1.101 -> 0.836 s |
| Gundam, 13 views | 9.588 -> 8.181 s | 5.127 -> 3.728 s |

Complete-request latency fell about **6.2% / 14.7%**, with unchanged output
lengths. This is a measured fixture result, not a general throughput guarantee.
Encoder/wait/allocation counts were unchanged: base **1,039 / 386 / 14** and
13-view gundam **3,496 / 388 / 18**. Tracked peaks remained **2,176,307,712 /
2,680,173,056 bytes**. Separate 1 ms Metal sampling also retained peaks of
**2,205,302,784 / 2,735,915,008 bytes**, with end-of-request usage **1,862,516,736
bytes**. No admission-estimate or crop-default change was needed.

#### Build, failure and cleanup checks

- Runtime-compiled and precompiled base/maximum-crop output IDs matched. The
  runtime-only probe used an empty resource directory and the regenerated
  source. Regeneration was checked for byte-for-byte idempotence.
- Four-crop maximum-view output matched the eight-crop baseline. A standalone
  probe checked v1 dispatch, v2 MMA, a noncanonical 10x10 window, and interpolated
  relative tables. Constant-V results were exact. An undersized relative slice
  rejected before encoding and allowed reuse after restoration.
- Injected matrix-pipeline and post-encoding failures cleaned up and allowed
  reuse, including a subsequent complete request matching all 385 baseline IDs.
  A 2.5 GB budget still rejected the eight-crop maximum request and accepted a
  smaller base request afterward. Actual GPU execution faults were not injected.
- Numerical, dispatch, failure and runtime maximum-crop probes passed Metal API
  validation. Release build and existing suites passed: **105 Python checks,
  7 opt-in skips**, the three expected warmup warnings, and **13 native checks**.
- Probes/libraries/logs and generated images under
  `data.tmp/probes/v2-q4-sam-window/` were removed after recording evidence. The
  persistent stress dataset and the user's concurrent notebook edits were not
  modified or staged.

CLIP specialization remains `todo`. References: [1], [2].

## Phase 5: Parallel DASH sparse attention

Outcome: decode attention has enough parallel work while retaining sparse
memory usage and the exact selection/bias contract.

The baseline assigns one SIMD group per query/head: only 10 SIMD groups for
decode, each visiting up to 512 selected/local keys serially. The accepted
specialization computes scores and stages values cooperatively while preserving
the original recurrence.

- [x] done: Benchmark multiple SIMD groups per head and small split-K variants,
  including stable merging of partial maxima, denominators and weighted outputs.
  Retain an order-preserving tiled variant instead of split-softmax reductions.
- [x] done: Preserve remote biases, causal bounds, empty partitions, and
  single inclusion of overlapping local tokens.
- [ ] todo: Separately investigate matrix-tiled sparse prefill where selections
  allow useful reuse; do not materialize a dense attention matrix.
- [x] done: Bound temporary storage in static threadgroup memory;
  the accepted kernel needs no device reduction scratch. Select dispatch from
  measured query shape and retain the scalar prefill/short-query path.

QA checkpoint:

- [x] done: Compare against current attention with fixed selections, overlap,
  empty remote sets, partial blocks, and extreme logits.
- [x] done: Compare SIMD-group/tile configurations and kernel/full-decode latency;
  reject configurations that regress the target shape. Hardware-counter
  occupancy was not measured.
- [x] done: Confirm unchanged page/EOS behavior and full OCR quality checks.

### Order-preserving DASH decode checkpoint evidence (2026-10-30)

Baseline: `ceed60c`, Apple M1 Pro with 16 GB RAM, the same pinned models and
Release/profiling configuration as phase 4. Production changes are limited to
`dash.metal`, its regenerated source, and DASH dispatch in `uocr_metal.m`.

#### Candidates and selected behavior

- Split-softmax variants used 2/4/8/16 SIMD groups per head. They merged partial
  maxima, denominators and weighted values in FP32, explicitly handling empty
  partitions. They passed `atol=0.003, rtol=0.003` against the scalar path and a
  separate FP64 reference, and the fastest variants were about 6-7x faster on
  the representative isolated decode shapes.
- The eight-group split variant reduced warmed base invoice time from about
  **4.028 to 3.359 s**, but moved several box coordinates, including a two-unit
  shift on the v2 0-1000 grid. Text remained unchanged. This variant was not
  accepted: an order-preserving alternative recovered most of its request-level
  benefit without the measured output drift.
- Precomputing scores alone retained bitwise outputs but gave only about 20%
  isolated improvement. Staging V alongside scores removed the remaining
  serialized device reads. Four consumer SIMD groups each handle 32 output
  dimensions while retaining the original key order and scalar FP32 operations.
- The accepted `uocr_dash_attend_decode_tiled_f16` uses **16 SIMD groups / 512
  threads**, a **32-key tile**, FP32 scores and FP16 staged V: **8,320 bytes** of
  static threadgroup storage. QK still uses the original four terms per lane
  and SIMD reduction. Softmax and weighted accumulation do not merge partial
  reductions or introduce probability downcasts.
- Tile widths 16/32/64 and 4/8/16/32-group variants were explored. A 64-key tile
  improved representative isolated timings only about 1-2% over the selected
  configuration while doubling shared storage; 32 groups did not improve it.
  The selected tile also matches the 32-token sparse block size.
- Dispatch uses the new kernel only when `n_queries == 1` and
  `query_start >= 127`, so a complete 128-token local window is available.
  Prefill and shorter queries retain the original kernel. Prefill measurements
  regressed with the decode specialization; matrix-tiled sparse prefill remains
  a separate `todo` rather than being enabled through this dispatch.
- Buffer bindings, selection order/biases, causal/local overlap handling,
  summaries, routing, page-aware RoPE, cache retention and page/EOS guards are
  unchanged. V1 does not use DASH. Pipeline/width/thread-count/shared-memory
  failures are explicit; there is no silent fallback after a selected pipeline
  fails. No new device buffers, reductions or CPU/GPU submissions are added.

#### Numerical and functional checks

- The production shader was compared against the prior scalar kernel and an
  independent FP64 sparse-attention reference on **16 cases**. These cover
  empty remote sets, uniform attention, large finite logits/biases, overlap,
  partial blocks/tiles, short queries, multiple queries, and scattered block
  selections with key counts up to **32,769**. Production output was **bitwise
  identical** to the scalar kernel; maximum absolute FP64 error was
  **0.000958801**. API and GPU shader validation passed in a separate run whose
  instrumented timings were excluded from performance evidence.
- A diagnostic backend compared outputs and selections on **1,536 real Q4
  attention calls**: 64 decode steps across 12 layers in each of base/gundam.
  Every output element and selection byte matched. Diagnostic synchronization
  and readback were excluded from timing.
- The host DASH bridge matched baseline outputs, selections and summaries on
  nine shapes, including the 127/128-key dispatch boundary, a partially
  overlapping remote block, multi-query prefill, and varying image-page ranges.
  Invalid query ranges left outputs untouched, and valid reuse succeeded.
- Complete natural-EOS/strict-parsing checks matched all baseline IDs:

| Path / fixture | Output tokens |
|---|---:|
| V2 Q4 invoice, base / 13-view / maximum 17-view gundam | 385 |
| V2 Q4 blank, base | 19 |
| V2 Q4 table, base | 448 |
| V2 Q4 edge text, gundam | 300 |
| V2 Q4 degraded scan, gundam | 385 |
| V2 Q4 tiny text, base | 1,172 |
| V2 Q8 invoice, base | 385 |
| V1 Q4 invoice, base / gundam | 381 |

Q8 matches this checkpoint's `ceed60c` baseline; the earlier phase 4 rounding
qualification is unchanged. Four-crop maximum-view output also matched the
eight-crop baseline. These checks do not establish universal/compiler-independent
bitwise equivalence or upstream/v2-FP16 end-to-end parity.

#### Timing and memory

Isolated probes timed repeated dependent dispatches with GPU timestamps, one
warmup command buffer and three measured buffers. Each measured tuning buffer
contained 256 serial dispatches; the table reports median time per dispatch.
These use fixed selections and include sparse attention only, not routing:

| Retained keys / single query | Scalar | Selected tiled kernel |
|---|---:|---:|
| 128, local only | 48.05 us | 22.18 us |
| 279, selected prefix plus local window | 100.05 us | 42.16 us |
| 1,519, 12 remote blocks plus local window | 178.40 us | 73.84 us |

The selected path is about **2.2-2.4x faster** in these isolated shapes. Whole
request A/B measurements warmed both libraries and alternated order across
three pairs at each output cap, without polling or detailed checkpoints:

| V2 Q4 invoice | Before | After |
|---|---:|---:|
| Base, cap 1 | 1.067 s | 1.052 s |
| Base, cap 65 | 1.511 s | 1.459 s |
| Base, natural EOS / 385 IDs | 4.083 s | 3.675 s |
| Gundam, cap 1 | 4.578 s | 4.577 s |
| Gundam, cap 65 | 5.163 s | 5.052 s |
| Gundam, natural EOS / 385 IDs | 8.260 s | 7.566 s |

Complete-request means improved **10.0% / 8.4%**. Base candidate timings ranged
from **3.586 to 3.832 s**; all three samples are retained in that mean. The
early incremental decode estimate `(T65-T1)/64` changed from **6.94 to 6.36
ms/token** in base and **9.14 to 7.42 ms/token** in gundam. First-token/prefill
work is unchanged; the small cap-1 difference is measurement variation. A
separate, non-interleaved warmed tiny-text check improved **10.534 to 8.863 s**
with exactly 1,172 IDs. These are fixture results, not general throughput claims.

Encoder/wait/allocation counts remained **1,039 / 386 / 14** for base and
**3,496 / 388 / 18** for gundam. Tracked peaks remained **2,176,307,712 /
2,680,173,056 bytes**. Separate 1 ms Metal sampling during 65-token requests
retained peaks **2,205,302,784 / 2,735,915,008 bytes**, and end-of-request Metal
usage **1,862,516,736 bytes**. Process RSS high-water marks were approximately
**2.131 -> 2.140 GB** for base and **2.214 -> 2.215 GB** for gundam; these include
Python/compiler/driver state and are not the native admission estimate.

#### Failure, build and cleanup checks

- Injected pipeline-lookup and post-dispatch failures returned errors, released
  request allocations, and allowed repeated reuse, followed by a complete
  385-ID request matching baseline. Actual GPU execution faults were not
  injected. A 2.5 GB budget rejected gundam and permitted base afterward.
- Runtime-only compilation from regenerated source in an empty resource
  directory matched precompiled base/gundam outputs. A separate profiling-OFF,
  runtime-compilation-OFF, precompiled build passed all 13 native checks and
  matched the complete base output. Source regeneration was byte-identical.
- Main Release build and existing suites passed: **105 Python checks, 7 opt-in
  skips**, the three expected warmup warnings, and **13 native checks**.
- Owned probes, private libraries/builds and logs under
  `data.tmp/probes/v2-q4-dash/` were removed after recording evidence. No
  permanent tests were added. The persistent stress dataset and concurrent
  notebook edits were not modified or staged by this work.

References: [2], [4].

## Phase 6: Occupancy-aware routed-MoE prefill

Outcome: small expert buckets avoid unnecessary 64-row padding while larger
buckets retain efficient tiled execution.

The base fixture averages `279 * 6 / 64 = 26.2` selected rows per expert;
actual routing distribution must be measured before choosing a dispatch policy.
Expert sorting/bucketing and fused Q4 MMA are already implemented.

- [x] done: Measure per-expert row-count distributions on representative prompts.
- [x] done: Benchmark 16/32-row variants alongside the existing 64-row path.
- [x] done: Retain GPU bucketing/prefix scheduling with a fixed 16-row v2 Q4
  tile; measured results did not justify a mixed-width policy or route readback.
- [x] done: Evaluate empty/partial-tile overhead. Smaller tiles address padding;
  compact/indirect dispatch is deferred because measured empty-work cost is small.
  Fused gate/up and deterministic expert combination are preserved.

QA checkpoint:

- [x] done: Exercise skewed routing, empty experts, and partial tiles against
  the current implementation using the same routes and quantized weights.
- [x] done: Measure complete MoE prefill including bucket/combine overhead,
  then verify end-to-end base/gundam latency and output quality.

### V2 Q4 M16 checkpoint evidence (2026-10-30)

Baseline: `f63be0b`, Apple M1 Pro with 16 GB RAM, pinned mixed-Q4 models and
Release/profiling build. Changes are limited to the Q4 Metal fragment, its
regenerated source, and the routed-Q4 prefill dispatch in `uocr_metal.m`.

#### Routing distribution and tile decision

A diagnostic backend collected counts for all 64 experts in each of 11 routed
layers. Counts and padding utilization were measured before choosing a policy;
production does not read routing data back to the CPU.

| Prompt | Min / median / max rows per expert | Empty buckets across 11 layers | Useful rows / padded rows, M64 -> M16 |
|---|---:|---:|---:|
| Invoice, base / 279 tokens | 0 / 25 / 117 | 18 of 704 | 41.0% -> 77.8% |
| Invoice, gundam / 1,519 tokens | 4 / 128 / 1,039 | 0 of 704 | 81.3% -> 94.9% |
| Blank, base / 279 tokens | 0 / 18 / 240 | 33 of 704 | 38.7% -> 76.4% |

GPU command-buffer timestamps measured all four MoE passes (bucket, fused
SwiGLU gate/up, down, combine), using identical real inputs, routes and weights.
Each layer/width had one warmup plus three measured runs; the table sums the
per-layer medians. Diagnostic readbacks and synchronization outside these
command buffers are not included:

| Invoice prompt | M64 | M32 | M16 |
|---|---:|---:|---:|
| Base, all 11 routed layers | 156.36 ms | 99.33 ms | 81.33 ms |
| Gundam, all 11 routed layers | 443.72 ms | 378.63 ms | 362.97 ms |

M16 also won or tied the tested skew cases: six expert slots with **0, 1, 16,
17, 32, 33, 65, 128, 279 or 1,519 rows each**, with the remaining experts empty.
Invalid pair IDs supplied the fully empty case and unused rows. Examples of
complete first-layer GPU time, M64 -> M16: **1.492 -> 0.431 ms** at one row,
**2.934 -> 2.720 ms** at 128 rows, and **33.820 -> 31.309 ms** at 1,519 rows.
All widths matched the reference combined outputs bit-for-bit.

No measured occupancy range favored adding a mixed-width scheduler. The
all-empty 279-token four-pass probe cost approximately **0.08 ms** for each
width, so indirect dispatch would target less than about 1 ms across 11 layers
in that probe. It was not added. Hardware-counter occupancy was not measured.

#### Implementation and scope

- New M16 gate/up and down kernels use **16x32x32 tiles and 128 threads**.
  Q4 unpacking/scales, BK=32, the per-output 8x8 MMA reduction sequence,
  FP32 accumulators, FP16 intermediate stores and final combine remain unchanged.
- Static threadgroup storage, including pair IDs, is **5,184 bytes gate/up /
  3,136 bytes down**, versus **16,640 / 8,448 bytes** for M64. The M16 staging
  buffers are sized for A/B inputs, which exceed the smaller output-tile sizes;
  simply scaling the M64 output-storage formula would underallocate them.
- The existing GPU histogram/scatter/prefix pass uses 16 rows per tile. The
  conservative launch bound remains `ceil(pair_count / rows_per_tile) + 64`;
  unused slots return before GEMM. There is no route readback, new buffer,
  persistent dequantized weight copy or additional dispatch.
- Only **v2 Q4 routed prefill** selects M16. Decode, v1, Q8/FP16, vision GEMMs
  and dense/shared MLPs retain their previous dispatch. Required pipelines,
  32-lane SIMD width, thread capacity and threadgroup memory are checked;
  failures are explicit rather than silently selecting M64.

#### Whole-request acceptance

Both libraries were warmed, then request order alternated over six pairs at
cap 1 and four pairs at natural EOS. Image preparation/model loading were
excluded, and detailed checkpoints/polling were disabled:

| V2 Q4 invoice | Before | After | Improvement |
|---|---:|---:|---:|
| Base, first output token | 1.067 s | 0.982 s | 8.0% |
| Gundam, first output token | 4.531 s | 4.447 s | 1.9% |
| Base, complete 385-ID output | 3.530 s | 3.455 s | 2.1% |
| Gundam, complete 385-ID output | 7.525 s | 7.450 s | 1.0% |

The approximately **85 ms first-token / 75 ms complete-request** savings are
fixture results, not a decode-speed or general throughput claim. Complete Q4
IDs matched on base/13-view/maximum-17-view invoices, blank (19 IDs), table
(448), edge text (300) and degraded scan (385), with natural EOS and strict
parsing. V1 base/gundam invoice IDs also matched (381).

Q8 remained on its original Q8 bucketed path, confirmed by profile events.
An initial baseline run itself varied two box coordinates by one unit between
cold/warm requests. Six further requests per library were checked; every
candidate output matched an observed baseline ID sequence, with no new output
variant. The initial baseline had two variants and the candidate repeats one.
This is not a claim of deterministic Q8 output; the cause of that existing
variation was not established in this checkpoint.

#### Memory, failure and build QA

- Production M16 combined outputs matched M64 bit-for-bit across **33 real
  layer inputs** (base/gundam invoices and base blank). Synthetic skew/empty/
  tail comparisons also matched. These are native-reference checks, not
  upstream or FP16-model numerical parity claims.
- Encoder/wait/allocation counts remain **1,039 / 386 / 14** for base and
  **3,496 / 388 / 18** for gundam. Tracked peaks remain **2,176,307,712 /
  2,680,173,056 bytes**. Separate 1 ms Metal sampling retained approximately
  **2.205 / 2.736 GB** peaks (one base sample was 16 KiB higher), with unchanged
  end usage **1,862,516,736 bytes**. RSS high-water marks were approximately
  **2.142 -> 2.139 GB** base and **2.224 -> 2.223 GB** gundam; these process
  measurements are not the native admission estimate.
- Injected M16 pipeline and post-gate-encoding failures released request
  allocations and allowed repeated reuse, then a complete matching 385-ID
  output. A 2.5 GB budget rejected gundam and accepted base afterward. Actual
  GPU execution faults were not injected.
- Metal API validation passed the comparison/failure probes. Production M16
  base prefill also passed GPU shader validation. **Limitation:** running the
  M64 comparison under GPU shader validation aborted because instrumentation
  expanded its gate kernel to **33,280 bytes**, above the device's **32,768-byte**
  threadgroup limit. M64 comparisons therefore used API validation only; the
  ordinary M64 allocation is within the hardware limit.
- Runtime-only source compilation from an empty resource directory matched
  precompiled base/gundam complete outputs. Regeneration was byte-identical.
  Profiling-OFF was not separately rebuilt in this checkpoint.
- Release build and existing suites passed: **105 Python checks, 7 opt-in
  skips**, the three expected warmup warnings, and **13 native checks**.
- Owned probes, private libraries and logs under
  `data.tmp/probes/v2-q4-moe-prefill/` were removed after recording evidence.
  No permanent tests were added. Persistent stress data and concurrent notebook
  edits were not modified or staged by this work.

References: [5], [6]. External performance figures are from different
models/hardware and are not expected speedups for this engine.

## Phase 7: Q4-specific vision GEMM staging

Outcome: faster vision GEMMs with unchanged group-64 half-split storage and no
persistent dequantized copies.

- [x] done: Benchmark vectorized packed loads, cheaper nibble conversion, and
  packed-byte/scale reuse across the two halves of a quantization group.
- [x] done: Evaluate one-group-ahead register prefetching, recording latency and
  static threadgroup storage. It did not improve the tested shapes.
- [x] done: Tune large vision GEMMs separately from small expert buckets.
- [ ] todo: If revisiting this work, collect hardware register/occupancy counters;
  register pressure was not directly measured in this investigation.
- [ ] todo: Qualify and land a production candidate whose whole-request benefit
  justifies the extra kernels/dispatch. Current prototypes are deferred.

Prior evidence in `howto_quant.md`: Q8 `BN=64` caused occupancy collapse and
`BK=64` was slightly slower. Revisit larger tiles only with a concrete
Q4-specific rationale; bigger tiles are not an optimization by themselves.

QA checkpoint:

- [x] done: Compare numerical results on actual quantized tensors, preserving
  SAM erf-GELU and CLIP QuickGELU behavior and all fused epilogues.
- [x] done: Report shape-specific GPU time and normal-run vision/request latency;
  reject variants that merely shift cost into staging or extra allocations.
- [x] done: Verify base/gundam OCR and runtime/precompiled shader parity for the
  leading prototype. This is not an accepted production checkpoint.

### Vision-GEMM investigation evidence (2026-10-30)

Baseline: `264afc8`, Apple M1 Pro with 16 GB RAM, pinned v2 mixed-Q4 model and
Release/profiling build. **No production code, shader, dispatch or estimator was
changed.** Private kernels/backends were used for the measurements below.

Tracing found **145 Q4 vision GEMM calls / 11 shapes** in base and **435 calls /
33 shapes** in 13-view gundam. These include SAM's padded-window QKV/project
shapes, SAM MLPs, CLIP QKV/project/MLPs, and the visual projector. For example,
CLIP uses K=1024/4096 with M=257/404/808, while large SAM matrices reach
M=14,112. All shapes use the existing group-64 half-split Q4 storage.

#### Candidates

Ten variants were evaluated, with no persistent dequantized copies:

1. Packed vector loads replacing scalar byte loads.
2. Reusing packed bytes and scale across low/high halves while retaining BK=32.
3. BK=64 staging, unpacking both halves together.
4. FP32 FMA-based nibble conversion.
5. One-group-ahead register prefetching.
6. BM=32/BK=32.
7. BM=32/BK=32 plus packed-byte/scale reuse.
8. Balanced BM=32/BK=64, which keeps staging within 8 KiB.
9. Contiguous B staging with transposed SIMD-matrix loads.
10. Contiguous/transposed B staging plus low/high-half reuse.

Vector loads and FMA conversion gave small gains on some shapes. Reuse with
BK=32 was usually about 1-2% faster. Register prefetching and transposed staging
did not improve the representative cases. BM=32/BK=32 regressed large SAM
matrices. BK=64 with BM=64 needed **12 KiB** rather than the baseline **8 KiB**
and was generally slower. Balancing it with BM=32 restored **8 KiB** staging
and helped small matrices, but still regressed large SAM shapes.

Representative isolated GPU times (M, N, K; median milliseconds per dispatch):

| Shape / epilogue | Baseline | BK32 reuse | BM32/BK64 |
|---|---:|---:|---:|
| 257, 4096, 1024 / QuickGELU | 1.211 | 1.194 | 1.129 |
| 4900, 2304, 768 / QKV split | 8.136 | 8.020 | 8.398 |
| 12800, 3072, 768 / erf-GELU | 27.169 | 26.772 | 28.123 |

Each case used one warmup command buffer and three measured buffers, each with
four dependent dispatches. These are synthetic-input kernel timings, not
complete vision latency. A broader three-way sweep covered all 33 traced
shapes. Weighting by actual call counts predicted Q4-GEMM time of **419.84 ->
407.15 ms** base and **2,271.47 -> 2,229.62 ms** gundam for the leading policy:
BM32/BK64 at M <= 512, BK32 reuse otherwise. This policy was a probe heuristic,
not a general shape-dispatch qualification.

#### Numerical and build checks

- Synthetic checks exercised bias-only, QKV-split, QuickGELU and erf-GELU
  epilogues, including a 65-row tail. All tested candidates matched baseline
  FP16 outputs bit-for-bit. An independent sampled FP64 reference included the
  existing FP16 dequant-on-stage rounding; maximum sampled error in the
  all-shape sweep was **0.000873** (`atol=0.003, rtol=0.003`).
- A separate small-shape sweep included zero scales, FP16 subnormal scales and
  larger finite scales, and passed Metal API/GPU shader validation for all ten
  variants. Maximum sampled FP64 error was **0.001764**. Instrumented timings
  were not used for performance decisions.
- The leading policy matched baseline outputs bit-for-bit on **44 actual
  quantized GEMM inputs**: one call per distinct base/gundam shape. Diagnostic
  synchronization/readbacks were excluded from timing.
- Complete base/gundam invoice outputs matched all **385 IDs**, reached natural
  EOS and passed strict parsing. Runtime-only compilation from the prototype
  source in an empty resource directory also matched precompiled outputs.
- V1/Q8/FP16, broader OCR fixtures, failure injection and profiling-OFF builds
  were not requalified for the prototype, since no production rollout was
  accepted. A future candidate still needs those applicable checks.

#### Whole-request measurements and decision

Both private libraries were warmed. Order alternated across eight pairs at cap
1 and four pairs at natural EOS, with detailed checkpoints and polling disabled.
Preparation/model loading were excluded; output lengths were identical:

| V2 Q4 invoice | Before | Candidate | Mean saving |
|---|---:|---:|---:|
| Base, first token | 0.995 s | 0.968 s | 27 ms / 2.7% |
| Gundam, first token | 4.452 s | 4.405 s | 47 ms / 1.1% |
| Base, complete request | 3.458 s | 3.450 s | 7 ms / 0.2% |
| Gundam, complete request | 7.444 s | 7.408 s | 36 ms / 0.5% |

Normal-run vision means in the complete requests were **833.88 -> 823.50 ms**
base and **3,745.74 -> 3,709.64 ms** gundam. Complete-request paired savings
ranged **-43.7 to +55.6 ms** in base and **-15.8 to +86.7 ms** in gundam;
paired standard deviations were **51.7 / 46.8 ms**. Every gundam cap-1 pair was
faster, so the result is not evidence of zero benefit. The complete-request
benefit was nevertheless too small and variable to justify two additional
kernels and a shape-dependent policy in this checkpoint. **Production rollout
is deferred**, rather than presenting isolated GEMM gains as an OCR speedup.

Tracked memory reports matched exactly. Complete-request encoder/wait/allocation
counts stayed **1,039 / 386 / 14** base and **3,496 / 388 / 18** gundam; tracked
peaks remained **2,176,307,712 / 2,680,173,056 bytes**. Separate Metal/RSS sampling
was not repeated for these discarded prototypes.

The unchanged production build passed **105 Python checks, 7 opt-in skips**,
the three expected warmup warnings, and **13 native checks**. Owned probes,
private libraries and logs under `data.tmp/probes/v2-q4-vision-gemm/` were removed
after recording evidence. No permanent tests were added; persistent stress data
and concurrent notebook edits were not modified or staged by this work.

Reference: [5].

## Phase 8: Q4 LM-head and decode GEMV tuning

Outcome: measured decode improvements without full-logit or dequantized-weight
buffers and without changing model-file layout.

The LM head reads approximately 88 MB of weights/scales per token. The baseline
used 1,024-thread groups; the accepted path uses 256 threads while preserving
32-row argmax partials. Existing packing and scale semantics are unchanged.

- [x] done: Benchmark smaller groups with multiple output rows per SIMD group
  and activation/preprocessing reuse across rows.
- [x] done: Investigate quantized dot-product instruction sequences while
  retaining current packing/scales. Reject slower signed-nibble conversion.
- [x] done: Preserve fused partial argmax and the existing logical geometry;
  partial-buffer sizing and the finalize pass do not change.
- [x] done: Apply activation-load reuse to routed/shared/dense gate/up GEMVs;
  retain the existing linear/down-combine paths and their group sizes.

QA checkpoint:

- [x] done: Verify tail dimensions, tied logits, invalid-token masks, and page/EOS
  filtering before argmax; compare against the existing quantized path.
- [x] done: Measure effective payload bandwidth and shape-specific GPU time,
  then confirm a complete-request improvement.
- [ ] todo: Collect hardware instruction/register/occupancy counters if further
  tuning is needed; these counters were not measured in this checkpoint.

### Q4 decode checkpoint evidence (2026-10-31)

Baseline: `e66094f` (production code from `264afc8`), Apple M1 Pro with 16 GB
RAM, pinned models and Release/profiling build. The saved native library and
metallib were used together for production A/B comparisons, since the LM-head
launch contract changes from 1,024 to 256 threads.

#### Selected changes and rejected alternatives

- The LM head now uses **eight 32-lane SIMD groups / 256 threads**. Each SIMD
  group visits four vocabulary rows sequentially, retaining the original
  per-lane dot/chunk order. It still produces one partial per **32 vocabulary
  rows**, or **4,040 partials** at vocabulary size 129,280.
- After an unconditional threadgroup barrier, the first SIMD group reduces the
  32 scores with `simd_max` and chooses the lowest matching valid ID with
  `simd_min`. The score is fetched from the winning row, preserving its bits
  even on signed-zero ties. Filtering and NaN exclusion still happen before
  reduction; fully masked groups retain the invalid-ID/-infinity sentinel.
- Scores and IDs still occupy **32,320 bytes total**. Dynamic threadgroup
  partial storage remains **256 bytes**. There is no full-logit buffer, new
  dispatch, persistent dequantized copy, format change or estimator change.
  Host validation requires 32-lane SIMD groups; compile-time assertions pin
  the 32-row partial/256-thread contract.
- Gate/up GEMVs now load and convert each activation chunk once for both dots.
  Gate and up have independent FP32 accumulators with the same chunk order,
  dot expressions and scale placement as before. SwiGLU/FP16 stores are
  unchanged. The linear/down-combine kernels are not modified.
- These shared Q4 primitives also serve **v1**. Unlike the earlier v2-only
  window/MoE-tile changes, this checkpoint intentionally applies the same
  arithmetic-preserving change to both versions; v1 output checks are below.
  Q8/FP16 paths and prefill GEMMs are unchanged.

LM-head candidates included 128/256/512-thread groups, two simultaneous rows
per SIMD group, canonical-hidden-size specialization, cached activation
vectors, 128/256-row physical tiles, and parallel partial reductions. Smaller
groups alone saved about 2-3%; parallel reduction supplied the rest. Wider
physical tiles and simultaneous-row computation did not beat the selected
32-row/256-thread path. Signed-nibble conversion and activation preloading were
about 45% slower on the full-vocabulary probe and were discarded. GEMV group
sizes 32/64/128/256 did not justify replacing the existing 128-thread dispatch.

#### Kernel measurements

The production LM-head probe used one warmup and three measured command
buffers, each with 24 dependent dispatches. At hidden size 1,280 / vocabulary
129,280, median GPU time changed **0.797 -> 0.748 ms**. The nominal weight/scale
payload is **87,910,400 bytes**, implying approximately **110 -> 118 GB/s**
effective payload bandwidth, not a measured DRAM counter. Forced-token cases
were also checked; they should not be interpreted as streaming bandwidth.

Small GEMV timings varied when each candidate reallocated buffers. A follow-up
kept identical buffers resident and alternated old/new pipelines over five
pairs after warmup, with 256 dependent dispatches per buffer:

| Gate/up shape | Before | After |
|---|---:|---:|
| Shared: K=1280, N=1792 | 23.65 us | 22.15 us |
| Dense: K=1280, N=6848 | 77.62 us | 72.75 us |
| Routed: six experts, K=1280, N=896 each | 66.99 us | 63.25 us |

These approximately **6%** isolated improvements are not claims about total
request speed. The unchanged down kernel also showed allocation/run-order
variation in early probes; that was not counted as an optimization.

#### Numerical, filters and compatibility

- LM-head partial scores and IDs matched the baseline **bit-for-bit**, including
  vocabulary sizes 1/31/33/65/129,280; hidden sizes 64/896/1,280; ties within and
  across partials; excluded IDs; forced PAGE/EOS/other IDs; an out-of-range forced
  ID (all masked); a NaN row; and all-NaN activations. Small cases also used an
  independent FP64 oracle for scores/selection.
- A diagnostic backend compared **130 real Q4 LM-head invocations**, 65 each
  from base/gundam, including initial prefill and 64 decode steps. Every score
  and ID in all **4,040 partials per invocation** matched. Diagnostic waits and
  readbacks were excluded from timing.
- Production GEMVs matched baseline FP16 outputs bit-for-bit on nine synthetic
  shapes, including partial output rows, partial lane coverage, real shared/
  dense/routed dimensions, and invalid expert IDs. Metal API/GPU shader
  validation passed the LM-head edge cases and GEMV checks.
- Complete natural-EOS/strict-parsing checks matched baseline IDs:

| Path / fixture | Output tokens |
|---|---:|
| V2 Q4 invoice, base / 13-view / maximum 17-view gundam | 385 |
| V2 Q4 blank, base | 19 |
| V2 Q4 table, base | 448 |
| V2 Q4 edge text, gundam | 300 |
| V2 Q4 degraded scan, gundam | 385 |
| V2 Q4 tiny text, base | 1,172 |
| V1 Q4 invoice, base / gundam | 381 |
| V1 Q4 table, base | 444 |
| V2 Q8 invoice, base | 385 |

Q8 matched the observed baseline output set; the earlier baseline-variation
caveat still applies. Universal/compiler-independent bitwise equivalence and
upstream/full-v2-FP16 numerical parity are not established by these checks.

#### Whole-request acceptance and memory

Final production A/B measurements warmed both engines, alternated order over
four complete-request pairs and disabled detailed checkpoints/polling:

| V2 Q4 invoice | Before | After | Mean saving |
|---|---:|---:|---:|
| Base, 385 IDs | 3.478 s | 3.414 s | 64 ms / 1.8% |
| Gundam, 385 IDs | 7.454 s | 7.392 s | 62 ms / 0.8% |

Each final production pair improved. Subtracting the unchanged vision-stage
timer left **2.630 -> 2.579 s** base and **3.699 -> 3.650 s** gundam, supporting
about 50 ms of post-vision benefit rather than attributing vision timing noise
to the new kernels. A separate warmed tiny-text comparison changed **8.815 ->
8.677 s** with 1,172 identical IDs. An interleaved v1 table check also improved
on all four pairs, but its high timing variation is not a general v1 speed claim.
An earlier prototype gundam A/B run timed out; acceptance uses the subsequently
completed production runs, not its incomplete samples.

Encoder/wait/allocation counts remained **1,039 / 386 / 14** base and
**3,496 / 388 / 18** gundam. Tracked peaks remained **2,176,307,712 /
2,680,173,056 bytes**. Separate 1 ms Metal sampling of 65-token requests retained
peaks **2,205,302,784 / 2,735,915,008 bytes** and end usage **1,862,516,736 bytes**.
RSS high-water marks were approximately **2.142 -> 2.132 GB** base and
**2.227 -> 2.213 GB** gundam; these include Python/compiler/driver state and
are not the native admission estimate.

#### Failure, builds and cleanup

- Injected LM-head pipeline-lookup and post-dispatch failures released request
  allocations and allowed repeated reuse, followed by a complete matching
  385-ID request. Actual GPU execution faults were not injected. Admission
  estimates/scratch geometry did not change; budget rejection was not
  separately repeated in this checkpoint.
- Runtime-only compilation from generated source in an empty resource directory
  matched precompiled base/gundam outputs. A profiling-OFF, runtime-compilation-
  OFF precompiled build passed all 13 native checks and matched complete base
  output. Regeneration was byte-identical.
- Release build and existing suites passed: **105 Python checks, 7 opt-in
  skips**, the three expected warmup warnings, and **13 native checks**.
- Owned probes, private builds/libraries and logs under
  `data.tmp/probes/v2-q4-decode-tuning/` were removed after recording evidence.
  No permanent tests were added. Persistent stress data and concurrent notebook
  edits were not modified or staged by this work.

Reference: [5].

## Phase 9: DASH router/projection specialization

Outcome: lower router overhead on decode and a qualified path for larger block
counts, without changing the sparse routing algorithm.

- [x] done: Specialize single-token router Q/K projections and fuse router
  normalization/RoPE with preserved rounding.
- [ ] todo: Maintain incremental tail-summary accumulator state rather than
  recomputing the partial block, with explicit ownership/reset at boundaries.
- [x] done: Benchmark parallel block scoring and SIMD top-16 insertion against
  serial scoring/lane-zero insertion; qualify bounded score tiles.
- [x] done: Preserve top-16 entmax, finite masked sentinels, stable ties, FP32
  norm semantics, sanitization order and page-bucket eligibility. Do not use
  the upstream full-candidate routing approximation.

QA checkpoint:

- [x] done: Compare projections, norm/RoPE, selected blocks and biases against
  the native baseline and independent reference calculations.
- [ ] todo: Compare incremental summary state against full tail recomputation
  when that separate checkpoint is implemented. Summary code is unchanged here.
- [x] done: Exercise 32-token boundaries, masked candidates, ties, extremes,
  failure cleanup and reset/reuse; retain the Q8 caveat below.
- [x] done: Measure single-page benefit separately from synthetic long-context
  results. Multi-page generation/continuation remain unimplemented.

### Projection and routing checkpoint evidence (2026-10-31)

Baseline: `7168088`, Apple M1 Pro with 16 GB RAM, pinned models and Release/
profiling build. Saved baseline native code and metallib were used together.

#### Selected implementation and scope

- `uocr_dash_project_decode_f16` replaces two 1x1280x128 MPS projections with one
  dispatch: 128 threads, four SIMD groups/output rows per group, separate Q/K
  grid coordinates. Inputs/weights/outputs remain FP16 and accumulation FP32.
  Multi-token prefill projections remain on MPS.
- `uocr_dash_norm_rope_decode_f16` fuses 128-dimensional Gemma RMSNorm and
  router page RoPE for one-token appends. Normalized values explicitly round
  to FP16 before the existing separately rounded RoPE products. FP32
  `1 + weight`, epsilon, frequencies, page parity and later sanitization are
  unchanged. Main Q/K RoPE is unchanged.
- SIMD lanes 0..15 retain ordered top-16 entries. A SIMD minimum finds insertion
  rank and shuffles shift entries. Strict comparison inserts after equal
  scores, preserving earlier-block ties. Shared helpers retain the original
  FP32 top-M entmax and centered-prior schedule after gathering the ranks.
- Prefill/small contexts use one 32-lane SIMD group per query. Single-query
  routing with **more than 256 keys** uses eight SIMD groups / 256 threads.
  They score eight adjacent blocks and merge them in original block order.
  Only **32 bytes** of threadgroup score storage are added; both barriers are
  unconditional and incomplete tiles are bounded.
- These are **v2 DASH** paths, shared by mixed Q4/Q8 and FP16 dispatch. V1,
  model packing/conversion, page guards, retained K/V, summary ownership,
  sparse attention arithmetic and local-overlap rules are unchanged. No new
  persistent buffer or estimator change is introduced. Full v2 FP16 OCR is
  unverified.

#### Benchmark decisions

Small projection GPU timings varied with allocation/run order: roughly 9-20 us
for two MPS calls versus 8-12 us for the combined projection. Complete OCR was
the acceptance gate. A projection-only warmed base invoice changed **3.383 ->
3.307 s**, with matching IDs. Four interleaved base pairs comparing projection-
only against projection-plus-fused-norm/RoPE changed **3.288 -> 3.268 s**.
Four gundam pairs adding tiled scoring to SIMD insertion changed **6.795 ->
6.725 s**. These separate ablations are not additive speed estimates.

Serial lane-zero top-16 insertion was the larger bottleneck. Parallel scoring
alone retained much of its cost. SIMD insertion removed more; combining both
helped further. Isolated GPU sweeps used one warmup and three measured command
buffers, each with 128 dependent dispatches:

| Keys / blocks | Original | Parallel scores only | SIMD insertion | Tiled scores + SIMD insertion |
|---|---:|---:|---:|---:|
| 279 / 9 | 36.9 us | 35.1 us | about 9.9 us | 8.5 us |
| 1024 / 32 | 93.7 us | 81.0 us | about 20.8 us | 13.3 us |
| 2048 / 64 | 161.2 us | 138.0 us | about 37.1 us | 22.1 us |
| 8192 / 256 | 521.2 us | 433.7 us | about 126.9 us | 67.5 us |
| 131072 / 4096 | 7.36 ms | 5.94 ms | about 2.02 ms | 1.06 ms |

The table combines successive isolated sweeps. Long-context numbers do not
qualify document scheduling. Hardware instruction/register/occupancy counters
were not collected.

Tail-summary recomputation measured **3.0 / 6.5 / 10.7 / 14.0 / 18.7 us** for
1/8/16/24/32 entries. Incremental summaries remain **todo**: they need owned
FP32 accumulator/max/denominator state, reset/boundary rules, estimator updates
if storage is added, and separate failure qualification. Reconstructing state
from normalized summaries would change rounding. This stateful work is a
separate checkpoint.

#### Numerical and edge qualification

- Four synthetic projection cases matched MPS FP16 bits. Real comparisons
  covered **768 Q/K pairs each** from Q4 base, Q4 gundam and Q8 base (64 decode
  steps x 12 layers). Fused norm/RoPE covered the same counts. All **589,824
  FP16 values** in each comparison category matched the baseline.
- Production selected blocks and biases matched bit-for-bit across **27,228
  real query/layer outputs**: prefill plus 64 decode steps on those three paths.
  Diagnostic waits/readbacks were excluded from timing.
- A further complete Q8 gundam diagnostic compared **4,608 projection pairs**,
  **4,608 norm/RoPE pairs** and **22,836 query/layer routing outputs**. Every
  compared value matched on identical inputs, and the final 385 IDs matched
  baseline. This is not a universal/compiler-independent equivalence proof.
- Ninety synthetic routing cases covered 15 context sizes from 1 to 131,072
  keys, block boundaries, ties, same-page masks, no-same-page fallback,
  all-masked/finite sentinels, extreme negative scores and NaN/infinite queries.
  Both new paths matched baseline bits. IDs matched an independent FP64 top-M
  entmax oracle; priors used `2e-4` absolute/relative tolerance.
- Forty-eight norm/RoPE cases covered token positions 0/1/32/131071, page parity,
  zero inputs, maximum finite FP16 values, and NaNs/infinities. Outputs matched
  the separate kernels bit-for-bit.
- Metal API/GPU shader validation passed synthetic projection, norm/RoPE and
  routing checks and a 65-token production Q4 base request.

#### Complete OCR and Q8 caveat

Q4 complete baseline IDs, natural EOS and strict parsing matched for base/
13-view/17-view invoices (385 IDs), blank (19), table (448), edge text (300),
degraded scan (385) and tiny text (1172). V1 base/gundam invoices matched all
381 IDs. Q8 base invoices matched all 385 IDs.

**Q8 gundam is not claimed token-identical.** One initial candidate request
changed one coordinate from **0.211 to 0.210**, with the other 384 IDs unchanged.
Seventeen further baseline requests exhibited 385/386-ID baseline variation,
but did not reproduce that specific coordinate. Four later candidate requests
and the full diagnostic matched the 385-ID baseline. The isolated outlier's
cause is unresolved; it is recorded under the existing one-coordinate-unit
Q8 compatibility tolerance, not dismissed as proven baseline variation.
Deterministic Q8 output and upstream numerical parity remain unverified.

#### Production timing and memory

Final A/B runs warmed both engines, alternated order over four complete invoice
pairs, and disabled diagnostic checkpoints and memory polling:

| V2 Q4 invoice | Before | After | Saving |
|---|---:|---:|---:|
| Base, 385 IDs | 3.494 s | 3.117 s | 376 ms / 10.8% |
| Gundam, 385 IDs | 7.440 s | 6.734 s | 706 ms / 9.5% |

Every pair improved. Subtracting vision time gave **2.646 -> 2.287 s** base and
**3.701 -> 2.985 s** gundam. A separate warmed tiny-text check changed **8.670 ->
7.513 s**, with 1172 identical IDs. The blank-page check improved rather than
regressed.

Encoders/waits/buffer allocations remained **1039 / 386 / 14** base and
**3496 / 388 / 18** gundam. MPS NDArray/NSArray counts each fell by **9216** per
385-ID request, consistent with two fewer MPS calls per layer/decode step.
Tracked peaks stayed **2,176,307,712 / 2,680,173,056 bytes**. Separate 1 ms Metal
sampling of 65-token requests retained peaks **2,205,302,784 / 2,735,915,008
bytes** and end usage **1,862,516,736 bytes**. RSS high-water marks were about
**2.130 -> 2.130 GB** base and **2.220 -> 2.227 GB** gundam; these include driver,
compiler and Python state and are not the native admission estimate.

#### Failure, builds and cleanup

- Injected projection-pipeline lookup, post-routing dispatch and post-norm/RoPE
  dispatch failures cleaned up request allocations and allowed repeated reuse.
  Reuse also covered complete invoice, blank, capped invoice and complete
  invoice requests. Actual GPU execution faults were not injected.
- Runtime-only compilation from generated source in an empty resource directory
  matched precompiled base/gundam IDs. Regeneration was byte-identical.
- Release build and existing suites passed: **105 Python checks, 7 opt-in
  skips**, three expected warmup warnings, and **13 native checks**. Profiling-
  OFF and budget-rejection probes were not repeated; no memory estimate or
  persistent allocation changed.
- Owned probes, private native/shader libraries and logs under
  `data.tmp/probes/v2-dash-router/` were removed after recording evidence. No
  permanent tests were added. Persistent stress data and concurrent notebook
  edits were not modified or staged by this work.

See [v2 research notes](unlimitedocr2_research.md) for the routing oracle and
continuation distinctions.

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
