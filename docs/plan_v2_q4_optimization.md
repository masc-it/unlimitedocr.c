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
kernel are implemented and verified. CLIP reuse, matrix-tiled sparse prefill,
and remaining staging/cached-buffer accounting are not done.

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
