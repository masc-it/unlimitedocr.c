# V2 mixed-Q4 optimization plan

Status: `todo` (not done)

Checkboxes: `- [x] done` means completed and verified; `- [ ] todo` means not
done. Initial research is complete. Phase 1's single-token submission and
encoder-consolidation checkpoint is implemented and verified. GPU decode
chunking is deferred after an unfavorable prototype tradeoff. Phase 2's shared
SAM relative-logit storage, shared local/global backing storage, and batch-abort
ownership fixes are implemented. Four-crop batches are qualified on the measured
M1 Pro. Remaining staging/cached-buffer accounting is not done.

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
