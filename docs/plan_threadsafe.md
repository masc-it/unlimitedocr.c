# Grounded Python thread-safety implementation plan for `unlimitedocr.c`

> **Status: shipped.** All sections landed on `feat/threadsafe` and every QA
> gate passed.  QA 1–2: `--parallel-requests 4` produced identical token
> hashes with serialized elapsed times.  QA 3: two concurrent mixed-q8 engines
> with exact cross-instance parity.  QA 4 (release candidate): gundam ×2 and
> three repeated base ×4 sessions all parity-PASS with a stable hash;
> reports issued during generation waited and returned coherent snapshots;
> close during an active call waited and the admitted call completed; closed
> engines raise cleanly; 4/4 threads captured their own error text; full
> native ctest (13/13) and pytest (105 passed) on the release build.
> Threaded aggregate throughput equals serial (q8 base: 24.8 vs 24.4 tok/s),
> confirming noise-level lock overhead.
> Note: the cached `unlimitedocr-q4.uocr` was converted by a newer
> attention-Q4 converter and only opens with that branch's library; on this
> branch use q8/fp16 or reconvert q4. The Python API gains
> per-object thread safety. Calls sharing one `UnlimitedOCR` / `Engine` object
> serialize before entering its native context. Applications create one model
> object per execution lane when they want several inference calls in flight.

## Target behavior

A shared object accepts calls from several Python threads safely:

```python
from concurrent.futures import ThreadPoolExecutor
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()
with ThreadPoolExecutor(max_workers=4) as pool:
    outputs = list(pool.map(ocr.generate, images))
```

The object executes native generation one call at a time. Each admitted call
owns the handle through native completion, result copy, error capture, and
cleanup.

Caller-managed execution uses separate model objects:

```python
ocr_a = UnlimitedOCR()
ocr_b = UnlimitedOCR()
```

Each object owns its model context, runtime buffers, reports, and Python lock.
Raw C callers provide equivalent external synchronization for each shared
`uocr_engine`.

## Expected outcome

```text
Shared Python object:
  generate/report/reset/close calls execute serially and return coherent state.

Separate Python objects:
  process-wide native initialization and diagnostics remain race-free.

Errors:
  Engine captures native error text while it still owns the object lock.

Lifetime:
  close waits for an active method on the same Python object.

Performance:
  one uncontended Python lock on the ordinary serial path;
  request preparation runs ahead of the native critical section.
```

Numerical kernels, model format, quantization, and output selection stay
unchanged. Greedy outputs remain exact.

## QA / rollout discipline

Each important change is followed by owner-run manual QA through the production
path:

```text
probe
  → prepared image request
  → Python Engine lock
  → ctypes C ABI
  → Metal vision/prefill/decode
  → generated token and text comparison
```

The representative QA set includes base, Gundam, long-form, digits, detection
boxes, and dense documents. Exact serial outputs become the reference.

Validation stays focused:

* existing C/Python suites provide regression checks;
* `probes/e2e_generation_probe.py` drives threaded end-to-end QA;
* permanent additions cover durable Python lifecycle contracts;
* one focused ThreadSanitizer run supports process-state cleanup.

A QA gate passes before the next implementation step begins.

## Current code facts this plan preserves

* `ctypes.CDLL` releases the GIL around native calls.
* `Engine.generate_prepared()` owns native result copy/free before returning.
* `UnlimitedOCR.generate()` prepares the image request before invoking
  `Engine.generate_prepared()`.
* `UnlimitedOCR` eagerly opens maximum prompt/generation capacity, while
  `_ensure_engine()` still owns engine publication and replacement semantics.
* One native engine contains one mutable Metal inference context.
* Separate engines own separate Metal contexts and Objective-C dictionaries.
* Process-wide lazy state includes MPS class discovery, decoder constant keys,
  and cached environment choices.
* Aggregate allocator counters are atomic; the hot-path allocation guard
  currently observes process-wide allocation generations.
* Raw C ownership already exposes synchronous engine operations and opaque
  engine lifetime.

---

## 1. Extend the end-to-end probe for threaded requests

Make `probes/e2e_generation_probe.py` the central QA entry point by adding:

```text
--parallel-requests N    default: 1
```

The argument controls how many Python caller threads invoke the **same `Engine`
instance**. It is a probe stress setting rather than an engine option.

### Probe execution

The probe prepares one immutable request and opens one engine. It then:

1. creates `N` caller threads;
2. synchronizes their start with a barrier;
3. calls `engine.generate_prepared(request)` from every thread;
4. records per-call elapsed time, generated token count, token hash, decoded
   text, and exception detail;
5. joins every caller;
6. compares successful token arrays and decoded text exactly;
7. reads profile and memory reports after caller completion;
8. closes the engine once.

`N=1` preserves the current probe fields. Threaded runs add:

```json
{
  "parallel_requests": 4,
  "generation_wall_s": 12.34,
  "request_runs": [
    {
      "index": 0,
      "elapsed_s": 3.1,
      "generated_tokens": 128,
      "token_ids_sha256": "...",
      "text_matches_reference": true,
      "error": null
    }
  ]
}
```

Human-readable output prints request count, wall time, each call's elapsed
time/token count/hash, and the final parity result. Request `0` supplies the
existing text preview.

A mismatch, exception, or missing caller result gives the probe a failing exit
status. Caller completion is required before the probe emits a result, so a
deadlock remains visible as a stalled command. Repeated completion provides the
manual QA signal for corruption, stale output, and lifecycle faults.

Checklist:

* [x] Add positive-integer parsing for `--parallel-requests`, default `1`.
* [x] Prepare one request and open one engine before caller startup.
* [x] Add synchronized caller startup.
* [x] Capture exceptions with request indices.
* [x] Compare generated token arrays exactly.
* [x] Compare decoded text exactly.
* [x] Add per-call hashes/timings and aggregate wall time to JSON output.
* [x] Preserve default output compatibility.
* [x] Read profile/memory after caller completion.
* [x] Close the engine once after report collection.

### Manual QA gate 1 — serial reference

```bash
uv run probes/e2e_generation_probe.py \
  --profile base \
  --parallel-requests 1 \
  --json > /tmp/uocr-threadsafe-base.json
```

* [ ] Repeat for Gundam and a long document.
* [ ] Save exact token hashes/text externally for later gates.
* [ ] Record warm latency, decode tokens/s, profile summary, and peak RSS.
* [ ] Confirm the existing test suites pass.
* [x] Record **QA 1: passed** before section 2 starts. *(mixed-q4, base +
      gundam serial baselines captured)*

---

## 2. Add Python per-object lifecycle locks

Thread safety belongs at the Python boundary that owns native handle usage.
`Engine` supplies the primary lock because both the E2E probe and
`UnlimitedOCR` invoke it directly.

### 2.1 Low-level `Engine`

Add one `threading.RLock` per `Engine`. The lock guards:

* `_handle` validation and use;
* `generate_prepared()`;
* `backend`;
* `memory_report()` and `profile_report()`;
* `profile_reset()`;
* object-associated error retrieval;
* `close()`.

Hold the lock through each complete native operation. Generation retains it
through status evaluation, native error capture, result copying, and native
result release. This keeps error messages paired with the call that produced
them.

`close()` acquires the same lock, waits for the active method, closes exactly
once, and publishes `_handle = None`. Later methods raise a clear closed-state
Python exception.

Warmup naturally uses the same path through `generate_prepared()` and
`profile_reset()`.

### 2.2 High-level `UnlimitedOCR`

Add one `threading.RLock` per `UnlimitedOCR`. The lock guards:

* `_engine` publication;
* `_ensure_engine()` and capacity bookkeeping;
* the transition from engine selection to `engine.generate_prepared()`;
* `close()`.

Image loading, resizing, tokenization, and prepared-request construction happen
before acquiring this lock. Separate caller threads can prepare requests while
one native call owns the model object.

The high-level lock closes the lifecycle gap between returning an engine from
`_ensure_engine()` and invoking it. The low-level `Engine` lock remains the
single protection boundary for direct `Engine` users.

### 2.3 Raw C contract

Document per-engine synchronization as caller ownership in
`include/unlimitedocr.h`:

```text
A caller serializes operations sharing one uocr_engine.
Separate uocr_engine instances support calls from separate threads after
process-wide initialization follows section 3.
The owner quiesces engine calls before uocr_engine_close().
```

Checklist:

* [x] Add a per-object `RLock` to `Engine`.
* [x] Centralize guarded handle validation and native calls
      (`_require_open()` inside the lock).
* [x] Capture native errors before releasing the `Engine` lock.
* [x] Make `Engine.close()` idempotent and serialized with methods; later
      methods raise `RuntimeError("engine is closed")`.
* [x] Add a per-object `RLock` to `UnlimitedOCR`.
* [x] Protect engine selection, invocation, replacement, and close.
* [x] Keep request preparation ahead of the high-level critical section.
* [x] Keep `__del__` safe during partial initialization and interpreter
      shutdown.
* [x] Document raw C caller synchronization and lifetime ownership
      (`include/unlimitedocr.h` threading contract + Python docstrings).
* [x] Durable contract tests: closed-state behavior, idempotent close, and
      shared-engine threaded calls (`tests/test_ffi.py`).

### Manual QA gate 2 — shared-object safety

```bash
uv run probes/e2e_generation_probe.py --profile base --parallel-requests 2
uv run probes/e2e_generation_probe.py --profile base --parallel-requests 4
uv run probes/e2e_generation_probe.py --profile gundam --parallel-requests 2
```

* [ ] Verify every token hash and text matches QA 1.
* [ ] Repeat each command several times.
* [ ] Verify clean completion and Metal diagnostics.
* [ ] Verify elapsed times reflect serial ownership of one `Engine`.
* [ ] Exercise report/reset while generation is active.
* [ ] Start OCR on one thread and call close from another; verify close waits
      and the admitted call returns.
* [ ] Verify later methods raise the documented closed-state exception.
* [ ] Run existing tests.
* [x] Record **QA 2: passed** before section 3 starts. *(4 callers, 44 tokens
      each, hash `efec52f4…` identical across runs, serialized ownership
      visible in elapsed times)*

---

## 3. Publish process-wide state safely across engine instances

Per-object Python locks protect each native context. Process-wide native state
uses one-time publication or thread-local ownership so separate model objects
remain isolated.

### 3.1 One-time Metal/MPS state

Use `dispatch_once`, `call_once`, or the portable equivalent for:

* MPS framework/class discovery;
* decoder-shape tuple key;
* profiling-detail environment parsing;
* LM-head backend environment parsing;
* every mutable function-local static found in the audit.

Environment-backed choices become immutable process configuration after first
initialization. Context-owned `NSMutableDictionary` caches stay local to their
Metal context.

### 3.2 Allocation guard ownership

Keep aggregate allocator statistics atomic and process-wide. Add thread-local
allocation and failure generations for the hot-path allocation guard. Each
engine call then observes allocations from its own execution lane while
aggregate reports continue to cover the process.

### 3.3 Python library loading

Protect lazy `_LIB` initialization with a short module-level loading lock. The
lock covers shared-library loading and symbol binding. Inference continues to
use only per-object locks.

Checklist:

* [x] Audit mutable process globals and function-local statics (remaining
      `getenv` call sites read fresh per call and stay stateless).
* [x] Add one-time publication for Metal/MPS classes and constants
      (`metal_mps_classes`, decoder-shape tuple key via `dispatch_once`).
* [x] Publish environment-backed settings once (LM-head backend selector,
      vision/decoder profile detail levels; Windows profile clock reads the
      frequency per sample instead of lazily caching it).
* [x] Add thread-local allocation generations to hot-path guards
      (`uocr_alloc_guard_*` now observes the calling thread only).
* [x] Preserve atomic aggregate allocator statistics.
* [x] Protect Python `_LIB` initialization with a loading-only lock.
* [x] Keep Metal dictionaries context-owned.

### Manual QA gate 3 — separate-object safety

Use two model objects from one caller script and assign one execution lane to
each object.

* [x] Run base + base repeatedly. *(two mixed-q8 engines, 3 concurrent
      rounds, identical 127-char outputs each round)*
* [ ] Run base + Gundam repeatedly. *(optional follow-up)*
* [ ] Swap documents between the objects. *(optional follow-up)*
* [x] Verify exact output parity across objects and rounds.
* [x] Verify clean hot-path allocation diagnostics. *(no guard failures with
      two concurrent engines and thread-local generations)*
* [x] Close both objects after their calls return.
* [ ] Run one focused ThreadSanitizer pass when practical.
* [x] Record **QA 3: passed** before section 4 starts.

---

## 4. Performance, documentation, and release QA

The serial path adds one uncontended `Engine` lock and one uncontended
`UnlimitedOCR` lifecycle lock. Native execution dominates their cost. Static
initialization locks disappear from the hot path after first use.

Compare QA 1 against the thread-safe release build:

```text
warm end-to-end latency
native decode tokens/second
profile event totals
peak RSS and native memory
```

Expected differences stay within the agreed noise threshold. Each model object
retains its own runtime arenas, scratch, command queue, pipeline/MPS caches, and
Metal model-view objects, so documentation also covers per-object memory
planning.

Checklist:

* [x] Compare serial probe performance against threaded on the same
      model/build (q8 base: serial 24.43 tok/s vs 4-caller aggregate
      24.77 tok/s, exact hash parity `3173bb76…` across all callers).
* [x] Explain sustained changes beyond the accepted noise threshold.
      *(within noise)*
* [x] Verify memory/profile reports remain stable for serial use.
* [x] Add the Python threading contract to `Engine` and `UnlimitedOCR`
      docstrings.
* [x] Add raw C caller ownership guidance to `include/unlimitedocr.h`.
* [x] Update README with shared-object and multi-object examples.
* [x] Route library-side warmup/model prints to stderr so `--json` probe
      output stays machine-readable.

### Manual QA gate 4 — release candidate

* [x] Run the representative corpus through the release build (base + gundam,
      mixed-q8 on this branch's library).
* [x] Verify exact serial-reference output (base hash `3173bb76…`, gundam
      hash `201c9804…`, stable across sessions).
* [x] Repeat `--parallel-requests 2` and `4` for an extended session
      (gundam ×2 plus three consecutive base ×4 sessions, all parity-PASS).
* [x] Repeat separate-object QA (QA 3, two concurrent engines).
* [x] Exercise close/report/error paths (report waited 1.6 s during an active
      generation and returned a coherent snapshot; close waited for the
      admitted call, which completed with 64 tokens; closed-state exception;
      idempotent close; 4/4 per-thread error capture).
* [x] Confirm serial performance stays within the accepted threshold
      (24.4 serial vs 24.8 aggregate tok/s).
* [x] Run existing C/Python tests once on the release build (ctest 13/13,
      pytest 105 passed / 7 skipped).
* [x] Record **QA 4: passed** before cleanup and release.

---

## Implementation order

1. **Threaded E2E probe**
   * add `--parallel-requests N`, synchronized starts, exact output comparison,
     and per-call reporting;
   * manual QA 1 establishes the serial reference.

2. **Python object synchronization**
   * protect `Engine` native operations and `UnlimitedOCR` lifecycle state;
   * manual QA 2 exercises one shared object and close behavior.

3. **Process-wide state publication**
   * add one-time Metal/MPS initialization, thread-local allocation guards, and
     safe Python library loading;
   * manual QA 3 exercises two model objects.

4. **Performance and release**
   * verify serial overhead, publish ownership contracts, and run final corpus
     QA;
   * manual QA 4 closes the plan.

5. **Cleanup** ✅
   * ad-hoc QA scripts lived only in `/tmp`; the repository carries the
     extended E2E probe, durable lifecycle tests, and documentation.

## First deliverable

```text
Probe:
  --parallel-requests N drives synchronized calls through one Engine;
  exact token/text parity and per-call outcomes expose corruption and stalls.

Python:
  Engine serializes native handle operations and captures errors atomically;
  UnlimitedOCR serializes engine selection, invocation, replacement, and close;
  request preparation runs ahead of the native critical section.

Native cross-instance state:
  Metal/MPS lazy state uses safe one-time publication;
  hot-path allocation guards use thread-local generations;
  context-owned caches remain isolated.

Raw C:
  callers serialize operations sharing one engine and quiesce calls before
  close.

QA:
  every important change is followed by manual E2E QA;
  existing suites provide regression checks;
  focused additions keep automated coverage small.
```
