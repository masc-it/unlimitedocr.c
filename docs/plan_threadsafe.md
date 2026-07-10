# Grounded Python thread-safety implementation plan for `unlimitedocr.c`

> **Status: in progress.** The threaded E2E probe is implemented on
> `feat/threadsafe`; manual QA gate 1 is pending. The Python API gains
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
* [ ] Record **QA 1: passed** before section 2 starts.

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

* [ ] Add a per-object `RLock` to `Engine`.
* [ ] Centralize guarded handle validation and native calls.
* [ ] Capture native errors before releasing the `Engine` lock.
* [ ] Make `Engine.close()` idempotent and serialized with methods.
* [ ] Add a per-object `RLock` to `UnlimitedOCR`.
* [ ] Protect engine selection, invocation, replacement, and close.
* [ ] Keep request preparation ahead of the high-level critical section.
* [ ] Keep `__del__` safe during partial initialization and interpreter
      shutdown.
* [ ] Document raw C caller synchronization and lifetime ownership.

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
* [ ] Record **QA 2: passed** before section 3 starts.

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

* [ ] Audit mutable process globals and function-local statics.
* [ ] Add one-time publication for Metal/MPS classes and constants.
* [ ] Publish environment-backed settings once.
* [ ] Add thread-local allocation generations to hot-path guards.
* [ ] Preserve atomic aggregate allocator statistics.
* [ ] Protect Python `_LIB` initialization with a loading-only lock.
* [ ] Keep Metal dictionaries context-owned.

### Manual QA gate 3 — separate-object safety

Use two model objects from one caller script and assign one execution lane to
each object.

* [ ] Run base + base repeatedly.
* [ ] Run base + Gundam repeatedly.
* [ ] Swap documents between the objects.
* [ ] Verify exact QA 1 token hashes/text.
* [ ] Verify clean hot-path allocation diagnostics.
* [ ] Close both objects after their calls return.
* [ ] Run one focused ThreadSanitizer pass when practical.
* [ ] Record **QA 3: passed** before section 4 starts.

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

* [ ] Compare serial probe performance before/after on the same model/build.
* [ ] Explain sustained changes beyond the accepted noise threshold.
* [ ] Verify memory/profile reports remain stable for serial use.
* [ ] Add the Python threading contract to `Engine` and `UnlimitedOCR`
      docstrings.
* [ ] Add raw C caller ownership guidance to `include/unlimitedocr.h`.
* [ ] Update README with shared-object and multi-object examples.

### Manual QA gate 4 — release candidate

* [ ] Run the complete representative corpus through the release build.
* [ ] Verify exact QA 1 output.
* [ ] Repeat `--parallel-requests 2` and `4` for an extended session.
* [ ] Repeat separate-object QA.
* [ ] Exercise close/report/error paths.
* [ ] Confirm serial performance stays within the accepted threshold.
* [ ] Run existing C/Python tests once on the release build.
* [ ] Record **QA 4: passed** before cleanup and release.

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

5. **Cleanup**
   * remove temporary logs, timing hooks, and ad-hoc QA scripts;
   * retain the extended E2E probe and durable lifecycle checks.

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
