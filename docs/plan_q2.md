# Grounded Q2 implementation plan for `unlimitedocr.c` (decoder only)

> **Status: draft — nothing implemented.**  This plan follows the exact
> methodology that shipped Q8 and Q4: **sim gate → probe-first kernels →
> cfg flip → manual notebook QA → commit → tick the checkbox here**.  Every
> step below ends with a config flip and a user QA gate; no step's checkbox is
> ticked until QA passes on the real path.

This plan adds **2-bit storage for selected decoder weights** on top of the
shipped mixed-Q4 pipeline (v0.4.0, `docs/plan_q4.md`).  Scope is the
**decoder only**: vision and the projector are explicitly out of scope (the
vision encoder feeds everything downstream and already sits at Q4 with no
decode-speed benefit to gain), and attention Q/K/V/O stays at its current
qtype (the Q4 attention-O experiment was reverted; 2-bit attention is not on
the table).

## Why 2-bit is a different animal than Q4 (best practices)

Q8 and Q4_0 shipped as *symmetric, scale-only, group-64* formats and passed
QA essentially for free (sim rmse 0.0022 at 4 bits).  The literature and
llama.cpp practice are unambiguous that this recipe **does not survive at
2 bits**:

* **2 bits = 4 levels.**  A symmetric scale-only grid `{-2,-1,0,1}` wastes a
  level on asymmetric weight distributions and clips outliers brutally.
  Best practice (llama.cpp `Q2_K`, GPTQ/AWQ literature) is **asymmetric
  quantization (scale + min) over small sub-blocks**, with the sub-block
  scales/mins themselves quantized against per-super-block fp16 constants —
  a two-level hierarchy that keeps metadata near 0.56 bits/weight instead of
  16 bits per 16-weight group.
* **Small groups are mandatory.**  Our group-64 convention amortizes one fp16
  scale over 64 weights; at 2 bits the within-group dynamic range dominates
  the error.  `Q2_K` uses 16-weight sub-blocks inside 256-weight super-blocks
  (2.5625 bits/weight effective).  We should expect the same shape.
* **Not all tensors tolerate 2 bits.**  llama.cpp's own `Q2_K` preset keeps
  embeddings, output head and attention at higher precision and reserves
  2-bit for the FFN bulk.  Mapped to this model, the only defensible
  candidate is the **routed MoE experts**: 2112 tensors, ~80% of decoder
  parameters, and the architecture provides redundancy (top-6 of 64 routing
  plus a full-precision-path shared expert running for every token).
  LM head at 2 bits is specifically dangerous here because our LM head is a
  fused **argmax** — near-tie token selection amplifies weight error into
  discrete output changes.
* **Expect ALU-bound kernels, not bandwidth-bound.**  The Q4 LM-head lesson
  (plan_q4 E1) generalizes: halving bytes only halves time if the kernel is
  bandwidth-limited *and* the unpack stays off the critical path.  2-bit
  unpack (crumb extraction: 4 weights/byte, two shifts+mask each) plus
  sub-block scale/min lookups is markedly more ALU work per weight than the
  Q4 nibble split.  The honest expectation is **footprint is the win**
  (~0.8 GB off the file) and decode speed is at best a modest bonus —
  measure, don't assume.

## Context this plan builds on (shipped facts)

* Mixed-q4 model file is **1.83 GB**; routed expert Q4 qweights are **1.21 GB**
  of it.  At ~2.56 bits/weight the expert payload drops to ~0.39 GB →
  **file ≈ 1.0 GB**.
* Decode expert traffic is ~10.3 MB/token at Q4 (top-6 of 64); Q2 reads
  ~5.3 MB/token.  The Q4 expert GEMV measured 2.7× vs Q8 because the Q8
  kernel was *not* bandwidth-saturated; that headroom is spent now, so the
  speedup estimate is 1.1–1.4× — probe before promising anything.
* Qtype ids in use: `F16=1, F32=2, Q8_0=3, Q4_0=4, Q4_1=5 (reserved)`.
  Next free id: **`UOCR_TENSOR_Q2_K = 6`**.
* `uocr_tensor_entry` carries `scale_offset/scale_size` **and**
  `min_offset/min_size` (unused since Q4_1 was never implemented) — enough
  for a packed sub-scale array plus a per-super-block constants array with
  **no format-version bump**.
* Kernel conventions are normative in `docs/howto_quant.md`: two workload
  regimes only (simdgroup-per-row GEMV for decode §3, 64×32×32 tiled MMA GEMM
  for prefill §4), everything fused §1.3, probe in `/tmp` before landing §6.
* The Q4 packing lesson (howto_quant §7, plan_q4 §1.1): **probe the packing
  before freezing the format** — interleaved nibbles measured 0.9×, the
  group-half split 2.7×.  A 2-bit format has even more packing freedom
  (crumb order within bytes, sub-scale packing) and the same rule applies.
* The sim-gate tool from Q4 was deleted after it served its purpose; its
  methodology is recoverable from git: `git show 7ec8ca4:src/unlimitedocr_c/q4_sim.py`
  (quantize→dequantize expert tensors in a cloned fp16 `.uocr`, QA end-to-end
  in the notebook before writing any kernel).

## QA / rollout discipline (identical to Q4)

`configs/quant-cfg.yaml` stays the single source of truth.  For every module
step: **land kernels behind the qtype dispatch → flip the module's `qtype:` in
the shipped cfg → user reconverts and QAs manually in the notebook → commit →
tick the checkbox here.**  A regression rolls the cfg flip back (one-line
revert) without touching kernels.  `mixed-q8_0` and `mixed-q4` outputs must
stay byte-identical throughout.

---

## 1. Sim gate first (no kernels until this passes)

Resurrect the Q4 sim methodology and extend it to 2-bit candidates.  This is
the cheapest possible way to kill the idea if quality doesn't hold — hours,
not days.

Candidate schemes to simulate on the routed experts (2112 tensors):

```text
q2_0    symmetric,  scale-only,   group 16      (3.0  bpw effective) — strawman
q2_1    asymmetric, scale+min,    group 16      (4.0  bpw effective) — accuracy ceiling check
q2_k    asymmetric, 16-wide sub-blocks, 4-bit quantized sub-scales/mins,
        fp16 super scale/min per 256-super-block (2.5625 bpw) — the real candidate
```

Checklist:

* [ ] Recover the sim tool (`git show 7ec8ca4:src/unlimitedocr_c/q4_sim.py`),
      generalize to `q2_sim.py` with the three schemes above; report
      rmse / max_abs_err per scheme vs the Q4 baseline numbers
      (q4_0 rmse 0.00219 is the reference point).
* [ ] Produce a simulated `.uocr` per candidate (clone of the shipped
      mixed-q4 file with expert payloads quantized→dequantized in place).
* [ ] **Manual QA (user)**: notebook runs on the sensitive corpus (digits,
      det boxes, long documents) against fp16/q8/q4 baselines, per candidate.
* [ ] Record the verdict here.  **Gate:** if `q2_k` fails QA, stop — the
      fallback ladder (§8) starts at group-8 sub-blocks and ends at
      "don't ship Q2".
* [ ] Commit the sim tool + verdict; tick this section.

**Sim verdict:** _pending_.

---

## 2. Q2_K format and `.uocr` metadata

Frozen only after §1 passes and the §5 packing probe confirms the layout.
Working proposal (llama.cpp-inspired, adapted to our conventions):

```text
qtype:        UOCR_TENSOR_Q2_K (= 6)
super-block:  128 weights along K.  (llama.cpp uses 256, but the down
              projection has K = 896 = 7 × 128, not a multiple of 256;
              128 divides both expert K dims — 1280 and 896 — cleanly and
              costs only +0.19 bpw of super-const metadata.)
sub-block:    16 weights; 8 sub-blocks per 128-super-block
weights:      2 bits, unsigned crumbs q ∈ [0,3], w ≈ d * sq * q − m * sm
sub-scales:   4-bit sq and 4-bit sm per sub-block, packed 1 byte/sub-block
super consts: fp16 d (scale) + fp16 m (min) per super-block
effective:    2 + 8/16 + 32/128 = 2.75 bpw (super 128)  |  2.5625 bpw (super 256)
```

Tensor-entry mapping (existing fields only, no format bump):

```text
payload_offset/size = packed 2-bit crumbs        (rows * cols / 4 bytes)
scale_offset/size   = packed 4-bit sub scales+mins (rows * cols/16 bytes)
min_offset/size     = fp16 super d,m pairs       (rows * cols/SUPER * 4 bytes)
block_size          = 16 (sub-block)
row_size            = cols / 4 (packed crumb bytes per row)
```

Checklist:

* [ ] `UOCR_TENSOR_Q2_K = 6` in `uocr_format.h` + converter constants +
      `uocr_tensor_qtype_name()`.
* [ ] Byte-count helpers (crumb bytes, sub-scale bytes, super-const bytes).
* [ ] Q2_K tensor-entry validator in `uocr_model_file.c`, **restricted to
      `MOE_EXPERT` gate/up/down** (family + projection), mirroring how Q4
      started; loosen per module only if extensions ever land.
* [ ] Loader tests (`tests/test_uocr_model_file.c`): valid entry, bad family,
      bad row_size, bad sub-scale size, min-array size mismatch.
* [ ] Commit; tick.

---

## 3. quant-cfg + converter

Checklist:

* [ ] `quant_cfg.py`: add `q2_k` to `QUANT_CFG_QTYPES`;
      `Q2_ALLOWED = {(MOE_EXPERT, GATE/UP/DOWN)}` (per-projection gating like
      `Q4_ALLOWED_PROJECTIONS`); schema errors for anything else.
* [ ] New qprofile `UOCR_QPROFILE_MIXED_Q2 = 4`, CLI `--qprofile mixed-q2`;
      planning = mixed-q4 planning with expert modules re-typed by cfg.
* [ ] `_quantize_q2_k_rows()` + `_stream_bf16_to_q2_k()` writer: per
      super-block fp16 d/m from sub-block scale/min extrema, 4-bit quantized
      sub-scales/mins, crumb packing (exact packing frozen by the §5 probe).
* [ ] Round-trip unit tests: pack/unpack identity, all-zero and all-equal
      sub-blocks, negative-only sub-blocks, layout offsets, planned vs
      written byte counts, dequant rmse matches the sim within tolerance.
* [ ] Expert interleaved slab layout: same `[expert][gate,up,down]` contract,
      now three side arrays (crumbs, sub-scales, super-consts) with per-slab
      strides — extend the slab math the way Q4 did (packed-byte strides).
* [ ] `mixed-q8_0`/`mixed-q4` plans byte-identical (regression assert in
      tests).
* [ ] Commit; tick.

---

## 4. Loader and Metal bindings

Checklist:

* [ ] Payload spans: map crumb + sub-scale + super-const ranges (first real
      user of `min_offset/min_size` → new third buffer/offset in the decoder
      binding struct).
* [ ] `metal_decoder_tensor_allows_q2()`: `MOE_EXPERT` gate/up/down only.
* [ ] Expert slab cache: Q2 strides; per-layer and cross-layer mixed-slab
      guards (a layer's experts are all Q2_K or all Q4_0/Q8_0, never mixed).
* [ ] Engine open accepts `UOCR_QPROFILE_MIXED_Q2` **only in the step-7
      commit** (fail loudly until kernels exist — same deferral Q4 used).
* [ ] Commit; tick.

---

## 5. Metal kernels (probe-first, per `docs/howto_quant.md` §6)

### 5.1 Packing probe — before freezing §2

The Q4 rule, applied harder: 2-bit unpack cost can eat the entire bandwidth
win.  Probe in `/tmp` (extend `/tmp/bench_moe_decode_gemv_q4.m`) at the real
expert shapes, comparing at minimum:

* crumb order: group-quarter split (byte j holds w[j], w[j+S/4], w[j+2S/4],
  w[j+3S/4] — the 2-bit analogue of the Q4 group-half split, one shift per
  `float4` extraction) vs sequential crumbs;
* sub-scale application: per-16 multiply inside the lane loop vs
  pre-multiplying `d*sq` into a small per-super register cache;
* load width per lane (`uchar4` = 16 weights vs `ushort4`/`uint2` = 32).

Checklist:

* [ ] `/tmp/bench_moe_decode_gemv_q2.m`: q2 candidates vs the shipped Q4 GEMV
      (bit-exactness against a CPU reference dequant, then timing).
* [ ] Record the winner + numbers here; freeze §2 packing accordingly.
* [ ] **Go/no-go:** if the best q2 GEMV is ≤1.0× vs Q4, Q2 ships as a
      *footprint-only* feature — that's acceptable, but it must be a
      deliberate, recorded decision, not a surprise.

**Probe verdict:** _pending_.

### 5.2 Decode GEMV (`decode_gemv_q2.metal`)

* [ ] `uocr_decode_gemv_q2_lane_dot` helper (crumb unpack + hierarchical
      scale, per the probe winner).
* [ ] `uocr_moe_decode_gate_up_gemv_q2_k_to_f16`,
      `uocr_moe_decode_down_combine_gemv_q2_k_to_f16` — same grids, fused
      SwiGLU / weighting / combine / residual as the Q8/Q4 twins.
* [ ] Dispatch routing by expert slab qtype (F16 | Q8_0 | Q4_0 | Q2_K).

### 5.3 Prefill bucketed GEMM (`gemm_q2.metal`)

* [ ] `uocr_gemm_q2_stage_b()`: dequant-on-stage into the shared `[BK][BN]`
      fp16 tile; BK=32 must stay inside one sub-scale run — with 16-wide
      sub-blocks each staged 8-run covers exactly half a sub-block, so one
      `d*sq`/`m*sm` pair per run (verify against the frozen packing).
* [ ] `uocr_moe_prefill_bucketed_swiglu_gate_up_gemm_q2_k_to_f16`,
      `uocr_moe_prefill_bucketed_down_gemm_q2_k_to_f16` — reuse
      `uocr_gemm_q8_mma_tile` and the qtype-agnostic bucketing/combine passes.
* [ ] Dispatch routing by slab qtype.

### 5.4 Build integration

* [ ] `decode_gemv_q2.metal` + `gemm_q2.metal` in `tools/gen_metal.py` `ORDER`
      (after their q4 siblings); regenerate `uocr_smoke.metal`;
      `ctest` 13/13; commit; tick.

---

## 6. Public API

Checklist:

* [ ] `UnlimitedOCR(quant="q2")` → cached `unlimitedocr-q2.uocr`
      (`--qprofile mixed-q2` on miss), mirroring `q4`.
* [ ] No C API changes (qprofile inferred from the file).
* [ ] Commit; tick.

---

## 7. Rollout: cfg flip + manual QA (the gate)

Checklist:

* [ ] Engine-open acceptance of `mixed-q2` lands in this commit (not before).
* [ ] Flip `moe_routed_experts: qtype: q2_k` in the shipped
      `configs/quant-cfg.yaml` **under the mixed-q2 profile only** —
      `mixed-q4` keeps experts at `q4_0` (per-profile qtype resolution,
      mirroring how `mixed-q8_0` ignores the q4 overrides today).
* [ ] **Manual QA (user)**: `UnlimitedOCR(quant="q2", force_reconvert=True)`
      on the sensitive corpus vs fp16/q8/q4 baselines; extra attention to
      digits and rare tokens (expert error is the whole delta here).
* [ ] Record file size + decode timings vs mixed-q4 here.
* [ ] Commit; tick.  **Q2 ships expert-only.**

---

## 8. Fallback ladder (if §1 sim or §7 QA regresses)

Cheapest first:

1. **Sub-block 8** (halve the sub-block; +0.5 bpw of sub-scale metadata,
   ~3.25 bpw effective — still ~20% smaller than Q4).
2. **Down projection stays Q4_0**, gate/up Q2_K (per-projection qtype is
   already representable; down is the error-summing projection).
3. **Importance-weighted quantization** (imatrix-style: weight the per
   sub-block fit by mean activation magnitude collected from a calibration
   run) — significant converter work; only if 1–2 fail and the footprint
   still justifies it.
4. **Don't ship Q2.**  The mixed-q4 profile at 1.83 GB stays the floor;
   record the numbers and close the plan.  Shipping nothing is an acceptable
   outcome of a probe-first methodology.

---

## Explicitly out of scope

* **Vision / projector**: prefill-compute-bound, quality-critical input
  stage; stays Q4/fp16.
* **Attention (all projections)**: reverted at Q4; not a Q2 candidate.
* **LM head**: fused argmax makes 2-bit weight error a token-selection
  hazard; stays Q4_0.
* **Embeddings, shared experts, dense MLP**: run for every token with no
  routing redundancy; revisit only if expert Q2 passes full-corpus QA with
  margin, and then only through their own sim gate first.

## Deliverable

```text
Model:
  mixed .uocr with qprofile UOCR_QPROFILE_MIXED_Q2
  routed experts (gate/up/down, 2112 tensors)  Q2_K (~2.6–2.75 bpw)
  everything else identical to shipped mixed-q4
  file size ~1.0 GB (from 1.83 GB mixed-q4)

Runtime:
  fused Q2 decode GEMV + bucketed prefill GEMM at the routed-expert call
  sites; no dequantized weight buffers; decode speed >= mixed-q4 (probe-
  verified) or a recorded footprint-only decision.

User API:
  UnlimitedOCR(quant="q2")

Reporting:
  sim rmse table, packing-probe table, per-qtype byte totals, decode
  tokens/s vs mixed-q4.
```
