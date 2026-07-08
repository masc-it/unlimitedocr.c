# Grounded Q4 implementation plan for `unlimitedocr.c`

> **Status: shipped.**  All seven steps landed on `feat/more-quant` and the
> mixed-q4 model passed end-to-end notebook QA.  Actuals: model file
> **2.24 GB** (from 3.45 GB mixed-q8_0), routed-expert decode GEMV measured
> **~2.7× vs Q8** on M1 Pro (group-half-split packing; the originally planned
> interleaved nibble packing measured 0.9× and was replaced — see §1.1).
> The q4_sim quality-gate tool served its purpose and was deleted in step 7.

This plan adds **int4 (Q4_0) storage for the routed MoE expert weights** on top
of the shipped mixed-Q8 pipeline.  Scope is the **decoder only**, and inside the
decoder only `TensorFamily.MOE_EXPERT` gate/up/down: the experts hold ~2.42 B of
the ~3 B decoder parameters (~80%), so they are the only module where 4-bit pays
for its risk.  Everything else keeps its current qtype: attention, dense MLP,
shared experts, embeddings, and LM head stay Q8_0; router, norms, and vision
stay fp16.

## Why this is justified (quality gate already passed)

The simulation gate ran before this plan (`src/unlimitedocr_c/q4_sim.py`,
branch `feat/more-quant`): expert weights were quantized→dequantized in a clone
of the fp16 `.uocr` (2112 tensors, group 64) and QA'd end-to-end through the
notebook.

```text
scheme  rmse       max_abs_err   end-to-end QA
q4_0    0.00219    0.065         pass (output parity with fp16 baseline)
q4_1    0.00206    0.054         pass
```

**Decision: ship Q4_0 (symmetric, scale-only) as the storage scheme.**  It
passed QA, matches the Q8_0 metadata conventions (scale only, no min), and
keeps the kernel inner loop cheapest.  Q4_1 (scale+min) is the documented
fallback if full-corpus QA on the real Q4 path regresses; the format reserves
its qtype id and the tensor entry already has `min_offset/min_size` fields.

The simulation tool was deleted at the end of this plan (step 7) — the gate
passed and Q4 landed properly.

## Expected wins

```text
model file:  3.45 GB (mixed-q8_0) → ~2.26 GB   (expert qweights 2.42 GB → 1.21 GB)
decode:      routed expert GEMV traffic halves; Q8 measured 0.11–0.13 ms/layer
             on M1 Pro → est. 0.06–0.08 ms (bandwidth-bound, near-peak already)
prefill:     bucketed expert GEMM B-tile staging bytes halve
load/page-in: ~1.2 GB less cold-start I/O
```

## QA / rollout discipline

Same as the Q8 rollout: `configs/quant-cfg.yaml` is the single source of truth
for what the runtime may consume.  Q4 for `moe_routed_experts` only flips on
after the fused Q4 kernels pass an end-to-end OCR run against both the fp16 and
mixed-q8_0 baselines on representative inputs (digits, det boxes, long
documents — the sensitive cases from the sim QA).

## Current code facts this plan must preserve

* Qtypes today: `UOCR_TENSOR_F16 = 1`, `UOCR_TENSOR_F32 = 2`, `UOCR_TENSOR_Q8_0 = 3`
  (`src/model/uocr_format.h`, mirrored in `src/unlimitedocr_c/convert.py`).
* Qprofiles today: `UOCR_QPROFILE_FP16 = 1`, `UOCR_QPROFILE_MIXED_Q8_0 = 2`.
* `uocr_tensor_entry` already carries `qtype`, `block_size`, `row_size`,
  `scale_offset/scale_size`, **and unused `min_offset/min_size`** — no new
  directory structure is needed for Q4_0 or a future Q4_1.
* `configs/quant-cfg.yaml` (schema version 1) flags modules `supported: true`
  for Q8_0; there is no per-module qtype yet.
* Q8 kernels are structured as focused sibling fragments:
  `embedding_q8.metal`, `lm_head_q8.metal`, `attention_q8.metal`,
  `dense_q8.metal`, `decode_gemv_q8.metal`, `gemm_q8.metal`, `moe_q8.metal`.
  Q4 follows the same convention with `*_q4.metal` fragments.
* Routed-expert execution paths that must gain Q4 variants:
  * decode: `uocr_moe_decode_gate_up_gemv_q8_0_to_f16` and
    `uocr_moe_decode_down_combine_gemv_q8_0_to_f16`
    (`decode_gemv_q8.metal`, simdgroup-per-row GEMV, 128 threads);
  * prefill: `uocr_moe_prefill_bucketed_swiglu_gate_up_gemm_q8_0_to_f16` and
    `uocr_moe_prefill_bucketed_down_gemm_q8_0_to_f16`
    (`gemm_q8.metal`, 64×32×32 simdgroup-MMA tiles, expert-bucketed pairs);
  * bucketing/combine kernels (`uocr_moe_prefill_bucket_pairs_*`,
    `uocr_moe_prefill_bucketed_combine_*`) are qtype-agnostic and unchanged.
* Expert weights live in interleaved expert-major slabs; the Q8 slab contract is
  qweights `[expert][gate,up,down][row][k]` plus a matching qscale slab, with
  strides resolved by `metal_expert_interleaved_slab_for_layer()` and
  `expert_stride_values` in the kernel params.
* Router and norms are validated fp16-only in
  `metal_validate_decoder_tensor_metadata()`; that stays.
* Adding a `.metal` fragment requires updating `tools/gen_metal.py` `ORDER` and
  regenerating `src/backend/metal/kernels/uocr_smoke.metal`.
* Kernel design follows **`docs/howto_quant.md`** (the distilled Q8 lessons):
  quantized inference is a memory problem (§1.1), there are exactly two
  workload regimes — simdgroup-per-row GEMV for decode and 64×32×32 tiled MMA
  GEMM for prefill (§1.2, §3, §4) — and dequantization is always fused (§1.3).
  Its §7 (“Applying this to dynamic Q4”) covers the Q4-specific decisions and
  is the reference for every kernel in section 4 of this plan.
* Performance rule carried over from Q8 (`docs/howto_quant.md` §1.3):
  **every Q4 compute path is fused**.  Nibble unpack, dequant, dot/MMA, SwiGLU,
  expert weighting, and combine happen inside one kernel; no dequantized weight
  buffers on CPU or GPU, no standalone dequant pass, no fp16/Q8 shadow copies.

---

## 1. Q4 format and `.uocr` metadata

### 1.1 Q4_0 storage format

Symmetric grouped int4 for routed-expert rank-2 weights:

```text
qtype:       UOCR_TENSOR_Q4_0 (= 4)
bits:        4 (two weights per byte)
group_size:  64
scale_type:  fp16
zero_point:  absent
levels:      [-8, 7]
quant_axis:  K/input dimension
layout:      row-major [out_features, in_features], packed along K
```

Per output row and 64-column group (ggml-style signed-max trick, exactly what
the sim validated):

```text
m     = group value with the largest |value|
scale = m / -8              (fp16; 1.0 for all-zero groups)
q     = round(w / scale), clamped to [-8, 7]
```

Packing (**group-half split**, llama.cpp-style): byte `j` of a 64-wide group
(j in 0..31) holds weight `g*64 + j` in the low nibble and `g*64 + 32 + j` in
the high nibble, stored as unsigned `q + 8`:

```text
byte[g*32 + j] = (uint8)(q[g*64 + j] + 8) | ((uint8)(q[g*64 + 32 + j] + 8) << 4)
```

This was probe-validated against the interleaved even/odd alternative: the
group-half split lets the GEMV unpack one `uchar4` load into two vectorized
`float4` halves that each pair with an *aligned* `half4` activation load
(`lo → x[k..k+3]`, `hi → x[k+32..k+35]`), measured **2.7× vs Q8 decode**,
while interleaved nibbles measured 0.9× (scalar unpack dominated).

Tensor-entry representation (existing fields only):

```text
payload_offset / payload_size = packed qweight bytes (= rows * cols / 2)
scale_offset   / scale_size   = qscale fp16 bytes    (= rows * cols/64 * 2)
block_size                    = 64
row_size                      = cols / 2   (packed bytes per row)
min_offset / min_size         = 0 for Q4_0
physical_shape = logical_shape = [rows, cols]
flags                         = UOCR_TENSOR_FLAG_ROW_MAJOR
```

Checklist:

* [x] Add `UOCR_TENSOR_Q4_0 = 4` to `uocr_tensor_qtype` and converter constants;
      reserve `UOCR_TENSOR_Q4_1 = 5` (do not implement).
* [x] Add `UOCR_Q4_GROUP_SIZE_DEFAULT = 64`, `UOCR_Q4_MIN = -8`, `UOCR_Q4_MAX = 7`.
* [x] Add `uocr_tensor_qtype_name()` support for `q4_0`.
* [x] Add converter/runtime helpers for Q4 qweight bytes, qscale bytes, totals.
* [x] Require `cols % 64 == 0` (both expert shapes qualify: gate/up
      `[896, 1280]`, down `[1280, 896]`).

### 1.2 Qprofile policy

Q4 ships as a third mixed profile layered on the Q8 one: experts Q4_0, the
rest of the Q8-supported decoder modules stay Q8_0, everything else fp16.

Checklist:

* [x] Add `UOCR_QPROFILE_MIXED_Q4 = 3` in `src/model/uocr_format.h` and
      `convert.py`; name `mixed-q4`.
* [x] Keep `UOCR_FORMAT_VERSION = 1` — the tensor entry already encodes
      everything Q4_0 needs.
* [x] Extend `uocr_model_file_validate_memory()` to accept
      `UOCR_QPROFILE_MIXED_Q4` and dispatch per-tensor validation by qtype
      (fp16 | q8_0 | q4_0).
* [x] Extend the provenance JSON `quantization` payload:

```json
{
  "quantization": {
    "mode": "mixed-q4",
    "group_size": 64,
    "policy": "embeddings+decoder",
    "expert_qtype": "q4_0",
    "lm_head_qtype": "q8_0",
    "router_qtype": "fp16",
    "converter_version": "0.3.0"
  }
}
```

### 1.3 Q4 tensor validation

Add a Q4_0 validator next to the fp16/Q8 ones in `src/model/uocr_model_file.c`.

Checklist:

* [x] Rank 2; logical == physical shape; row-major, not transposed.
* [x] `block_size == 64`; `cols % 64 == 0`; `cols % 2 == 0`.
* [x] `row_size == cols / 2`.
* [x] `payload_size == rows * cols / 2`.
* [x] `scale_size == rows * (cols / 64) * sizeof(uint16_t)`.
* [x] `payload_offset`/`scale_offset` 16-byte aligned, inside tensor-data section.
* [x] `min_offset == 0 && min_size == 0`.
* [x] Restrict Q4_0 to `TensorFamily.MOE_EXPERT` in decoder-binding validation;
      any other family with qtype Q4_0 is a validation error for this
      deliverable.

---

## 2. quant-cfg schema and converter

### 2.1 quant-cfg v2: per-module qtype

Extend the schema so a supported module can name its qtype (default `q8_0`
keeps every existing entry valid-by-default after the version bump):

```yaml
version: 2
group_size: 64

modules:
  - name: moe_routed_experts
    family: MOE_EXPERT
    projections: [GATE, UP, DOWN]
    supported: true
    qtype: q4_0        # new field; every other module omits it → q8_0
```

Checklist:

* [x] Bump quant-cfg `version` to 2 in `configs/quant-cfg.yaml` and
      `src/unlimitedocr_c/quant_cfg.py`; keep accepting version 1 (implicit
      `qtype: q8_0`).
* [x] `QuantModuleSpec` gains `qtype`; `QuantConfig.supported_pairs` becomes a
      mapping `(family, projection) → qtype`.
* [x] Reject `qtype: q4_0` on any family other than `MOE_EXPERT` (schema-level
      guard mirroring the runtime validator).
* [x] `moe_routed_experts` stayed `qtype: q8_0` in the shipped cfg until the
      full Q4 path passed QA (rollout discipline), then flipped to `q4_0`.

### 2.2 Converter CLI

```bash
uv run tools/uocr-convert \
  --qprofile mixed-q4 \
  --quant-cfg configs/quant-cfg.yaml \
  --out unlimitedocr-mixed-q4.uocr \
  --overwrite \
  --dump-quant-summary
```

Checklist:

* [x] Extend `--qprofile` choices to `fp16,mixed-q8_0,mixed-q4`.
* [x] `mixed-q4` plans exactly like `mixed-q8_0`, except modules whose cfg
      qtype is `q4_0` get Q4 planning; the policy stays
      `embeddings+decoder`.
* [x] Bump `CONVERTER_VERSION` to `(0, 3, 0)`.

### 2.3 Planning and layout

Checklist:

* [x] Extend `TensorPlan` byte-count helpers with `q4_qweight_bytes`
      (`rows * cols / 2`), `q4_qscale_bytes`, `q4_total_bytes`.
* [x] `_layout_dry_run_file()` allocates qweight + qscale ranges for Q4 tensors
      in the tensor-data section, 16-byte aligned, same as Q8.
* [x] Preserve the interleaved expert-major ordering: per layer, the routed
      expert Q4 qweight slab is `[expert][gate,up,down]` contiguous, followed by
      the matching contiguous qscale slab — same contract as Q8, halved
      qweight strides.
* [x] Update qtype histograms, `q8_summary` → generalized quant summary with
      per-qtype byte totals.

### 2.4 Quantization writer

Checklist:

* [x] Add `_stream_bf16_to_q4_0()` next to `_stream_bf16_to_q8_0()`: chunked
      row-major BF16→fp32, per-group signed-max scale, nibble packing
      (`q + 8`, low nibble = even k), qweight and qscale streamed to their
      offsets without materializing full tensors.
* [x] Reject non-finite values, mirror the Q8 error handling.
* [x] Extend `compare_single_tensor_conversion()` with Q4 dequant comparison
      (max_abs_error/rmse against source BF16); record tolerances in tests.
* [x] Unit tests in `tests/test_convert.py`: pack/unpack round-trip, scale
      edge cases (all-zero group, negative max), layout offsets, planned vs
      written byte counts.

---

## 3. Loader and Metal weight views

### 3.1 Model views

Checklist:

* [x] `payload_tensor_count()`/`build_payload_spans()` already emit spans for
      qweight + qscale ranges by qtype; extend the qtype switch to Q4_0.
* [x] Binding construction resolves both ranges to Metal buffers/offsets for
      Q4_0 (same two-buffer shape as Q8; `min` ranges stay unmapped).

### 3.2 Decoder bindings and expert slabs

Checklist:

* [x] `metal_validate_decoder_tensor_metadata()` accepts qtype Q4_0 **only**
      for `MOE_EXPERT` gate/up/down; all other families keep their current
      fp16/Q8 rules.
* [x] Extend the decoder-binding cache with Q4 shape/group metadata
      (`group_size`, `groups_per_row`, packed `row_bytes`).
* [x] Extend `metal_expert_interleaved_slab_for_layer()` and the fast expert
      slab cache for Q4: qweight slab stride is
      `expert_stride_bytes = 3 * projection_rows * cols / 2`, qscale slab
      stride unchanged from Q8 (`3 * projection_rows * cols / 64 * 2`).
* [x] Mixed-layer guard: within one layer all routed experts must share one
      qtype (Q4_0 or Q8_0); reject mixed slabs at binding time.

---

## 4. Metal kernels

> **Reference:** `docs/howto_quant.md` is normative for everything in this
> section — §3 for the decode GEMV shape, §4 for the prefill GEMM shape and
> the gathered-rows bucketed variant (§4.3), §5 for host-side integration
> rules, and §6 for the validation methodology (probe in `/tmp` first, compare
> bit-level vs a reference, then wire the dispatch).  §7 lists the Q4-specific
> design decisions (nibble packing, unpack cost, load widths).

New fragments, mirroring the Q8 structure:

```text
src/backend/metal/kernels/decode_gemv_q4.metal   decode GEMV expert kernels
src/backend/metal/kernels/gemm_q4.metal          bucketed prefill expert GEMM
```

Shared Q4 helpers live at the top of `decode_gemv_q4.metal` (or `common.metal`
if prefill needs them first in `gen_metal.py` order):

```metal
/* Unpack 8 weights from 4 packed bytes; nibbles are stored as q+8. */
static inline void uocr_q4_0_unpack8(uchar4 packed, thread float *w);  /* w[8] */

/* Lane-local Q4_0 GEMV dot: one simdgroup owns one output row; each lane
 * consumes 8-weight (4-byte) runs inside 64-wide groups, multiplies by the
 * group's fp16 scale, and the caller reduces with simd_sum. */
static inline float uocr_decode_gemv_q4_0_lane_dot(device const half *x,
                                                   device const uchar *qw_row,
                                                   device const half *qscale_row,
                                                   uint k, uint lane, uint simd_width);
```

Design notes grounded in measurements:

* The Q8 decode GEMV kernels run at near-peak bandwidth, so the Q4 win comes
  purely from halving bytes; the nibble unpack must stay off the critical path.
  Use ≥4-byte loads per lane (8 weights) — the Q8 LM-head probe showed 4-byte
  loads leave ~5–10% on the table versus 8-byte loads, so prefer `uchar4`/
  `ushort2` or wider per iteration and measure with a `/tmp` probe before
  finalizing, per the usual workflow.
* Group 64 with 32 lanes: each lane handles one 4-byte run per group
  (2 lanes × 8 weights × 4 runs… choose stride so a lane's run stays inside one
  group and multiplies by a single scale — same structure as the Q8 lane dot).

### 4.1 Decode (simdgroup-per-row GEMV)

Checklist:

* [x] `uocr_moe_decode_gate_up_gemv_q4_0_to_f16` — mirrors the Q8 kernel:
      grid `top_k * 896` rows, 128 threads, fused SwiGLU, fp32 accumulate,
      fp16 `mid` out.
* [x] `uocr_moe_decode_down_combine_gemv_q4_0_to_f16` — mirrors the Q8 kernel:
      1280 rows, fused expert weighting + combine + optional residual.
* [x] Route the decode expert dispatch (`metal_run_decode_moe_*`) by expert
      slab qtype: F16 → f16 GEMV, Q8_0 → q8 GEMV, Q4_0 → q4 GEMV.
* [x] `/tmp` probe comparing q8 vs q4 decode GEMV before wiring, per the
      methodology in `docs/howto_quant.md` §6 (expect ~1.6–2.0×; if unpack
      costs eat the win, widen loads before landing).

### 4.2 Prefill (expert-bucketed tiled GEMM)

Checklist:

* [x] `uocr_gemm_q4_stage_b()` in `gemm_q4.metal`: stage a dequantized fp16
      B tile `[BK][BN]` from packed nibbles + scales into threadgroup memory —
      dequant-on-stage exactly like `uocr_gemm_q8_stage_b()`; the MMA loop
      (`uocr_gemm_q8_mma_tile`) is reused unchanged.
* [x] `uocr_moe_prefill_bucketed_swiglu_gate_up_gemm_q4_0_to_f16` — same
      64×32×32 tile geometry, gathered A rows, fused SwiGLU epilogue.
* [x] `uocr_moe_prefill_bucketed_down_gemm_q4_0_to_f16` — same shape, fused
      down projection into per-pair rows.
* [x] Bucketing pass and combine kernels are reused as-is (qtype-agnostic).
* [x] Route the prefill bucketed dispatch by expert slab qtype.
* [x] Transient scratch preallocation (`metal_preallocate_moe_bucketed_prefill_
      transient_scratch`) is qtype-independent and unchanged.

### 4.3 Build integration

Checklist:

* [x] Add `decode_gemv_q4.metal` after `decode_gemv_q8.metal` and
      `gemm_q4.metal` after `gemm_q8.metal` in `tools/gen_metal.py` `ORDER`.
* [x] Regenerate `src/backend/metal/kernels/uocr_smoke.metal`.
* [x] Full pipeline warmup (`uocr_metal_context_compile_all_pipelines`) picks
      the new kernels up automatically.

---

## 5. Public API

Checklist:

* [x] No C API changes: qprofile/qtype are inferred from the `.uocr` file, same
      as Q8.
* [x] Python convenience: `UnlimitedOCR(quant="q4")` resolves/produces
      `unlimitedocr-q4.uocr` in the cache dir (converting with
      `--qprofile mixed-q4` on miss), mirroring `quant="q8"`.
* [x] `resolve_model_path(...)` learns the `q4` filename.

```python
from unlimitedocr_c import UnlimitedOCR
ocr = UnlimitedOCR(quant="q4")    # experts Q4_0, rest of decoder Q8_0
text = ocr.generate("page.png")
```

---

## 6. Memory accounting and profiling

Checklist:

* [x] Model-view admission uses the actual mixed-q4 tensor-data size (nothing
      to change structurally; verify the reported bytes drop by ~1.2 GB).
* [x] Quant summary reports per-qtype tensor counts and byte totals
      (fp16 / q8_0 / q4_0).
* [x] Profile event names: `metal.decode.moe_routed_experts_q4`,
      `metal.prefill.moe_bucketed_q4`.

---

## 7. Implementation order

Each step gates the next; the cfg flip is last.

1. **Format/validator groundwork** ✅
   * qtype/qprofile constants, name helpers, byte-count helpers;
   * Q4_0 tensor-entry validator + loader rejection tests.

2. **quant-cfg v2 + converter** ✅
   * per-module qtype schema;
   * `_stream_bf16_to_q4_0()` writer + expert slab layout;
   * pack/round-trip/layout unit tests; `--qprofile mixed-q4` dry-run summary.

3. **Loader / weight views / expert slabs** ✅
   * Q4 spans, bindings, slab strides + mixed-slab guards;
   * fp16 and Q8 paths still pass unchanged (full `ctest`).

4. **Decode GEMV Q4** ✅ (probe-first per `docs/howto_quant.md` §6)
   * `/tmp` probe vs Q8 GEMV caught the packing problem: interleaved nibbles
     0.9×, group-half split 2.7× — packing was revised before landing;
   * `decode_gemv_q4.metal` kernels + dispatch routing, bit-exact vs Q8 on
     identical dequant values.

5. **Prefill bucketed GEMM Q4** ✅
   * `gemm_q4.metal` dequant-on-stage stage-B + kernels + dispatch routing;
   * bucketing/combine passes reused unchanged.

6. **End-to-end QA, then flip the cfg** ✅
   * real `mixed-q4` model converted (2.24 GB) and QA'd in the notebook
     against the q8/fp16 baselines;
   * `moe_routed_experts: qtype: q4_0` flipped in the shipped
     `configs/quant-cfg.yaml`; `quant="q4"` wired.

7. **Cleanup** ✅
   * `src/unlimitedocr_c/q4_sim.py` and the `*-q4sim-*.uocr` cache artifacts
     deleted — the simulation was the gate, not the product;
   * README quant table + `docs/howto_quant.md` §7 outcome notes updated.

---

## Fallback ladder (if full-corpus QA regresses at step 6)

In order, cheapest first — all validated paths exist in the sim data:

1. Switch experts to **Q4_1** (scale+min; `UOCR_TENSOR_Q4_1 = 5`,
   `min_offset/min_size` already in the tensor entry; sim rmse 0.00206 vs
   0.00219).
2. **Down projection stays Q8_0**, gate/up Q4_0 (per-projection qtype is
   representable in quant-cfg v2 by splitting the module).
3. **Group size 32** for experts (doubles scale bytes, still ~45% smaller than
   Q8).

## First deliverable

```text
Model:
  mixed .uocr with qprofile UOCR_QPROFILE_MIXED_Q4
  routed experts (gate/up/down, 2112 tensors)   Q4_0 group 64
  embeddings / attention / dense / shared / LM head   Q8_0 (unchanged)
  router / norms / vision / projector                 fp16 (unchanged)
  file size ~2.26 GB (from 3.45 GB mixed-q8_0)

Runtime:
  fused Q4 decode GEMV (gate/up + down/combine) and bucketed prefill GEMM
  at the existing routed-expert call sites; no dequantized weight buffers;
  fp16 activations/KV/arenas unchanged.

User API:
  UnlimitedOCR(quant="q4") resolves/produces a cached mixed-q4 model.

Reporting:
  per-qtype byte totals, Q4 kernel timings, decode tokens/s vs mixed-q8_0.
```
