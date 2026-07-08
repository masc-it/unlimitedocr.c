# How to write performant quantized kernels for unlimitedocr.c

This guide condenses everything learned while building the mixed-Q8_0 path
(decoder, vision encoders, MoE, LM head) on the Metal backend.  It is the
starting point for the next quantization effort (dynamic Q4).  It goes from
foundations to details; every recommendation here was validated with measured
numbers on real shapes.

Companion documents: `docs/plan_q8.md` (decoder rollout), `docs/plan_q8_vision.md`
(vision rollout + performance postmortems).

---

## 1. Foundations: the three facts that decide everything

### 1.1 Quantized inference is a memory problem, not a compute problem

Weights are read from DRAM; activations are tiny by comparison.  The entire
design space is governed by one question: **how many times does each weight
byte cross the memory bus?**

* Decode (1 token): every weight is read exactly once per token, no reuse is
  possible.  The job is to *stream* weights at peak bandwidth.  Per decoded
  token, mixed-Q8 reads ~573 MB (LM head 165 MB, routed MoE 227 MB, shared MLP
  76 MB, QKV/O 79 MB, dense L0 26 MB).  At ~160 GB/s that is a ~3.6 ms/token
  floor; a kernel at 15% of bandwidth turns that into 24 ms.
* Prefill / vision (many rows): weights *can* be reused across rows.  A kernel
  that re-reads the weight matrix per row multiplies traffic by the row count.
  The original vision Q8 kernels re-read ~10 GB per SAM block linear
  (4096 rows × 2.4 MB); the tiled fix reads each weight tile once per 64-row
  block.

### 1.2 There are exactly two workload regimes — one kernel shape each

| regime | rows (M) | correct pattern | wrong pattern & measured cost |
|---|---|---|---|
| **GEMV** (decode, M == 1) | 1 | simdgroup-per-row streaming dot | threadgroup-per-output + tree reduction: 11–44% of BW |
| **GEMM** (prefill, vision, M ≫ 1) | 64 – 4096+ | tiled simdgroup-MMA with weight-tile reuse | GEMV kernel per row: 10 GB re-read per linear, 18–60× slower |

Never let one serve the other.  Every host dispatch site branches on
`n_tokens > 1` (or is a vision batch site, which is always GEMM-sized).

### 1.3 The dequantization must be fused — always

No kernel ever materializes a dequantized weight buffer in device memory, and
no MPS matmul ever runs on dequantized weights.  Dequantization happens:

* **GEMV**: in registers, inside the dot-product loop.
* **GEMM**: into threadgroup memory, once per output tile, shared by all rows
  of the tile.

This is a hard project rule (see the plans); it is also what makes quantization
pay off at all — Q8 halves the bytes, Q4 will halve them again.

---

## 2. Where quantized weights live

### 2.1 Format contract (Q8_0 today; the pattern generalizes)

```text
qtype        UOCR_TENSOR_Q8_0 (= 3)
layout       row-major [out_features, in_features]; no transpose
group_size   64 along the input (K) dimension
qweight      int8, payload_offset/payload_size in the tensor entry
qscale       fp16, one per (row, K-group); scale_offset/scale_size
range        [-127, 127]; dequant = int8 * scale
```

Key derived guarantees the kernels rely on:

* `K % 64 == 0` for every quantized tensor (validated at load).
* Payloads are 16-byte aligned (`UOCR_TENSOR_PAYLOAD_ALIGNMENT`); row starts
  are `K`-byte multiples, so `char4`/`half4` vector loads are always aligned.
* **An aligned run of 4 or 8 consecutive K elements never crosses a scale
  group** (4 and 8 divide 64).  This is what lets a vector load use a single
  scale lookup.  Preserve an equivalent property in any new format.

All model shapes used by quantized kernels (K and N per call site):

| call site | K | N | notes |
|---|---|---|---|
| decoder QKV / O | 1280 | 1280 | no bias |
| dense L0 gate/up/down | 1280 / 6848 | 6848 / 1280 | SwiGLU |
| shared MLP | 1280 / 1792 | 1792 / 1280 | SwiGLU |
| MoE expert (×64, interleaved slab) | 1280 / 896 | 896 / 1280 | gate,up,down per expert |
| LM head | 1280 | 129280 | argmax fused |
| token embedding | — | — | gather (get_rows), not a matmul |
| projector | 2048 | 1280 | bias |
| CLIP QKV / O / fc1 / fc2 | 1024 / 4096 | 3072 / 1024 / 4096 / 1024 | bias; QuickGELU on fc1 |
| SAM QKV / proj / lin1 / lin2 | 768 / 3072 | 2304 / 768 / 3072 / 768 | bias; erf-GELU on lin1 |

All K values are multiples of 64; all N values are multiples of 32.  Verify the
same before choosing tile sizes for a new format.

### 2.2 Loader plumbing (already dtype-aware — reuse it)

* `.uocr` tensor entries carry `qtype`, `block_size`, `row_size`,
  `scale_offset/scale_size` — a new qtype reuses the same fields.
* Model views map both qweight and qscale ranges; the binding caches
  (`uocr_metal_decoder_binding`, `uocr_metal_vision_binding`) expose
  `{qtype, rows, cols, group_size, groups_per_row, buffer/offset,
  scale_buffer/scale_offset}` per tensor.
* Validation is fail-fast and per-role: a quantized tensor that reaches a call
  site without a fused kernel must fail at bind/validate time with a clear
  message, never fall back silently (`metal_vision_tensor_allows_q8`,
  `metal_decoder_tensor_allows_q8` are the templates).
* `configs/quant-cfg.yaml` is the single source of truth for which
  (family, projection) pairs the converter may quantize.  A module flips to
  `supported: true` only after its fused kernels exist and end-to-end QA
  passed.  Q4 must get its own module gating (and likely its own qtype +
  profile), rolled out module by module exactly like Q8.

---

## 3. The GEMV pattern (decode, M == 1)

File: `src/backend/metal/kernels/decode_gemv_q8.metal`.

### 3.1 Shape

* **One simdgroup per output row.**  128-thread threadgroups = 4 rows each.
* Host grid: `ceil(rows / (128 / threadExecutionWidth))` threadgroups
  (`metal_decode_gemv_rows_per_threadgroup`).
* **No threadgroup memory, no barriers.**  The only reduction is `simd_sum`.
* Vectorized loads: `char4` weights + `half4` activations per lane; a
  32-lane simdgroup issues 128 contiguous weight bytes per instruction.

The core (everything else is epilogue variation):

```metal
static inline float uocr_decode_gemv_q8_lane_dot(device const half *x,
                                                 device const char *w_row,
                                                 device const half *s_row,
                                                 uint k, uint lane, uint simd_width) {
    device const char4 *w4 = (device const char4 *)w_row;
    device const half4 *x4 = (device const half4 *)x;
    float sum = 0.0f;
    for (uint c = lane; c < k / 4u; c += simd_width) {
        const char4 w = w4[c];
        const half4 xv = x4[c];
        const float scale = float(s_row[(c * 4u) / 64u]);   // 4-run never crosses a group
        sum += (float(xv.x)*float(w.x) + float(xv.y)*float(w.y) +
                float(xv.z)*float(w.z) + float(xv.w)*float(w.w)) * scale;
    }
    return sum;   // caller: simd_sum(...)
}
```

### 3.2 Why the old pattern was slow (do not repeat it)

The first-generation decode kernels used one threadgroup (64–256 threads) per
*output element* with a tree reduction: for K=1280 each thread did ~5 MACs and
then paid multiple barrier rounds; weight loads were scalar bytes.  Measured on
real shapes:

| decode kernel | old (per-output TG) | new (simdgroup GEMV) | gain |
|---|---|---|---|
| attention QKV | 0.272 ms · 18 GB/s | 0.086 ms · 57 GB/s | 3.2× |
| attention O | 0.069 ms · 24 GB/s | 0.035 ms · 47 GB/s | 2.0× |
| shared MLP g/u/d | 0.235 ms · 29 GB/s | 0.111 ms · 62 GB/s | 2.1× |
| MoE routed top-6 | 0.526 ms · 39 GB/s | 0.213 ms · 97 GB/s | 2.5× |
| LM head argmax | 2.392 ms · 69 GB/s | 1.059 ms · **156 GB/s** | 2.3× |

Streaming-read reference on the test machine: ~159 GB/s.  Large-N dispatches
(LM head) reach it; small-N dispatches (1280 rows = 320 threadgroups) cannot
fully hide latency and land at 45–60 GB/s — that is expected, not a bug.

### 3.3 Epilogue variants (fuse, don't add passes)

* fused QKV: `row / n` selects projection weight/scale/dst (one dispatch for
  Q, K, V).
* residual add: lane 0 adds `residual[row]` before the fp16 store.
* SwiGLU: one simdgroup computes the gate dot then the up dot for the same
  column; `silu(g) * u` at lane 0.  Two sequential dots per simdgroup is fine —
  it is still one weight-read per weight.
* MoE: expert id lookup per (rank, col); pointer bases derived from the
  interleaved expert slab (`expert * expert_stride + {0, up_offset,
  down_offset}` — the `col * K` term commutes, so `up_row = gate_row +
  up_offset` works on already-offset pointers).
* MoE down+combine: accumulate *lane partials* across all top-k experts
  weighted by `top_weights[rank]`, then one `simd_sum` at the end (the weighted
  sum is linear — don't reduce per expert).
* LM head: dot + argmax fused; NaN-safe compare with index tie-breaking, two
  stages (per-tile partials, then a small reduce).

---

## 4. The GEMM pattern (prefill and vision, M ≫ 1)

File: `src/backend/metal/kernels/gemm_q8.metal`.

### 4.1 Shape (the configuration that won)

```text
tile      BM=64 rows × BN=32 cols, BK=32 K-slice
threads   128 = 4 simdgroups in a 2×2 grid of 32×16 sub-tiles
per sg    4×2 simdgroup_float8x8 accumulators (fp16 MMA in, fp32 out)
smem      8 KB total, aliased:
            during K loop: [sa 64×32 half | sb 32×32 half]
            epilogue:      sc 64×32 float (over the same bytes, after barrier)
A stage   half4 cooperative loads, zero-fill rows ≥ M
B stage   one (column, 8-wide K run) per thread: 8 int8 loads + 1 scale,
            dequantized transposed into sb[k][n] (k-major for B-side MMA loads)
```

Measured on the heaviest shape (SAM lin1, 4096×768→3072):

| variant | ms | GFLOP/s | verdict |
|---|---|---|---|
| GEMV-style tile4 (first attempt) | ≫100 | reduction/traffic-bound | 18–60× too slow |
| tiled MMA, 14 KB separate smem, scalar A loads | 16.9 | 1146 | good |
| **+ aliased 8 KB smem + half4 A loads (shipped)** | **8.8** | **2234** | ~75% of MPS fp16 |
| BM=64,BN=64 (16 accumulators/sg, 24 KB smem) | 35.3 | 547 | occupancy collapse — don't |
| BK=64 (12 KB smem, fewer barriers) | 9.4 | 2051 | slightly worse |
| MPS fp16 reference, same shape | 6.5 | 2975 | Q8 reads half the bytes |

Lessons baked into those rows:

* **Threadgroup memory is the occupancy budget.**  Aliasing staging and the
  output tile (8 KB total) was worth 2×.  Bigger tiles that push smem past
  ~12 KB or accumulators past 8 mats/simdgroup lose more to occupancy than
  they gain in reuse.
* fp16 MMA inputs with **fp32 accumulators** (`simdgroup_float8x8`) keep
  accuracy at ~5e-4 max rel. error over K up to 4096 — same class as MPS.
* Stage B transposed (`sb[k][n]`) so `simdgroup_load` for the B side is a
  plain k-major 8×8.
* M is handled by zero-fill + bounds-checked epilogue; N by `ceil` grid +
  bounds checks — kernels are shape-generic across every call site above.

### 4.2 Epilogues

Same philosophy as GEMV: `simdgroup_store` the accumulators to the (aliased)
fp32 tile, barrier, then 128 threads apply bias / activation / residual /
split and write fp16.  Variants shipped: bias+activation(+QKV 3-way split)
for vision, plain / residual / dual-accumulator SwiGLU for the decoder.

**Activation parity is part of correctness**: CLIP uses QuickGELU
(`x·sigmoid(1.702x)`), SAM uses erf-GELU, the decoder uses SiLU.  Match the
fp16 path's exact function (`uocr_quickgelu`, `uocr_gelu_erf`), not a generic
"gelu".

### 4.3 Gathered rows: the MoE bucketed GEMM (mul_mm_id)

Routed-MoE prefill can't tile naively because consecutive rows use different
expert weights.  The shipped solution (~18× at 1024 tokens: 528 → 29 ms/layer):

1. **Bucket pass** (single 256-thread threadgroup): histogram of
   `(token, expert)` pairs by expert via threadgroup atomics → serial prefix
   sums (`expert_offsets[65]`, `tile_prefix[65]` in units of 64-row tiles) →
   atomic-cursor scatter of pair ids into `pair_rows`.  Invalid expert ids are
   dropped here and re-checked in the combine.
2. **Bucketed GEMMs**: grid.y is *overdispatched* to
   `ceil(pairs/64) + expert_count` tile slots; each threadgroup binary-searches
   `tile_prefix` for its (expert, local tile), stages its 64 pair ids into a
   small threadgroup array, and runs the standard tiled GEMM with a *gathered*
   A stage (`row = pair_rows[...] / top_k` for token rows, `/1` for mid rows).
3. **Combine** (elementwise): `dst = shared + residual + Σ_rank w·down_out`.

Scratch (`pair_rows`, prefix arrays, `down_out = pairs × hidden fp16`) lives in
the TRANSIENT scratch slot; all four passes are encoded back-to-back in one
command buffer, so nothing else can clobber it.

This machinery is format-agnostic — Q4 only changes the weight-tile
dequantization step.

---

## 5. Host-side integration rules

* **Dispatch by qtype at every call site**; keep the fp16 path intact next to
  the quantized branch.  Selection: `weight->qtype == UOCR_TENSOR_Q8_0`
  (or per-projection for mixed blocks).
* **Dispatch by regime**: `n_tokens > 1` → GEMM; decode → GEMV.  Vision batch
  sites are always GEMM.
* Params structs are 16/32/48-byte packed structs with `_Static_assert`ed
  sizes, mirrored field-for-field in the kernel fragment.  When a fragment
  needs a struct defined in a *later* fragment (concatenated translation unit,
  order in `tools/gen_metal.py`), define a same-layout local mirror
  (see `UocrMoeBucketedGemmQ8Params`).
* Kernel fragments are focused files; regenerate `uocr_smoke.metal` with
  `uv run tools/gen_metal.py` after any change, and verify **both** the
  runtime-compiled and the precompiled (`xcrun metal`) builds.
* Threadgroup arrays declared inside the kernel (compile-time sizes) don't need
  `setThreadgroupMemoryLength`; runtime-sized ones do.
* Add a `uocr_profile` event per new quantized path
  (`metal.vision.projector_q8`, `metal.prefill.moe_routed_bucketed_q8`, …) —
  they are how QA confirms the path is actually taken.
* Encoder churn costs ~4 µs each; decode issues ~100 per token (~0.4 ms).
  Acceptable today, but don't add passes casually — fuse into epilogues.

---

## 6. Validation methodology (non-negotiable)

Every kernel shipped in the Q8 effort was validated **before** end-to-end QA
with a standalone Objective-C probe that compiles `uocr_smoke.metal` at
runtime (`newLibraryWithSource`; use `newFunctionWithName:constantValues:`
with empty constants for kernels referencing function constants):

1. Fill random fp16 activations, int8 weights, small positive fp16 scales.
2. Run the kernel on the *real call-site shapes* (including M-edge cases like
   101/257/273 rows, split epilogues, both activations, invalid MoE experts).
3. Compare against a scalar CPU reference **that models the kernel's rounding**
   (e.g. dequantized weights pass through fp16 when the kernel stages them as
   half).  Accept max rel. error < 0.02; healthy kernels sit at ~5e-4.
4. Benchmark against a streaming-read reference kernel on the same machine and
   report **GB/s and % of reference** — "% of bandwidth" is the only number
   that tells you whether a memory-bound kernel is done.

Then follow the rollout discipline: flip the module in `quant-cfg.yaml`,
`UnlimitedOCR(quant="q8", force_reconvert=True)`, user QA on representative
OCR inputs vs the fp16 baseline, only then move to the next module.

Performance targets to hold a new kernel to:

| workload | target |
|---|---|
| GEMV, large N (LM head) | ≥ 85% of streaming reference |
| GEMV, small N (1280–3840 rows) | ≥ 30–40% of reference (latency-bound tail is real) |
| GEMM, K ≥ 768 | ≥ 60–75% of MPS fp16 GFLOP/s on the same shape |
| MoE prefill | within ~2× of the equivalent dense GEMM GFLOP/s |

---

## 7. Applying this to dynamic Q4

What "Q4" changes and what it doesn't:

### 7.1 Unchanged (reuse as-is)

* Both kernel skeletons (simdgroup GEMV, tiled MMA GEMM), the bucketed-MoE
  machinery, all epilogues, the host dispatch/regime logic, binding caches,
  scratch handling, probes, and the module-by-module rollout.
* The format plumbing: `.uocr` tensor entries already carry
  `block_size/row_size/scale_offset/scale_size` and an unused
  `min_offset/min_size` pair — the latter is the natural home for Q4
  zero-points/minimums if the scheme is asymmetric.
* Host param structs for Q4 already exist as placeholders
  (`uocr_metal_dense_q4_params`, `uocr_metal_moe_selected_q4_params`, …) —
  keep their 32-byte ABI discipline.

### 7.2 New design decisions Q4 forces

* **Packing layout.**  Two nibbles per byte.  Pack along K
  (`byte = w[k] | (w[k+1] << 4)` or the llama.cpp low/high-half split).
  Choose so that one vector load (`uchar4` = 8 weights) still maps to **one
  scale lookup**: an aligned 8-weight run must stay inside one group — with
  group 64 and nibble packing, 4-byte runs cover 8 K elements, so the Q8
  alignment argument carries over unchanged.
* **Dequant cost doubles per byte.**  Q4 GEMV loads half the bytes but does the
  same MACs plus nibble extraction (`(b & 0xF) - 8`, `(b >> 4) - 8`).  Decode
  becomes even more bandwidth-bound → expect up to ~2× decode gain *if* the
  unpack stays cheap; keep it in registers, sign-extend with a subtract, avoid
  per-element branches.
* **Zero-point/asymmetry.**  If dynamic Q4 is asymmetric
  (`w = q*scale + min`), the dot gains a correction term:
  `Σ x·(q·s + m) = s·Σ x·q + m·Σ x`.  Precompute the per-group activation sum
  `Σ x` **once per row-group, not per output row** — in GEMV compute it per
  simdgroup pass; in GEMM stage it alongside the A tile.
* **Dynamic activation quantization** (if "dynamic" means int8 activations):
  add a tiny pre-pass per matmul input that computes per-token (or per-group)
  activation scales and quantizes x to int8, then the inner loop becomes
  int8×int4 with one fp multiply per group.  That pre-pass is elementwise and
  cheap; keep it fused into the preceding norm kernel if possible.  Accumulate
  in int32 within a group, fp32 across groups.
* **Accuracy budget.**  Q8 landed at ~5e-4 kernel-level rel. error and passed
  OCR QA everywhere.  Q4 will not; expect per-module quality decisions — which
  is exactly why the quant-cfg gating exists.  Start with the most tolerant,
  highest-byte modules (MoE experts: 227 MB/token; LM head: 165 MB/token) and
  QA each alone before combining.
* **Converter**: new qtype id + `uocr_q4_*_bytes` helpers mirrored in
  `src/unlimitedocr_c/convert.py`, dry-run planning tests, and compare support
  (dequantized diff against BF16 source) before any runtime work — same
  sequence as Q8.

### 7.3 Suggested Q4 kernel order (highest value first)

1. MoE expert slab (decode GEMV + bucketed GEMM) — 227 MB/token decode, the
   single biggest reader.
2. LM head argmax — 165 MB/token, and the GEMV already runs at full bandwidth,
   so Q4 bytes convert ~1:1 into time.
3. Shared MLP, dense L0, QKV/O — decode GEMV variants.
4. Prefill/vision GEMM stage-B dequant variants — lowest urgency (prefill is
   already fast and vision is near MPS parity; Q4 there is a memory-footprint
   win more than a speed win).

---

## 8. File map

| file | contents |
|---|---|
| `src/backend/metal/kernels/decode_gemv_q8.metal` | simdgroup-per-row decode GEMV family |
| `src/backend/metal/kernels/gemm_q8.metal` | tiled MMA GEMM core + vision/decoder/bucketed-MoE kernels |
| `src/backend/metal/kernels/dense_q8.metal`, `attention_q8.metal`, `moe_q8.metal` | remaining first-gen kernels (embedding gather, legacy decode variants) |
| `src/backend/metal/kernels/embedding_q8.metal`, `lm_head_q8.metal` | token-embedding gather; fused LM-head argmax |
| `src/backend/metal/uocr_metal.m` | binding caches, validation, dispatch-by-qtype/regime, params structs |
| `src/model/uocr_format.h` ↔ `src/unlimitedocr_c/convert.py` | format contract (mirrored) |
| `configs/quant-cfg.yaml`, `src/unlimitedocr_c/quant_cfg.py` | runtime-safe module gating |
| `tools/gen_metal.py` | fragment order → `uocr_smoke.metal` (fragment order = struct/helper visibility) |
