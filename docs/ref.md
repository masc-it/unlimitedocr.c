# Refactor Targets: Compile-Time Fast Paths and Dead-Code Removal

This is an actionable backlog for moving production OCR away from runtime dispatch, shape checks, and historical fallbacks. The focus is not new math optimization; it is making hot paths explicit, fixed-shape, and easier to reason about.

## Current assumptions

- Model shape is fixed by `src/model/uocr_constants.h`:
  - decoder: `hidden=1280`, `layers=12`, `heads=10`, `head_dim=128`, `moe_top_k=6`
  - production max batch default is `1`
  - vision constants are fixed for the shipped model family
- Current long-output profiles are dominated by decode:
  - `metal.decode_loop` ~15-17s
  - `metal.chunk.wait` ~14-16s
  - decode CPU encoding/step events are smaller but still repeated ~1k tokens
- Production Metal path now uses GPU-resident chunked decode, MPS global SAM attention, tiled MoE decode down-combine, and decode-only MoE top-k.
- Second-pass concrete code findings:
  - `src/backend/metal/uocr_metal.m` still starts with `#define UOCR_METAL_ENABLE_DIAGNOSTIC_API 1`, so production currently compiles the diagnostic surface unconditionally.
  - `metal_run_decoder_decode_one_f16()` still rebuilds/validates scratch slices and resolves layer binding indices for every generated token and layer.
  - `metal_prewarm_integrated_decoder_pipelines()` still prewarms broad kernel sets; selected-kernel prewarm has improved, but the list still includes non-selected alternatives.
  - Current release builds still warn about unused Metal helpers, including old SAM helpers and `metal_select_next_token_from_hidden_slice_f16()`.

---

## A. Remove per-op runtime backend decisions from decode

### A1. Split decode and prefill projection wrappers

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- `src/backend/metal/kernels/uocr_smoke.metal`

**Problem:** Functions such as `metal_run_attention_qkv_buffer_f16()`, `metal_run_attention_output_residual_buffer_f16()`, `metal_run_dense_swiglu_buffer_f16()`, and `metal_run_moe_router_buffer_f16()` still check shape/backend policy at runtime. For decode, `n_tokens == 1` is known, so MPS threshold checks and fallback policy are unnecessary.

**Action:**
- Add decode-only wrappers:
  - `metal_run_decode_attention_qkv_one_f16()`
  - `metal_run_decode_attention_output_one_f16()`
  - `metal_run_decode_dense_swiglu_one_f16()`
  - `metal_run_decode_moe_router_one_f16()`
- Route `metal_run_decoder_decode_one_f16()` to these wrappers directly.
- Leave current generic wrappers for prefill/diagnostic paths only.

**Acceptance criteria:**
- No call to `metal_mps_matmul_nt_f16_should_use()` from the decode-token path.
- Normal long-output run still produces sane output.
- `metal.decode.layer` CPU-side time does not regress.

### A2. Cache or remove `UOCR_METAL_MPS_MATMUL_MIN_FLOPS` parsing from hot paths

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** `metal_mps_matmul_min_flops()` calls `getenv()`/`strtoull()` whenever MPS threshold decisions are made. This is not a GPU bottleneck, but it is unnecessary runtime policy in hot wrapper code.

**Action:**
- Cache parsed threshold in a static integer, same style as decoder/vision profile-detail parsing.
- Or, after A1, restrict this function to prefill/diagnostic paths only.

**Acceptance criteria:**
- `metal_mps_matmul_min_flops()` does not parse the environment repeatedly.
- No behavior change when `UOCR_METAL_MPS_MATMUL_MIN_FLOPS` is unset.

---

## B. Remove repeated decoder binding lookup from decode

### B1. Precompute per-layer decoder fast bindings

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** Profiles show many repeated binding-cache lookups, e.g. `metal.decoder_binding_cache.direct_lookup` with ~100k+ calls in long decode. `metal_run_decoder_decode_one_f16()` repeatedly calls `metal_require_decoder_binding_index()` for every token/layer/op even though layer weights do not change.

**Action:**
- Add a fast per-layer struct in `uocr_metal_context`, for example:
  ```c
  typedef struct uocr_metal_decoder_layer_fast_bindings {
      const uocr_metal_decoder_binding *input_norm;
      const uocr_metal_decoder_binding *post_attention_norm;
      const uocr_metal_decoder_binding *q_proj;
      const uocr_metal_decoder_binding *k_proj;
      const uocr_metal_decoder_binding *v_proj;
      const uocr_metal_decoder_binding *o_proj;
      const uocr_metal_decoder_binding *dense_gate;
      const uocr_metal_decoder_binding *dense_up;
      const uocr_metal_decoder_binding *dense_down;
      const uocr_metal_decoder_binding *moe_router;
      const uocr_metal_decoder_binding *moe_shared_gate;
      const uocr_metal_decoder_binding *moe_shared_up;
      const uocr_metal_decoder_binding *moe_shared_down;
  } uocr_metal_decoder_layer_fast_bindings;
  ```
- Populate it once after model bindings are loaded/validated.
- In the decode loop, index this table directly.

**Acceptance criteria:**
- Decode token loop no longer calls `metal_require_decoder_binding_index()` per layer.
- Startup/open-model validation catches missing bindings before generation.
- `metal.decoder_binding_cache.direct_lookup` disappears or drops sharply in normal profiles.

### B2. Precompute per-layer expert slabs

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** `metal_expert_interleaved_slab_for_layer()` is still invoked inside MoE decode. Even if cheap, expert slab selection is fixed by layer.

**Action:**
- Store `uocr_metal_buffer_slice routed_expert_slab[UOCR_DECODER_LAYERS]` in the context.
- Fill during model binding/prewarm.
- Use direct array lookup in decode and prefill.

**Acceptance criteria:**
- No slab lookup helper call in `metal_run_decoder_decode_one_f16()`.
- Output unchanged.

### B3. Precompute decode scratch slices per slot

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** `metal_run_decoder_decode_one_f16()` calls `metal_hidden_arena_segment_slice()` for every scratch segment on every generated token. In long decode this repeats thousands of validity checks and offset calculations for fixed slot/segment/one-token slices.

**Action:**
- Add a per-slot decode scratch view, for example:
  ```c
  typedef struct uocr_metal_decode_slot_fast_slices {
      uocr_metal_buffer_slice hidden_segment[UOCR_METAL_HIDDEN_SCRATCH_SEGMENTS];
      uocr_metal_buffer_slice norm_scratch;
      uocr_metal_buffer_slice selection_scratch;
      uocr_metal_buffer_slice router_top_ids;
      uocr_metal_buffer_slice router_top_weights;
      uocr_metal_buffer_slice moe_mid;
  } uocr_metal_decode_slot_fast_slices;
  ```
- Populate it when runtime arenas are prepared or at decode-loop entry.
- Pass these slices into `metal_run_decoder_decode_one_f16()` instead of recomputing them.

**Acceptance criteria:**
- `metal_run_decoder_decode_one_f16()` has no loop that recomputes all hidden scratch segment slices.
- Long decode output remains sane.
- CPU-side `metal.chunk.step.decode` / `metal.decode.layer` profile time drops or stays neutral.

---

## C. Compile decode kernels for fixed one-token shapes

### C1. Add decode-only RMSNorm kernel

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/kernels/uocr_smoke.metal`
- `src/backend/metal/uocr_metal.m`

**Problem:** RMSNorm kernels are generic over rows/tokens and output type. Decode always normalizes one `UOCR_HIDDEN_SIZE=1280` vector.

**Action:**
- Add `uocr_rmsnorm_one_f16_to_f16` specialized for one row and decoder hidden size via function constants.
- Host wrapper should take only input, weight, output.
- Use this in decode attention norm, post-attention norm, final norm.

**Acceptance criteria:**
- Decode path does not pass `n_tokens=1` into generic RMSNorm.
- Targeted token profile shows `attn_norm`, `post_norm`, and final norm not worse.

### C2. Add decode-only QKV projection kernel without `projection_count` branches

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/kernels/uocr_smoke.metal`
- `src/backend/metal/uocr_metal.m`

**Problem:** `uocr_attention_qkvo_f16_to_f16` supports Q/K/V/O through `projection_count` and runtime branch selection. Decode QKV always does exactly Q,K,V for one token.

**Action:**
- Add `uocr_decode_attention_qkv_one_f16`.
- Dispatch three fixed projection regions or one kernel with separate explicit loops, but do not branch on `projection == ...` inside each output element.
- Keep generic kernel for prefill and diagnostics.

**Acceptance criteria:**
- Decode QKV wrapper uses the new kernel.
- Targeted token profile `d.l*.attn_qkv` improves or stays neutral.

### C3. Add decode-only attention output kernel

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/kernels/uocr_smoke.metal`
- `src/backend/metal/uocr_metal.m`

**Problem:** Attention output residual projection is generic over `n_tokens`; decode is one token with fixed hidden size.

**Action:**
- Add `uocr_decode_attention_output_residual_one_f16`.
- Remove token division and `token >= n_tokens` checks from the decode kernel.

**Acceptance criteria:**
- Decode path no longer calls generic `uocr_attention_output_residual_f16_to_f16`.
- `d.l*.attn_out` profile improves or stays neutral.

### C4. Add decode-only RoPE+KV-write fused kernel

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/kernels/uocr_smoke.metal`
- `src/backend/metal/uocr_metal.m`

**Problem:** Decode currently encodes RoPE and KV write as separate operations. Token/layer count makes this repeated overhead visible even when each op is small.

**Action:**
- Add a kernel that applies RoPE to Q/K and writes K/V to KV cache in one pass.
- Keep separate generic RoPE and KV write for prefill/diagnostics.

**Acceptance criteria:**
- Decode path has one checkpoint/event instead of separate `rope` and `kv_write`.
- Correctness unchanged on short probe and long screenshot.

---

## D. Collapse MoE compile-time feature switches

### D1. Remove unfused routed MoE decode fallback

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** `UOCR_METAL_ENABLE_MOE_FUSED_DOWN_COMBINE` is now expected to be `1`; the `#else` path still keeps old `metal_run_moe_interleaved_buffer_f16()` + `metal_run_moe_combine_buffer_f16()` code alive.

**Action:**
- Delete the `#else` path in decode and prefill once parity is accepted.
- Delete or move `metal_run_moe_interleaved_buffer_f16()` and `metal_run_moe_combine_buffer_f16()` to diagnostics.

**Acceptance criteria:**
- `rg "UOCR_METAL_ENABLE_MOE_FUSED_DOWN_COMBINE" src/backend/metal/uocr_metal.m` shows no production branch.
- Build passes.

### D2. Make decode MoE top-k a named product choice

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- `src/model/uocr_constants.h` optional

**Problem:** `UOCR_METAL_DECODE_MOE_TOP_K` is a backend-local compile-time choice. If it is accepted as production behavior, it should be documented as a product/perf tradeoff rather than an experimental knob.

**Action:**
- Add a short comment near the define documenting chosen value, speed impact, and quality acceptance.
- Optionally move it to `uocr_constants.h` as `UOCR_DECODE_MOE_TOP_K`.

**Acceptance criteria:**
- The chosen top-k value is documented with benchmark reference.
- There is no environment variable for this path.

### D3. Remove non-selected MoE tile kernel from production prewarm

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- `src/backend/metal/kernels/uocr_smoke.metal`

**Problem:** Both tile4 and tile8 decode down-combine kernels are available/prewarmed, but production uses only `UOCR_METAL_MOE_ROUTED_DOWN_COMBINE_KERNEL`.

**Action:**
- Prewarm only the selected kernel macro, not both tile4 and tile8.
- Keep the alternative kernel behind an explicit compile-time experiment macro or remove it after settling.

**Acceptance criteria:**
- Pipeline cache count at startup decreases by one.
- No hot-path pipeline-cache growth warning.

### D4. Collapse always-on MoE feature macros

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** Several production MoE switches default to enabled and now describe the expected path:
- `UOCR_METAL_ENABLE_MOE_DECODE_KERNELS`
- `UOCR_METAL_ENABLE_MOE_FUSED_ROUTER`
- `UOCR_METAL_ENABLE_MOE_SHARED_GATE_UP_TILED`
- `UOCR_METAL_ENABLE_MOE_ROUTED_GATE_UP_TILED`

These keep fallback branches alive and make production control flow harder to audit.

**Action:**
- After quality/perf acceptance, remove the fallback branches guarded by these macros.
- If an experiment is still needed, invert it into a local branch-only patch rather than production compile-time knob.

**Acceptance criteria:**
- Production decode MoE/router code has one path per operation.
- `rg "UOCR_METAL_ENABLE_MOE_" src/backend/metal/uocr_metal.m` only shows constants that are genuinely build-product choices.

---

## E. Remove diagnostic/parity code from production builds

### E1. Split diagnostic Metal API implementations out of `uocr_metal.m`

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- new file, e.g. `src/backend/metal/uocr_metal_diagnostic.m`
- `src/backend/metal/uocr_metal.h`
- `CMakeLists.txt`

**Problem:** `uocr_metal.m` includes a very large diagnostic/parity surface even though `uocr_metal.h` already gates prototypes with `UOCR_METAL_ENABLE_DIAGNOSTIC_API`. The implementation currently forces diagnostics on with `#define UOCR_METAL_ENABLE_DIAGNOSTIC_API 1` at the top of `uocr_metal.m`, while `CMakeLists.txt` compiles only one Metal Objective-C implementation file. This increases compile time, code size, and makes production path harder to audit.

**Action:**
- Remove the unconditional `#define UOCR_METAL_ENABLE_DIAGNOSTIC_API 1` from production `uocr_metal.m`.
- Move all `uocr_metal_context_diagnostic_*` implementations to `uocr_metal_diagnostic.m`.
- Compile that file only for `test_metal`/diagnostic builds or behind a new CMake option.
- Production library should include only integrated GPU-resident APIs.

**Acceptance criteria:**
- Production `uocr_metal.m` contains no `uocr_metal_context_diagnostic_*` definitions.
- Diagnostic tests still build when explicitly enabled.

### E2. Remove legacy host-pointer vision workspace token path from production

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** The comment around legacy diagnostic helpers notes that host pointers and opaque private workspace tokens coexist. Production vision uses private GPU workspace and should not need the host-pointer compatibility path.

**Action:**
- After E1, move `metal_private_workspace_token()` and host-pointer workspace compatibility helpers into diagnostic-only code if production no longer references them.
- Keep production workspace as explicit `MTLBuffer + offset` slices.

**Acceptance criteria:**
- Production vision workspace helpers do not cast fake pointers.
- Diagnostic path retains compatibility if needed.

### E3. Split diagnostic Metal kernels from production kernels

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/kernels/uocr_smoke.metal`
- new file, e.g. `src/backend/metal/kernels/uocr_diagnostic.metal`
- `CMakeLists.txt`

**Problem:** `uocr_smoke.metal` contains both production kernels and diagnostic/parity variants such as f32-output dense/attention helpers. Production builds compile all of them into one library.

**Action:**
- Move f32-output and diagnostic-only kernels to a separate diagnostic Metal source.
- Keep production metallib focused on kernels reachable from public OCR generation.
- Update `test_metal`/diagnostic builds to include the diagnostic metallib/source.

**Acceptance criteria:**
- Production metallib function count drops.
- `test_metal` still compiles all diagnostic kernels when diagnostics are enabled.

---

## F. Delete old vision kernels and fallbacks after production path is fixed

### F1. Remove unused SAM neck scalar conv helpers

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- `src/backend/metal/kernels/uocr_smoke.metal`

**Problem:** Several older SAM conv/layernorm helpers are unused or retained for diagnostics after BHWC/im2col/MPS changes. Examples include unused/static helper warnings and earlier NCHW scalar paths.

**Action:**
- Audit these functions with `rg`:
  - `metal_context_sam_transformer_workspace_f16_to_slice`
  - `metal_context_sam_neck_conv1x1_batch_f16_to_slice`
  - `metal_context_sam_layernorm2d_batch_f16_to_slice`
  - `metal_context_sam_neck_conv3x3_batch_f16_to_slice`
  - `metal_context_sam_stride2_conv_batch_f16_to_slice`
  - `metal_context_sam_conv3x3_im2col_mps_batch_f16_to_slice`
- Move diagnostic-only ones to the diagnostic file or delete if truly unreachable.

**Acceptance criteria:**
- Release build no longer emits these unused-function warnings.
- Vision output remains sane on `docs/test.png` and long screenshot.

### F2. Remove obsolete NCHW/BHWC conversion fallbacks from production vision

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- `src/backend/metal/kernels/uocr_smoke.metal`

**Problem:** Production SAM neck now intends to remain in BHWC. Old conversion/fallback kernels increase code surface and risk layout bugs.

**Action:**
- Identify conversions used only by diagnostics or obsolete paths:
  - `uocr_bhwc_to_nchw_f16`
  - old NCHW SAM conv kernels
- Keep only conversions required by the final production concat/projector path.

**Acceptance criteria:**
- Production SAM neck layout is documented as BHWC end-to-end.
- No production call path re-enters NCHW scalar conv fallback.

---

## G. Simplify selection/LM-head backend policy

### G1. Remove `UOCR_METAL_LM_HEAD_SELECTION_BACKEND` from production

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** Next-token selection currently has an env-selectable MPS LM-head path and a custom fused path. The fused custom path is the production default and avoids large logits materialization.

**Action:**
- Move MPS LM-head path to diagnostic/experiment code or delete it.
- Remove `metal_lm_head_backend_is_mps()` from production selection.

**Acceptance criteria:**
- Decode selection has one backend in production.
- No `UOCR_METAL_LM_HEAD_SELECTION_BACKEND` env var in production code.

### G2. Compile no-repeat mode as request policy, not per-kernel fallback

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- `src/backend/metal/kernels/uocr_smoke.metal`

**Problem:** Selection code carries banned-candidate/no-repeat branches into every token. For the common request profile, no-repeat parameters are known before generation.

**Action:**
- Split selection into two host paths:
  - no-repeat disabled
  - no-repeat enabled
- In the disabled path, skip ban-flag scratch and no-repeat kernels entirely.

**Acceptance criteria:**
- Requests with `no_repeat_ngram_size == 0` execute fewer selection kernels.
- Existing default request behavior unchanged.

### G3. Delete unused old per-token selection helper

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** Release builds warn that `metal_select_next_token_from_hidden_slice_f16()` is unused. Chunked GPU-resident decode uses `metal_encode_chunk_step_selection_f16()` instead.

**Action:**
- Confirm no diagnostic/test path calls `metal_select_next_token_from_hidden_slice_f16()`.
- Delete it or move it to diagnostic code.

**Acceptance criteria:**
- Release build no longer warns about this function.
- Chunked decode selection still works.

---

## H. Production prewarm should match exactly the production fast path

### H1. Replace broad pipeline prewarm list with selected-fast-path prewarm

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`

**Problem:** `metal_prewarm_integrated_decoder_pipelines()` prewarms both generic and experimental kernels. Recent pipeline-cache warnings came from mismatch between prewarm specialization and runtime specialization.

**Action:**
- Separate prewarm into:
  - production-required pipelines
  - diagnostic/experimental pipelines
- For function-constant kernels, prewarm with the exact runtime key used by production.
- Add an assertion/test that generation does not grow pipeline cache in warmup.

**Acceptance criteria:**
- `metal_hot_path_alloc_guard_end()` never reports pipeline-cache growth in warmup or first generation.
- Pipeline cache count is stable and documented.

### H2. Add a pipeline-cache no-growth regression test

**Status:** [ ] Not started

**Files:**
- `tests/test_metal.c`
- `src/backend/metal/uocr_metal.h`
- `src/backend/metal/uocr_metal.m`

**Problem:** Pipeline-cache growth warnings have recurred when compile-time choices changed. This should be caught by a small test instead of by long probe runs.

**Action:**
- Expose a test-only helper that records pipeline cache count before/after a minimal warm generation or decode step.
- Assert the count is unchanged after prewarm.

**Acceptance criteria:**
- `ctest -R uocr_metal` fails if a production hot path lazily creates a pipeline.
- The test does not require the long screenshot probe.

---

## I. Lower-priority cleanup

### I1. Remove `UOCR_METAL_ENABLE_MPP_TENSOROPS` dead experiment if unused

**Status:** [ ] Not started

**Files:**
- `src/backend/metal/uocr_metal.m`
- `src/backend/metal/kernels/uocr_smoke.metal`

**Problem:** TensorOps support is guarded by `UOCR_METAL_ENABLE_MPP_TENSOROPS` and defaults to `0`. If no active experiment uses it, it is dead code.

**Action:**
- Confirm no build/profile requires it.
- Delete include/preprocessor plumbing or move to a branch.

**Acceptance criteria:**
- Build passes with no `UOCR_METAL_ENABLE_MPP_TENSOROPS` references.

### I2. Keep CPU reference backend, but keep it out of Metal production decisions

**Status:** [ ] Not started

**Files:**
- `src/core/uocr_api.c`
- `src/backend/cpu_ref/uocr_cpu_ref.c`

**Problem:** CPU-ref is useful for tests/parity but should not shape Metal production fast-path code.

**Action:**
- Leave CPU-ref as a separate backend.
- Avoid adding CPU-ref fallback checks inside Metal hot code.

**Acceptance criteria:**
- Metal backend has no CPU-ref fallback branches.

---

## Suggested execution order

1. [ ] B1/B2/B3: precompute decoder bindings, slabs, and scratch slices.
2. [ ] G3/F1: delete unused helpers that already trigger release warnings.
3. [ ] H1/H2: exact selected-fast-path prewarm plus no-growth regression test.
4. [ ] A1: split decode wrappers and remove MPS threshold checks from decode.
5. [ ] C1-C4: fixed one-token decode kernels for norm/projections/RoPE+KV.
6. [ ] D1-D4: collapse MoE feature/fallback branches after tile/top-k is settled.
7. [ ] E1-E3: split diagnostic implementation and kernels out of production.
8. [ ] F2: delete obsolete vision layout fallbacks.
9. [ ] G1-G2: simplify LM-head/selection policy.

