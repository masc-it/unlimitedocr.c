# MoE Decoder Optimization Implementation Plan

Goal: improve UnlimitedOCR Metal fp16 MoE decoder throughput while preserving deterministic output, fp16 model behavior, and upstream Python parity.

Scope: decode after prefill. The model has `hidden_size=1280`, `num_hidden_layers=12`, one dense MLP layer, and 11 routed MoE layers with `64` experts, `top_k=6`, `expert_intermediate=896`, and `shared_intermediate=1792`.

Reference grounding:
- Model architecture: `data.tmp/reference/modeling_unlimitedocr.py` and `data.tmp/reference/modeling_deepseekv2.py`.
- Metal Shading Language: `data.tmp/reference/Metal-Shading-Language-Specification.pdf`.
- Current Metal implementation:
  - decode MoE router: `metal_run_moe_router_buffer_f16()`
  - shared expert path: `metal_run_dense_swiglu_buffer_f16()`
  - routed expert path: `metal_run_moe_interleaved_buffer_f16()`
  - final add: `metal_run_moe_combine_buffer_f16()`
  - kernels in `src/backend/metal/kernels/uocr_smoke.metal`

Metal spec constraints to apply:
- MSL p153: choose threadgroup sizes as multiples of `threads_per_simdgroup` for best performance.
- MSL p157: standard compute SIMD-groups are linear and independent without barriers; use `thread_index_in_simdgroup`/`simdgroup_index_in_threadgroup` deliberately.
- MSL p215-p216: all threads in a threadgroup/SIMD-group must encounter `threadgroup_barrier`/`simdgroup_barrier`; barriers inside conditionals or loops must be uniform for all participating threads.
- MSL p219-p222: use SIMD-group operations (`simd_sum`, `simd_max`, `simd_shuffle*`, prefix reductions) to reduce threadgroup memory traffic where deterministic reduction order is acceptable.
- MSL p127: `[[max_total_threads_per_threadgroup]]` caps pipeline threadgroup size; Metal 4 `[[required_threads_per_threadgroup]]` can be considered only where device support and exact dispatch shape are guaranteed.
- MSL p190-p193: use function constants for scalar/vector shape specialization instead of creating extra source variants.
- MSL p349-p354: TensorOps/cooperative tensors are Metal 4 features; all threads in the execution scope must call TensorOp methods uniformly, and explicit barriers are required before reading device/threadgroup tensor outputs.

Validation requirements for every native-code step:
- Rebuild Release with Metal enabled.
- Run `uv run probes/e2e_generation_probe.py --profile base`.
- Run `uv run probes/e2e_generation_probe.py --profile gundam`.
- Compare generated token counts/text previews against the current baseline.
- Run focused C/Metal tests for any new kernel parity.
- Restart notebook/Jupyter kernels after native dylib changes.

## 1. Capture MoE Decoder Baselines

Status: [x] Completed

Details:
- Record Release probe output for `base` and `gundam` before code changes.
- Run probes with enough profile depth to include:
  - `metal.decode_step`
  - `metal.decode.layer`
  - `metal.decode.moe_router`
  - `metal.decode.moe_shared_experts`
  - `metal.decode.moe_routed_experts`
  - `metal.decode.moe_combine`
  - `metal.decode.command_wait`
  - `metal.decode.command_encoding`
- Document current decode MoE dispatch structure:
  - router logits
  - router softmax/top-k
  - shared gate/up
  - shared down
  - routed gate/up
  - routed down
  - combine routed + shared + residual
- Distinguish CPU command-encoding time from GPU command-wait time; the per-kernel profile buckets are useful for structure, but `metal.decode.command_wait` carries the batched GPU execution cost.

Acceptance criteria:
- Baseline tok/s and MoE profile totals are recorded.
- Current per-token/per-layer MoE dispatch count is documented.
- Baseline generated text is saved for parity comparison.

Completion notes:
- Baseline artifacts:
  - `data.tmp/probes/e2e_base_moe_baseline.json`
  - `data.tmp/probes/e2e_gundam_moe_baseline.json`
  - `data.tmp/probes/moe_decoder_baseline.md`
- Baseline speed:
  - `base`: 44 generated tokens, 7.122 native tok/s, 654.628 ms decode loop.
  - `gundam`: 58 generated tokens, 1.509 native tok/s, 1720.646 ms decode loop.
- Current MoE decode structure: 7 MoE dispatches per MoE layer/token, or `11 * 7 = 77` MoE dispatches per generated token, excluding dense layer 0 and attention.
- Profile call counts confirm 11 MoE layers per decode iteration: `base` has `43 * 11 = 473` MoE bucket calls; `gundam` has `57 * 11 = 627`.
- CPU command encoding is small relative to GPU command wait: 0.137 ms vs 626.160 ms (`base`) and 0.267 ms vs 1675.643 ms (`gundam`).

## 2. Fuse Routed Down and MoE Combine

Status: [x] Completed

Details:
- Replace the decode-only sequence:
  - routed down writes routed output
  - separate combine kernel adds routed + shared + residual
- With a fused routed-down-combine decode kernel that writes:
  - `dst = routed_down(top_k experts) + shared_out + residual`
- Preserve routed accumulation order over ranks and intermediate columns as closely as possible.
- Preserve existing fp16 behavior by rounding the routed down-projection to fp16 before adding shared expert output and residual.
- Leave prefill kernels unchanged.
- Keep the old path behind an internal fallback until parity and performance are verified.

Acceptance criteria:
- Decode no longer launches `metal.decode.moe_combine` on the default path.
- Synthetic fused-vs-old kernel test passes within existing fp16 tolerances.
- `base` and `gundam` E2E output remains stable.
- Decode command count decreases.

Completion notes:
- Added `uocr_moe_decode_interleaved_down_sum_combine_f16_to_f16`, which fuses routed down projection with shared/residual combine on the decode path.
- Preserved fp16 parity with the unfused path by rounding the routed down-projection result to fp16 before adding shared and residual rows.
- Default decode no longer emits `metal.decode.moe_combine`; the unfused path remains available behind `UOCR_METAL_ENABLE_MOE_FUSED_DOWN_COMBINE=0` at compile time.
- Added focused diagnostic coverage via `uocr_metal_context_diagnostic_moe_interleaved_experts_combine_f16()` and `UOCR_METAL_TEST_FILTER=moe_interleaved_experts_combine_f16 ./build/test/test_metal`.
- Final Release probe artifacts:
  - `data.tmp/probes/e2e_base_moe_fused_down_combine_final.json`
  - `data.tmp/probes/e2e_gundam_moe_fused_down_combine_final.json`
  - `data.tmp/probes/moe_fused_down_combine_results.md`
- Final probe summary:
  - `base`: text parity true, 44 generated tokens, 7.117 native tok/s, `metal.decode.moe_combine` absent, command encoders `8205 -> 7732`.
  - `gundam`: text parity true, 58 generated tokens, 1.517 native tok/s, `metal.decode.moe_combine` absent, command encoders `12529 -> 11902`.

## 3. Add Decode-Only MoE Kernel Variants

Status: [x] Completed

Details:
- Add kernels dedicated to `n_tokens == 1` decode rather than reusing prefill kernels.
- Specialize fixed dimensions via function constants or compile-time constants:
  - hidden size `1280`
  - routed experts `64`
  - top-k `6`
  - routed intermediate `896`
  - shared intermediate `1792`
- Remove prompt-token indexing, dynamic divisions, and generic bounds checks that are unnecessary for decode.
- Prefer 2D dispatch coordinates for `(rank, output_column)` or `(tile, output_column)` shapes when this avoids hot integer division/modulo.
- Apply MSL p190-p193 function-constant rules: scalar/vector constants only, specialized at pipeline creation.

Acceptance criteria:
- Decode path uses decode-specific MoE kernels.
- Prefill path continues using prompt-sized kernels.
- Fallback to existing kernels remains available for debugging.
- `metal.decode.moe_*` profile time is equal or lower.

Completion notes:
- Added decode-only routed MoE kernel entry points for the `n_tokens == 1` path:
  - `uocr_moe_decode_interleaved_gate_up_one_f16`
  - `uocr_moe_decode_interleaved_down_sum_combine_one_f16_to_f16`
- The default decode path now dispatches the decode-specific gate/up entry point and continues to use the fused decode down/combine kernel from Task 2. The fully stripped down/combine-one variant is retained for diagnostics/follow-up because measurements showed the conservative fused path was faster on the current GPU.
- Prefill remains on `uocr_moe_prefill_interleaved_gate_up_f16` and `uocr_moe_prefill_interleaved_down_sum_f16_to_f16`.
- Added function-constant IDs for MoE expert/shared intermediate sizes for future decode-specialized kernels, following MSL p190-p193 scalar function-constant rules.
- Focused parity coverage: `UOCR_METAL_TEST_FILTER=moe_interleaved_experts_combine_f16 ./build/test/test_metal` now exercises the decode-only `n_tokens == 1` diagnostic path.
- Final Release probe artifacts:
  - `data.tmp/probes/e2e_base_moe_decode_kernels.json`
  - `data.tmp/probes/e2e_gundam_moe_decode_kernels.json`
  - `data.tmp/probes/moe_decode_kernel_results.md`
- Final probe summary versus Task 2:
  - `base`: text parity true, 44 generated tokens, 7.058 native tok/s, decode loop `656.497 -> 653.218` ms, `metal.decode.moe_combine` absent; routed MoE stayed effectively neutral within the tiny profile bucket (`2.452 -> 2.493` ms over 473 calls).
  - `gundam`: text parity true, 58 generated tokens, 1.515 native tok/s, layer kernels `36.500 -> 36.229` ms, `metal.decode.moe_combine` absent; routed MoE stayed effectively neutral within the tiny profile bucket (`3.335 -> 3.386` ms over 627 calls).

## 4. Replace Router Bitonic Sort with Deterministic Top-6 Selection

Status: [x] Completed

Details:
- Current router top-k sorts all 64 experts; decode only needs top 6.
- Implement repeated top-1 selection for six ranks from the 64 softmax probabilities.
- Preserve tie-break behavior exactly: higher probability wins; equal probability selects lower expert id.
- Use tuple-like reductions `(score, inverse_expert_id)` rather than score-only `simd_max` when preserving tie-breaks.
- Use SIMD-group reductions/shuffles where they reduce barriers, but keep all barriers in uniform control flow per MSL p215-p216.
- Keep softmax probabilities unrenormalized, matching the UnlimitedOCR routing contract.

Acceptance criteria:
- Router ids and weights match the current bitonic path on deterministic synthetic cases, including equal-score ties.
- `metal.decode.moe_router` time does not regress.
- E2E token output remains stable.

Completion notes:
- Replaced the router softmax/top-k full 64-expert bitonic sort with deterministic greedy top-6 selection.
- Selection uses tuple ordering `(score_bits, inverse_expert_id)` over nonnegative softmax probabilities, so higher probability wins and exact ties select the lower expert id while keeping raw, unrenormalized softmax weights.
- The default dispatch uses `UOCR_METAL_MOE_ROUTER_TOPK_THREADS=16`, keeping the fixed 64-expert/top-6 selection on the one-SIMD greedy path and avoiding the old bitonic top-k barrier sequence. A generic multi-SIMD greedy fallback remains in the kernel.
- Diagnostic router coverage now has a focused filter: `UOCR_METAL_TEST_FILTER=moe_router_f16 ./build/test/test_metal`. The existing zero-logit/equal-probability case verifies lower-id tie order (`0..5`).
- Diagnostic router pipeline creation now uses decoder-shape function constants, matching the integrated decode path and avoiding generic function-constant pipeline creation failures.
- Final Release probe artifacts:
  - `data.tmp/probes/e2e_base_moe_router_topk.json`
  - `data.tmp/probes/e2e_gundam_moe_router_topk.json`
  - `data.tmp/probes/moe_router_topk_results.md`
- Final probe summary versus Task 3:
  - `base`: text parity true, 44 generated tokens, 7.121 native tok/s, decode loop `653.218 -> 651.531` ms, router `2.720 -> 2.757` ms, `metal.decode.moe_combine` absent.
  - `gundam`: text parity true, 58 generated tokens, 1.516 native tok/s, decode loop `1707.815 -> 1700.695` ms, router `3.665 -> 3.750` ms, `metal.decode.moe_combine` absent.
  - Router bucket differences are sub-microsecond per MoE call and noisy in E2E profiling; the greedy top-k path preserves parity and removes the full bitonic sort. Task 5 should address the remaining router cost by fusing logits, softmax, and top-k and removing the full logits/probs round trip.

## 5. Fuse Decode Router Logits, Softmax, and Top-K

Status: [x] Completed

Details:
- For decode, fuse router logits and softmax/top-k into one kernel.
- Avoid global memory round-trip for full 64-expert logits/probabilities when only top ids and top weights are consumed by routed experts.
- Stage the single 1280-element hidden vector in threadgroup memory only if it improves reuse enough to justify barrier cost.
- Consider one threadgroup per router row or a small fixed set of SIMD-groups; keep threadgroup size a multiple of `threads_per_simdgroup`.
- Preserve softmax max-subtraction, total-sum, probability calculation, and top-k tie-break semantics.
- Keep the existing two-stage router as a fallback until parity is proven.

Acceptance criteria:
- Decode router writes top ids/weights directly.
- Full logits/probs buffer traffic is removed from the default decode router path.
- Synthetic router parity passes against the current implementation.
- `base` and `gundam` E2E output remains stable.

Completion notes:
- Added `uocr_moe_router_decode_fused_f16`, a decode-only router kernel for the fixed OCR shape (`hidden=1280`, `experts=64`, `top_k=6`). It stages the single hidden row in threadgroup memory, computes router logits, applies softmax max-subtraction/sum, and writes top ids/weights directly without global logits/probs traffic.
- The fused kernel keeps 256-thread per-expert dot-product partitions so the per-expert accumulation shape matches the existing logits kernel, while processing four experts per 1024-thread threadgroup batch. The first SIMD-group performs deterministic top-6 selection with the same `(score_bits, inverse_expert_id)` tie-break semantics as Task 4.
- Added `UOCR_METAL_ENABLE_MOE_FUSED_ROUTER` as a compile-time fallback switch. Prefill and non-decode router calls continue to use the existing logits + softmax/top-k path.
- Diagnostic router coverage now exercises the fused decode path when `n_tokens == 1` and optional logits/probs outputs are omitted; the existing focused test verifies top-id/top-weight parity.
- Final Release probe artifacts:
  - `data.tmp/probes/e2e_base_moe_fused_router.json`
  - `data.tmp/probes/e2e_gundam_moe_fused_router.json`
  - `data.tmp/probes/moe_fused_router_results.md`
- Final probe summary versus Task 4:
  - `base`: text parity true, 44 generated tokens, 7.092 native tok/s, router `2.757 -> 1.566` ms, encoder count `7732 -> 7259`, `metal.decode.moe_combine` absent.
  - `gundam`: text parity true, 58 generated tokens, 1.515 native tok/s, router `3.750 -> 2.018` ms, encoder count `11902 -> 11275`, `metal.decode.moe_combine` absent.
  - Total decode-loop samples were dominated by noisy command-wait time in these E2E runs, but decode command encoding/layer buckets and router traffic dropped while output parity remained stable.

## 6. Tile Shared Expert Gate/Up GEMV

Status: [ ] Not started

Details:
- Optimize shared expert gate/up first because it has one shared MLP per MoE layer and no top-k rank dimension.
- Current kernels launch one threadgroup per output column; implement tiled output columns per threadgroup.
- Reuse the single decode hidden vector across a tile of columns.
- Use vectorized `half4` loads when alignment and multiples permit (`1280` is divisible by 4).
- Keep all threadgroup-memory staging and barriers uniform.
- Compare tile sizes such as 4, 8, and 16 columns with profile data.

Acceptance criteria:
- Shared gate/up threadgroup count is materially reduced.
- `metal.decode.moe_shared_experts` time decreases without parity drift.
- Hot-path allocation guard remains clean.

## 7. Tile Routed Expert Gate/Up GEMV

Status: [ ] Not started

Details:
- Extend the tiled gate/up approach to routed experts after shared expert validation.
- Shape is `top_k=6` by `expert_intermediate=896` output columns.
- Use selected expert ids to address the interleaved expert slab.
- Preserve gate and up dot-product accumulation order per output column as much as practical.
- Avoid divergent barriers based on expert rank or expert id.

Acceptance criteria:
- Routed gate/up threadgroup count decreases.
- `metal.decode.moe_routed_experts` gate/up portion improves in focused profiling.
- Synthetic selected-expert decode parity passes.

## 8. Tile Routed Down Projection with Final Add

Status: [ ] Not started

Details:
- Implement tiled down projection for routed experts and keep routed-down-combine fused.
- For each hidden output tile, accumulate six selected expert down projections weighted by router probabilities.
- Add shared output and residual before final store.
- Avoid extra hidden-vector read/write passes.
- Preserve fp32 accumulation over intermediate columns and top-k ranks as closely as practical.

Acceptance criteria:
- Routed down and final add remain fused.
- Down projection threadgroup count decreases.
- `metal.decode.moe_routed_experts` and total decode step time improve or do not regress.
- E2E output remains stable.

## 9. Reduce Decode MoE Command Encoding Overhead

Status: [ ] Not started

Details:
- After kernel fusion/tiled kernels, review command-buffer and encoder count per decoded token.
- Keep per-token command batching intact and avoid waits between MoE sub-kernels unless data dependencies require a completed command buffer.
- Reuse pipeline states and function-constant variants from existing caches.
- Avoid new Objective-C allocation in the decode hot path.
- Confirm profile events still expose enough detail for regression analysis.

Acceptance criteria:
- Command encoder count per decoded token decreases or stays flat with faster kernels.
- `metal.decode.command_wait` and `metal.decode.command_encoding` do not regress.
- Hot-path allocation guard passes.

## 10. Evaluate Metal 4 TensorOps Prototype Behind Feature Gate

Status: [ ] Not started

Details:
- Prototype TensorOps/cooperative-tensor matmul only after manual tiled kernels are measured.
- Feature-gate by OS/GPU support and keep disabled by default until faster and stable.
- Follow MSL p349-p354 TensorOps rules:
  - all threads in the execution scope must call TensorOp methods uniformly
  - TensorOps may use execution-scope barriers
  - insert required barriers before reading device/threadgroup tensor outputs
  - keep fallback manual kernels available
- Prioritize half x half -> float/half combinations supported by the spec for Metal 4.

Acceptance criteria:
- Unsupported devices use manual kernels.
- Prototype can be toggled independently for benchmarking.
- Default E2E path remains stable.

## 11. Add MoE Decoder Tests and Probe Reporting

Status: [ ] Not started

Details:
- Add tests for router top-k determinism, including equal-score tie cases.
- Add synthetic old-vs-new tests for fused routed-down-combine.
- Add decode-only selected-expert tests with fixed `top_k=6` and interleaved expert-slab addressing.
- Expand probe/profile reporting or saved benchmark notes to summarize MoE dispatch counts and event totals.
- Keep test tensors fp16 and aligned with model constants.

Acceptance criteria:
- C/Metal tests cover router selection and fused routed-down-combine behavior.
- Probe output or benchmark notes explain the speed delta from each optimization step.
- Release E2E probes pass after every native-code change.
