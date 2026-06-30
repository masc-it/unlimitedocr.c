# Metal fp16 kernel implementation plan

Reference: `data.tmp/reference/Metal-Shading-Language-Specification.pdf`

## 1. Make flash attention SIMD-width correct

**Goal:** Remove the hard dependency on 32-wide SIMD groups while preserving the fast path on current Apple GPUs.

**Details:**
- Audit all flash-attention kernels using `UOCR_FLASH_SIMD_WIDTH 32u`.
- Add host-side validation against `pipeline.threadExecutionWidth` for current kernels.
- Either fail clearly on non-32 SIMD width or add an adaptive kernel path using `threads_per_simdgroup`.

**Implementation:**
- [x] Remove hardcoded 32-wide flash-attention dispatches; derive threadgroup size from `pipeline.threadExecutionWidth`.
- [x] Replace misleading `maxTotalThreadsPerThreadgroup`-only checks.
- [x] Add diagnostics for unsupported SIMD width/head-dim combinations.

## 2. Replace threadgroup tree reductions with SIMD-group reductions

**Goal:** Reduce barriers, threadgroup memory use, and latency in fp16/fp32 reductions.

**Details:**
- Prefer `simd_sum`, `simd_max`, and SIMD shuffle/reduction APIs for per-SIMD reductions.
- Use threadgroup memory only to combine per-SIMD partials when needed.
- Use correct active-lane handling for partial SIMD groups.
- Target dense dots, norm kernels, router/top-k, argmax, and attention fallback reductions.

**Implementation:**
- [x] Add shared SIMD reduction helpers in the Metal source.
- [x] Convert dense/norm reductions first.
- [x] Convert router/top-k and argmax reductions.
- [x] Benchmark barrier count and runtime before/after.

## 3. Prefer optimized MPS/MPP matmul for large fp16 GEMMs

**Goal:** Ensure large fp16 matrix multiplies use Apple-tuned primitives instead of slow scalar custom kernels.

**Details:**
- Current MPSNDArray matmul usage is directionally correct.
- Revisit `UOCR_METAL_MPS_MATMUL_MIN_FLOPS`; the fallback one-output-per-threadgroup GEMM is not competitive for many shapes.
- For Metal 4+ targets, evaluate MPP TensorOps `matmul2d`; keep MPSNDArray fallback for broader OS support.
- The MPP prototype is opt-in behind `UOCR_METAL_ENABLE_MPP_TENSOROPS`; default builds keep MPSNDArray.
- Respect TensorOps execution-scope rules: all threads in the selected scope must call `run`, and explicit barriers are needed before reading device/threadgroup tensor outputs.

**Implementation:**
- [x] Benchmark MPS threshold across representative decoder/vision GEMMs.
- [x] Lower or autotune `UOCR_METAL_MPS_MATMUL_MIN_FLOPS`.
- [x] Add a feature-gated MPP TensorOps prototype for Metal 4+.
- [x] Keep current MPSNDArray path as default portable optimized path.

## 4. Vectorize fp16 memory-bound kernels

**Goal:** Improve bandwidth efficiency for simple fp16 elementwise/copy kernels.

**Details:**
- Use `half2`/`half4` vector loads and stores where buffers are aligned and widths are multiples of 2/4.
- Keep scalar tail handling for odd sizes.
- Target residual adds, bias adds, KV-cache writes, prompt assembly, row copies, and format/concat kernels.

**Implementation:**
- [x] Identify kernels with contiguous fp16 loads/stores and minimal arithmetic.
- [x] Add `half4` fast paths with scalar tails.
- [x] Validate alignment assumptions against Metal buffer offsets.
- [x] Benchmark memory bandwidth and end-to-end latency.

## 5. Make Metal math mode explicit

**Goal:** Make fp16/fp32 numerical-performance tradeoffs intentional and reproducible.

**Details:**
- Runtime compilation now sets `MTLCompileOptions` math mode explicitly; precompile passes matching `metal` flags.
- Default `UOCR_METAL_MATH_MODE=FAST` matches the prior compiler default for inference speed.
- `RELAXED` preserves NaN/Inf while allowing unsafe FP optimization; `SAFE` uses precise fp32 functions for validation/debug builds.

**Implementation:**
- [x] Set explicit fast math options for runtime compilation where supported.
- [x] Add matching flags to the CMake Metal precompile command.
- [x] Document expected numerical behavior and tolerances.
- [x] Add a debug/safe-math switch if needed for validation.

## 6. Specialize hot kernels with function constants and threadgroup attributes

**Goal:** Let the compiler optimize fixed OCR shapes and remove runtime branches from hot kernels.

**Details:**
- Added decoder-shape function constants for fixed model values: hidden size, vocab size, heads, head dim, ring window, RoPE/attention scale, MoE experts/top-k, LM-head tile shape, and QKV projection count.
- Integrated hot-path dispatch uses specialized pipeline states while diagnostics and generic APIs keep unspecialized fallback pipelines.
- Added `[[max_total_threads_per_threadgroup]]` to fixed-size hot kernels; `[[required_threads_per_threadgroup]]` remains deferred because it would constrain adaptive SIMD-width flash dispatch.

**Implementation:**
- [x] Identify hot kernels with runtime params that are actually fixed for the model.
- [x] Add function-constant pipeline variants for the highest-impact paths.
- [x] Cache specialized pipeline states by constant tuple.
- [x] Benchmark runtime, prewarm overhead, and memory usage with the E2E generation probe.

## 7. Evaluate native packed numeric and tensor-blockwise quantized paths

**Goal:** Improve quantized fp16 kernels by using native Metal packed/tensor formats where available.

**Details:**
- Metal 4+ exposes packed numeric types (`int4b_format`, `uint4b_format`, fp4/fp8) and TensorOps matmul combinations for half × packed formats.
- Current Q4/Q8 kernels manually unpack bytes/nibbles; native formats may reduce decode overhead and improve bandwidth.
- This must be feature-gated because support depends on OS, SDK, and GPU family, and Q4_K layout may need repacking.

**Implementation:**
- [ ] Map current Q4_K/Q8_0 layouts to native formats or define a repack step.
- [ ] Prototype Metal 4+ packed numeric unpack for local quantized dot kernels.
- [ ] Prototype MPP TensorOps for supported half × int4/uint4 shapes.
- [ ] Keep existing manual decode path as compatibility fallback.
