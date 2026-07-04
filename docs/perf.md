# Performance

## Overview

End-to-end generation has three major stages:

1. **Frontend preprocessing** — image loading, view preparation, tokenization.
2. **Vision encoding** — SAM patch-embed + 12-block SAM transformer + SAM neck →
   CLIP embedding + 24-block CLIP transformer → CLIP/SAM concat → projector.
3. **Decoder prefill + decode loop** — prompt assembly, prefill attention, then
   auto-regressive decode with attention + MoE layers.

## Current Baseline (2026-07-04)

Measured with `uv run probes/e2e_generation_probe.py --profile base` on a
single 1024×1024 test image (one global view, `base` profile).

> **Blatant issue discovered during analysis:** `uocr_metal_context_warmup_model_views()`
> exists, is publicly declared in `uocr_metal.h` and tested in `tests/test_metal.c`,
> but is **never called** from the Python FFI, the probe, or any generation path.
> The 6.2 GiB model runs completely cold — every GPU first-touch triggers a
> page fault. Wiring this one call eliminates ~4 s of GPU stall. See root cause 2.

```
Timings:
  resolve_model_s                   0.327 ms
  load_image_s                     10.453 ms
  prepare_request_s               272.488 ms
  engine_open_model_load_s       6016.378 ms
  generate_native_s              6513.824 ms
  decode_s                          0.820 ms
  engine_close_s                    1.776 ms
  total_s                       12816.519 ms
```

### Breakdown of `generate_native_s = 6513 ms`

| Stage | Time | % of generate |
|---|---|---|
| Vision encoding | 4294 ms | 66 % |
| Decoder prefill | 1326 ms | 20 % |
| Decode loop (43 iters) | 689 ms | 11 % |
| Other (prompt assembly, etc.) | ~200 ms | 3 % |

### Vision encoding breakdown

```
metal.vision                                       4294 ms
  metal.vision.chunk_processing                    4294 ms
    metal.vision.global_shape_group_command_wait   4156 ms  ← 97 % of vision
    metal.vision.chunk_encode                       132 ms  ← CPU submit
    metal.vision.formatter                           ~6 ms
```

### Per-operation GPU compute (inside chunk_encode)

| Operation | Time |
|---|---|
| `metal.vision.chunk_encode` (total submission) | 132 ms |
| metal.vision.sam_patch_batch | not isolated |
| metal.vision.sam_transformer_batch | 67 ms |
| metal.vision.sam_block.00 | **64.8 ms** (280× slower than avg) |
| metal.vision.sam_block (01–11 avg) | ~0.23 ms each |
| metal.vision.sam_neck | 23 ms |
| metal.vision.clip_frontend_batch | not isolated |
| metal.vision.clip_transformer_batch | not isolated |
| metal.vision.concat_batch | not isolated |
| metal.vision.projector_batch | not isolated |

> **Key observation:** SAM block 0 takes **64.8 ms CPU encode time** while the
> other 11 blocks average **0.23 ms** — a 280× penalty. This is **not** Metal
> pipeline compilation (those are cached via `metal_get_pipeline()`). It is
> **MPSNDArrayMatrixMultiplication JIT kernel compilation** on first
> `encodeToCommandBuffer:` with each distinct `(rows, in_features, out_features)`
> tuple. The MPS framework compiles internal GPU kernels lazily; no public API
> exists to force this ahead of time beyond a dummy encode+wait per shape.

### Top-level command buffer wait times

| Wait | Time | Calls |
|---|---|---|
| `metal.command_buffer_wait` | 6144 ms | 47 calls |
| └ vision `global_shape_group_command_wait` | 4156 ms | 1 call |
| └ prefill `command_wait` | 1324 ms | 1 call |
| └ decode `command_wait` | 662 ms | 44 calls (~15 ms avg) |

---

## Vision Encoding Architecture

### Per-view compute budget

Each view (local 640×640 or global 1024×1024) goes through a two-stage vision
encoder:

| Stage | Model | Params | ~FLOPs/view |
|---|---|---|---|
| SAM Patch Embed | Conv2d 3→768, 16×16 kernel | 59K | 384K |
| SAM Transformer | 12 blocks, 768-dim, 12 heads | ~47M | ~1.2B |
| SAM Neck | Conv1x1→LN2d→Conv3x3→net2→net3, 1024 out | ~2M | ~200M |
| CLIP Embed+Pos | 1024-dim class+spatial embed | ~1K | ~250K |
| CLIP Transformer | 24 blocks, 1024-dim, 16 heads | ~86M | ~5.3B |
| Projector | Linear 2048→1280 | ~2.6M | ~5M |
| **Total per view** | | **~138M** | **~6.8B** |

For the current9-view workload (8 local +1 global): **~61B FLOPs**. At
~3 TFLOPS fp16 on Apple Silicon, the theoretical minimum is ~20s. The measured
4.3s indicates the batched views and MPS optimized paths are effective, but the
pipeline is memory-bandwidth-bound.

### Memory bandwidth floor

~150 MB of SAM/CLIP/projector weights are loaded repeatedly for9 views. At
~300 GB/s peak bandwidth on Apple Silicon: ~500ms minimum for weight loads
alone. This is the practical floor for vision encoding time.

### Workspace allocation between chunks

The vision pipeline processes local and global chunks sequentially with different
workspace shapes:

```
for each chunk (local then global):
  if chunk_kind != previous_kind:
    release scratch[VISION]          ← frees previous workspace
    reinit_for_chunk_kind(...)       ← allocates new workspace
    prebuild_mps_vision_ndarrays()   ← re-creates MPS descriptors
```

The `vision_workspace_capacity = 0.00 B` in the profile output confirms the
workspace is allocated and released within each shape group rather than being
persisted. This adds allocation overhead at the local→global boundary.

### Why the global_shape_group_command_wait dominates

The profiler shows:

```
metal.vision.chunk_processing                      4294 ms  (total)
metal.vision.global_shape_group_command_wait        4156 ms  (GPU sync)
metal.vision.chunk_encode                            132 ms  (CPU dispatch only)
```

CPU encodes all GPU commands in 132 ms, then blocks waiting for the GPU to
finish in a single command buffer. The 4156 ms breaks down as:

| Component | Time | Evidence |
|---|---|---|
| SAM transformer GPU compute | ~67 ms | `sam_transformer_batch` profile event |
| SAM neck GPU compute | ~23 ms | `sam_neck` profile event |
| CLIP + projector GPU compute | ~200–400 ms | estimate from FLOPs |
| **MPSNDArray JIT compilation** (first-use per shape) | **~500–1500 ms** | block 0 alone is 64.8 ms CPU; GPU-side compilation similar magnitude |
| **GPU page faults (mmap cold)** | **~2000–3000 ms** | 6.2 GiB model, 16 KB pages, ~0.01–0.1 ms per page fault |

**Page fault cost calculation:** The model is ~6.2 GiB mapped via
`newBufferWithBytesNoCopy:` from an mmap'd file. On first GPU access, every
unresided 16 KB page triggers a fault. At ~400K pages and ~5–10 µs per
minor fault (plus DSM promotion latency on Apple Silicon), the total easily
reaches 2–3 seconds. This is the dominant term.

**SAM block 0 is the first block to touch** the SAM QKV weight (~3.4 MB),
projection weight (~1.1 MB), and MLP weights (~9.4 MB) for hidden size 768.
Its 64.8 ms CPU time combines MPS JIT compilation with the setup overhead
of ~10 distinct first-use MPS matmul encodes (LayerNorm, Q/K/V, proj, MLP
lin1/lin2).

---

## Root Causes of Slow Vision Encoding

### 1. No vision pipeline prewarm

The engine calls `metal_prewarm_integrated_decoder_pipelines()` which compiles
all **decoder** Metal compute pipelines (rmsnorm, attention, MoE kernels, etc.)
ahead of time via `metal_get_pipeline()`. **No equivalent prewarm exists for
vision Metal pipelines.**

However, the dominant first-use cost is **not** Metal pipeline compilation
(Metal pipelines ARE cached by `metal_get_pipeline()` and individual vision
kernels are compiled on demand — fast). The dominant cost is
**MPSNDArrayMatrixMultiplication JIT kernel compilation** inside the MPS
framework. MPSNDArray creates internal GPU kernels lazily the first time
`encodeToCommandBuffer:` is called with a given shape; there is no public
MPS API to pre-compile these.

Every vision compute Metal pipeline is compiled on first GPU dispatch:

| Vision kernel family | Pipelines compiled lazily |
|---|---|
| SAM patch embed | 1 convolution kernel (`metal_get_pipeline`) |
| SAM add_abs_pos | 1 custom kernel |
| SAM transformer (×12 blocks) | LayerNorm custom kernels + **MPSNDArray QKV/proj/MLP matmuls**(~72 total, ~40 distinct shapes) |
| SAM neck | conv1x1, layernorm2d, conv3x3, net2 conv, net3 conv |
| CLIP embed_sam | 1 custom kernel |
| CLIP add_abs_pos | 1 custom kernel |
| CLIP pre layernorm | 1 custom kernel |
| CLIP transformer (×24 blocks) | LayerNorm + **MPSNDArray matmuls** (~144 total, ~24 distinct shapes) |
| CLIP/SAM concat | 1 custom kernel |
| Projector | 1 MPS matmul |

The first SAM block's 64.8 ms vs 0.23 ms for subsequent blocks directly
demonstrates this: the 64.8 ms includes on-demand MPSNDArray kernel JIT for
QKV, projection, and MLP matmuls with SAM hidden size 768. The Metal custom
kernels (LayerNorm, attention, etc.) compile quickly via `metal_get_pipeline()`
and are not the bottleneck.

### 2. No model-view GPU warmup (the blatant issue)

The 6.21 GiB model is mapped during `uocr_metal_context_map_model()` using
`newBufferWithBytesNoCopy:`. This creates Metal buffer wrappers around the
mmap'd file data, but **the GPU never touches these pages before the first
inference**.

When the vision command buffer executes, the GPU's first access to each page
of the SAM/CLIP/projector weight tensors triggers:

- Page faults and memory promotion in the Apple Unified Memory subsystem
- TLB/cache warmup for GPU page table walks

This affects not just the first vision dispatch but also the first decoder
prefill, which touches the remaining ~3 GiB of decoder weights (embedding
table, attention QKV/output projections, MoE expert weights).

The combined GPU page-fault cost can add **several seconds** to the first
inference.

#### The fix already exists — it's just not wired

The public C function `uocr_metal_context_warmup_model_views()`
(`src/backend/metal/uocr_metal.m:7478`) already implements exactly what is
needed: it dispatches a `uocr_touch_u32` GPU compute kernel that reads one
`uint32_t` per page from every mapped model view, forcing all pages resident
before inference begins. It is tested in `tests/test_metal.c:14047`.

**But it is never called.** The Python FFI (`src/unlimitedocr_c/ffi.py`) does
not expose it, the probe (`probes/e2e_generation_probe.py`) does not call it,
and the engine open path does not invoke it. The six-second `engine_open`
time compiles pipelines and creates MPS NDArray wrappers but touches zero
model data on the GPU.

```c
// Already exists, just needs to be called after map_model():
uocr_metal_context_warmup_model_views(ctx, 0 /* max bytes */, error, sizeof(error));
```

**Estimated impact:** Eliminates ~2–3 s of GPU page-fault stalls from vision
encoding and another ~1 s from prefill — roughly **halving `generate_native_s`
from 6.5 s to ~3 s** on first inference.

### 3. MPSNDArrayMatrixMultiplication lazy kernel compilation

Vision uses MPSNDArray for ~40 distinct matrix multiply configurations:

- SAM: Q, K, V, projection, MLP lin1, MLP lin2 — per block type (global vs window)
- CLIP: Q, K, V, output projection, MLP fc1, MLP fc2
- Projector

Each distinct `(rows, in_features, out_features)` tuple triggers a one-time
MPS internal kernel compilation on first `encodeToCommandBuffer:`. These
compilations are serialised inside the single vision command buffer's
execution. The compile cost is visible in the SAM block-0 timing:
the first MPS matmul encode on each distinct SAM hidden-size shape
(768→2304 QKV, 768→768 proj, 768→3072 MLP lin1, 3072→768 MLP lin2) each
compiles its MPS GPU kernel.

Unlike Metal compute pipelines (compiled via `metal_get_pipeline()` ->
`newComputePipelineStateWithFunction:`), MPSNDArray kernels have no
programmatic pre-compilation API. The only way to front-load the cost is
to run a dummy encode+wait per distinct shape during warmup.

**The existing `metal_prebuild_mps_vision_workspace_ndarrays()` creates MPS
NDArray wrappers but does NOT compile the matmul kernels** — that requires
actual `encodeToCommandBuffer:` calls.

### 4. Single command buffer serialises all overhead

The vision path encodes **all** operations for a shape group into one command
buffer, then calls `waitUntilCompleted` once:

```
vision chunk
  ├── sam_patch_embed (1 dispatch)
  ├── sam_add_abs_pos (1 dispatch)
  ├── sam_transformer (12 blocks × ~10 ops each = ~120 dispatches)
  ├── sam_neck (5 dispatches)
  ├── clip_embed_sam (1 dispatch)
  ├── clip_add_abs_pos (1 dispatch)
  ├── clip_pre_layernorm (1 dispatch)
  ├── clip_transformer (24 blocks × ~10 ops each = ~240 dispatches)
  ├── concat (1 dispatch)
  └── projector (1 dispatch)
  │
  └── commit + waitUntilCompleted  ← 4156 ms
```

Approximately **380 GPU dispatches** in one command buffer. Any single slow
dispatch (first-use compilation, page fault) stalls the entire command buffer.

Additionally, each MPS matmul creates its own `MTLComputeCommandEncoder`
(via `metal_compute_command_encoder()`) even though all share the same
command buffer. Each encoder creation and `[enc endEncoding]` adds ~5–10 µs
of CPU overhead. With ~217 MPS matmuls across SAM + CLIP, that's ~1–2 ms
of CPU encoding overhead — negligible.

Splitting into smaller batches with intermediate commits would let later
dispatches benefit from earlier compilations while earlier waits overlap.

### 5. Workspace thrash between shape groups

At the local→global shape-group boundary, the code:

```c
if (active_chunk_kind != (int)chunk->kind) {
    uocr_metal_context_release_scratch(ctx, UOCR_METAL_SCRATCH_VISION);  // free
    metal_vision_workspace_init_for_chunk_kind(...);                        // realloc
    metal_prebuild_mps_vision_workspace_ndarrays(...);                      // rebuild MPS cache
}
```

This releases and re-allocates the entire `UOCR_METAL_SCRATCH_VISION` buffer
(~138 MiB peak), and invalidates the MPS workspace NDArray cache, forcing a
rebuild for the new shape. The allocation itself is fast (MTLBuffer creation),
but the MPS NDArray rebuild iterates over all matmul I/O pairs for the new
shape group, touching dozens of cache entries.

While this is not the dominant cost, it adds measurable CPU time and prevents
keeping a warm MPS descriptor cache across shape groups.

---

## Prefill also affected

```
metal.prefill                     1326 ms
metal.prefill.command_wait        1324 ms  ← 99.8 % wait
```

The prefill command buffer also suffers from first-use compilation and
memory-page faults for decoder weights not touched during vision encoding.
Unlike vision, the decoder pipelines ARE prewarmed (`pipeline_prewarm` in
`generate_f16_impl`), but the prewarm happens *after* vision has already
paid the price. Moreover, model-view pages remain unwarmed.

---

## Decode loop

```
metal.decode_loop                  689 ms  (43 iterations)
metal.decode.command_wait          662 ms  (44 calls, ~15 ms avg)
metal.decode_iteration             674 ms  (43 calls)
metal.next_token_selection         664 ms  (44 calls)
```

The decode loop is well-pipelined: each iteration commits a small command
buffer with attention + MoE dispatches and waits. Per-iteration wait (~15 ms)
is dominated by actual GPU compute for 11 MoE layers + attention + LM head.
No excessive overhead — this is expected performance.

---

## Recommendations

### Immediate (quick wins)

1. **Wire `uocr_metal_context_warmup_model_views()` into the Python API** —
   the C function already exists and does exactly what's needed. Add it to:
   - `src/unlimitedocr_c/ffi.py` as an `Engine` method
   - The `Engine.__init__()` path after `map_model()` (called during `__enter__`)
   - `probes/e2e_generation_probe.py` before the first generation
   
   ```python
   # In Engine.open() or after model load:
   engine.warmup_model_views(max_bytes_per_view=0)  # 0 = warm whole model
   ```
   
   **Estimated impact:** Eliminates ~3–4 s of GPU page-fault stalls (vision +
   prefill). Halves `generate_native_s` from 6.5 s to ~3 s on first inference.

### High priority

2. **Add MPS matmul shape warmup** — after `metal_prebuild_mps_vision_workspace_ndarrays()`
   creates NDArray wrappers, issue a dummy `encodeToCommandBuffer:waitUntilCompleted`
   for each distinct MPS matmul shape that will be used in the vision pipeline.
   This front-loads MPSNDArray kernel JIT compilation so it doesn't stall the
   first inference. The ~40 distinct SAM/CLIP/projector shapes can be encoded
   in a single throwaway command buffer during engine init.

3. **Call `metal_get_pipeline()` for all vision kernel function names** during
   engine warmup — analogous to the decoder prewarm. This ensures the Metal
   compute pipelines (LayerNorm, attention, conv, etc.) are pre-compiled.
   Currently only decoder pipelines are prewarmed.

### Medium priority

4. **Split the vision command buffer** — commit and wait after SAM transformer
   completion, then encode CLIP + projector in a second command buffer. This
   lets CLIP MPS compilation overlap with SAM execution. The natural split
   point is after `sam_neck`. However, this requires keeping the SAM output
   (net3 features) alive across the split, which means the vision scratch
   buffer must persist between chunks.

### Low priority

5. **Persist vision scratch across shape groups** — instead of releasing
   `UOCR_METAL_SCRATCH_VISION` at the local→global boundary, allocate a single
   workspace large enough for the larger of the two shapes and reuse it. This
   avoids MTLBuffer reallocation and MPS NDArray cache rebuild. The current
   `vision_workspace_capacity = 0.00 B` in the profile shows the buffer is
   freed between shape groups.

---

## Measuring

Run the probe with JSON output for detailed analysis:

```sh
uv run probes/e2e_generation_probe.py --profile base --json
```

Key metrics to track per change:

- `metal.vision` total time
- `metal.vision.global_shape_group_command_wait` / `metal.vision.local_shape_group_command_wait`
- `metal.vision.chunk_encode`
- `metal.vision.sam_block.00` vs `metal.vision.sam_block.01` (first-use gap)
- `metal.prefill.command_wait`
- Total `metal.command_buffer_wait` count and sum
- `tokens_per_second_native`
