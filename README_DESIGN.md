# Architecture Design: unlimitedocr.c

## 1. Overview

**unlimitedocr.c** is a multi-backend inference engine for the
[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) model.  It
runs locally on Apple Silicon (Mac) with a native C/Metal core and a thin Python
API for OCR workflows.

### Design Goals

| Goal | Approach |
|------|----------|
| **On-device, private** | No network calls; model and inference run entirely on the local machine. |
| **Low latency** | Native C code, GPU execution via Apple Metal, no-copy model tensor views, and a bounded-memory vision pipeline. |
| **Memory aware** | Every allocation is tracked by category.  Requests are admitted or rejected based on a fixed memory budget, preventing OOM at runtime. |
| **Quantized** | An optional mixed-Q8_0 profile stores weight matrices as int8 + fp16 group scales (~½ model size).  Dequantization is fused inside the GPU kernels; activations, KV cache, and numerics-sensitive tensors stay fp16. |
| **Multi-backend** | A clean C ABI lets the same engine dispatch to different backends.  Today only **Metal** (Apple GPU) is production-ready; a **CPU ref** backend exists for validation and parity testing.  Future backends (CUDA, ANE, etc.) slot in behind the same interface. |
| **Portable Python API** | The user-facing `UnlimitedOCR` class accepts images as paths, URLs, bytes, PIL images, or base64 strings. |

### Two-Tier Architecture

```
┌─────────────────────────────────────────────┐
│              Python Frontend                │
│   Image loading, resizing, normalization,   │
│   tokenization, prompt assembly             │
│                    │ PreparedRequest        │
│                    ▼                        │
│                 C ABI                       │
│         Engine open / generate              │
│                    │                        │
│                    ▼                        │
│              Engine Core                    │
│   Request validation, memory admission,     │
│   vision scheduling, backend dispatch       │
│                    │                        │
│           ┌────────┴────────┐               │
│           ▼                 ▼               │
│        Metal             CPU Ref            │
│   (Apple GPU)      (validation-only)        │
│        + future backends                    │
└─────────────────────────────────────────────┘
```

Python handles the **user-facing** pieces—image I/O, resize/normalize,
tokenization, and text decoding.  The C library owns **model loading, memory
management, KV cache, logits processing, and GPU execution**.  The two halves
communicate through a stable C ABI defined in `unlimitedocr.h`.

---

## 2. High-Level Design

### 2.1 System Layers

| Layer | Files | Responsibility |
|-------|-------|----------------|
| **Python frontend** | `api.py`, `frontend.py`, `ocr.py`, `ffi.py` | Image loading, preprocessing, tokenization, prompt assembly, result decoding, ctypes bridge. |
| **C ABI** | `include/unlimitedocr.h` | Public API: engine open/close, generate, profiling, memory reporting. |
| **Engine core** | `src/core/uocr_api.c` | Request validation, memory estimation, admission control, vision scheduling, backend dispatch. |
| **Memory management** | `src/runtime/uocr_memory.c` | Per-category tracking, peak recording, budget enforcement, component-wise memory estimation. |
| **Vision runtime** | `src/runtime/uocr_vision.c` | View schedule planning, chunked execution, final feature assembly. |
| **Sequence runtime** | `src/runtime/uocr_sequence.c` | Prompt+generated token state. |
| **Model file** | `src/model/uocr_model_file.c` | Memory-mapped `.uocr` model parsing: header, sections, tensor directory. |
| **Metal backend** | `src/backend/metal/` | GPU context, buffer management, kernel pipelines, integrated fp16 decode and image generation. |
| **CPU ref backend** | `src/backend/cpu_ref/` | Pure C reference implementation for validation and correctness parity. |

### 2.2 Data Flow

```
User image (path/URL/bytes/PIL)
    │
    ▼
Python: load_user_image()
    │
    ▼
Python: prepare_image()
  • Resize to preset (1024×1024 base, or 1024×1024→crop 640×640 for gundam)
  • Normalize pixels to [-1, 1], NCHW layout
  • Tokenize prompt with <image> placeholder
  • Build PreparedRequest (input_ids, image_mask, views, config)
    │
    ▼
Python → C via ctypes (ffi.py → C ABI)
    │
    ▼
C: uocr_generate_prepared()
    │
    ├─ uocr_profile_begin_request()
    ├─ Build request limits (max_prompt_tokens, max_gen_tokens, etc.)
    │
    ├─ For each request in batch:
    │     ├─ uocr_validate_prepared_request()
    │     ├─ uocr_build_sequence_state()
    │     ├─ uocr_plan_vision_schedule()
    │     │     • Groups views into chunks (local crops first, then global)
    │     │     • Computes projected token counts per chunk
    │     └─ Aggregate max tokens, visual rows, chunk shapes, view sizes
    │
    ├─ uocr_estimate_runtime_memory_with_vision_chunk_shapes()
    │     • Estimates total GPU memory need
    │     • Rejects if over budget (detailed error with breakdown)
    │
    ├─ If text-only (no views, no image placeholders) ──────────┐
    │     └─ generate_metal_text_fp16()                         │
    │           ├─ ensure_metal_runtime_arenas_for_request()    │
    │           │     (grow KV cache / prompt / decoder arenas) │
    │           ├─ Allocate generated-token buffer              │
    │           └─ uocr_metal_context_generate_f16()            │
    │                                                           │
    └─ If image present (views or image placeholders) ──────────┘
          └─ generate_metal_image_fp16()
                ├─ uocr_build_sequence_state()  (validate image span)
                ├─ uocr_plan_vision_schedule()  (full schedule)
                ├─ Release existing runtime arenas
                ├─ Allocate generated-token buffer
                └─ uocr_metal_context_generate_image_f16_deferred_runtime()
                      ├── Metal: run SAM → CLIP → Projector on each chunk
                      │     (GPU scratch is per-chunk, bounded)
                      ├── Callback → prepare_metal_runtime_arenas_after_vision()
                      │     (deferred arena allocation sized after vision)
                      ├── Splice visual features into prompt arena
                      ├── Decoder prefill (all prompt tokens → KV cache)
                      ├── Decode loop (one token at a time):
                      │     ├── Embed last token
                      │     ├── For each decoder layer:
                      │     │     ├── RMS norm → QKV → RoPE
                      │     │     ├── KV append → flash attention → output proj
                      │     │     ├── RMS norm → MoE (router + shared + routed)
                      │     │     └── Residual add
                      │     ├── RMS norm → LM head → greedy argmax (fused on GPU)
                      │     └── Stop if EOS or max_new_tokens reached
                      └── Return generated tokens
    │
    ├─ uocr_result_create_from_generated()
    │     (copies generated token IDs into result struct)
    │
    └─ Return uocr_result
          │
          ▼
Python: _copy_result_tokens() → decode_generated_ids() → text
```

### 2.3 Memory Management Architecture

Memory is the central resource managed by the engine.  Every allocation belongs
to one of **ten categories**, each tracked with live/peak bytes:

| # | Category | Contents |
|---|----------|---------|
| 0 | Model Views | GPU-visible mappings of model tensor data (read-only, no copy) |
| 1 | KV Cache | Key-value cache for all decoder layers and batch slots |
| 2 | Prompt Embeddings | Embedded prompt token features |
| 3 | Vision GPU Workspace | SAM/CLIP/projector intermediate tensors |
| 4 | Vision Final Features | Concatenated visual feature rows for decoder |
| 5 | Vision Host Staging | CPU-side staging for visual features (if needed) |
| 6 | Decoder Scratch | Layer intermediate buffers (hidden states, attention) |
| 7 | MoE Scratch | Router logits, expert output combiners |
| 8 | Logits Readback | CPU-visible logits/token readback buffer |
| 9 | Transient Buffers | Short-lived allocations (generated token buffers, etc.) |

Before executing a request, the engine computes a **memory estimate** for every
component.  If the estimate exceeds the configurable `memory_budget_bytes`, the
request is **rejected with a detailed error** showing the breakdown.  This
guarantees that no runtime OOM can occur during inference.

The Metal backend further subdivides its GPU memory into:

- **Scratch buffers** (7 slots): reusable per-operation temporary storage
  (vision, decoder, MoE, logits, transient, vision-final, LM-head logits).
- **Runtime arenas** (7 slots): long-lived allocations sized to the batch/prompt
  capacity (KV cache, prompt embeddings, hidden ping-pong, router top-k,
  MoE intermediate, vision scratch, logits readback).

Runtime arenas are **grown on demand** when a request exceeds current capacity,
then tracked against the budget.  For image requests all arenas are allocated
together **after** the vision pipeline finishes, via a deferred callback — the
vision pipeline uses separate scratch buffers whose size is bounded by the
largest single chunk, so decoder arena sizing does not need to account for
peak vision memory simultaneously.

### 2.4 Vision Pipeline

The Unlimited-OCR model uses a **dual vision encoder**:

1. **SAM** (Segment Anything Model) — windowed attention, 12 blocks, hidden 768,
   patch size 16.  Produces multi-scale image features.
2. **CLIP** (ViT-L/14-style) — global attention, 24 blocks, hidden 1024.
   Takes SAM features as input and produces a sequence of visual tokens.

The vision pipeline processes image **views** (1024×1024 global view and/or
640×640 local crop views) in **chunks** to bound GPU memory:

- For the **"base"** profile: one global 1024×1024 view → SAM → CLIP → Projector.
- For the **"gundam"** (crop) profile: N local 640×640 crop views (SAM→CLIP→Projector)
  in row-major order, then one global 1024×1024 view.

Each chunk is projected through the encoder stacks independently into a reusable
scratch buffer, then scattered into the final `[visual_tokens × 1280]` feature
buffer.  This keeps encoder temporaries bounded by the largest single chunk.

### 2.5 Decoder Pipeline

The decoder is a **DeepSeek V2**-style transformer with:

- **12 layers**, hidden size 1280, 10 attention heads, head dim 128.
- **Multi-Head Attention (MHA)** with RoPE position encoding.
- **1 dense layer** followed by **11 MoE layers**.
- **64 routed experts** with top-6 routing, **2 shared experts**, and
  intermediate size 896 (routed) / 1792 (shared).
- **Ring-buffer KV cache** (capacity = max_prompt_tokens, with a ring window for
  generated tokens).

The decode loop runs entirely on GPU: prompt tokens are prefilled in one pass,
then each generated token runs a full forward pass through all layers, followed
by a fused LM-head projection + greedy argmax.

---

## 3. Model Architecture

### 3.1 Unified Vision-Language Model

The `baidu/Unlimited-OCR` model combines a vision encoder and a language decoder
into a single end-to-end architecture:

```
Input Image
    │
    ▼
┌────────────────────────────────────────┐
│  SAM Vision Encoder (12 blocks)        │
│  • Windowed attention (window 14×14)   │
│  • Global attention at blocks 2,5,8,11 │
│  • Neck conv (256ch) + MLP → features  │
└────────────────┬───────────────────────┘
                 │ SAM features
                 ▼
┌────────────────────────────────────────┐
│  CLIP Vision Encoder (24 blocks)       │
│  • Full global attention               │
│  • Pre-norm, QKV projection           │
│  • MLP intermediate 4096               │
└────────────────┬───────────────────────┘
                 │ CLIP features
                 ▼
┌────────────────────────────────────────┐
│  Projector MLP                         │
│  • 2048 → 1280 (hidden)               │
└────────────────┬───────────────────────┘
                 │ Visual tokens [N × 1280]
                 │  (+ image_newline tokens, view_separators)
                 ▼
┌────────────────────────────────────────┐
│  Decoder (DeepSeek V2, 12 layers)      │
│  • Layer 0: dense MLP                  │
│  • Layers 1-11: MoE                    │
│  • MHA with RoPE                       │
│  • Ring-buffer KV cache                │
│  • RMS norm every sub-layer            │
└────────────────┬───────────────────────┘
                 │ Hidden → LM head
                 ▼
            Output logits
```

### 3.2 Constants (`uocr_constants.h`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `UOCR_VOCAB_SIZE` | 129280 | Total vocabulary (LM head output size) |
| `UOCR_HIDDEN_SIZE` | 1280 | Decoder hidden dimension |
| `UOCR_DECODER_LAYERS` | 12 | Number of transformer layers |
| `UOCR_ATTENTION_HEADS` / `UOCR_KV_HEADS` | 10 / 10 | MHA head counts |
| `UOCR_HEAD_DIM` | 128 | Dimension per attention head |
| `UOCR_MAX_POSITIONS` | 32768 | Maximum supported sequence length |
| `UOCR_GENERATED_RING_WINDOW` | 128 | Ring-buffer window for generated tokens |
| `UOCR_DENSE_FIRST_LAYERS` | 1 | Number of dense layers at start |
| `UOCR_ROUTED_EXPERTS` | 64 | Number of routed MoE experts |
| `UOCR_MOE_TOP_K` | 6 | Top-k experts selected per token |
| `UOCR_MOE_EXPERT_INTERMEDIATE` | 896 | Hidden dim inside routed experts |
| `UOCR_MOE_SHARED_EXPERTS` | 2 | Number of shared experts |
| `UOCR_MOE_SHARED_INTERMEDIATE` | 1792 | Hidden dim inside shared experts |
| `UOCR_GLOBAL_VIEW_SIZE` | 1024 | Global image view resolution |
| `UOCR_LOCAL_VIEW_SIZE` | 640 | Local crop view resolution |
| `UOCR_VISION_PATCH_SIZE` | 16 | SAM/CLIP patch size |
| `UOCR_DOWNSAMPLE_RATIO` | 4 | Downsample ratio for visual token grid |
| `UOCR_CLIP_HIDDEN_SIZE` | 1024 | CLIP encoder hidden dimension |
| `UOCR_CLIP_BLOCKS` | 24 | Number of CLIP transformer blocks |
| `UOCR_SAM_HIDDEN_SIZE` | 768 | SAM encoder hidden dimension |
| `UOCR_SAM_BLOCKS` | 12 | Number of SAM transformer blocks |

### 3.3 The `.uocr` File Format

The model is distributed as a **single `.uocr` file** — a custom packed format
designed for **zero-copy memory mapping**:

```
┌──────────────────────┐
│ File Header (112 B)  │  magic "UOCR", version, qprofile, offsets
├──────────────────────┤
│ Section Directory    │  array of {type, offset, size, alignment}
├──────────────────────┤
│ Config Record        │  vocab size, hidden dim, layer count, MoE params
├──────────────────────┤
│ Tensor Directory     │  header + N tensor entries (140 B each)
├──────────────────────┤
│ Tokenizer Metadata   │  hash, vocab sizes, BOS/EOS/PAD/IMAGE ids
├──────────────────────┤
│ Provenance Record    │  source safetensors info, config hashes
├──────────────────────┤
│ Tensor Payloads      │  aligned fp16 / Q8_0 tensor data
└──────────────────────┘
```

The file carries a **quantization profile** (`qprofile`): `fp16` (all tensors
fp16) or `mixed-q8_0`.  In the mixed profile, selected rank-2 weight matrices
are stored as **Q8_0**: row-major int8 qweights plus one fp16 scale per
64-element group along the input dimension (`payload_offset/size` for
qweights, `scale_offset/size` for scales in the same tensor entry).  Which
modules the converter may quantize is gated by `configs/quant-cfg.yaml` — a
module is enabled only after fused runtime kernels exist and it passed
end-to-end QA.

**Key design decisions:**

- **Aligned offsets** (4096 B for sections, 16 B for tensor payloads) enable
  direct GPU buffer aliasing without copying.
- **Tensor entries** carry rich metadata: family (attention, MoE, vision, etc.),
  projection (Q/K/V/O, gate/up/down, etc.), layer index, logical & physical
  shape, and quantization profile.
- **No-copy loading**: the file is `mmap`'d, then Metal buffers are created that
  point directly at the file pages.  Model tensor data is never copied into
  separate allocations.
- **Provenance**: the file records the original Hugging Face safetensors paths,
  config hash, and converter version for reproducibility.

---

## 4. Metal Backend

### 4.1 GPU Memory Model

The Metal backend divides GPU memory into three tiers:

| Tier | Allocation | Lifetime | Contents |
|------|-----------|----------|----------|
| **No-copy model views** | `MTLBuffer` aliasing `mmap`'d file pages | Engine lifetime | All model weight tensors (read-only) |
| **Scratch buffers** | `MTLBuffer` per slot | Per-operation, reused | SAM/CLIP intermediates, decoder hidden states, MoE scratch, logits |
| **Runtime arenas** | `MTLBuffer` per slot | Sized to max batch/prompt capacity | KV cache, prompt embeddings, hidden ping-pong, router top-k |

**Scratch slots** (`uocr_metal_scratch_slot`):

| Slot | Usage |
|------|-------|
| 0 = VISION | SAM/CLIP/projector per-chunk temporaries |
| 1 = DECODER | Decoder layer intermediate hidden states |
| 2 = MOE | MoE expert intermediate activations |
| 3 = LOGITS | CPU-visible logits buffer |
| 4 = TRANSIENT | Short-lived kernel temporaries |
| 5 = VISION_FINAL | Concatenated visual features for decoder |
| 6 = LM_HEAD_LOGITS | LM head projection output |

**Runtime arena slots** (`uocr_metal_runtime_arena_slot`):

| Slot | Contents | Sizing |
|------|----------|--------|
| 0 = KV_CACHE | Key-value cache, all layers | `layers × batch × prompt_capacity × kv_heads × head_dim × 2` |
| 1 = PROMPT_EMBEDDINGS | Embedded prompt tokens | `batch × prompt_capacity × hidden_size` |
| 2 = HIDDEN_PINGPONG | Two hidden state buffers | `2 × batch × prompt_capacity × hidden_size` |
| 3 = ROUTER_TOPK | MoE router top-k indices/scores | `batch × prompt_capacity × top_k` |
| 4 = MOE_INTERMEDIATE | MoE expert intermediate | Per-batch sized to max tokens |
| 5 = VISION_SCRATCH | Vision workspace | Sized after vision schedule |
| 6 = LOGITS_READBACK | CPU-visible logits | `batch × vocab_size` |

### 4.2 Vision Execution

The Metal vision path runs the full SAM→CLIP→Projector pipeline on GPU:

```
For each vision chunk:
  1. Upload view pixels (fp16 NCHW, already normalized)
  2. SAM encoder:
     a. Patch embedding (conv)
     b. For each SAM block (0-11):
        - Global attention at blocks 2, 5, 8, 11; window attention otherwise
        - MLP + residual
     c. Neck conv (reduce channels to 256)
     d. Upsample + net convs → output features
  3. CLIP encoder:
     a. Position embedding + class token
     b. For each CLIP block (0-23):
        - Layer norm → QKV projection → attention → out projection
        - Layer norm → MLP (fc1+fc2)
  4. Projector MLP (2048 → 1280)
  5. Write projected rows into final visual feature buffer
```

The final feature buffer interleaves image-newline and view-separator tokens in
the order expected by the decoder.  Once vision is complete, the visual features
are spliced directly into the prompt embedding arena (no host round-trip).

### 4.3 Decoder Execution

**Prefill** processes all prompt tokens in one forward pass:

```
For each decoder layer:
  • RMS norm( hidden ) → QKV projection (attention)
  • Apply RoPE to Q and K
  • Flash attention: Q × K cache → softmax → V cache → output
  • Output projection + residual
  • RMS norm → MoE:
      ┌─ Router: linear → softmax → top-6 expert selection
      ├─ Shared expert: gate+up → SiLU → down (2 shared experts)
      ├─ Routed expert: for each selected expert, gate+up → SiLU → down
      └─ Combine: shared_output + sum(routed_outputs * routing_weight)
  • Residual add
```

**Decode** (autoregressive, one token at a time):

```
For each generated token:
  1. Embed token (lookup table → hidden)
  2. For each layer:
     a. RMS norm → QKV → RoPE
     b. Append K,V to ring-buffer KV cache
     c. Flash attention over all cache positions (or ring window)
     d. Output projection + residual
     e. MoE (same as prefill)
  3. RMS norm → LM head → greedy argmax (fused on GPU) → next token
  4. If EOS or max_new_tokens → stop
```

### 4.4 Metal Kernels

The Metal shader sources live in `src/backend/metal/kernels/`.  Each file
contains one or more related GPU functions:

| File | Kernels |
|------|---------|
| `common.metal` | Shared constants, function constants, threadgroup helpers |
| `attention_prefill.metal` | Flash attention for prompt tokens |
| `attention_decode.metal` | Flash attention for single-token decode |
| `attention.metal` | Shared QKVO projection kernels |
| `dense.metal` | Linear layers (matrix multiply + bias) |
| `norm.metal` | RMS norm, layer norm |
| `moe.metal` | Router, gated MLP (shared + routed experts) |
| `rope.metal` | Rotary position embedding |
| `embedding.metal` | Token embedding lookup |
| `embedding_q8.metal` | Fused Q8 token-embedding gather |
| `attention_q8.metal`, `dense_q8.metal`, `moe_q8.metal` | Fused Q8 decode projection/MLP/MoE variants |
| `decode_gemv_q8.metal` | Simdgroup-per-row Q8 GEMV family for single-token decode |
| `gemm_q8.metal` | Tiled simdgroup-MMA Q8 GEMM (prefill + vision) and expert-bucketed MoE prefill |
| `lm_head_q8.metal` | Fused Q8 LM-head projection + argmax |
| `sam.metal` | SAM patch embedding |
| `sam_attention.metal` | SAM window attention with relative position bias |
| `sam_conv.metal` | SAM conv layers |
| `sam_position.metal` | SAM position embedding |
| `sam_window.metal` | SAM window reshape utilities |
| `clip.metal` | CLIP encoder blocks |
| `clip_sam.metal` | Shared SAM/CLIP pointwise ops |
| `kv_cache.metal` | KV cache read/write, ring buffer management |
| `layout.metal` | Layout conversion helpers |
| `prompt_assembly.metal` | Image feature splice, prompt embedding assembly |
| `sampling.metal` | Argmax, top-k, top-p sampling |
| `uocr_smoke.metal` | Diagnostic smoke-test kernels |

The engine uses **function constants** (`[[function_constant(index)]]`) to
compile specialized pipeline variants at runtime for the exact model
architecture (hidden size, head count, ring window, etc.), avoiding runtime
branches inside hot kernels.

### 4.5 Q8 Quantization Support

The Metal backend executes the **mixed-Q8_0** model profile with fused
quantized kernels.  All rank-2 weight matrices are quantized — the full
decoder path (token embedding, attention Q/K/V/O, dense MLP, shared and
routed MoE experts, LM head) and the full vision path (visual projector, CLIP
attention QKV/O and MLP fc1/fc2, SAM attention QKV/O and MLP lin1/lin2).
Norms, biases, position/relative-position embeddings, the MoE router, the
rank-4 SAM conv/patch tensors, and all activations and the KV cache stay fp16.

Design rules (see `docs/howto_quant.md` for the full guide):

- **Fused dequantization.**  No kernel materializes a dequantized weight
  buffer; int8 qweights and fp16 group scales are read and dequantized inside
  the same kernel that performs the dot product (registers for GEMV,
  threadgroup tiles for GEMM).  MPS matmul is never fed dequantized copies.
- **Two kernel regimes, selected per dispatch.**  Single-token decode uses
  simdgroup-per-row GEMV kernels (`decode_gemv_q8.metal`: char4 weight /
  half4 activation loads, `simd_sum` only — 45–156 GB/s effective weight
  bandwidth, the LM head at full streaming bandwidth).  Multi-row prefill and
  vision batches use a tiled simdgroup-MMA GEMM (`gemm_q8.metal`: 64×32×32
  tiles, fp16 MMA with fp32 accumulators, each weight tile dequantized once
  per 64-row block — ~2.2 TFLOP/s, ≈ 75% of MPS fp16 on the same shapes
  while reading half the bytes).
- **Expert-bucketed MoE prefill.**  Routed-expert prefill buckets
  (token, expert) pairs by expert on GPU, then runs the tiled GEMM per bucket
  with gathered rows (~18× faster than per-pair GEMV at 1024 prompt tokens).
- **Cfg-gated rollout.**  `configs/quant-cfg.yaml` lists the runtime-safe
  (family, projection) modules; the converter quantizes only those, the
  loader validates Q8 metadata per role, and a quantized tensor reaching a
  call site without a fused kernel fails at bind time rather than falling
  back silently.

### 4.6 Orchestration


The top-level Metal generation paths are:

- **`uocr_metal_context_generate_f16()`** — text-only (no image views).  Runs
  prompt prefill then decode loop entirely on GPU.
- **`uocr_metal_context_generate_image_f16()`** — image-to-text.  Runs the
  full vision pipeline first, then allocates decoder runtime arenas (deferred,
  since vision scratch size is not known until the schedule is built), then
  splices visual features into the prompt arena and runs decode.
- **`uocr_metal_context_generate_image_f16_deferred_runtime()`** — variant that
  accepts a callback to prepare runtime arenas after vision completes, used by
  the engine core to coordinate memory tracking across the vision→decoder
  boundary.

### 4.7 Profiling & Diagnostics

The backend supports runtime profiling via `uocr_profile_state`.  Every
significant operation records a named event with duration, call count, min/max.
Profiling is enabled by setting `UOCR_PROFILE=1` or passing `profile=1` in
engine options.  The report includes:

- Per-event timing (calls, total/min/max ms)
- Metal buffer allocation count and bytes
- Metal command buffer and encoder counts
- Per-category memory live/peak bytes
- Vision workspace high-water mark

---

*This document describes the architecture as of unlimitedocr.c v0.3.0.  The
multi-backend design intentionally separates frontend, engine core, and backend
concerns so that new backends (CUDA, ANE, Vulkan, etc.) can be added by
implementing the backend interface without modifying the shared C core or Python
frontend.*
