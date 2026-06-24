# unlimitedocr.c architecture

This is a refined architecture for a narrow, Metal-first C inference engine for `baidu/Unlimited-OCR`. The engine should be model-specific: the converter absorbs Hugging Face naming/layout complexity, and the runtime executes a small number of fixed Unlimited-OCR stages.

## Refined design commitments

1. **Two contracts, not a generic graph runtime**
   - Offline contract: HF config/tokenizer/safetensors -> validated `.uocr` file.
   - Online contract: Python-prepared OCR request (`input_ids`, image mask/spans, preprocessed views) -> vision features -> prompt embeddings -> decoder prefill/decode.

2. **Converter owns model complexity**
   - Tensor name mapping, transposition, expert packing, qtype selection, alignment padding, tokenizer metadata, and provenance live in the converter.
   - Runtime uses stable tensor ids and prepacked tensor slabs.

3. **Python frontend, C inference core**
   - Python owns image loading/preprocessing, prompt rendering, tokenization, output token decoding, and user-facing ergonomics.
   - C owns model loading, Metal/CUDA inference, KV cache, generation loop, logits processors, and prepared-request batching.
   - A native tokenizer/image frontend can be added later for standalone CLI use, but it is not on the critical path.

4. **Metal first, but backend boundary is narrow**
   - Shared C owns prepared-request validation, batching policy, generation state, and no-repeat-ngram logic.
   - Metal/CUDA own buffers, kernels, command scheduling, and backend-specific scratch.

5. **Correctness path is mandatory**
   - `fp16` `.uocr` and Python-dumped intermediates come before quantization.
   - Decoder can be validated with Python-dumped image embeddings before the vision stack is ported.

6. **Dynamic q8/q4 is per tensor/expert/projection**
   - MoE router weights (`mlp.gate.weight`, not expert `gate_proj`), norms, biases, positional tables, newline/separator embeddings stay fp16.
   - Q4_K is used only where inner dimensions are aligned or explicitly padded.
   - Calibration promotes monotonically: `q4 -> q8 -> fp16`.

## Proposed source layout

```text
include/unlimitedocr.h          prepared-request C API
python/unlimitedocr_c/          Python frontend: PIL/tokenizer/preprocess/decoding/bindings
src/core/                       errors, logging, spans, mmap, allocators, tensor views
src/model/                      fixed Unlimited-OCR dimensions and tensor registry
src/convert/                    safetensors/config/tokenizer-metadata -> .uocr
src/quant/                      Q8_0/Q4_K/custom padding/calibration helpers
src/runtime/                    prepared request validation, session, scheduler, generation, logits processors
src/backend/cpu_ref/            slow correctness backend / tensor checks
src/backend/metal/              Objective-C runtime + Metal kernels
src/backend/cuda/               later CUDA backend
src/frontend_native/            optional later native tokenizer/image frontend
tools/uocr-convert              model converter
tools/uocr-calibrate            dynamic quant calibration/promotions
tools/uocr-dump                 inspect .uocr files
tools/uocr-ref-dump             Python-side golden tensor dumper
tests/                          Python frontend, converter, prepared-request, parity tests
probes/                         one-off perf/parity probes
```

## Build system

Use **plain CMake first**, with **uv** for Python development. Add **scikit-build-core later** when the C API and install layout are stable enough to package.

Rationale:

- CMake is the real native build system: C, Objective-C/Metal, CUDA later, tools, CTest tests, install layout.
- `uv` remains the Python environment/test/package command runner.
- The Python wrapper should initially load the build-tree shared library with `ctypes`/`cffi`, avoiding CPython extension complexity.
- `scikit-build-core` is useful for Python packaging, but it should not be required for day-one native bring-up.
- Avoid C++ in the inference core. CUDA later may require `.cu` files compiled as CUDA/C++, but those should expose a narrow `extern "C"` backend boundary.

Phased build plan:

1. **Native bring-up: CMake only**
   - Build `libunlimitedocr.dylib` / `.so`, native tools, and CTest tests directly with CMake.
   - Debug native compiler/linker/Metal issues without a Python packaging layer in the way.
2. **Python wrapper over build-tree library**
   - `python/unlimitedocr_c/ffi.py` locates `build/.../libunlimitedocr.dylib` via an env var or a small search path.
   - Python frontend can develop prepared-request logic immediately.
3. **Packaging phase: add scikit-build-core**
   - Once the C ABI, resource paths, and Metal shader strategy are stable, switch `pyproject.toml` to `scikit-build-core` so `uv pip install -e .` and wheels build/package the native library cleanly.

Native targets:

```text
uocr_core          object/static library: model loader, runtime, quant, CPU ref helpers
uocr_metal         Objective-C Metal backend, macOS only
uocr_cuda          CUDA backend, later, optional
unlimitedocr       shared library exporting include/unlimitedocr.h
tools/uocr-dump    native model inspection tool
tests/*            native unit/parity probes via CTest
```

Python package layout:

```text
python/unlimitedocr_c/__init__.py
python/unlimitedocr_c/frontend.py       PIL + tokenizer + prepared request creation
python/unlimitedocr_c/ffi.py            ctypes/cffi loader for libunlimitedocr
python/unlimitedocr_c/lib/              installed dylib/so and Metal resources, later
python/unlimitedocr_c/tools/            convert/calibrate/ref-dump CLIs
```

### scikit-build-core pros/cons

Pros:

- Python install/build commands can drive the CMake build.
- Editable installs and wheels can bundle the native library and Metal resources.
- Avoids custom `setup.py` build glue.
- Does not require `pybind11`; we can still expose/load a plain C ABI.

Cons:

- Adds another layer while native code is still changing.
- Native dylib packaging on macOS needs care: `@rpath`, framework links, resource paths, possible codesigning/notarization later.
- CUDA wheels are complicated; local CUDA builds are easier than portable binary wheels.
- Build errors can be harder to debug through Python packaging than direct CMake.

Recommended future `pyproject.toml` direction after phase 3:

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "setuptools-scm>=8"]
build-backend = "scikit_build_core.build"

[project]
dependencies = [
  "numpy>=2",
  "pillow>=10",
  "tokenizers>=0.20",
  "safetensors>=0.4",
]

[tool.scikit-build]
cmake.version = ">=3.27"
cmake.build-type = "RelWithDebInfo"
wheel.packages = ["python/unlimitedocr_c"]
```

CMake options:

```text
UOCR_ENABLE_METAL=AUTO    macOS Metal backend
UOCR_ENABLE_CUDA=OFF      later CUDA backend
UOCR_ENABLE_CPU_REF=ON    correctness backend/tests
UOCR_BUILD_TOOLS=ON
UOCR_BUILD_TESTS=ON
UOCR_METAL_RUNTIME_COMPILE=ON   DS4-style source compile during development
UOCR_METAL_PRECOMPILE=OFF       optional .metallib for release wheels
UOCR_SANITIZE=OFF               asan/ubsan for CPU-ref/debug builds
```

Metal shader strategy:

- Development default: package `.metal` sources and compile at runtime like DS4. This keeps device feature decisions simple and avoids fragile build-time SDK assumptions.
- Release option: CMake can precompile a `unlimitedocr.metallib` with `xcrun metal`/`metallib`; runtime falls back to source compilation if the metallib is missing or incompatible.
- Python should pass the package/resource directory or source-tree directory into `uocr_engine_open`, so the C library can find `.metal` sources / `.metallib` independent of working directory.

Typical commands during phases 1-2:

```sh
# Python/dev environment
uv sync --extra dev

# Native debug build
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DUOCR_ENABLE_METAL=ON
cmake --build build/debug -j
ctest --test-dir build/debug --output-on-failure

# Python tests using build-tree libunlimitedocr
UOCR_LIBRARY_PATH=$PWD/build/debug/libunlimitedocr.dylib uv run pytest
```

Typical command after phase 3:

```sh
uv pip install -e .
uv run pytest
```

A small top-level `Makefile` may be useful as a command wrapper (`make build`, `make test`, `make wheel`), but it should delegate to CMake/uv rather than become the real build system.

## Public API shape

Keep the C API low-level and prepared-request-oriented. Python provides the user-facing API.

```c
typedef struct uocr_engine uocr_engine;
typedef struct uocr_result uocr_result;

typedef enum {
    UOCR_PIXEL_F16_NCHW,
    UOCR_PIXEL_F32_NCHW,
} uocr_pixel_format;

typedef enum {
    UOCR_VIEW_GLOBAL,
    UOCR_VIEW_LOCAL,
} uocr_view_kind;

typedef struct {
    const void *pixels;          // contiguous [3,H,W], normalized to [-1,1]
    uint32_t width;
    uint32_t height;
    uocr_pixel_format format;
    uocr_view_kind kind;
} uocr_image_view;

typedef struct {
    const int32_t *input_ids;    // includes BOS and image placeholder ids
    const uint8_t *image_mask;   // 1 where input_ids is an image placeholder
    uint32_t n_tokens;

    const uocr_image_view *views;
    uint32_t n_views;
    uint32_t crop_grid_w;        // 1 for non-crop/global-only
    uint32_t crop_grid_h;

    uint32_t max_new_tokens;
    uint32_t no_repeat_ngram_size;
    uint32_t no_repeat_window;
} uocr_prepared_request;

typedef struct {
    const char *model_path;
    const char *backend;         // "metal", later "cuda", "cpu-ref"
    uint32_t max_batch;
    uint32_t max_prompt_tokens;
    uint32_t max_gen_tokens;
} uocr_engine_opts;

uocr_engine *uocr_engine_open(const uocr_engine_opts *opts);
void uocr_engine_close(uocr_engine *e);
int uocr_engine_memory_report(const uocr_engine *e, uocr_memory_report *out);
int uocr_generate_prepared(uocr_engine *e,
                           const uocr_prepared_request *reqs,
                           uint32_t n,
                           uocr_result **out);
const int32_t *uocr_result_tokens(const uocr_result *r, uint32_t index, uint32_t *n_tokens);
void uocr_result_free(uocr_result *r);
```

The Python package wraps this as `ocr.generate(image, prompt, preset=...)`, decodes output token ids with the HF tokenizer, exposes memory estimates/reports for admission-control debugging, and handles streaming text later by receiving generated token ids incrementally.

## `.uocr` model file

Use a simple mmap-friendly binary file rather than loading HF files at runtime.

Required sections:

1. **Header**
   - magic/version/endian/alignment
   - model hash/source model id
   - qprofile id: `fp16`, `dyn-q8`, `dyn-q4`, custom
2. **Config block**
   - decoder constants: vocab `129280`, hidden `1280`, layers `12`, heads `10`, head dim `128`, RoPE theta `10000`, ring window `128`
   - MoE constants: dense first layer `1`, routed experts `64`, top-k `6`, expert dim `896`, shared experts `2`
   - vision constants: SAM/CLIP dimensions, patch size `16`, downsample ratio `4`, supported image sizes
3. **Tensor directory**
   - stable tensor id / family / layer / expert / projection
   - logical shape and physical packed shape
   - dtype/qtype
   - byte offset/size/alignment
   - q block size, scale/min offsets if separate
   - qtype reason/promote reason
4. **Packed tensor payloads**
   - fp16 payload arena
   - Q8_0 arena
   - Q4_K / padded-Q4_K arena
   - optional calibration/imatrix arena
5. **Tokenizer metadata / optional payload**
   - tokenizer file hash and expected special ids (`BOS=0`, `EOS=1`, `PAD=2`, `<image>=128815`)
   - optional embedded tokenizer payload for future standalone/native frontend
   - C inference does not require tokenizer tables in v1 because Python owns tokenization/decoding
6. **Provenance**
   - HF commit/file hashes
   - converter version
   - calibration corpus id and thresholds

### Tensor registry and packing

Runtime should use enum-like tensor ids, not names. Suggested families:

```text
TOK_EMBED, LM_HEAD, FINAL_NORM
LAYER[i].ATTN.{Q,K,V,O}
LAYER[i].NORM.{INPUT,POST_ATTN}
LAYER[0].DENSE_MLP.{GATE,UP,DOWN}
LAYER[1..11].MOE.ROUTER
LAYER[1..11].MOE.EXPERTS.{GATE,UP,DOWN}
LAYER[1..11].MOE.SHARED.{GATE,UP,DOWN}
VISION.SAM.*, VISION.CLIP.*, PROJECTOR, IMAGE_NEWLINE, VIEW_SEPARATOR
```

Routed experts are packed as interleaved expert-major slabs for selected-expert kernels:

```text
[layer][expert][projection: gate, up, down][out_row][packed_input]
```

This keeps each expert's `gate_proj`, `up_proj`, and `down_proj` payloads contiguous; Metal can fetch selected expert `gate` and `up` together for fused projection and use the same expert stride for the down projection.

### Quantized physical shapes

Store both logical and physical input width. This is important because DS4/GGML Q4_K requires an inner dimension multiple of `256`:

- Q4_K-aligned examples: attention projections `1280`, LM head input `1280`, routed expert `gate/up` input `1280`, shared expert down input `1792`, projector input `2048`.
- Not Q4_K-aligned: routed expert `down_proj` input `896`, dense layer-0 `down_proj` input `6848`.

Initial policy should keep unaligned down projections in `Q8_0`, unless the converter emits `PADDED_Q4_K` and kernels zero-pad activations to the physical width. Converter plans expose `logical_input_width`, `physical_input_width`, and `input_padding_width` explicitly for every quantized tensor; the `.uocr` tensor directory stores the same contract as the final dimension of `logical_shape` and `physical_shape`, with C helpers to read those widths for Metal kernels.

## Converter design

`uocr-convert` pipeline:

1. Load `config.json`, `processor_config.json`, `tokenizer.json` metadata, and safetensors.
2. Validate all expected tensor names and shapes.
3. Map HF names to tensor registry ids.
4. Convert BF16 -> fp16/f32 temporary rows.
5. Pack tensors into runtime layout:
   - routed expert slabs
   - optional gate/up paired layout
   - contiguous vision/decoder arenas
6. Apply qtype policy:
   - `fp16`: every weight fp16
   - `dyn-q8`: large linears/conv weights Q8_0, sensitive tensors fp16
   - `dyn-q4`: Q4_K where aligned and calibrated; Q8_0/fp16 elsewhere
7. Optionally precompute common interpolation tables:
   - SAM absolute pos for `64x64` and `40x40`
   - CLIP absolute pos for `16x16` and `10x10`
   - SAM relative-pos helpers for `40x40` if useful
8. Emit `.uocr` with provenance and qtype decisions.

Useful DS4 converter features to copy:

- dry-run tensor plan
- bounded streaming writer that never allocates a full-model temporary buffer
- per-prefix qtype overrides
- single-tensor compare
- imatrix/calibration metadata
- strict shape checking with clear failures

## Dynamic quantization policy

### `fp16`

Everything fp16 except integer metadata. This is the parity baseline.

### `dyn-q8`

- Large linear/conv tensors: Q8_0.
- MoE router weights (`mlp.gate.weight`, not expert `gate_proj`), norms, biases, positional tables, image newline/separator: fp16.
- LM head: Q8_0.
- Token embedding: Q8_0 or fp16 if embedding-output drift appears.
- Vision: Q8_0 except position/norm/bias tensors.

### `dyn-q4`

- Attention projections: Q4_K candidate, promote Q/K/V first if logits drift.
- Routed MoE experts:
  - `gate_proj`, `up_proj`: Q4_K default candidates.
  - `down_proj`: Q8_0 initially because input `896` is not Q4_K-aligned; optional padded-Q4_K later.
- Shared experts: Q8_0 first, then Q4_K where calibration allows.
- Dense layer-0 MLP:
  - `gate/up`: Q4_K candidates.
  - `down`: Q8_0 or padded/custom q4 because input `6848` is not Q4_K-aligned.
- LM head: Q8_0 initially; q4 only if generation parity survives.
- Vision: Q8_0 initially; selective q4 after vision feature drift is understood.
- Sensitive keep-list remains fp16.

Calibration records:

- tensor/module output RMSE and cosine
- per-layer hidden-state drift
- router top-6 agreement and selected expert frequency
- logits KL/top-k agreement
- generated markdown/layout agreement
- per-expert/per-projection error and traffic

Promotion is monotonic: `q4 -> q8 -> fp16`.

## Python tokenizer and prompt construction

The Python wrapper owns tokenization and output decoding in v1. It should use the HF `tokenizer.json`/`tokenizers` implementation, not a fresh C tokenizer. The C engine only receives token ids and an image-placeholder mask.

Tokenizer facts to validate in Python and `.uocr` metadata:

- BPE vocab `128000`, total vocab `129280`.
- Special ids: BOS `0`, EOS `1`, PAD `2`, `<image>` `128815`.
- Pre-tokenizer sequence is JoyAI/DeepSeek byte-level BPE; exactness comes from the HF tokenizer implementation.

Prompt construction must match the upstream Python code:

1. Apply `format_messages(..., sft_format="plain", system_prompt="")`; this effectively returns the user prompt text only.
2. Split on literal `<image>`.
3. Tokenize text pieces with `add_special_tokens=false`.
4. Insert repeated `128815` placeholders for the required visual token count.
5. Prepend BOS id `0` manually.
6. Do **not** append EOS to the prompt.
7. Pass `input_ids` plus a same-length `image_mask` to C.

For C runtime, placeholders are only a length/span contract. During prompt embedding assembly, write text embeddings and vision embeddings directly instead of doing `masked_scatter_`.

A native DS4-style tokenizer remains useful later for a standalone CLI, but it is intentionally out of scope for the first inference engine.

## Python image/view planning

The Python wrapper owns PIL-compatible image loading and preprocessing in v1:

- open image, EXIF transpose, RGB conversion
- `ImageOps.pad` / resize semantics
- dynamic crop-grid selection
- normalization with mean/std `(0.5,0.5,0.5)` into contiguous NCHW `float16` or `float32`

Single-image presets:

- `base`: `base_size=1024`, `image_size=1024`, `crop_mode=false`.
- `gundam`: `base_size=1024`, `image_size=640`, `crop_mode=true`.

Crop mode:

- If the source image is at most `640x640`, use crop ratio `[1, 1]`.
- Otherwise choose a dynamic grid up to 32 local `640x640` crops.
- Always create a padded global `1024x1024` view.
- Local crops are resized from the original image to `(640 * W, 640 * H)` then split row-major.

Python passes `uocr_image_view[]` to C. C runs the vision encoder and assembles visual features in the required order:

- Non-crop or multi-page image: `[global rows with newline, view_separator]`.
- Crop mode with real local crops: `[local rows with newline, global rows with newline, view_separator]`.

This order is more important than the repeated placeholder ids; all placeholders have the same token id.

## Runtime execution stages

### 1. Prepared request validation

Python has already loaded/preprocessed images and tokenized the prompt. C validates:

- `input_ids` length and `image_mask` length.
- Placeholder count equals the visual token count implied by views/crop grid.
- View dimensions/formats are supported (`1024x1024` global, `640x640` locals for gundam, `1024x1024` base).
- Prompt length and generation budget fit the engine's KV/scratch capacity.
- no-repeat-ngram config is valid.

Then C creates sequence state:

- prompt token count
- text token ranges
- image feature ranges
- max generation length
- no-repeat-ngram config

### 2. Vision encode

For each view:

- SAM-like ViT-B:
  - `16x16` stride-16 patch embed
  - 12 transformer blocks, window size 14 except global blocks `[2,5,8,11]`
  - neck/net convs to produce `1024`-channel downsampled map
- CLIP-like ViT-L:
  - consumes SAM downsampled feature map as patch embeddings
  - 24 transformer blocks, hidden `1024`, heads `16`, MLP `4096`
- Concatenate CLIP token features (without CLS) and SAM feature-map tokens: `1024 + 1024 = 2048`.
- Project `2048 -> 1280`.
- Add row newline embeddings and view separator according to feature order above.

Bring-up shortcut: allow a test mode where Python-dumped projected image embeddings are loaded from disk and fed directly into decoder prefill.

### 3. Prompt embedding assembly

Backend should create a contiguous `[total_prompt_tokens, 1280]` activation buffer:

```text
[BOS/text token embeddings before image]
[vision embeddings for image span]
[text token embeddings after image]
```

For multi-image, all image feature blocks are concatenated at the single `<image>` location, matching the Python multi-image path.

### 4. Decoder prefill

Run all prompt embeddings through 12 decoder layers:

- RMSNorm with fp32 variance.
- Llama-style MHA (`use_mla=false`): separate Q/K/V/O projections, 10 heads, head dim 128.
- RoPE over the full 128-dim head.
- Causal full-prompt attention.
- KV cache write.
- Layer 0 dense SwiGLU MLP.
- Layers 1-11 MoE:
  - router fp16/fp32 softmax
  - top-6 greedy, `sorted=false` equivalent; order does not matter for sum
  - no top-k renormalization (`norm_topk_prob=false`)
  - shared expert added to routed result

### 5. Decode loop

For active sequences:

- Decode one token per step initially.
- Compute logits with LM head.
- Apply no-repeat-ngram processor on CPU or GPU.
- Greedy argmax first; sampling later.
- Stop on EOS id `1` or max tokens.

## Memory management

Use DS4's memory discipline, scaled down for OCR: mmap model weights, preallocate runtime state, reuse scratch, and make memory budgets explicit before accepting a request.

### Weight memory

- `.uocr` should be opened once from `uocr_engine_opts.model_path` and kept mmap-backed for the engine lifetime.
- Account resident tensor-data bytes in the engine memory report and reject configured model-open budgets that cannot cover the mapped tensor-data view.
- On Metal, wrap host-VM-page-aligned ranges of the mmap as no-copy `MTLBuffer`s. Do not eagerly copy the full model into private buffers.
- Store per-tensor `{view, inner_offset}` after load so hot kernels do not search views.
- Keep the tensor payload separate from metadata/tokenizer so only tensor pages are exposed to Metal.
- Request Metal residency / perform a coarse model-view warmup when useful, but treat this as a startup hint, not a full prefetch.

### Runtime arenas

Allocate long-lived buffers at engine/session creation:

- KV cache per batch slot.
- Prompt embedding buffer sized by `max_prompt_tokens`.
- Decode hidden-state ping-pong buffers.
- Router/top-k buffers.
- MoE intermediate buffers for selected experts.
- Vision view buffers and vision scratch for a bounded view chunk.
- Logits buffer and next-token buffer.

Avoid allocation in the token loop. Add a debug allocation guard like DS4's CPU guard for decode/prefill hot paths.

### Scratch policy

- Use a small scratch allocator or explicit named scratch buffers, not arbitrary per-op allocations.
- Scratch buffers grow to high-water mark and are reused.
- Use Metal private storage for GPU-only scratch when possible; use shared storage only for CPU-filled/readback buffers.
- Keep CPU-filled transient buffers alive until the command buffer completes.
- Track live/peak bytes by category and print a memory report.
- Use the shared C `uocr_memory_tracker` categories (`model-views`, `kv-cache`, `prompt-embeddings`, `vision-scratch`, `decoder-scratch`, `moe-scratch`, `logits-readback`, `transient-buffers`) for backend-independent accounting before adding backend-specific details.

### KV cache budget

OCR has a simple Llama MHA KV cache. Per sequence token:

```text
2 K/V * 12 layers * 10 heads * 128 dim * 2 bytes = 61,440 bytes/token
```

Examples:

- `273` visual tokens -> ~16.8 MiB KV before text/generated tokens.
- Worst crop-mode prompts can be ~3.5k visual tokens -> ~215 MiB KV per sequence.
- Batch/context limits must reserve KV before running vision or prefill.

The cache layout is fixed per batch slot:

```text
[layer][slot][max_prompt_tokens + 128][kv_head=10][head_dim=128]
```

The prompt region is immutable after prefill; generated tokens append until the 128-token ring is full, then overwrite ring slots.

### Vision memory

Crop mode can create up to 32 local views plus one global view. Do not process all views blindly if memory is tight:

- Plan views on CPU.
- Process local crops in chunks.
- Append projected `[visual_tokens,1280]` features into the final feature buffer.
- Release/reuse SAM/CLIP scratch after each view chunk.

### Admission control

Before running a request/batch, estimate:

```text
resident weights
+ KV cache for max prompt/gen window
+ prompt embeddings
+ vision scratch for chosen view chunk
+ decoder/MoE scratch
+ logits/readback buffers
+ safety margin
```

Reject or downsize batches when the estimate exceeds a configured limit or the Metal backend default budget. When `uocr_engine_opts.memory_budget_bytes == 0`, Metal uses 95% of `recommendedMaxWorkingSetSize` if the device reports it; the runtime estimate already carries its own safety margin.

## What DS4 does for memory

`data/ds4/` is the main reference for memory policy:

- **Mmap model loading**: `ds4.c` maps the GGUF once, parses metadata/tensor descriptors, and leaves tensor bytes in the file mapping.
- **No-copy Metal model views**: `ds4_metal.m` wraps overlapping page-aligned slices of the mmap with `newBufferWithBytesNoCopy`, because full models can exceed `maxBufferLength`. Each tensor is guaranteed to live wholly inside one view.
- **Residency/warmup**: on newer macOS it optionally builds a Metal residency set for model views and runs a coarse touch kernel to move first-use VM/driver costs out of measured inference.
- **Exact view cache**: for paths needing exact small model ranges, DS4 caches no-copy range wrappers and evicts them when a byte budget is exceeded.
- **Preallocated scratch**: named global scratch buffers grow to the needed size and are reused; memory reports include scratch categories.
- **Transient buffer lifetime**: CPU-filled Metal buffers are retained in an array until command completion, avoiding use-after-free with asynchronous command buffers.
- **Runtime tensor accounting**: GPU tensor allocations/views are tracked with live and peak byte counters.

For Unlimited-OCR, copy the mmap/no-copy views, scratch reuse, transient lifetime, accounting, and admission-control ideas.

## Backend interface

The backend should expose stage-level calls, not generic ops:

```c
typedef struct {
    int  (*init)(uocr_backend *, const uocr_model *);
    void (*destroy)(uocr_backend *);
    int  (*upload_or_map_model)(uocr_backend *, const uocr_model *);

    int (*vision_encode)(uocr_backend *, const uocr_view_batch *, uocr_gpu_tensor *features);
    int (*assemble_prompt)(uocr_backend *, const uocr_prepared_batch *, const uocr_gpu_tensor *features);
    int (*prefill)(uocr_backend *, const uocr_prefill_batch *);
    int (*decode_one)(uocr_backend *, const uocr_decode_batch *, uocr_next_token_batch *out);
} uocr_backend_vtable;
```

The Metal backend owns MTL buffers, command queues, pipelines, argument buffers, scratch arenas, and KV cache buffers.

## Metal backend design

Core Objective-C objects:

- `uocr_metal_device`
- `uocr_metal_model_buffers`
- `uocr_metal_pipelines`
- `uocr_metal_scratch`
- `uocr_metal_kv_cache`
- `uocr_metal_command_plan`

Kernel priority order:

1. **Decoder fp16 text-only**
   - get rows embedding
   - RMSNorm
   - fp16 linear/matvec
   - RoPE
   - SDPA prefill/decode
   - SwiGLU
   - router softmax/top-6
   - LM head + argmax
2. **Decoder with Python image embeddings**
   - prompt assembly with image spans
   - full prefill parity
3. **Q8_0 decoder**
   - adapt DS4 dense/MoE Q8_0 kernels
4. **Q4_K decoder/MoE**
   - Q4_K for aligned tensors
   - q8/padded-q4 path for unaligned down projections
5. **Vision fp16/q8**
   - conv2d patch/neck/net kernels
   - SAM window/global attention with relative pos
   - CLIP attention/MLP
   - projector and visual-token formatting
6. **Batch optimization**
   - packed prefill
   - active decode compaction
   - expert grouping across requests

Precision rules:

- Store activations fp16 where possible.
- Accumulate matmul/dot products in fp32 where parity requires it.
- Softmax and router scores in fp32.
- Norm variance in fp32.
- Final logits as fp32 before logits processors/argmax.

Metal references:

- DS4: Q8_0/Q4_K dense and routed MoE patterns, top-k/router helper patterns, get_rows, RMSNorm/SwiGLU.
- gradients.c: RoPE, SDPA prefill/decode, KV-cache-oriented attention kernels.

## KV cache design

Allocate per batch slot and per layer:

```text
K: [max_prompt_tokens + 128, 10, 128] fp16
V: [max_prompt_tokens + 128, 10, 128] fp16
```

Operational behavior must match Python `SlidingWindowLlamaAttention`:

- Prefill writes all prompt tokens; prompt remains fully attendable.
- Decode warmup appends generated K/V until `prefill_len + 128` is reached.
- After warmup, generated slots are overwritten as a 128-token ring.
- Attention sees all prompt K/V plus the current generated ring contents.
- Position ids continue increasing monotonically even when cache slots are overwritten.

Memory cost is about:

```text
2 * K/V * 10 heads * 128 dim * 2 bytes * 12 layers = 61,440 bytes/token/sequence
```

A 273-token single-image prompt is ~16.8 MB of KV; large crop/multi-page prompts are much larger, so prompt-length limits must be explicit.

## Batching model

Implement progressively:

1. Single request.
2. Static batch with one preset and bounded prompt lengths.
3. Variable-length packed prefill:
   - prompt offsets
   - sequence-to-batch slot map
   - per-sequence image feature ranges
4. Decode active-sequence compaction.
5. MoE batching:
   - for small batch decode, selected-expert matvec is enough
   - for larger batches, group `(sequence, token, expert)` pairs by expert and run expert-major matmul
6. Continuous batching later.

## Parity and test strategy

Golden Python/prepared-request dump must include:

- rendered prompt string
- token ids and image mask/span lengths
- crop grid and preprocessed pixel tensors
- SAM output, CLIP output, projected visual features
- final prompt embeddings
- per-layer decoder hidden states after attention and after MLP/MoE
- router logits/probs/top-6 ids/weights
- logits before processors
- generated token ids/text

Test layers:

1. Python frontend parity against upstream HF wrapper:
   - token ids
   - image mask/span lengths
   - crop grid
   - preprocessed pixels
2. Prepared-request C validation tests.
3. Vision projected-feature parity.
4. Decoder fp16 parity text-only.
5. Decoder fp16 parity with dumped image embeddings.
6. Quantized decoder parity:
   - layer drift
   - router top-k agreement
   - logits top-k/KL
7. End-to-end OCR markdown/layout parity through Python API on small fixtures.

Large model tests should be opt-in; Python frontend and file-format tests should be fast and always-on.

## Milestone plan

1. Python frontend + golden prepared-request fixture format.
2. `.uocr` spec implementation, fp16 converter, strict loader/dumper.
3. C prepared-request API and validation tests.
4. Tensor registry and routed-expert packing.
5. CPU reference for decoder subsets and tensor-order validation.
6. Metal fp16 text-only decoder.
7. Metal decoder with Python-dumped image embeddings.
8. Q8_0 converter + Metal decoder kernels.
9. Q4_K aligned-tensor support; q8/padded-q4 policy for unaligned down projections.
10. Metal vision encoder fp16, then q8.
11. End-to-end single-image Python OCR API.
12. Static and variable-length batching for prepared requests.
13. Optional native tokenizer/image frontend for standalone CLI.
14. CUDA backend mirroring the stabilized backend interface.
