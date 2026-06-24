# Unlimited-OCR C implementation plan

This plan is grounded in the current design docs and reference code:

- `docs/foundations.md`
- `docs/architecture.md`
- `data/context/*` Hugging Face remote-code/config/tokenizer context
- `data/ds4/*` DwarfStar mmap/Metal/quantization reference
- `/Users/mascit/projects/gradients.c/src/ops/*` Metal RoPE/SDPA kernel references
- `/Users/mascit/Downloads/ocrbaidu/src/ocrbaidu/*` local Python wrapper/postprocess reference

The plan is intentionally model-specific. The converter absorbs Hugging Face naming/layout complexity; the runtime executes fixed Unlimited-OCR stages.

## 0. Refined execution order and acceptance gates

- [ ] **Gate A: frontend fixtures are exact.** Python prepared-request fixtures must match upstream for prompt text, token ids, image mask, crop grid, and preprocessed pixels before C inference work depends on them.
- [ ] **Gate B: fp16 `.uocr` is structurally exact.** The converter/loader must account for all `2710` safetensors tensors from the current header, either as runtime tensors or explicitly marked unused/provenance-only.
- [ ] **Gate C: integrated decoder works without vision.** The public Metal fp16 text-only generation path in `uocr_generate_prepared()` must match Python layer/logit/generated-id dumps before image paths are introduced.
- [ ] **Gate D: integrated decoder works with dumped image embeddings.** The same generation loop, fed with Python-dumped visual embeddings, must match Python prompt/layer/router/logit/generated-id dumps before implementing the Metal vision stack.
- [ ] **Gate E: vision works in fp16.** SAM -> CLIP -> projector -> newline/separator formatting must match Python visual-feature dumps for `1024` global and `640` local views.
- [ ] **Gate F: q8 works before q4.** `dyn-q8` converter/kernels must pass layer/logit/generation parity before enabling any q4 policy.
- [ ] **Gate G: q4 is calibrated, not global.** `dyn-q4` may only ship as a mixed profile with monotonic promotions and explicit reasons.
- [ ] **Gate H: batching follows single-request parity.** Static/variable batching starts only after single-request fp16 and q8 paths are stable.

First implementation slice, in order:

- [x] Land CMake skeleton and `libunlimitedocr` stub.
- [x] Land Python prepared-request builder and fixture writer.
- [x] Land synthetic `.uocr` loader/dumper tests without full weights.
- [x] Land C prepared-request validator and Python `ctypes` smoke test.
- [x] Land Metal backend init/source-compile/no-copy-mmap smoke test.
- [x] Land fp16 converter dry-run against cached safetensors header/index before requiring the full weight file.

Current priority slice, in order. The coding agent should take the first unchecked item here before starting vision, quantization, batching, CUDA, or high-level OCR API work:

- [x] Define internal integrated Metal decoder request/result structs and one orchestration entry point; keep existing per-op Metal functions as diagnostic/parity helpers.
- [x] Validate and cache all decoder-required mapped tensor bindings (`TOK_EMBED`, `FINAL_NORM`, `LM_HEAD`, layer norms, attention, dense MLP, MoE router/shared/expert weights) before running the integrated decoder.
- [x] Implement single-request, slot-0, text-only prompt prefill orchestration over all 12 layers using persistent arenas and KV cache; no image features and no batching yet.
- [x] Implement single-token decode-step orchestration using the existing KV ring helpers, prompt+generated-ring attention, final norm, LM head, no-repeat banning, greedy argmax, and EOS handling.
- [x] Wire the Metal backend path in `uocr_generate_prepared()` for `n_requests=1`, `views=0`, `fp16` model, and `max_new_tokens>0`; return generated ids with `uocr_result_create_from_generated()` instead of `UOCR_ERROR_NOT_IMPLEMENTED` for that narrow case.
- [x] Add an opt-in full-model public API parity test for text-only generated ids (`UOCR_RUN_LARGE_TESTS=1`, `UOCR_MODEL_PATH`, `UOCR_LAYER_DUMP_DIR` or equivalent fixture).
- [x] Promote the Python-dumped visual-embedding path from direct Metal tests into the same integrated decoder runner via an internal/test-only fixture adapter, without changing the stable v1 prepared-request image API.
- [x] Add opt-in generated-id/text parity for the integrated dumped-visual-embedding path.
- [x] Start section 16 Metal vision only after the integrated dumped-visual-embedding generation path passes.

## 1. Critical facts to encode as asserts

- [ ] Assert model constants from `data/context/config.json` and `configuration_deepseek_v2.py` defaults:
  - [ ] vocab `129280`, hidden `1280`, layers `12`, heads `10`, KV heads `10`, head dim `128`
  - [x] max positions `32768`, generated ring window `128`, RoPE theta `10000`
  - [ ] dense layer-0 intermediate `6848`
  - [ ] MoE layers `1..11`, routed experts `64`, top-k `6`, routed intermediate `896`, shared experts `2`
  - [ ] `hidden_act="silu"`, `rms_norm_eps=1e-6`, `attention_bias=false`, `attention_dropout=0.0`
  - [ ] MoE `scoring_func="softmax"`, `topk_method="greedy"`, `routed_scaling_factor=1.0`, `norm_topk_prob=false`
- [ ] Assert tokenizer facts:
  - [ ] BPE vocab `128000`, total vocab `129280`, added tokens `830`
  - [ ] BOS `0`, EOS `1`, PAD `2`, `<image>` `128815`
- [ ] Assert current safetensors facts:
  - [x] total weight payload `6,672,212,480` bytes from `model.safetensors.index.json`
  - [x] `2710` tensor entries in the safetensors header / index
  - [x] all source tensors are BF16 in the current checkpoint header
- [x] Assert prompt-template facts:
  - [x] `plain` conversation template has empty roles and empty separators
  - [x] upstream strips each message content and strips the final prompt
  - [x] assistant empty message contributes no text
  - [x] BOS is manually prepended after text/image expansion
- [ ] Assert generation/cache facts:
  - [ ] upstream `max_length` is total sequence length (`prompt + generated`), while the C API uses `max_new_tokens`
  - [ ] Python disables `config.sliding_window` for generation prefill and stores `_ring_window`
  - [ ] prefill tokens remain fully attendable
  - [ ] only generated tokens use the 128-token ring overwrite behavior
- [ ] Assert crop/view facts:
  - [ ] crop-mode source feature order is local rows first, then global rows, then view separator
  - [ ] placeholder id order is not semantically meaningful because all visual placeholders use the same id
  - [ ] C must validate visual feature count, not trust placeholder construction blindly

## 2. Non-negotiable constraints

- [ ] Keep the inference core narrow and Unlimited-OCR-specific; do not build a generic transformer runtime.
- [ ] Keep Python as the v1 frontend: image loading/preprocessing, prompt rendering, tokenization, and output decoding stay in Python.
- [ ] Keep C as the inference core: model loading, memory management, Metal/CUDA backends, KV cache, generation loop, logits processors, and batching.
- [ ] Start with Metal; design a narrow backend boundary so CUDA can mirror it later.
- [ ] Start with fp16 weights/activations for parity before any q8/q4 work.
- [ ] Keep MoE router weights `model.layers.*.mlp.gate.weight` fp16; do not confuse them with expert `gate_proj` weights.
- [ ] Keep norms, biases, RoPE/position data, image newline, and view separator tensors fp16 unless a later calibration proves otherwise.
- [ ] Use DS4/GGML quant layouts where they fit: `Q8_0` first, `Q4_K` only when inner dimension is multiple of `256` or explicitly padded.
- [ ] Do not allocate in the token loop; preallocate KV/cache/scratch and add debug allocation guards.
- [ ] Do not eagerly copy the whole `.uocr` model into RAM/GPU private buffers; keep model weights mmap-backed and wrap model ranges as Metal no-copy buffers.
- [ ] Preserve correctness before speed; no fast path is accepted with unexplained attention, KV, router, or logits drift.
- [ ] Do not implement the Metal vision encoder, q8/q4, batching, CUDA, or high-level OCR API until the fp16 single-request integrated decoder works first for text-only and then for dumped visual embeddings.

## 3. Repository and build bootstrap

- [ ] Create the planned source layout:
  - [x] `include/unlimitedocr.h`
  - [x] `src/core/`
  - [x] `src/model/`
  - [x] `src/convert/`
  - [x] `src/quant/`
  - [x] `src/runtime/`
  - [x] `src/backend/cpu_ref/`
  - [x] `src/backend/metal/`
  - [x] `src/backend/metal/kernels/`
  - [x] `src/backend/cuda/` as an empty later-backend placeholder
  - [x] `src/frontend_native/` as an optional later standalone frontend placeholder
  - [ ] `python/unlimitedocr_c/`
  - [x] `tools/`
  - [x] `tests/`
  - [x] `probes/`
- [x] Add a plain CMake build first; do not introduce `scikit-build-core` until the native ABI/resource layout stabilizes.
- [ ] Add CMake targets:
  - [x] `uocr_core` object/static library
  - [x] `uocr_metal` Objective-C/Metal backend on macOS
  - [ ] `uocr_cuda` optional later backend, default off
  - [x] `unlimitedocr` shared library exporting the C ABI
  - [x] `uocr-dump` model-file inspection tool
  - [x] native CTest tests
- [x] Add CMake options:
  - [x] `UOCR_ENABLE_METAL=AUTO`
  - [x] `UOCR_ENABLE_CUDA=OFF`
  - [x] `UOCR_ENABLE_CPU_REF=ON`
  - [x] `UOCR_BUILD_TOOLS=ON`
  - [x] `UOCR_BUILD_TESTS=ON`
  - [x] `UOCR_METAL_RUNTIME_COMPILE=ON`
  - [x] `UOCR_METAL_PRECOMPILE=OFF`
  - [x] `UOCR_SANITIZE=OFF`
- [x] Configure Objective-C compilation for `src/backend/metal/*.m` and link Apple frameworks: `Metal`, `Foundation`, and required system frameworks.
- [x] Use runtime Metal source compilation during development, DS4-style, so `.metal` edits do not require a full packaging flow.
- [ ] Add an optional CMake path to precompile `unlimitedocr.metallib` with `xcrun metal`/`metallib` for future release builds.
- [x] Keep `pyproject.toml` uv-friendly for now; add Python deps for frontend/tests without switching the build backend yet.
- [ ] Add Python extras:
  - [x] runtime/frontend: `numpy`, `pillow`, `tokenizers`, `safetensors`
  - [ ] reference/golden: `torch`, `transformers`, `einops`, `addict`, `easydict`
  - [ ] pdf/postprocess optional: `pymupdf`
  - [x] dev/test: `pytest`
- [x] Add a build-tree library loading convention for Python:
  - [x] `UOCR_LIBRARY_PATH=/path/to/libunlimitedocr.dylib`
  - [x] fallback search under `build/debug`, `build/release`, and package-installed `python/unlimitedocr_c/lib/` later
- [x] Add a tiny top-level `Makefile` only as a wrapper around `cmake`/`ctest`/`uv`, not as the real build system.

## 4. Python frontend and prepared-request fixtures

- [x] Implement `src/unlimitedocr_c/frontend.py` as the v1 public input path (current uv package layout; docs may move this to `python/` later).
- [x] Implement preset definitions matching upstream:
  - [x] `gundam`: `base_size=1024`, `image_size=640`, `crop_mode=True`, single-image only
  - [x] `base`: `base_size=1024`, `image_size=1024`, `crop_mode=False`
  - [x] multi-page/PDF: base mode only, `image_size=1024`
- [x] Implement tokenizer loading with HF `tokenizers` from `tokenizer.json`.
- [x] Validate tokenizer metadata at startup:
  - [x] BPE base vocab size `128000`
  - [x] total vocab size `129280`
  - [x] BOS id `0`
  - [x] EOS id `1`
  - [x] PAD id `2`
  - [x] `<image>` id `128815`
- [x] Implement prompt rendering compatible with upstream `format_messages(..., sft_format="plain", system_prompt="")`:
  - [x] strip each message content before appending, as upstream does
  - [x] strip the final prompt
  - [x] ignore role strings in `plain` mode
  - [x] emit no assistant prefix for the empty assistant message
- [x] Implement prompt token construction exactly:
  - [x] split rendered prompt on literal `<image>`
  - [x] tokenize text pieces with `add_special_tokens=false`
  - [x] insert repeated `128815` placeholders for the computed visual token count
  - [x] prepend BOS id `0`
  - [x] do not append EOS to the prompt
  - [x] emit same-length `image_mask`
- [x] Port upstream `dynamic_preprocess()` exactly for crop-mode grid selection:
  - [x] build target ratios for `min_num=2`, `max_num=32`
  - [x] choose closest aspect ratio with the same area tie-break
  - [x] resize source image to `(640 * W, 640 * H)`
  - [x] split row-major into `640x640` local crops
  - [x] if source image is at most `640x640`, use crop ratio `[1,1]`
- [x] Match upstream PIL preprocessing:
  - [x] open image
  - [x] EXIF transpose if needed
  - [x] RGB conversion
  - [x] `ImageOps.pad(..., color=(127,127,127))` because mean is `(0.5,0.5,0.5)`
  - [x] upstream default resize interpolation for `image.resize()` and `ImageOps.pad()`
  - [x] `ToTensor()` semantics: channel-first float in `[0,1]`
  - [x] normalize with mean/std `(0.5,0.5,0.5)` -> values in `[-1,1]`
  - [x] output contiguous NCHW `float16` for normal runtime calls
  - [x] allow `float32` prepared fixtures for preprocessing parity/debug; C/Metal may convert to fp16 on upload
- [x] Implement single-image base visual-token count:
  - [x] `num_queries = ceil((1024 / 16) / 4) = 16`
  - [x] placeholders `([id] * 16 + [id]) * 16 + [id] = 273`
  - [x] one global `1024x1024` view
- [x] Implement single-image gundam visual-token count:
  - [x] global `1024` rows/newlines: `16 * (16 + 1) = 272`
  - [x] final view separator: `1`
  - [x] local crops: `num_queries=10` for each `640` view
  - [x] if real local grid `W x H` exists, add `(10 * W + 1) * (10 * H)` local feature positions
  - [x] if crop grid is `[1,1]`, do not add local features; total remains `273`
  - [x] total for real crop grid is `(10 * W + 1) * (10 * H) + 272 + 1`
  - [x] remember upstream placeholder construction emits global placeholders first, but `masked_scatter_` fills them with source features ordered local-first/global-second/separator-last; C must reproduce the final feature order, not the placeholder-construction order
- [x] Implement multi-page visual-token construction:
  - [x] one `<image>` placeholder in prompt
  - [x] all page/global features concatenated at that span in page order
  - [x] each `1024` page contributes `273` visual positions
  - [x] no crop mode
- [x] Define the Python-to-C view order contract:
  - [x] base single image: one global `1024x1024` view
  - [x] multi-page: one global `1024x1024` view per page, in page order
  - [x] gundam `[1,1]`: one global `1024x1024` view, no local view
  - [x] gundam real crop: `W*H` local `640x640` views in row-major order, followed by one global `1024x1024` view
- [x] Define a Python `PreparedRequest` object containing:
  - [x] `input_ids: int32[n_tokens]`
  - [x] `image_mask: uint8[n_tokens]`
  - [x] `views: list[ImageView]` with `kind`, `width`, `height`, `format`, and contiguous pixel array
  - [x] `crop_grid_w`, `crop_grid_h`
  - [x] `mode/preset`
  - [x] `max_new_tokens`
  - [x] optional original upstream-style `max_length` in fixture metadata for parity debugging
  - [x] `no_repeat_ngram_size`
  - [x] `no_repeat_window`
- [ ] Add fixture serialization:
  - [x] `manifest.json` with prompt, preset, grid, image mask counts, expected visual count, dtype, shape, and source image metadata; token ids/mask are stored as `.npy` arrays
  - [x] one `.npy` per token/mask array plus a single `.npz` with named view arrays
  - [ ] optional `expected_output_ids.npy` and `expected_text.txt`
- [x] Add `tools/uocr-ref-dump` Python CLI to emit prepared-request fixtures without requiring C.
- [ ] Add pytest frontend parity tests against the upstream remote code path in `data/context/modeling_unlimitedocr.py`:
  - [x] local frontend smoke tests for rendered prompt, token ids, image mask, crop grid, pixels, fixture roundtrip
  - [x] rendered prompt equality
  - [ ] token ids equality
  - [ ] image mask equality
  - [ ] crop grid equality
  - [ ] pixel tensor equality within exact/near-exact tolerance
  - [ ] visual placeholder count equality
- [ ] Keep the native tokenizer/image frontend out of v1; record it under later tasks only.

## 5. Python golden tensor dumper

- [ ] Implement a reference dump mode that loads the official HF model from local/HF context and runs with eager attention.
- [ ] Use `uv run ...` for all Python scripts and tests.
- [ ] Add hooks or patched forward calls to dump:
  - [ ] rendered prompt string
  - [ ] input token ids
  - [ ] image mask
  - [ ] crop grid
  - [ ] preprocessed view tensors
  - [ ] SAM output after `sam_model(view)`
  - [ ] CLIP output after `vision_model(view, sam_features)`
  - [ ] projected visual features after `projector`
  - [ ] formatted visual features after newline/separator insertion
  - [ ] final prompt embeddings after Python `masked_scatter_`
  - [ ] per-layer hidden state after attention
  - [ ] per-layer hidden state after dense MLP/MoE
  - [ ] router logits, probabilities, top-6 expert ids, and top-6 weights for MoE layers
  - [ ] logits before logits processors
  - [ ] banned token ids from the no-repeat processor
  - [ ] generated token ids and decoded text
- [ ] Dump small, deterministic fixtures first:
  - [ ] text-only prompt fixture
  - [ ] single base `1024` image fixture
  - [ ] single gundam crop fixture with `1x1` crop
  - [ ] single gundam crop fixture with multi-crop grid
  - [ ] two-page base fixture
- [ ] Use fp32 dumps for comparison-critical tensors where feasible, even if model weights are BF16/fp16.
- [ ] Add an explicit fp16-reference mode for C parity:
  - [ ] either cast the HF model weights to fp16 before dumping, or
  - [ ] dump BF16 official tensors plus clear tolerances for the BF16->FP16 converter/runtime difference
  - [ ] record the reference dtype mode in every fixture manifest
- [ ] Store metadata with dtype, shape, max/mean statistics, and source module path for every dumped tensor.
- [ ] Add a fixture reader shared by Python tests and C parity tools.

## 6. `.uocr` model format

- [x] Define `src/model/uocr_format.h` with a versioned binary file format.
- [x] Use a mmap-friendly layout with a fixed header plus section directory.
- [x] Define header fields:
  - [x] magic, e.g. `UOCR`
  - [x] format version
  - [x] endian marker
  - [x] required alignment
  - [x] model id/source hash
  - [x] qprofile id: `fp16`, `dyn-q8`, `dyn-q4`, custom
  - [x] section count and offsets
- [x] Define a config section with fixed Unlimited-OCR constants:
  - [x] vocab `129280`
  - [x] hidden `1280`
  - [x] decoder layers `12`
  - [x] attention heads `10`
  - [x] KV heads `10`
  - [x] head dim `128`
  - [x] RoPE theta `10000`
  - [x] max positions `32768`
  - [x] generated ring window `128`
  - [x] dense-first layers `1`
  - [x] routed experts `64`
  - [x] top-k `6`
  - [x] MoE expert intermediate `896`
  - [x] shared experts `2`
  - [x] dense layer-0 intermediate `6848`
  - [x] projector `2048 -> 1280`
  - [x] vision patch size `16`
  - [x] downsample ratio `4`
  - [x] supported global view `1024`
  - [x] supported local view `640`
- [x] Define tensor directory entries:
  - [x] stable tensor id
  - [x] family/layer/expert/projection fields
  - [x] logical shape
  - [x] physical packed shape
  - [x] dtype/qtype
  - [x] payload offset/size
  - [x] block size and row size for quantized tensors
  - [x] scale/min offsets if stored separately
  - [x] qtype reason and promotion reason
  - [x] runtime usage flag: used by inference, unused-but-preserved, or omitted-with-provenance
- [x] Define qtype enum:
  - [x] `UOCR_TENSOR_F16`
  - [x] `UOCR_TENSOR_F32`
  - [x] `UOCR_TENSOR_Q8_0`
  - [x] `UOCR_TENSOR_Q4_K`
  - [x] `UOCR_TENSOR_PADDED_Q4_K`
  - [x] later placeholders for `Q2_K` and `IQ2_XXS`
- [x] Define tokenizer metadata section:
  - [x] tokenizer file hash
  - [x] BOS/EOS/PAD/image token ids
  - [x] optional embedded tokenizer payload for future native frontend
  - [x] explicit flag that C v1 does not require tokenizer tables
- [x] Define provenance section:
  - [x] HF model id/commit if known
  - [x] safetensors file hashes
  - [x] converter version
  - [x] config/tokenizer hashes
  - [x] qprofile and calibration corpus id
  - [x] complete source tensor accounting: every one of the current `2710` safetensors tensors is mapped to a runtime tensor, marked unused-but-preserved, or marked omitted-with-reason
- [x] Keep tensor payloads page-aligned enough for Metal no-copy view wrapping.
- [x] Keep tensor payload separate from metadata/tokenizer so Metal only wraps tensor pages.
- [x] Write a standalone `uocr-dump` tool to print:
  - [x] header/config
  - [x] tensor count
  - [x] qtype histogram
  - [x] tensor offsets/sizes/alignment
  - [x] largest tensors
  - [x] expected memory footprint
  - [x] provenance
  - [x] source tensor accounting summary: runtime / preserved-unused / omitted

## 7. Tensor registry and shape validation

- [x] Implement enum-like tensor ids in `src/model/`, not string lookups in hot runtime paths.
- [x] Include tensor families:
  - [x] `TOK_EMBED`
  - [x] `LM_HEAD`
  - [x] `FINAL_NORM`
  - [x] `LAYER[i].ATTN.{Q,K,V,O}`
  - [x] `LAYER[i].NORM.{INPUT,POST_ATTN}`
  - [x] `LAYER[0].DENSE_MLP.{GATE,UP,DOWN}`
  - [x] `LAYER[1..11].MOE.ROUTER`
  - [x] `LAYER[1..11].MOE.EXPERTS.{GATE,UP,DOWN}`
  - [x] `LAYER[1..11].MOE.SHARED.{GATE,UP,DOWN}`
  - [x] `VISION.SAM.*`
  - [x] `VISION.CLIP.*`
  - [x] `PROJECTOR`
  - [x] `IMAGE_NEWLINE`
  - [x] `VIEW_SEPARATOR`
- [x] Generate or hardcode expected tensor shapes from `data/context/model.safetensors.header`.
- [x] Validate key known shapes:
  - [x] `lm_head.weight`: `[129280,1280]`
  - [x] `model.embed_tokens.weight`: `[129280,1280]`
  - [x] decoder attention projection weights: `[1280,1280]`
  - [x] layer-0 dense `down_proj`: `[1280,6848]`
  - [x] MoE router `mlp.gate.weight`: `[64,1280]`
  - [x] routed expert `down_proj`: `[1280,896]`
  - [x] routed expert `gate_proj/up_proj`: `[896,1280]`
  - [x] shared expert intermediate: `1792`
  - [x] projector weight: `[1280,2048]`
  - [x] projector bias: `[1280]`
  - [x] image newline/view separator: `[1280]`
  - [x] SAM patch embed: `[768,3,16,16]`
  - [x] CLIP qkv: `[3072,1024]`
- [x] Add a converter validation report that fails if expected tensor names are missing or unexpected shapes appear.
- [x] Identify tensors that are present but not used in the normal upstream OCR path, especially CLIP pixel patch embedding (`vision_model.embeddings.patch_embedding.*`) because CLIP receives SAM feature maps as `patch_embeds`; either preserve them as unused or omit them with explicit provenance.
- [x] Add a deterministic tensor-order table used by both converter and runtime.
- [ ] Pack routed experts expert-major:
  - [ ] `[layer][projection][expert][out_row][packed_input]`
- [ ] Pack or colocate expert `gate_proj` and `up_proj` so Metal selected-expert kernels can read them together.
- [ ] Store both logical and physical input widths for quantized tensors.
- [ ] Explicitly mark unaligned `Q4_K` hazards:
  - [ ] routed expert `down_proj` input `896`
  - [ ] dense layer-0 `down_proj` input `6848`

## 8. fp16 converter

- [x] Implement `tools/uocr-convert` dry-run CLI (Python first; native conversion can follow once layout is stable).
- [ ] Read `config.json`, `processor_config.json`, tokenizer metadata, and safetensors metadata.
- [x] Support the current one-file safetensors layout `model-00001-of-000001.safetensors`; keep multi-file safetensors support possible through the index file.
- [ ] Parse safetensors without loading all weights at once:
  - [x] read header length
  - [x] parse JSON header
  - [x] validate that current source tensors are BF16 unless explicitly exempted by a future checkpoint
  - [x] map or stream individual tensor byte ranges
  - [x] convert BF16 rows to fp16 payload rows
  - [x] never allocate a second full-model-sized temporary buffer
- [x] Implement BF16 -> fp16 conversion using bounded NumPy chunk conversion for the Python converter writer.
- [ ] Record conversion statistics per tensor: source dtype, output dtype/qtype, min/max if cheap, NaN/Inf count, and exact byte count.
- [ ] Preserve row-major matrix layout where runtime kernels expect `[out, in]`.
- [ ] Transpose only where an explicit runtime kernel requires it; record the decision in tensor metadata.
- [x] Emit pure fp16 `.uocr` first when the safetensors payload is present; always-on tests use tiny synthetic safetensors payloads.
- [x] Add converter flags:
  - [x] `--hf-dir PATH`
  - [x] `--out PATH`
  - [x] `--qprofile fp16|dyn-q8|dyn-q4`
  - [x] `--dry-run`
  - [x] `--tensor NAME_OR_ID` for single-tensor debugging
  - [x] `--dump-plan`
  - [x] `--overwrite`
- [ ] Add dry-run output similar to DS4 quantizer:
  - [x] tensor name -> tensor id mapping
  - [x] source shape/dtype
  - [x] output shape/qtype
  - [x] output bytes
  - [x] qtype/promotion reason
  - [x] source accounting status: runtime / preserved-unused / omitted
  - [x] planned `.uocr` section layout and tensor payload offsets/alignment
- [ ] Add single-tensor compare mode for converter development.
- [x] Add strict loader tests using only a tiny synthetic `.uocr` file, so CI does not require full weights.
- [x] Add writer tests using tiny synthetic safetensors payloads so `.uocr` serialization and BF16->fp16 streaming are covered without full weights.
- [ ] Add an opt-in full-model converter smoke test for local machines with the full safetensors file.
- [x] Keep converter/loader unit tests runnable without full weights by using synthetic `.uocr` files, synthetic safetensors payloads, and the cached safetensors header/index.

## 9. C public API and Python FFI

- [x] Write `include/unlimitedocr.h` with opaque types:
  - [x] `uocr_engine`
  - [x] `uocr_result`
- [x] Define C enums:
  - [x] `uocr_pixel_format`: `UOCR_PIXEL_F16_NCHW`, `UOCR_PIXEL_F32_NCHW`
  - [x] `uocr_view_kind`: `UOCR_VIEW_GLOBAL`, `UOCR_VIEW_LOCAL`
  - [x] backend ids or backend string handling: `metal`, `cpu-ref`, later `cuda`
- [ ] Define `uocr_image_view`:
  - [x] pointer to contiguous `[3,H,W]` normalized pixels
  - [x] width/height
  - [x] format
  - [x] kind
  - [ ] no strides in v1; Python must pass contiguous arrays and C validates expected byte size from format/shape
- [x] Define `uocr_prepared_request`:
  - [x] `input_ids`
  - [x] `image_mask`
  - [x] `n_tokens`
  - [x] `views`
  - [x] `n_views`
  - [x] `crop_grid_w`, `crop_grid_h`
  - [x] `max_new_tokens`
  - [x] `no_repeat_ngram_size`
  - [x] `no_repeat_window`
- [x] Define `uocr_engine_opts`:
  - [x] `model_path`
  - [x] `backend`
  - [x] `resource_path` for Metal source/metallib lookup
  - [x] `max_batch`
  - [x] `max_prompt_tokens`
  - [x] `max_gen_tokens`
  - [x] optional memory budget override
- [x] Export:
  - [x] `uocr_engine_open`
  - [x] `uocr_engine_close`
  - [x] `uocr_generate_prepared`
  - [x] `uocr_result_tokens`
  - [x] `uocr_result_free`
  - [x] `uocr_last_error` or per-engine error retrieval
- [x] Keep C ABI stable and plain; no `pybind11`.
- [x] Implement `src/unlimitedocr_c/ffi.py` with `ctypes` first.
- [x] Ensure Python holds NumPy arrays alive for the duration of the C call.
- [x] Add Python tests that call C validation with synthetic prepared requests before real inference exists.

## 10. Prepared-request validation and runtime state

- [x] Validate `input_ids` pointer, `image_mask` pointer, and `n_tokens`.
- [x] Validate token ids are in `[0,129280)`.
- [x] Validate BOS id `0` is present at position 0 for normal image/text generation.
- [x] Count `image_mask == 1` positions.
- [x] Compute expected visual token count from views and crop grid:
  - [x] base/global `1024`: `273`
  - [x] local grid `W x H` with `640` crops, only when `W*H > 1`: `(10*W + 1) * (10*H)`
  - [x] crop mode with real locals: local tokens + `272` global row/newline tokens + `1` separator
  - [x] no-crop/multi-page: each global page contributes `272` global row/newline tokens + `1` separator
- [x] Validate placeholder count equals expected visual feature count.
- [x] Validate supported view dimensions:
  - [x] `1024x1024` global
  - [x] `640x640` local
- [ ] Validate view order rules:
  - [ ] base single image: exactly one global `1024x1024` view
  - [x] multi-page: one or more global `1024x1024` views, all in page order
  - [x] gundam `[1,1]`: exactly one global `1024x1024` view
  - [x] gundam real crop: exactly `W*H` local `640x640` views in row-major order followed by one global `1024x1024` view
  - [x] reject mixed local/global orders that do not match this contract
- [x] Validate prompt length plus `max_new_tokens` fits engine limits.
- [ ] Validate KV budget before running vision/prefill.
- [x] Validate no-repeat config:
  - [x] allow zero disabled values
  - [x] support upstream defaults `35/128` and `35/1024`
  - [x] implement CPU no-repeat processor over the current full token sequence (`prompt + generated`) with the same sliding-window scan as upstream `LogitsProcessor`
- [ ] Build per-sequence state:
  - [x] prompt token count
  - [x] text ranges around the single visual span
  - [x] image feature span range
  - [x] max generation length
  - [ ] no-repeat state over prompt plus generated ids
  - [ ] generated token buffer
  - [x] position counter
  - [x] EOS status

## 11. Model loader and memory management

- [x] Implement `.uocr` mmap loader in `src/core/`/`src/model/`.
- [x] Parse header, section directory, config, tensor directory, tokenizer metadata, and provenance.
- [x] Validate `.uocr` config against compiled Unlimited-OCR constants.
- [x] Validate tensor payload ranges do not exceed file size.
- [x] Keep tensor data in mmap; do not copy tensors during load.
- [x] Build runtime tensor table mapping tensor id -> mmap offset/size/qtype/shape.
- [x] Implement DS4-style allocation wrappers:
  - [x] `uocr_malloc`
  - [x] `uocr_calloc`
  - [x] `uocr_realloc`
  - [x] `uocr_malloc_zeroed` using `malloc + memset`, not lazy `calloc`, for large hot-state buffers
  - [x] internal live/peak/total/failure allocation counters
  - [x] no-allocation guard primitive for future hot-path checks
- [ ] Add allocation guard around prefill/decode hot paths.
- [x] Implement runtime memory categories and live/peak counters:
  - [x] model views
  - [x] KV cache
  - [x] prompt embeddings
  - [x] vision scratch
  - [x] decoder scratch
  - [x] MoE scratch
  - [x] logits/readback
  - [x] transient buffers
- [x] Add `uocr_engine_memory_report()` or log output at open and after first run.
- [ ] Implement admission control:
  - [x] minimal request memory-budget enforcement before generation/ABI smoke path
  - [x] resident weights estimate from `.uocr` tensor-data section and model-open budget rejection
  - [x] KV cache estimate
  - [x] prompt embedding estimate
  - [x] vision scratch estimate for one-view chunk scheduling
  - [x] decoder/MoE scratch estimate scaffold
  - [x] logits/readback estimate
  - [x] safety margin
  - [x] Metal `recommendedMaxWorkingSetSize` fraction if available
- [x] Implement KV size formula in code comments/tests:
  - [x] `2 * 12 * 10 * 128 * 2 = 61,440 bytes/token/sequence`
- [x] Allocate long-lived runtime arenas at engine/session creation:
  - [x] KV cache per batch slot
  - [x] prompt embedding buffer sized by `max_prompt_tokens`
  - [x] hidden-state ping-pong buffers
  - [x] router/top-k buffers
  - [x] MoE intermediate buffers
  - [x] vision scratch buffers
  - [x] logits and next-token buffers

## 12. Metal backend skeleton and DS4 memory policy

- [x] Implement `src/backend/metal/uocr_metal.m` with Objective-C kept at the backend boundary.
- [x] Initialize Metal device and command queue.
- [x] Compile `.metal` source files at runtime in development mode.
- [x] Add pipeline cache keyed by function name; function-constant variants will extend the same key scheme when needed.
- [x] Add backend resource lookup from `uocr_engine_opts.resource_path`.
- [x] Implement mmap-backed model view planning, following DS4:
  - [x] page-align view ranges to host VM page size
  - [x] account for device `maxBufferLength`
  - [x] create coalesced views that cover all tensor payloads
  - [x] ensure every tensor lives wholly within one view
  - [x] wrap views with `newBufferWithBytesNoCopy`
  - [x] cache per-tensor `{id<MTLBuffer>, inner_offset}` after load
- [ ] Add optional exact-range wrapper cache only if needed; default kernels should use precomputed model views.
- [x] Add optional Metal model-view warmup hook; residency-set integration remains a future macOS 15+ hint if needed.
- [x] Add DS4-style transient buffer tracking:
  - [x] retain CPU-filled shared buffers until command buffer completion
  - [x] release transients in completion handlers
- [x] Add named scratch buffers that grow to high-water mark and are reused.
- [x] Prefer `MTLResourceStorageModePrivate` for GPU-only scratch where possible.
- [x] Use shared storage only for CPU-filled/readback buffers.
- [x] Add command-buffer ownership rules and a simple synchronous path first.
- [x] Add Metal smoke tests:
  - [x] allocate/free scratch
  - [x] compile kernels
  - [x] wrap a synthetic mmap range as no-copy buffer
  - [x] run a tiny copy/fill kernel

## 13. CPU reference and diagnostic kernels

- [ ] Implement CPU reference code only for correctness diagnostics; do not aim for full fast CPU inference.
- [ ] Implement scalar conversions:
  - [ ] BF16 -> F32
  - [ ] F32 -> F16
  - [ ] F16 -> F32
- [ ] Implement fp32/fp16 tensor helpers for small fixtures.
- [ ] Implement CPU RMSNorm with fp32 variance and eps `1e-6`.
- [ ] Implement Llama/DeepSeek RoPE using split-half layout, not interleaved layout.
- [ ] Implement CPU causal SDPA for tiny text-only fixtures.
- [ ] Implement CPU KV cache append/ring behavior for tiny fixtures.
- [ ] Implement CPU dense SwiGLU for layer 0.
- [ ] Implement CPU MoE router:
  - [ ] router matmul in fp32
  - [ ] softmax in fp32
  - [ ] top-6 greedy
  - [ ] no top-k renormalization
  - [ ] shared expert added after routed sum
- [x] Implement no-repeat-ngram processor matching upstream `SlidingWindowNoRepeatNgramProcessor`.
- [ ] Copy/adapt DS4 CPU quant dot helpers for `Q8_0` and `Q4_K` once quantization starts.
- [ ] Add CPU reference tests against Python dumped tiny tensors.

## 14. Metal fp16 decoder bring-up

### 14.1 Text embedding and prompt assembly

- [x] Adapt DS4 `metal/get_rows.metal` for token embedding gather.
- [x] Support fp16 embedding weights and fp16/fp32 output activations.
- [x] Implement direct prompt embedding assembly:
  - [x] write text token embeddings from token ids
  - [x] write image features into image spans when provided
  - [x] avoid emulating Python `masked_scatter_`
- [x] Add a text-only path where no image features are present.
- [x] Bind mmap-backed token embeddings for production prompt assembly.
- [x] Write mapped prompt embeddings into the persistent Metal prompt arena.
- [x] Compare prompt embeddings against Python dumps.

### 14.2 Decoder layer primitives

- [x] Adapt DS4 `metal/norm.metal` or write OCR-specific RMSNorm for hidden `1280`.
- [x] Ensure RMSNorm variance accumulation is fp32.
- [x] Implement fp16 dense matrix-vector and matrix-matrix kernels for `[out,in]` row-major weights.
- [x] Implement attention Q/K/V/O projections for hidden `1280`.
- [x] Adapt gradients.c `qkv_split_rope` logic:
  - [x] split-half RoPE layout
  - [x] head dim `128`
  - [x] heads `10`
  - [x] position ids monotonically increasing
  - [x] theta `10000`
- [x] Implement KV cache write kernel.
- [x] Allocate KV cache layout:
  - [x] K: `[layer][slot][max_prompt_tokens + 128][10][128]` fp16
  - [x] V: `[layer][slot][max_prompt_tokens + 128][10][128]` fp16
  - [x] per sequence, ring slots start at actual `prefill_len`, not at `max_prompt_tokens`
  - [x] attention length uses actual prompt tokens plus live generated-ring tokens; unused prompt capacity is never attended
- [x] Implement prompt prefill attention over full prompt with causal mask.
- [x] Adapt gradients.c `sdpa_varlen` for prefill where useful.
- [x] Implement decode attention for one token attending to full prompt plus generated ring.
- [x] Adapt gradients.c `sdpa_decode` window/prefix logic to OCR's rule: prompt always attendable, generated ring size `128`.
- [x] Implement attention output projection and residual add.

### 14.3 MLP and MoE

- [x] Implement layer-0 dense SwiGLU:
  - [x] `gate_proj`
  - [x] `up_proj`
  - [x] SiLU activation from DeepSeekV2Config default `hidden_act="silu"`; assert this at model load
  - [x] elementwise multiply
  - [x] `down_proj`
- [x] Implement MoE router for layers `1..11`:
  - [x] fp16 router weight storage, fp32 matmul accumulation
  - [x] softmax over `64` experts in fp32
  - [x] top-6 selection; order can differ because weighted sum is commutative
  - [x] multiply selected probabilities by `routed_scaling_factor=1.0`
  - [x] no probability renormalization (`norm_topk_prob=false`)
  - [x] no DS4 router-specific sqrt/softplus/bias behavior
- [x] Adapt DS4 `argsort.metal` / `dsv4_misc.metal` top-k patterns for OCR's simpler top-6.
- [x] Implement selected-expert fp16 path for decode.
- [x] Implement token-batched expert-major routed-expert path for prefill; true group-by-expert optimization can follow after parity.
- [x] Implement shared experts for every MoE layer with intermediate `1792`.
- [x] Add routed expert result + shared expert result.

### 14.4 LM head and generation loop

- [x] Implement final RMSNorm.
- [x] Implement fp16 LM head matvec to vocab `129280` with fp32 logits.
- [x] Implement greedy argmax.
- [x] Implement no-repeat-ngram banning before argmax:
  - [x] CPU version first using logits readback if needed
  - [x] later GPU version for speed
- [x] Stop on EOS id `1` or max new tokens.
- [x] Return generated token ids from the diagnostic/parity selection helper.
- [x] Wire final RMSNorm, LM head, no-repeat bans, greedy argmax, and EOS handoff in a Metal next-token selection helper.
- [x] Add text-only fp16 parity test:
  - [x] prompt embeddings
  - [x] layer-0 dense decoder output
  - [x] layer-1 MoE decoder output
  - [x] remaining layers 2..11 MoE outputs
  - [x] logits top-k
  - [x] generated token ids

### 14.5 Integrated public text decoder path

These tasks turn the fp16 decoder primitives/parity helpers into the first real `uocr_generate_prepared()` generation path. Keep the scope narrow: Metal only, fp16 only, single request, text-only, no vision, no batching.

- [x] Define an internal decoder orchestration boundary, e.g. `uocr_metal_context_generate_text_f16()` or equivalent, that owns prefill/decode scheduling instead of exposing another per-op diagnostic helper.
- [x] Build and validate a decoder tensor-binding table from stable tensor ids once per mapped model; fail early if any text decoder tensor is missing, not fp16, or has an unexpected shape.
- [x] Assemble text-only prompt embeddings into the persistent prompt arena for slot `0`.
- [x] Run full-prompt prefill through all 12 decoder layers, writing prompt K/V for every layer and carrying hidden-state ping-pong buffers in Metal arenas.
- [x] Run decode steps from the last accepted token, writing generated K/V into the 128-token ring and attending to all prompt tokens plus live generated ring tokens.
- [x] Maintain a preallocated generated-token buffer and prompt+generated no-repeat state; no allocation is allowed inside the per-token loop after initial bring-up.
- [x] Stop on EOS id `1` or `max_new_tokens` and return generated ids through `uocr_result_create_from_generated()`.
- [x] Wire only this narrow path into `uocr_generate_prepared()`; keep CPU-ref and unsupported Metal cases returning clear `UOCR_ERROR_NOT_IMPLEMENTED` messages.
- [x] Add opt-in full-model text-only generated-id parity against Python fixtures.

## 15. Decoder with Python-dumped image embeddings

- [x] Add a C/Metal test mode accepting precomputed projected visual embeddings from Python fixtures.
- [x] Bypass C vision encoder in this mode.
- [x] Assemble prompt embeddings with direct spans:
  - [x] text embeddings before image
  - [x] dumped visual embeddings
  - [x] text embeddings after image
- [x] Validate image placeholder count against dumped visual feature length.
- [x] Run full decoder prefill/decode with dumped image embeddings.
- [ ] Compare against Python:
  - [x] final prompt embeddings
  - [x] per-layer hidden states
  - [x] router top-6 agreement
  - [x] logits
  - [x] generated ids/text
- [ ] Keep this mode available permanently as a parity/debug path even after Metal vision exists.

### 15.1 Integrated dumped-embedding generation path

Do this immediately after section 14.5 and before porting SAM/CLIP. The goal is to prove the real decoder/generation loop works for image prompts while vision is still bypassed.

- [x] Add an internal/test-only request adapter that supplies preformatted fp16 visual feature rows `[image_tokens,1280]` alongside a normal prepared image request.
- [x] Reuse the integrated text decoder runner after prompt assembly; do not maintain a separate image-specific decoder loop.
- [x] Assemble text tokens plus dumped visual rows into the same prompt arena layout that future Metal vision will produce.
- [x] Validate image span length equals dumped visual feature rows and that the prepared request's view/crop metadata still passes normal validation.
- [x] Run prefill/decode/generation through the integrated decoder for dumped image embeddings.
- [x] Compare generated ids/text against Python dumped-image fixtures; logits/top-k and router checks remain mandatory diagnostics when generated ids drift.
- [ ] Keep this path as a permanent opt-in parity mode after Metal vision lands.

## 16. Metal fp16 vision encoder

Do not begin this section until sections 14.5 and 15.1 pass for fp16 single-request generation. Vision should replace dumped visual embeddings, not introduce a second decoder/generation path.

### 16.1 Vision scheduling and memory

- [x] Implement view-chunk scheduling so crop mode does not force all local views through vision at once.
- [x] Process local crops in chunks, append projected features into final visual feature buffer, and reuse SAM/CLIP scratch.
- [x] Support one global `1024x1024` view and zero or more local `640x640` views.
- [x] Add vision memory estimates to admission control.

### 16.2 SAM-like encoder

- [x] Implement SAM patch embed conv:
  - [x] weight shape `[768,3,16,16]`
  - [x] stride `16`
  - [x] output grid `64x64` for `1024`, `40x40` for `640`
  - [x] layout conversion to BHWC as upstream does
- [x] Add SAM absolute position embedding:
  - [x] use `get_abs_pos_sam` bicubic interpolation semantics
  - [x] precompute/store common `64x64` and `40x40` tables in converter if practical (not practical for the current v1 `.uocr` writer; runtime Metal interpolation is used until derived tensors are added)
- [x] Implement SAM transformer blocks `0..11`:
  - [x] LayerNorm eps `1e-6`
  - [x] QKV linear with bias
  - [x] window attention size `14` for non-global blocks
  - [x] global attention for blocks `[2,5,8,11]`
  - [x] decomposed relative position bias from `rel_pos_h` and `rel_pos_w`
  - [x] GELU MLP with `mlp_ratio=4`
  - [x] residual connections
- [x] Implement window partition/unpartition with padding exactly like upstream.
- [x] Implement SAM neck/net:
  - [x] `1x1` conv `768 -> 256`, no bias
  - [x] `LayerNorm2d(256, eps=1e-6)`
  - [x] `3x3` conv `256 -> 256`, padding `1`, no bias
  - [x] `LayerNorm2d(256, eps=1e-6)`
  - [x] `net_2`: `3x3` stride-2 `256 -> 512`, padding `1`
  - [x] `net_3`: `3x3` stride-2 `512 -> 1024`, padding `1`
- [ ] Validate SAM output against Python dumps for `1024` and `640` views.

### 16.3 CLIP-like encoder

- [x] Implement CLIP embeddings using SAM output as `patch_embeds`; do not use raw-pixel patch embedding in the normal path.
- [x] Treat `vision_model.embeddings.patch_embedding.*` as unused in normal OCR inference unless a later parity test proves an alternate path needs it.
- [x] Flatten SAM features to token sequence and prepend class embedding.
- [x] Add CLIP absolute position embedding using upstream bicubic interpolation with antialias and `align_corners=False`.
- [x] Support token lengths:
  - [x] `257` for `16x16 + CLS`
  - [x] `101` for `10x10 + CLS`
- [x] Implement pre-LayerNorm eps `1e-5`.
- [x] Implement 24 CLIP transformer blocks:
  - [x] LayerNorm eps `1e-5`
  - [x] QKV projection `[3072,1024]` with bias
  - [x] full attention, 16 heads, head dim `64`
  - [x] output projection
  - [x] QuickGELU: `x * sigmoid(1.702*x)`
  - [x] MLP `1024 -> 4096 -> 1024`
  - [x] residual connections
- [ ] Validate CLIP output against Python dumps.

### 16.4 Projector and visual feature formatting

- [x] Concatenate CLIP token features without CLS and SAM feature-map tokens: `1024 + 1024 = 2048`.
- [x] Implement linear projector `2048 -> 1280` with bias.
- [ ] Validate projected features against Python dumps.
- [x] Format global view features:
  - [x] reshape to grid `16x16` for `1024`
  - [x] append `image_newline` after each row -> `272` tokens
  - [x] append `view_seperator` -> `273` tokens
- [x] Format local crop features:
  - [x] each local `640` crop produces `10x10`
  - [x] reshape local crops as upstream: `view(height_crop_num, width_crop_num, h2, w2, dim).permute(0,2,1,3,4)`
  - [x] append newline after each stitched local row
- [x] Crop-mode final feature order must be local features first, then global row/newline features, then separator.
- [x] Non-crop/multi-page final feature order must be per-image global row/newline features plus separator.
- [ ] Validate final visual feature buffer length and values against Python dumps.

## 17. Dynamic q8/q4 converter and kernels

### 17.1 Quantization foundations

- [x] Vendor/adapt DS4 `data/ds4/gguf-tools/quants.[ch]` into `src/quant/` with attribution/license notes.
- [x] Keep only needed initial formats:
  - [x] `Q8_0`, block size `32`, type size `34`
  - [x] `Q4_K`, block size `256`, type size `144`
- [x] Keep `Q2_K` and `IQ2_XXS` code disabled or behind later experimental flags.
- [x] Implement qtype row-size helpers and alignment validation.
- [x] Add quantized tensor metadata to `.uocr` with logical and physical widths.

### 17.2 `dyn-q8` profile

- [x] Implement qtype policy for `dyn-q8`:
  - [x] large decoder linears -> `Q8_0`
  - [x] LM head -> `Q8_0`
  - [x] token embedding -> `Q8_0` or fp16 if embedding parity drifts; q8 embedding gather needs a dequantizing get-rows kernel
  - [x] vision conv/linear weights -> `Q8_0`
  - [x] MoE router weights -> fp16
  - [x] norms/biases/position/newline/separator -> fp16
- [x] Add converter dry-run qtype histogram and memory estimate.
- [x] Emit q8 `.uocr`.
- [x] Add CPU dequant/dot tests for q8 tensors.
- [x] Add Metal tests for Q8_0 get-rows/dequant because token embeddings and possibly LM-head-adjacent paths need it.

### 17.3 `dyn-q4` profile

- [ ] Implement conservative q4 policy:
  - [ ] attention projections `1280` inner dim -> `Q4_K` candidates
  - [ ] routed expert `gate_proj/up_proj` inner dim `1280` -> `Q4_K` candidates
  - [ ] routed expert `down_proj` inner dim `896` -> keep `Q8_0` initially
  - [ ] dense layer-0 `down_proj` inner dim `6848` -> keep `Q8_0` initially
  - [ ] shared expert down inner dim `1792` -> `Q4_K` candidate after q8 parity
  - [ ] LM head -> keep `Q8_0` initially
  - [ ] vision -> keep `Q8_0` initially, selective q4 later
- [ ] Add `PADDED_Q4_K` design but do not enable by default:
  - [ ] physical width rounded to multiple of `256`
  - [ ] activation zero-fill/padding in kernels
  - [ ] logical width retained for output correctness
- [ ] Add promotion metadata with reasons: sensitive, unaligned, calibration drift, manual override.

### 17.4 Metal quantized kernels

- [x] Adapt DS4 `metal/dense.metal` Q8_0 matvec for OCR shapes.
- [x] Add Q8_0 linear kernels for decode and prefill.
- [x] Adapt DS4 Q8_0 shared gate/up SwiGLU pattern for OCR shared experts if helpful.
- [x] Adapt DS4 `metal/moe.metal` Q4_K selected-expert kernels for OCR routed experts.
- [x] Ensure OCR MoE routing math stays OCR-specific; do not import DS4 router softplus/sqrt/scaling behavior.
- [x] Implement q8 fallback path for unaligned down projections.
- [ ] Add optional padded q4 kernels only after q8/q4 baseline is correct.
- [ ] Compare q8/q4 layer drift against fp16 golden dumps.

## 18. Calibration and dynamic promotion loop

- [ ] Build a representative OCR calibration corpus:
  - [ ] small scanned document
  - [ ] dense text page
  - [ ] table/form page
  - [ ] multilingual/CJK page
  - [ ] flyer/graphic-heavy image
  - [ ] multi-page PDF subset
  - [ ] crop-mode high aspect-ratio image
- [ ] Add `tools/uocr-calibrate` to run fp16 and candidate quantized models over fixtures.
- [ ] Add optional DS4-style activation-importance collection for routed experts:
  - [ ] for routed `gate_proj`/`up_proj`, accumulate squared FFN-normalized input columns per expert
  - [ ] for routed `down_proj`, accumulate squared routed SwiGLU/down-input columns after route weighting
  - [ ] pack per-expert vectors into calibration metadata so the quantizer can slice `[expert][column]`
  - [ ] use weight-energy fallback only when no activation corpus is available
- [ ] Record metrics:
  - [ ] tensor/module RMSE
  - [ ] tensor/module cosine similarity
  - [ ] per-layer hidden-state drift
  - [ ] router top-6 agreement
  - [ ] selected expert frequency
  - [ ] logits KL/top-k agreement
  - [ ] generated token id agreement or longest common prefix
  - [ ] decoded markdown/layout agreement
- [ ] Add per-expert/per-projection error and traffic stats.
- [ ] Implement monotonic promotion:
  - [ ] `q4 -> q8`
  - [ ] `q8 -> fp16`
  - [ ] never demote automatically in the same calibration pass
- [ ] Persist calibration results/provenance into `.uocr`.
- [ ] Add manual override file support for qtype decisions.

## 19. Batching and scheduler

- [ ] Implement single-request path first.
- [ ] Implement static batch with same preset and bounded prompt lengths.
- [ ] Allocate per-slot KV caches up front.
- [ ] Add packed prefill data structures:
  - [ ] sequence offsets
  - [ ] per-sequence prompt lengths
  - [ ] per-sequence image feature ranges
  - [ ] slot map
- [ ] Extend SDPA prefill to variable-length packed prompts.
- [ ] Implement decode active-sequence compaction:
  - [ ] skip completed EOS sequences
  - [ ] keep per-sequence position counters
  - [ ] keep per-sequence ring positions
- [ ] Add MoE batching progression:
  - [ ] selected-expert matvec per token first
  - [ ] group `(sequence, token, expert)` pairs by expert for larger batches later
  - [ ] expert-major batched matmul after correctness
- [ ] Add admission control for batches before running vision.
- [ ] Add continuous batching only after static/variable batching is stable.

## 20. Python end-to-end API

Do not start the high-level OCR API until `uocr_generate_prepared()` can return generated ids for the fp16 single-request Metal path. A thin `ctypes` `Engine` already exists in `src/unlimitedocr_c/ffi.py`; this section is about stable user-facing ergonomics after the native pipeline works.

- [ ] Decide whether to keep the existing `src/unlimitedocr_c/ffi.py::Engine` as the public engine or move/wrap it in `python/unlimitedocr_c/engine.py` during packaging cleanup.
- [ ] Implement `generate_prepared()` returning generated token ids.
- [ ] Implement high-level `ocr.generate(image, prompt, preset=...)`:
  - [ ] build prepared request
  - [ ] call C engine
  - [ ] decode returned token ids with HF tokenizer
  - [ ] strip trailing EOS text if present
- [ ] Implement `ocr_image()` matching local wrapper defaults:
  - [ ] prompt `"<image>document parsing."`
  - [ ] gundam preset default
  - [ ] upstream-style `max_length=32768` converted to C `max_new_tokens = max_length - prompt_tokens`, capped by engine limit
  - [ ] `no_repeat_ngram_size=35`
  - [ ] `ngram_window=128`
- [ ] Implement `ocr_pages()` matching upstream multi-page defaults:
  - [ ] prompt `"<image>Multi page parsing."`
  - [ ] base mode
  - [ ] upstream-style `max_length=32768` converted to C `max_new_tokens = max_length - prompt_tokens`, capped by engine limit
  - [ ] `ngram_window=1024`
- [ ] Implement `ocr_pdf()` using PyMuPDF in Python, not C.
- [ ] Port useful postprocessing from `/Users/mascit/Downloads/ocrbaidu/src/ocrbaidu/postprocess.py`:
  - [ ] `parse_regions`
  - [ ] `clean_lines`
  - [ ] `merge_lines`
- [ ] Keep offer/flyer two-stage recovery as a Python-only convenience path after one-pass OCR works.
- [ ] Add a simple CLI using the Python API, not the C frontend.

## 21. Test and parity gates

- [ ] Add fast always-on tests:
  - [ ] tokenizer metadata validation
  - [ ] prompt construction
  - [ ] visual token count formulas
  - [ ] image preprocessing against upstream for tiny fixtures
  - [ ] `.uocr` synthetic loader/dumper
  - [ ] prepared-request validation
  - [ ] no-repeat-ngram processor
  - [ ] KV ring index arithmetic
- [ ] Add native CTest tests:
  - [ ] endian/alignment parsing
  - [ ] tensor directory range validation
  - [ ] qtype row-size math
  - [x] tensor id lookup
  - [x] allocation wrapper accounting/overflow/guard tests
  - [x] runtime memory accounting and KV formula tests
  - [ ] CPU reference tiny ops
- [ ] Add Metal tests that do not need full model weights:
  - [ ] compile all kernels
  - [x] run RMSNorm on synthetic tensor
  - [x] run get_rows on synthetic embedding table
  - [ ] run small RoPE kernel
  - [ ] run small SDPA prefill/decode kernel
  - [ ] run q8/q4 dot kernels once implemented
- [x] Add opt-in full-model tests guarded by env vars:
  - [x] `UOCR_MODEL_PATH`
  - [x] `UOCR_HF_DIR`
  - [x] `UOCR_RUN_LARGE_TESTS=1`
- [ ] Add opt-in integrated fp16 generation tests:
  - [x] public `uocr_generate_prepared()` text-only generated ids match Python fixture
  - [x] integrated dumped-visual-embedding generated ids/text match Python fixture
  - [ ] unsupported public paths still return clear `UOCR_ERROR_NOT_IMPLEMENTED` messages
  - [ ] no-allocation guard passes around the decode token loop
- [ ] Define fp16 parity thresholds per stage:
  - [ ] prompt embeddings: near exact within dtype tolerance
  - [ ] vision projected features: small absolute/relative tolerance after fp16 conversion
  - [ ] router top-6: exact ids for fp16 baseline
  - [ ] logits: top-k agreement and bounded max error
  - [ ] generated ids: exact for deterministic greedy smoke fixtures where possible
- [ ] Define q8/q4 parity thresholds separately:
  - [ ] router top-6 agreement must remain high because router itself is fp16
  - [ ] logits top-k/KL within calibrated thresholds
  - [ ] generated OCR text/layout stable on calibration set
- [ ] Add perf smoke tests:
  - [ ] model open time
  - [ ] first-token latency after warmup
  - [ ] decode tokens/sec single request
  - [ ] peak memory report
- [ ] Keep large conversion and full inference tests manual/opt-in until weights are available locally.

## 22. Documentation and developer workflow

- [ ] Keep `docs/foundations.md` updated when source facts change.
- [ ] Keep `docs/architecture.md` updated when API/build/memory decisions change.
- [ ] Keep this implementation plan updated as tasks complete or split.
- [ ] Add `README.md` development commands:
  - [ ] `uv sync --extra dev`
  - [ ] CMake configure/build commands
  - [ ] CTest command
  - [ ] Python pytest command
  - [ ] converter command examples
  - [ ] Python API examples
- [ ] Document full-weight download/conversion requirement once the actual safetensors file is available.
- [ ] Document memory estimates for fp16/q8/q4 and KV cache.
- [ ] Document known unsupported v1 features:
  - [ ] native C tokenizer/image loader
  - [ ] CUDA backend
  - [ ] continuous batching
- [ ] Add comments near inference code explaining non-obvious shape/order/cache choices, following DS4's quality rules.

## 23. Later / explicitly deferred work

- [ ] Add `scikit-build-core` packaging after C ABI/resource layout stabilizes.
- [ ] Bundle `libunlimitedocr.dylib/.so` and Metal resources inside Python wheels.
- [ ] Add release-path `.metallib` precompilation and fallback to source compilation.
- [ ] Add native C tokenizer by adapting DS4's JoyAI/DeepSeek byte-level BPE only if a standalone C CLI is required.
- [ ] Add native image loading/preprocessing only if a standalone C CLI is required.
- [ ] Add CUDA backend after Metal fp16 and quantized paths are stable.
- [ ] Consider experimental `Q2_K` / `IQ2_XXS` routed-expert quantization after dynamic q4 is robust.
- [ ] Consider server/OpenAI-compatible API only after the Python API is stable.
