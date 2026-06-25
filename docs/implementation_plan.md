# Unlimited-OCR C implementation plan: fp16 Metal image-to-text engine

This plan is now intentionally narrow.  The only active goal is:

> **Real image in -> Python-prepared request -> C/Metal fp16 inference -> generated token ids -> decoded text out.**

Implementation work only.  Do not spend time adding new test suites, parity fixture
infrastructure, q8/q4 runtime work, batching, CUDA, or packaging polish until the
fp16 Metal image path can be used directly on real images.

The frontend remains Python for v1.  The inference core remains C/Metal.

## 0. Definition of done for this slice

End goal: a developer can point the Python API at a normal image and a converted
fp16 `.uocr` model and receive decoded text without touching internal debug APIs.

- [x] A full HF checkpoint can be downloaded outside the repo metadata cache.
- [x] The checkpoint can be converted to an fp16 `.uocr` model.
- [x] The C engine can open an fp16 `.uocr` model with the Metal backend.
- [x] The public Python wrapper can prepare a real image into a stable C request.
- [x] Public `uocr_generate_prepared()` has an image path instead of returning `UOCR_ERROR_NOT_IMPLEMENTED` for fp16 Metal image requests.
- [x] The Metal vision runner is wired before the decoder in the public path.
- [x] Generated token ids are returned through `uocr_result` and decoded in Python.
- [ ] A real single-image call such as `ocr_image("page.png", model_path="dist/unlimitedocr-fp16.uocr")` runs to completion and returns a decoded sequence.
- [ ] The same public path works with `preset="base"` and `preset="gundam"`.
- [ ] The public path can generate more than one token for useful OCR output, not just a first-token smoke run.
- [ ] The implementation handles a failed run with actionable user-facing errors: model missing, Metal unavailable, out of memory, invalid request, or unsupported qprofile.

Canonical user command once this slice is complete:

```sh
UOCR_LIBRARY_PATH=build/debug/libunlimitedocr.dylib uv run python - <<'PY'
from unlimitedocr_c.ocr import ocr_image

result = ocr_image(
    "path/to/image.png",
    model_path="dist/unlimitedocr-fp16.uocr",
    max_gen_tokens=128,
)
print(result.text)
PY
```

## 1. Hard scope boundaries

End goal: keep all work directed at the fp16 Metal image-to-text path.

- [x] Python owns image loading, preprocessing, prompt rendering, tokenization, and output decoding.
- [x] C owns model loading, request validation, memory accounting, Metal execution, KV cache, generation, and result ownership.
- [x] Start and stay on fp16 weights/activations for this slice.
- [x] Use Metal as the only production backend for this slice.
- [x] Keep the runtime model-specific to Unlimited-OCR; do not generalize into a transformer framework.
- [x] Keep q8/q4 code that already exists, but do not route public image OCR through it for this slice.
- [x] Keep CUDA, batching, calibration, server APIs, native C tokenization, native C image loading, and PDF/postprocess polish out of scope.
- [ ] Remove or bypass any remaining implementation blockers that still assume image generation is a diagnostic-only path.
- [ ] Keep all new work on the direct public path: Python API -> C ABI -> Metal fp16 -> decoded text.

## 2. Local model artifacts needed for real use

End goal: a developer has a real fp16 `.uocr` file that the C engine can mmap and
Metal can bind without copying the full model.

- [x] Download the full `baidu/Unlimited-OCR` HF checkpoint into a local directory.
  - Current local path used during bring-up: `data/hf/Unlimited-OCR`.
  - Required payload: `model-00001-of-000001.safetensors`.
- [x] Avoid relying on the repo's `data/context` metadata-only cache for inference; it has no full safetensors payload.
- [x] Convert the full checkpoint to fp16 `.uocr`.
  - Current local artifact: `dist/unlimitedocr-fp16.uocr`.
  - Current conversion stats: `dist/unlimitedocr-fp16.conversion-stats.json`.
- [x] Validate model-file structure with `uocr-dump`.
- [x] Preserve all current checkpoint tensor accounting: `2710` source tensors, `2709` runtime tensors, `1` preserved-unused tensor, `0` omitted tensors.
- [x] Store fp16 tensor payloads in a page-aligned tensor-data section suitable for Metal `newBufferWithBytesNoCopy` views.
- [ ] Add a small model-discovery helper in Python so `model_path` can default to `UOCR_MODEL_PATH` or `dist/unlimitedocr-fp16.uocr` when present.
- [ ] Make converter/runtime diagnostics explicitly tell the user when they accidentally pass the metadata cache directory or an unconverted safetensors file instead of `.uocr`.
- [ ] Document the minimum disk footprint for the local workflow: HF safetensors plus fp16 `.uocr` are both about 6.2 GiB.

## 3. Public Python entrypoint

End goal: the normal user-facing Python path accepts an image path/PIL image and
returns a decoded generated sequence.

- [x] Implement `src/unlimitedocr_c/frontend.py` as the v1 frontend.
- [x] Load the HF `tokenizer.json` with `tokenizers`.
- [x] Validate tokenizer constants: BOS `0`, EOS `1`, PAD `2`, `<image>` `128815`, vocab `129280`.
- [x] Render the upstream-compatible plain prompt.
- [x] Build prompt token ids with repeated image placeholders and a matching `image_mask`.
- [x] Preprocess images with PIL: EXIF transpose, RGB conversion, padding, resize, `[-1,1]` normalization, contiguous NCHW.
- [x] Support `base` preset: one global `1024x1024` view and `273` visual tokens.
- [x] Support `gundam` preset: local `640x640` crops first, global `1024x1024` view last, and correct visual-token count.
- [x] Support `prepare_pages()` for base-mode multi-page requests.
- [x] Implement Python `ctypes` bindings that keep NumPy buffers alive for the duration of the C call.
- [x] Implement `Engine.generate_prepared()` returning generated token-id arrays.
- [x] Implement `ocr.generate()` for single-image convenience.
- [x] Implement `ocr_image()` with upstream-style defaults.
- [x] Implement `ocr_pages()` and `ocr_pdf()` as Python-side conveniences.
- [x] Decode returned ids with the same tokenizer and trim one trailing EOS marker.
- [ ] Add default model-path lookup in `ocr.generate()`, `ocr_image()`, and `ocr_pages()`.
- [ ] Add a minimal CLI wrapper around `ocr_image()` for manual use, e.g. `uv run tools/uocr-ocr image.png --model dist/unlimitedocr-fp16.uocr`.
- [ ] Ensure Python errors include the native `uocr_last_error()` detail without hiding the actionable part.
- [ ] Make the default generation budget useful for OCR while still allowing `max_gen_tokens=1` for quick first-token bring-up.

## 4. C ABI and public generation dispatch

End goal: public C ABI accepts prepared image requests and dispatches exactly one
fp16 Metal image request through vision and decoder.

- [x] Define stable public types in `include/unlimitedocr.h`:
  - [x] `uocr_engine`
  - [x] `uocr_result`
  - [x] `uocr_image_view`
  - [x] `uocr_prepared_request`
  - [x] `uocr_engine_opts`
- [x] Export public engine/result functions:
  - [x] `uocr_engine_open`
  - [x] `uocr_engine_close`
  - [x] `uocr_generate_prepared`
  - [x] `uocr_result_tokens`
  - [x] `uocr_result_free`
  - [x] `uocr_last_error`
  - [x] `uocr_engine_memory_report`
- [x] Validate token ids, BOS, prompt length, generation length, image mask, view dimensions, and view order before inference.
- [x] Build a sequence state with prompt length, contiguous image span, generated buffer metadata, no-repeat config, position counters, and EOS status.
- [x] Reject unsupported batch sizes for generation with a clear message.
- [x] Route single-request text-only fp16 Metal calls through the integrated decoder.
- [x] Route single-request image fp16 Metal calls through vision encoding and then the integrated decoder.
- [x] Allocate and own temporary generated-token buffers before packaging `uocr_result`.
- [x] Validate image span length equals final formatted visual-feature rows before decoder prefill.
- [ ] Harden repeated `uocr_generate_prepared()` calls on one engine by explicitly resetting all per-run Metal context state before vision and decoder work.
- [ ] Ensure public generation errors distinguish vision failure, prompt assembly failure, prefill failure, decode failure, OOM, and unsupported qprofile.
- [ ] Add a public way to retrieve last generated token count and EOS status if useful for debugging long OCR calls.

## 5. `.uocr` loader and model binding

End goal: model open builds all validated fp16 tensor bindings needed by the
public E2E image path and fails early if any required tensor is missing.

- [x] Implement mmap loader for `.uocr` header, sections, config, tokenizer metadata, provenance, tensor directory, and tensor payloads.
- [x] Validate compiled Unlimited-OCR constants against the `.uocr` config.
- [x] Validate tensor payload ranges and alignment.
- [x] Keep tensor payloads mmap-backed; do not copy full weights into CPU or GPU memory.
- [x] Plan page-aligned Metal no-copy model views.
- [x] Wrap model views with `newBufferWithBytesNoCopy`.
- [x] Cache per-tensor Metal binding `{buffer, inner_offset, payload_size}`.
- [x] Build and validate decoder tensor bindings for:
  - [x] token embedding
  - [x] final norm
  - [x] LM head
  - [x] all decoder layer norms
  - [x] all Q/K/V/O projections
  - [x] layer-0 dense MLP
  - [x] MoE routers
  - [x] routed experts
  - [x] shared experts
- [x] Build and validate vision tensor bindings for:
  - [x] SAM encoder
  - [x] CLIP encoder
  - [x] projector
  - [x] `IMAGE_NEWLINE`
  - [x] `VIEW_SEPARATOR`
- [x] Treat CLIP raw-pixel patch embedding as preserved-unused for the normal OCR path.
- [x] Fail engine open early when an fp16 model lacks required vision bindings.
- [ ] Confirm all production-required binding errors include the source tensor name or stable tensor id.
- [ ] Keep q8/q4 `.uocr` artifacts rejected by public fp16 image generation with a precise message.

## 6. Metal backend runtime foundation

End goal: Metal context has all reusable resources needed to run one fp16 image
request without building a new runtime for each token.

- [x] Create Metal device and command queue.
- [x] Compile Metal source at runtime in development builds.
- [x] Support precompiled `.metallib` as a release path.
- [x] Cache compute pipelines by function name.
- [x] Track transient shared buffers until command-buffer completion.
- [x] Allocate named scratch buffers that can grow to high-water marks.
- [x] Allocate long-lived runtime arenas for prompt embeddings, hidden-state ping-pong, KV cache, router/top-k, MoE, logits, and token slots.
- [x] Use mmap-backed no-copy buffers for model weights.
- [x] Use private storage for GPU-only scratch where practical and shared storage for CPU-filled/readback buffers.
- [x] Maintain memory accounting estimates and expose public memory reports.
- [x] Use Metal recommended working-set size to choose a default memory budget when available.
- [ ] Ensure real model open plus one image generation fits the default memory budget on target machines; otherwise improve the default or error message.
- [ ] Reduce unnecessary per-call CPU allocations in the image path once correctness is established.
- [ ] Keep the per-token decode loop allocation-free; move any accidental new allocation outside the loop.

## 7. Metal vision encoder: public pixels to formatted visual features

End goal: public `uocr_image_view` pixels become a contiguous fp16 feature buffer
`[image_tokens,1280]` in the exact order the decoder prompt expects.

- [x] Accept public `UOCR_PIXEL_F16_NCHW` pixels.
- [x] Accept public `UOCR_PIXEL_F32_NCHW` pixels.
- [x] Support one global `1024x1024` view.
- [x] Support zero or more local `640x640` views followed by one global view.
- [x] Process local crops in chunks to avoid holding all crop intermediates at once.
- [x] Implement SAM patch embedding from pixels.
- [x] Add/interpolate SAM absolute position embeddings.
- [x] Implement all SAM transformer blocks with local/global attention, relative position bias, LayerNorm, GELU MLP, and residuals.
- [x] Implement SAM neck and downsampling stack to produce `1024 x grid x grid` features.
- [x] Implement CLIP embeddings from SAM features, including CLS and interpolated absolute positions.
- [x] Implement all CLIP transformer blocks with full attention, LayerNorm, QuickGELU MLP, and residuals.
- [x] Concatenate CLIP token features and SAM spatial features into `2048`-wide projector input rows.
- [x] Apply projector `2048 -> 1280` with bias.
- [x] Format global view features as `16` rows of `16` projected tokens plus `IMAGE_NEWLINE`, followed by `VIEW_SEPARATOR`.
- [x] Format local crop features in upstream stitched row order, with newline after each stitched local row.
- [x] Use crop-mode final order: local rows first, then global rows/newlines, then separator.
- [x] Use non-crop/multi-page order: each global page rows/newlines plus separator in page order.
- [x] Return final formatted visual features to the public image generation path.
- [ ] Keep all vision temporary storage in bounded/reusable arenas instead of ad-hoc heap allocation where possible.
- [ ] Add clearer runtime diagnostics that name the failed vision stage: SAM patch, SAM transformer, SAM neck, CLIP, projector, or formatter.
- [ ] Profile first real run and remove obvious serial CPU/GPU synchronization points that make image encoding unusably slow.

## 8. Metal decoder: prompt embeddings to generated token ids

End goal: text embeddings plus formatted image features run through the fp16
Unlimited-OCR decoder and produce greedy generated ids.

- [x] Gather token embeddings from mmap-backed fp16 token embedding weights.
- [x] Assemble prompt embeddings directly into the Metal prompt arena.
- [x] Replace the contiguous image span with formatted visual feature rows.
- [x] Implement fp16 RMSNorm with fp32 accumulation.
- [x] Implement decoder attention Q/K/V/O projections.
- [x] Implement split-half RoPE with theta `10000`.
- [x] Implement KV cache writes for prompt and generated tokens.
- [x] Implement full-prompt causal prefill attention.
- [x] Implement decode attention over full prompt plus generated-ring tokens.
- [x] Implement layer-0 dense SwiGLU MLP.
- [x] Implement MoE router softmax and top-6 greedy expert selection.
- [x] Implement selected routed expert execution.
- [x] Implement shared experts and routed/shared/residual combine.
- [x] Implement final RMSNorm and LM head.
- [x] Implement no-repeat-ngram banning before greedy argmax.
- [x] Stop generation on EOS id `1` or `max_new_tokens`.
- [x] Return generated ids via `uocr_result_create_from_generated()`.
- [x] Use a generated-token ring window of `128` while keeping prompt tokens fully attendable.
- [ ] Run the public real-image path long enough to expose and fix multi-token decode-state bugs.
- [ ] Confirm repeated calls do not leak old KV cache, image features, or generated-token state across requests.
- [ ] Keep CPU readbacks limited to what the current greedy/no-repeat implementation needs.
- [ ] If generated text is empty or immediately EOS for normal OCR images, inspect prompt construction, image feature assembly, LM-head logits, and no-repeat settings in that order.

## 9. Request and visual-token contract

End goal: every supported Python-prepared image request has exactly the visual
feature span expected by C/Metal before inference starts.

- [x] Require BOS at token position `0`.
- [x] Count `image_mask == 1` placeholders.
- [x] Require one contiguous image placeholder span for image generation.
- [x] Validate token ids are within `[0,129280)`.
- [x] Validate global views are exactly `1024x1024`.
- [x] Validate local views are exactly `640x640`.
- [x] Validate base single image: one global view, `273` visual tokens.
- [x] Validate gundam `[1,1]`: one global view, `273` visual tokens.
- [x] Validate gundam real crop: row-major locals then global, with local visual count plus global/separator count.
- [x] Validate multi-page base: global views in page order, `273` visual tokens per page.
- [x] Validate prompt length plus generation length within engine and model limits.
- [x] Validate KV budget before running vision/prefill.
- [x] Build no-repeat config for upstream defaults `35/128` and `35/1024`.
- [ ] Make request validation messages concise enough to show directly in Python exceptions.
- [ ] Ensure high-level Python wrappers choose engine `max_prompt_tokens` and `max_gen_tokens` from the prepared request so ordinary real images are not rejected by default engine limits.

## 10. Python decoding and result ergonomics

End goal: generated ids from C turn into the same decoded text shape a user
expects from the upstream wrapper.

- [x] Store generated ids in `uocr_result` with one output sequence per request.
- [x] Copy generated ids into NumPy arrays on the Python side before freeing `uocr_result`.
- [x] Decode generated ids with the HF tokenizer.
- [x] Decode with `skip_special_tokens=False` to match the current upstream-style path.
- [x] Trim one trailing EOS marker if present.
- [x] Strip surrounding whitespace from the decoded output.
- [x] Return a `GenerationResult` with both `token_ids` and `text`.
- [ ] Preserve enough metadata in `GenerationResult` for practical debugging: prompt length, generated count, stopped-on-EOS if exposed, preset, view count, visual token count.
- [ ] Add optional printing of generated token ids for manual bring-up without requiring users to import internal helpers.
- [ ] Keep output postprocessing out of this slice except EOS/whitespace cleanup.

## 11. Immediate implementation sequence

End goal: turn the current mostly-wired code into a usable real-image OCR call.

- [x] Download full HF checkpoint to `data/hf/Unlimited-OCR`.
- [x] Convert full checkpoint to `dist/unlimitedocr-fp16.uocr`.
- [x] Confirm `uocr-dump dist/unlimitedocr-fp16.uocr` loads and reports valid fp16 tensor accounting.
- [ ] Run one real image through `ocr.generate(..., preset="base", max_new_tokens=1)` with `dist/unlimitedocr-fp16.uocr`.
- [ ] If engine open fails, fix model binding, qprofile, resource path, or memory-budget handling first.
- [ ] If vision fails, fix the named SAM/CLIP/projector/formatter stage in the public `encode_visual_features_f16` path.
- [ ] If prompt assembly or prefill fails, fix the decoder input span and arena setup before changing decoder math.
- [ ] If first-token generation fails, fix final norm / LM head / argmax / no-repeat wiring.
- [ ] Increase to `max_new_tokens=8` and fix decode-loop state issues.
- [ ] Increase to `max_gen_tokens=128` through `ocr_image()` and fix performance or memory issues that block useful text output.
- [ ] Run the same public wrapper with `preset="gundam"` on a normal document-like image.
- [ ] Run a multi-page `ocr_pages()` request if single-image base/gundam are stable.
- [ ] Add default model lookup and a minimal CLI only after direct Python function calls work.

## 12. Known implementation risks to resolve in this slice

End goal: prevent likely real-use failures from becoming hidden side quests.

- [ ] **Memory pressure:** fp16 weights plus KV/vision/decoder scratch may exceed default Metal budget on smaller machines.  Prefer clear errors and smaller generation defaults over silent failure.
- [ ] **First-run latency:** runtime Metal compilation plus model mapping plus vision may be slow.  Keep it acceptable for manual use before optimizing deeply.
- [ ] **Per-call vision allocation:** current public vision path should be hardened so large crop grids do not repeatedly allocate/free large CPU buffers.
- [ ] **Repeated generation:** persistent arenas and KV state must be reset or overwritten deterministically between calls.
- [ ] **Long generation:** multi-token decode must preserve position ids, ring indexing, no-repeat history, and generated token embeddings correctly.
- [ ] **Prompt/image span mismatch:** image placeholders, final visual-feature rows, and prompt assembly must agree for base, gundam, and pages.
- [ ] **Immediate EOS/empty text:** if real images generate EOS immediately, inspect input prompt, image feature values/order, logits, and no-repeat banning before changing sampling behavior.
- [ ] **User-facing errors:** Python exceptions should include the native detail and the remedial action when obvious.

## 13. Explicitly deferred until fp16 image-to-text works

End goal: avoid diluting the E2E fp16 work.

- [ ] q8 public inference and q8 parity work.
- [ ] q4 public inference, q4 calibration, and promotion policies.
- [ ] Static or continuous batching.
- [ ] CUDA backend.
- [ ] Native C tokenizer or native C image loader.
- [ ] Rich OCR postprocessing, layout recovery, and offer/flyer multi-stage flows.
- [ ] Wheel packaging and bundled Metal resources.
- [ ] Server APIs.
- [ ] Additional automated test infrastructure or new parity fixture generation.
