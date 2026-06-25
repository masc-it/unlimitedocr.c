# Unlimited-OCR C implementation plan: fp16 Metal image-to-text engine

This plan tracks the remaining implementation work for one focused product path:

> **Real image in -> Python-prepared request -> reusable C/Metal fp16 engine -> generated token ids until EOS or max length -> raw decoded model sequence out.**

The plan is implementation-oriented. Validation for this
slice is manual through direct Python invocations on `docs/test.png` and related
real images.

## 0. Definition of done for this slice

End goal: a developer supplies a pre-converted fp16 `.uocr` model and a normal
image to the public Python API, then receives the raw decoded Unlimited-OCR model
sequence.

- [x] A full HF checkpoint can be downloaded outside the repo metadata cache.
- [x] The checkpoint can be converted to an fp16 `.uocr` model.
- [x] The C engine can open an fp16 `.uocr` model with the Metal backend.
- [x] The public Python wrapper can prepare a real image into a stable C request.
- [x] Public `uocr_generate_prepared()` routes fp16 Metal image requests through an image path.
- [x] The Metal vision runner is wired before the decoder in the public path.
- [x] Generated token ids are returned through `uocr_result` and decoded in Python.
- [x] Manual one-token public image generation has run on synthetic image inputs for both base/global and gundam-style requests.
- [x] Manual multi-token public image generation has produced a raw decoded model sequence on a synthetic image.
- [x] `docs/test.png` runs through `ocr_image(..., model_path="dist/unlimitedocr-fp16.uocr")` and returns a raw decoded sequence.
- [x] Generation continues until EOS or the request's max length / max-new-token budget.
- [ ] The same reusable `Engine` instance processes multiple image requests sequentially.
- [ ] Each request resets or overwrites all per-request Metal state: KV cache metadata, prompt state, image feature state, generated-token state, and error state.
- [ ] Failure modes produce clear actionable errors: pass a pre-converted `.uocr`, use fp16 qprofile, build/find `libunlimitedocr`, provide Metal resources, reduce `max_length` when memory is insufficient, or use a Metal-capable machine.

Canonical user command once this slice is complete:

```sh
UOCR_LIBRARY_PATH=build/debug/libunlimitedocr.dylib uv run python - <<'PY'
from unlimitedocr_c.ocr import ocr_image

result = ocr_image(
    "docs/test.png",
    model_path="dist/unlimitedocr-fp16.uocr",
    max_length=32768,
)
print(result.text)  # raw decoded model sequence, e.g. <|det|>...<|/det|>...
PY
```

## 1. Active execution contract

End goal: every implementation task supports the direct fp16 Metal image-to-text
contract.

- [x] Python owns image loading, preprocessing, prompt rendering, tokenization, and output decoding.
- [x] C owns model loading, request validation, memory accounting, Metal execution, KV cache, generation, and result ownership.
- [x] The active production path uses fp16 weights and activations.
- [x] The active production backend is Metal.
- [x] The runtime stays model-specific to Unlimited-OCR.
- [x] The caller supplies an explicit pre-converted `.uocr` model path for owned-engine OCR helpers.
- [x] The v1 output is the raw decoded model sequence, including model markup tokens such as `<|det|>` regions.
- [ ] All new work lands on the direct public path: Python API -> C ABI -> reusable Metal fp16 engine -> raw decoded text.
- [ ] The current implementation path for image generation is promoted from bring-up behavior to normal public API behavior.

## 2. Required model and runtime artifacts

End goal: the user explicitly passes a local fp16 `.uocr` file that the C engine
can mmap and Metal can bind with no-copy model views.

- [x] Download the full `baidu/Unlimited-OCR` HF checkpoint into a local directory.
  - Current local path used during bring-up: `data/hf/Unlimited-OCR`.
  - Required payload: `model-00001-of-000001.safetensors`.
- [x] Convert the full checkpoint to fp16 `.uocr`.
  - Current local artifact: `dist/unlimitedocr-fp16.uocr`.
  - Current conversion stats: `dist/unlimitedocr-fp16.conversion-stats.json`.
- [x] Validate model-file structure with `uocr-dump`.
- [x] Preserve all current checkpoint tensor accounting: `2710` source tensors, `2709` runtime tensors, `1` preserved-unused tensor, `0` omitted tensors.
- [x] Store fp16 tensor payloads in a page-aligned tensor-data section suitable for Metal `newBufferWithBytesNoCopy` views.
- [ ] Keep `model_path` explicit in public OCR calls for owned engines.
- [ ] Produce a direct error for an empty `model_path`: pass a pre-converted fp16 `.uocr` path via `model_path=...`.
- [ ] Produce a direct error for directories, safetensors payloads, metadata cache paths, and wrong-format files, including the conversion command the user should run first.
- [ ] Reject non-fp16 `.uocr` qprofiles with a precise message saying public image OCR currently requires fp16.
- [ ] Document that the local workflow uses a converted fp16 `.uocr` artifact at inference time.

## 3. Public Python entrypoint

End goal: the normal user-facing Python path accepts an image path/PIL image and
an explicit pre-converted model path, then returns the raw decoded generated
sequence.

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
- [x] Implement `ocr_image()` with upstream-style prompt defaults.
- [x] Implement `ocr_pages()` and `ocr_pdf()` as Python-side conveniences.
- [x] Decode returned ids with the same tokenizer and trim one trailing EOS marker.
- [ ] Make explicit `model_path` mandatory whenever the helper opens its own engine.
- [ ] Use `max_length` semantics in high-level OCR helpers: `max_new_tokens = max_length - prompt_tokens`.
- [ ] Generate through the full length-derived budget unless EOS stops generation first.
- [ ] Keep smaller generation budgets explicit through caller-provided `max_new_tokens` or equivalent arguments.
- [ ] Ensure Python exceptions include the native `uocr_last_error()` detail and the concrete user action.
- [ ] Add a minimal CLI wrapper after the Python API path works, with required `--model /path/to/model.uocr`.

## 4. Reusable engine contract

End goal: one opened Metal engine runs multiple image requests sequentially while
reusing the loaded model and long-lived arenas.

- [x] Expose `Engine` and `EngineOptions` in Python.
- [x] Allow `generate_prepared(request, engine=engine)` to use an existing engine.
- [x] Keep the `.uocr` model mmap and Metal no-copy views alive for the engine lifetime.
- [x] Allocate long-lived Metal arenas at engine creation.
- [ ] Document the reusable-engine usage pattern for multiple images.
- [ ] Ensure `ocr_image(..., engine=engine)` and `ocr.generate(..., engine=engine)` use the already-loaded model in the engine.
- [ ] Ensure high-level helpers size owned engines from the prepared request.
- [ ] Ensure reusable engines reject oversize requests with an actionable error including actual and configured limits.
- [ ] Reset per-request sequence metadata before each call: prompt length, image span, generated count, EOS flag, no-repeat history, and last-token state.
- [ ] Reset or overwrite per-request Metal metadata before each call: `has_integrated_prefill`, prefill slot, prefill token count, final hidden segment, KV write positions, and token slot contents.
- [ ] Leave the engine in a known state after any vision, prefill, or decode failure.
- [ ] Reuse model tensor bindings across requests on the same engine.

Reusable-engine target API:

```python
from unlimitedocr_c.ffi import Engine, EngineOptions
from unlimitedocr_c.ocr import ocr_image, default_resource_path

with Engine(EngineOptions(
    model_path="dist/unlimitedocr-fp16.uocr",
    backend="metal",
    resource_path=default_resource_path("metal"),
    max_batch=1,
    max_prompt_tokens=4096,
    max_gen_tokens=512,
)) as engine:
    first = ocr_image("docs/test.png", engine=engine, max_length=4096)
    second = ocr_image("docs/test.png", engine=engine, max_length=4096)
```

## 5. C ABI and public generation dispatch

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
- [ ] Clear previous per-run error and generation state at the start of every `uocr_generate_prepared()` call.
- [ ] Reset Metal integrated decoder bookkeeping for the target slot before every prefill.
- [ ] Distinguish public generation errors by category: model open failure, invalid request, vision failure, prompt assembly failure, prefill failure, decode failure, OOM, and unsupported qprofile.
- [ ] Add a public way to retrieve generated count and stopped-on-EOS status if useful for debugging long OCR calls.

## 6. `.uocr` loader and model binding

End goal: model open builds all validated fp16 tensor bindings needed by the
public E2E image path and fails early if any required tensor is missing.

- [x] Implement mmap loader for `.uocr` header, sections, config, tokenizer metadata, provenance, tensor directory, and tensor payloads.
- [x] Validate compiled Unlimited-OCR constants against the `.uocr` config.
- [x] Validate tensor payload ranges and alignment.
- [x] Keep tensor payloads mmap-backed.
- [x] Plan page-aligned Metal no-copy model views.
- [x] Wrap model views with `newBufferWithBytesNoCopy`.
- [x] Cache per-tensor Metal binding `{buffer, inner_offset, payload_size}`.
- [x] Build and validate decoder tensor bindings for token embedding, final norm, LM head, all layer norms, Q/K/V/O projections, layer-0 dense MLP, MoE routers, routed experts, and shared experts.
- [x] Build and validate vision tensor bindings for SAM, CLIP, projector, `IMAGE_NEWLINE`, and `VIEW_SEPARATOR`.
- [x] Treat CLIP raw-pixel patch embedding as preserved-unused for the normal OCR path.
- [x] Fail engine open early when an fp16 model lacks required vision bindings.
- [ ] Confirm all production-required binding errors include the source tensor name or stable tensor id.
- [ ] Reject missing, unreadable, wrong-magic, wrong-version, or non-fp16 `.uocr` files with direct instructions for the user.

## 7. Metal backend runtime foundation

End goal: Metal context has all reusable resources needed to run image requests
while reusing the runtime across tokens and images.

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
- [ ] Ensure real model open plus `docs/test.png` generation fits the default memory budget on the target machine.
- [ ] When memory admission fails, error with the estimated bytes and tell the user to lower `max_length` / `max_new_tokens` or use a machine with more memory.
- [ ] When Metal resources are missing or shader compilation fails, error with the resource path that was searched and how to pass `resource_path`.
- [ ] Reduce unnecessary per-call CPU allocations in the image path once correctness is established.
- [ ] Keep the per-token decode loop allocation-free; move any accidental allocation outside the loop.

## 8. Metal vision encoder: public pixels to formatted visual features

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
- [ ] Keep all vision temporary storage in bounded/reusable arenas where possible.
- [ ] Add clearer runtime diagnostics that name the failed vision stage: SAM patch, SAM transformer, SAM neck, CLIP, projector, or formatter.
- [ ] Profile `docs/test.png` and remove obvious serial CPU/GPU synchronization points that make image encoding unusably slow.

## 9. Metal decoder: prompt embeddings to generated token ids

End goal: text embeddings plus formatted image features run through the fp16
Unlimited-OCR decoder and produce greedy generated ids until EOS or max length.

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
- [ ] Ensure high-level OCR requests set `max_new_tokens = max_length - prompt_tokens` when max length semantics are requested.
- [ ] Generate until EOS or the full max-new-token budget.
- [ ] Run `docs/test.png` long enough to expose and fix multi-token decode-state bugs.
- [ ] Confirm repeated calls start from clean KV cache metadata, image features, and generated-token state.
- [ ] Keep CPU readbacks limited to what the current greedy/no-repeat implementation needs.
- [ ] If generated text is empty or immediately EOS for `docs/test.png`, inspect prompt construction, image feature assembly, LM-head logits, and no-repeat settings in that order.

## 10. Request and visual-token contract

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
- [ ] Ensure owned Python engines choose `max_prompt_tokens` and `max_gen_tokens` exactly from the prepared request when opening an engine.
- [ ] Ensure reusable engines reject oversize prompts or generation budgets with an error that includes actual and configured limits.

## 11. Python decoding and result ergonomics

End goal: generated ids from C turn into the raw decoded Unlimited-OCR sequence.
This raw model markup is the accepted v1 output for this slice.

- [x] Store generated ids in `uocr_result` with one output sequence per request.
- [x] Copy generated ids into NumPy arrays on the Python side before freeing `uocr_result`.
- [x] Decode generated ids with the HF tokenizer.
- [x] Decode with `skip_special_tokens=False` to preserve raw model markup.
- [x] Trim one trailing EOS marker if present.
- [x] Strip surrounding whitespace from the decoded output.
- [x] Return a `GenerationResult` with both `token_ids` and `text`.
- [ ] Preserve enough metadata in `GenerationResult` for practical debugging: prompt length, generated count, stopped-on-EOS if exposed, preset, view count, visual token count.
- [ ] Add optional printing of generated token ids for manual bring-up through public helpers.
- [ ] Keep clean-text/layout postprocessing in a future phase; raw decoded markup is enough for this slice.

## 12. Immediate implementation sequence

End goal: turn the current mostly-wired code into a reusable real-image OCR call
on `docs/test.png`.

- [x] Download full HF checkpoint to `data/hf/Unlimited-OCR`.
- [x] Convert full checkpoint to `dist/unlimitedocr-fp16.uocr`.
- [x] Confirm `uocr-dump dist/unlimitedocr-fp16.uocr` loads and reports valid fp16 tensor accounting.
- [x] Run one synthetic/file-path image through `ocr.generate(..., preset="base", max_new_tokens=1)` with `dist/unlimitedocr-fp16.uocr`.
- [x] Run one synthetic image through `ocr.generate(..., preset="gundam", max_new_tokens=1)` with `dist/unlimitedocr-fp16.uocr`.
- [x] Run one synthetic image through multi-token public generation and return raw decoded markup.
- [ ] Run `docs/test.png` through `ocr_image(..., model_path="dist/unlimitedocr-fp16.uocr", max_length=32768)`.
- [ ] If engine open fails, fix model binding, qprofile, resource path, or memory-budget handling first.
- [ ] If vision fails, fix the named SAM/CLIP/projector/formatter stage in the public `encode_visual_features_f16` path.
- [ ] If prompt assembly or prefill fails, fix the decoder input span and arena setup before changing decoder math.
- [ ] If first-token generation fails, fix final norm / LM head / argmax / no-repeat wiring.
- [ ] If long generation fails, fix decode-loop position ids, generated-ring indexing, no-repeat history, token embedding for generated tokens, or arena slice selection.
- [ ] Run `docs/test.png` twice through the same reusable `Engine` and confirm both calls produce independent decoded sequences.
- [ ] Run `docs/test.png` with `preset="base"` and `preset="gundam"` if the image dimensions exercise both paths usefully.
- [ ] Add a minimal CLI requiring `--model` after direct Python function calls work.

## 13. Operational diagnostics for the real-image path

End goal: a failed or slow `docs/test.png` run tells the developer where to act.

- [ ] Report or log timing for model open, preprocessing, vision encode, prompt prefill, first token, and decode tokens/sec.
- [ ] Include memory estimate details in OOM/admission failures.
- [ ] Include searched library path details when `libunlimitedocr` cannot be loaded.
- [ ] Include searched Metal resource paths when Metal source/metallib loading fails.
- [ ] Include the exact request limits when prompt length or generation length exceeds the engine configuration.
- [ ] Include the failing vision stage name for SAM/CLIP/projector/formatter failures.
- [ ] Include selected backend, qprofile, prompt tokens, image tokens, view count, and requested max-new-token count in manual debug output.

## 14. Future phases after fp16 image-to-text

End goal: keep follow-up work ordered after the fp16 Metal E2E engine is usable.

- [ ] q8 public inference and q8 parity work.
- [ ] q4 public inference, q4 calibration, and promotion policies.
- [ ] Static or continuous batching.
- [ ] CUDA backend.
- [ ] Native C tokenizer or native C image loader.
- [ ] Clean-text/layout postprocessing and offer/flyer multi-stage flows.
- [ ] Wheel packaging and bundled Metal resources.
- [ ] Server APIs.
- [ ] Additional automated test infrastructure or new parity fixture generation.

## 15. Profiling baseline for future optimization

End goal: every real-image run can produce enough speed and memory visibility to
rank future code and kernel optimization work by measured impact.

- [ ] Add an opt-in profiling mode for the public Python path, e.g. `profile=True` or `UOCR_PROFILE=1`.
- [ ] Emit one structured profiling record per request, preferably JSON plus concise human-readable summary.
- [ ] Include request metadata: image path, preset, view count, crop grid, prompt tokens, image tokens, requested max-new-token count, generated count, EOS status, backend, qprofile, and model path basename.
- [ ] Record wall-clock timings for Python preprocessing, engine open, request validation, memory admission, vision encoding, prompt assembly, decoder prefill, first token, full decode loop, total generation, and Python decoding.
- [ ] Record Metal-stage timings for SAM patch embedding, SAM positional embedding, each SAM transformer block group, SAM neck, CLIP embedding, each CLIP transformer block group, projector, visual formatter, prompt assembly, each decoder layer prefill, final norm, LM head, no-repeat processing, argmax, and each decode step.
- [ ] Record throughput metrics: image tokens/sec for vision, prompt tokens/sec for prefill, decode tokens/sec, first-token latency, average decode-token latency, p50/p95 decode-token latency, and total tokens/sec.
- [ ] Record memory metrics: model view bytes, KV cache bytes, prompt arena bytes, vision scratch high-water mark, decoder scratch high-water mark, MoE scratch high-water mark, logits/readback bytes, transient buffer high-water mark, total estimated bytes, and total live/peak bytes.
- [ ] Record Metal resource metrics when available: recommended working set, selected memory budget, model view count, tensor binding count, scratch buffer capacities, command-buffer count, and CPU readback byte count.
- [ ] Persist profiling artifacts under a user-selected directory such as `dist/profiles/`, with filenames including timestamp, preset, generated token count, and image stem.
- [ ] Add a compact summary table that ranks the top time-consuming stages and top memory-consuming arenas for `docs/test.png`.
- [ ] Produce baseline profiles for `docs/test.png` with `preset="base"`, `preset="gundam"`, `max_length=4096`, and the default OCR settings.
- [ ] Use profiling output to label future optimization targets by stage: vision kernels, decoder prefill, decode loop, LM head/readback, memory allocation, or Python frontend.
