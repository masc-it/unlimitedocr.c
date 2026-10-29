# Unlimited-OCR-2 research notes

Checkpoint/inference and scheduler source research complete; numerical inference parity remains unverified.

Implementation scope and order: [plan_unlimitedocr2.md](plan_unlimitedocr2.md).

## Goal and scope

Add `baidu/Unlimited-OCR-2` to the native C/Metal engine while preserving v1 behavior and existing model files. Required scope includes single-page, base-mode multi-page inference, and bounded-memory document scheduling with upstream-compatible continuation. Single-window inference is an intermediate checkpoint. Keep model selection separate from image profile (`base`/`gundam`) and quantization (`fp16`/`q8`/`q4`).

## Research evidence

- [done] Inspected the Hugging Face model card, configuration, tokenizer, checkpoint index, both safetensors headers, and inference source.
- [done] Compared these with `data/context/` and the current converter, frontend, runtime, and Metal implementation.
- [done] Confirmed that all 2,710 v1 tensor names exist in v2 with identical shapes and dtypes. This does not imply identical weight values.
- [done] Inspected the pinned customized SGLang wheel's window planner, KV copy/splice, page guard, and cache relocation. Exact contract is recorded below.
- [todo] Run upstream inference and capture numerical reference fixtures. No full weights were downloaded or model inference executed during this investigation.

Upstream revision: `c14071b549f4fcab10909f3c4acc6a2edb366947`.
Local research downloads are temporary files under `/tmp/unlimitedocr2-research-c14071b/`, not committed fixtures. Only source, tokenizer, metadata, weight headers, and the SGLang wheel were downloaded. The wheel was extracted for inspection, not installed or executed.

Sources:

- [Model card](https://huggingface.co/baidu/Unlimited-OCR-2/blob/c14071b549f4fcab10909f3c4acc6a2edb366947/README.md)
- [Configuration](https://huggingface.co/baidu/Unlimited-OCR-2/blob/c14071b549f4fcab10909f3c4acc6a2edb366947/config.json)
- [Inference/router implementation](https://huggingface.co/baidu/Unlimited-OCR-2/blob/c14071b549f4fcab10909f3c4acc6a2edb366947/modeling_deepseekv2.py)
- [Page coordinates and base decoder](https://huggingface.co/baidu/Unlimited-OCR-2/blob/c14071b549f4fcab10909f3c4acc6a2edb366947/_sparse_base_impl.py)
- [Router modules and normalization](https://huggingface.co/baidu/Unlimited-OCR-2/blob/c14071b549f4fcab10909f3c4acc6a2edb366947/dash_attention.py)
- [Input preparation and generation](https://huggingface.co/baidu/Unlimited-OCR-2/blob/c14071b549f4fcab10909f3c4acc6a2edb366947/modeling_unlimitedocr.py)
- [Checkpoint index](https://huggingface.co/baidu/Unlimited-OCR-2/blob/c14071b549f4fcab10909f3c4acc6a2edb366947/model.safetensors.index.json)

## Architecture comparison

| Component | Current v1 | v2 |
| --- | --- | --- |
| Decoder | 12 layers, hidden 1280, 10 attention/KV heads, head dim 128 | Same dimensions |
| MoE | 64 routed experts, top-6, 2 shared; first layer dense | Same dimensions/configuration |
| Vision | SAM + CLIP, linear 2048-to-1280 projector | Same; encoder source differs only in comments |
| Views | Global 1024, local 640; 273 global visual tokens | Same sizes/count; default crop cap becomes 16 instead of 32 |
| Vocabulary | 129280 output rows; 830 added-token entries | Same output rows; 834 added-token entries |
| Checkpoint | 2710 tensors; 6,672,212,480 payload bytes; one shard | 2819 tensors; 6,687,958,536 bytes; two shards |
| Prefill attention | Dense causal | DASH/SR-SWA sparse attention in all 12 layers |
| Decode attention | All prompt tokens + 128 generated-token ring | Selected remote blocks + 128-token local window |
| Positions | 1D RoPE, theta 10000; cap 32768 | Page-aware 2D RoPE; configured cap 131072 |

### New tensors

Each of the 12 decoder layers adds:

- `self_attn.dash_router.q_proj.weight` and `k_proj.weight`: `[128, 1280]`.
- `self_attn.dash_router.q_norm.weight`, `k_norm.weight`, and `head_cls`: `[128]`.
- `self_attn.indexer.q_proj.weight` and `k_proj.weight`: `[128, 1280]`.
- `self_attn.indexer.q_norm.weight` and `k_norm.weight`: `[128]`.

These are 108 BF16 tensors. The remaining addition is `model.vision_model.embeddings.position_ids`, I64 `[1, 257]`.

The configured DASH inference path uses the DASH projections/norms/head_cls. The indexer object also supplies page/position helpers, but its separate learned projections are not used by that path. Classify preserved or omitted tensors explicitly rather than silently discarding them. The position-ID buffer is deterministic; validate it and either regenerate it or record its omission instead of passing it through BF16 conversion.

### Sparse attention math to reproduce

The authoritative path is `configure_dash_inference` plus `_infer_prefill_attend` / `_dash_decode_forward`, not a generic call to the training `forward` method.

1. Project each hidden state to 128-dimensional router Q/K and apply Gemma-style RMSNorm (`1 + weight`, unlike the decoder's regular RMSNorm).
2. Apply page-aware RoPE to router Q/K as well as main attention Q/K.
3. Partition keys into 32-token blocks. The learned `head_cls` produces a softmax-weighted router-key summary per block.
4. Score remote blocks and apply the page-aware image candidate mask. Preserve upstream fallback when a query has no allowed candidates.
5. Select the top 16 scores after finite candidate masking for entmax-1.5, then retain up to 12 remote blocks. Reapply candidacy after selection: the -10000 masked sentinel can outrank extremely negative valid scores. The per-layer configured scale is 128 and prior floor is 0.0001.
6. Convert priors into centered log-prior attention biases. Attend to selected blocks plus the causal local 128-token window. Local entries have zero bias and take precedence on overlap.

The read set is at most 12 * 32 + 128 = 512 tokens per query, before accounting for overlap or fewer valid blocks. Routing and candidate-state management still have their own cost; bounded reads do not alone prove bounded total runtime or storage.

The native primitive follows HF `_infer_route_select` and the wheel's Torch `DashRouterMath.route_and_select`: compute entmax over top-M, then center log priors over valid selected slots. Equal scores use earlier-block-first ordering natively; Torch `topk` does not promise stable tie ordering.

The pinned wheel's optimized `route_triton.py::_entmax_kernel` differs: it bisects an entmax threshold over the full candidate domain, caps the selected support separately, and packs blocks in index order. Its `M` is an output width, not a pre-entmax top-M truncation. Saturated support or ties can therefore differ from HF. Keep HF as the single-window math oracle and use the wheel for continuation-state semantics; do not assume exact scheduler text parity without accounting for this difference.

### Position semantics

The config advertises theta 500000 and YaRN factor 4 for the 1D path, but normal page-marked OCR uses custom `page2d` RoPE:

- 63 frequency pairs for within-page token offsets, theta 10000, resampled frequencies.
- 1 frequency pair for page position, page theta 256, page coordinate modulo 2.
- Page 0 is the prompt/instruction bucket; image and output tokens for a page share a page bucket, with output offsets continuing after its prefill tokens.
- Page transitions update per-sequence state when `<PAGE>` is generated.

Simply replacing `UOCR_ROPE_THETA` or increasing `UOCR_MAX_POSITIONS` is incorrect. Implement the configured fallback too if text-only or custom markerless prompts are supported; otherwise reject unsupported request modes explicitly.

## Required changes by subsystem

### Model identity, conversion, and loading

Files: `src/unlimitedocr_c/api.py`, `convert.py`, `tensor_registry.py`, `frontend.py`; `src/model/uocr_format.h`, `uocr_constants.h`, `uocr_tensor_registry.h`, `uocr_model_file.c`.

- Implemented: explicit `model="v1"|"v2"` selection and source resolution. Both checkpoints report `model_type=unlimited-ocr`; conversion distinguishes them by the validated sparse-attention contract. Native v2 generation remains guarded until decoder support is ready.
- Implemented: v2 uses `.uocr` file version 2, retaining the common version-1 subrecords and appending a 60-byte `uocr_v2_config_record` to the config section. It records variant, page token, sparse dimensions/budgets, page RoPE, crop cap, and router scale/floor/sigma. The native loader validates the supported contract; engine/Metal inference currently rejects v2 explicitly.
- Implemented: merge both shard headers and retain per-tensor source-file/offset information. Conversion validates variant-specific counts, sizes, config, and tokenizer. V2's absent `processor_config.json` is replaced by the pinned preprocessing contract, not disabled validation.
- Implemented: new families `DASH_ROUTER=15` and `SPARSE_INDEXER=16`, with per-layer local IDs 60-64 and 70-73. Existing IDs remain unchanged. Deterministic `position_ids` gets fixed ID 299999, outside the sorted CLIP numbering, and is omitted only after validating its I64 payload.
- Implemented: converted files live under `<cache>/<variant>/<repo-revision-hash>/<qprofile>/<filename>`. Default v1 also checks legacy locations; v2 and custom source/revision selections do not. Tokenizers come directly from repository/revision-specific HF snapshots. The official v2 source and tokenizer default to the pinned revision. Known wrong-variant files/tokenizers are rejected, and reconversion failures leave the old model intact.
- Keep the new router tensors unquantized initially. Reuse existing matrix kernels and quantization for unchanged families only after parity checks.

### Frontend and page representation

Files: `src/unlimitedocr_c/frontend.py`, `api.py`, `ffi.py`, `ocr.py`; `include/unlimitedocr.h`; `src/runtime/uocr_request_validation.c`, `uocr_sequence.[ch]`, `uocr_vision.c`; `src/core/uocr_api.c`; Metal prompt assembly and embedding paths.

- Load the v2 tokenizer: added tokens are `<PAGE>` 128827, `<BLOCK>` 128828, `<FORK>` 128829, `<CHILD>` 128830. The BPE vocabulary and other tokenizer components are unchanged in the comparison.
- Match the default prompt: `"<PAGE><image>" * n_pages + "Multi page parsing."`, including single-page default inference.
- Preserve page markers in decoded output. Match the crop limit of 16 and upstream image preparation; base multi-page input pads original images directly to 1024.
- Support image spans separated by text markers. Both request validation and sequence construction currently reject noncontiguous image placeholders; Metal prompt assembly also assumes a single contiguous copy.
- Use a per-span mapping from prompt positions to visual feature rows, or a mask-driven scatter. Keep `<PAGE>` embeddings intact and assign page-local positional metadata.
- Add a user-facing multi-page method; the existing low-level `prepare_pages` is not exposed through the single-image `generate` API. Start with base-mode pages. Multi-page tiling additionally needs per-page crop grids instead of the current single request-wide grid.

### Runtime and Metal

Files: `src/backend/metal/uocr_metal.m`, `kernels/rope.metal`, `attention_prefill.metal`, `attention_decode.metal`, `kv_cache.metal`, new router kernels; `src/backend/cpu_ref/`; `src/runtime/uocr_memory.[ch]`, `uocr_sequence.[ch]`, `uocr_logits_processor.[ch]`; `src/core/uocr_api.c`.

- Use small Python reference calculations to check the new math; do not build a second C inference path. Reuse SAM/CLIP/projector/MoE and main projection kernels.
- Dispatch prefill and decode attention by model variant. New scratch/cache data includes router keys or summaries, block/page metadata, selected indices/biases, and per-page position counters.
- Redesign v2 KV retention separately from v1's `prompt + 128` layout. All state belongs to an engine/request; do not copy upstream's process-global inference switch into the thread-safe native API.
- Make positional and allocation limits model-specific and enforce the memory budget before allocating.
- Implement upstream page-count guarding before greedy argmax: suppress premature EOS and force a page marker or EOS at the configured per-page output cap (default 20000). Match optional repetition processing separately; upstream native inference defaults it off.

## Memory caveat and scope boundary

The HF `_uocr_generate` reference explicitly disables the ring window and uses `DynamicCache`; `_dash_decode_forward` appends router keys and main K/V. Its reference attention also uses dense masked math in some paths. It is a correctness oracle, not evidence of constant-memory native inference.

At this architecture's FP16 KV size (61,440 bytes/token), retaining 131072 tokens costs 7.5 GiB for main K/V alone. Increasing the position cap without a retention design would be unsuitable for many target Macs.

The customized SGLang path uses page-window scheduling and KV-tail reuse, specified below. Do not discard generated tokens older than 128 within an active window: the reference router has fallback candidate behavior. Reclaim old-window data only at the verified continuation boundary.

## Verified upstream scheduler contract

Source: `wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl` at the pinned HF revision. SHA256: `576c71dab3cacad843f6870739494e13b2e8169a69cc038f84395d6503e801ea`.

Relevant files inside the wheel:

- `sglang/srt/entrypoints/openai/doc_split.py`: contiguous windows and output budgets.
- `sglang/srt/entrypoints/openai/serving_chat.py`: lookahead and streaming output ownership.
- `sglang/srt/session/uocr_doc_reuse.py`: retained ranges, rewritten prompts, copied text, and page state.
- `sglang/srt/sampling/custom_logit_processor.py`: page/window guard.
- `sglang/srt/mem_cache/memory_pool.py`: move main K/V and router K together.

### Window layout and ownership

- Own consecutive pages with zero overlap. Default wheel settings are one owned page and one lookahead page; the model card's launch example overrides owned pages to ten. Lookahead is omitted at document end.
- Only transcribe owned pages. The lookahead lets the final image block see the next page's opening. Cache its encoded/prefilled representation for the next window.
- Keep a cumulative document output budget separate from the per-window budget. Upstream caps window output at `own_pages * max_page_tokens + 256`, further constrained by remaining context. A length-stopped incomplete page is not a valid continuation checkpoint.

### Exactly what survives a completed window

For the last owned image page, let E be the position just after its image tokens. KV allocation and sparse routing both use 32-token blocks.

1. Keep that page's `<PAGE>` marker K/V.
2. Keep image tokens `[E - 128 - E % 32, E)`: 128 to 159 tokens to preserve allocation alignment.
3. Keep the first lookahead page's marker and image K/V, if present.
4. Keep at most 128 trailing generated tokens from the last owned page, stopping after its last `<PAGE>` marker. Copy only tokens whose K/V has actually been computed; exclude EOS and the marker beyond the window's owned pages.
5. Free the remaining allocation blocks. Physical retained spans round outward to 32-token boundaries.

### Next-window execution

- Rewrite the prompt as `[previous-page marker | image tail | next pages including lookahead]`; drop BOS and the instruction. The first window already provided the instruction.
- Copy main K/V and router K for the retained image prefix and reused lookahead, then prefill uncached image tokens. Upstream leaves one last image token to prefill if otherwise the new prompt would be entirely cached.
- Append the previous page's text-tail K/V after the new prompt, then force a fresh `<PAGE>` decode input. Ignore the sampled prefill token. Do not re-emit the copied text or count it against the user's new-output budget. The output formatter supplies the owned page's marker once.
- Rebuild image-page eligibility, marker counters, and per-page token-position counters; recompute `head_cls` summaries for relocated blocks. Relocating only main K/V is incorrect.
- Set the rewritten prompt's page base to `first_owned_global_page - 1`. Recomputed windows use `first_owned_global_page`. Local page IDs plus this base preserve global page parity; copied keys are not re-rotated.
- Stop a lookahead/reused window at the marker beyond its owned pages, force EOS, and remove that stop marker from output and the retained text. The page-length cap can also force a turn/EOS. Do not confuse the window's EOS with document completion.
- Reuse only a successfully finished, contiguous predecessor with usable cached text. Upstream can fall back to recomputation for missing state or allocation failures. Our initial implementation should report an incomplete document/failure rather than silently discard continuation context when this contract cannot be met.

Upstream explicitly describes copy-splice reuse as approximate, validated by OCR accuracy. Matching its scheduler does not imply bitwise or semantic equivalence to full-cache HF inference. Cross-window OCR and bounded-memory behavior still require actual runtime verification.

### Native implementation boundary

Use one request-owned session: window buffers plus a retained-tail record (source slots, image/router K/V, text token IDs, global next-page index, and page counters). Consume at most one window plus one lookahead from the input iterable. Skip SGLang's server queues, cross-request caches, CUDA graphs, and asynchronous prefetch; they are not needed for the native single-session path. Validate capacity for the active window and source/destination tail copies before starting, and fail cleanly if that cannot fit.

