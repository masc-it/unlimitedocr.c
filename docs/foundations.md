# Unlimited-OCR C inference engine: foundations

This note captures the starting point for a Metal-first, quantized C inference engine for `baidu/Unlimited-OCR`.

## Source/context reviewed

- Local Python runner: `/Users/mascit/Downloads/ocrbaidu/`
- Hugging Face remote code cached under `data/context/`:
  - `modeling_unlimitedocr.py`
  - `modeling_deepseekv2.py`
  - `deepencoder.py`
  - `configuration_deepseek_v2.py`
  - `config.json`
  - `processor_config.json`
  - tokenizer/special-token configs and safetensors index/header
- Metal reference kernels in `/Users/mascit/projects/gradients.c/`, especially f16 RMSNorm, RoPE, SDPA prefill/decode, matmul/linear, embedding, and KV-cache kernels.

## DS4 reference notes

`data/ds4/` is directly useful as a reference, not just generally inspiring:

- **Tokenizer**: Unlimited-OCR's `tokenizer.json` uses the same JoyAI/DeepSeek byte-level BPE shape DS4 implements in `data/ds4/ds4.c`:
  - BPE vocab size `128000` plus added special tokens up to vocab `129280`.
  - Pre-tokenizer sequence: `\p{N}{1,3}`, CJK/kana isolation, JoyAI regex split, then byte-level encoding.
  - Decoder is byte-level.
  - Special ids include BOS `0`, EOS `1`, PAD `2`, and image token `<image>` id `128815`.
  - We can defer a native C tokenizer because the v1 Python frontend will use HF/tokenizers directly; if we later add a standalone C CLI, adapt DS4's compact tokenizer and load from HF `tokenizer.json` instead of GGUF metadata.
- **Quantizer**: `data/ds4/gguf-tools/quants.[ch]` already provides small C implementations of `q8_0`, `q4_K`, `q2_K`, and `iq2_xxs`, with imatrix-aware weighted quantization.
- **Quant policy/tooling**: `data/ds4/gguf-tools/deepseek4-quantize.c` is a useful template for per-family/per-prefix qtype policy, tensor-name mapping, safetensors reading, dry-run planning, and imatrix provenance metadata.
- **Metal kernels**:
  - `metal/dense.metal`: q8/f16 matvec/matmul and fused shared-expert gate/up SwiGLU.
  - `metal/moe.metal`: routed-expert q4/q2/iq2 kernels, selected-expert dispatch, pair gate/up projection, SwiGLU weighting, and sum-6 down projection.
  - `metal/argsort.metal`, `metal/softmax.metal`, `metal/dsv4_misc.metal`: top-k and router-support patterns.
  - `metal/norm.metal`, `metal/glu.metal`, `metal/get_rows.metal`: RMSNorm, SwiGLU, embedding gather.

Important differences to account for:

- DS4 is DeepSeek V4 specific; Unlimited-OCR has smaller shapes (`hidden=1280`, `experts=64`, `topk=6`, expert intermediate `896`) and a vision stack.
- DS4 router logic is not identical; OCR uses `softmax(hidden @ gate_weight.T)` then greedy top-6 without renormalization/scaling.
- DS4's q2 recipe is useful later, but the initial OCR requirement should stay dynamic q8/q4 with sensitive tensors promoted.

## High-level pipeline

Unlimited-OCR is a vision-language autoregressive decoder:

1. **Host preprocessing**
   - Load image, EXIF transpose, RGB.
   - Normalize pixels with mean/std `(0.5, 0.5, 0.5)`: `x = pixel / 127.5 - 1`.
   - Single image supports:
     - `base`: `base_size=1024`, `image_size=1024`, `crop_mode=false`.
     - `gundam`: `base_size=1024`, `image_size=640`, `crop_mode=true`.
   - In crop mode, large images are resized to a dynamic grid of up to 32 `640x640` local crops; a `1024x1024` padded global view is always included.

2. **Vision encoder**
   - SAM-like ViT-B encoder over each view.
   - CLIP-like ViT-L transformer consumes the SAM downsampled feature map as patch embeddings.
   - Concatenate CLIP token features and SAM feature-map tokens: `1024 + 1024 = 2048`.
   - Linear projector maps `2048 -> 1280`.
   - Add row newline embeddings and a view separator embedding.

3. **Prompt embedding splice**
   - The rendered prompt uses the `plain` conversation template: effectively the user prompt text, with no role prefix; the wrapper prepends BOS id `0` manually after token/image expansion.
   - Prompt construction splits on literal `<image>`, tokenizes text pieces with `add_special_tokens=false`, and inserts id `128815` repeated once per visual feature row/token. These ids are placeholders only.
   - The Python model replaces placeholder embeddings via `masked_scatter_`. Source feature order, not the placeholder-token construction order, determines the final embedding order.
   - We should avoid scatter in C: build final prompt embedding sequence directly from text-token embeddings and vision features.

4. **Text decoder / generation**
   - DeepSeekV2-derived decoder, but configured as standard Llama-style MHA (`use_mla=false`).
   - 12 layers, hidden `1280`, 10 heads, head dim `128`.
   - Layer 0 has dense SwiGLU MLP; layers 1-11 are MoE.
   - Greedy generation by default, with optional sliding-window no-repeat n-gram processor.
   - The no-repeat processor bans any token that would recreate an `ngram_size`-gram within the last `window` tokens (`35/128` single image, `35/1024` multi-page in the wrapper).
   - Stop on EOS id `1` (`<｜end▁of▁sentence｜>`).

## Model shape summary

From `config.json` and safetensors header:

- Total parameters: `3,336,106,240` BF16 params (`6.672 GB`).
- Vocab: `129,280`.
- Hidden size: `1280`.
- Max positions: `32768`.
- Decoder layers: `12`.
- Attention: MHA, `10` Q heads, `10` KV heads, head dim `128`, RoPE theta `10000`.
- Sliding window: `128` generated tokens, but prompt/prefill remains fully attendable.
- MoE: layers 1-11, `64` routed experts, top-`6`, expert intermediate `896`, plus `2` shared experts (`1792` intermediate).

Parameter breakdown:

| Component | Params | BF16 size |
|---|---:|---:|
| Routed MoE experts | 2.422B | 4.844 GB |
| CLIP-like vision transformer | 303.2M | 0.606 GB |
| Token embedding | 165.5M | 0.331 GB |
| LM head | 165.5M | 0.331 GB |
| SAM-like vision encoder | 95.6M | 0.191 GB |
| Decoder attention | 78.6M | 0.157 GB |
| MoE shared experts | 75.7M | 0.151 GB |
| Dense layer-0 MLP | 26.3M | 0.053 GB |
| Projector | 2.6M | 0.005 GB |
| Gates/norms/special embeddings | <1M | small |

## Visual token counts

The feature grid is downsampled by `patch_size=16` then `downsample_ratio=4`:

- `1024` view -> `16x16` feature grid -> `(16 + newline) * 16 + separator = 273` tokens.
- `640` local crop -> `10x10` feature grid.
- `base` single image / each multi-page image: `273` visual tokens per page.
- `gundam` crop mode:
  - local grid `W x H` crops -> `(10*W + 1) * (10*H)` local tokens,
  - plus `272` global row/newline tokens,
  - plus `1` separator.

Implementation detail: in Python, the actual feature order inserted in crop mode is **local features first, then global features, then separator**. The repeated image token ids are identical, so the feature order is what matters.

## Core operations to implement

### Host/runtime foundations

- Model file converter:
  - Read HF safetensors BF16.
  - Emit a custom `.uocr`/GGUF-like file with tensor metadata, alignment, quant type, scales, and optional tokenizer metadata/payload.
  - Precompute fixed positional interpolations for common sizes:
    - SAM abs pos: `64x64` and `40x40`.
    - CLIP abs pos: `16x16` and `10x10`.
    - SAM global rel-pos for `40x40` if we want to avoid runtime interpolation.
- Python frontend:
  - Owns image loading, EXIF transpose, RGB conversion, PIL-compatible resizing/padding/crop planning, normalization, prompt rendering, HF tokenization, and output token decoding.
  - Passes prepared requests to C: `input_ids`, `image_mask`, crop/grid metadata, and contiguous preprocessed image views.
- C inference core:
  - Does not need a native tokenizer or image loader in v1.
  - Validates prepared requests, runs vision encoder, assembles prompt embeddings, performs decoder prefill/decode, applies token-id logits processors, and returns generated token ids.
- Inference scheduler:
  - Single prepared request first.
  - Batched prepared requests need per-sequence prompt lengths, KV cache offsets, and per-step active sequence tracking.

### Metal kernels needed

Can reuse/adapt patterns from `gradients.c`:

- Quantized linear/matmul (`int8`, `int4`) with fp16 activations and fp32 accumulation where needed.
- Embedding lookup/dequant.
- RMSNorm and LayerNorm.
- RoPE for Llama-style Q/K.
- SDPA prefill and decode:
  - Causal prefill over full prompt.
  - Decode attends to full prompt + generated ring window.
- KV cache append/ring update.
- SwiGLU/SiLU fused MLP path.
- MoE gate softmax + top-6.
- MoE token dispatch/group-by-expert and weighted combine.
- Vision conv2d kernels:
  - SAM patch embed `16x16 stride 16`.
  - SAM neck/net convs: `1x1`, `3x3`, stride-2 convs.
- Vision attention:
  - SAM window attention (`14x14` windows) + global attention blocks with decomposed relative position bias.
  - CLIP full attention over `257` or `101` tokens.
- GELU and QuickGELU.
- LM head matvec + argmax/logits processors.

## Dynamic mixed quantization strategy

We should use a dynamic/selective quantization scheme similar in spirit to Unsloth dynamic quants: most large weights are compressed, but numerically sensitive tensors stay at higher precision. "Dynamic" here means **per-tensor / per-module qtype selection at conversion time**, driven by heuristics and calibration, not one fixed global q4/q8 policy.

Recommended staged approach:

1. **Correctness baseline**: fp16 runtime weights/activations, loaded from BF16 converted to fp16.
2. **Dynamic int8 profile**:
   - Weight-only q8 for large linear/conv tensors.
   - Keep all required-sensitive tensors fp16.
   - Use this as the first compressed parity target.
3. **Dynamic int4 profile**:
   - Default large low-risk matrices to q4 group-wise.
   - Promote sensitive tensors/layers to q8 or fp16 based on calibration.
   - Prioritize routed MoE experts because they dominate memory.
4. **Calibration/tuning loop**:
   - Run representative OCR images/prompts through fp16 and candidate quantized models.
   - Compare module outputs, next-token logits, generated text, and layout boxes.
   - Escalate tensors from q4 -> q8 -> fp16 when error crosses thresholds.

Initial mandatory fp16 keep-list:

- MoE router weights: `model.layers.*.mlp.gate.weight` (not expert `gate_proj`).
  - Router logits/top-k must be computed in fp32/fp16 to avoid expert-selection drift.
- Norm weights and norm math: RMSNorm/LayerNorm weights, fp32 variance.
- Biases, positional embeddings, relative position tables, RoPE/cos/sin tables.
- Image newline/view separator embeddings.
- Small tensors where quantization overhead exceeds savings.

Initial conservative qtype policy:

| Tensor family | Initial qtype | Notes |
|---|---:|---|
| MoE router weight | fp16 | mandatory; top-k stability |
| Norms, biases, position tables | fp16 | mandatory |
| Token embeddings | q8 or fp16 | start q8; q4 only after parity |
| LM head | q8 | generation-sensitive; q4 later if validated |
| Decoder attention projections | q4/q8 dynamic | promote q/k/v if logits drift |
| Dense layer-0 MLP | q4/q8 dynamic | `gate/up` are Q4_K-aligned; `down` input dim `6848` needs q8 or padded/custom q4 |
| MoE shared experts | q8 first | always active; q4 after calibration; shared `down` input dim `1792` is Q4_K-aligned |
| MoE routed experts | q4/q8 dynamic | `gate/up` are Q4_K-aligned; routed `down` input dim `896` needs q8 or padded/custom q4 |
| Vision conv/linear weights | q8 first | vision parity is critical; q4 later selectively |
| Vision attention/MLP weights | q8/q4 dynamic | promote layers with feature drift |
| Projector `2048 -> 1280` | q8/fp16 | small, OCR-sensitive |

Quant format foundations:

- Reuse DS4/GGML-compatible block formats where possible:
  - `Q8_0` for q8 baseline / sensitive large matrices; block size `32`, compatible with OCR's important inner dimensions.
  - `Q4_K` for q4 where the matrix inner dimension is a multiple of `256`.
  - Keep `Q2_K` / `IQ2_XXS` as future experiments, mainly for routed experts after q4 is solid.
- Important q4 alignment caveat: OCR has key inner dimensions that are **not** Q4_K-aligned, especially routed expert `down_proj` input `896` and dense layer-0 `down_proj` input `6848`. Initial dynamic-q4 should either keep these tensors `Q8_0` or write a padded-Q4_K/custom-q4 format with explicit logical vs physical input width.
- Use fused dequant-matmul/conv kernels; never fully dequantize large tensors at runtime.
- Conv weights are flattened to `[out_channels, in_channels * kh * kw]` for quantization.
- The model file stores qtype per tensor plus logical shape, physical packed shape, scale offsets, block size, optional imatrix/calibration metadata, and promotion reason.
- The DS4 quantizer structure is the preferred starting point: policy by tensor family/prefix, dry-run plan, single-tensor compare, and optional imatrix input.

MoE-specific dynamic quantization:

- Keep the MoE router weight fp16, compute softmax/top-6 in fp32.
- Quantize expert `gate_proj`, `up_proj`, `down_proj` independently.
- Calibration should track expert frequency and output error per expert.
- Promote high-impact or high-error experts/projections to q8/fp16 while leaving low-risk experts q4.

Approximate memory before scales if every large tensor used one type:

- BF16/fp16: `6.67 GB`.
- Full int8 weight-only: `~3.34 GB` + scales.
- Full int4 weight-only: `~1.67 GB` + scales.
- Dynamic q4 will land between q4 and q8 depending on promoted tensors.

## Decoder details that matter for parity

- RMSNorm computes variance in fp32, then casts back.
- Llama attention uses RoPE over full head dim `128`.
- Attention mask:
  - Prefill: causal mask over full prompt.
  - Decode: no causal mask needed for `q_len=1`.
- Sliding/ring behavior:
  - Python disables HF cache truncation during prefill.
  - Records prefill length per layer.
  - Generated tokens append until `prefill_len + 128`; then overwrite a 128-token ring.
  - Attention always includes all prompt/prefill tokens plus the ring window of recent generated tokens.
- MoE gate:
  - `softmax(hidden @ gate_weight.T)` in fp32.
  - `topk=6`, `sorted=False` in PyTorch; order does not matter for summed output.
  - Selected weights are not renormalized (`norm_topk_prob=false`).

## Milestones

1. **Python frontend + reference extraction**
   - Add Python code to produce prepared-request fixtures: token ids, image masks, crop grids, preprocessed pixels.
   - Add Python scripts to dump intermediate tensors for one tiny image run: image features, prompt embeddings, layer outputs, logits.
   - This is essential for C/Metal parity tests.
2. **Model converter + metadata parser**
   - Parse safetensors and emit fp16 `.uocr` first.
   - Add dynamic quantized variants after fp16 works, with qtype stored per tensor.
3. **Prepared C API + CPU reference engine**
   - Implement the low-level prepared-request API.
   - Implement enough fp32/fp16 CPU ops to validate architecture and tensor order.
4. **Metal decoder first**
   - Embeddings, RMSNorm, Q/K/V/O, RoPE, SDPA, dense MLP, MoE, KV cache, LM head.
   - Test with text-only token-id fixtures, then with precomputed image embeddings.
5. **Metal vision encoder**
   - SAM then CLIP then projector.
   - Validate against Python image embeddings from prepared views.
6. **Quantized Metal kernels**
   - Start by adapting DS4 `Q8_0` and `Q4_K` dense/MoE kernels to OCR shapes.
   - q8 first, q4 after.
7. **Batching**
   - Prefix/prompt batching, variable-length prefill, decode scheduler, MoE batching across requests.
8. **Optional native frontend**
   - Native tokenizer/image loading only if we want a standalone C CLI.
9. **CUDA backend**
   - Mirror backend abstractions after Metal path stabilizes.

## Main risks

- Vision parity: SAM relative position bias, window partition padding, and positional interpolation must match closely.
- Image preprocessing differences from Pillow can shift OCR outputs; keeping preprocessing in Python avoids this in v1.
- MoE routing is sensitive; quantizing router weights or using low-precision gate logits may change selected experts.
- Long prompts/multi-page runs make KV cache large because all visual prompt tokens remain attendable.
- Python tokenizer exactness is mandatory for prompt formatting, placeholder counts, EOS handling, and output decoding.
