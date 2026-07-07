# Grounded Q8 implementation plan for vision encoders in `unlimitedocr.c`

This plan extends the completed decoder/text Q8 work to the **Metal vision path**:
SAM image encoder, CLIP vision transformer, and the visual projector.  It keeps
all runtime activations fp16, keeps the public API unchanged, and uses the same
module-by-module QA discipline as `docs/plan_q8.md`.

The first vision deliverable should **not** attempt a pure-Q8 model.  It should
produce a mixed fp16/Q8_0 `.uocr` model where decoder/text Q8 remains enabled
and selected vision/projector weights become Q8 only after their fused Metal
paths pass end-to-end QA.

## QA / rollout discipline

Vision Q8 support must roll out **module by module**, not as one large switch.
Each module lands behind the quantization config, is converted into a real
`.uocr`, and is validated end-to-end before the next vision module is enabled.

The workflow for every vision Q8 module is:

1. Implement the fused Q8 Metal kernel(s) for that exact call site.
2. Flip `supported: true` for that module in the quantization config.
3. Re-convert with `--qprofile mixed-q8_0` or `UnlimitedOCR(quant="q8", force_reconvert=True)`.
4. QA test representative OCR inputs against the fp16 baseline.
5. Only after QA passes, move on to the next vision module.

`configs/quant-cfg.yaml` remains the source of truth for what is runtime-safe to
quantize.  Vision modules should be added to it as explicit module entries, all
starting with `supported: false`.  A module never becomes `supported: true`
until its fused kernels are wired and a real converted model passes QA.

## Current code facts this plan must preserve

* Q8 format, qprofile, tensor-entry metadata, model-view span planning, and
  qweight/qscale mapping already exist for `UOCR_TENSOR_Q8_0`.
* Decoder/text Q8 is implemented and QA'd for token embedding, attention QKV/O,
  dense/shared MLP, routed MoE, and LM head.
* The current converter policy is `embeddings+decoder`; vision tensors are
  currently preserved fp16.
* Vision tensor families are `TensorFamily.VISION_SAM`,
  `TensorFamily.VISION_CLIP`, and `TensorFamily.PROJECTOR` in
  `src/unlimitedocr_c/tensor_registry.py` and C mirrors.
* Current vision tensor registry entries use `TensorProjection.NONE` for most
  SAM/CLIP tensors, which is too coarse for safe cfg-gated Q8 selection.
* Metal vision binding currently requires fp16:
  `metal_validate_vision_tensor_metadata()` rejects any non-`UOCR_TENSOR_F16`
  qtype with `production Metal vision requires f16`.
* Vision weight caches are fp16-specific (`uocr_metal_vision_tensor_f16`,
  `uocr_metal_vision_weights_f16`) and many helpers read `host_f16` for
  precomputed position tables or CPU-side conv repacks.
* The vision path uses a mix of custom Metal kernels and MPS matmuls:
  * SAM patch embedding uses MPS matmul after patch extraction.
  * SAM neck 1x1 and 3x3 conv paths use MPS matmul / im2col and CPU-side
    repacked fp16 weights.
  * SAM transformer QKV/O and MLP linears use MPS matmuls.
  * CLIP transformer QKV/O and MLP linears use MPS matmuls.
  * The visual projector uses MPS matmul plus bias add.
* MPS matmul cannot consume the existing Q8_0 qweight/qscale representation.
  Q8 paths must therefore be fused custom kernels; they must not dequantize a
  full fp16 copy of weights into global memory.
* Runtime prompt assembly, image newline, view separator, decoder KV/cache, and
  decoder runtime arenas stay fp16.
* Metal source generation still flows through focused kernel fragments and
  regenerated `src/backend/metal/kernels/uocr_smoke.metal`.

---

## 1. Q8 format and vision tensor metadata

### 1.1 Storage format

Reuse the existing `UOCR_TENSOR_Q8_0` representation:

```text
qtype:       UOCR_TENSOR_Q8_0
bits:        8
group_size:  64
scale_type:  fp16
zero_point:  absent
range:       [-127, 127]
quant_axis:  K/input dimension
layout:      row-major [out_features, in_features]
```

For the first vision pass, quantize **rank-2 linear/projector weights only**.
Keep convolution, patch embedding, positional/relative embeddings, norms, and
bias tensors fp16 until dedicated kernels and validation are planned.

Rank-2 Q8 tensors keep the current metadata contract:

```text
payload_offset / payload_size = qweight int8 bytes
scale_offset   / scale_size   = qscale fp16 bytes
block_size                  = 64
row_size                    = cols
logical_shape              = [rows, cols]
physical_shape             = [rows, cols]
flags                      = UOCR_TENSOR_FLAG_ROW_MAJOR
min_offset / min_size       = 0
```

Checklist:

* [ ] Reuse `UOCR_TENSOR_Q8_0`; do not add a new qtype for vision.
* [ ] Keep `UOCR_QPROFILE_MIXED_Q8_0`; do not add a separate vision qprofile.
* [ ] Require `cols % 64 == 0` for first-pass vision Q8 weights.
* [ ] Keep all rank-1 tensors fp16: biases, norm weights, image newline,
      view separator, class embeddings.
* [ ] Keep positional and relative-position embeddings fp16 even when rank-2.
* [ ] Keep rank-4 SAM patch/neck conv tensors fp16 for the first deliverable.

### 1.2 Vision registry roles

Current SAM/CLIP registry entries mostly have `projection=NONE`; that is not
specific enough to safely select Q8 modules.  Before enabling vision Q8, add
structured roles/projections for vision weights.

Recommended projection additions:

```text
VISION_ATTN_QKV
VISION_ATTN_O
VISION_MLP_FC1
VISION_MLP_FC2
VISION_PATCH_EMBED
VISION_CONV_1X1
VISION_CONV_3X3
PROJECTOR_WEIGHT  (or reuse WEIGHT for TensorFamily.PROJECTOR)
```

For first-pass rank-2 Q8, only these should be eligible:

```text
PROJECTOR / WEIGHT
VISION_CLIP / VISION_ATTN_QKV
VISION_CLIP / VISION_ATTN_O
VISION_CLIP / VISION_MLP_FC1
VISION_CLIP / VISION_MLP_FC2
VISION_SAM  / VISION_ATTN_QKV
VISION_SAM  / VISION_ATTN_O
VISION_SAM  / VISION_MLP_FC1
VISION_SAM  / VISION_MLP_FC2
```

Checklist:

* [x] Extend `TensorProjection` in Python and C mirrors with vision-specific
      projections, or an equivalent stable role field.
* [x] Map SAM block tensors:
      attention qkv weight, attention projection weight, MLP lin1/lin2 weights.
* [x] Map CLIP block tensors:
      attention qkv weight, attention output weight, MLP fc1/fc2 weights.
* [x] Keep SAM/CLIP norms, biases, position embeddings, relative position
      tables, class embedding, and conv/patch weights out of first-pass Q8.
* [x] Add tests that registry roles match the expected tensor names and shapes.

### 1.3 Validator policy

The model-file Q8 validator already supports rank-2 Q8 tensors globally.  The
Metal vision binding layer must independently reject Q8 for unsupported vision
roles until kernels exist.

Checklist:

* [ ] Keep global Q8 tensor-entry validation unchanged for rank-2 weights.
* [ ] Update Metal vision metadata validation to allow Q8 only for cfg-enabled
      rank-2 vision/projector roles.
* [ ] Preserve strict fp16 validation for norms, biases, position embeddings,
      relative-position tensors, image newline, view separator, and first-pass
      rank-4 conv/patch weights.
* [ ] Add negative tests for malformed vision Q8 metadata: wrong block size,
      wrong scale size, unsupported role, rank-1 Q8, and rank-4 Q8 before conv
      support lands.

---

## 2. Converter and quantization config

### 2.1 CLI and policy

Extend the existing converter instead of adding a new tool or public API.

Recommended usage:

```bash
uv run tools/uocr-convert \
  /path/to/hf/snapshot \
  --qprofile mixed-q8_0 \
  --quant-policy embeddings+decoder+vision \
  --quant-cfg configs/quant-cfg.yaml \
  --out unlimitedocr-mixed-q8_0-vision.uocr \
  --overwrite \
  --dump-quant-summary
```

Checklist:

* [ ] Extend `--quant-policy` choices to include `embeddings+decoder+vision`.
* [ ] Keep `--qprofile mixed-q8_0` as the switch; do not add `--quant` aliases.
* [ ] Keep the existing `UnlimitedOCR(quant="q8")` API; users force vision-Q8
      reconversion with `force_reconvert=True` after cfg changes.
* [ ] Bump converter version, e.g. to `(0, 3, 0)`, once vision Q8 planning/writing lands.
* [ ] Extend quant summary JSON with vision-family qweight/qscale bytes and
      fp16-equivalent savings.

### 2.2 Quantization config modules

Add vision modules to `configs/quant-cfg.yaml`, initially disabled:

```yaml
  - name: visual_projector
    family: PROJECTOR
    projections: [WEIGHT]
    supported: false

  - name: clip_mlp
    family: VISION_CLIP
    projections: [VISION_MLP_FC1, VISION_MLP_FC2]
    supported: false

  - name: clip_attention
    family: VISION_CLIP
    projections: [VISION_ATTN_QKV, VISION_ATTN_O]
    supported: false

  - name: sam_mlp
    family: VISION_SAM
    projections: [VISION_MLP_FC1, VISION_MLP_FC2]
    supported: false

  - name: sam_attention
    family: VISION_SAM
    projections: [VISION_ATTN_QKV, VISION_ATTN_O]
    supported: false
```

Later optional modules, also disabled by default:

```yaml
  - name: sam_patch_embed
    family: VISION_SAM
    projections: [VISION_PATCH_EMBED]
    supported: false

  - name: sam_neck_convs
    family: VISION_SAM
    projections: [VISION_CONV_1X1, VISION_CONV_3X3]
    supported: false
```

Checklist:

* [x] Add disabled vision module entries only after registry roles exist.
* [x] Ensure default config stays monotonic: a module only flips to `true` after
      fused kernels and QA.
* [x] Add converter dry-run tests showing vision modules remain fp16 while
      disabled and become Q8 only when their module is enabled.

### 2.3 Tensor selection

First-pass Q8 candidates:

* visual projector: `model.projector.layers.weight` `[1280, 2048]`
* CLIP transformer linears:
  * attention qkv weight `[3072, 1024]`
  * attention output weight `[1024, 1024]`
  * MLP fc1 weight `[4096, 1024]`
  * MLP fc2 weight `[1024, 4096]`
* SAM transformer linears:
  * attention qkv weight `[2304, 768]`
  * attention projection weight `[768, 768]`
  * MLP lin1 weight `[3072, 768]`
  * MLP lin2 weight `[768, 3072]`

Keep fp16:

* SAM patch embedding and all neck/net conv weights for the first deliverable.
* CLIP class and position embeddings.
* SAM absolute/relative position embeddings.
* Norm weights and all biases.
* Image newline and view separator.

Checklist:

* [ ] Implement selection through registry roles, not name globs in hot paths.
* [ ] Reject Q8 for shapes whose input dimension is not divisible by 64.
* [ ] Preserve existing tensor order and file layout guarantees.
* [ ] Add tests for each module's tensor count and bytes.

### 2.4 Quantization implementation

Reuse the existing BF16→Q8_0 streaming writer for rank-2 weights.

Checklist:

* [ ] Reuse row-major Q8 writer for rank-2 vision/projector weights.
* [ ] Keep BF16→FP16 path for unsupported or disabled vision tensors.
* [ ] Add dequantized compare support for selected vision tensors.
* [ ] Add summary sections grouped by `PROJECTOR`, `VISION_CLIP`, and
      `VISION_SAM`.
* [ ] Do not quantize rank-4 conv/patch tensors until a flattening and kernel
      contract is explicitly implemented.

---

## 3. Loader and Metal vision weight views

### 3.1 Vision binding cache

Current vision binding structures store only fp16 payloads.  Replace or extend
these with dtype-aware views, mirroring the decoder binding approach.

Suggested structure:

```c
typedef struct uocr_metal_vision_weight_view {
    uint32_t tensor_id;
    uint32_t qtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
    uint32_t groups_per_row;

    id<MTLBuffer> data_buffer;     /* fp16 payload for F16; qweight for Q8_0 */
    NSUInteger data_offset;
    uint64_t data_size;

    id<MTLBuffer> scale_buffer;    /* Q8 only */
    NSUInteger scale_offset;
    uint64_t scale_size;

    const uint16_t *host_f16;      /* only valid for fp16 tensors that need CPU precompute */
} uocr_metal_vision_weight_view;
```

Checklist:

* [x] Store qtype, rows, cols, group_size, groups_per_row for vision weights.
* [x] Resolve both qweight and qscale buffers for Q8 vision tensors.
* [x] Keep `host_f16` only for fp16 tensors; never expose Q8 as fp16 host data.
* [x] Keep precomputed SAM/CLIP position tables fp16.
* [x] Keep CPU-side conv repack code fp16 until conv Q8 support lands.

### 3.2 Validation and fallback

Vision modules must dispatch by qtype at each call site.  If a tensor is Q8 and
there is no Q8 kernel, fail early during binding/validation rather than silently
falling back through MPS.

Checklist:

* [x] Add `metal_vision_tensor_allows_q8(family, projection, rank, dims)`.
* [x] Require fp16 for disabled/unsupported roles.
* [x] Fail with clear errors when a Q8 tensor reaches an fp16-only path.
* [x] Keep all existing fp16 vision tests passing unchanged.

---

## 4. Metal kernels and integration

Performance rule: every Q8 vision compute path must be fused.  A Q8 path reads
int8 qweights and fp16 scales, dequantizes inside the same kernel that performs
the dot product/projection, accumulates in fp32, and writes the existing fp16
activation output.  It must not materialize a dequantized fp16 weight tensor on
CPU or GPU and must not run MPS matmul on dequantized global-memory weights.

### 4.1 Build/source integration

Checklist:

* [ ] Add focused fragments, not a monolithic file:
  * [x] `projector_q8.metal`
  * [x] `clip_q8.metal`
  * [ ] `sam_q8.metal`
  * optional later: `sam_conv_q8.metal`
* [x] Add fragments to `tools/gen_metal.py` near the matching fp16 files.
* [x] Regenerate `src/backend/metal/kernels/uocr_smoke.metal`.
* [x] Verify both source-compiled and precompiled Metal builds.

### 4.2 Visual projector Q8

Current path: `metal_context_visual_projector_f16_to_slice()` runs MPS matmul
`[rows,2048] @ [1280,2048]^T`, then bias add.

Checklist:

* [x] Add fused Q8 projector kernel: fp16 input `[rows,2048]`, Q8 weight
      `[1280,2048]`, fp16 bias `[1280]`, fp16 output `[rows,1280]`.
* [x] Accumulate in fp32, add bias, write fp16.
* [x] Dispatch Q8 projector when `projector_weight.qtype == UOCR_TENSOR_Q8_0`.
* [x] Keep projector bias fp16.
* [x] Enable `visual_projector.supported: true`; end-to-end QA passed.

### 4.3 CLIP MLP Q8

Current path: CLIP MLP uses MPS matmuls for fc1/fc2 around GELU.

Checklist:

* [x] Add Q8 fc1 kernel: fp16 hidden → fp16 intermediate with bias and
      QuickGELU (matching the fp16 path activation).
* [x] Add Q8 fc2 kernel: fp16 intermediate → fp16 hidden update, preserving
      residual/add semantics at current call sites.
* [x] Dispatch per-linear when the fc1/fc2 weight is Q8_0.
* [x] Preserve CLIP LayerNorm and biases fp16.
* [x] Enable `clip_mlp.supported: true` for end-to-end QA.

### 4.4 CLIP attention Q8

Current path: CLIP attention uses MPS matmul for fused QKV and output projection;
attention math itself remains custom/fp16/fp32 as currently implemented.

Checklist:

* [ ] Add fused Q8 QKV projection kernel for CLIP: hidden `[tokens,1024]`,
      Q8 qkv weight `[3072,1024]`, fp16 qkv bias `[3072]`, output existing Q/K/V
      scratch layout.
* [ ] Add Q8 output projection kernel: attention context `[tokens,1024]`, Q8
      weight `[1024,1024]`, fp16 bias `[1024]`, residual output fp16.
* [ ] Preserve attention softmax/scaling/head layout unchanged.
* [ ] Dispatch Q8 only when the block's QKV and output weights are Q8_0.
* [ ] Flip `clip_attention.supported: true` only after QA.

### 4.5 SAM MLP Q8

Current path: SAM transformer MLP uses MPS matmul for lin1/lin2 around GELU.

Checklist:

* [ ] Add Q8 SAM lin1 kernel: fp16 hidden `[tokens,768]`, Q8 weight
      `[3072,768]`, fp16 bias, GELU, fp16 intermediate.
* [ ] Add Q8 SAM lin2 kernel: fp16 intermediate `[tokens,3072]`, Q8 weight
      `[768,3072]`, fp16 bias, residual output fp16.
* [ ] Support both global and windowed SAM block token layouts used by current
      kernels.
* [ ] Preserve SAM LayerNorm fp16.
* [ ] Flip `sam_mlp.supported: true` only after QA.

### 4.6 SAM attention Q8

Current path: SAM attention has global and window variants.  QKV/O are MPS
matmuls; packing/windowing, relative-position handling, softmax, and unpacking
are custom kernels.

Checklist:

* [ ] Add Q8 SAM QKV kernels for global and window attention paths.
* [ ] Add Q8 SAM output projection kernels for global and window attention
      paths.
* [ ] Preserve relative-position tables, attention masks/window packing, and
      residual handling unchanged.
* [ ] Dispatch Q8 only when the block's QKV and output weights are Q8_0.
* [ ] Flip `sam_attention.supported: true` only after QA.

### 4.7 Optional later: SAM patch and conv Q8

Rank-4 conv/patch weights need a separate contract.  Do not include them in the
first vision-Q8 deliverable.

Possible contract:

```text
logical_shape / physical_shape = original rank-4 shape
qweight payload flattened by output channel as [out, in * kh * kw]
row_size = in * kh * kw
scale_size = out * ceil(row_size / 64) * sizeof(fp16)
```

Checklist:

* [ ] Extend Q8 tensor validation for selected rank-4 conv/patch tensors.
* [ ] Add converter rank-4 flattening writer.
* [ ] Add Q8 patch embedding kernel, replacing patch extraction + MPS matmul.
* [ ] Add Q8 1x1/3x3 conv kernels or fused im2col-dot kernels.
* [ ] Remove CPU-side fp16 repack requirement for Q8 conv tensors.
* [ ] Flip `sam_patch_embed` / `sam_neck_convs` only after separate QA.

---

## 5. Public API

No public API change is required.

Checklist:

* [ ] Keep `uocr_engine_opts` unchanged.
* [ ] Keep `UnlimitedOCR(quant="q8")` as the high-level entry point.
* [ ] Require `force_reconvert=True` in QA when changing cfg-supported vision
      modules to avoid stale cached `unlimitedocr-q8.uocr` files.
* [ ] Keep qprofile inference from `.uocr`; do not add runtime quantization.

---

## 6. Memory accounting and profiling

Checklist:

* [ ] Extend converter summaries with vision/projector Q8 bytes and savings.
* [ ] Add runtime/profile events for Q8 vision kernels, for example:
  * `metal.vision.projector_q8`
  * `metal.vision.clip_mlp_q8`
  * `metal.vision.clip_attention_q8`
  * `metal.vision.sam_mlp_q8`
  * `metal.vision.sam_attention_q8`
* [ ] Keep existing runtime arena accounting unchanged.
* [ ] Track model-view bytes from actual tensor-data size, including Q8 scales.
* [ ] Compare peak memory and wall-clock profile against the current full-Q8
      decoder baseline.

---

## 7. Implementation order

1. **Registry/config groundwork** — complete
   * add stable vision projections/roles;
   * add disabled vision modules to `configs/quant-cfg.yaml`;
   * add converter dry-run tests.

2. **Vision loader groundwork** — complete
   * add dtype-aware vision weight views;
   * allow Q8 only for selected roles;
   * keep fp16 vision path passing.

3. **Visual projector Q8** — complete (QA'd)
   * implement `projector_q8.metal`;
   * wire dtype-aware dispatch;
   * enable `visual_projector` and QA.

4. **CLIP MLP Q8** — implemented, awaiting QA
   * implement fc1/fc2 Q8 kernels;
   * wire dispatch;
   * enable `clip_mlp` and QA.

5. **CLIP attention Q8**
   * implement QKV/O Q8 kernels;
   * wire dispatch;
   * enable `clip_attention` and QA.

6. **SAM MLP Q8**
   * implement lin1/lin2 Q8 kernels;
   * wire dispatch for global/window block usage;
   * enable `sam_mlp` and QA.

7. **SAM attention Q8**
   * implement global/window QKV/O Q8 kernels;
   * wire dispatch;
   * enable `sam_attention` and QA.

8. **Optional rank-4 conv/patch Q8**
   * only after first-pass rank-2 vision Q8 is stable and profiled.

---

## First vision deliverable

```text
Model:
  mixed fp16/Q8_0 .uocr with qprofile UOCR_QPROFILE_MIXED_Q8_0

Already-QA'd Q8 modules:
  token embedding
  decoder attention Q/K/V/O
  decoder dense/shared MLP
  decoder routed MoE experts
  LM head

New vision-Q8 target modules, QA-gated in order:
  visual projector                      QA'd
  CLIP MLP fc1/fc2                      enabled for user QA
  CLIP attention QKV/O                  pending
  SAM MLP lin1/lin2                     pending
  SAM attention QKV/O                   pending

Still fp16 in first vision deliverable:
  SAM patch embedding and neck/net convs
  SAM/CLIP position and relative-position embeddings
  all norms and biases
  image newline and view separator
  runtime activations, visual feature rows, prompt arena, decoder KV cache

Runtime:
  Metal fused Q8 kernels at each enabled vision call site; no standalone
  dequantization kernels and no dequantized Q8 weight buffers.

QA:
  each enabled module is validated by reconverting with
  UnlimitedOCR(quant="q8", force_reconvert=True) and running representative
  single-image, crop-mode, and multi-page OCR inputs against the fp16 baseline.
```
