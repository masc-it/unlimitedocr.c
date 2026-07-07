# Grounded Q8 implementation plan for `unlimitedocr.c`

This plan is corrected against the current tree.  It assumes **pre-quantized `.uocr` model files**, **fp16 activations/KV/runtime arenas**, and **Metal-only Q8 execution** for the decoder/text path while vision/projector/norm/router weights initially remain fp16.

## Current code facts this plan must preserve

* `.uocr` format is defined in `src/model/uocr_format.h` and mirrored in `src/unlimitedocr_c/convert.py`.
* `UOCR_FORMAT_VERSION` is currently `1`; `uocr_model_file_validate_memory()` rejects any header whose `qprofile != UOCR_QPROFILE_FP16`.
* Tensor entries already have quantization-related fields: `qtype`, `block_size`, `row_size`, `scale_offset`, `scale_size`, `min_offset`, `min_size`. Do **not** add a parallel `uocr_tensor_quant_info` directory structure for the first pass.
* Existing tensor qtype values are `UOCR_TENSOR_F16 = 1` and `UOCR_TENSOR_F32 = 2`. Q8 uses `UOCR_TENSOR_Q8_0 = 3`.
* Existing converter entry points are `tools/uocr-convert` and `python -m unlimitedocr_c.convert`; the CLI currently exposes `--qprofile fp16`, not `tools/convert_uocr.py --quant q8`.
* Tensor registry names/families are in `src/unlimitedocr_c/tensor_registry.py` and mirrored by C tensor-id helpers in `src/model/uocr_tensor_registry.h`.
* Metal model mapping currently maps only `payload_offset/payload_size`; Q8 scales at `scale_offset/scale_size` must be added to the span/view/binding logic.
* Integrated decoder bindings currently require `tensor->qtype == UOCR_TENSOR_F16` in `metal_validate_decoder_tensor_metadata()` and expose only one fp16 buffer/offset.
* Decoder execution is not a single generic linear stack:
  * attention Q/K/V prefill uses `metal_run_attention_qkv_buffer_f16()` with MPS matmuls;
  * attention O prefill uses `metal_run_attention_output_residual_buffer_f16()`;
  * decode has custom attention QKV/O kernels;
  * dense/shared MLP gate+up are fused SwiGLU kernels;
  * routed MoE experts use interleaved expert-major slabs and fused selected-expert kernels;
  * LM-head selection has a custom fused argmax path plus an env-selectable MPS path.
* Metal runtime compilation reads the generated combined source `kernels/uocr_smoke.metal`; adding a new `.metal` fragment also requires updating `tools/gen_metal.py` and regenerating/precompiling the combined source.
* Public C options struct is named `uocr_engine_opts`, not `uocr_engine_options`; Python mirrors it as `CEngineOpts`/`EngineOptions` in `src/unlimitedocr_c/ffi.py`.

---

## 1. Q8 format and `.uocr` metadata

### 1.1 Q8_0 storage format

Use symmetric grouped Q8 for selected rank-2 weights:

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

For each output row and each 64-column group:

```text
scale = max(abs(w_group)) / 127
q     = round(w / scale), clamped to [-127, 127]
scale = 1.0 for all-zero groups
```

Represent Q8 with the existing `uocr_tensor_entry` fields:

```text
payload_offset / payload_size = qweight int8 bytes
scale_offset   / scale_size   = qscale fp16 bytes
block_size                  = group_size (64)
row_size                    = qweight bytes per row (= cols)
min_offset / min_size        = 0 for Q8_0
physical_shape              = [rows, cols]
logical_shape               = [rows, cols]
flags                       = UOCR_TENSOR_FLAG_ROW_MAJOR
```

Checklist:

* [x] Add `UOCR_TENSOR_Q8_0 = 3` to `uocr_tensor_qtype` and converter constants.
* [x] Add `UOCR_Q8_GROUP_SIZE_DEFAULT = 64`, `UOCR_Q8_MIN = -127`, `UOCR_Q8_MAX = 127`.
* [x] Add `uocr_tensor_qtype_name()` support for `q8_0`.
* [x] Add converter/runtime helpers for Q8 qweight bytes, qscale bytes, and total bytes.
* [x] Require `cols % group_size == 0` for first-pass Q8 tensors.
* [ ] Keep rank-1 tensors, biases, norms, router weights, projector, and vision tensors fp16.

### 1.2 Qprofile/version policy

The first usable Q8 model is mixed fp16/Q8, not pure Q8, because vision/projector/norm/router tensors remain fp16.

Checklist:

* [x] Add `UOCR_QPROFILE_MIXED_Q8_0 = 2` in `src/model/uocr_format.h` and `convert.py`.
* [x] Update `uocr_qprofile_name()`.
* [x] Keep `UOCR_FORMAT_VERSION = 1`; the existing tensor-entry fields already encode Q8 qweights and qscales, so Q8 is a new qprofile within the current file structure.
* [x] Extend `uocr_model_file_validate_memory()` so it accepts `UOCR_QPROFILE_MIXED_Q8_0` and dispatches per-tensor validation by qtype.
* [x] Preserve the current fp16 validation path unchanged for `UOCR_QPROFILE_FP16`.
* [ ] Use the existing provenance JSON payload for quantization metadata instead of adding binary provenance fields.

Add this provenance JSON payload:

```json
{
  "quantization": {
    "mode": "mixed-q8_0",
    "group_size": 64,
    "policy": "embeddings+decoder",
    "lm_head_qtype": "q8_0",
    "router_qtype": "fp16",
    "converter_version": "0.2.0",
    "source_model_hash": "..."
  }
}
```

### 1.3 Q8 tensor validation

Add a Q8 validator next to `validate_fp16_tensor_entry()` in `src/model/uocr_model_file.c`.

Checklist:

* [x] Validate rank is 2 for Q8_0.
* [x] Validate logical and physical shapes match.
* [x] Validate row-major flag and no transposed flag.
* [x] Validate `block_size == 64`.
* [x] Validate `row_size == cols`.
* [x] Validate `payload_size == rows * cols`.
* [x] Validate `scale_size == rows * (cols / 64) * sizeof(uint16_t)`.
* [x] Validate `payload_offset` and `scale_offset` are 16-byte aligned and inside the tensor-data section.
* [x] Validate `min_offset == 0 && min_size == 0`.

---

## 2. Converter changes

### 2.1 Grounded CLI

Extend the existing converter module and wrapper:

```bash
uv run tools/uocr-convert \
  /path/to/hf/snapshot \
  --qprofile mixed-q8_0 \
  --quant-group-size 64 \
  --quant-policy embeddings+decoder \
  --out unlimitedocr-mixed-q8_0.uocr \
  --overwrite \
  --dump-quant-summary
```

Checklist:

* [x] Extend `--qprofile` choices from `fp16` to `fp16,mixed-q8_0`.
* [x] Do not add a separate `--quant` alias; `--qprofile mixed-q8_0` is the single converter switch for this deliverable.
* [x] Add `--quant-group-size`, default `64`.
* [x] Add `--quant-policy embeddings+decoder` as the only accepted Q8 policy for this deliverable; reject other policy strings with a clear error.
* [x] Quantize LM head unconditionally under `mixed-q8_0`.
* [x] Keep router weights fp16 unconditionally under `mixed-q8_0`.
* [x] Add `--dump-quant-summary` and `--dry-run` behavior consistent with existing planning.
* [x] Bump `CONVERTER_VERSION` to `(0, 2, 0)`.

### 2.2 Tensor selection using the registry

Quantize only runtime rank-2 weights whose `TensorFamily`/`TensorProjection` match the policy.

For `embeddings+decoder`, Q8 candidates are:

* token embedding: `TensorFamily.TOK_EMBED`
* LM head: `TensorFamily.LM_HEAD`
* attention projections: `TensorFamily.LAYER_ATTN`, projections Q/K/V/O
* dense layer-0 MLP: `TensorFamily.LAYER_DENSE_MLP`, projections GATE/UP/DOWN
* routed experts: `TensorFamily.MOE_EXPERT`, projections GATE/UP/DOWN
* shared experts: `TensorFamily.MOE_SHARED`, projections GATE/UP/DOWN

Keep fp16:

* `TensorFamily.MOE_ROUTER`, always fp16 in `mixed-q8_0`
* norms, bias vectors, image newline/view separator, projector, SAM/CLIP vision tensors, tokenizer/provenance/config metadata
* any non-rank-2 tensor

### 2.3 Planning and layout

Current `TensorPlan` has one payload. Q8 needs qweight and qscale metadata.

Checklist:

* [x] Extend `TensorPlan`/`_UocrTensorPayloadView` with `scale_offset`, `scale_size`, and Q8-specific byte counts.
* [x] Update `_TENSOR_ENTRY_STRUCT.pack/unpack` usage; the binary struct already has fields for scales.
* [x] Update `_layout_dry_run_file()` to allocate both qweight and qscale ranges for Q8 tensors in the single tensor-data section.
* [x] Keep all offsets absolute file offsets, matching current fp16 `payload_offset` semantics.
* [x] Preserve existing interleaved expert-major ordering for routed MoE qweight payloads. Store qscales in a matching contiguous expert-major scale slab immediately after the layer's routed-expert qweight slab.

### 2.4 Quantization implementation

The source checkpoint is expected BF16. The current writer streams BF16→FP16; Q8 must convert enough of each selected tensor to fp32 to compute per-group scales.

Checklist:

* [x] Add BF16→fp32 row/chunk conversion helpers.
* [x] Quantize row-major `[out, in]`; do not transpose.
* [x] Compute one fp16 scale per row and 64-column group.
* [x] Write qweight int8 bytes to `payload_offset`.
* [x] Write qscale fp16 bytes to `scale_offset`.
* [x] For fp16 tensors, keep the current BF16→FP16 streaming path.
* [x] Update qtype histograms and family summaries.
* [x] Add comparison/inspection support for Q8 metadata and dequantized numeric comparison against source BF16 with fixed tolerances recorded in tests.

---

## 3. Loader and Metal model views

### 3.1 Model view planning

Current Metal mapping ignores `scale_offset/scale_size`; fix this before any Q8 kernel work.

Checklist:

* [x] Update `payload_tensor_count()`/span planning to count every non-empty tensor byte range: fp16 payloads, Q8 qweights, and Q8 qscales.
* [x] Update `build_payload_spans()` to emit separate spans for qweight and qscale ranges.
* [x] Update binding construction so a Q8 tensor can resolve both qweight and qscale to Metal buffers/offsets.
* [x] Keep `ctx->model_view_bytes = tensor_data->size` and use the tensor-data section size consistently for admission and reports.

### 3.2 Weight-view abstraction

Replace `uocr_metal_decoder_binding` for decoder weights with dtype-aware views.

```c
typedef struct uocr_metal_weight_view {
    uint32_t tensor_id;
    uint32_t qtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
    uint32_t groups_per_row;

    id<MTLBuffer> data_buffer;   /* fp16 payload for F16; qweight for Q8_0 */
    NSUInteger data_offset;
    uint64_t data_size;

    id<MTLBuffer> scale_buffer;  /* Q8 only */
    NSUInteger scale_offset;
    uint64_t scale_size;
} uocr_metal_weight_view;
```

Checklist:

* [x] Store qtype/shape in decoder binding cache.
* [x] Allow fp16 and Q8_0 in `metal_validate_decoder_tensor_metadata()` for selected families.
* [x] Keep strict fp16 requirements for norms and router.
* [x] Update expert slab cache for Q8: routed MoE kernels currently assume one fp16 expert-major slab; Q8 needs qweight slab plus qscale slab and new stride math.

---

## 4. Metal kernels and integration

Do not plan this as only a generic GEMV/GEMM replacement. The current backend relies on fused specialized kernels and MPS paths, so Q8 must be integrated at those call sites.

Performance rule: every Q8 compute path must be fused. A Q8 kernel must read int8 qweights and fp16 scales, dequantize inside the same Metal kernel that performs the embedding lookup, dot product, SwiGLU, expert combine, residual-producing projection, or LM-head argmax, accumulate in fp32, and write only the required fp16/f32 final output. The implementation must not materialize a full fp16 copy of Q8 weights on CPU or GPU, must not run a standalone dequantization kernel before an MPS matmul, and must not store dequantized intermediate weight tiles to global memory.

### 4.1 Build/source integration

Checklist:

* [x] Add Q8 kernel code as focused sibling fragments under `src/backend/metal/kernels/` (for example `embedding_q8.metal`, `lm_head_q8.metal`, `attention_q8.metal`, `dense_q8.metal`, `moe_q8.metal`) instead of a single monolithic `q8.metal`.
* [x] Add each Q8 fragment to `tools/gen_metal.py` `ORDER` near its fp16 counterpart (for example `embedding_q8.metal` immediately after `embedding.metal`).
* [x] Regenerate `src/backend/metal/kernels/uocr_smoke.metal` so runtime source compilation sees the new kernels.
* [x] Precompiled builds are already covered by CMake glob dependencies, but still depend on `gen_metal.py` ordering.

### 4.2 Q8 embedding

Current fp16 embedding path uses `uocr_get_rows_f16_to_f16` over `model.embed_tokens.weight`.

Checklist:

* [x] Add fused `uocr_get_rows_q8_0_to_f16_h1280_g64`; it reads qweight/qscale and writes fp16 token embeddings without a dequantized table.
* [x] Add a dtype-aware embedding wrapper in `uocr_metal.m` for prompt assembly and generated-token embedding.
* [x] Preserve output layout `[tokens, 1280]` fp16 and image-feature splice layout.

### 4.3 Attention projections

Checklist:

* [ ] For prefill Q/K/V, add a fused Q8 QKV kernel and dispatch it from `metal_run_attention_qkv_buffer_*` when all three projection weights are Q8_0.
* [ ] For decode Q/K/V, add fused Q8 variants of the custom decode QKV path.
* [ ] For O projection, add fused Q8 dequant-dot kernels before residual add for both decode and prefill.
* [ ] Preserve RoPE, KV-cache write, flash attention, and residual add unchanged.

### 4.4 Dense/shared MLP

Current gate+up is fused and down projection is a separate MPS/custom path depending on decode/prefill.

Checklist:

* [ ] Add fused Q8 gate+up SwiGLU kernels that read two Q8 weight views, dequantize inside the dot loops, apply SwiGLU, and write fp16 mid.
* [ ] Add fused Q8 down projection kernels for fp16 mid → fp16 hidden; no dequantized down-weight buffer is allowed.
* [ ] Implement a fused Q8 version of the current tiled shared-expert decode kernel so decode keeps the same tile-column scheduling.
* [ ] Route `metal_run_decode_dense_swiglu_one_f16()` and `metal_run_dense_swiglu_buffer_f16()` through dtype-aware fp16/Q8 variants.

### 4.5 Routed MoE experts

Current routed expert path depends on an interleaved fp16 expert-major slab.

Checklist:

* [ ] Define a Q8 expert slab contract: qweights are still `[expert][gate,up,down][row][k]`; scales are `[expert][gate,up,down][row][k_group]`.
* [ ] Extend `metal_expert_interleaved_slab_for_layer()` to return both qweight and qscale slabs/strides.
* [ ] Add fused Q8 selected-expert gate/up and down/combine kernels; dequantization, expert weighting, and combine happen inside the selected-expert kernels.
* [ ] Preserve router/top-k buffers and weighting semantics.

### 4.6 LM head

Current fast path is `uocr_lm_head_argmax_f16`; the env-selectable MPS path writes logits then argmax.

Checklist:

* [ ] Add fused Q8 LM-head argmax kernel for qweight/qscale `[vocab, hidden]`; dequantization, dot product, no-repeat-ngram banning, and per-tile argmax stay in the kernel path without materializing full logits. The kernel writes only the existing partial score/id outputs.
* [ ] Force the custom Q8 LM-head argmax path when LM-head qtype is Q8_0; ignore `UOCR_METAL_LM_HEAD_SELECTION_BACKEND=mps` for Q8_0 and emit a profile/debug note.
* [ ] Preserve no-repeat-ngram banning and partial argmax output flow.

---

## 5. Public API

Runtime quantization is inferred from the model file; no runtime quantization is performed by `uocr_engine_open()`.

Checklist:

* [ ] Do **not** add `quantization` and `quantization_policy` fields to `uocr_engine_opts`.
* [ ] Do not change `include/unlimitedocr.h`, `src/unlimitedocr_c/ffi.py::CEngineOpts`, and `EngineOptions` for Q8 loading.
* [ ] Python high-level loading is:

```python
ocr = UnlimitedOCR(model_path="unlimitedocr-mixed-q8_0.uocr", backend="metal")
```


---

## 6. Memory accounting and profiling

Checklist:

* [ ] Update model-view admission to use the actual mixed-Q8 tensor-data size from the loaded model.
* [ ] Add Q8 count/bytes to converter summary and runtime debug/profile reporting.
* [ ] Keep existing runtime arena, KV cache, prompt embedding, logits, and transient accounting unchanged.
* [ ] Add these profile event names: `metal.decode.attention_qkv_q8`, `metal.prefill.attention_qkv_q8`, `metal.decode.moe_routed_experts_q8`, `metal.decode.lm_head_argmax_q8`.

---

## 7. Implementation order

1. **Format/validator groundwork**
   * add qtype/qprofile constants;
   * add Q8 tensor-entry validation;
   * update name helpers and converter mirrors.

2. **Converter planning/writing**
   * extend `TensorPlan` for scale ranges;
   * implement tensor selection from registry metadata;
   * implement BF16→Q8_0 writer;
   * emit quant summary JSON.

3. **Metal loader**
   * map qweight and qscale ranges;
   * add dtype-aware weight views/bindings;
   * keep fp16 decoder and vision paths passing unchanged.

4. **Embedding + LM-head Q8**
   * implement the isolated Q8 embedding and Q8 LM-head kernels;
   * verify prompt text-only generation with Q8 embeddings and Q8 LM-head.

5. **Attention Q8**
   * Q/K/V and O for decode and prefill.

6. **Dense/shared MLP Q8**
   * fused gate+up Q8 kernels and fused down Q8 kernels.

7. **Routed MoE Q8**
   * Q8 expert slabs and selected-expert kernels.

8. **Docs/tests/profiling**
   * converter docs;
   * Python loading example;
   * loader rejection tests for malformed Q8 metadata;
   * numeric smoke/parity tests against fp16.

---

## First deliverable

A grounded first deliverable is:

```text
Model:
  mixed fp16/Q8_0 .uocr with qprofile UOCR_QPROFILE_MIXED_Q8_0

Quantized families:
  token embedding
  decoder attention projections
  decoder dense MLP
  decoder MoE routed experts
  decoder MoE shared experts
  LM head Q8; router fp16

Runtime:
  Metal fused Q8 embedding
  fused Q8 attention/dense/MoE/LM-head paths integrated at current call sites
  no standalone dequantization kernels and no dequantized Q8 weight buffers
  fp16 activations, KV cache, image features, runtime arenas
  fp16 vision/projector/norm/router paths unchanged

User API:
  load by model_path; quantization inferred from .uocr qprofile/tensor qtypes

Reporting:
  q8 tensor count, qweight bytes, qscale bytes, fp16-equivalent bytes,
  estimated savings, and Q8 kernel timings
```
