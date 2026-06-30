# Quantization cleanup plan

## Production status

Runtime generation is already effectively **fp16-only**:
- `src/core/uocr_api.c`
  - rejects non-`UOCR_QPROFILE_FP16` for Metal text/image generation.

So most quant code is converter/planning, metadata validation, diagnostics, tests, and tools.

## Main removal targets

### C quant layer
- `src/quant/uocr_quant.c`
- `src/quant/uocr_quant.h`
- `src/quant/README.md`
- CMake entries:
  - `src/quant/uocr_quant.c`
  - `test_quant`

### Model format / metadata
- `src/model/uocr_format.h`
  - `UOCR_QPROFILE_DYN_Q8`
  - `UOCR_QPROFILE_DYN_Q4`
  - `UOCR_TENSOR_Q8_0`
  - `UOCR_TENSOR_Q4_K`
  - `UOCR_TENSOR_PADDED_Q4_K`
  - `UOCR_TENSOR_Q2_K`
  - `UOCR_TENSOR_IQ2_XXS`
  - quant reason/promotion enums if no longer useful
- `src/model/uocr_model_file.c`
  - quant type naming
  - quant metadata validation via `uocr_quant_*`
  - should become simple fp16-only validation.

### Metal kernels
In `src/backend/metal/kernels/uocr_smoke.metal`:
- helpers:
  - `uocr_q8_0_load_value`
  - `uocr_q4_k_get_scale_min`
  - `uocr_q4_k_load_value`
- kernels:
  - `uocr_get_rows_q8_0_to_f16`
  - `uocr_get_rows_q8_0_to_f32`
  - `uocr_dense_q8_0_to_f16`
  - `uocr_dense_q8_0_to_f32`
  - `uocr_dense_q4_k_to_f16`
  - `uocr_dense_q4_k_to_f32`
  - `uocr_dense_swiglu_gate_up_q8_0`
  - `uocr_moe_selected_gate_up_q4_k`
  - `uocr_moe_selected_down_sum_q8_0_to_f16/f32`
  - `uocr_moe_selected_down_sum_q4_k_to_f16/f32`
  - `uocr_moe_prefill_selected_gate_up_q4_k`
  - `uocr_moe_prefill_selected_down_sum_q8_0_to_f16/f32`
  - `uocr_moe_prefill_selected_down_sum_q4_k_to_f16/f32`

### Metal diagnostic API
- `src/backend/metal/uocr_metal.h`
- `src/backend/metal/uocr_metal.m`

Remove diagnostic functions for:
- Q8 get rows
- Q8/Q4 dense
- Q8 shared experts
- Q4 selected experts decode/prefill
- Q4+Q8 fallback
- padded Q4 paths

### Python converter
- `src/unlimitedocr_c/convert.py`
  - remove `--qprofile dyn-q8/dyn-q4`
  - remove Q8 packer
  - remove Q4 planning/hazard logic
  - keep only fp16 conversion.

### Quant parity/calibration tooling
Likely removable:
- `src/unlimitedocr_c/calibrate.py`
- `src/unlimitedocr_c/drift.py`
- quant parts of `src/unlimitedocr_c/parity_thresholds.py`
- exports in `src/unlimitedocr_c/__init__.py`
- tools:
  - `tools/uocr-calibrate`
  - `tools/uocr-drift`

Keep `parity.py` only if still useful for fp16 diagnostics.

### Tests
Remove/simplify:
- `tests/test_quant.c`
- quant sections of:
  - `tests/test_convert.py`
  - `tests/test_metal.c`
  - `tests/test_uocr_model_file.c`
- likely remove:
  - `tests/test_calibrate.py`
  - `tests/test_drift.py`
  - quant-specific parts of `tests/test_parity_thresholds.py`

### Reference/vendor quant files
Likely removable if no longer needed:
- `data/ds4/gguf-tools/quants.c`
- `data/ds4/gguf-tools/quants.h`
- `data/ds4/gguf-tools/deepseek4-quantize.c`

## Suggested cleanup order

1. Remove Metal quant kernels + diagnostic APIs.
2. Remove C quant module and CMake/test references.
3. Simplify model file validation to fp16-only.
4. Simplify converter to fp16-only.
5. Remove quant calibration/drift tools and exports.
6. Delete quant tests and docs.
7. Rebuild Release and run E2E probe.
