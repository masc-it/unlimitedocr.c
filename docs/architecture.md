# Architecture

UnlimitedOCR C is an fp16 native runtime with a Python frontend and a Metal backend.

## Components

- `src/core/`: public C API implementation, request validation, memory/profile accounting.
- `src/model/`: `.uocr` file layout, tensor registry, and fp16 model validation.
- `src/runtime/`: sequence construction, logits processing, vision request helpers.
- `src/backend/metal/`: Metal context, fp16 runtime orchestration, and kernels.
- `src/backend/cpu_ref/`: CPU reference helpers used by tests.
- `src/unlimitedocr_c/`: Python API, frontend preprocessing, converter, FFI loader, parity/perf helpers.
- `tools/uocr-dump`: native model-inspection tool.

## Model format

`.uocr` is a stable, page-aligned container for fp16 OCR weights:

1. file header with format/version/profile metadata
2. section directory
3. config record
4. tokenizer metadata record
5. provenance record
6. tensor directory
7. tensor-data section

Only the `fp16` profile is supported. Tensor entries keep logical and physical shapes identical, with zero block/row/side-buffer metadata.

## Converter

`tools/uocr-convert` / `src/unlimitedocr_c/convert.py` plans the `.uocr` layout from safetensors metadata and streams BF16 source payloads to fp16 tensor payloads. The converter preserves source row-major order and assigns stable tensor ids through `tensor_registry.py`.

## Metal backend

The Metal backend owns:

- device/resource setup
- reusable runtime arenas
- fp16 vision/projector/decoder pipelines
- prompt assembly from model embeddings
- KV-cache and generation loops
- diagnostic fp16 parity helpers

Hot kernels are specialized with function constants where shapes are fixed by the OCR model.

## Python loading

Source-tree Python use must load `build/release/libunlimitedocr.dylib`; debug or stale native libraries are rejected to avoid dylib/metallib mismatches.

## End-to-end check

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DUOCR_ENABLE_METAL=ON -DUOCR_BUILD_TOOLS=ON -DUOCR_BUILD_TESTS=OFF -DUOCR_METAL_PRECOMPILE=ON
cmake --build build/release -j
uv run probes/e2e_generation_probe.py --profile base
```
