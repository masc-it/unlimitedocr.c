# Foundations

This project is a Metal-first fp16 C inference core for `baidu/Unlimited-OCR`.

## Scope

- Load a `.uocr` model file with fp16 tensor payloads.
- Run OCR preprocessing from the Python frontend.
- Execute the vision/projector/decoder path through the native C and Metal runtime.
- Keep runtime model metadata simple: one supported profile, `fp16`.

## Model file

The `.uocr` file is a page-aligned, mmap-friendly container with:

- file header
- config record
- tokenizer metadata record
- provenance record
- tensor directory
- fp16 tensor-data section

Tensor payloads preserve safetensors row-major order. Runtime tensors use stable tensor ids from `src/model/uocr_tensor_registry.h` / `src/unlimitedocr_c/tensor_registry.py`.

## Runtime priorities

1. Keep Python loading pinned to the Release native library in source-tree use.
2. Bind mmap-backed model tensors directly where possible.
3. Reuse Metal arenas across generation requests.
4. Prefer specialized fp16 Metal pipelines for hot decoder shapes.
5. Keep diagnostic APIs focused on fp16 parity and performance checks.

## Validation

Primary validation is end-to-end generation through:

```sh
uv run probes/e2e_generation_probe.py --profile base
```
