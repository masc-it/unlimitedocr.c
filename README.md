# unlimitedocr.c

Metal-first C inference core for `baidu/Unlimited-OCR`.

This project is intentionally narrow and model-specific. Python owns the v1 frontend
(image loading, preprocessing, prompt rendering, tokenization, decoding); the C
library owns prepared-request validation, model loading, Metal/CUDA backends,
KV cache, generation, logits processors, and batching.

## Current status

Implemented first native scaffold:

- plain CMake build
- `include/unlimitedocr.h` C ABI
- `libunlimitedocr` shared library stub
- CPU reference backend placeholder
- macOS Metal backend skeleton with runtime source compilation, mmap/no-copy smoke test, `.uocr` no-copy model view mapping/warmup, transient retain tracking, reusable named scratch buffers, long-lived runtime arenas, and fp16 get-rows gather
- prepared-request validation foundation plus per-sequence prompt/image span state
- upstream-compatible sliding-window no-repeat n-gram logits processor
- DS4-style internal allocation wrappers with live/peak accounting and no-allocation guard primitive
- runtime memory category accounting, KV/vision/decoder/MoE/logits estimates, public memory reports, Metal working-set default budgets, and memory-budget checks
- mmap-backed `.uocr` header/config/tokenizer/provenance/tensor-directory loader
- C engine `model_path` loading with resident tensor-data memory accounting
- page-aligned tensor-data section validation for Metal no-copy model views
- stable tensor registry and binary-search tensor-id lookup shared by converter planning and runtime metadata
- fp16 converter dry-run against cached safetensors header/index with planned page-aligned `.uocr` layout
- fp16 `.uocr` writer that streams real safetensors tensor ranges with BF16->FP16 conversion when weights are available
- Python prepared-request frontend and fixture writer
- Python `ctypes` wrapper for build-tree `libunlimitedocr`
- `uocr-dump` header/config/tokenizer/provenance/tensor-layout inspection tool
- native CTest and pytest smoke tests

Real inference kernels are not implemented yet. Full-model conversion is implemented for fp16 layout/streaming but requires the real safetensors payload, which is not checked into this repo.

## Build

```sh
make test
```

Equivalent direct commands:

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DUOCR_ENABLE_METAL=AUTO
cmake --build build/debug -j
ctest --test-dir build/debug --output-on-failure
uv run pytest
```

The build emits `build/debug/libunlimitedocr.dylib` on macOS. Python can also
locate the library via `UOCR_LIBRARY_PATH=/path/to/libunlimitedocr.dylib`.

## Prepared fixtures

```sh
uv run tools/uocr-ref-dump --image path/to/page.png --out fixtures/page --max-new-tokens 0
```

## Converter

Plan conversion from the cached header/index without full weights:

```sh
uv run tools/uocr-convert --dry-run
```

Write fp16 `.uocr` when the real safetensors payload is present:

```sh
uv run tools/uocr-convert --out /path/to/unlimitedocr-fp16.uocr --overwrite
```

Single-tensor debugging can use `--tensor NAME_OR_ID`.
