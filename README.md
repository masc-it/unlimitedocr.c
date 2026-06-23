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
- macOS Metal backend skeleton with runtime source compilation and mmap/no-copy smoke test
- prepared-request validation foundation
- minimal mmap-backed `.uocr` header/config loader
- fp16 converter dry-run against cached safetensors header/index
- Python prepared-request frontend and fixture writer
- Python `ctypes` wrapper for build-tree `libunlimitedocr`
- `uocr-dump` header/config inspection tool
- native CTest and pytest smoke tests

Real inference kernels and the full converter are not implemented yet.

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

## Converter dry-run

```sh
uv run tools/uocr-convert --dry-run
```
