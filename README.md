# unlimitedocr.c

An optimized C inference engine for the `baidu/Unlimited-OCR` model, with a
Python frontend for practical OCR workflows.

The product goal is simple: run UnlimitedOCR locally with low-overhead native
inference. Python handles image loading, prompt construction, tokenization, and
text decoding; the C library owns model loading, memory management, KV cache,
logits processing, and GPU execution.

## Backend support

| Backend | Status |
| --- | --- |
| Metal | Supported today on macOS |

The runtime executes fp16 `.uocr` model files.

## Highlights

- Native C ABI in `include/unlimitedocr.h`.
- Metal backend with precompiled shader support.
- mmap-backed fp16 `.uocr` tensor views.
- Python API that accepts paths, URLs, bytes, file-like objects, base64/data URI
  strings, and `PIL.Image.Image` inputs.
- Cache-aware model resolution with Hugging Face download/conversion.
- Engine reuse for repeated OCR calls.
- Native profile and memory reports.

## Requirements

- macOS with a Metal-capable GPU.
- Python 3.12+.
- CMake and Xcode Command Line Tools.
- `uv` for local development commands.

## Build from source

Build the Release native library and precompile the Metal library:

```sh
uv sync
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DUOCR_ENABLE_METAL=ON \
  -DUOCR_BUILD_TOOLS=ON \
  -DUOCR_METAL_PRECOMPILE=ON
cmake --build build/release -j
```

When running from the source tree, Python loads the Release library from
`build/release/libunlimitedocr.dylib`. Use `UOCR_LIBRARY_PATH` to point Python
at this Release build explicitly.

## Quick start from Python

```python
from unlimitedocr_c import UnlimitedOCR

with UnlimitedOCR() as ocr:
    text = ocr.generate("page.png", profile="base")
    print(text)
```

`UnlimitedOCR()` resolves the model automatically. Resolution order is:

1. explicit `model_path=...`
2. `UOCR_MODEL_PATH`
3. local `dist/unlimitedocr-fp16.uocr`
4. UnlimitedOCR cache directories
5. preconverted model download from Hugging Face
6. upstream `baidu/Unlimited-OCR` download followed by fp16 `.uocr` conversion

The high-level wrapper keeps the native engine open and reuses it across calls;
release resources with `close()` or context-manager exit.

## Input examples

```python
from io import BytesIO
from PIL import Image
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()

print(ocr.generate("/path/to/page.png"))          # local path
print(ocr.generate("https://example.com/page.jpg"))  # URL
print(ocr.generate(open("page.png", "rb").read()))   # bytes
print(ocr.generate(BytesIO(b"...image bytes...")))   # file-like
print(ocr.generate(Image.open("page.png")))          # PIL image

ocr.close()
```

Profiles:

- `profile="base"` — standard single 1024px global view.
- `profile="gundam"` — crop-aware profile for larger or denser pages.

## Lower-level API with profiling

Use `ocr_image()` when you want the decoded text plus token ids and native
profile data:

```python
from unlimitedocr_c import ocr_image, resolve_model_path

model_path = resolve_model_path()
result = ocr_image(
    "page.png",
    model_path=model_path,
    preset="gundam",
    max_gen_tokens=512,
    profile=True,
)

print(result.text)
print(result.token_ids.tolist())
if result.profile:
    print(result.profile.summary())
```

## Reusing an engine explicitly

```python
from unlimitedocr_c import (
    Engine,
    EngineOptions,
    default_resource_path,
    generate_prepared,
    prepare_image,
    resolve_model_path,
)

request = prepare_image("page.png", preset="base", max_new_tokens=512)

engine = Engine(
    EngineOptions(
        model_path=str(resolve_model_path()),
        backend="metal",
        resource_path=default_resource_path("metal"),
        max_batch=1,
        max_prompt_tokens=request.n_tokens,
        max_gen_tokens=request.max_new_tokens,
        profile=True,
    )
)
try:
    result = generate_prepared(request, engine=engine, profile=True)
    print(result.text)
finally:
    engine.close()
```

## Multi-page and PDF OCR

```python
from unlimitedocr_c import ocr_pages, ocr_pdf, resolve_model_path

model_path = resolve_model_path()

pages = ocr_pages(["page-1.png", "page-2.png"], model_path=model_path)
print(pages.text)

# Requires PyMuPDF (`uv add pymupdf` or `pip install pymupdf`).
pdf = ocr_pdf("document.pdf", model_path=model_path, dpi=300)
print(pdf.text)
```

## Useful environment variables

- `UOCR_MODEL_PATH` — path to an fp16 `.uocr` model.
- `UOCR_CACHE_DIR` — cache directory for downloaded/converted models.
- `UOCR_MODEL_REPO_ID` / `UOCR_MODEL_REVISION` — preconverted model source.
- `UOCR_SOURCE_MODEL_REPO_ID` / `UOCR_SOURCE_MODEL_REVISION` — upstream model
  source used for local conversion.
- `UOCR_LIBRARY_PATH` — native `libunlimitedocr` path.
- `UOCR_METAL_RESOURCE_PATH` — directory containing `unlimitedocr.metallib` or
  Metal source kernels.
- `UOCR_PROFILE=1` — enable profile reporting through the environment.

## Model conversion tools

Preview the conversion layout:

```sh
uv run tools/uocr-convert --dry-run
```

Write an fp16 `.uocr` model when the upstream safetensors payload is available:

```sh
uv run tools/uocr-convert --out dist/unlimitedocr-fp16.uocr --overwrite
```

Inspect a converted model:

```sh
build/release/uocr-dump dist/unlimitedocr-fp16.uocr
```

## Verification

Run the end-to-end Metal generation probe:

```sh
uv run probes/e2e_generation_probe.py --profile base
```

For native and Python tests during development, configure with
`-DUOCR_BUILD_TESTS=ON`, then run `ctest` and `uv run pytest`.
