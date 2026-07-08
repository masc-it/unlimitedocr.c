# unlimitedocr.c

`unlimitedocr.c` runs the `baidu/Unlimited-OCR` model locally with a native C/Metal
inference engine and a small Python API for OCR workflows.

Python handles the user-facing pieces — image loading, prompt construction,
tokenization, and text decoding. The native library handles model loading, memory
management, KV cache, logits processing, and GPU execution.

## Installation

```bash
uv add unlimitedocr-c
```

> `unlimitedocr-c` is published on PyPI. Version `0.3.0` is the latest stable release.

## Quick start

```python
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()
text = ocr.generate("page.png")
print(text)
ocr.close()
```

Create one `UnlimitedOCR` instance, reuse it for as many images as you want, and
call `close()` when finished.

```python
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()

for image_path in ["page-1.png", "page-2.png", "page-3.png"]:
    print(ocr.generate(image_path, profile="base"))

ocr.close()
```

## Quantization (Q8 / Q4)

The engine supports three model profiles:

| Profile | Weights | Model file | Usage |
| --- | --- | --- | --- |
| fp16 (default) | fp16 | ~6.7 GB | `UnlimitedOCR()` |
| mixed Q8_0 | int8 weights + fp16 scales | ~3.5 GB | `UnlimitedOCR(quant="q8")` |
| mixed Q4 | Q8 + int4 MoE experts | ~2.2 GB | `UnlimitedOCR(quant="q4")` |

```python
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR(quant="q4")
text = ocr.generate("page.png")
ocr.close()
```

The first use converts the Hugging Face checkpoint into a cached
`unlimitedocr-q8.uocr` / `unlimitedocr-q4.uocr` model file; pass
`force_reconvert=True` to rebuild it.

Q8 quantizes all decoder and vision-encoder weight matrices (attention, MLPs,
MoE experts, LM head, embeddings, projector) with group-64 Q8_0.  Norms,
biases, position embeddings, convolutions, and all runtime activations stay
fp16.  Dequantization is fused inside the Metal kernels — quantization roughly
halves model memory and speeds up token generation, which is
memory-bandwidth-bound.

Mixed Q4 additionally stores the routed MoE expert weights (~80% of the
decoder parameters) as group-64 Q4_0 — symmetric int4 with fp16 scales and a
group-half-split nibble packing chosen for vectorized dequantization in the
fused Metal kernels.  The routed-expert decode step measured ~2.7× faster
than Q8 on M1 Pro; everything else stays Q8/fp16 as above.

## Input types

`generate()` accepts the common image forms directly:

| Input type | Example |
| --- | --- |
| local path | `ocr.generate("page.png")` |
| URL | `ocr.generate("https://example.com/page.jpg")` |
| bytes | `ocr.generate(open("page.png", "rb").read())` |
| file-like object | `ocr.generate(BytesIO(image_bytes))` |
| PIL image | `ocr.generate(Image.open("page.png"))` |
| base64/data URI string | `ocr.generate("data:image/png;base64,...")` |

Example:

```python
from io import BytesIO
from PIL import Image
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()

text_from_path = ocr.generate("page.png")
text_from_url = ocr.generate("https://example.com/page.jpg")
text_from_bytes = ocr.generate(open("page.png", "rb").read())
text_from_file = ocr.generate(BytesIO(open("page.png", "rb").read()))
text_from_pil = ocr.generate(Image.open("page.png"))

ocr.close()
```
