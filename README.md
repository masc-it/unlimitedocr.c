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

> `unlimitedocr-c` is published on PyPI. Version `0.4.1` is the latest stable release.

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

## Threading

One `UnlimitedOCR` instance is safe to share across threads.  A per-object
lock serializes native inference, so shared calls are correct but execute one
at a time; `close()` waits for the active call.  Image loading and request
preparation run outside the lock and overlap freely.

```python
from concurrent.futures import ThreadPoolExecutor
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()
with ThreadPoolExecutor(max_workers=4) as pool:
    texts = list(pool.map(ocr.generate, ["a.png", "b.png", "c.png", "d.png"]))
ocr.close()
```

For calls that actually run concurrently, create one instance per execution
lane and assign work on your side.  Each instance owns its own engine and
runtime memory (KV cache, scratch, caches), so budget memory per instance;
the mmap-backed model weights are shared by the OS across instances.

```python
from concurrent.futures import ThreadPoolExecutor
from unlimitedocr_c import UnlimitedOCR

lanes = [UnlimitedOCR(quant="q8"), UnlimitedOCR(quant="q8")]
with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
    texts = list(pool.map(lambda t: t[0].generate(t[1]), zip(lanes, ["a.png", "b.png"])))
for lane in lanes:
    lane.close()
```

At the C ABI, callers serialize operations sharing one `uocr_engine` and
quiesce calls before `uocr_engine_close()`; separate engines may be used from
separate threads (see the threading contract in `include/unlimitedocr.h`).

## Quantization (Q8 / dynamic Q4)

The engine supports three model profiles:

| Profile | Weights | Model file | Usage |
| --- | --- | --- | --- |
| fp16 (default) | fp16 | ~6.7 GB | `UnlimitedOCR()` |
| mixed Q8_0 | int8 weights + fp16 scales | ~3.5 GB | `UnlimitedOCR(quant="q8")` |
| dynamic mixed Q4 | int4 nearly everywhere, Q8 attention | ~1.8 GB | `UnlimitedOCR(quant="q4")` |

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

Dynamic mixed Q4 stores the routed MoE experts, shared experts, dense MLP,
LM head, token embedding and vision encoders as group-64 Q4_0 — symmetric
int4 with fp16 scales and a group-half-split nibble packing chosen for
vectorized dequantization in the fused Metal kernels.  The exact Q4 subset is
cfg-gated by `configs/quant-cfg.yaml`, so the converter only emits Q4 for
modules with runtime-safe fused kernels; attention projections stay Q8_0
(highest quality sensitivity); norms, biases and convolutions stay fp16.
On M1 Pro the routed-expert decode step measured ~2.7× faster than Q8 and
the fused LM-head argmax ~1.2× faster.

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
