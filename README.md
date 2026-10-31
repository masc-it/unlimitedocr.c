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

> `unlimitedocr-c` is published on PyPI. Version `0.6.0` is the latest release; v2 inference is experimental and single-page only.
>
> [0.6.0 release notes](docs/release_0.6.0.md): faster mixed-Q4 inference, reduced vision memory, and a Q8/FP16 token-selection race fix.

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

## Unlimited-OCR-2 (experimental single-page support)

Experimental native v2 **single-page** OCR works in `base` and `gundam` modes.
Mixed Q4 is the primary v2 quantization profile. Both Q4 and Q8 have passed
base/gundam OCR on rendered invoices, including outputs longer than 128 tokens.
Upstream numerical parity and v2 FP16 OCR remain unverified. The default model
remains v1; select Q4 explicitly.

```python
from unlimitedocr_c import UnlimitedOCR

with UnlimitedOCR(model="v2", quant="q4") as ocr:
    print(ocr.generate("page.png", profile="base"))
```

V2 requires Metal and preserves its page/block markup in output. It warms GPU
pipelines on the first image request. Multi-page generation and bounded-memory
cross-window continuation are **not implemented yet**; separate `generate` calls
do not preserve document context.

To prepare only the converted model (downloads the checkpoint on first use):

```python
from unlimitedocr_c import resolve_model_path

model_path = resolve_model_path(model="v2", quant="q4")
```

The official v2 weights and tokenizer use the same pinned Hugging Face revision.
Converted files are cached separately by variant, repository/revision, and
quantization. Default v1 still finds its older cache files. Tokenizers use the
repository-specific Hugging Face snapshot cache, not a shared copied file.
For `UnlimitedOCR`, `download=False` requires both the model and tokenizer to be
available locally.
`force_reconvert=True` replaces the selected model only after conversion succeeds.

Progress and remaining work: [implementation plan](docs/plan_unlimitedocr2.md).

## Typed OCR output

`generate()` returns raw markup. `parse_ocr()` auto-detects v1/v2 and returns
immutable `Document -> Page -> Block` records, in generated reading order:

```python
from dataclasses import asdict
from unlimitedocr_c import UnlimitedOCR, parse_ocr

with UnlimitedOCR(model="v2", quant="q4") as ocr:
    document = parse_ocr(ocr.generate("page.png"))
for page in document.pages:
    for block in page.blocks:
        print(block.kind, block.bbox, block.heading_level, block.content)

payload = asdict(document)  # Ready for json.dumps().
```

- `BBox(x1, y1, x2, y2)` uses page-relative coordinates in **[0, 1]**, with a
  top-left origin. V1 coordinates are divided by **999**; v2 by **1000**.
  Boxes are not clipped or reordered. Zero-width/height boxes are retained.
- V1 produces one implicit page. V2 preserves explicit page boundaries,
  including blank pages. A v2 block without coordinates has `bbox=None`.
- Labels (including unknown labels and `cross_`/`append_` prefixes), heading
  levels and content are preserved. Content includes its original HTML/LaTeX
  and whitespace. `block.merge_previous` records a same-page `<MERGE>` link;
  parsing does not combine blocks or cross-page table fragments.
- Invalid/unsupported layout markup raises `OCRParseError`, a `ValueError`
  subclass with a zero-based character `offset`. There is no partial-result
  or repair mode. Non-string input raises `TypeError`. Only single rectangular
  boxes are supported; `<CHILD>` hierarchy is rejected because its semantics
  are not defined by the upstream postprocessor.
- Embedded HTML/LaTeX is opaque, not validated. A plain string cannot establish
  whether generation reached EOS: truncation inside otherwise valid content
  is undetectable. Reserved layout controls in content are interpreted as
  structure, not literal text.

Format references and verification: [parser contract](docs/plan_ocr_parsing.md).

To generate synthetic documents, run OCR and save raw/parsed predictions:

```bash
uv run python tools/stress_ocr.py --models v1 v2
```

See the [stress runner guide](docs/synthetic_ocr_stress.md) for cases, saved
artifacts and failure reporting. It defaults to mixed Q4 and cached models.

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
modules with runtime-safe fused kernels; decoder attention projections stay Q8_0
(highest quality sensitivity); norms, biases and convolutions stay fp16.
The precision mix is selected during conversion, not adapted during inference.
V2 keeps all 60 DASH router tensors in FP16. Its mixed-Q4 model file is about
1.85 GB; observed tracked native peaks on the invoice fixtures were 3.33 GB in
base mode and 4.14 GB in gundam mode. These are fixture measurements, not fixed
limits; admission includes request capacity and a safety margin.

On M1 Pro the routed-expert decode step measured ~2.7× faster than Q8 and
the fused LM-head argmax ~1.2× faster. These are kernel measurements, not
end-to-end v2 timings.

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

## Acknowledgements

This would not have been possible without these projects as solid foundations to give AI high quality context and techniques:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [ds4](https://github.com/antirez/ds4)

