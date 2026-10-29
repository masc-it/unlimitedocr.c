# unlimitedocr-c 0.5.0

## Highlights

- Experimental **single-page Unlimited-OCR-2** inference on Apple Metal, in base and gundam modes. Mixed Q4 is the primary v2 profile; Q4 and Q8 were exercised end-to-end.
- Explicit `model="v1" | "v2"` selection, pinned v2 checkpoint/tokenizer resolution and separate model/revision/quant caches. The default model remains v1 and the default quantization remains FP16.
- Native page-aware RoPE, DASH sparse attention, retained request state and page/EOS guards, reusing existing quantized vision, decoder and MoE kernels.
- `parse_ocr(text)` returns immutable typed pages/blocks with normalized bounding boxes. It accepts v1 and v2 layout formats and raises `OCRParseError` on malformed or unsupported structure.
- Updated demo notebook and `tools/stress_ocr.py` for reproducible synthetic documents, stored predictions, parsed layouts, box overlays and failure reports.

```python
from unlimitedocr_c import UnlimitedOCR, parse_ocr

with UnlimitedOCR(model="v2", quant="q4") as ocr:
    text = ocr.generate("page.png", profile="base")
    document = parse_ocr(text)
```

## Fixes

- Corrected vision memory admission to include SAM global-attention and convolution workspaces.
- Corrected generic Metal pipeline specialization for optional function constants.
- Corrected source-tensor comparison when filtered plans relocate quantization scale slabs.
- Bundled the quantization policy in the wheel. Earlier wheels omitted it and silently used an embeddings-only conversion policy outside the checkout. Missing policy data now raises instead of changing the requested precision mix.

Existing converted files are not rewritten automatically. If an older installed
wheel created a model with the embeddings-only policy, rebuild the selected
cache with `force_reconvert=True` (requires `download=True`; existing cached HF
weights can be reused). Explicit `model_path` overrides are not reconverted.

## Scope and known limitations

- V2 multi-page generation and bounded-memory cross-window continuation are **not included**. Separate `generate()` calls do not preserve document context. These remain required follow-up work.
- V2 FP16 end-to-end OCR and numerical parity against upstream execution remain unverified. Synthetic fixture success is not a general OCR accuracy guarantee.
- Literal OCR tag spellings printed inside documents can collide with structural markup in decoded text. The strict parser raises rather than guessing; unsupported `<CHILD>` hierarchy also raises.
- V2 repeated itself and reached the token cap on the adversarial literal-markup fixture. Parser success alone cannot establish that generation reached EOS.
- The wheel targets **macOS 15+ on Apple Silicon**, requires Python 3.12+, and includes the native dylib and precompiled Metal library. No local compiler is required for wheel inference.

## Verification

- Existing Python checks: 105 passed, 7 opt-in checks skipped; three expected invalid-limit warmup warnings.
- Native checks: 13 passed again during release preparation.
- Synthetic stress: eight document types, v1/v2 Q4, base/gundam, 32 predictions. 28 parsed successfully; four literal-markup cases failed parsing, including two capped v2 generations. No tracked-memory estimate overruns.
- Real v1 image output matched a baseline build exactly during integration. Model/tokenizer cache isolation and v2 request cleanup were exercised.

The actual macOS arm64 release wheel was installed in an isolated environment
and exercised from `/tmp`, outside the source checkout. Checks confirmed package
resource resolution, the bundled Q4 policy, failure when that policy is missing,
v2 planning and a real shard-2 Q4 tensor conversion. V1 Q4 and v2 Q4/Q8 each
passed base/gundam OCR and strict parsing of a ten-item invoice (six runs,
11 ordered blocks each). Wheel RECORD hashes and metadata were verified.

The missing-policy bug was reproduced first using the installed 0.4.1 wheel:
its default policy had only one module. The 0.5.0 installed-wheel checks then
passed with the intended Q4/Q8 mix and all 60 v2 DASH tensors retained in FP16.

Artifact: `unlimitedocr_c-0.5.0-py3-none-macosx_15_0_arm64.whl`

SHA256: `b362b2d5e24fef650692510c2c714d8cb4e1e639c42945d59ff6090edd0f6458`

For subsequent releases, use `bash tools/publish.sh --build-only`, verify the
installed wheel outside the checkout, then publish that exact file with
`uv publish <wheel>`. Build-only preserves prior versions' distribution files.
