# unlimitedocr-c 0.7.0

## Highlights

- Experimental v2 multi-page OCR in base mode. Each page has its own image-token
  span, page positions and output markers.
- New `UnlimitedOCR.stream_pages()` and `generate_pages()` APIs. Streams consume
  ordered page images through bounded windows, with one lookahead page and
  retained attention state between windows. Chunks contain complete-window text.
- Native document sessions reuse main attention K/V, router keys and the
  lookahead cache. They retain only the previous page's visual/text tail rather
  than the complete document history.
- Document memory admission includes retained state, prompt rewriting, pixel
  uploads, patch extraction and cached position tables. Budget failures report
  required/configured bytes and leave the engine reusable.
- Fixed Float32 image inputs being interpreted as Float16 during SAM patch
  extraction. The existing Float16 kernel is unchanged.

## Usage

```python
from contextlib import closing
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR(model="v2", quant="q4")
pages = (f"page{i}.png" for i in range(1, 7))

with closing(ocr.stream_pages(pages, window_pages=1)) as chunks:
    for text in chunks:
        print(text, end="", flush=True)

ocr.close()
```

`generate_pages()` joins the same chunks and intentionally retains all output.
Streams must be consumed and closed on the thread that started them. A stream
owns its engine until completion or closure; other threads wait. `closing`
releases the stream even when the loop exits early or raises an exception.

## Compatibility and limits

- Defaults remain v1 and FP16. Existing single-image APIs, model files,
  quantization policy and model caches are preserved; no reconversion is needed.
  Native document functions are additive; public C struct layouts are unchanged.
- V2 remains experimental. Document streaming requires Metal and base-mode page
  images. Multi-page gundam and PDF rendering are not included.
- Continuation is approximate and retains limited context. Numerical parity with
  upstream or full-history inference is not claimed. CUDA runtime comparison is
  out of scope; full v2 FP16-model OCR remains unqualified.
- `max_page_tokens` defaults to 20000. Reaching this cap can force a page turn or
  EOS and truncate content. `max_output_tokens` caps cumulative emitted output;
  an incomplete window raises without yielding its partial text. Earlier chunks
  then describe an incomplete document.
- Working memory depends on window/token limits and input sizes. The native
  budget is not a process-RSS or total-device-memory cap. Caller-retained images
  and collected output also consume memory outside the streaming working set.
- Streaming has not been benchmarked against independent per-page calls. This
  release makes no throughput claim for that comparison.

The wheel targets **macOS 15+ on Apple Silicon**, requires Python 3.12+, and
bundles the native library, precompiled Metal library and quantization policy.
Profiling and runtime Metal compilation are disabled. No local compiler is
needed for installed-wheel inference.

## Implementation qualification

The three activities in the
[multi-page plan](https://github.com/masc-it/unlimitedocr.c/blob/v0.7.0/docs/plan_v2_multi_page.md)
passed native behavioral E2E probes, including page ordering, blank pages,
consecutive cache handoffs, streamed/joined equality, bounded input consumption,
early close, cancellation, input errors, budget rejection and engine recovery.
V1 Q4/FP16 and v2 Q4 single-page IDs matched frozen baselines; v2 Q8 passed
multi-window and single-page OCR. Float16/Float32 pixel inputs matched in the
tested cases.

On M1 Pro / macOS 26.0.1, increasing the same document from three to 30 pages with
one-page windows and a 768-token page cap kept tracked native peak at 2.482 GB
and sampled Metal peak at 2.500 GB. Sampled RSS peaks were 2.259/2.269 GB. This is
fixed-setting source-build evidence, not a universal memory bound or a fresh
release-wheel memory measurement.

## Release-wheel verification

- Installed the exact wheel into a fresh Python 3.12.10 environment and ran from
  `/tmp` on macOS 26.0.1. Both native artifacts target macOS 15.0 / arm64; a
  physical macOS 15 host was not tested.
- Verified the platform tag, all 22 RECORD entries, packaged dylib/metallib
  resolution and bundled quantization policy. Installed resources matched the
  wheel byte-for-byte; the policy matched the source file.
- Ten prepared-request cases covered v1 Q4/Q8/FP16 and v2 Q4/Q8 in base/gundam.
  All token IDs matched the source build and outputs parsed into the expected
  15 invoice blocks. Four Q8 repeats and five base-mode Float32 input checks
  also matched, for 19 installed-wheel prepared requests in total.
- Three high-level v1 calls passed: Q4 base/gundam and FP16 base. The remaining
  broad high-level checks were stopped to avoid unnecessary memory pressure.
- A smaller Q4-only check used one engine, one-page windows, a 256-token page
  cap, a 768-token document cap and a 2.7 GB native budget. Three distinct pages
  crossed two consecutive handoffs with correct content/order; streamed and
  joined output matched. The native budget is not a process-memory cap.
- No long-document memory measurement or broad test-suite rerun was performed
  for the release wheel. Earlier implementation evidence is listed separately
  above. Model caches were reused without reconversion.

Artifact: `unlimitedocr_c-0.7.0-py3-none-macosx_15_0_arm64.whl`

SHA256: `fc69919117ee563dacd669d55ba4a10476d9fd48bc744a6d9c0fd2163ca67bd7`
