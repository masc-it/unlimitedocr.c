# Synthetic OCR stress runner

`tools/stress_ocr.py` is one standalone runner for document rendering, native
inference, strict parsing, and artifact storage. Run it from the repository's
Python environment on a Metal-capable Mac with the native library built.

```bash
# Cached v2 mixed Q4; eight cases in base and gundam modes.
uv run python tools/stress_ocr.py

# Compare both variants on identical images. The output directory must be new.
uv run python tools/stress_ocr.py --models v1 v2 --output data.tmp/synthetic-ocr/comparison

# Preview the generated dataset without loading weights or using Metal.
uv run python tools/stress_ocr.py --render-only --output data.tmp/synthetic-ocr/preview

# Repeat a smaller selection with the same engine and a different token cap.
uv run python tools/stress_ocr.py --cases invoice tiny_text --repeats 3 --max-new-tokens 4096
```

Defaults are `--models v2 --quant q4 --profiles base gundam --seed 42`, one
repetition and 2048 generated tokens per prediction. Without `--output`, the
runner creates a UTC-timestamped directory under `data.tmp/synthetic-ocr/`.
Existing output directories are rejected; there is no implicit overwrite or
resume. Artifacts are ignored by Git and remain on disk after the run.

Models and tokenizers must already be cached unless `--download` is supplied.
`--font` selects a TTF/OTF file; otherwise rendering uses Pillow's bundled
scalable font. The seed, Pillow version, font hash when supplied, script hash,
model/tokenizer hashes, model paths and image hashes are recorded. This suite
uses ASCII text; it does not evaluate multilingual OCR.

## Cases

- `invoice`: headings, addresses, item quantities and monetary totals.
- `columns`: two columns of wrapped paragraphs with distinct reference numbers.
- `table`: an 18-row ruled table with quantities, unit prices and totals.
- `tiny_text`: 48 small-print audit records with unique serial numbers.
- `degraded_scan`: the same invoice content with reduced contrast, blur and JPEG artifacts.
- `edge_text`: near-border text, form fields, checkboxes and ambiguous glyphs.
- `code_markup`: printed literal OCR control tags and HTML inside source code.
- `blank`: an empty page.

The last two are intentional edge cases. In particular, literal tag text may
be indistinguishable from actual controls after decoding. Parser errors are
retained rather than repaired. The public `parse_ocr()` function still raises;
only this batch runner catches its exception to save the failure and continue.

## Saved files

```text
runner.py                      exact script snapshot for this run
run.json                       configuration and model provenance
images/<case>.png              generated input
images/<case>.txt              reference text in intended reading order
images/<case>.json             reference lines, normalized boxes, transforms
dataset.json                   case paths and input hashes
results.jsonl                  one completed prediction record per line
summary.json                   status counts, EOS failures, memory overruns
predictions/<model>/<profile>/r000-<case>/
    request.json               prepared prompt/view metadata
    tokens.json                generated token IDs, including EOS if emitted
    raw.txt                    decoded output, never repaired
    parsed.json                typed document serialized with dataclasses.asdict
    boxes.png                  predicted boxes drawn over a single-page input
    result.json                status, timings, metrics, memory and errors
```

`parsed.json` and `boxes.png` are absent when parsing fails. An inference error
can leave only the request/result files. Each prediction starts with a
`status: running` record, so an interrupted request can be identified; completed
records and raw predictions are persisted before moving to the next case.
The JSONL index is streamed to disk; only summary counters are retained as
repetitions increase. Setup failures are recorded in `summary.json`.

The runner calls `Engine.generate_prepared`, the same native image-generation
path used by `UnlimitedOCR.generate`. This exposes token IDs and an explicit
output cap, allowing it to distinguish EOS from a truncated but parseable
string. One engine is reused per variant; variants run sequentially. No warmup
is performed, so first-use generation times include pipeline setup.

Exit code 1 means at least one setup, inference, parsing, completion, page-count
or memory-estimate check failed. Results are still saved. Exit 130 indicates a
keyboard interrupt. The adversarial default suite can legitimately exit 1.

## What the metrics mean

- `word_recall` is case-insensitive, bag-of-words reference coverage. Duplicate
  words are counted; table HTML tags are removed for scoring only. It does not
  measure precision, reading order, punctuation fidelity or general accuracy.
  No score is calculated when parsing fails. Blank pages have no recall score.
- `blank_has_content` flags recognized words on a blank reference page.
- Reference boxes describe drawn text lines; model boxes may group entire
  paragraphs or tables. They are saved for inspection, not scored as matching
  segmentation. `boxes.png` labels predicted blocks in generated order.
- EOS and token-limit status are independent of parsing success. Inspect both.
- Memory reports contain native tracked allocations, not process RSS or a
  proof of bounded long-document continuation. This runner makes independent
  single-page requests; it does not exercise the unfinished document scheduler.

## Initial Q4 run

Command:

```bash
uv run --extra dev python tools/stress_ocr.py --models v1 v2 --output data.tmp/synthetic-ocr/q4-stress
```

All 32 predictions completed and remain in that directory:

- 28 parsed successfully; four failed on `code_markup` (both models/profiles).
- V1 reached EOS on the markup page but emitted literal reserved tags within
  content. V2 became repetitive and hit the 2048-token cap in both profiles.
  No parser repairs or generation-policy changes were made.
- The v2/base markup prediction contains two decoded `<PAGE>` strings but only
  one PAGE special-token ID. This demonstrates the information lost when
  parsing decoded text alone. Saved token IDs allow further investigation.
- Invoice, columns, table and degraded-scan predictions had full reference word
  coverage. Tiny-text base recall was approximately 99.32% for v1 and 99.55%
  for v2; both gundam runs reached 100%. The ambiguous-glyph form scored 95.56%
  in all four combinations (`0O` and `1Il` were misread).
- Blank pages produced an empty-content image block, with no hallucinated text.
- No native estimate overruns occurred. Maximum tracked peaks were
  4,158,762,496 bytes for v1 and 4,139,379,200 bytes for v2.

Runner verification also covered deterministic image generation across two
fresh output directories, unclipped reference boxes, refusal to overwrite,
and a two-token native run that retained parser failures and continued to the
next case. Two repeated native predictions verified per-round artifact paths,
index/result consistency and stable live native allocation. A deliberately
insufficient budget verified both models' setup failures and empty-result
summaries. Script snapshot hashes matched run manifests. Disposable verification
artifacts were removed; the full 32-result stress run is intentionally retained.

Existing checks: `uv run --extra dev python -m pytest tests -q` with large/perf
opt-ins disabled: 105 passed, 7 skipped, 3 expected invalid-limit warmup warnings.
