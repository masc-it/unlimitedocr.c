# Typed OCR parsing

Status: `done`

## Contract

- `parse_ocr(text: str) -> Document` is a pure, standard-library Python function.
- Frozen `Document`, `Page`, `Block` and `BBox` dataclasses preserve page/block order. No inference API changes.
- V1 detection headers use one implicit page and coordinates divided by 999. Support both inline labels and the legacy `<|ref|>label<|/ref|><|det|>box<|/det|>` spelling.
- V2 requires `<PAGE>` and `<BLOCK>metadata<FORK>content`. Normalize coordinates by 1000. Preserve blank pages, heading levels, unknown labels, `cross_`/`append_` labels and same-page `<MERGE>` links (`Block.merge_previous`). Do not merge geometry across pages.
- Preserve block content verbatim, including HTML/LaTeX and inter-block whitespace. Missing boxes are represented by `None` where the v2 header omits one.
- Malformed/unsupported structural syntax, invalid coordinates and mixed formats raise `OCRParseError(ValueError)` with a character offset. No repair, clamping or partial result. Non-string input raises `TypeError`.
- Reject `<CHILD>`: the tokenizer contains it, but the upstream postprocessor defines no hierarchy semantics. Do not invent a tree.
- Only structural errors are detectable. A string alone cannot prove that generation reached EOS; truncated prose or embedded HTML can remain structurally valid. Embedded markup is preserved, not validated.

## Sources

- V1: pinned local `data/context/modeling_unlimitedocr.py`, `draw_bounding_boxes`, divides coordinates by 999.
- V2: [upstream utility documentation](https://github.com/baidu/Unlimited-OCR/blob/f37d525c995b12179248512dcb9505e8ec915990/utils/README.md) and [postprocessor](https://github.com/baidu/Unlimited-OCR/blob/f37d525c995b12179248512dcb9505e8ec915990/utils/process_unlimitedOCR2.py): 0-1000 coordinates, optional box/level in either order, blank pages, merge and continuation labels. Unlike its evaluation cleanup, this API must raise on errors.

## Phase 1: Parser and public types

- [done] Add `src/unlimitedocr_c/parsing.py` and package exports.
- [done] Verify both formats, legacy references, normalized endpoints, optional boxes, headings, blank pages, merge links and untouched content with a disposable probe that first fails.
- [done] Verify malformed/truncated headers, out-of-range/reversed/nonfinite boxes, mixed formats and unsupported controls raise without partial results.

## Phase 2: Usable example

- [done] Document the API, strictness and normalization; add parsed output to `notebooks/demo.ipynb` without changing image selections.
- [done] Run actual v1/v2 Q4 OCR through the parser, inspect normalized boxes, run existing Python checks and remove owned probes.

QA evidence:

- The disposable parser probe first failed with a missing `parse_ocr`/type import, then passed after implementation. Equivalent v1/v2 inputs produce equal typed documents. Covered legacy references, optional boxes, levels in both orders, blank pages, raw HTML/LaTeX/whitespace, unknown/continuation labels, merge links, frozen records, JSON serialization, malformed headers, invalid coordinates (including bool/NaN/Inf), mixed formats, unsupported controls, non-string inputs and 10,000 ordered blocks. Separate checks verified Unicode character offsets and opaque `<BLOCKQUOTE>` content. Document/error pickle round-trips passed; an initially failing error round-trip was fixed by retaining both constructor arguments in the exception.
- Real cached v1 and v2 mixed-Q4 models generated a ten-item rendered invoice in both base and gundam modes. All four outputs parsed as one page with 11 ordered blocks, all expected text and normalized boxes in [0, 1]. Inference was not mocked.
- Notebook code cells compile; its new parser cell executed verbatim on the real v2 output and matched the directly parsed document. No personal images were used or changed.
- `UOCR_RUN_LARGE_TESTS=0 UOCR_RUN_PERF_TESTS=0 uv run --extra dev python -m pytest tests -q`: 105 passed, 7 skipped, 3 expected invalid-limit warmup warnings. The initial command without `--extra dev` could not find pytest; the declared project extra resolved it.
- Disposable probes, generated image and OCR outputs removed after verification. No permanent tests or new dependencies added.
