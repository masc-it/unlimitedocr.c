"""Strict parsing of Unlimited-OCR v1/v2 layout output, without inference dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


@dataclass(frozen=True, slots=True)
class BBox:
    """Page-relative (left, top, right, bottom), normalized to [0, 1]."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class Block:
    """One layout region; content is verbatim, including HTML/LaTeX/whitespace.

    ``kind`` retains the source label, including unknown and cross_/append_
    labels. ``merge_previous`` records a v2 same-page <MERGE> relationship;
    blocks and their boxes are never combined by the parser.
    """

    kind: str
    bbox: BBox | None
    content: str
    heading_level: int | None = None
    merge_previous: bool = False


@dataclass(frozen=True, slots=True)
class Page:
    """Blocks in generated reading order. An explicitly blank v2 page is empty."""

    blocks: tuple[Block, ...]


@dataclass(frozen=True, slots=True)
class Document:
    """Pages in source order; v1 has one implicit page."""

    pages: tuple[Page, ...]


class OCRParseError(ValueError):
    """Malformed or unsupported OCR layout syntax at a character offset."""

    def __init__(self, message: str, offset: int) -> None:
        self.offset = offset
        super().__init__(message, offset)

    def __str__(self) -> str:
        return f"{self.args[0]} at character {self.offset}"


_KIND = re.compile(r"[A-Za-z_][A-Za-z_0-9-]*")
_LEVEL = re.compile(r"<LEVEL([0-9]+)>")
# Reserve layout controls (including malformed/unterminated spellings), but
# leave HTML such as <table>, <figure> and <BLOCKQUOTE> inside content untouched.
_CONTROL = re.compile(
    r"<(?:/?(?:PAGE|BLOCK|FORK|CHILD|MERGE|LEVEL)(?=[\s/>0-9]|$)"
    r"|\|/?(?:det|ref)(?=[\s|>]|$))[^<>]*>?"
)


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _metadata(
    text: str, offset: int, scale: int, *, require_bbox: bool, kind: str | None = None
) -> tuple[str, BBox | None, int | None]:
    pos = _skip_space(text, 0)
    if kind is None:
        label = _KIND.match(text, pos)
        if label is None:
            raise OCRParseError("expected a block label", offset + pos)
        kind = label.group()
        pos = label.end()
    bbox = None
    level = None
    while (pos := _skip_space(text, pos)) < len(text):
        if text[pos] == "[":
            if bbox is not None:
                raise OCRParseError("duplicate bounding box", offset + pos)
            try:
                coords, end = json.JSONDecoder().raw_decode(text, pos)
            except (ValueError, RecursionError) as exc:
                raise OCRParseError("invalid bounding box", offset + pos) from exc
            if not isinstance(coords, list) or len(coords) != 4:
                raise OCRParseError("expected four bounding-box coordinates", offset + pos)
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= scale
                for v in coords
            ):
                raise OCRParseError(
                    f"bounding-box coordinates must be finite numbers in [0, {scale}]", offset + pos
                )
            x1, y1, x2, y2 = coords
            if x1 > x2 or y1 > y2:
                raise OCRParseError("bounding-box coordinates are reversed", offset + pos)
            bbox = BBox(x1 / scale, y1 / scale, x2 / scale, y2 / scale)
            pos = end
        else:
            match = _LEVEL.match(text, pos)
            if match is None:
                raise OCRParseError("unexpected block metadata", offset + pos)
            if level is not None:
                raise OCRParseError("duplicate heading level", offset + pos)
            try:
                level = int(match.group(1))
            except ValueError as exc:
                raise OCRParseError("invalid heading level", offset + pos) from exc
            if level < 1:
                raise OCRParseError("heading level must be positive", offset + pos)
            pos = match.end()
    if require_bbox and bbox is None:
        raise OCRParseError("v1 detection header requires a bounding box", offset)
    return kind, bbox, level


def _closing(text: str, start: int, marker: str) -> int:
    end = text.find(marker, start)
    if end < 0:
        raise OCRParseError(f"missing {marker}", start)
    return end


def _content_end(text: str, start: int) -> int:
    marker = _CONTROL.search(text, start)
    return marker.start() if marker is not None else len(text)


def _unexpected(offset: int, text: str) -> OCRParseError:
    if text.startswith("<CHILD>", offset):
        return OCRParseError("unsupported <CHILD> hierarchy", offset)
    return OCRParseError("unexpected or mixed-format layout control", offset)


def _parse_v1(text: str, pos: int) -> Document:
    blocks: list[Block] = []
    while pos < len(text):
        if text.startswith("<|ref|>", pos):
            label_start = pos + len("<|ref|>")
            label_end = _closing(text, label_start, "<|/ref|>")
            label = text[label_start:label_end].strip()
            if _KIND.fullmatch(label) is None:
                raise OCRParseError("invalid reference label", label_start)
            pos = _skip_space(text, label_end + len("<|/ref|>"))
            if not text.startswith("<|det|>", pos):
                raise OCRParseError("reference label requires a detection header", pos)
            start = pos + len("<|det|>")
            end = _closing(text, start, "<|/det|>")
            kind, bbox, level = _metadata(
                text[start:end], start, 999, require_bbox=True, kind=label
            )
        elif text.startswith("<|det|>", pos):
            start = pos + len("<|det|>")
            end = _closing(text, start, "<|/det|>")
            kind, bbox, level = _metadata(text[start:end], start, 999, require_bbox=True)
        else:
            raise _unexpected(pos, text)
        start = end + len("<|/det|>")
        pos = _content_end(text, start)
        blocks.append(Block(kind, bbox, text[start:pos], level))
    return Document((Page(tuple(blocks)),))


def _parse_v2(text: str, pos: int) -> Document:
    pages: list[Page] = []
    blocks: list[Block] = []
    merge_offset: int | None = None
    # Consume the first page marker here, so every subsequent one ends a page,
    # including blank pages. A trailing <PAGE> therefore adds an empty page.
    pos = _skip_space(text, pos + len("<PAGE>"))
    while pos < len(text):
        if text.startswith("<PAGE>", pos):
            if merge_offset is not None:
                raise OCRParseError("<MERGE> cannot cross a page boundary", merge_offset)
            pages.append(Page(tuple(blocks)))
            blocks = []
            pos = _skip_space(text, pos + len("<PAGE>"))
        elif text.startswith("<MERGE>", pos):
            if not blocks or merge_offset is not None:
                raise OCRParseError("<MERGE> requires a preceding block", pos)
            merge_offset = pos
            pos = _skip_space(text, pos + len("<MERGE>"))
        elif text.startswith("<BLOCK>", pos):
            start = pos + len("<BLOCK>")
            end = _closing(text, start, "<FORK>")
            kind, bbox, level = _metadata(text[start:end], start, 1000, require_bbox=False)
            start = end + len("<FORK>")
            pos = _content_end(text, start)
            blocks.append(Block(kind, bbox, text[start:pos], level, merge_offset is not None))
            merge_offset = None
        else:
            raise _unexpected(pos, text)
    if merge_offset is not None:
        raise OCRParseError("<MERGE> requires a following block", merge_offset)
    pages.append(Page(tuple(blocks)))
    return Document(tuple(pages))


def parse_ocr(text: str) -> Document:
    """Parse raw ``generate()`` output from v1 or v2 into immutable records.

    V1 detection boxes use a 0-999 scale; v2 block boxes use 0-1000. The result
    always uses [0, 1], with no clipping, box reordering or heuristic rescaling.
    Degenerate (zero-width/height) boxes are retained. V2 may omit a box.

    Raises ``OCRParseError`` for malformed/unsupported layout syntax, empty or
    unstructured output, invalid boxes, mixed formats and <CHILD>. Non-string
    input raises ``TypeError``. Content is opaque: embedded HTML/LaTeX is not
    validated. Without token/EOS metadata, truncation inside otherwise valid
    content cannot be detected. Reserved layout tokens cannot be literal text.
    """
    if not isinstance(text, str):
        raise TypeError("OCR output must be a string")
    pos = _skip_space(text, 0)
    if pos == len(text):
        raise OCRParseError("empty OCR output", pos)
    if text.startswith("<PAGE>", pos):
        return _parse_v2(text, pos)
    if text.startswith(("<|det|>", "<|ref|>"), pos):
        return _parse_v1(text, pos)
    raise OCRParseError("expected v1 detection markup or a v2 <PAGE> marker", pos)


__all__ = ["BBox", "Block", "Page", "Document", "OCRParseError", "parse_ocr"]
