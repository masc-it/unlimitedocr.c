#!/usr/bin/env python3
"""Render synthetic single-page documents, run native OCR, and retain every result.

Run from the repository with `uv run python tools/stress_ocr.py`. Defaults to
cached v2 mixed Q4, base + gundam, eight cases, and 2048 generated tokens.
No predictions are repaired. A parser error is recorded without stopping the
remaining cases. Exit 1 signals generation/parser/completion/layout/memory
failures; imperfect word recall alone is a diagnostic, not a failing threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import platform
import random
import re
import sys
import time

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, __version__ as pillow_version

from unlimitedocr_c import (
    EOS_TOKEN_ID, Document, Engine, EngineOptions, ModelName, OCRParseError, ProfileName, decode_generated_ids,
    default_resource_path, parse_ocr, prepare_image, resolve_model_path,
)
from unlimitedocr_c.frontend import default_tokenizer_path

CASES = ("invoice", "columns", "table", "tiny_text", "degraded_scan", "edge_text", "code_markup", "blank")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Canvas:
    """Draw text and retain its actual raster bounding box as reference data."""

    def __init__(self, width: int, height: int, font_path: Path | None):
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.font_path = font_path
        self.fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self.lines: list[dict] = []

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self.fonts:
            self.fonts[size] = (ImageFont.truetype(str(self.font_path), size)
                                if self.font_path else ImageFont.load_default(size=size))
        return self.fonts[size]

    def text(self, x: int, y: int, text: str, size: int = 28, kind: str = "text") -> None:
        font = self.font(size)
        box = self.draw.textbbox((x, y), text, font=font, anchor="lt")
        width, height = self.image.size
        if not (0 <= box[0] <= box[2] <= width and 0 <= box[1] <= box[3] <= height):
            raise ValueError(f"synthetic text would be clipped: {text!r}")
        self.draw.text((x, y), text, font=font, fill="black", anchor="lt")
        self.lines.append({"text": text, "kind": kind, "font_size": size,
                           "bbox": [box[0] / width, box[1] / height, box[2] / width, box[3] / height]})

    def paragraph(self, x: int, y: int, text: str, width: int, size: int = 26) -> int:
        line = ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if line and self.font(size).getlength(candidate) > width:
                self.text(x, y, line, size)
                y += size + 12
                line = word
            else:
                line = candidate
        if line:
            self.text(x, y, line, size)
            y += size + 12
        return y


def render_case(name: str, seed: int, font_path: Path | None) -> tuple[Image.Image, dict]:
    content_case = "invoice" if name == "degraded_scan" else name
    rng = random.Random(f"{seed}:{content_case}")
    c = Canvas(1200, 1600, font_path)
    transforms = []
    if name in ("invoice", "degraded_scan"):
        c.text(60, 65, "NORTHWIND INVOICE", 48, "title")
        c.text(60, 155, f"Invoice: INV-{rng.randrange(10000, 99999)}")
        c.text(60, 210, "Date: 2026-10-29")
        c.text(60, 310, "Bill to: Ada Example")
        c.text(60, 365, "17 Maple Street, London")
        total = 0
        for n in range(1, 9):
            qty, cents = rng.randint(1, 5), rng.randint(100, 2500)
            amount = qty * cents
            total += amount
            c.text(60, 480 + n * 70, f"Item {n:02}   Quantity {qty}   USD {amount // 100}.{amount % 100:02}")
        c.text(60, 1230, f"TOTAL DUE: USD {total // 100}.{total % 100:02}", 40, "title")
        c.text(60, 1440, "Payment terms: net 30 days. Thank you.", 24)
        if name == "degraded_scan":
            image = ImageEnhance.Contrast(c.image).enhance(0.45).filter(ImageFilter.GaussianBlur(0.7))
            encoded = BytesIO()
            image.save(encoded, format="JPEG", quality=45)
            encoded.seek(0)
            with Image.open(encoded) as decoded:
                c.image = decoded.convert("RGB")
            transforms = ["contrast=0.45", "gaussian_blur=0.7px", "jpeg_quality=45"]
    elif name == "columns":
        c.text(60, 55, "OPERATIONS BULLETIN", 44, "title")
        for column, x in enumerate((60, 650), start=1):
            y = 160
            c.text(x, y, f"COLUMN {column}", 34, "title")
            y += 70
            for n in range(1, 7):
                key = rng.randrange(10000, 99999)
                text = (f"Section {column}.{n}: Reference {key}. The team reviewed the sample, "
                        "checked every measurement, and approved the next delivery.")
                y = c.paragraph(x, y, text, 490, 25) + 35
    elif name == "table":
        c.text(60, 55, "WAREHOUSE STOCK REPORT", 44, "title")
        edges = [60, 540, 700, 920, 1140]
        top, step, count = 180, 56, 18
        for x in edges:
            c.draw.line((x, top, x, top + (count + 1) * step), fill="black", width=2)
        for row in range(count + 2):
            y = top + row * step
            c.draw.line((edges[0], y, edges[-1], y), fill="black", width=2)
        for x, value in zip(edges, ("Product", "Qty", "Unit USD", "Total USD")):
            c.text(x + 12, top + 17, value, 24)
        total = 0
        for n in range(1, count + 1):
            qty, cents = rng.randint(1, 9), rng.randint(100, 3000)
            amount = qty * cents
            total += amount
            values = (f"Widget {n:02}", str(qty), f"{cents / 100:.2f}", f"{amount / 100:.2f}")
            for x, value in zip(edges, values):
                c.text(x + 12, top + n * step + 17, value, 24)
        c.text(60, 1390, f"Grand total: USD {total / 100:.2f}", 34)
    elif name == "tiny_text":
        c.text(60, 55, "SMALL PRINT AUDIT LOG", 44, "title")
        for n in range(1, 49):
            serial = rng.randrange(100000, 999999)
            c.text(60, 140 + (n - 1) * 28, f"Record {n:03} | Serial {serial} | Status VERIFIED | Amount {n}.25 USD", 17)
        c.text(60, 1530, "END OF LOG - 48 RECORDS", 22)
    elif name == "edge_text":
        c.text(4, 4, "TOP LEFT / REF 001", 18)
        c.text(850, 4, "TOP RIGHT / REF 002", 18)
        c.text(60, 190, "FORM AND EDGE CASES", 46, "title")
        for n, label in enumerate(("Name: Alex Example", "Account: AB-019283-Z", "Code: 0O 1Il 5S 8B",
                                   "Punctuation: (A/B) [C] {D} + 15%", "Email: example@example.test")):
            c.text(80, 340 + n * 110, label, 30)
        for n, label in enumerate(("Approved", "Pending", "Rejected")):
            y = 1040 + n * 80
            c.draw.rectangle((80, y, 110, y + 30), outline="black", width=2)
            if n == 0:
                c.draw.line((82, y + 15, 92, y + 27, 108, y + 3), fill="black", width=3)
            c.text(135, y, label, 28)
        c.text(4, 1560, "BOTTOM LEFT / REF 003", 18)
        c.text(850, 1560, "BOTTOM RIGHT / REF 004", 18)
    elif name == "code_markup":
        c.text(60, 65, "LITERAL MARKUP IN SOURCE CODE", 42, "title")
        lines = ('page = "<PAGE>"', 'block = "<BLOCK>text[0,0,999,999]<FORK>"',
                 'legacy = "<|det|>text[1,2,3,4]<|/det|>"', 'child = "<CHILD>"',
                 'html = "<table><tr><td>42</td></tr></table>"', 'if x < 10 and y > 2:',
                 '    total = (x + y) / 3.0', '    print("Result:", total)',
                 'These angle-bracket strings are printed content, not document structure.')
        for n, line in enumerate(lines):
            c.text(60, 190 + n * 100, line, 25)
    elif name != "blank":
        raise ValueError(f"unknown case: {name}")
    return c.image, {"case": name, "seed": seed, "width": c.image.width, "height": c.image.height,
                     "lines": c.lines, "text": "\n".join(line["text"] for line in c.lines),
                     "transforms": transforms, "bbox_frame": "original page; normalized xyxy"}


def word_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower()))


def content_metrics(document: Document, reference: dict) -> dict:
    contents = []
    for page in document.pages:
        for block in page.blocks:
            content = block.content
            if block.kind in ("table", "cross_table", "append_table"):
                content = html.unescape(re.sub(r"<[^>]*>", " ", content))
            contents.append(content)
    # Bag-of-words coverage, not CER, precision, reading-order or layout accuracy.
    expected, predicted = word_counts(reference["text"]), word_counts("\n".join(contents))
    hits = sum((expected & predicted).values())
    missing = expected - predicted
    return {"expected_words": sum(expected.values()), "predicted_words": sum(predicted.values()),
            "word_recall": hits / sum(expected.values()) if expected else None,
            "missing_words": dict(missing), "blank_has_content": not expected and bool(predicted),
            "pages": len(document.pages), "blocks": sum(len(p.blocks) for p in document.pages),
            "blocks_without_bbox": sum(b.bbox is None for p in document.pages for b in p.blocks)}


def save_overlay(image_path: Path, document: Document, path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("red", "blue", "green", "purple")
    for n, block in enumerate(b for page in document.pages for b in page.blocks):
        if block.bbox is not None:
            b = block.bbox
            box = (b.x1 * image.width, b.y1 * image.height, b.x2 * image.width, b.y2 * image.height)
            draw.rectangle(box, outline=colors[n % len(colors)], width=2)
            draw.text((box[0] + 2, box[1] + 2), str(n), fill=colors[n % len(colors)])
    image.save(path)


def predict(engine: Engine, variant: ModelName, profile: ProfileName, tokenizer: Path, case: dict,
            root: Path, destination: Path, max_new_tokens: int, repeat: int) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    result = {"model": variant, "profile": profile, "case": case["case"], "repeat": repeat,
              "directory": str(destination.relative_to(root)), "status": "running"}
    write_json(destination / "result.json", result)
    stage = "prepare"
    try:
        start = time.perf_counter()
        request = prepare_image(root / case["image"], model=variant, preset=profile,
                                tokenizer_path=tokenizer, max_new_tokens=max_new_tokens)
        result["prepare_seconds"] = time.perf_counter() - start
        write_json(destination / "request.json", request.manifest())
        result.update(prompt_tokens=request.n_tokens, views=len(request.views))
        stage = "generate"
        engine.profile_reset()
        start = time.perf_counter()
        ids = engine.generate_prepared(request)[0]
        result["generate_seconds"] = time.perf_counter() - start
        write_json(destination / "tokens.json", ids.tolist())
        result.update(output_tokens=len(ids), eos=bool(len(ids) and ids[-1] == EOS_TOKEN_ID))
        result["hit_token_limit"] = not result["eos"] and len(ids) >= max_new_tokens
        memory = engine.memory_report()
        result["memory"] = asdict(memory)
        result["memory_within_estimate"] = memory.total_peak_bytes <= memory.estimated_total_bytes
        stage = "decode"
        text = decode_generated_ids(ids, tokenizer)
        (destination / "raw.txt").write_text(text, encoding="utf-8")
    except (RuntimeError, ValueError, OSError) as error:
        result.update(status="inference_error", error={"stage": stage, "type": type(error).__name__, "message": str(error)})
        write_json(destination / "result.json", result)
        return result

    start = time.perf_counter()
    try:
        document = parse_ocr(text)
    except OCRParseError as error:
        result["parse_seconds"] = time.perf_counter() - start
        result.update(status="parse_error", error={"stage": "parse", "type": type(error).__name__,
                                                   "message": str(error), "offset": error.offset})
    else:
        result["parse_seconds"] = time.perf_counter() - start
        write_json(destination / "parsed.json", asdict(document))
        reference = json.loads((root / case["reference"]).read_text(encoding="utf-8"))
        result["metrics"] = content_metrics(document, reference)
        result["status"] = "ok" if len(document.pages) == 1 else "page_count_mismatch"
        if len(document.pages) == 1:
            save_overlay(root / case["image"], document, destination / "boxes.png")
    if result["status"] == "ok" and not result["eos"]:
        result["status"] = "incomplete"
    if result["status"] == "ok" and not result["memory_within_estimate"]:
        result["status"] = "memory_estimate_exceeded"
    write_json(destination / "result.json", result)
    return result


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory (existing paths are rejected)")
    parser.add_argument("--models", choices=("v1", "v2"), nargs="+", default=["v2"])
    parser.add_argument("--quant", choices=("q4", "q8", "fp16"), default="q4")
    parser.add_argument("--profiles", choices=("base", "gundam"), nargs="+", default=["base", "gundam"])
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=positive, default=1)
    parser.add_argument("--max-new-tokens", type=positive, default=2048)
    parser.add_argument("--memory-budget-mib", type=positive, help="per-engine native memory budget")
    parser.add_argument("--font", type=Path, help="TTF/OTF font; default is Pillow's bundled scalable font")
    parser.add_argument("--download", action="store_true", help="allow missing models/tokenizers to download/convert")
    parser.add_argument("--render-only", action="store_true", help="save images/references without loading a model")
    args = parser.parse_args()
    for field in ("models", "profiles", "cases"):
        if len(getattr(args, field)) != len(set(getattr(args, field))):
            parser.error(f"--{field} must not contain duplicates")
    if args.font is not None and not args.font.is_file():
        parser.error("--font must be an existing font file")
    if args.max_new_tokens > 32768 - 4096:
        parser.error("--max-new-tokens must be <= 28672 (fits both variants with 4096 prompt slots)")
    created = datetime.now(timezone.utc)
    root = (args.output or Path("data.tmp/synthetic-ocr") / created.strftime("%Y%m%dT%H%M%S.%fZ")).resolve()
    if root.exists():
        parser.error(f"output already exists: {root}")
    root.mkdir(parents=True)
    (root / "runner.py").write_bytes(Path(__file__).read_bytes())
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {"created_utc": created.isoformat(), "settings": settings,
                "python": platform.python_version(), "platform": platform.platform(), "pillow": pillow_version,
                "script_sha256": sha256(root / "runner.py"),
                "font_sha256": sha256(args.font) if args.font else None,
                "inference_api": "Engine.generate_prepared (same image-generation core as UnlimitedOCR.generate)",
                "models": {}}
    write_json(root / "run.json", manifest)
    dataset = []
    images = root / "images"
    images.mkdir()
    for name in args.cases:
        image, reference = render_case(name, args.seed, args.font)
        path = images / f"{name}.png"
        image.save(path)
        image.close()
        write_json(images / f"{name}.json", reference)
        (images / f"{name}.txt").write_text(reference["text"], encoding="utf-8")
        dataset.append({"case": name, "image": f"images/{name}.png", "reference": f"images/{name}.json",
                        "image_sha256": sha256(path)})
    write_json(root / "dataset.json", dataset)
    print(json.dumps({"event": "rendered", "output": str(root), "cases": len(dataset)}), flush=True)
    if args.render_only:
        return 0

    statuses: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    setup_errors = []
    (root / "results.jsonl").touch()
    interrupted = False
    try:
        for variant in args.models:
            engine = None
            try:
                try:
                    model = resolve_model_path(model=variant, quant=args.quant, download=args.download)
                    tokenizer = default_tokenizer_path(model=variant, local_files_only=not args.download)
                    resource = default_resource_path("metal")
                    manifest["models"][variant] = {"path": str(model), "bytes": model.stat().st_size,
                                                   "sha256": sha256(model), "tokenizer": str(tokenizer),
                                                   "tokenizer_sha256": sha256(tokenizer), "resource_path": resource}
                    write_json(root / "run.json", manifest)
                    options = EngineOptions(model_path=str(model), backend="metal", resource_path=resource,
                                            max_prompt_tokens=4096, max_gen_tokens=args.max_new_tokens,
                                            memory_budget_bytes=(args.memory_budget_mib or 0) * 1024**2, warmup=False)
                    engine = Engine(options)
                except (RuntimeError, ValueError, OSError) as error:
                    setup_errors.append({"model": variant, "type": type(error).__name__, "message": str(error)})
                    print(json.dumps({"event": "setup_error", **setup_errors[-1]}), flush=True)
                    continue
                for repeat in range(args.repeats):
                    for case in dataset:
                        for profile in args.profiles:
                            destination = root / "predictions" / variant / profile / f"r{repeat:03}-{case['case']}"
                            result = predict(engine, variant, profile, tokenizer, case, root, destination, args.max_new_tokens, repeat)
                            statuses[result["status"]] += 1
                            totals["completed_predictions"] += 1
                            totals["no_eos"] += result.get("eos") is False
                            totals["estimate_overruns"] += result.get("memory_within_estimate") is False
                            totals["blank_with_content"] += result.get("metrics", {}).get("blank_has_content", False)
                            with (root / "results.jsonl").open("a", encoding="utf-8") as output:
                                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                            print(json.dumps({"event": "prediction", "model": variant, "profile": profile,
                                              "case": case["case"], "repeat": repeat, "status": result["status"],
                                              "output_tokens": result.get("output_tokens"), "eos": result.get("eos"),
                                              "word_recall": result.get("metrics", {}).get("word_recall")}), flush=True)
            finally:
                if engine is not None:
                    engine.close()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        summary = {"output": str(root), "interrupted": interrupted,
                   "planned_predictions": len(args.models) * len(args.profiles) * len(dataset) * args.repeats,
                   "completed_predictions": totals["completed_predictions"], "statuses": dict(statuses),
                   "setup_errors": setup_errors,
                   "no_eos": totals["no_eos"],
                   "estimate_overruns": totals["estimate_overruns"],
                   "blank_with_content": totals["blank_with_content"],
                   "notes": ["Word recall is bag-of-words coverage, not an accuracy benchmark.",
                             "Table HTML tags are removed only for word scoring; raw and parsed outputs are unchanged.",
                             "Reference boxes are rendered text-line boxes, not expected model block segmentation.",
                             "Parser success does not establish completion; EOS is checked separately.",
                             "Memory counters are native tracked allocations, not process RSS.",
                             "No warmup is run; first-use generation timings include pipeline setup."]}
        write_json(root / "summary.json", summary)
        print(json.dumps({"event": "summary", **summary}), flush=True)
    return 130 if interrupted else int(bool(setup_errors) or any(status != "ok" for status in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
