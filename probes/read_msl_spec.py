# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pymupdf>=1.24.0",
# ]
# ///

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def snippets(text: str, pattern: re.Pattern[str], window: int = 420) -> list[str]:
    out: list[str] = []
    for m in pattern.finditer(text):
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        out.append(normalize(text[start:end]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract/search Apple Metal Shading Language spec with PyMuPDF")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--toc", action="store_true")
    parser.add_argument("--terms", nargs="*", default=[])
    parser.add_argument("--pages", nargs="*", type=int, default=[], help="1-based pages to print")
    parser.add_argument("--max-snippets", type=int, default=3)
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    print(f"PDF: {args.pdf} pages={doc.page_count} metadata_title={doc.metadata.get('title')!r}")

    if args.toc:
        print("\n# Table of contents")
        for level, title, page in doc.get_toc(simple=True):
            indent = "  " * (level - 1)
            print(f"{indent}- p{page}: {title}")

    for p in args.pages:
        if not (1 <= p <= doc.page_count):
            continue
        text = doc[p - 1].get_text("text")
        print(f"\n\n===== PAGE {p} =====\n")
        print(text)

    for term in args.terms:
        # Allow regex terms. Case insensitive.
        pat = re.compile(term, re.IGNORECASE)
        hit_count = 0
        printed = 0
        print(f"\n# TERM /{term}/")
        for i in range(doc.page_count):
            text = doc[i].get_text("text")
            hits = snippets(text, pat)
            if hits:
                hit_count += len(hits)
                if printed < args.max_snippets:
                    print(f"\n-- page {i + 1} ({len(hits)} hit(s)) --")
                    for snip in hits[: max(0, args.max_snippets - printed)]:
                        print(snip)
                        print()
                        printed += 1
                    if printed >= args.max_snippets:
                        # Still count pages below? Cheap enough, keep scanning.
                        pass
        print(f"TOTAL_HITS {hit_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
