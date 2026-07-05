#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "unlimitedocr-c>=0.2.3",
#   "pillow>=10",
# ]
# ///
"""Quick e2e smoke test: OCR a screenshot from a fresh PyPI install."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from unlimitedocr_c import UnlimitedOCR

# The test image used during development.
IMAGE = Path(
    "/Users/mascit/Documents/Screenshot 2026-07-05 at 12.49.33.png"
)
if not IMAGE.exists():
    print(f"test image not found: {IMAGE}", file=sys.stderr)
    # Fall back to the repo docs image when running from the source tree.
    fallback = (
        Path(__file__).resolve().parents[1] / "docs" / "test.png"
    )
    if fallback.exists():
        IMAGE = fallback
    else:
        print("no test image available", file=sys.stderr)
        sys.exit(1)

print(f"image: {IMAGE}")
print()

ocr = UnlimitedOCR()
text = ocr.generate(str(IMAGE))
print(text)
ocr.close()
