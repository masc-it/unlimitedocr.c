#!/usr/bin/env python3
"""Concatenate UnlimitedOCR Metal kernel fragments into one compilable source."""

from __future__ import annotations

import argparse
from pathlib import Path

KERNELS = Path("src/backend/metal/kernels")
DEFAULT_OUTPUT = KERNELS / "uocr_smoke.metal"

# This order keeps shared parameter/helper definitions before users.  The split
# files are development fragments; the runtime/precompiled Metal library is a
# single translation unit.
ORDER = [
    "common.metal",
    "dense.metal",
    "norm.metal",
    "sam.metal",
    "sam_attention.metal",
    "sam_position.metal",
    "attention_decode.metal",
    "attention_prefill.metal",
    "attention.metal",
    "clip.metal",
    "clip_sam.metal",
    "embedding.metal",
    "embedding_q8.metal",
    "kv_cache.metal",
    "layout.metal",
    "moe.metal",
    "prompt_assembly.metal",
    "rope.metal",
    "sampling.metal",
    "sam_conv.metal",
    "sam_window.metal",
    "smoke.metal",
    "mpp_prototypes.metal",
]


def stripped_content(path: Path, *, is_common: bool) -> str:
    content = path.read_text(encoding="utf-8")
    if is_common:
        return content.strip()
    return "\n".join(
        line for line in content.splitlines() if line.strip() != '#include "common.metal"'
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=KERNELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    kernels = args.kernels
    output = args.output
    available = {path.name for path in kernels.glob("*.metal")}
    # uocr_smoke.metal is the generated runtime translation unit, not a source
    # fragment.  Keep smoke/debug kernels in smoke.metal to make regeneration
    # idempotent and avoid recursively embedding an older generated file.
    available.discard(DEFAULT_OUTPUT.name)
    order = [name for name in ORDER if name in available]
    for name in sorted(available - set(order)):
        order.insert(-1, name)

    lines: list[str] = [
        "// Auto-generated UnlimitedOCR Metal translation unit.",
        "// Edit src/backend/metal/kernels/*.metal and rerun tools/gen_metal.py.",
        f"// Generated from: {', '.join(order)}",
        "",
    ]
    for name in order:
        source = kernels / name
        lines.extend([
            "// ═══════════════════════════════════════════",
            f"//  {name}",
            "// ═══════════════════════════════════════════",
            "",
            stripped_content(source, is_common=(name == "common.metal")),
            "",
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} from {len(order)} Metal fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
