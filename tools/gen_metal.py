#!/usr/bin/env python3
"""Concatenate all semantic .metal files into uocr_smoke.metal.

Usage: cd /Users/mascit/projects/unlimitedocr.c && python3 tools/gen_metal.py

Order: common.metal first, then alphasorted (but uocr_smoke.metal last,
since it contains smoke kernels that reference everything else).
"""

import os, glob

KERNELS = "src/backend/metal/kernels"
OUTPUT = f"{KERNELS}/uocr_smoke.metal"

# Collect .metal files
files = [f for f in os.listdir(KERNELS) if f.endswith(".metal")]
files.sort()

# Order: common.metal first, uocr_smoke.metal last
if "common.metal" in files:
    files.remove("common.metal")
    files.insert(0, "common.metal")
if "uocr_smoke.metal" in files:
    files.remove("uocr_smoke.metal")
    files.append("uocr_smoke.metal")

print(f"Concatenating {len(files)} files into {OUTPUT}:")

lines_out = []
lines_out.append(f"// uocr_smoke.metal — Auto-generated concatenation of all semantic kernel files.")
lines_out.append(f"// Do not edit directly. Edit the source files in {KERNELS}/ and run tools/gen_metal.py")
lines_out.append(f"// Generated from: {', '.join(files)}")
lines_out.append("")

for fname in files:
    path = os.path.join(KERNELS, fname)
    with open(path) as f:
        content = f.read()
    lines_out.append(f"// ═══════════════════════════════════════════")
    lines_out.append(f"//  {fname}")
    lines_out.append(f"// ═══════════════════════════════════════════")
    lines_out.append("")
    lines_out.append(content.strip())
    lines_out.append("")
    print(f"  {fname}")

with open(OUTPUT, 'w') as f:
    f.write('\n'.join(lines_out) + '\n')

print(f"\nDone. {OUTPUT} written ({len(lines_out)} lines).")
