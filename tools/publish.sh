#!/usr/bin/env bash
set -euo pipefail

# Build the native library + metallib, assemble a macOS arm64 wheel, publish to PyPI.
#
# Usage: tools/publish.sh
#
# Requires: uv, xcrun, and PyPI credentials (UV_PUBLISH_TOKEN or ~/.pypirc).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# 1. Build the Release native library and precompiled Metal library.
echo "==> Building native library + metallib…"
UOCR_ENABLE_PROFILING=OFF tools/build.sh -j

# 2. Stage native artifacts into the package so uv_build includes them.
echo "==> Staging native artifacts…"
mkdir -p src/unlimitedocr_c/lib src/unlimitedocr_c/metal
cp build/release/libunlimitedocr.0.1.0.dylib src/unlimitedocr_c/lib/libunlimitedocr.dylib
cp build/release/unlimitedocr.metallib    src/unlimitedocr_c/metal/unlimitedocr.metallib

cleanup() {
    rm -rf src/unlimitedocr_c/lib src/unlimitedocr_c/metal
}
trap cleanup EXIT

# 3. Read version from pyproject.toml.
VERSION="$(python3 -c "import tomllib,sys; print(tomllib.load(sys.stdin.buffer)['project']['version'])" < pyproject.toml)"
echo "==> Package version: ${VERSION}"

# 4. Build the wheel.
echo "==> Building wheel…"
rm -rf dist/*.whl dist/*.tar.gz
uv build --wheel

# 5. Retag the wheel as macOS arm64 (uv_build emits py3-none-any).
PYTHON_REQ=">=3.12"
SRC_WHL="dist/unlimitedocr_c-${VERSION}-py3-none-any.whl"
DST_WHL="dist/unlimitedocr_c-${VERSION}-py3-none-macosx_12_0_arm64.whl"

python3 - <<PY "${SRC_WHL}" "${DST_WHL}"
import base64, csv, hashlib, io, sys, zipfile
from pathlib import Path
import shutil, tempfile

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
tmp = Path(tempfile.mkdtemp())
try:
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)
    dist_info = next(tmp.glob("*.dist-info"))
    wheel = dist_info / "WHEEL"
    wheel.write_text(wheel.read_text().replace("Tag: py3-none-any", "Tag: py3-none-macosx_12_0_arm64"))
    rows = []
    for path in sorted(p for p in tmp.rglob("*") if p.is_file()):
        rel = path.relative_to(tmp).as_posix()
        if rel.endswith("RECORD"):
            rows.append([rel, "", ""])
        else:
            data = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append([rel, f"sha256={digest}", str(len(data))])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    (dist_info / "RECORD").write_text(buf.getvalue())
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(p for p in tmp.rglob("*") if p.is_file()):
            z.write(path, path.relative_to(tmp).as_posix())
finally:
    shutil.rmtree(tmp)
src.unlink()
print(dst)
PY

echo "==> Wheel: ${DST_WHL}"

# 6. Publish to PyPI.
echo "==> Publishing to PyPI…"
uv publish "${DST_WHL}"

echo "==> Published ${DST_WHL}"