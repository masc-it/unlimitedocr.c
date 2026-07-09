#!/usr/bin/env bash
set -euo pipefail

# Build the native library + metallib, assemble a macOS arm64 wheel, publish to PyPI.
#
# Usage: tools/publish.sh
#
# Requires: uv, cmake, xcrun, and PyPI credentials (UV_PUBLISH_TOKEN or ~/.pypirc).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
MACOSX_DEPLOYMENT_TAG="${MACOSX_DEPLOYMENT_TARGET//./_}"
WHEEL_PLATFORM_TAG="macosx_${MACOSX_DEPLOYMENT_TAG}_arm64"
BUILD_DIR="${UOCR_PUBLISH_BUILD_DIR:-build/release-publish}"

cleanup() {
    rm -rf src/unlimitedocr_c/lib src/unlimitedocr_c/metal
}
cleanup
trap cleanup EXIT

# 1. Build a clean Release native library + precompiled Metal library for the
#    same deployment target we are going to advertise in the wheel tag. Do not
#    reuse build/release: its CMake cache may have been configured with a newer
#    SDK/deployment target, which would silently produce an incompatible wheel.
echo "==> Building native library + metallib for ${WHEEL_PLATFORM_TAG}…"
rm -rf "${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
  -DUOCR_ENABLE_PROFILING=OFF \
  -DUOCR_METAL_PRECOMPILE=ON \
  -DUOCR_METAL_RUNTIME_COMPILE=OFF \
  -DUOCR_BUILD_TESTS=OFF \
  -DUOCR_BUILD_TOOLS=OFF
cmake --build "${BUILD_DIR}" --parallel

BUILT_DYLIB="${BUILD_DIR}/libunlimitedocr.0.1.0.dylib"
BUILT_METALLIB="${BUILD_DIR}/unlimitedocr.metallib"

if [[ ! -f "${BUILT_DYLIB}" ]]; then
    echo "missing native library: ${BUILT_DYLIB}" >&2
    exit 1
fi
if [[ ! -f "${BUILT_METALLIB}" ]]; then
    echo "missing Metal library: ${BUILT_METALLIB}" >&2
    exit 1
fi

# 2. Refuse to publish if the native artifacts do not match the advertised tag.
echo "==> Verifying native deployment targets…"
DYLIB_MINOS="$(otool -l "${BUILT_DYLIB}" | awk '/LC_BUILD_VERSION/{in_build=1} in_build && $1 == "minos" {print $2; exit}')"
if [[ "${DYLIB_MINOS}" != "${MACOSX_DEPLOYMENT_TARGET}" ]]; then
    echo "dylib deployment target mismatch: got ${DYLIB_MINOS}, expected ${MACOSX_DEPLOYMENT_TARGET}" >&2
    exit 1
fi

METALLIB_TARGET_VERSION="${MACOSX_DEPLOYMENT_TARGET}"
if [[ "${METALLIB_TARGET_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    METALLIB_TARGET_VERSION="${METALLIB_TARGET_VERSION}.0"
fi
# Avoid `strings | grep -q` under pipefail: grep exits early and can SIGPIPE
# strings, making a successful match look like a failed validation.
if ! grep -a -q "apple-macosx${METALLIB_TARGET_VERSION}" "${BUILT_METALLIB}"; then
    echo "metallib deployment target mismatch: expected apple-macosx${METALLIB_TARGET_VERSION}" >&2
    exit 1
fi

# 3. Stage native artifacts into the package so uv_build includes them.
echo "==> Staging native artifacts…"
mkdir -p src/unlimitedocr_c/lib src/unlimitedocr_c/metal
cp "${BUILT_DYLIB}" src/unlimitedocr_c/lib/libunlimitedocr.dylib
cp "${BUILT_METALLIB}" src/unlimitedocr_c/metal/unlimitedocr.metallib

# 4. Read version from pyproject.toml.
VERSION="$(uv run python -c "import tomllib,sys; print(tomllib.load(sys.stdin.buffer)['project']['version'])" < pyproject.toml)"
echo "==> Package version: ${VERSION}"

# 5. Build the wheel.
echo "==> Building wheel…"
rm -rf dist/*.whl dist/*.tar.gz
uv build --wheel

# 6. Retag the wheel as macOS arm64 (uv_build emits py3-none-any because the
#    dylib is packaged as data, not as a Python extension module). The validation
#    above makes this retag explicit and checked instead of optimistic.
SRC_WHL="dist/unlimitedocr_c-${VERSION}-py3-none-any.whl"
DST_WHL="dist/unlimitedocr_c-${VERSION}-py3-none-${WHEEL_PLATFORM_TAG}.whl"

uv run python - <<PY "${SRC_WHL}" "${DST_WHL}" "${WHEEL_PLATFORM_TAG}"
import base64, csv, hashlib, io, sys, zipfile
from pathlib import Path
import shutil, tempfile

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
platform_tag = sys.argv[3]
tmp = Path(tempfile.mkdtemp())
try:
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)
    dist_info = next(tmp.glob("*.dist-info"))
    wheel = dist_info / "WHEEL"
    wheel_text = wheel.read_text()
    expected = "Tag: py3-none-any"
    replacement = f"Tag: py3-none-{platform_tag}"
    if expected not in wheel_text:
        raise SystemExit(f"expected wheel tag not found: {expected}")
    wheel.write_text(wheel_text.replace(expected, replacement))

    required = [
        "unlimitedocr_c/lib/libunlimitedocr.dylib",
        "unlimitedocr_c/metal/unlimitedocr.metallib",
    ]
    missing = [rel for rel in required if not (tmp / rel).is_file()]
    if missing:
        raise SystemExit("wheel is missing required artifacts: " + ", ".join(missing))

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

# 7. Publish to PyPI.
echo "==> Publishing to PyPI…"
uv publish "${DST_WHL}"

echo "==> Published ${DST_WHL}"
