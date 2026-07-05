#!/usr/bin/env bash
set -euo pipefail

# Keep this script aligned with probes/e2e_generation_probe.py, which loads
# build/release/libunlimitedocr.dylib by default when running from the source tree.
BUILD_DIR="build/release"
UOCR_ENABLE_PROFILING="${UOCR_ENABLE_PROFILING:-ON}"

cmake -S . -B "${BUILD_DIR}" \
  -DUOCR_ENABLE_PROFILING="${UOCR_ENABLE_PROFILING}" \
  -DUOCR_METAL_PRECOMPILE=ON
cmake --build "${BUILD_DIR}" "$@"
