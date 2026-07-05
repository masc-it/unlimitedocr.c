#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build/release}"
UOCR_ENABLE_PROFILING="${UOCR_ENABLE_PROFILING:-ON}"

cmake -S . -B "${BUILD_DIR}" -DUOCR_ENABLE_PROFILING="${UOCR_ENABLE_PROFILING}"
cmake --build "${BUILD_DIR}" "$@"
