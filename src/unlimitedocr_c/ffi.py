"""ctypes binding for the narrow `libunlimitedocr` C ABI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import os
import sys
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .frontend import PreparedRequest, project_root

UOCR_OK = 0
UOCR_ERROR_INVALID_ARGUMENT = -1
UOCR_ERROR_UNSUPPORTED = -2
UOCR_ERROR_OUT_OF_MEMORY = -3
UOCR_ERROR_NOT_IMPLEMENTED = -4
UOCR_ERROR_INTERNAL = -5

UOCR_PIXEL_F16_NCHW = 0
UOCR_PIXEL_F32_NCHW = 1
UOCR_VIEW_GLOBAL = 0
UOCR_VIEW_LOCAL = 1

UOCR_MEMORY_MODEL_VIEWS = 0
UOCR_MEMORY_KV_CACHE = 1
UOCR_MEMORY_PROMPT_EMBEDDINGS = 2
UOCR_MEMORY_VISION_SCRATCH = 3
UOCR_MEMORY_DECODER_SCRATCH = 4
UOCR_MEMORY_MOE_SCRATCH = 5
UOCR_MEMORY_LOGITS_READBACK = 6
UOCR_MEMORY_TRANSIENT_BUFFERS = 7
UOCR_MEMORY_CATEGORY_COUNT = 8


class CMemoryReport(ct.Structure):
    _fields_ = [
        ("category_live_bytes", ct.c_uint64 * UOCR_MEMORY_CATEGORY_COUNT),
        ("category_peak_bytes", ct.c_uint64 * UOCR_MEMORY_CATEGORY_COUNT),
        ("total_live_bytes", ct.c_uint64),
        ("total_peak_bytes", ct.c_uint64),
        ("estimated_model_views_bytes", ct.c_uint64),
        ("estimated_kv_cache_bytes", ct.c_uint64),
        ("estimated_prompt_embeddings_bytes", ct.c_uint64),
        ("estimated_vision_scratch_bytes", ct.c_uint64),
        ("estimated_decoder_scratch_bytes", ct.c_uint64),
        ("estimated_moe_scratch_bytes", ct.c_uint64),
        ("estimated_logits_readback_bytes", ct.c_uint64),
        ("estimated_transient_bytes", ct.c_uint64),
        ("estimated_safety_margin_bytes", ct.c_uint64),
        ("estimated_total_bytes", ct.c_uint64),
        ("memory_budget_bytes", ct.c_uint64),
        ("recommended_working_set_bytes", ct.c_uint64),
    ]


class CImageView(ct.Structure):
    _fields_ = [
        ("pixels", ct.c_void_p),
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("format", ct.c_int),
        ("kind", ct.c_int),
    ]


class CPreparedRequest(ct.Structure):
    _fields_ = [
        ("input_ids", ct.POINTER(ct.c_int32)),
        ("image_mask", ct.POINTER(ct.c_uint8)),
        ("n_tokens", ct.c_uint32),
        ("views", ct.POINTER(CImageView)),
        ("n_views", ct.c_uint32),
        ("crop_grid_w", ct.c_uint32),
        ("crop_grid_h", ct.c_uint32),
        ("max_new_tokens", ct.c_uint32),
        ("no_repeat_ngram_size", ct.c_uint32),
        ("no_repeat_window", ct.c_uint32),
    ]


class CEngineOpts(ct.Structure):
    _fields_ = [
        ("model_path", ct.c_char_p),
        ("backend", ct.c_char_p),
        ("resource_path", ct.c_char_p),
        ("max_batch", ct.c_uint32),
        ("max_prompt_tokens", ct.c_uint32),
        ("max_gen_tokens", ct.c_uint32),
        ("memory_budget_bytes", ct.c_uint64),
    ]


@dataclass(frozen=True)
class MemoryReport:
    category_live_bytes: tuple[int, ...]
    category_peak_bytes: tuple[int, ...]
    total_live_bytes: int
    total_peak_bytes: int
    estimated_model_views_bytes: int
    estimated_kv_cache_bytes: int
    estimated_prompt_embeddings_bytes: int
    estimated_vision_scratch_bytes: int
    estimated_decoder_scratch_bytes: int
    estimated_moe_scratch_bytes: int
    estimated_logits_readback_bytes: int
    estimated_transient_bytes: int
    estimated_safety_margin_bytes: int
    estimated_total_bytes: int
    memory_budget_bytes: int
    recommended_working_set_bytes: int


@dataclass(frozen=True)
class EngineOptions:
    model_path: str | None = None
    backend: str = "cpu-ref"
    resource_path: str | None = None
    max_batch: int = 1
    max_prompt_tokens: int = 4096
    max_gen_tokens: int = 512
    memory_budget_bytes: int = 0


@dataclass
class _PreparedKeepalive:
    input_ids: NDArray[np.int32]
    image_mask: NDArray[np.uint8]
    view_pixels: tuple[NDArray[np.float16] | NDArray[np.float32], ...]
    c_views: ct.Array[CImageView] | None
    c_request: CPreparedRequest


def _shared_library_names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libunlimitedocr.dylib",)
    if os.name == "nt":
        return ("unlimitedocr.dll", "libunlimitedocr.dll")
    return ("libunlimitedocr.so",)


def candidate_library_paths() -> list[Path]:
    env = os.environ.get("UOCR_LIBRARY_PATH")
    candidates: list[Path] = [Path(env)] if env else []
    root = project_root()
    for build_dir in ("debug", "release", "relwithdebinfo", "cpu"):
        for name in _shared_library_names():
            candidates.append(root / "build" / build_dir / name)
    package_lib = Path(__file__).resolve().parent / "lib"
    for name in _shared_library_names():
        candidates.append(package_lib / name)
    return candidates


def find_library_path() -> Path:
    for candidate in candidate_library_paths():
        if candidate.exists():
            return candidate
    formatted = "\n  ".join(str(path) for path in candidate_library_paths())
    raise FileNotFoundError(
        "could not find libunlimitedocr; set UOCR_LIBRARY_PATH or build the native target. Tried:\n  " + formatted
    )


def _bind_library(path: Path) -> ct.CDLL:
    lib = ct.CDLL(str(path))
    lib.uocr_abi_version.argtypes = []
    lib.uocr_abi_version.restype = ct.c_uint32
    lib.uocr_status_string.argtypes = [ct.c_int]
    lib.uocr_status_string.restype = ct.c_char_p
    lib.uocr_engine_open.argtypes = [ct.POINTER(CEngineOpts)]
    lib.uocr_engine_open.restype = ct.c_void_p
    lib.uocr_engine_close.argtypes = [ct.c_void_p]
    lib.uocr_engine_close.restype = None
    lib.uocr_last_error.argtypes = [ct.c_void_p]
    lib.uocr_last_error.restype = ct.c_char_p
    lib.uocr_engine_backend.argtypes = [ct.c_void_p]
    lib.uocr_engine_backend.restype = ct.c_char_p
    lib.uocr_memory_category_name.argtypes = [ct.c_int]
    lib.uocr_memory_category_name.restype = ct.c_char_p
    lib.uocr_engine_memory_report.argtypes = [ct.c_void_p, ct.POINTER(CMemoryReport)]
    lib.uocr_engine_memory_report.restype = ct.c_int
    lib.uocr_generate_prepared.argtypes = [ct.c_void_p, ct.POINTER(CPreparedRequest), ct.c_uint32, ct.POINTER(ct.c_void_p)]
    lib.uocr_generate_prepared.restype = ct.c_int
    lib.uocr_result_count.argtypes = [ct.c_void_p]
    lib.uocr_result_count.restype = ct.c_uint32
    lib.uocr_result_tokens.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint32)]
    lib.uocr_result_tokens.restype = ct.POINTER(ct.c_int32)
    lib.uocr_result_free.argtypes = [ct.c_void_p]
    lib.uocr_result_free.restype = None
    return lib


_LIB: ct.CDLL | None = None


def load_library(path: str | Path | None = None) -> ct.CDLL:
    global _LIB
    if path is not None:
        return _bind_library(Path(path))
    if _LIB is None:
        _LIB = _bind_library(find_library_path())
    return _LIB


def _bytes_or_null(value: str | None) -> bytes | None:
    return value.encode("utf-8") if value else None


def _pixel_format(format_name: str) -> int:
    if format_name == "f16_nchw":
        return UOCR_PIXEL_F16_NCHW
    if format_name == "f32_nchw":
        return UOCR_PIXEL_F32_NCHW
    raise ValueError(f"unsupported pixel format {format_name!r}")


def _view_kind(kind: str) -> int:
    if kind == "global":
        return UOCR_VIEW_GLOBAL
    if kind == "local":
        return UOCR_VIEW_LOCAL
    raise ValueError(f"unsupported view kind {kind!r}")


def _copy_result_tokens(lib: ct.CDLL, result: ct.c_void_p) -> list[NDArray[np.int32]]:
    count = lib.uocr_result_count(result)
    outputs: list[NDArray[np.int32]] = []
    for i in range(count):
        n_tokens = ct.c_uint32()
        ptr = lib.uocr_result_tokens(result, i, ct.byref(n_tokens))
        if n_tokens.value == 0 or not ptr:
            outputs.append(np.empty((0,), dtype=np.int32))
        else:
            array = np.ctypeslib.as_array(ptr, shape=(n_tokens.value,)).copy()
            outputs.append(array.astype(np.int32, copy=False))
    return outputs


def as_c_request(request: PreparedRequest) -> _PreparedKeepalive:
    input_ids = np.ascontiguousarray(request.input_ids, dtype=np.int32)
    image_mask = np.ascontiguousarray(request.image_mask, dtype=np.uint8)
    if input_ids.ndim != 1 or image_mask.ndim != 1 or input_ids.shape[0] != image_mask.shape[0]:
        raise ValueError("input_ids and image_mask must be same-length 1D arrays")

    view_pixels: list[NDArray[np.float16] | NDArray[np.float32]] = []
    c_views: ct.Array[CImageView] | None = None
    if request.views:
        c_views = (CImageView * len(request.views))()
        for i, view in enumerate(request.views):
            pixels = np.ascontiguousarray(view.pixels)
            view_pixels.append(pixels)
            c_views[i] = CImageView(
                pixels=ct.c_void_p(int(pixels.ctypes.data)),
                width=view.width,
                height=view.height,
                format=_pixel_format(view.format),
                kind=_view_kind(view.kind),
            )

    c_request = CPreparedRequest(
        input_ids=input_ids.ctypes.data_as(ct.POINTER(ct.c_int32)),
        image_mask=image_mask.ctypes.data_as(ct.POINTER(ct.c_uint8)),
        n_tokens=ct.c_uint32(input_ids.shape[0]),
        views=ct.cast(c_views, ct.POINTER(CImageView)) if c_views is not None else None,
        n_views=ct.c_uint32(len(request.views)),
        crop_grid_w=ct.c_uint32(request.crop_grid_w),
        crop_grid_h=ct.c_uint32(request.crop_grid_h),
        max_new_tokens=ct.c_uint32(request.max_new_tokens),
        no_repeat_ngram_size=ct.c_uint32(request.no_repeat_ngram_size),
        no_repeat_window=ct.c_uint32(request.no_repeat_window),
    )
    return _PreparedKeepalive(input_ids, image_mask, tuple(view_pixels), c_views, c_request)


class Engine:
    def __init__(self, options: EngineOptions | None = None, *, library_path: str | Path | None = None) -> None:
        self._handle: ct.c_void_p | None = None
        self._lib = load_library(library_path)
        opts = options or EngineOptions()
        c_opts = CEngineOpts(
            model_path=_bytes_or_null(opts.model_path),
            backend=_bytes_or_null(opts.backend),
            resource_path=_bytes_or_null(opts.resource_path),
            max_batch=opts.max_batch,
            max_prompt_tokens=opts.max_prompt_tokens,
            max_gen_tokens=opts.max_gen_tokens,
            memory_budget_bytes=opts.memory_budget_bytes,
        )
        self._handle = self._lib.uocr_engine_open(ct.byref(c_opts))
        if not self._handle:
            raise RuntimeError(self.last_error(global_error=True))

    @property
    def backend(self) -> str:
        raw = self._lib.uocr_engine_backend(self._handle)
        return raw.decode("utf-8") if raw else ""

    def memory_category_name(self, category: int) -> str:
        raw = self._lib.uocr_memory_category_name(category)
        return raw.decode("utf-8") if raw else ""

    def memory_report(self) -> MemoryReport:
        report = CMemoryReport()
        status = self._lib.uocr_engine_memory_report(self._handle, ct.byref(report))
        if status != UOCR_OK:
            raise RuntimeError(f"uocr_engine_memory_report failed ({status}): {self.last_error()}")
        return MemoryReport(
            category_live_bytes=tuple(int(report.category_live_bytes[i]) for i in range(UOCR_MEMORY_CATEGORY_COUNT)),
            category_peak_bytes=tuple(int(report.category_peak_bytes[i]) for i in range(UOCR_MEMORY_CATEGORY_COUNT)),
            total_live_bytes=int(report.total_live_bytes),
            total_peak_bytes=int(report.total_peak_bytes),
            estimated_model_views_bytes=int(report.estimated_model_views_bytes),
            estimated_kv_cache_bytes=int(report.estimated_kv_cache_bytes),
            estimated_prompt_embeddings_bytes=int(report.estimated_prompt_embeddings_bytes),
            estimated_vision_scratch_bytes=int(report.estimated_vision_scratch_bytes),
            estimated_decoder_scratch_bytes=int(report.estimated_decoder_scratch_bytes),
            estimated_moe_scratch_bytes=int(report.estimated_moe_scratch_bytes),
            estimated_logits_readback_bytes=int(report.estimated_logits_readback_bytes),
            estimated_transient_bytes=int(report.estimated_transient_bytes),
            estimated_safety_margin_bytes=int(report.estimated_safety_margin_bytes),
            estimated_total_bytes=int(report.estimated_total_bytes),
            memory_budget_bytes=int(report.memory_budget_bytes),
            recommended_working_set_bytes=int(report.recommended_working_set_bytes),
        )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.uocr_engine_close(handle)
            self._handle = None

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def __del__(self) -> None:
        self.close()

    def last_error(self, *, global_error: bool = False) -> str:
        handle = None if global_error else self._handle
        raw = self._lib.uocr_last_error(handle)
        return raw.decode("utf-8") if raw else ""

    def generate_prepared(self, requests: PreparedRequest | Sequence[PreparedRequest]) -> list[NDArray[np.int32]]:
        if isinstance(requests, PreparedRequest):
            request_list = [requests]
        else:
            request_list = list(requests)
        if not request_list:
            raise ValueError("at least one prepared request is required")

        keepalive = [as_c_request(req) for req in request_list]
        c_requests = (CPreparedRequest * len(keepalive))(*(item.c_request for item in keepalive))
        result = ct.c_void_p()
        status = self._lib.uocr_generate_prepared(self._handle, c_requests, len(keepalive), ct.byref(result))
        if status != UOCR_OK:
            raise RuntimeError(f"uocr_generate_prepared failed ({status}): {self.last_error()}")
        try:
            return _copy_result_tokens(self._lib, result)
        finally:
            self._lib.uocr_result_free(result)
