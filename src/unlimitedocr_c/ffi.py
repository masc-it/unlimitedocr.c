"""ctypes binding for the narrow `libunlimitedocr` C ABI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import os
import sys
import threading
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .frontend import GLOBAL_VIEW_SIZE, LOCAL_VIEW_SIZE, PreparedRequest, SINGLE_PROMPT, default_tokenizer_path, is_source_tree_package, prepare_image, project_root

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
UOCR_MEMORY_VISION_GPU_WORKSPACE = 3
UOCR_MEMORY_VISION_FINAL_FEATURES = 4
UOCR_MEMORY_VISION_HOST_STAGING = 5
UOCR_MEMORY_DECODER_SCRATCH = 6
UOCR_MEMORY_MOE_SCRATCH = 7
UOCR_MEMORY_LOGITS_READBACK = 8
UOCR_MEMORY_TRANSIENT_BUFFERS = 9
UOCR_MEMORY_CATEGORY_COUNT = 10
UOCR_MEMORY_VISION_SCRATCH = UOCR_MEMORY_VISION_GPU_WORKSPACE

UOCR_PROFILE_EVENT_NAME_SIZE = 64
UOCR_PROFILE_MAX_EVENTS = 256


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
        ("estimated_vision_gpu_workspace_bytes", ct.c_uint64),
        ("estimated_vision_final_features_bytes", ct.c_uint64),
        ("estimated_vision_host_staging_bytes", ct.c_uint64),
        ("estimated_decoder_scratch_bytes", ct.c_uint64),
        ("estimated_moe_scratch_bytes", ct.c_uint64),
        ("estimated_logits_readback_bytes", ct.c_uint64),
        ("estimated_transient_bytes", ct.c_uint64),
        ("estimated_safety_margin_bytes", ct.c_uint64),
        ("estimated_total_bytes", ct.c_uint64),
        ("memory_budget_bytes", ct.c_uint64),
        ("recommended_working_set_bytes", ct.c_uint64),
        ("vision_workspace_capacity_bytes", ct.c_uint64),
        ("vision_workspace_high_watermark_bytes", ct.c_uint64),
    ]


class CProfileEvent(ct.Structure):
    _fields_ = [
        ("name", ct.c_char * UOCR_PROFILE_EVENT_NAME_SIZE),
        ("calls", ct.c_uint64),
        ("total_ms", ct.c_double),
        ("min_ms", ct.c_double),
        ("max_ms", ct.c_double),
    ]


class CProfileReport(ct.Structure):
    _fields_ = [
        ("enabled", ct.c_uint32),
        ("event_count", ct.c_uint32),
        ("dropped_event_count", ct.c_uint32),
        ("reserved0", ct.c_uint32),
        ("generation_index", ct.c_uint64),
        ("events", CProfileEvent * UOCR_PROFILE_MAX_EVENTS),
        ("metal_buffer_allocation_count", ct.c_uint64),
        ("metal_buffer_allocation_bytes", ct.c_uint64),
        ("metal_command_buffer_count", ct.c_uint64),
        ("metal_command_encoder_count", ct.c_uint64),
        ("metal_command_buffer_wait_count", ct.c_uint64),
        ("metal_mps_descriptor_count", ct.c_uint64),
        ("metal_mps_ndarray_count", ct.c_uint64),
        ("metal_nsarray_count", ct.c_uint64),
        ("metal_transient_retain_object_count", ct.c_uint64),
        ("memory", CMemoryReport),
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
    ]


class CEngineOpts(ct.Structure):
    _fields_ = [
        ("model_path", ct.c_char_p),
        ("backend", ct.c_char_p),
        ("resource_path", ct.c_char_p),
        ("max_batch", ct.c_uint32),
        ("max_prompt_tokens", ct.c_uint32),
        ("max_gen_tokens", ct.c_uint32),
        ("profile", ct.c_uint32),
        ("reserved0", ct.c_uint32),
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
    estimated_vision_gpu_workspace_bytes: int
    estimated_vision_final_features_bytes: int
    estimated_vision_host_staging_bytes: int
    estimated_decoder_scratch_bytes: int
    estimated_moe_scratch_bytes: int
    estimated_logits_readback_bytes: int
    estimated_transient_bytes: int
    estimated_safety_margin_bytes: int
    estimated_total_bytes: int
    memory_budget_bytes: int
    recommended_working_set_bytes: int
    vision_workspace_capacity_bytes: int = 0
    vision_workspace_high_watermark_bytes: int = 0


@dataclass(frozen=True)
class ProfileEvent:
    name: str
    calls: int
    total_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "name": self.name,
            "calls": self.calls,
            "total_ms": self.total_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True)
class ProfileReport:
    enabled: bool
    generation_index: int
    events: tuple[ProfileEvent, ...]
    dropped_event_count: int
    metal_buffer_allocation_count: int
    metal_buffer_allocation_bytes: int
    metal_command_buffer_count: int
    metal_command_encoder_count: int
    metal_command_buffer_wait_count: int
    metal_mps_descriptor_count: int
    metal_mps_ndarray_count: int
    metal_nsarray_count: int
    metal_transient_retain_object_count: int
    memory: MemoryReport

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "generation_index": self.generation_index,
            "events": [event.as_dict() for event in self.events],
            "dropped_event_count": self.dropped_event_count,
            "metal_buffer_allocation_count": self.metal_buffer_allocation_count,
            "metal_buffer_allocation_bytes": self.metal_buffer_allocation_bytes,
            "metal_command_buffer_count": self.metal_command_buffer_count,
            "metal_command_encoder_count": self.metal_command_encoder_count,
            "metal_command_buffer_wait_count": self.metal_command_buffer_wait_count,
            "metal_mps_descriptor_count": self.metal_mps_descriptor_count,
            "metal_mps_ndarray_count": self.metal_mps_ndarray_count,
            "metal_nsarray_count": self.metal_nsarray_count,
            "metal_transient_retain_object_count": self.metal_transient_retain_object_count,
            "memory": self.memory.__dict__,
        }

    def summary(self, *, limit: int = 12) -> str:
        ordered = sorted(self.events, key=lambda event: event.total_ms, reverse=True)
        event_bits = ", ".join(f"{event.name}={event.total_ms:.3f}ms/{event.calls}" for event in ordered[:limit])
        return (
            f"profile enabled={self.enabled} generation={self.generation_index} "
            f"events=[{event_bits}] metal_buffers={self.metal_buffer_allocation_count} "
            f"metal_commands={self.metal_command_buffer_count} encoders={self.metal_command_encoder_count} "
            f"peak_memory={self.memory.total_peak_bytes}B"
        )


@dataclass(frozen=True)
class EngineOptions:
    model_path: str | None = None
    backend: str = "cpu-ref"
    resource_path: str | None = None
    max_batch: int = 1
    max_prompt_tokens: int = 4096
    max_gen_tokens: int = 512
    memory_budget_bytes: int = 0
    profile: bool = False
    warmup: bool = True


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


def _release_library_dir() -> Path:
    return project_root() / "build" / "release"


def _release_library_paths() -> list[Path]:
    release_dir = _release_library_dir()
    return [release_dir / name for name in _shared_library_names()]


def _is_source_tree_release_library(path: Path) -> bool:
    release_dir = _release_library_dir().resolve(strict=False)
    return path.expanduser().resolve(strict=False).parent == release_dir


def _require_source_tree_release_library(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if is_source_tree_package() and not _is_source_tree_release_library(expanded):
        expected = ", ".join(str(candidate) for candidate in _release_library_paths())
        raise FileNotFoundError(f"{label} must point to the Release native library ({expected})")
    return expanded


def candidate_library_paths() -> list[Path]:
    env = os.environ.get("UOCR_LIBRARY_PATH")
    if env:
        return [_require_source_tree_release_library(Path(env), "UOCR_LIBRARY_PATH")]

    candidates: list[Path] = _release_library_paths()
    if not is_source_tree_package():
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
    lib.uocr_engine_profile_report.argtypes = [ct.c_void_p, ct.POINTER(CProfileReport)]
    lib.uocr_engine_profile_report.restype = ct.c_int
    lib.uocr_engine_profile_reset.argtypes = [ct.c_void_p]
    lib.uocr_engine_profile_reset.restype = ct.c_int
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
_LIB_LOCK = threading.Lock()


def load_library(path: str | Path | None = None) -> ct.CDLL:
    global _LIB
    if path is not None:
        return _bind_library(_require_source_tree_release_library(Path(path), "library_path"))
    with _LIB_LOCK:
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


def _validate_c_contiguous_array(name: str, array: np.ndarray, dtype: np.dtype[np.generic], ndim: int) -> None:
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous; v1 C ABI does not accept strides")


def _validate_view_pixels(request: PreparedRequest) -> None:
    for index, view in enumerate(request.views):
        pixels = view.pixels
        if view.format == "f16_nchw":
            expected_dtype = np.dtype(np.float16)
        elif view.format == "f32_nchw":
            expected_dtype = np.dtype(np.float32)
        else:
            raise ValueError(f"unsupported pixel format {view.format!r}")
        _validate_c_contiguous_array(f"view {index} pixels", pixels, expected_dtype, 3)
        expected_shape = (3, int(view.height), int(view.width))
        if pixels.shape != expected_shape:
            raise ValueError(f"view {index} pixels must have shape {expected_shape}, got {pixels.shape}")
        if view.kind == "global":
            if view.width != GLOBAL_VIEW_SIZE or view.height != GLOBAL_VIEW_SIZE:
                raise ValueError(
                    f"global view {index} must be {GLOBAL_VIEW_SIZE}x{GLOBAL_VIEW_SIZE}, got {view.width}x{view.height}"
                )
        elif view.kind == "local":
            if view.width != LOCAL_VIEW_SIZE or view.height != LOCAL_VIEW_SIZE:
                raise ValueError(
                    f"local view {index} must be {LOCAL_VIEW_SIZE}x{LOCAL_VIEW_SIZE}, got {view.width}x{view.height}"
                )
        else:
            raise ValueError(f"unsupported view kind {view.kind!r}")


def _validate_public_view_contract(request: PreparedRequest) -> None:
    _validate_view_pixels(request)
    if request.mode == "text-only":
        if request.views:
            raise ValueError("text-only requests must not include image views")
        return
    if request.mode == "base":
        if request.crop_grid_w != 1 or request.crop_grid_h != 1:
            raise ValueError(f"base image requests must use crop grid 1x1, got {request.crop_grid_w}x{request.crop_grid_h}")
        if len(request.views) != 1 or request.views[0].kind != "global":
            raise ValueError("base image requests must have exactly one global 1024x1024 view")
        return
    if request.mode == "multi-page-base":
        if request.crop_grid_w != 1 or request.crop_grid_h != 1:
            raise ValueError(
                f"multi-page base requests must use crop grid 1x1, got {request.crop_grid_w}x{request.crop_grid_h}"
            )
        if not request.views:
            raise ValueError("multi-page base requests require at least one global view")
        for index, view in enumerate(request.views):
            if view.kind != "global":
                raise ValueError(f"multi-page base view {index} must be global")
            if view.source_index != index:
                raise ValueError(f"multi-page base view {index} must have source_index {index}, got {view.source_index}")
        return
    if request.mode == "gundam":
        if request.crop_grid_w <= 0 or request.crop_grid_h <= 0:
            raise ValueError(f"gundam requests require a positive crop grid, got {request.crop_grid_w}x{request.crop_grid_h}")
        expected_locals = int(request.crop_grid_w) * int(request.crop_grid_h)
        if expected_locals == 1:
            if len(request.views) != 1 or request.views[0].kind != "global":
                raise ValueError("gundam 1x1 requests must have exactly one global 1024x1024 view")
            return
        if len(request.views) != expected_locals + 1:
            raise ValueError(
                f"gundam crop grid {request.crop_grid_w}x{request.crop_grid_h} expects "
                f"{expected_locals} local views plus one global view, got {len(request.views)} views"
            )
        for index, view in enumerate(request.views[:-1]):
            if view.kind != "local":
                raise ValueError(f"gundam crop view {index} must be local")
        if request.views[-1].kind != "global":
            raise ValueError("gundam crop final view must be the global view")


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
    input_ids = request.input_ids
    image_mask = request.image_mask
    _validate_c_contiguous_array("input_ids", input_ids, np.dtype(np.int32), 1)
    _validate_c_contiguous_array("image_mask", image_mask, np.dtype(np.uint8), 1)
    if input_ids.shape[0] != image_mask.shape[0]:
        raise ValueError("input_ids and image_mask must be same-length 1D arrays")
    _validate_public_view_contract(request)

    view_pixels: list[NDArray[np.float16] | NDArray[np.float32]] = []
    c_views: ct.Array[CImageView] | None = None
    if request.views:
        c_views = (CImageView * len(request.views))()
        for i, view in enumerate(request.views):
            pixels = view.pixels
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
    )
    return _PreparedKeepalive(input_ids, image_mask, tuple(view_pixels), c_views, c_request)


class Engine:
    """Python handle for one native `uocr_engine`.

    Thread safety: one `Engine` may be shared by several threads.  A per-object
    reentrant lock serializes every native operation (`generate_prepared`,
    reports, reset, `close`), so calls sharing this object execute one at a
    time and `close()` waits for the active call.  Native error text is read
    while the failing call still owns the lock, keeping messages paired with
    the call that produced them.  Applications that want several inference
    calls in flight create one `Engine` per execution lane.
    """

    def __init__(self, options: EngineOptions | None = None, *, library_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
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
            profile=1 if opts.profile else 0,
            reserved0=0,
            memory_budget_bytes=opts.memory_budget_bytes,
        )
        self._handle = self._lib.uocr_engine_open(ct.byref(c_opts))
        if not self._handle:
            raise RuntimeError(self.last_error(global_error=True))

        if opts.warmup and self.backend != "cpu-ref":
            self._warmup()

    def _require_open(self) -> ct.c_void_p:
        handle = self._handle
        if not handle:
            raise RuntimeError("engine is closed")
        return handle

    @property
    def backend(self) -> str:
        with self._lock:
            raw = self._lib.uocr_engine_backend(self._require_open())
            return raw.decode("utf-8") if raw else ""

    def memory_category_name(self, category: int) -> str:
        raw = self._lib.uocr_memory_category_name(category)
        return raw.decode("utf-8") if raw else ""

    @staticmethod
    def _memory_report_from_c(report: CMemoryReport) -> MemoryReport:
        return MemoryReport(
            category_live_bytes=tuple(int(report.category_live_bytes[i]) for i in range(UOCR_MEMORY_CATEGORY_COUNT)),
            category_peak_bytes=tuple(int(report.category_peak_bytes[i]) for i in range(UOCR_MEMORY_CATEGORY_COUNT)),
            total_live_bytes=int(report.total_live_bytes),
            total_peak_bytes=int(report.total_peak_bytes),
            estimated_model_views_bytes=int(report.estimated_model_views_bytes),
            estimated_kv_cache_bytes=int(report.estimated_kv_cache_bytes),
            estimated_prompt_embeddings_bytes=int(report.estimated_prompt_embeddings_bytes),
            estimated_vision_scratch_bytes=int(report.estimated_vision_scratch_bytes),
            estimated_vision_gpu_workspace_bytes=int(report.estimated_vision_gpu_workspace_bytes),
            estimated_vision_final_features_bytes=int(report.estimated_vision_final_features_bytes),
            estimated_vision_host_staging_bytes=int(report.estimated_vision_host_staging_bytes),
            estimated_decoder_scratch_bytes=int(report.estimated_decoder_scratch_bytes),
            estimated_moe_scratch_bytes=int(report.estimated_moe_scratch_bytes),
            estimated_logits_readback_bytes=int(report.estimated_logits_readback_bytes),
            estimated_transient_bytes=int(report.estimated_transient_bytes),
            estimated_safety_margin_bytes=int(report.estimated_safety_margin_bytes),
            estimated_total_bytes=int(report.estimated_total_bytes),
            memory_budget_bytes=int(report.memory_budget_bytes),
            recommended_working_set_bytes=int(report.recommended_working_set_bytes),
            vision_workspace_capacity_bytes=int(report.vision_workspace_capacity_bytes),
            vision_workspace_high_watermark_bytes=int(report.vision_workspace_high_watermark_bytes),
        )

    def memory_report(self) -> MemoryReport:
        with self._lock:
            report = CMemoryReport()
            status = self._lib.uocr_engine_memory_report(self._require_open(), ct.byref(report))
            if status != UOCR_OK:
                raise RuntimeError(f"uocr_engine_memory_report failed ({status}): {self.last_error()}")
        return self._memory_report_from_c(report)

    def profile_reset(self) -> None:
        with self._lock:
            status = self._lib.uocr_engine_profile_reset(self._require_open())
            if status != UOCR_OK:
                raise RuntimeError(f"uocr_engine_profile_reset failed ({status}): {self.last_error()}")

    def profile_report(self) -> ProfileReport:
        with self._lock:
            report = CProfileReport()
            status = self._lib.uocr_engine_profile_report(self._require_open(), ct.byref(report))
            if status != UOCR_OK:
                raise RuntimeError(f"uocr_engine_profile_report failed ({status}): {self.last_error()}")
        event_count = min(int(report.event_count), UOCR_PROFILE_MAX_EVENTS)
        events = []
        for i in range(event_count):
            event = report.events[i]
            events.append(
                ProfileEvent(
                    name=bytes(event.name).split(b"\0", 1)[0].decode("utf-8", errors="replace"),
                    calls=int(event.calls),
                    total_ms=float(event.total_ms),
                    min_ms=float(event.min_ms),
                    max_ms=float(event.max_ms),
                )
            )
        return ProfileReport(
            enabled=bool(report.enabled),
            generation_index=int(report.generation_index),
            events=tuple(events),
            dropped_event_count=int(report.dropped_event_count),
            metal_buffer_allocation_count=int(report.metal_buffer_allocation_count),
            metal_buffer_allocation_bytes=int(report.metal_buffer_allocation_bytes),
            metal_command_buffer_count=int(report.metal_command_buffer_count),
            metal_command_encoder_count=int(report.metal_command_encoder_count),
            metal_command_buffer_wait_count=int(report.metal_command_buffer_wait_count),
            metal_mps_descriptor_count=int(report.metal_mps_descriptor_count),
            metal_mps_ndarray_count=int(report.metal_mps_ndarray_count),
            metal_nsarray_count=int(report.metal_nsarray_count),
            metal_transient_retain_object_count=int(report.metal_transient_retain_object_count),
            memory=self._memory_report_from_c(report.memory),
        )

    def _warmup(self) -> None:
        """Run a synthetic generation to warm up Metal pipelines and MPS JIT kernels."""
        try:
            print("  warmup: compiling Metal pipelines...", end="", flush=True)
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (224, 224), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((80, 40), "A", fill=(0, 0, 0))

            request = prepare_image(
                img,
                prompt=SINGLE_PROMPT,
                preset="base",
                tokenizer_path=default_tokenizer_path(),
                max_new_tokens=2,
                dtype=np.float16,
            )
            _ = self.generate_prepared(request)
            self.profile_reset()
            print(" done")
        except Exception as exc:
            import warnings
            warnings.warn(f"GPU warmup failed (non-fatal): {exc}")

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
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
        with self._lock:
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
        with self._lock:
            result = ct.c_void_p()
            status = self._lib.uocr_generate_prepared(self._require_open(), c_requests, len(keepalive), ct.byref(result))
            if status != UOCR_OK:
                raise RuntimeError(f"uocr_generate_prepared failed ({status}): {self.last_error()}")
            try:
                return _copy_result_tokens(self._lib, result)
            finally:
                self._lib.uocr_result_free(result)
