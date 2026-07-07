"""Opt-in Metal parity helpers.

These helpers are intentionally developer-facing rather than part of the stable
OCR API.  They keep the Python-dumped image-embedding path available after the
public Metal vision path exists, so decoder regressions can be isolated from
SAM/CLIP/projector regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import json
import os

import numpy as np
from numpy.typing import NDArray

from .ffi import UOCR_OK, load_library
from .frontend import project_root
from .golden import (
    GENERATED_IDS_BIN,
    HIDDEN_SIZE,
    MODEL_VOCAB_SIZE,
    ImagePromptEmbeddingDump,
    load_image_prompt_embedding_dump,
)


class CModelFile(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("size", ct.c_size_t),
        ("owns_mapping", ct.c_int),
    ]
    if os.name != "nt":
        _fields_.append(("fd", ct.c_int))
    _fields_.extend(
        [
            ("header", ct.c_void_p),
            ("sections", ct.c_void_p),
            ("config", ct.c_void_p),
            ("tokenizer_metadata", ct.c_void_p),
            ("tokenizer_payload", ct.c_void_p),
            ("tokenizer_payload_size", ct.c_size_t),
            ("provenance", ct.c_void_p),
            ("provenance_json", ct.c_void_p),
            ("provenance_json_size", ct.c_size_t),
            ("tensor_directory", ct.c_void_p),
            ("tensors", ct.c_void_p),
            ("tensor_count", ct.c_uint32),
        ]
    )


class CMetalDecoderRequestF16(ct.Structure):
    _fields_ = [
        ("input_ids", ct.POINTER(ct.c_int32)),
        ("image_mask", ct.POINTER(ct.c_uint8)),
        ("image_features_f16", ct.POINTER(ct.c_uint16)),
        ("n_tokens", ct.c_uint32),
        ("max_new_tokens", ct.c_uint32),
        ("slot", ct.c_uint32),
        ("image_span_start", ct.c_uint32),
        ("image_span_length", ct.c_uint32),
        ("reserved0", ct.c_uint32),
    ]


class CMetalDecoderResultF16(ct.Structure):
    _fields_ = [
        ("generated_ids", ct.POINTER(ct.c_int32)),
        ("generated_scores_f32_or_null", ct.POINTER(ct.c_float)),
        ("generated_capacity", ct.c_uint32),
        ("generated_count", ct.c_uint32),
        ("stopped_on_eos", ct.c_uint32),
        ("last_token_id", ct.c_uint32),
        ("last_score_f32", ct.c_float),
        ("reserved0", ct.c_uint32),
    ]


@dataclass(frozen=True)
class DumpedImageEmbeddingGenerationResult:
    """Output from the dumped-image-embedding Metal decoder parity path."""

    generated_ids: NDArray[np.int32]
    generated_scores: NDArray[np.float32]
    stopped_on_eos: bool
    last_token_id: int | None
    last_score: float | None


def default_metal_resource_path() -> Path:
    return project_root() / "src" / "backend" / "metal"


def _bind_internal_metal_symbols(lib: ct.CDLL) -> None:
    required = [
        "uocr_metal_is_available",
        "uocr_model_file_open",
        "uocr_model_file_close",
        "uocr_metal_context_create",
        "uocr_metal_context_destroy",
        "uocr_metal_context_map_model",
        "uocr_metal_context_allocate_runtime_arenas",
        "uocr_metal_context_generate_f16",
    ]
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        raise RuntimeError("libunlimitedocr is missing internal Metal parity symbols: " + ", ".join(missing))

    lib.uocr_metal_is_available.argtypes = []
    lib.uocr_metal_is_available.restype = ct.c_int
    lib.uocr_model_file_open.argtypes = [ct.c_char_p, ct.POINTER(CModelFile), ct.c_char_p, ct.c_size_t]
    lib.uocr_model_file_open.restype = ct.c_int
    lib.uocr_model_file_close.argtypes = [ct.POINTER(CModelFile)]
    lib.uocr_model_file_close.restype = None
    lib.uocr_metal_context_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.uocr_metal_context_create.restype = ct.c_void_p
    lib.uocr_metal_context_destroy.argtypes = [ct.c_void_p]
    lib.uocr_metal_context_destroy.restype = None
    lib.uocr_metal_context_map_model.argtypes = [ct.c_void_p, ct.POINTER(CModelFile), ct.c_char_p, ct.c_size_t]
    lib.uocr_metal_context_map_model.restype = ct.c_int
    lib.uocr_metal_context_allocate_runtime_arenas.argtypes = [
        ct.c_void_p,
        ct.c_uint32,
        ct.c_uint32,
        ct.c_char_p,
        ct.c_size_t,
    ]
    lib.uocr_metal_context_allocate_runtime_arenas.restype = ct.c_int
    lib.uocr_metal_context_generate_f16.argtypes = [
        ct.c_void_p,
        ct.POINTER(CMetalDecoderRequestF16),
        ct.POINTER(CMetalDecoderResultF16),
        ct.c_char_p,
        ct.c_size_t,
    ]
    lib.uocr_metal_context_generate_f16.restype = ct.c_int


def _load_manifest(root: Path) -> dict[str, object]:
    with (root / "manifest.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {root / 'manifest.json'}")
    return data


def load_dumped_image_embedding_expected_ids(fixture_dir: str | Path) -> NDArray[np.int32]:
    """Load generated-id expectations without requiring layer/logit dumps."""

    root = Path(fixture_dir)
    manifest = _load_manifest(root)
    golden = manifest.get("golden_tensors", {})
    generated = golden.get("generated_ids", {}) if isinstance(golden, dict) else {}
    if not isinstance(generated, dict):
        raise ValueError(f"missing generated_ids metadata in {root}")
    shape = generated.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"invalid generated_ids shape metadata in {root}")
    n_ids = int(shape[0])
    filename = generated.get("file", GENERATED_IDS_BIN)
    if not isinstance(filename, str):
        raise ValueError(f"invalid generated_ids filename in {root}")
    ids = np.fromfile(root / filename, dtype=np.dtype("<i4"))
    if ids.shape != (n_ids,):
        raise ValueError(f"generated_ids shape mismatch in {root}: ids={ids.shape}, expected {(n_ids,)}")
    if n_ids <= 0 or np.any(ids < 0) or np.any(ids >= MODEL_VOCAB_SIZE):
        raise ValueError(f"invalid generated_ids in {root}")
    return np.ascontiguousarray(ids, dtype=np.int32)


def _validate_dump_for_metal(dump: ImagePromptEmbeddingDump) -> None:
    if dump.input_ids.ndim != 1 or dump.image_mask.shape != dump.input_ids.shape:
        raise ValueError("dump input_ids/image_mask must be same-length 1D arrays")
    if dump.visual_features_f16_bits.ndim != 2 or dump.visual_features_f16_bits.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"dumped visual features must have shape [image_tokens,{HIDDEN_SIZE}], "
            f"got {dump.visual_features_f16_bits.shape}"
        )
    if dump.image_span_length <= 0 or dump.image_span_length != dump.visual_features_f16_bits.shape[0]:
        raise ValueError("dumped visual feature rows must match the image span length")
    if dump.image_span_start < 0 or dump.image_span_start + dump.image_span_length > int(dump.input_ids.size):
        raise ValueError("dumped image span is outside the prompt")
    image_mask_count = int(np.count_nonzero(dump.image_mask))
    if image_mask_count != dump.image_span_length:
        raise ValueError(f"dump image mask has {image_mask_count} placeholders, expected {dump.image_span_length}")


def generate_from_dumped_image_embeddings(
    fixture_dir: str | Path,
    *,
    model_path: str | Path,
    resource_path: str | Path | None = None,
    max_new_tokens: int | None = None,
    library_path: str | Path | None = None,
) -> DumpedImageEmbeddingGenerationResult:
    """Run Metal fp16 generation using Python-dumped final visual features.

    This bypasses the C/Metal vision encoder and feeds ``visual_features_f16.bin``
    from ``fixture_dir`` into the same integrated decoder used by public image
    generation.  It is meant for opt-in parity debugging, not for normal OCR.
    """

    root = Path(fixture_dir)
    dump = load_image_prompt_embedding_dump(root)
    _validate_dump_for_metal(dump)
    if max_new_tokens is None:
        max_new_tokens = int(load_dumped_image_embedding_expected_ids(root).size)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    input_ids = np.ascontiguousarray(dump.input_ids, dtype=np.int32)
    image_mask = np.ascontiguousarray(dump.image_mask, dtype=np.uint8)
    visual = np.ascontiguousarray(dump.visual_features_f16_bits, dtype=np.dtype("<u2"))
    generated = np.zeros((int(max_new_tokens),), dtype=np.int32)
    scores = np.zeros((int(max_new_tokens),), dtype=np.float32)

    lib = load_library(library_path)
    _bind_internal_metal_symbols(lib)
    if lib.uocr_metal_is_available() == 0:
        raise RuntimeError("Metal device is not available")

    resolved_resource_path = Path(resource_path) if resource_path is not None else default_metal_resource_path()
    error = ct.create_string_buffer(2048)
    model = CModelFile()
    model_open = False
    ctx: ct.c_void_p | None = None
    try:
        status = lib.uocr_model_file_open(str(model_path).encode("utf-8"), ct.byref(model), error, len(error))
        if status != UOCR_OK:
            raise RuntimeError(f"uocr_model_file_open failed ({status}): {error.value.decode('utf-8', errors='replace')}")
        model_open = True

        ctx = lib.uocr_metal_context_create(str(resolved_resource_path).encode("utf-8"), error, len(error))
        if not ctx:
            raise RuntimeError(f"uocr_metal_context_create failed: {error.value.decode('utf-8', errors='replace')}")
        if lib.uocr_metal_context_map_model(ctx, ct.byref(model), error, len(error)) != 1:
            raise RuntimeError(f"uocr_metal_context_map_model failed: {error.value.decode('utf-8', errors='replace')}")
        if lib.uocr_metal_context_allocate_runtime_arenas(ctx, 1, int(input_ids.size), error, len(error)) != 1:
            raise RuntimeError(
                f"uocr_metal_context_allocate_runtime_arenas failed: {error.value.decode('utf-8', errors='replace')}"
            )

        request = CMetalDecoderRequestF16(
            input_ids=input_ids.ctypes.data_as(ct.POINTER(ct.c_int32)),
            image_mask=image_mask.ctypes.data_as(ct.POINTER(ct.c_uint8)),
            image_features_f16=visual.ctypes.data_as(ct.POINTER(ct.c_uint16)),
            n_tokens=ct.c_uint32(input_ids.size),
            max_new_tokens=ct.c_uint32(max_new_tokens),
            slot=ct.c_uint32(0),
            image_span_start=ct.c_uint32(dump.image_span_start),
            image_span_length=ct.c_uint32(dump.image_span_length),
            reserved0=ct.c_uint32(0),
        )
        result = CMetalDecoderResultF16(
            generated_ids=generated.ctypes.data_as(ct.POINTER(ct.c_int32)),
            generated_scores_f32_or_null=scores.ctypes.data_as(ct.POINTER(ct.c_float)),
            generated_capacity=ct.c_uint32(max_new_tokens),
            generated_count=ct.c_uint32(0),
            stopped_on_eos=ct.c_uint32(0),
            last_token_id=ct.c_uint32(0),
            last_score_f32=ct.c_float(0.0),
            reserved0=ct.c_uint32(0),
        )
        if lib.uocr_metal_context_generate_f16(ctx, ct.byref(request), ct.byref(result), error, len(error)) != 1:
            raise RuntimeError(f"uocr_metal_context_generate_f16 failed: {error.value.decode('utf-8', errors='replace')}")

        count = int(result.generated_count)
        return DumpedImageEmbeddingGenerationResult(
            generated_ids=np.ascontiguousarray(generated[:count], dtype=np.int32),
            generated_scores=np.ascontiguousarray(scores[:count], dtype=np.float32),
            stopped_on_eos=bool(result.stopped_on_eos),
            last_token_id=int(result.last_token_id) if count else None,
            last_score=float(result.last_score_f32) if count else None,
        )
    finally:
        if ctx:
            lib.uocr_metal_context_destroy(ctx)
        if model_open:
            lib.uocr_model_file_close(ct.byref(model))
