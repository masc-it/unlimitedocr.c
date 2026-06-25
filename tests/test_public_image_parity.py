from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pytest

from unlimitedocr_c.ffi import (
    CImageView,
    CPreparedRequest,
    Engine,
    EngineOptions,
    UOCR_OK,
    as_c_request,
    find_library_path,
    load_library,
)
from unlimitedocr_c.frontend import (
    GLOBAL_VISUAL_TOKENS,
    ImageView,
    PreparedRequest,
    default_tokenizer_path,
    load_prepared_fixture,
    project_root,
)
from unlimitedocr_c.golden import (
    CLIP_GLOBAL_TOKENS,
    CLIP_HIDDEN_SIZE,
    CLIP_LOCAL_TOKENS,
    PROJECTED_GLOBAL_ROWS,
    PROJECTED_LOCAL_ROWS,
    GENERATED_IDS_BIN,
    GENERATED_TEXT_TXT,
    HIDDEN_SIZE,
    SAM_FEATURE_CHANNELS,
    SAM_GLOBAL_GRID,
    SAM_LOCAL_GRID,
    VISUAL_FEATURES_BIN,
    decode_generated_text,
    load_clip_features_dump,
    load_projected_features_dump,
    load_sam_features_dump,
)


def native_library_available() -> bool:
    try:
        find_library_path()
    except FileNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")


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


def _bind_internal_metal_symbols(lib: ct.CDLL) -> None:
    try:
        lib.uocr_metal_is_available
        lib.uocr_model_file_open
        lib.uocr_model_file_close
        lib.uocr_metal_context_create
        lib.uocr_metal_context_destroy
        lib.uocr_metal_context_map_model
        lib.uocr_metal_context_encode_visual_features_f16
        lib.uocr_metal_context_encode_sam_features_f16
        lib.uocr_metal_context_encode_clip_features_f16
        lib.uocr_metal_context_encode_projected_features_f16
    except AttributeError as exc:
        pytest.skip(f"Metal/internal model-file symbols are unavailable in this build: {exc}")

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
    lib.uocr_metal_context_encode_visual_features_f16.argtypes = [
        ct.c_void_p,
        ct.POINTER(CPreparedRequest),
        ct.c_uint32,
        ct.POINTER(ct.c_uint16),
        ct.c_uint32,
        ct.c_char_p,
        ct.c_size_t,
    ]
    lib.uocr_metal_context_encode_visual_features_f16.restype = ct.c_int
    lib.uocr_metal_context_encode_sam_features_f16.argtypes = [
        ct.c_void_p,
        ct.POINTER(CImageView),
        ct.POINTER(ct.c_uint16),
        ct.c_uint32,
        ct.c_uint32,
        ct.c_char_p,
        ct.c_size_t,
    ]
    lib.uocr_metal_context_encode_sam_features_f16.restype = ct.c_int
    lib.uocr_metal_context_encode_clip_features_f16.argtypes = [
        ct.c_void_p,
        ct.POINTER(CImageView),
        ct.POINTER(ct.c_uint16),
        ct.c_uint32,
        ct.c_char_p,
        ct.c_size_t,
    ]
    lib.uocr_metal_context_encode_clip_features_f16.restype = ct.c_int
    lib.uocr_metal_context_encode_projected_features_f16.argtypes = [
        ct.c_void_p,
        ct.POINTER(CImageView),
        ct.POINTER(ct.c_uint16),
        ct.c_uint32,
        ct.c_char_p,
        ct.c_size_t,
    ]
    lib.uocr_metal_context_encode_projected_features_f16.restype = ct.c_int


def _require_large_metal_inputs() -> tuple[str, str]:
    if os.environ.get("UOCR_RUN_LARGE_TESTS") != "1":
        pytest.skip("set UOCR_RUN_LARGE_TESTS=1 to run full-model Metal image parity")
    model_path = os.environ.get("UOCR_MODEL_PATH")
    if not model_path:
        pytest.skip("set UOCR_MODEL_PATH to the fp16 .uocr model for Metal image parity")
    if not Path(model_path).exists():
        pytest.fail(f"UOCR_MODEL_PATH does not exist: {model_path}")
    resource_path = os.environ.get("UOCR_METAL_RESOURCE_PATH") or str(project_root() / "src" / "backend" / "metal")
    if not Path(resource_path).exists():
        pytest.fail(f"Metal resource path does not exist: {resource_path}")
    return model_path, resource_path


def _fixture_dir_from_env(label: str, env_names: tuple[str, ...]) -> Path:
    for name in env_names:
        value = os.environ.get(name)
        if value:
            root = Path(value)
            if not root.exists():
                pytest.fail(f"{name} for {label} does not exist: {root}")
            return root
    joined = ", ".join(env_names)
    pytest.skip(f"set one of {joined} to run {label} image parity")


def _optional_fixture_dir(label: str, env_name: str) -> Path | None:
    value = os.environ.get(env_name)
    if not value:
        return None
    root = Path(value)
    if not root.exists():
        pytest.fail(f"{env_name} for {label} does not exist: {root}")
    return root


def _require_fixture_files(root: Path) -> None:
    required = ["manifest.json", "views.npz", VISUAL_FEATURES_BIN, GENERATED_IDS_BIN]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        pytest.fail(f"image parity fixture {root} is missing required files: {', '.join(missing)}")


def _tokenizer_path_from_manifest(manifest: dict[str, object]) -> str:
    raw = manifest.get("tokenizer_path")
    if isinstance(raw, str) and raw and Path(raw).exists():
        return raw
    return str(default_tokenizer_path())


def _prepared_request_from_fixture(root: Path, *, max_new_tokens: int) -> PreparedRequest:
    manifest, input_ids, image_mask, view_arrays = load_prepared_fixture(root)
    views_meta = manifest.get("views")
    if not isinstance(views_meta, list):
        raise ValueError(f"fixture {root} manifest is missing views metadata")

    views: list[ImageView] = []
    for index, meta_any in enumerate(views_meta):
        if not isinstance(meta_any, dict):
            raise ValueError(f"fixture {root} view {index} metadata is invalid")
        name = str(meta_any.get("name", f"view_{index}"))
        if name not in view_arrays:
            raise ValueError(f"fixture {root} is missing pixel array {name!r} in views.npz")
        pixels = np.ascontiguousarray(view_arrays[name])
        views.append(
            ImageView(
                pixels=pixels,
                width=int(meta_any["width"]),
                height=int(meta_any["height"]),
                kind=str(meta_any["kind"]),  # type: ignore[arg-type]
                format=str(meta_any["format"]),  # type: ignore[arg-type]
                source_index=int(meta_any.get("source_index", 0)),
            )
        )

    return PreparedRequest(
        input_ids=np.ascontiguousarray(input_ids, dtype=np.int32),
        image_mask=np.ascontiguousarray(image_mask, dtype=np.uint8),
        views=tuple(views),
        crop_grid_w=int(manifest.get("crop_grid_w", 1)),
        crop_grid_h=int(manifest.get("crop_grid_h", 1)),
        mode=str(manifest.get("mode", "image-parity-fixture")),
        prompt=str(manifest.get("prompt", "")),
        rendered_prompt=str(manifest.get("rendered_prompt", "")),
        max_new_tokens=max_new_tokens,
        max_length=int(manifest["max_length"]) if manifest.get("max_length") is not None else None,
        no_repeat_ngram_size=int(manifest.get("no_repeat_ngram_size", 0)),
        no_repeat_window=int(manifest.get("no_repeat_window", 0)),
        tokenizer_path=_tokenizer_path_from_manifest(manifest),
        source_images=tuple(manifest.get("source_images", ())),  # type: ignore[arg-type]
        model_vocab_size=int(manifest.get("model_vocab_size", 129280)),
    )


def _load_visual_features(root: Path, expected_rows: int) -> np.ndarray:
    visual = np.fromfile(root / VISUAL_FEATURES_BIN, dtype=np.dtype("<u2"))
    expected_values = expected_rows * HIDDEN_SIZE
    if visual.size != expected_values:
        pytest.fail(f"{VISUAL_FEATURES_BIN} has {visual.size} values, expected {expected_values}")
    return np.ascontiguousarray(visual.reshape((expected_rows, HIDDEN_SIZE)), dtype=np.dtype("<u2"))


def _load_first_generated_id(root: Path) -> np.ndarray:
    ids = np.fromfile(root / GENERATED_IDS_BIN, dtype=np.dtype("<i4"))
    if ids.ndim != 1 or ids.size == 0:
        pytest.fail(f"{GENERATED_IDS_BIN} in {root} must contain at least one int32 token id")
    return np.ascontiguousarray(ids[:1], dtype=np.int32)


def _load_reference_text(root: Path) -> str | None:
    path = root / GENERATED_TEXT_TXT
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _encode_metal_visual_features(model_path: str, resource_path: str, request: PreparedRequest) -> np.ndarray:
    lib = load_library()
    _bind_internal_metal_symbols(lib)
    if lib.uocr_metal_is_available() == 0:
        pytest.skip("Metal device is not available")

    error = ct.create_string_buffer(2048)
    model = CModelFile()
    status = lib.uocr_model_file_open(model_path.encode("utf-8"), ct.byref(model), error, len(error))
    if status != UOCR_OK:
        pytest.fail(f"uocr_model_file_open failed ({status}): {error.value.decode('utf-8', errors='replace')}")

    ctx = None
    try:
        ctx = lib.uocr_metal_context_create(resource_path.encode("utf-8"), error, len(error))
        if not ctx:
            pytest.fail(f"uocr_metal_context_create failed: {error.value.decode('utf-8', errors='replace')}")
        if lib.uocr_metal_context_map_model(ctx, ct.byref(model), error, len(error)) != 1:
            pytest.fail(f"uocr_metal_context_map_model failed: {error.value.decode('utf-8', errors='replace')}")

        rows = int(request.expected_visual_tokens)
        out = np.empty((rows, HIDDEN_SIZE), dtype=np.dtype("<u2"))
        keepalive = as_c_request(request)
        ok = lib.uocr_metal_context_encode_visual_features_f16(
            ctx,
            ct.byref(keepalive.c_request),
            ct.c_uint32(1),
            out.ctypes.data_as(ct.POINTER(ct.c_uint16)),
            ct.c_uint32(rows),
            error,
            len(error),
        )
        if ok != 1:
            pytest.fail(
                "uocr_metal_context_encode_visual_features_f16 failed: "
                + error.value.decode("utf-8", errors="replace")
            )
        return out
    finally:
        if ctx:
            lib.uocr_metal_context_destroy(ctx)
        lib.uocr_model_file_close(ct.byref(model))


def _encode_metal_sam_features(
    model_path: str,
    resource_path: str,
    request: PreparedRequest,
    *,
    view_index: int,
    grid_size: int,
) -> np.ndarray:
    lib = load_library()
    _bind_internal_metal_symbols(lib)
    if lib.uocr_metal_is_available() == 0:
        pytest.skip("Metal device is not available")

    error = ct.create_string_buffer(2048)
    model = CModelFile()
    status = lib.uocr_model_file_open(model_path.encode("utf-8"), ct.byref(model), error, len(error))
    if status != UOCR_OK:
        pytest.fail(f"uocr_model_file_open failed ({status}): {error.value.decode('utf-8', errors='replace')}")

    ctx = None
    try:
        ctx = lib.uocr_metal_context_create(resource_path.encode("utf-8"), error, len(error))
        if not ctx:
            pytest.fail(f"uocr_metal_context_create failed: {error.value.decode('utf-8', errors='replace')}")
        if lib.uocr_metal_context_map_model(ctx, ct.byref(model), error, len(error)) != 1:
            pytest.fail(f"uocr_metal_context_map_model failed: {error.value.decode('utf-8', errors='replace')}")

        if view_index < 0 or view_index >= len(request.views):
            pytest.fail(f"view_index {view_index} is outside request views")
        out = np.empty((SAM_FEATURE_CHANNELS, grid_size, grid_size), dtype=np.dtype("<u2"))
        keepalive = as_c_request(request)
        assert keepalive.c_views is not None
        ok = lib.uocr_metal_context_encode_sam_features_f16(
            ctx,
            ct.byref(keepalive.c_views[view_index]),
            out.ctypes.data_as(ct.POINTER(ct.c_uint16)),
            ct.c_uint32(grid_size),
            ct.c_uint32(grid_size),
            error,
            len(error),
        )
        if ok != 1:
            pytest.fail(
                "uocr_metal_context_encode_sam_features_f16 failed: "
                + error.value.decode("utf-8", errors="replace")
            )
        return out
    finally:
        if ctx:
            lib.uocr_metal_context_destroy(ctx)
        lib.uocr_model_file_close(ct.byref(model))


def _encode_metal_clip_features(
    model_path: str,
    resource_path: str,
    request: PreparedRequest,
    *,
    view_index: int,
    token_count: int,
) -> np.ndarray:
    lib = load_library()
    _bind_internal_metal_symbols(lib)
    if lib.uocr_metal_is_available() == 0:
        pytest.skip("Metal device is not available")

    error = ct.create_string_buffer(2048)
    model = CModelFile()
    status = lib.uocr_model_file_open(model_path.encode("utf-8"), ct.byref(model), error, len(error))
    if status != UOCR_OK:
        pytest.fail(f"uocr_model_file_open failed ({status}): {error.value.decode('utf-8', errors='replace')}")

    ctx = None
    try:
        ctx = lib.uocr_metal_context_create(resource_path.encode("utf-8"), error, len(error))
        if not ctx:
            pytest.fail(f"uocr_metal_context_create failed: {error.value.decode('utf-8', errors='replace')}")
        if lib.uocr_metal_context_map_model(ctx, ct.byref(model), error, len(error)) != 1:
            pytest.fail(f"uocr_metal_context_map_model failed: {error.value.decode('utf-8', errors='replace')}")

        if view_index < 0 or view_index >= len(request.views):
            pytest.fail(f"view_index {view_index} is outside request views")
        out = np.empty((token_count, CLIP_HIDDEN_SIZE), dtype=np.dtype("<u2"))
        keepalive = as_c_request(request)
        assert keepalive.c_views is not None
        ok = lib.uocr_metal_context_encode_clip_features_f16(
            ctx,
            ct.byref(keepalive.c_views[view_index]),
            out.ctypes.data_as(ct.POINTER(ct.c_uint16)),
            ct.c_uint32(token_count),
            error,
            len(error),
        )
        if ok != 1:
            pytest.fail(
                "uocr_metal_context_encode_clip_features_f16 failed: "
                + error.value.decode("utf-8", errors="replace")
            )
        return out
    finally:
        if ctx:
            lib.uocr_metal_context_destroy(ctx)
        lib.uocr_model_file_close(ct.byref(model))


def _encode_metal_projected_features(
    model_path: str,
    resource_path: str,
    request: PreparedRequest,
    *,
    view_index: int,
    row_count: int,
) -> np.ndarray:
    lib = load_library()
    _bind_internal_metal_symbols(lib)
    if lib.uocr_metal_is_available() == 0:
        pytest.skip("Metal device is not available")

    error = ct.create_string_buffer(2048)
    model = CModelFile()
    status = lib.uocr_model_file_open(model_path.encode("utf-8"), ct.byref(model), error, len(error))
    if status != UOCR_OK:
        pytest.fail(f"uocr_model_file_open failed ({status}): {error.value.decode('utf-8', errors='replace')}")

    ctx = None
    try:
        ctx = lib.uocr_metal_context_create(resource_path.encode("utf-8"), error, len(error))
        if not ctx:
            pytest.fail(f"uocr_metal_context_create failed: {error.value.decode('utf-8', errors='replace')}")
        if lib.uocr_metal_context_map_model(ctx, ct.byref(model), error, len(error)) != 1:
            pytest.fail(f"uocr_metal_context_map_model failed: {error.value.decode('utf-8', errors='replace')}")

        if view_index < 0 or view_index >= len(request.views):
            pytest.fail(f"view_index {view_index} is outside request views")
        out = np.empty((row_count, HIDDEN_SIZE), dtype=np.dtype("<u2"))
        keepalive = as_c_request(request)
        assert keepalive.c_views is not None
        ok = lib.uocr_metal_context_encode_projected_features_f16(
            ctx,
            ct.byref(keepalive.c_views[view_index]),
            out.ctypes.data_as(ct.POINTER(ct.c_uint16)),
            ct.c_uint32(row_count),
            error,
            len(error),
        )
        if ok != 1:
            pytest.fail(
                "uocr_metal_context_encode_projected_features_f16 failed: "
                + error.value.decode("utf-8", errors="replace")
            )
        return out
    finally:
        if ctx:
            lib.uocr_metal_context_destroy(ctx)
        lib.uocr_model_file_close(ct.byref(model))


def _assert_feature_bits_close(
    actual_bits: np.ndarray,
    expected_bits: np.ndarray,
    *,
    label: str,
    env_prefix: str,
    default_atol: float,
    default_p99_atol: float,
    default_mean_atol: float,
    default_rtol: float,
) -> None:
    actual = actual_bits.view(np.float16).astype(np.float32)
    expected = expected_bits.view(np.float16).astype(np.float32)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        pytest.fail(f"{label} features contain non-finite values")
    diff = np.abs(actual - expected)
    ref_scale = np.maximum(np.abs(expected), np.float32(1.0e-3))
    rel = diff / ref_scale
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    p99_abs = float(np.quantile(diff, 0.99))
    max_rel = float(np.max(rel))

    atol = float(os.environ.get(f"{env_prefix}_ATOL", str(default_atol)))
    p99_atol = float(os.environ.get(f"{env_prefix}_P99_ATOL", str(default_p99_atol)))
    mean_atol = float(os.environ.get(f"{env_prefix}_MEAN_ATOL", str(default_mean_atol)))
    rtol = float(os.environ.get(f"{env_prefix}_RTOL", str(default_rtol)))
    if max_abs > atol or p99_abs > p99_atol or mean_abs > mean_atol or max_rel > rtol:
        pytest.fail(
            f"{label} feature mismatch: max_abs={max_abs:.6g} (limit {atol}), "
            f"p99_abs={p99_abs:.6g} (limit {p99_atol}), "
            f"mean_abs={mean_abs:.6g} (limit {mean_atol}), max_rel={max_rel:.6g} (limit {rtol})"
        )


def _assert_visual_features_close(actual_bits: np.ndarray, expected_bits: np.ndarray, *, label: str) -> None:
    _assert_feature_bits_close(
        actual_bits,
        expected_bits,
        label=label,
        env_prefix="UOCR_IMAGE_VISUAL",
        default_atol=0.08,
        default_p99_atol=0.02,
        default_mean_atol=0.004,
        default_rtol=0.08,
    )


def _assert_sam_features_close(actual_bits: np.ndarray, expected_bits: np.ndarray, *, label: str) -> None:
    _assert_feature_bits_close(
        actual_bits,
        expected_bits,
        label=label,
        env_prefix="UOCR_SAM_FEATURE",
        default_atol=0.08,
        default_p99_atol=0.02,
        default_mean_atol=0.004,
        default_rtol=0.08,
    )


def _assert_clip_features_close(actual_bits: np.ndarray, expected_bits: np.ndarray, *, label: str) -> None:
    _assert_feature_bits_close(
        actual_bits,
        expected_bits,
        label=label,
        env_prefix="UOCR_CLIP_FEATURE",
        default_atol=0.12,
        default_p99_atol=0.03,
        default_mean_atol=0.006,
        default_rtol=0.10,
    )


def _assert_projected_features_close(actual_bits: np.ndarray, expected_bits: np.ndarray, *, label: str) -> None:
    _assert_feature_bits_close(
        actual_bits,
        expected_bits,
        label=label,
        env_prefix="UOCR_PROJECTED_FEATURE",
        default_atol=0.14,
        default_p99_atol=0.035,
        default_mean_atol=0.007,
        default_rtol=0.12,
    )


def _run_public_generation(model_path: str, resource_path: str, request: PreparedRequest) -> np.ndarray:
    with Engine(
        EngineOptions(
            model_path=model_path,
            backend="metal",
            resource_path=resource_path,
            max_batch=1,
            max_prompt_tokens=request.n_tokens,
            max_gen_tokens=request.max_new_tokens,
            memory_budget_bytes=(1 << 64) - 1,
        )
    ) as engine:
        outputs = engine.generate_prepared(request)
    assert len(outputs) == 1
    return np.ascontiguousarray(outputs[0], dtype=np.int32)


def _run_image_parity_case(label: str, root: Path) -> None:
    model_path, resource_path = _require_large_metal_inputs()
    _require_fixture_files(root)
    expected_ids = _load_first_generated_id(root)
    request = _prepared_request_from_fixture(root, max_new_tokens=int(expected_ids.size))
    assert request.expected_visual_tokens > 0
    expected_visual = _load_visual_features(root, request.expected_visual_tokens)

    actual_visual = _encode_metal_visual_features(model_path, resource_path, request)
    _assert_visual_features_close(actual_visual, expected_visual, label=label)

    actual_ids = _run_public_generation(model_path, resource_path, request)
    np.testing.assert_array_equal(actual_ids, expected_ids)

    reference_text = _load_reference_text(root)
    actual_text = decode_generated_text(actual_ids, request.tokenizer_path)
    if reference_text is not None:
        assert actual_text == reference_text
    print(f"{label} image parity generated ids={actual_ids.tolist()} decoded={actual_text!r}")


def test_public_metal_image_base_global_parity_fixture() -> None:
    root = _fixture_dir_from_env(
        "base/global",
        ("UOCR_IMAGE_PARITY_BASE_DIR", "UOCR_IMAGE_PARITY_DIR", "UOCR_LAYER_DUMP_DIR"),
    )
    manifest, _, _, _ = load_prepared_fixture(root)
    if int(manifest.get("image_mask_count", 0)) != GLOBAL_VISUAL_TOKENS:
        pytest.fail(f"base/global fixture must have {GLOBAL_VISUAL_TOKENS} visual tokens")
    views = manifest.get("views")
    if not isinstance(views, list) or len(views) != 1 or not isinstance(views[0], dict) or views[0].get("kind") != "global":
        pytest.fail("base/global fixture must contain exactly one global view")
    _run_image_parity_case("base/global", root)


def test_public_metal_image_optional_gundam_parity_fixtures() -> None:
    cases = [
        ("gundam [1,1]", _optional_fixture_dir("gundam [1,1]", "UOCR_IMAGE_PARITY_GUNDAM_1X1_DIR")),
        ("gundam multi-crop", _optional_fixture_dir("gundam multi-crop", "UOCR_IMAGE_PARITY_GUNDAM_MULTICROP_DIR")),
    ]
    selected = [(label, root) for label, root in cases if root is not None]
    if not selected:
        pytest.skip(
            "set UOCR_IMAGE_PARITY_GUNDAM_1X1_DIR and/or UOCR_IMAGE_PARITY_GUNDAM_MULTICROP_DIR "
            "to run optional gundam image parity"
        )
    for label, root in selected:
        _run_image_parity_case(label, root)


def _first_view_index_with_kind(request: PreparedRequest, kind: str) -> int:
    for index, view in enumerate(request.views):
        if view.kind == kind:
            return index
    pytest.fail(f"fixture has no {kind!r} view")


def _run_sam_parity_case(
    label: str,
    root: Path,
    *,
    view_index: int,
    expected_kind: str,
    expected_grid: int,
) -> None:
    model_path, resource_path = _require_large_metal_inputs()
    request = _prepared_request_from_fixture(root, max_new_tokens=1)
    if view_index < 0 or view_index >= len(request.views):
        pytest.fail(f"{label} SAM parity view index {view_index} is outside fixture views")
    view = request.views[view_index]
    if view.kind != expected_kind:
        pytest.fail(f"{label} SAM parity requires a {expected_kind} view, got {view.kind!r} at index {view_index}")
    expected_size = 640 if expected_kind == "local" else 1024
    if view.width != expected_size or view.height != expected_size:
        pytest.fail(f"{label} SAM parity view must be {expected_size}x{expected_size}, got {view.width}x{view.height}")

    expected = load_sam_features_dump(root, view_index=view_index, grid_size=expected_grid)
    actual = _encode_metal_sam_features(
        model_path,
        resource_path,
        request,
        view_index=view_index,
        grid_size=expected_grid,
    )
    _assert_sam_features_close(actual, expected, label=label)
    print(f"{label} SAM parity matched shape={actual.shape} view_index={view_index}")


def test_public_metal_sam_global_1024_parity_fixture() -> None:
    root = _fixture_dir_from_env("SAM global 1024", ("UOCR_SAM_PARITY_1024_DIR",))
    view_index = int(os.environ.get("UOCR_SAM_PARITY_1024_VIEW_INDEX", "0"))
    _run_sam_parity_case(
        "SAM global 1024",
        root,
        view_index=view_index,
        expected_kind="global",
        expected_grid=SAM_GLOBAL_GRID,
    )


def test_public_metal_sam_local_640_parity_fixture() -> None:
    root = _fixture_dir_from_env("SAM local 640", ("UOCR_SAM_PARITY_640_DIR",))
    request = _prepared_request_from_fixture(root, max_new_tokens=1)
    view_index = int(os.environ["UOCR_SAM_PARITY_640_VIEW_INDEX"]) if os.environ.get("UOCR_SAM_PARITY_640_VIEW_INDEX") else _first_view_index_with_kind(request, "local")
    _run_sam_parity_case(
        "SAM local 640",
        root,
        view_index=view_index,
        expected_kind="local",
        expected_grid=SAM_LOCAL_GRID,
    )


def _run_clip_parity_case(
    label: str,
    root: Path,
    *,
    view_index: int,
    expected_kind: str,
    expected_tokens: int,
) -> None:
    model_path, resource_path = _require_large_metal_inputs()
    request = _prepared_request_from_fixture(root, max_new_tokens=1)
    if view_index < 0 or view_index >= len(request.views):
        pytest.fail(f"{label} CLIP parity view index {view_index} is outside fixture views")
    view = request.views[view_index]
    if view.kind != expected_kind:
        pytest.fail(f"{label} CLIP parity requires a {expected_kind} view, got {view.kind!r} at index {view_index}")
    expected_size = 640 if expected_kind == "local" else 1024
    if view.width != expected_size or view.height != expected_size:
        pytest.fail(f"{label} CLIP parity view must be {expected_size}x{expected_size}, got {view.width}x{view.height}")

    expected = load_clip_features_dump(root, view_index=view_index, token_count=expected_tokens)
    actual = _encode_metal_clip_features(
        model_path,
        resource_path,
        request,
        view_index=view_index,
        token_count=expected_tokens,
    )
    _assert_clip_features_close(actual, expected, label=label)
    print(f"{label} CLIP parity matched shape={actual.shape} view_index={view_index}")


def test_public_metal_clip_global_1024_parity_fixture() -> None:
    root = _fixture_dir_from_env("CLIP global 1024", ("UOCR_CLIP_PARITY_1024_DIR",))
    view_index = int(os.environ.get("UOCR_CLIP_PARITY_1024_VIEW_INDEX", "0"))
    _run_clip_parity_case(
        "CLIP global 1024",
        root,
        view_index=view_index,
        expected_kind="global",
        expected_tokens=CLIP_GLOBAL_TOKENS,
    )


def test_public_metal_clip_local_640_parity_fixture() -> None:
    root = _fixture_dir_from_env("CLIP local 640", ("UOCR_CLIP_PARITY_640_DIR",))
    request = _prepared_request_from_fixture(root, max_new_tokens=1)
    view_index = int(os.environ["UOCR_CLIP_PARITY_640_VIEW_INDEX"]) if os.environ.get("UOCR_CLIP_PARITY_640_VIEW_INDEX") else _first_view_index_with_kind(request, "local")
    _run_clip_parity_case(
        "CLIP local 640",
        root,
        view_index=view_index,
        expected_kind="local",
        expected_tokens=CLIP_LOCAL_TOKENS,
    )


def _run_projected_parity_case(
    label: str,
    root: Path,
    *,
    view_index: int,
    expected_kind: str,
    expected_rows: int,
) -> None:
    model_path, resource_path = _require_large_metal_inputs()
    request = _prepared_request_from_fixture(root, max_new_tokens=1)
    if view_index < 0 or view_index >= len(request.views):
        pytest.fail(f"{label} projected-feature parity view index {view_index} is outside fixture views")
    view = request.views[view_index]
    if view.kind != expected_kind:
        pytest.fail(f"{label} projected-feature parity requires a {expected_kind} view, got {view.kind!r} at index {view_index}")
    expected_size = 640 if expected_kind == "local" else 1024
    if view.width != expected_size or view.height != expected_size:
        pytest.fail(f"{label} projected-feature parity view must be {expected_size}x{expected_size}, got {view.width}x{view.height}")

    expected = load_projected_features_dump(root, view_index=view_index, row_count=expected_rows)
    actual = _encode_metal_projected_features(
        model_path,
        resource_path,
        request,
        view_index=view_index,
        row_count=expected_rows,
    )
    _assert_projected_features_close(actual, expected, label=label)
    print(f"{label} projected-feature parity matched shape={actual.shape} view_index={view_index}")


def test_public_metal_projected_global_1024_parity_fixture() -> None:
    root = _fixture_dir_from_env("projected global 1024", ("UOCR_PROJECTED_PARITY_1024_DIR",))
    view_index = int(os.environ.get("UOCR_PROJECTED_PARITY_1024_VIEW_INDEX", "0"))
    _run_projected_parity_case(
        "projected global 1024",
        root,
        view_index=view_index,
        expected_kind="global",
        expected_rows=PROJECTED_GLOBAL_ROWS,
    )


def test_public_metal_projected_local_640_parity_fixture() -> None:
    root = _fixture_dir_from_env("projected local 640", ("UOCR_PROJECTED_PARITY_640_DIR",))
    request = _prepared_request_from_fixture(root, max_new_tokens=1)
    view_index = int(os.environ["UOCR_PROJECTED_PARITY_640_VIEW_INDEX"]) if os.environ.get("UOCR_PROJECTED_PARITY_640_VIEW_INDEX") else _first_view_index_with_kind(request, "local")
    _run_projected_parity_case(
        "projected local 640",
        root,
        view_index=view_index,
        expected_kind="local",
        expected_rows=PROJECTED_LOCAL_ROWS,
    )
