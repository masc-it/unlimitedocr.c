from __future__ import annotations

import ctypes as ct
import os

import numpy as np
import pytest

from unlimitedocr_c.ffi import Engine, EngineOptions, UOCR_MEMORY_KV_CACHE, _copy_result_tokens, find_library_path
from unlimitedocr_c.frontend import (
    GLOBAL_VISUAL_TOKENS,
    MODEL_VOCAB_SIZE,
    load_tokenizer,
    prepare_image,
    prepare_pages,
    prepare_text,
    project_root,
)
from PIL import Image


def native_library_available() -> bool:
    try:
        find_library_path()
    except FileNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")


def _large_metal_test_enabled() -> bool:
    return os.environ.get("UOCR_RUN_LARGE_TESTS") == "1" and bool(os.environ.get("UOCR_MODEL_PATH"))


def _gradient_image(width: int, height: int, phase: int = 0) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = ((x[None, :].astype(np.uint16) + phase) % 256).astype(np.uint8)
    pixels[..., 1] = ((y.astype(np.uint16) + 2 * phase) % 256).astype(np.uint8)
    pixels[..., 2] = ((pixels[..., 0].astype(np.uint16) + pixels[..., 1].astype(np.uint16)) // 2).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _assert_single_generated_id(generated: np.ndarray, tokenizer_path: str) -> tuple[int, str]:
    assert generated.shape == (1,)
    token_id = int(generated[0])
    assert 0 <= token_id < MODEL_VOCAB_SIZE
    tokenizer = load_tokenizer(tokenizer_path)
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    assert isinstance(decoded, str)
    return token_id, decoded


class _FakeResultLib:
    def __init__(self) -> None:
        self._arrays = [
            (ct.c_int32 * 3)(17, 42, 1),
            (ct.c_int32 * 0)(),
            (ct.c_int32 * 2)(128815, 2),
        ]

    def uocr_result_count(self, result: ct.c_void_p) -> int:
        assert result.value == 1234
        return len(self._arrays)

    def uocr_result_tokens(self, result: ct.c_void_p, index: int, n_tokens: ct.c_void_p) -> ct.POINTER(ct.c_int32):
        assert result.value == 1234
        array = self._arrays[index]
        n_tokens._obj.value = len(array)  # type: ignore[attr-defined]
        if len(array) == 0:
            return ct.POINTER(ct.c_int32)()
        return ct.cast(array, ct.POINTER(ct.c_int32))


def test_copy_result_tokens_handles_non_empty_outputs() -> None:
    fake = _FakeResultLib()
    outputs = _copy_result_tokens(fake, ct.c_void_p(1234))  # type: ignore[arg-type]
    assert [out.dtype for out in outputs] == [np.dtype(np.int32)] * 3
    assert [out.tolist() for out in outputs] == [[17, 42, 1], [], [128815, 2]]

    fake._arrays[0][0] = 999
    assert outputs[0].tolist() == [17, 42, 1]


def test_ctypes_text_smoke() -> None:
    req = prepare_text("hello", max_new_tokens=0)
    with Engine(EngineOptions(backend="cpu-ref", max_prompt_tokens=64, max_gen_tokens=4)) as engine:
        assert engine.backend == "cpu-ref"
        outputs = engine.generate_prepared(req)
    assert len(outputs) == 1
    assert outputs[0].shape == (0,)


def test_ctypes_memory_report_smoke() -> None:
    req = prepare_text("hello", max_new_tokens=0)
    with Engine(EngineOptions(backend="cpu-ref", max_prompt_tokens=64, max_gen_tokens=4)) as engine:
        assert engine.memory_category_name(UOCR_MEMORY_KV_CACHE) == "kv-cache"
        before = engine.memory_report()
        outputs = engine.generate_prepared(req)
        after = engine.memory_report()
    assert len(outputs) == 1
    assert before.memory_budget_bytes == 0
    assert before.recommended_working_set_bytes == 0
    assert before.estimated_kv_cache_bytes > 0
    assert after.estimated_total_bytes < before.estimated_total_bytes
    assert len(after.category_live_bytes) == 8


def test_ctypes_image_validation_smoke() -> None:
    image = Image.new("RGB", (64, 64), (1, 2, 3))
    req = prepare_image(image, max_new_tokens=0)
    with Engine(EngineOptions(backend="cpu-ref", max_prompt_tokens=512, max_gen_tokens=4)) as engine:
        outputs = engine.generate_prepared(req)
    assert len(outputs) == 1
    assert outputs[0].shape == (0,)


@pytest.mark.skipif(
    not _large_metal_test_enabled(),
    reason="set UOCR_RUN_LARGE_TESTS=1 and UOCR_MODEL_PATH to run the full-model Metal public image smoke test",
)
def test_public_metal_image_generation_smoke() -> None:
    req = prepare_image(_gradient_image(160, 96), preset="base", max_new_tokens=1)
    assert req.expected_visual_tokens == GLOBAL_VISUAL_TOKENS
    assert len(req.views) == 1
    assert req.views[0].kind == "global"

    model_path = os.environ["UOCR_MODEL_PATH"]
    resource_path = project_root() / "src" / "backend" / "metal"
    with Engine(
        EngineOptions(
            model_path=model_path,
            backend="metal",
            resource_path=str(resource_path),
            max_batch=1,
            max_prompt_tokens=req.n_tokens,
            max_gen_tokens=1,
            memory_budget_bytes=(1 << 64) - 1,
        )
    ) as engine:
        outputs = engine.generate_prepared(req)

    assert len(outputs) == 1
    token_id, decoded = _assert_single_generated_id(outputs[0], req.tokenizer_path)
    print(f"public Metal image smoke generated token id={token_id} decoded={decoded!r}")


@pytest.mark.skipif(
    not _large_metal_test_enabled(),
    reason="set UOCR_RUN_LARGE_TESTS=1 and UOCR_MODEL_PATH to run the extended full-model Metal image smoke test",
)
def test_public_metal_image_generation_extended_smoke() -> None:
    gundam_1x1 = prepare_image(_gradient_image(512, 384, phase=3), preset="gundam", max_new_tokens=1)
    assert gundam_1x1.expected_visual_tokens == GLOBAL_VISUAL_TOKENS
    assert (gundam_1x1.crop_grid_w, gundam_1x1.crop_grid_h) == (1, 1)
    assert [view.kind for view in gundam_1x1.views] == ["global"]

    gundam_multicrop = prepare_image(_gradient_image(960, 640, phase=7), preset="gundam", max_new_tokens=1)
    assert gundam_multicrop.expected_visual_tokens > GLOBAL_VISUAL_TOKENS
    assert gundam_multicrop.crop_grid_w > 1 or gundam_multicrop.crop_grid_h > 1
    assert gundam_multicrop.views[-1].kind == "global"
    assert all(view.kind == "local" for view in gundam_multicrop.views[:-1])

    pages = prepare_pages([_gradient_image(128, 128, phase=11), _gradient_image(96, 160, phase=17)], max_new_tokens=1)
    assert pages.expected_visual_tokens == 2 * GLOBAL_VISUAL_TOKENS
    assert [view.kind for view in pages.views] == ["global", "global"]

    cases = [
        ("gundam [1,1]", gundam_1x1),
        ("gundam multi-crop", gundam_multicrop),
        ("multi-page base", pages),
    ]

    model_path = os.environ["UOCR_MODEL_PATH"]
    resource_path = project_root() / "src" / "backend" / "metal"
    with Engine(
        EngineOptions(
            model_path=model_path,
            backend="metal",
            resource_path=str(resource_path),
            max_batch=1,
            max_prompt_tokens=max(request.n_tokens for _, request in cases),
            max_gen_tokens=1,
            memory_budget_bytes=(1 << 64) - 1,
        )
    ) as engine:
        for label, request in cases:
            outputs = engine.generate_prepared(request)
            assert len(outputs) == 1
            token_id, decoded = _assert_single_generated_id(outputs[0], request.tokenizer_path)
            print(
                f"public Metal image extended smoke {label}: "
                f"views={len(request.views)} grid={request.crop_grid_w}x{request.crop_grid_h} "
                f"visual={request.expected_visual_tokens} token id={token_id} decoded={decoded!r}"
            )
