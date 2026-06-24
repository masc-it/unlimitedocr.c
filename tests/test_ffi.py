from __future__ import annotations

import ctypes as ct
import os

import numpy as np
import pytest

from unlimitedocr_c.ffi import Engine, EngineOptions, UOCR_MEMORY_KV_CACHE, _copy_result_tokens, find_library_path
from unlimitedocr_c.frontend import MODEL_VOCAB_SIZE, load_tokenizer, prepare_image, prepare_text, project_root
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
    width, height = 160, 96
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = x[None, :]
    pixels[..., 1] = y
    pixels[..., 2] = ((pixels[..., 0].astype(np.uint16) + pixels[..., 1].astype(np.uint16)) // 2).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")

    req = prepare_image(image, preset="base", max_new_tokens=1)
    assert req.expected_visual_tokens == 273
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
    generated = outputs[0]
    assert generated.shape == (1,)
    token_id = int(generated[0])
    assert 0 <= token_id < MODEL_VOCAB_SIZE

    tokenizer = load_tokenizer(req.tokenizer_path)
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    print(f"public Metal image smoke generated token id={token_id} decoded={decoded!r}")
    assert isinstance(decoded, str)
