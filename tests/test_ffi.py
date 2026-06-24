from __future__ import annotations

import ctypes as ct

import numpy as np
import pytest

from unlimitedocr_c.ffi import Engine, EngineOptions, UOCR_MEMORY_KV_CACHE, _copy_result_tokens, find_library_path
from unlimitedocr_c.frontend import prepare_image, prepare_text
from PIL import Image


def native_library_available() -> bool:
    try:
        find_library_path()
    except FileNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")


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
