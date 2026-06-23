from __future__ import annotations

import pytest

from unlimitedocr_c.ffi import Engine, EngineOptions, find_library_path
from unlimitedocr_c.frontend import prepare_image, prepare_text
from PIL import Image


def native_library_available() -> bool:
    try:
        find_library_path()
    except FileNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")


def test_ctypes_text_smoke() -> None:
    req = prepare_text("hello", max_new_tokens=0)
    with Engine(EngineOptions(backend="cpu-ref", max_prompt_tokens=64, max_gen_tokens=4)) as engine:
        assert engine.backend == "cpu-ref"
        outputs = engine.generate_prepared(req)
    assert len(outputs) == 1
    assert outputs[0].shape == (0,)


def test_ctypes_image_validation_smoke() -> None:
    image = Image.new("RGB", (64, 64), (1, 2, 3))
    req = prepare_image(image, max_new_tokens=0)
    with Engine(EngineOptions(backend="cpu-ref", max_prompt_tokens=512, max_gen_tokens=4)) as engine:
        outputs = engine.generate_prepared(req)
    assert len(outputs) == 1
    assert outputs[0].shape == (0,)
