from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

import unlimitedocr_c.ocr as ocr
from unlimitedocr_c.ffi import find_library_path
from unlimitedocr_c.frontend import EOS_TOKEN_ID, MODEL_VOCAB_SIZE


def native_library_available() -> bool:
    try:
        find_library_path()
    except FileNotFoundError:
        return False
    return True


def _large_metal_test_enabled() -> bool:
    return os.environ.get("UOCR_RUN_LARGE_TESTS") == "1" and bool(os.environ.get("UOCR_MODEL_PATH"))


def _gradient_image(width: int, height: int) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = x[None, :]
    pixels[..., 1] = y
    pixels[..., 2] = ((pixels[..., 0].astype(np.uint16) + pixels[..., 1].astype(np.uint16)) // 2).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_decode_generated_ids_strips_one_trailing_eos() -> None:
    assert ocr.decode_generated_ids([128818]) == "<|det|>"
    assert ocr.decode_generated_ids([128818, EOS_TOKEN_ID]) == "<|det|>"


def test_decode_generated_ids_validates_shape_and_vocab() -> None:
    with pytest.raises(ValueError, match="rank-1"):
        ocr.decode_generated_ids(np.zeros((1, 1), dtype=np.int32))
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        ocr.decode_generated_ids([MODEL_VOCAB_SIZE])


@pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")
def test_generate_cpu_ref_image_validation_path() -> None:
    result = ocr.generate(Image.new("RGB", (64, 64), (1, 2, 3)), backend="cpu-ref", max_new_tokens=0)
    assert result.token_ids.dtype == np.dtype(np.int32)
    assert result.token_ids.shape == (0,)
    assert result.text == ""


@pytest.mark.skipif(
    not _large_metal_test_enabled(),
    reason="set UOCR_RUN_LARGE_TESTS=1 and UOCR_MODEL_PATH to run the full-model Metal ocr.generate smoke test",
)
def test_generate_public_metal_image_smoke() -> None:
    result = ocr.generate(
        _gradient_image(160, 96),
        model_path=os.environ["UOCR_MODEL_PATH"],
        preset="base",
        max_new_tokens=1,
        memory_budget_bytes=(1 << 64) - 1,
    )
    assert result.token_ids.shape == (1,)
    token_id = int(result.token_ids[0])
    assert 0 <= token_id < MODEL_VOCAB_SIZE
    assert isinstance(result.text, str)
    print(f"ocr.generate Metal image smoke generated token id={token_id} decoded={result.text!r}")
