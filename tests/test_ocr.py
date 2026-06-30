from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import unlimitedocr_c.ocr as ocr
from unlimitedocr_c.ffi import find_library_path
from unlimitedocr_c.frontend import GLOBAL_VISUAL_TOKENS, EOS_TOKEN_ID, MODEL_VOCAB_SIZE, MULTI_PROMPT, SINGLE_PROMPT


def native_library_available() -> bool:
    try:
        find_library_path()
    except FileNotFoundError:
        return False
    return True


def _large_metal_test_enabled() -> bool:
    return os.environ.get("UOCR_RUN_LARGE_TESTS") == "1" and bool(os.environ.get("UOCR_MODEL_PATH"))


class _FakeEngine:
    def __init__(self, generated: np.ndarray | None = None) -> None:
        self.generated = np.asarray(generated if generated is not None else [EOS_TOKEN_ID], dtype=np.int32)
        self.requests: list[object] = []

    def generate_prepared(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return [self.generated]


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


def test_ocr_image_uses_upstream_defaults_and_caps_generation() -> None:
    fake = _FakeEngine()
    result = ocr.ocr_image(Image.new("RGB", (320, 240), (1, 2, 3)), engine=fake, max_gen_tokens=7)
    assert result.token_ids.tolist() == [EOS_TOKEN_ID]
    assert result.text == ""
    assert len(fake.requests) == 1

    request = fake.requests[0]
    assert request.prompt == SINGLE_PROMPT
    assert request.mode == "gundam"
    assert request.expected_visual_tokens == GLOBAL_VISUAL_TOKENS
    assert request.max_length == 32768
    assert request.max_new_tokens == 7
    assert request.no_repeat_ngram_size == 35
    assert request.no_repeat_window == 128


def test_ocr_image_rejects_negative_generation_cap() -> None:
    with pytest.raises(ValueError, match="max_gen_tokens must be non-negative"):
        ocr.ocr_image(Image.new("RGB", (64, 64), (1, 2, 3)), engine=_FakeEngine(), max_gen_tokens=-1)


def test_ocr_pages_uses_upstream_defaults_and_caps_generation() -> None:
    fake = _FakeEngine()
    pages = [Image.new("RGB", (96, 64), (1, 2, 3)), Image.new("RGB", (64, 96), (4, 5, 6))]
    result = ocr.ocr_pages(pages, engine=fake, max_gen_tokens=9)
    assert result.token_ids.tolist() == [EOS_TOKEN_ID]
    assert result.text == ""
    assert len(fake.requests) == 1

    request = fake.requests[0]
    assert request.prompt == MULTI_PROMPT
    assert request.mode == "multi-page-base"
    assert request.expected_visual_tokens == 2 * GLOBAL_VISUAL_TOKENS
    assert request.max_length == 32768
    assert request.max_new_tokens == 9
    assert request.no_repeat_ngram_size == 35
    assert request.no_repeat_window == 1024
    assert request.crop_grid_w == 1
    assert request.crop_grid_h == 1
    assert [view.kind for view in request.views] == ["global", "global"]


def test_ocr_pages_rejects_negative_generation_cap() -> None:
    with pytest.raises(ValueError, match="max_gen_tokens must be non-negative"):
        ocr.ocr_pages([Image.new("RGB", (64, 64), (1, 2, 3))], engine=_FakeEngine(), max_gen_tokens=-1)


def test_pdf_to_images_rasterizes_with_lazy_pymupdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class _FakePixmap:
        def __init__(self, page_index: int) -> None:
            self.page_index = page_index

        def save(self, path: str) -> None:
            Path(path).write_bytes(f"page {self.page_index}".encode("ascii"))

    class _FakePage:
        def __init__(self, page_index: int) -> None:
            self.page_index = page_index

        def get_pixmap(self, *, matrix: object, alpha: bool) -> _FakePixmap:
            calls.setdefault("pixmap_calls", []).append((self.page_index, matrix, alpha))  # type: ignore[union-attr]
            return _FakePixmap(self.page_index)

    class _FakeDocument:
        def __init__(self) -> None:
            self.closed = False
            self.pages = [_FakePage(1), _FakePage(2)]

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self.pages)

        def close(self) -> None:
            self.closed = True
            calls["closed"] = True

    fake_doc = _FakeDocument()

    def fake_open(path: str) -> _FakeDocument:
        calls["open_path"] = path
        return fake_doc

    def fake_matrix(x_scale: float, y_scale: float) -> tuple[float, float]:
        calls["matrix"] = (x_scale, y_scale)
        return (x_scale, y_scale)

    fake_fitz = types.SimpleNamespace(open=fake_open, Matrix=fake_matrix)
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    out_dir = tmp_path / "pages"
    paths = ocr.pdf_to_images(tmp_path / "doc.pdf", dpi=144, out_dir=out_dir)

    assert calls["open_path"] == str(tmp_path / "doc.pdf")
    assert calls["matrix"] == (2.0, 2.0)
    assert calls["closed"] is True
    assert paths == [out_dir / "page_0001.png", out_dir / "page_0002.png"]
    assert [path.read_bytes() for path in paths] == [b"page 1", b"page 2"]
    assert calls["pixmap_calls"] == [(1, (2.0, 2.0), False), (2, (2.0, 2.0), False)]


def test_pdf_to_images_rejects_invalid_dpi() -> None:
    with pytest.raises(ValueError, match="dpi must be positive"):
        ocr.pdf_to_images("document.pdf", dpi=0)


def test_ocr_pdf_renders_then_uses_multi_page_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_pdf_to_images(pdf_path: str | Path, *, dpi: int = 300, out_dir: str | Path | None = None) -> list[Image.Image]:
        calls["pdf_path"] = pdf_path
        calls["dpi"] = dpi
        calls["out_dir"] = out_dir
        return [Image.new("RGB", (96, 64), (1, 2, 3)), Image.new("RGB", (64, 96), (4, 5, 6))]

    monkeypatch.setattr(ocr, "pdf_to_images", fake_pdf_to_images)
    fake = _FakeEngine()
    page_dir = tmp_path / "rendered-pages"
    result = ocr.ocr_pdf(tmp_path / "doc.pdf", engine=fake, dpi=200, page_output_dir=page_dir, max_gen_tokens=11)

    assert calls == {"pdf_path": tmp_path / "doc.pdf", "dpi": 200, "out_dir": page_dir}
    assert result.token_ids.tolist() == [EOS_TOKEN_ID]
    assert result.text == ""
    assert len(fake.requests) == 1

    request = fake.requests[0]
    assert request.prompt == MULTI_PROMPT
    assert request.mode == "multi-page-base"
    assert request.expected_visual_tokens == 2 * GLOBAL_VISUAL_TOKENS
    assert request.max_new_tokens == 11
    assert request.no_repeat_ngram_size == 35
    assert request.no_repeat_window == 1024


@pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")
def test_ocr_image_cpu_ref_validation_path() -> None:
    result = ocr.ocr_image(Image.new("RGB", (64, 64), (1, 2, 3)), backend="cpu-ref", max_gen_tokens=0)
    assert result.token_ids.dtype == np.dtype(np.int32)
    assert result.token_ids.shape == (0,)
    assert result.text == ""


@pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")
def test_ocr_pages_cpu_ref_validation_path() -> None:
    pages = [Image.new("RGB", (64, 64), (1, 2, 3)), Image.new("RGB", (96, 64), (4, 5, 6))]
    result = ocr.ocr_pages(pages, backend="cpu-ref", max_gen_tokens=0)
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


@pytest.mark.skipif(
    not _large_metal_test_enabled(),
    reason="set UOCR_RUN_LARGE_TESTS=1 and UOCR_MODEL_PATH to run the full-model Metal ocr_pages smoke test",
)
def test_ocr_pages_public_metal_image_smoke() -> None:
    result = ocr.ocr_pages(
        [_gradient_image(128, 128), _gradient_image(96, 160)],
        model_path=os.environ["UOCR_MODEL_PATH"],
        max_gen_tokens=1,
        memory_budget_bytes=(1 << 64) - 1,
    )
    assert result.token_ids.shape == (1,)
    token_id = int(result.token_ids[0])
    assert 0 <= token_id < MODEL_VOCAB_SIZE
    assert isinstance(result.text, str)
    print(f"ocr_pages Metal image smoke generated token id={token_id} decoded={result.text!r}")
