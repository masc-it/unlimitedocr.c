from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from unlimitedocr_c import UnlimitedOCR
import unlimitedocr_c.api as api
from unlimitedocr_c.frontend import EOS_TOKEN_ID, GLOBAL_VISUAL_TOKENS


class _FakeEngine:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self.closed = False

    def generate_prepared(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return [np.asarray([EOS_TOKEN_ID], dtype=np.int32)]

    def close(self) -> None:
        self.closed = True


class _TestOCR(UnlimitedOCR):
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.fake_engine = _FakeEngine()
        self.open_calls: list[dict[str, int]] = []
        super().__init__(**kwargs)

    def _open_engine(self, *, prompt_capacity: int, gen_capacity: int):  # type: ignore[no-untyped-def]
        self.open_calls.append({"prompt_capacity": prompt_capacity, "gen_capacity": gen_capacity})
        return self.fake_engine


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), (12, 34, 56))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_load_user_image_accepts_base64_data_uri_and_bytesio() -> None:
    payload = _png_bytes()

    raw_b64 = base64.b64encode(payload).decode("ascii")
    assert api.load_user_image(raw_b64).size == (32, 24)

    data_uri = "data:image/png;base64," + raw_b64
    assert api.load_user_image(data_uri).size == (32, 24)

    stream = BytesIO(payload)
    assert api.load_user_image(stream).size == (32, 24)
    assert stream.tell() == 0


def test_resolve_model_path_prefers_cache_without_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UOCR_MODEL_FILENAME", "cached-test-model.uocr")
    cached = tmp_path / "cached-test-model.uocr"
    cached.write_bytes(b"uocr")

    def fail_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("download should not be attempted")

    monkeypatch.setattr(api, "_download_and_convert_source_model", fail_download)
    assert api.resolve_model_path(cache_dir=tmp_path, download=True) == cached.resolve()


def test_resolve_model_path_uses_download_when_cache_misses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UOCR_MODEL_FILENAME", "downloaded-test-model.uocr")
    downloaded = tmp_path / "hf-cache" / "downloaded-test-model.uocr"
    downloaded.parent.mkdir()
    downloaded.write_bytes(b"uocr")

    def fake_download(cache_dir: Path, filename: str, **kwargs):  # type: ignore[no-untyped-def]
        assert cache_dir == tmp_path
        assert filename == "downloaded-test-model.uocr"
        return downloaded

    monkeypatch.setattr(api, "_download_and_convert_source_model", fake_download)
    assert api.resolve_model_path(cache_dir=tmp_path, download=True) == downloaded


def test_resolve_model_path_q8_uses_separate_cache_filename(tmp_path: Path) -> None:
    q8_cached = tmp_path / api.DEFAULT_Q8_MODEL_FILENAME
    q8_cached.write_bytes(b"uocr")
    resolved = api.resolve_model_path(cache_dir=tmp_path, download=False, quant="q8")
    assert resolved == q8_cached.resolve()


def test_resolve_model_path_q8_does_not_match_fp16_cache(tmp_path: Path) -> None:
    fp16_cached = tmp_path / api.DEFAULT_MODEL_FILENAME
    fp16_cached.write_bytes(b"uocr")
    with pytest.raises(api.ModelResolutionError, match="could not find"):
        api.resolve_model_path(cache_dir=tmp_path, download=False, quant="q8")


def test_unlimitedocr_generate_returns_decoded_string_and_uses_profile(tmp_path: Path) -> None:
    model = tmp_path / "model.uocr"
    model.write_bytes(b"uocr")
    ocr = _TestOCR(model_path=model, download=False, backend="cpu-ref")

    text = ocr.generate(Image.new("RGB", (96, 64), (1, 2, 3)), profile="base")

    assert text == ""
    assert len(ocr.fake_engine.requests) == 1
    request = ocr.fake_engine.requests[0]
    assert request.mode == "base"
    assert request.expected_visual_tokens == GLOBAL_VISUAL_TOKENS
    assert request.max_length == api.DEFAULT_MAX_LENGTH
    assert request.max_new_tokens == api.DEFAULT_MAX_LENGTH - request.n_tokens
    assert ocr.open_calls == [
        {"prompt_capacity": request.n_tokens, "gen_capacity": request.max_new_tokens}
    ]


def test_unlimitedocr_rejects_unknown_profile(tmp_path: Path) -> None:
    model = tmp_path / "model.uocr"
    model.write_bytes(b"uocr")
    ocr = _TestOCR(model_path=model, download=False, backend="cpu-ref")

    with pytest.raises(ValueError, match="unknown profile"):
        ocr.generate(Image.new("RGB", (32, 32), (1, 2, 3)), profile="small")  # type: ignore[arg-type]
