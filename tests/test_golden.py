from __future__ import annotations

import json
import struct

import numpy as np

from unlimitedocr_c.frontend import PreparedRequest
from unlimitedocr_c.golden import (
    dump_prompt_embedding_fixture,
    load_prompt_embedding_dump,
    read_bf16_rows_as_f16_bits,
    read_bf16_tensor_as_f16_bits,
)


def _bf16_payload(values: np.ndarray) -> bytes:
    fp32 = np.asarray(values, dtype=np.dtype("<f4"))
    return ((fp32.view(np.dtype("<u4")) >> np.uint32(16)).astype(np.dtype("<u2"))).tobytes()


def _expected_f16_bits_from_bf16_trunc(values: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(values, dtype=np.dtype("<f4"))
    bf16 = (fp32.view(np.dtype("<u4")) >> np.uint32(16)).astype(np.dtype("<u2"))
    rounded_fp32 = (bf16.astype(np.uint32) << np.uint32(16)).astype(np.dtype("<u4")).view(np.dtype("<f4"))
    return rounded_fp32.astype(np.dtype("<f2")).view(np.dtype("<u2"))


def _write_tiny_embedding_safetensors(root, values: np.ndarray) -> None:
    payload = _bf16_payload(values)
    header = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [0, len(payload)],
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with (root / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def _tiny_text_request(input_ids: np.ndarray) -> PreparedRequest:
    return PreparedRequest(
        input_ids=np.asarray(input_ids, dtype=np.int32),
        image_mask=np.zeros(input_ids.shape[0], dtype=np.uint8),
        views=(),
        crop_grid_w=1,
        crop_grid_h=1,
        mode="text-only",
        prompt="fixture",
        rendered_prompt="fixture",
        max_new_tokens=0,
        max_length=0,
        no_repeat_ngram_size=0,
        no_repeat_window=0,
        tokenizer_path="synthetic-tokenizer.json",
        model_vocab_size=6,
    )


def test_read_bf16_tensor_as_f16_bits(tmp_path) -> None:
    values = np.array([[0.0, 1.0, -1.0], [2.0, -3.0, 0.25]], dtype=np.float32)
    payload = _bf16_payload(values)
    header = {
        "__metadata__": {"format": "pt"},
        "custom.weight": {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [0, len(payload)],
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with (tmp_path / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)

    actual = read_bf16_tensor_as_f16_bits(tmp_path, "custom.weight", expected_shape=(2, 3))
    expected = _expected_f16_bits_from_bf16_trunc(values)

    np.testing.assert_array_equal(actual, expected)


def test_read_bf16_rows_as_f16_bits(tmp_path) -> None:
    values = np.array(
        [
            [0.0, 1.0, -1.0, 0.5],
            [2.0, 3.0, -4.0, 8.0],
            [0.25, -0.25, 16.0, -16.0],
            [5.5, -5.5, 7.25, -7.25],
            [10.0, 11.0, 12.0, 13.0],
            [-2.5, 4.5, -6.5, 9.5],
        ],
        dtype=np.float32,
    )
    _write_tiny_embedding_safetensors(tmp_path, values)

    rows = np.array([0, 3, 5, 3], dtype=np.int32)
    actual = read_bf16_rows_as_f16_bits(tmp_path, rows, expected_shape=(6, 4))
    expected = _expected_f16_bits_from_bf16_trunc(values[rows])

    np.testing.assert_array_equal(actual, expected)


def test_dump_prompt_embedding_fixture_writes_native_c_files(tmp_path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4) / np.float32(4.0)
    _write_tiny_embedding_safetensors(tmp_path, values)
    request = _tiny_text_request(np.array([0, 5, 2], dtype=np.int32))
    out = tmp_path / "dump"

    dump_prompt_embedding_fixture(request, out, tmp_path, expected_shape=(6, 4))
    loaded = load_prompt_embedding_dump(out)

    np.testing.assert_array_equal(loaded.input_ids, request.input_ids)
    np.testing.assert_array_equal(loaded.image_mask, request.image_mask)
    np.testing.assert_array_equal(
        loaded.prompt_embeddings_f16_bits,
        _expected_f16_bits_from_bf16_trunc(values[request.input_ids]),
    )
    assert loaded.manifest["golden_tensors"]["prompt_embeddings"]["shape"] == [3, 4]
    assert loaded.manifest["native_binary_arrays"]["input_ids"]["file"] == "input_ids_i32.bin"
