"""Small Python golden-tensor dump helpers for C/Metal parity tests.

The full HF reference dumper will eventually hook model forward passes.  This
module starts with the narrow text-only prompt-embedding dump needed by the
Metal decoder bring-up: read selected BF16 token-embedding rows from the local
HF safetensors file, convert them to fp16 exactly as the `.uocr` converter does,
and write a compact fixture that native tests can consume without parsing
`.npy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json
import struct

import numpy as np
from numpy.typing import NDArray

from .frontend import MODEL_VOCAB_SIZE, PreparedRequest, save_prepared_request

HIDDEN_SIZE = 1280


TOK_EMBED_TENSOR_NAME = "model.embed_tokens.weight"
PROMPT_EMBEDDINGS_BIN = "prompt_embeddings_f16.bin"
INPUT_IDS_BIN = "input_ids_i32.bin"
IMAGE_MASK_BIN = "image_mask_u8.bin"


@dataclass(frozen=True)
class SafetensorRange:
    path: Path
    header_length: int
    tensor_start: int
    tensor_end: int
    shape: tuple[int, ...]
    dtype: str

    @property
    def data_start(self) -> int:
        return 8 + self.header_length


@dataclass(frozen=True)
class PromptEmbeddingDump:
    manifest: dict[str, Any]
    input_ids: NDArray[np.int32]
    image_mask: NDArray[np.uint8]
    prompt_embeddings_f16_bits: NDArray[np.uint16]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _safetensors_candidates(hf_dir: Path, tensor_name: str) -> list[Path]:
    index = _read_json_if_exists(hf_dir / "model.safetensors.index.json")
    if index is not None:
        weight_map = index.get("weight_map", {})
        if isinstance(weight_map, dict):
            filename = weight_map.get(tensor_name)
            if isinstance(filename, str):
                indexed = hf_dir / filename
                if indexed.exists():
                    return [indexed]

    candidates = [hf_dir / "model-00001-of-000001.safetensors", hf_dir / "model.safetensors"]
    candidates.extend(sorted(path for path in hf_dir.glob("*.safetensors") if not path.name.endswith(".header")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            unique.append(candidate)
    return unique


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len == 0 or header_len > 256 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length {header_len} in {path}")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"truncated safetensors header in {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"expected safetensors header object in {path}")
    return header, int(header_len)


def find_safetensors_tensor(hf_dir: str | Path, tensor_name: str = TOK_EMBED_TENSOR_NAME) -> SafetensorRange:
    """Locate a tensor in local HF safetensors without loading its payload."""

    root = Path(hf_dir)
    for path in _safetensors_candidates(root, tensor_name):
        header, header_len = _read_safetensors_header(path)
        entry = header.get(tensor_name)
        if not isinstance(entry, dict):
            continue
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"malformed safetensors metadata for {tensor_name} in {path}")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end <= start:
            raise ValueError(f"invalid data offsets for {tensor_name} in {path}: {offsets}")
        return SafetensorRange(
            path=path,
            header_length=header_len,
            tensor_start=start,
            tensor_end=end,
            shape=tuple(int(dim) for dim in shape),
            dtype=dtype,
        )
    raise FileNotFoundError(f"could not find tensor {tensor_name!r} under {root}")


def bf16_words_to_f16_bits(words: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Convert little-endian BF16 words to fp16 bit patterns via fp32."""

    bf16 = np.asarray(words, dtype=np.dtype("<u2"))
    fp32_bits = (bf16.astype(np.uint32) << np.uint32(16)).astype(np.dtype("<u4"), copy=False)
    fp32 = fp32_bits.view(np.dtype("<f4"))
    return fp32.astype(np.dtype("<f2")).view(np.dtype("<u2"))


def read_bf16_rows_as_f16_bits(
    hf_dir: str | Path,
    row_ids: Sequence[int] | NDArray[np.integer[Any]],
    *,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> NDArray[np.uint16]:
    """Read selected rows from a BF16 safetensors matrix as fp16 bit patterns."""

    row_array = np.asarray(row_ids, dtype=np.int64)
    if row_array.ndim != 1 or row_array.size == 0:
        raise ValueError("row_ids must be a non-empty 1-D array")

    tensor = find_safetensors_tensor(hf_dir, tensor_name)
    if tensor.dtype != "BF16":
        raise ValueError(f"{tensor_name} must be BF16, got {tensor.dtype}")
    if tensor.shape != expected_shape:
        raise ValueError(f"{tensor_name} shape mismatch: expected {expected_shape}, got {tensor.shape}")

    vocab_size, hidden_size = expected_shape
    if np.any(row_array < 0) or np.any(row_array >= vocab_size):
        bad = int(row_array[(row_array < 0) | (row_array >= vocab_size)][0])
        raise ValueError(f"row id {bad} is outside [0,{vocab_size})")

    row_bytes = hidden_size * 2
    expected_bytes = vocab_size * row_bytes
    if tensor.tensor_end - tensor.tensor_start != expected_bytes:
        raise ValueError(
            f"{tensor_name} byte-size mismatch: expected {expected_bytes}, got {tensor.tensor_end - tensor.tensor_start}"
        )

    out = np.empty((int(row_array.size), hidden_size), dtype=np.dtype("<u2"))
    with tensor.path.open("rb") as f:
        for index, row_id in enumerate(row_array):
            offset = tensor.data_start + tensor.tensor_start + int(row_id) * row_bytes
            f.seek(offset)
            raw = f.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError(f"failed to read row {int(row_id)} from {tensor.path}")
            bf16_words = np.frombuffer(raw, dtype=np.dtype("<u2"), count=hidden_size)
            out[index, :] = bf16_words_to_f16_bits(bf16_words)
    return out


def _f16_stats(bits: NDArray[np.uint16]) -> dict[str, float]:
    values = np.asarray(bits, dtype=np.dtype("<u2")).view(np.dtype("<f2")).astype(np.float32)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def dump_prompt_embedding_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    *,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> Path:
    """Write a prepared-request fixture plus text-only prompt embeddings.

    The current Metal parity target is text-only prompt assembly.  Image prompt
    dumps need projected visual features and will be added with the image-embed
    debug path.
    """

    if np.any(request.image_mask != 0):
        raise NotImplementedError("prompt embedding dumps currently require text-only requests")

    out = Path(out_dir)
    save_prepared_request(request, out)
    ids_le = np.asarray(request.input_ids, dtype=np.dtype("<i4"))
    mask_u8 = np.asarray(request.image_mask, dtype=np.uint8)
    embeddings = read_bf16_rows_as_f16_bits(hf_dir, ids_le, tensor_name=tensor_name, expected_shape=expected_shape)
    hidden_size = expected_shape[1]

    (out / INPUT_IDS_BIN).write_bytes(ids_le.tobytes())
    (out / IMAGE_MASK_BIN).write_bytes(mask_u8.tobytes())
    (out / PROMPT_EMBEDDINGS_BIN).write_bytes(np.asarray(embeddings, dtype=np.dtype("<u2")).tobytes())

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["golden_tensors"] = {
        "prompt_embeddings": {
            "file": PROMPT_EMBEDDINGS_BIN,
            "dtype": "float16",
            "storage": "uint16_le",
            "shape": [int(request.n_tokens), hidden_size],
            "source": tensor_name,
            "reference_dtype_mode": "BF16 safetensors rows converted to FP16",
            "stats": _f16_stats(embeddings),
        }
    }
    manifest["native_binary_arrays"] = {
        "input_ids": {"file": INPUT_IDS_BIN, "dtype": "int32_le", "shape": [int(request.n_tokens)]},
        "image_mask": {"file": IMAGE_MASK_BIN, "dtype": "uint8", "shape": [int(request.n_tokens)]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_prompt_embedding_dump(fixture_dir: str | Path) -> PromptEmbeddingDump:
    root = Path(fixture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    input_ids = np.fromfile(root / INPUT_IDS_BIN, dtype=np.dtype("<i4")).astype(np.int32, copy=False)
    image_mask = np.fromfile(root / IMAGE_MASK_BIN, dtype=np.uint8)
    embeddings = np.fromfile(root / PROMPT_EMBEDDINGS_BIN, dtype=np.dtype("<u2"))
    if input_ids.ndim != 1 or input_ids.size == 0:
        raise ValueError(f"empty input ids in {root}")
    golden = manifest.get("golden_tensors", {}).get("prompt_embeddings", {}) if isinstance(manifest, dict) else {}
    shape = (
        golden.get("shape", [int(input_ids.size), HIDDEN_SIZE])
        if isinstance(golden, dict)
        else [int(input_ids.size), HIDDEN_SIZE]
    )
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"invalid prompt embedding shape metadata in {root}")
    n_tokens = int(shape[0])
    hidden_size = int(shape[1])
    if n_tokens != int(input_ids.size) or hidden_size <= 0:
        raise ValueError(f"prompt embedding shape {shape} does not match {input_ids.size} input ids")
    expected_values = n_tokens * hidden_size
    if embeddings.size != expected_values:
        raise ValueError(f"prompt embedding size mismatch: expected {expected_values} f16 values, got {embeddings.size}")
    return PromptEmbeddingDump(
        manifest=manifest,
        input_ids=input_ids,
        image_mask=image_mask,
        prompt_embeddings_f16_bits=embeddings.reshape((n_tokens, hidden_size)),
    )
