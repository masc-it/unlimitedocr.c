"""Small Python golden-tensor dump helpers for C/Metal parity tests.

The full HF reference dumper will eventually hook model forward passes.  This
module starts with the narrow text-only dumps needed by Metal decoder bring-up:
read selected BF16 token-embedding rows and layer-0 weights from the local HF
safetensors file, convert them to fp16 exactly as the `.uocr` converter does,
and write compact fixtures that native tests can consume without parsing
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
TEXT_LAYER0_HIDDEN_BIN = "layer_0_hidden_f16.bin"
INPUT_IDS_BIN = "input_ids_i32.bin"
IMAGE_MASK_BIN = "image_mask_u8.bin"

ATTENTION_HEADS = 10
HEAD_DIM = 128
DENSE_LAYER0_INTERMEDIATE = 6848
ROPE_THETA = 10000.0
RMS_NORM_EPS = 1.0e-6

LAYER0_TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "model.layers.0.input_layernorm.weight": (HIDDEN_SIZE,),
    "model.layers.0.self_attn.q_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
    "model.layers.0.self_attn.k_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
    "model.layers.0.self_attn.v_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
    "model.layers.0.self_attn.o_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
    "model.layers.0.post_attention_layernorm.weight": (HIDDEN_SIZE,),
    "model.layers.0.mlp.gate_proj.weight": (DENSE_LAYER0_INTERMEDIATE, HIDDEN_SIZE),
    "model.layers.0.mlp.up_proj.weight": (DENSE_LAYER0_INTERMEDIATE, HIDDEN_SIZE),
    "model.layers.0.mlp.down_proj.weight": (HIDDEN_SIZE, DENSE_LAYER0_INTERMEDIATE),
}


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


@dataclass(frozen=True)
class TextLayer0Dump(PromptEmbeddingDump):
    layer0_hidden_f16_bits: NDArray[np.uint16]


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


def read_bf16_tensor_as_f16_bits(
    hf_dir: str | Path,
    tensor_name: str,
    *,
    expected_shape: tuple[int, ...],
) -> NDArray[np.uint16]:
    """Read a full BF16 safetensors tensor as fp16 bit patterns."""

    tensor = find_safetensors_tensor(hf_dir, tensor_name)
    if tensor.dtype != "BF16":
        raise ValueError(f"{tensor_name} must be BF16, got {tensor.dtype}")
    if tensor.shape != expected_shape:
        raise ValueError(f"{tensor_name} shape mismatch: expected {expected_shape}, got {tensor.shape}")

    value_count = int(np.prod(np.asarray(expected_shape, dtype=np.int64)))
    expected_bytes = value_count * 2
    if tensor.tensor_end - tensor.tensor_start != expected_bytes:
        raise ValueError(
            f"{tensor_name} byte-size mismatch: expected {expected_bytes}, got {tensor.tensor_end - tensor.tensor_start}"
        )

    with tensor.path.open("rb") as f:
        f.seek(tensor.data_start + tensor.tensor_start)
        raw = f.read(expected_bytes)
    if len(raw) != expected_bytes:
        raise ValueError(f"failed to read {tensor_name} from {tensor.path}")
    bf16_words = np.frombuffer(raw, dtype=np.dtype("<u2"), count=value_count)
    return bf16_words_to_f16_bits(bf16_words).reshape(expected_shape)


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


def _f16_to_f32(bits: NDArray[np.uint16]) -> NDArray[np.float32]:
    return np.asarray(bits, dtype=np.dtype("<u2")).view(np.dtype("<f2")).astype(np.float32)


def _f32_to_f16_bits(values: NDArray[np.floating[Any]]) -> NDArray[np.uint16]:
    return np.asarray(values, dtype=np.float32).astype(np.dtype("<f2")).view(np.dtype("<u2"))


def _f16_stats(bits: NDArray[np.uint16]) -> dict[str, float]:
    values = _f16_to_f32(bits)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _rmsnorm_f16_bits(src_bits: NDArray[np.uint16], weight_bits: NDArray[np.uint16]) -> NDArray[np.uint16]:
    src = _f16_to_f32(src_bits)
    weight = _f16_to_f32(weight_bits)
    variance = np.sum(src * src, axis=1, keepdims=True, dtype=np.float32) / np.float32(src.shape[1])
    scale = (np.float32(1.0) / np.sqrt(variance + np.float32(RMS_NORM_EPS))).astype(np.float32)
    return _f32_to_f16_bits(src * scale * weight.reshape((1, -1)))


def _dense_f16_bits(src_bits: NDArray[np.uint16], weight_bits: NDArray[np.uint16]) -> NDArray[np.uint16]:
    src = _f16_to_f32(src_bits)
    weight = _f16_to_f32(weight_bits)
    out = np.matmul(src, weight.T).astype(np.float32, copy=False)
    return _f32_to_f16_bits(out)


def _rope_qk_f16_bits(
    q_bits: NDArray[np.uint16],
    k_bits: NDArray[np.uint16],
    *,
    position_start: int = 0,
) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    n_tokens = int(q_bits.shape[0])
    q = _f16_to_f32(q_bits).reshape((n_tokens, ATTENTION_HEADS, HEAD_DIM))
    k = _f16_to_f32(k_bits).reshape((n_tokens, ATTENTION_HEADS, HEAD_DIM))
    q_out = q.copy()
    k_out = k.copy()
    half_dim = HEAD_DIM // 2
    pairs = np.arange(half_dim, dtype=np.float32)
    freq_scale = np.float32(-2.0) * np.float32(np.log2(np.float32(ROPE_THETA))) / np.float32(HEAD_DIM)
    inv_freq = np.exp2(pairs * freq_scale).astype(np.float32)
    for token in range(n_tokens):
        angles = (np.float32(position_start + token) * inv_freq).astype(np.float32)
        c = np.cos(angles).astype(np.float32)
        s = np.sin(angles).astype(np.float32)
        q0 = q[token, :, :half_dim]
        q1 = q[token, :, half_dim:]
        k0 = k[token, :, :half_dim]
        k1 = k[token, :, half_dim:]
        q_out[token, :, :half_dim] = q0 * c.reshape((1, half_dim)) - q1 * s.reshape((1, half_dim))
        q_out[token, :, half_dim:] = q0 * s.reshape((1, half_dim)) + q1 * c.reshape((1, half_dim))
        k_out[token, :, :half_dim] = k0 * c.reshape((1, half_dim)) - k1 * s.reshape((1, half_dim))
        k_out[token, :, half_dim:] = k0 * s.reshape((1, half_dim)) + k1 * c.reshape((1, half_dim))
    return _f32_to_f16_bits(q_out.reshape((n_tokens, HIDDEN_SIZE))), _f32_to_f16_bits(
        k_out.reshape((n_tokens, HIDDEN_SIZE))
    )


def _prefill_attention_f16_bits(
    q_bits: NDArray[np.uint16],
    k_bits: NDArray[np.uint16],
    v_bits: NDArray[np.uint16],
) -> NDArray[np.uint16]:
    n_tokens = int(q_bits.shape[0])
    q = _f16_to_f32(q_bits).reshape((n_tokens, ATTENTION_HEADS, HEAD_DIM))
    k = _f16_to_f32(k_bits).reshape((n_tokens, ATTENTION_HEADS, HEAD_DIM))
    v = _f16_to_f32(v_bits).reshape((n_tokens, ATTENTION_HEADS, HEAD_DIM))
    out = np.empty_like(q, dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(float(HEAD_DIM)))
    for token in range(n_tokens):
        for head in range(ATTENTION_HEADS):
            scores = np.empty((token + 1,), dtype=np.float32)
            for key in range(token + 1):
                scores[key] = np.sum(q[token, head, :] * k[key, head, :], dtype=np.float32) * scale
            scores -= np.max(scores)
            weights = np.exp(scores).astype(np.float32)
            weights /= np.sum(weights, dtype=np.float32)
            out[token, head, :] = np.sum(weights.reshape((-1, 1)) * v[: token + 1, head, :], axis=0, dtype=np.float32)
    return _f32_to_f16_bits(out.reshape((n_tokens, HIDDEN_SIZE)))


def _attention_output_residual_f16_bits(
    context_bits: NDArray[np.uint16],
    o_weight_bits: NDArray[np.uint16],
    residual_bits: NDArray[np.uint16],
) -> NDArray[np.uint16]:
    context = _f16_to_f32(context_bits)
    o_weight = _f16_to_f32(o_weight_bits)
    projected = np.matmul(context, o_weight.T).astype(np.float32, copy=False)
    residual = _f16_to_f32(residual_bits)
    return _f32_to_f16_bits(projected + residual)


def _dense_swiglu_residual_f16_bits(
    src_bits: NDArray[np.uint16],
    gate_weight_bits: NDArray[np.uint16],
    up_weight_bits: NDArray[np.uint16],
    down_weight_bits: NDArray[np.uint16],
    residual_bits: NDArray[np.uint16],
) -> NDArray[np.uint16]:
    src = _f16_to_f32(src_bits)
    gate_weight = _f16_to_f32(gate_weight_bits)
    up_weight = _f16_to_f32(up_weight_bits)
    gate = np.matmul(src, gate_weight.T).astype(np.float32, copy=False)
    up = np.matmul(src, up_weight.T).astype(np.float32, copy=False)
    silu = gate / (np.float32(1.0) + np.exp(-gate).astype(np.float32))
    mid_bits = _f32_to_f16_bits(silu * up)
    mid = _f16_to_f32(mid_bits)
    down_weight = _f16_to_f32(down_weight_bits)
    down = np.matmul(mid, down_weight.T).astype(np.float32, copy=False)
    residual = _f16_to_f32(residual_bits)
    return _f32_to_f16_bits(down + residual)


def compute_text_layer0_hidden_f16_bits(
    prompt_embeddings_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
) -> NDArray[np.uint16]:
    """Run the fp16 text-only decoder layer 0 reference used by Metal parity tests."""

    prompt = np.asarray(prompt_embeddings_f16_bits, dtype=np.dtype("<u2"))
    if prompt.ndim != 2 or prompt.shape[1] != HIDDEN_SIZE or prompt.shape[0] <= 0:
        raise ValueError(f"prompt embeddings must have shape [n,{HIDDEN_SIZE}], got {prompt.shape}")

    weights = {
        name: read_bf16_tensor_as_f16_bits(hf_dir, name, expected_shape=shape)
        for name, shape in LAYER0_TENSOR_SHAPES.items()
    }
    normed = _rmsnorm_f16_bits(prompt, weights["model.layers.0.input_layernorm.weight"])
    q = _dense_f16_bits(normed, weights["model.layers.0.self_attn.q_proj.weight"])
    k = _dense_f16_bits(normed, weights["model.layers.0.self_attn.k_proj.weight"])
    v = _dense_f16_bits(normed, weights["model.layers.0.self_attn.v_proj.weight"])
    q, k = _rope_qk_f16_bits(q, k)
    context = _prefill_attention_f16_bits(q, k, v)
    attn_hidden = _attention_output_residual_f16_bits(
        context,
        weights["model.layers.0.self_attn.o_proj.weight"],
        prompt,
    )
    mlp_input = _rmsnorm_f16_bits(attn_hidden, weights["model.layers.0.post_attention_layernorm.weight"])
    return _dense_swiglu_residual_f16_bits(
        mlp_input,
        weights["model.layers.0.mlp.gate_proj.weight"],
        weights["model.layers.0.mlp.up_proj.weight"],
        weights["model.layers.0.mlp.down_proj.weight"],
        attn_hidden,
    )


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


def dump_text_layer0_fixture(request: PreparedRequest, out_dir: str | Path, hf_dir: str | Path) -> Path:
    """Write prompt embeddings plus the fp16 output of decoder layer 0.

    This is intentionally text-only and fp16-specific. It gives the Metal
    decoder bring-up a compact layer-output target without requiring the full HF
    model object or vision stack in native tests.
    """

    out = dump_prompt_embedding_fixture(request, out_dir, hf_dir)
    prompt_dump = load_prompt_embedding_dump(out)
    layer0 = compute_text_layer0_hidden_f16_bits(prompt_dump.prompt_embeddings_f16_bits, hf_dir)
    (out / TEXT_LAYER0_HIDDEN_BIN).write_bytes(np.asarray(layer0, dtype=np.dtype("<u2")).tobytes())

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("golden_tensors", {})["layer_0_hidden"] = {
        "file": TEXT_LAYER0_HIDDEN_BIN,
        "dtype": "float16",
        "storage": "uint16_le",
        "shape": [int(prompt_dump.input_ids.size), HIDDEN_SIZE],
        "source": "fp16 numpy reference over BF16 safetensors converted to FP16",
        "reference_dtype_mode": "FP16 weights/activations with FP32 reductions; layer 0 text-only prefill",
        "stats": _f16_stats(layer0),
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


def load_text_layer0_dump(fixture_dir: str | Path) -> TextLayer0Dump:
    prompt = load_prompt_embedding_dump(fixture_dir)
    root = Path(fixture_dir)
    golden = prompt.manifest.get("golden_tensors", {}).get("layer_0_hidden", {})
    shape = golden.get("shape", [int(prompt.input_ids.size), HIDDEN_SIZE]) if isinstance(golden, dict) else []
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"invalid layer_0_hidden shape metadata in {root}")
    n_tokens = int(shape[0])
    hidden_size = int(shape[1])
    if n_tokens != int(prompt.input_ids.size) or hidden_size != HIDDEN_SIZE:
        raise ValueError(f"layer_0_hidden shape {shape} does not match prompt shape")
    layer0 = np.fromfile(root / TEXT_LAYER0_HIDDEN_BIN, dtype=np.dtype("<u2"))
    expected_values = n_tokens * hidden_size
    if layer0.size != expected_values:
        raise ValueError(f"layer_0_hidden size mismatch: expected {expected_values} f16 values, got {layer0.size}")
    return TextLayer0Dump(
        manifest=prompt.manifest,
        input_ids=prompt.input_ids,
        image_mask=prompt.image_mask,
        prompt_embeddings_f16_bits=prompt.prompt_embeddings_f16_bits,
        layer0_hidden_f16_bits=layer0.reshape((n_tokens, hidden_size)),
    )
