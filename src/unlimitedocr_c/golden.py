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

from .frontend import EOS_TOKEN_ID, MODEL_VOCAB_SIZE, PreparedRequest, load_tokenizer, save_prepared_request

HIDDEN_SIZE = 1280


TOK_EMBED_TENSOR_NAME = "model.embed_tokens.weight"
FINAL_NORM_TENSOR_NAME = "model.norm.weight"
LM_HEAD_TENSOR_NAME = "lm_head.weight"
PROMPT_EMBEDDINGS_BIN = "prompt_embeddings_f16.bin"
TEXT_LAYER0_HIDDEN_BIN = "layer_0_hidden_f16.bin"
TEXT_LAYER1_HIDDEN_BIN = "layer_1_hidden_f16.bin"
LOGITS_TOPK_IDS_BIN = "logits_topk_ids_i32.bin"
LOGITS_TOPK_SCORES_BIN = "logits_topk_scores_f32.bin"
GENERATED_IDS_BIN = "generated_ids_i32.bin"
GENERATED_TEXT_TXT = "generated_text.txt"
VISUAL_FEATURES_BIN = "visual_features_f16.bin"
SAM_FEATURES_BIN = "sam_features_f16.bin"
SAM_FEATURE_CHANNELS = 1024
SAM_GLOBAL_GRID = 16
SAM_LOCAL_GRID = 10
CLIP_FEATURES_BIN = "clip_features_f16.bin"
CLIP_HIDDEN_SIZE = 1024
CLIP_GLOBAL_TOKENS = 257
CLIP_LOCAL_TOKENS = 101
TEXT_DECODER_LAYER_COUNT = 12
DEFAULT_LOGITS_TOP_K = 16
INPUT_IDS_BIN = "input_ids_i32.bin"
IMAGE_MASK_BIN = "image_mask_u8.bin"
ROUTER_TOP_IDS_DTYPE = np.dtype("<u4")
ROUTER_TOP_WEIGHTS_DTYPE = np.dtype("<f4")

ATTENTION_HEADS = 10
HEAD_DIM = 128
DENSE_LAYER0_INTERMEDIATE = 6848
MOE_EXPERTS = 64
MOE_TOP_K = 6
MOE_EXPERT_INTERMEDIATE = 896
MOE_SHARED_INTERMEDIATE = 1792
ROPE_THETA = 10000.0
RMS_NORM_EPS = 1.0e-6


def _attention_tensor_shapes(layer: int) -> dict[str, tuple[int, ...]]:
    prefix = f"model.layers.{layer}"
    return {
        f"{prefix}.input_layernorm.weight": (HIDDEN_SIZE,),
        f"{prefix}.self_attn.q_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        f"{prefix}.self_attn.k_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        f"{prefix}.self_attn.v_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        f"{prefix}.self_attn.o_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        f"{prefix}.post_attention_layernorm.weight": (HIDDEN_SIZE,),
    }


LAYER0_TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    **_attention_tensor_shapes(0),
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


@dataclass(frozen=True)
class TextLayer1Dump(TextLayer0Dump):
    layer1_hidden_f16_bits: NDArray[np.uint16]


@dataclass(frozen=True)
class TextDecoderLayersDump(PromptEmbeddingDump):
    layer_hidden_f16_bits: tuple[NDArray[np.uint16], ...]


@dataclass(frozen=True)
class TextLogitsTopKDump(TextDecoderLayersDump):
    logits_topk_ids: NDArray[np.int32]
    logits_topk_scores_f32: NDArray[np.float32]


@dataclass(frozen=True)
class TextGeneratedIdsDump(TextLogitsTopKDump):
    generated_ids: NDArray[np.int32]


@dataclass(frozen=True)
class ImagePromptEmbeddingDump(PromptEmbeddingDump):
    visual_features_f16_bits: NDArray[np.uint16]
    image_span_start: int
    image_span_length: int


@dataclass(frozen=True)
class ImageDecoderLayersDump(ImagePromptEmbeddingDump):
    layer_hidden_f16_bits: tuple[NDArray[np.uint16], ...]
    router_top_ids: dict[int, NDArray[np.uint32]]
    router_top_weights_f32: dict[int, NDArray[np.float32]]


@dataclass(frozen=True)
class ImageLogitsTopKDump(ImageDecoderLayersDump):
    logits_topk_ids: NDArray[np.int32]
    logits_topk_scores_f32: NDArray[np.float32]


@dataclass(frozen=True)
class ImageGeneratedIdsDump(ImageLogitsTopKDump):
    generated_ids: NDArray[np.int32]
    generated_text: str


def text_layer_hidden_filename(layer: int) -> str:
    if layer < 0 or layer >= TEXT_DECODER_LAYER_COUNT:
        raise ValueError(f"decoder layer out of range: {layer}")
    return f"layer_{layer}_hidden_f16.bin"


def router_top_ids_filename(layer: int) -> str:
    if layer <= 0 or layer >= TEXT_DECODER_LAYER_COUNT:
        raise ValueError(f"MoE router layer out of range: {layer}")
    return f"layer_{layer}_router_top_ids_u32.bin"


def router_top_weights_filename(layer: int) -> str:
    if layer <= 0 or layer >= TEXT_DECODER_LAYER_COUNT:
        raise ValueError(f"MoE router layer out of range: {layer}")
    return f"layer_{layer}_router_top_weights_f32.bin"


def sam_features_filename(view_index: int) -> str:
    if view_index < 0:
        raise ValueError(f"view_index must be non-negative, got {view_index}")
    return SAM_FEATURES_BIN if view_index == 0 else f"sam_features_view_{view_index}_f16.bin"


def clip_features_filename(view_index: int) -> str:
    if view_index < 0:
        raise ValueError(f"view_index must be non-negative, got {view_index}")
    return CLIP_FEATURES_BIN if view_index == 0 else f"clip_features_view_{view_index}_f16.bin"


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


def _validate_hidden_bits(name: str, hidden_bits: NDArray[np.uint16]) -> NDArray[np.uint16]:
    hidden = np.asarray(hidden_bits, dtype=np.dtype("<u2"))
    if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_SIZE or hidden.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [n,{HIDDEN_SIZE}], got {hidden.shape}")
    return hidden


def _read_named_tensors(hf_dir: str | Path, shapes: dict[str, tuple[int, ...]]) -> dict[str, NDArray[np.uint16]]:
    return {name: read_bf16_tensor_as_f16_bits(hf_dir, name, expected_shape=shape) for name, shape in shapes.items()}


def _attention_block_f16_bits(
    hidden_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    layer: int,
) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    hidden = _validate_hidden_bits("attention input", hidden_bits)
    weights = _read_named_tensors(hf_dir, _attention_tensor_shapes(layer))
    prefix = f"model.layers.{layer}"
    normed = _rmsnorm_f16_bits(hidden, weights[f"{prefix}.input_layernorm.weight"])
    q = _dense_f16_bits(normed, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = _dense_f16_bits(normed, weights[f"{prefix}.self_attn.k_proj.weight"])
    v = _dense_f16_bits(normed, weights[f"{prefix}.self_attn.v_proj.weight"])
    q, k = _rope_qk_f16_bits(q, k)
    context = _prefill_attention_f16_bits(q, k, v)
    attn_hidden = _attention_output_residual_f16_bits(
        context,
        weights[f"{prefix}.self_attn.o_proj.weight"],
        hidden,
    )
    mlp_input = _rmsnorm_f16_bits(attn_hidden, weights[f"{prefix}.post_attention_layernorm.weight"])
    return attn_hidden, mlp_input


def compute_text_layer0_hidden_f16_bits(
    prompt_embeddings_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
) -> NDArray[np.uint16]:
    """Run the fp16 text-only decoder layer 0 reference used by Metal parity tests."""

    prompt = _validate_hidden_bits("prompt embeddings", prompt_embeddings_f16_bits)
    attn_hidden, mlp_input = _attention_block_f16_bits(prompt, hf_dir, layer=0)
    weights = _read_named_tensors(
        hf_dir,
        {
            name: shape
            for name, shape in LAYER0_TENSOR_SHAPES.items()
            if ".mlp." in name
        },
    )
    return _dense_swiglu_residual_f16_bits(
        mlp_input,
        weights["model.layers.0.mlp.gate_proj.weight"],
        weights["model.layers.0.mlp.up_proj.weight"],
        weights["model.layers.0.mlp.down_proj.weight"],
        attn_hidden,
    )


def _moe_router_f16(
    hidden_bits: NDArray[np.uint16],
    router_weight_bits: NDArray[np.uint16],
) -> tuple[NDArray[np.float32], NDArray[np.uint32], NDArray[np.float32]]:
    hidden = _f16_to_f32(hidden_bits)
    router_weight = _f16_to_f32(router_weight_bits)
    logits = np.matmul(hidden, router_weight.T).astype(np.float32, copy=False)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted).astype(np.float32)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True, dtype=np.float32)
    top_ids = np.empty((hidden.shape[0], MOE_TOP_K), dtype=np.uint32)
    top_weights = np.empty((hidden.shape[0], MOE_TOP_K), dtype=np.float32)
    expert_ids = np.arange(MOE_EXPERTS, dtype=np.uint32)
    for token in range(hidden.shape[0]):
        order = np.lexsort((expert_ids, -probs[token]))
        chosen = order[:MOE_TOP_K].astype(np.uint32, copy=False)
        top_ids[token, :] = chosen
        top_weights[token, :] = probs[token, chosen]
    return probs, top_ids, top_weights


def _expert_weight_names(layer: int, expert: int) -> tuple[str, str, str]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return (
        f"{prefix}.gate_proj.weight",
        f"{prefix}.up_proj.weight",
        f"{prefix}.down_proj.weight",
    )


def _read_expert_weights(
    hf_dir: str | Path,
    layer: int,
    expert: int,
) -> tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16]]:
    gate_name, up_name, down_name = _expert_weight_names(layer, expert)
    gate = read_bf16_tensor_as_f16_bits(
        hf_dir,
        gate_name,
        expected_shape=(MOE_EXPERT_INTERMEDIATE, HIDDEN_SIZE),
    )
    up = read_bf16_tensor_as_f16_bits(
        hf_dir,
        up_name,
        expected_shape=(MOE_EXPERT_INTERMEDIATE, HIDDEN_SIZE),
    )
    down = read_bf16_tensor_as_f16_bits(
        hf_dir,
        down_name,
        expected_shape=(HIDDEN_SIZE, MOE_EXPERT_INTERMEDIATE),
    )
    return gate, up, down


def _moe_selected_routed_f16_bits(
    hidden_bits: NDArray[np.uint16],
    top_ids: NDArray[np.uint32],
    top_weights: NDArray[np.float32],
    hf_dir: str | Path,
    *,
    layer: int,
) -> NDArray[np.uint16]:
    hidden = _validate_hidden_bits("MoE input", hidden_bits)
    hidden_f32 = _f16_to_f32(hidden)
    routed = np.zeros((hidden.shape[0], HIDDEN_SIZE), dtype=np.float32)
    weight_cache: dict[int, tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16]]] = {}
    for token in range(hidden.shape[0]):
        token_src = hidden_f32[token : token + 1]
        for rank in range(MOE_TOP_K):
            expert = int(top_ids[token, rank])
            weights = weight_cache.get(expert)
            if weights is None:
                weights = _read_expert_weights(hf_dir, layer, expert)
                weight_cache[expert] = weights
            gate_bits, up_bits, down_bits = weights
            gate = np.matmul(token_src, _f16_to_f32(gate_bits).T).astype(np.float32, copy=False)
            up = np.matmul(token_src, _f16_to_f32(up_bits).T).astype(np.float32, copy=False)
            silu = gate / (np.float32(1.0) + np.exp(-gate).astype(np.float32))
            mid_bits = _f32_to_f16_bits(silu * up)
            mid = _f16_to_f32(mid_bits)
            down = np.matmul(mid, _f16_to_f32(down_bits).T).astype(np.float32, copy=False)
            routed[token, :] += down[0, :] * np.float32(top_weights[token, rank])
    return _f32_to_f16_bits(routed)


def _moe_shared_f16_bits(hidden_bits: NDArray[np.uint16], hf_dir: str | Path, *, layer: int) -> NDArray[np.uint16]:
    prefix = f"model.layers.{layer}.mlp.shared_experts"
    weights = _read_named_tensors(
        hf_dir,
        {
            f"{prefix}.gate_proj.weight": (MOE_SHARED_INTERMEDIATE, HIDDEN_SIZE),
            f"{prefix}.up_proj.weight": (MOE_SHARED_INTERMEDIATE, HIDDEN_SIZE),
            f"{prefix}.down_proj.weight": (HIDDEN_SIZE, MOE_SHARED_INTERMEDIATE),
        },
    )
    zeros = np.zeros_like(_validate_hidden_bits("shared expert input", hidden_bits), dtype=np.dtype("<u2"))
    return _dense_swiglu_residual_f16_bits(
        hidden_bits,
        weights[f"{prefix}.gate_proj.weight"],
        weights[f"{prefix}.up_proj.weight"],
        weights[f"{prefix}.down_proj.weight"],
        zeros,
    )


def _moe_layer_f16_bits_with_router(
    mlp_input_bits: NDArray[np.uint16],
    residual_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    layer: int,
) -> tuple[NDArray[np.uint16], NDArray[np.uint32], NDArray[np.float32]]:
    prefix = f"model.layers.{layer}.mlp"
    router_weight = read_bf16_tensor_as_f16_bits(
        hf_dir,
        f"{prefix}.gate.weight",
        expected_shape=(MOE_EXPERTS, HIDDEN_SIZE),
    )
    _, top_ids, top_weights = _moe_router_f16(mlp_input_bits, router_weight)
    routed = _moe_selected_routed_f16_bits(mlp_input_bits, top_ids, top_weights, hf_dir, layer=layer)
    shared = _moe_shared_f16_bits(mlp_input_bits, hf_dir, layer=layer)
    combined = _f16_to_f32(routed) + _f16_to_f32(shared) + _f16_to_f32(residual_bits)
    return _f32_to_f16_bits(combined), top_ids, top_weights


def _moe_layer_f16_bits(
    mlp_input_bits: NDArray[np.uint16],
    residual_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    layer: int,
) -> NDArray[np.uint16]:
    hidden, _, _ = _moe_layer_f16_bits_with_router(mlp_input_bits, residual_bits, hf_dir, layer=layer)
    return hidden


def compute_text_layer1_hidden_f16_bits(
    layer0_hidden_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
) -> NDArray[np.uint16]:
    """Run the fp16 text-only decoder layer 1 MoE reference used by Metal parity tests."""

    layer0 = _validate_hidden_bits("layer0 hidden", layer0_hidden_f16_bits)
    attn_hidden, mlp_input = _attention_block_f16_bits(layer0, hf_dir, layer=1)
    return _moe_layer_f16_bits(mlp_input, attn_hidden, hf_dir, layer=1)


def compute_text_decoder_layer_hidden_f16_bits(
    prompt_embeddings_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    layer_count: int = TEXT_DECODER_LAYER_COUNT,
) -> tuple[NDArray[np.uint16], ...]:
    """Run the fp16 text-only decoder reference through ``layer_count`` layers."""

    if layer_count < 1 or layer_count > TEXT_DECODER_LAYER_COUNT:
        raise ValueError(f"layer_count must be in [1,{TEXT_DECODER_LAYER_COUNT}], got {layer_count}")
    hidden = compute_text_layer0_hidden_f16_bits(prompt_embeddings_f16_bits, hf_dir)
    layers: list[NDArray[np.uint16]] = [hidden]
    for layer in range(1, layer_count):
        attn_hidden, mlp_input = _attention_block_f16_bits(hidden, hf_dir, layer=layer)
        hidden = _moe_layer_f16_bits(mlp_input, attn_hidden, hf_dir, layer=layer)
        layers.append(hidden)
    return tuple(layers)


def _lm_head_topk_from_normed_f16_bits(
    normed_hidden_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    top_k: int = DEFAULT_LOGITS_TOP_K,
    chunk_rows: int = 4096,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    normed = np.asarray(normed_hidden_f16_bits, dtype=np.dtype("<u2"))
    if normed.shape != (HIDDEN_SIZE,):
        raise ValueError(f"normalized hidden row must have shape [{HIDDEN_SIZE}], got {normed.shape}")
    if top_k <= 0 or top_k > MODEL_VOCAB_SIZE:
        raise ValueError(f"top_k must be in [1,{MODEL_VOCAB_SIZE}], got {top_k}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    tensor = find_safetensors_tensor(hf_dir, LM_HEAD_TENSOR_NAME)
    expected_shape = (MODEL_VOCAB_SIZE, HIDDEN_SIZE)
    if tensor.dtype != "BF16":
        raise ValueError(f"{LM_HEAD_TENSOR_NAME} must be BF16, got {tensor.dtype}")
    if tensor.shape != expected_shape:
        raise ValueError(f"{LM_HEAD_TENSOR_NAME} shape mismatch: expected {expected_shape}, got {tensor.shape}")

    row_bytes = HIDDEN_SIZE * 2
    expected_bytes = MODEL_VOCAB_SIZE * row_bytes
    if tensor.tensor_end - tensor.tensor_start != expected_bytes:
        raise ValueError(
            f"{LM_HEAD_TENSOR_NAME} byte-size mismatch: expected {expected_bytes}, "
            f"got {tensor.tensor_end - tensor.tensor_start}"
        )

    hidden = _f16_to_f32(normed)
    best_ids = np.empty((0,), dtype=np.int32)
    best_scores = np.empty((0,), dtype=np.float32)
    with tensor.path.open("rb") as f:
        for row_start in range(0, MODEL_VOCAB_SIZE, chunk_rows):
            rows = min(chunk_rows, MODEL_VOCAB_SIZE - row_start)
            f.seek(tensor.data_start + tensor.tensor_start + row_start * row_bytes)
            raw = f.read(rows * row_bytes)
            if len(raw) != rows * row_bytes:
                raise ValueError(f"failed to read {LM_HEAD_TENSOR_NAME} rows {row_start}..{row_start + rows}")
            bf16_words = np.frombuffer(raw, dtype=np.dtype("<u2"), count=rows * HIDDEN_SIZE).reshape(
                (rows, HIDDEN_SIZE)
            )
            weight = _f16_to_f32(bf16_words_to_f16_bits(bf16_words))
            scores = np.matmul(weight, hidden).astype(np.float32, copy=False)
            scores = np.where(np.isfinite(scores), scores, np.float32(-np.inf)).astype(np.float32, copy=False)
            ids = (np.arange(rows, dtype=np.int32) + np.int32(row_start)).astype(np.int32, copy=False)
            if best_ids.size:
                ids = np.concatenate((best_ids, ids))
                scores = np.concatenate((best_scores, scores))
            order = np.lexsort((ids, -scores))[:top_k]
            best_ids = ids[order].astype(np.int32, copy=False)
            best_scores = scores[order].astype(np.float32, copy=False)
    return best_ids, best_scores


def compute_text_logits_topk_f32(
    final_hidden_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    top_k: int = DEFAULT_LOGITS_TOP_K,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Compute fp32 LM-head top-k logits for the next token after a text-only prompt."""

    hidden = _validate_hidden_bits("final decoder hidden", final_hidden_f16_bits)
    final_norm_weight = read_bf16_tensor_as_f16_bits(
        hf_dir,
        FINAL_NORM_TENSOR_NAME,
        expected_shape=(HIDDEN_SIZE,),
    )
    normed = _rmsnorm_f16_bits(hidden[-1:, :], final_norm_weight)[0]
    return _lm_head_topk_from_normed_f16_bits(normed, hf_dir, top_k=top_k)


def compute_text_generated_ids(
    final_hidden_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    max_new_tokens: int = 1,
) -> NDArray[np.int32]:
    """Compute deterministic greedy generated ids supported by the current text-only parity path."""

    if max_new_tokens != 1:
        raise ValueError("text generated-id parity currently supports exactly one greedy token")
    ids, _ = compute_text_logits_topk_f32(final_hidden_f16_bits, hf_dir, top_k=1)
    return ids.astype(np.int32, copy=False)


def decode_generated_text(generated_ids: NDArray[np.integer[Any]], tokenizer_path: str | Path) -> str:
    """Decode generated token ids the same way the upstream image path returns text."""

    ids = np.asarray(generated_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"generated ids must be rank-1, got shape {ids.shape}")
    if np.any(ids < 0) or np.any(ids >= MODEL_VOCAB_SIZE):
        raise ValueError("generated ids contain values outside the model vocabulary")
    tokenizer = load_tokenizer(tokenizer_path)
    text = tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=False)
    eos_text = tokenizer.decode([EOS_TOKEN_ID], skip_special_tokens=False)
    if eos_text and text.endswith(eos_text):
        text = text[: -len(eos_text)]
    return text.strip()


def _contiguous_image_span(mask: NDArray[np.uint8]) -> tuple[int, int]:
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.ndim != 1:
        raise ValueError(f"image mask must be rank-1, got shape {mask_arr.shape}")
    image_indices = np.flatnonzero(mask_arr != 0)
    if image_indices.size == 0:
        raise ValueError("image prompt fixture requires at least one image placeholder")
    if np.any(mask_arr[image_indices] != 1):
        raise ValueError("image mask values must be 0 or 1")
    start = int(image_indices[0])
    end = int(image_indices[-1]) + 1
    if end - start != int(image_indices.size):
        raise ValueError("image placeholders must occupy one contiguous span")
    return start, end - start


def _validate_visual_features_f16_bits(
    visual_features_f16_bits: NDArray[np.uint16],
    *,
    expected_rows: int,
    hidden_size: int,
) -> NDArray[np.uint16]:
    visual = np.asarray(visual_features_f16_bits, dtype=np.dtype("<u2"))
    if visual.ndim == 1:
        if visual.size != expected_rows * hidden_size:
            raise ValueError(
                f"visual feature flat size mismatch: expected {expected_rows * hidden_size}, got {visual.size}"
            )
        visual = visual.reshape((expected_rows, hidden_size))
    if visual.shape != (expected_rows, hidden_size):
        raise ValueError(f"visual features must have shape {(expected_rows, hidden_size)}, got {visual.shape}")
    return np.ascontiguousarray(visual, dtype=np.dtype("<u2"))


def compute_prompt_embeddings_with_visual_features_f16_bits(
    request: PreparedRequest,
    visual_features_f16_bits: NDArray[np.uint16],
    hf_dir: str | Path,
    *,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> NDArray[np.uint16]:
    """Assemble prompt embeddings by splicing dumped visual features into one image span."""

    if request.image_mask.shape != request.input_ids.shape:
        raise ValueError("input_ids and image_mask must have matching shapes")
    image_span_start, image_span_length = _contiguous_image_span(request.image_mask)
    hidden_size = int(expected_shape[1])
    visual = _validate_visual_features_f16_bits(
        visual_features_f16_bits,
        expected_rows=image_span_length,
        hidden_size=hidden_size,
    )
    embeddings = read_bf16_rows_as_f16_bits(
        hf_dir,
        np.asarray(request.input_ids, dtype=np.dtype("<i4")),
        tensor_name=tensor_name,
        expected_shape=expected_shape,
    )
    embeddings = np.array(embeddings, dtype=np.dtype("<u2"), copy=True)
    embeddings[image_span_start : image_span_start + image_span_length, :] = visual
    return embeddings


def dump_prompt_embedding_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    *,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> Path:
    """Write a prepared-request fixture plus text-only prompt embeddings."""

    if np.any(request.image_mask != 0):
        raise NotImplementedError(
            "text prompt embedding dumps require text-only requests; use dump_image_prompt_embedding_fixture"
        )

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


def dump_image_prompt_embedding_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    visual_features_f16_bits: NDArray[np.uint16],
    *,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> Path:
    """Write an image-prompt fixture using precomputed formatted visual feature rows.

    visual_features_f16_bits must already be the final projected/formatted
    visual token sequence, including newline/separator rows, in the same order
    as the image placeholder span. This bypasses the C vision encoder while
    keeping prompt assembly and decoder inputs identical to the full path.
    """

    out = Path(out_dir)
    save_prepared_request(request, out)
    ids_le = np.asarray(request.input_ids, dtype=np.dtype("<i4"))
    mask_u8 = np.asarray(request.image_mask, dtype=np.uint8)
    image_span_start, image_span_length = _contiguous_image_span(mask_u8)
    hidden_size = int(expected_shape[1])
    visual = _validate_visual_features_f16_bits(
        visual_features_f16_bits,
        expected_rows=image_span_length,
        hidden_size=hidden_size,
    )
    embeddings = compute_prompt_embeddings_with_visual_features_f16_bits(
        request,
        visual,
        hf_dir,
        tensor_name=tensor_name,
        expected_shape=expected_shape,
    )

    (out / INPUT_IDS_BIN).write_bytes(ids_le.tobytes())
    (out / IMAGE_MASK_BIN).write_bytes(mask_u8.tobytes())
    (out / VISUAL_FEATURES_BIN).write_bytes(visual.tobytes())
    (out / PROMPT_EMBEDDINGS_BIN).write_bytes(np.asarray(embeddings, dtype=np.dtype("<u2")).tobytes())

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["image_embedding_fixture"] = {
        "bypasses_c_vision_encoder": True,
        "image_span_start": image_span_start,
        "image_span_length": image_span_length,
        "visual_features_file": VISUAL_FEATURES_BIN,
        "feature_order": "preformatted rows matching the contiguous image placeholder span",
    }
    manifest["golden_tensors"] = {
        "visual_features": {
            "file": VISUAL_FEATURES_BIN,
            "dtype": "float16",
            "storage": "uint16_le",
            "shape": [image_span_length, hidden_size],
            "source": "precomputed Python visual encoder/projector fixture",
            "reference_dtype_mode": "already formatted projected visual features in FP16",
            "stats": _f16_stats(visual),
        },
        "prompt_embeddings": {
            "file": PROMPT_EMBEDDINGS_BIN,
            "dtype": "float16",
            "storage": "uint16_le",
            "shape": [int(request.n_tokens), hidden_size],
            "source": f"{tensor_name} plus {VISUAL_FEATURES_BIN}",
            "reference_dtype_mode": "BF16 text-token rows converted to FP16 with visual rows spliced directly",
            "stats": _f16_stats(embeddings),
        },
    }
    manifest["native_binary_arrays"] = {
        "input_ids": {"file": INPUT_IDS_BIN, "dtype": "int32_le", "shape": [int(request.n_tokens)]},
        "image_mask": {"file": IMAGE_MASK_BIN, "dtype": "uint8", "shape": [int(request.n_tokens)]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _add_hidden_tensor_manifest(
    manifest: dict[str, Any],
    key: str,
    filename: str,
    n_tokens: int,
    stats_bits: NDArray[np.uint16],
    reference_dtype_mode: str,
) -> None:
    manifest.setdefault("golden_tensors", {})[key] = {
        "file": filename,
        "dtype": "float16",
        "storage": "uint16_le",
        "shape": [int(n_tokens), HIDDEN_SIZE],
        "source": "fp16 numpy reference over BF16 safetensors converted to FP16",
        "reference_dtype_mode": reference_dtype_mode,
        "stats": _f16_stats(stats_bits),
    }


def dump_image_decoder_layers_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    visual_features_f16_bits: NDArray[np.uint16],
    *,
    layer_count: int = TEXT_DECODER_LAYER_COUNT,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> Path:
    """Write image prompt embeddings plus fp16 decoder layer outputs.

    This is the image-embedding counterpart to
    :func:`dump_text_decoder_layers_fixture`: visual features are supplied as
    already formatted fp16 rows and spliced into the prompt before running the
    same fp16 decoder reference.  It keeps the C/Metal parity path independent
    of the unfinished native vision encoder.
    """

    out = dump_image_prompt_embedding_fixture(
        request,
        out_dir,
        hf_dir,
        visual_features_f16_bits,
        tensor_name=tensor_name,
        expected_shape=expected_shape,
    )
    if layer_count < 1 or layer_count > TEXT_DECODER_LAYER_COUNT:
        raise ValueError(f"layer_count must be in [1,{TEXT_DECODER_LAYER_COUNT}], got {layer_count}")
    prompt_dump = load_image_prompt_embedding_dump(out)
    hidden = compute_text_layer0_hidden_f16_bits(prompt_dump.prompt_embeddings_f16_bits, hf_dir)
    layer_outputs: list[NDArray[np.uint16]] = [hidden]
    router_topk: dict[str, Any] = {}
    for layer in range(1, layer_count):
        attn_hidden, mlp_input = _attention_block_f16_bits(hidden, hf_dir, layer=layer)
        hidden, top_ids, top_weights = _moe_layer_f16_bits_with_router(mlp_input, attn_hidden, hf_dir, layer=layer)
        ids_file = router_top_ids_filename(layer)
        weights_file = router_top_weights_filename(layer)
        np.asarray(top_ids, dtype=ROUTER_TOP_IDS_DTYPE).tofile(out / ids_file)
        np.asarray(top_weights, dtype=ROUTER_TOP_WEIGHTS_DTYPE).tofile(out / weights_file)
        router_topk[f"layer_{layer}"] = {
            "ids_file": ids_file,
            "weights_file": weights_file,
            "ids_dtype": "uint32_le",
            "weights_dtype": "float32_le",
            "shape": [int(prompt_dump.input_ids.size), MOE_TOP_K],
            "source": f"model.layers.{layer}.mlp.gate.weight softmax/top-{MOE_TOP_K}",
            "reference_dtype_mode": "FP16 router inputs/weights with FP32 logits and softmax",
        }
        layer_outputs.append(hidden)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for layer, hidden in enumerate(layer_outputs):
        filename = text_layer_hidden_filename(layer)
        (out / filename).write_bytes(np.asarray(hidden, dtype=np.dtype("<u2")).tobytes())
        _add_hidden_tensor_manifest(
            manifest,
            f"layer_{layer}_hidden",
            filename,
            int(prompt_dump.input_ids.size),
            hidden,
            f"FP16 weights/activations with FP32 reductions; layer {layer} image-embedding decoder prefill",
        )
    manifest["image_decoder_layer_count"] = int(layer_count)
    manifest["router_topk"] = router_topk
    manifest["image_embedding_fixture"]["decoder_layers_file_pattern"] = "layer_{layer}_hidden_f16.bin"
    manifest["image_embedding_fixture"]["router_topk_file_pattern"] = "layer_{layer}_router_top_{ids,weights}.{u32,f32}.bin"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def dump_image_logits_topk_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    visual_features_f16_bits: NDArray[np.uint16],
    *,
    top_k: int = DEFAULT_LOGITS_TOP_K,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> Path:
    """Write image-embedding decoder layers plus next-token LM-head top-k logits."""

    out = dump_image_decoder_layers_fixture(
        request,
        out_dir,
        hf_dir,
        visual_features_f16_bits,
        layer_count=TEXT_DECODER_LAYER_COUNT,
        tensor_name=tensor_name,
        expected_shape=expected_shape,
    )
    layers = load_image_decoder_layers_dump(out)
    ids, scores = compute_text_logits_topk_f32(layers.layer_hidden_f16_bits[-1], hf_dir, top_k=top_k)
    np.asarray(ids, dtype=np.dtype("<i4")).tofile(out / LOGITS_TOPK_IDS_BIN)
    np.asarray(scores, dtype=np.dtype("<f4")).tofile(out / LOGITS_TOPK_SCORES_BIN)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("golden_tensors", {})["logits_topk"] = {
        "ids_file": LOGITS_TOPK_IDS_BIN,
        "scores_file": LOGITS_TOPK_SCORES_BIN,
        "ids_dtype": "int32_le",
        "scores_dtype": "float32_le",
        "shape": [int(top_k)],
        "source": f"image-embedding final RMSNorm {FINAL_NORM_TENSOR_NAME} and LM head {LM_HEAD_TENSOR_NAME}",
        "reference_dtype_mode": "FP16 image decoder final norm activations/weights with FP32 LM-head reductions",
        "rank_order": "score descending, token id ascending on ties",
    }
    manifest.setdefault("image_embedding_fixture", {})["logits_topk_ids_file"] = LOGITS_TOPK_IDS_BIN
    manifest.setdefault("image_embedding_fixture", {})["logits_topk_scores_file"] = LOGITS_TOPK_SCORES_BIN
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def dump_image_generated_ids_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    visual_features_f16_bits: NDArray[np.uint16],
    *,
    top_k: int = DEFAULT_LOGITS_TOP_K,
    tensor_name: str = TOK_EMBED_TENSOR_NAME,
    expected_shape: tuple[int, int] = (MODEL_VOCAB_SIZE, HIDDEN_SIZE),
) -> Path:
    """Write image logits top-k plus the first greedy generated token id and decoded text."""

    out = dump_image_logits_topk_fixture(
        request,
        out_dir,
        hf_dir,
        visual_features_f16_bits,
        top_k=max(1, int(top_k)),
        tensor_name=tensor_name,
        expected_shape=expected_shape,
    )
    logits = load_image_logits_topk_dump(out)
    generated = np.asarray(logits.logits_topk_ids[:1], dtype=np.dtype("<i4"))
    generated.tofile(out / GENERATED_IDS_BIN)
    generated_text = decode_generated_text(generated, request.tokenizer_path)
    (out / GENERATED_TEXT_TXT).write_text(generated_text, encoding="utf-8")

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("golden_tensors", {})["generated_ids"] = {
        "file": GENERATED_IDS_BIN,
        "dtype": "int32_le",
        "shape": [int(generated.size)],
        "source": "greedy argmax over image-embedding next-token logits without sampling or no-repeat bans",
        "reference_dtype_mode": "first deterministic image-embedding generated token from fp16 parity logits",
    }
    manifest.setdefault("golden_tensors", {})["generated_text"] = {
        "file": GENERATED_TEXT_TXT,
        "encoding": "utf-8",
        "shape": [1],
        "source": f"{GENERATED_IDS_BIN} decoded with {request.tokenizer_path}",
        "decode_mode": "skip_special_tokens=False, trim one trailing EOS marker, then strip whitespace",
    }
    manifest.setdefault("image_embedding_fixture", {})["generated_ids_file"] = GENERATED_IDS_BIN
    manifest.setdefault("image_embedding_fixture", {})["generated_text_file"] = GENERATED_TEXT_TXT
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
    _add_hidden_tensor_manifest(
        manifest,
        "layer_0_hidden",
        TEXT_LAYER0_HIDDEN_BIN,
        int(prompt_dump.input_ids.size),
        layer0,
        "FP16 weights/activations with FP32 reductions; layer 0 text-only prefill",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def dump_text_layer1_fixture(request: PreparedRequest, out_dir: str | Path, hf_dir: str | Path) -> Path:
    """Write prompt embeddings plus layer-0 and layer-1 fp16 decoder outputs."""

    out = dump_text_layer0_fixture(request, out_dir, hf_dir)
    layer0_dump = load_text_layer0_dump(out)
    layer1 = compute_text_layer1_hidden_f16_bits(layer0_dump.layer0_hidden_f16_bits, hf_dir)
    (out / TEXT_LAYER1_HIDDEN_BIN).write_bytes(np.asarray(layer1, dtype=np.dtype("<u2")).tobytes())

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _add_hidden_tensor_manifest(
        manifest,
        "layer_1_hidden",
        TEXT_LAYER1_HIDDEN_BIN,
        int(layer0_dump.input_ids.size),
        layer1,
        "FP16 weights/activations with FP32 reductions; layer 1 text-only MoE prefill",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def dump_text_decoder_layers_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    *,
    layer_count: int = TEXT_DECODER_LAYER_COUNT,
) -> Path:
    """Write prompt embeddings plus fp16 text-only decoder outputs for all layers."""

    out = dump_prompt_embedding_fixture(request, out_dir, hf_dir)
    prompt_dump = load_prompt_embedding_dump(out)
    layer_outputs = compute_text_decoder_layer_hidden_f16_bits(
        prompt_dump.prompt_embeddings_f16_bits,
        hf_dir,
        layer_count=layer_count,
    )

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for layer, hidden in enumerate(layer_outputs):
        filename = text_layer_hidden_filename(layer)
        (out / filename).write_bytes(np.asarray(hidden, dtype=np.dtype("<u2")).tobytes())
        _add_hidden_tensor_manifest(
            manifest,
            f"layer_{layer}_hidden",
            filename,
            int(prompt_dump.input_ids.size),
            hidden,
            f"FP16 weights/activations with FP32 reductions; layer {layer} text-only decoder prefill",
        )
    manifest["text_decoder_layer_count"] = int(layer_count)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def dump_text_logits_topk_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    *,
    top_k: int = DEFAULT_LOGITS_TOP_K,
) -> Path:
    """Write all text-only decoder layers plus next-token LM-head top-k logits."""

    out = dump_text_decoder_layers_fixture(request, out_dir, hf_dir, layer_count=TEXT_DECODER_LAYER_COUNT)
    layers = load_text_decoder_layers_dump(out)
    ids, scores = compute_text_logits_topk_f32(layers.layer_hidden_f16_bits[-1], hf_dir, top_k=top_k)
    np.asarray(ids, dtype=np.dtype("<i4")).tofile(out / LOGITS_TOPK_IDS_BIN)
    np.asarray(scores, dtype=np.dtype("<f4")).tofile(out / LOGITS_TOPK_SCORES_BIN)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["golden_tensors"]["logits_topk"] = {
        "ids_file": LOGITS_TOPK_IDS_BIN,
        "scores_file": LOGITS_TOPK_SCORES_BIN,
        "ids_dtype": "int32_le",
        "scores_dtype": "float32_le",
        "shape": [int(top_k)],
        "source": f"final RMSNorm {FINAL_NORM_TENSOR_NAME} and LM head {LM_HEAD_TENSOR_NAME}",
        "reference_dtype_mode": "FP16 final norm activations/weights with FP32 LM-head reductions",
        "rank_order": "score descending, token id ascending on ties",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def dump_text_generated_ids_fixture(
    request: PreparedRequest,
    out_dir: str | Path,
    hf_dir: str | Path,
    *,
    top_k: int = DEFAULT_LOGITS_TOP_K,
) -> Path:
    """Write text-only decoder layers, logits top-k, and the first greedy generated token id."""

    out = dump_text_logits_topk_fixture(request, out_dir, hf_dir, top_k=max(1, int(top_k)))
    logits = load_text_logits_topk_dump(out)
    generated = np.asarray(logits.logits_topk_ids[:1], dtype=np.dtype("<i4"))
    generated.tofile(out / GENERATED_IDS_BIN)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["golden_tensors"]["generated_ids"] = {
        "file": GENERATED_IDS_BIN,
        "dtype": "int32_le",
        "shape": [1],
        "source": "greedy argmax over next-token logits without sampling or no-repeat bans",
        "reference_dtype_mode": "first deterministic text-only generated token from fp16 parity logits",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _infer_sam_grid_from_manifest(manifest: dict[str, Any], view_index: int) -> int | None:
    views = manifest.get("views") if isinstance(manifest, dict) else None
    if not isinstance(views, list) or view_index < 0 or view_index >= len(views):
        return None
    meta = views[view_index]
    if not isinstance(meta, dict):
        return None
    width = int(meta.get("width", 0))
    height = int(meta.get("height", 0))
    kind = str(meta.get("kind", ""))
    if width == 1024 and height == 1024 and kind == "global":
        return SAM_GLOBAL_GRID
    if width == 640 and height == 640 and kind == "local":
        return SAM_LOCAL_GRID
    return None


def _sam_feature_meta(manifest: dict[str, Any], view_index: int) -> dict[str, Any]:
    golden = manifest.get("golden_tensors", {}).get("sam_features", {}) if isinstance(manifest, dict) else {}
    if not isinstance(golden, dict):
        return {}
    views = golden.get("views")
    if isinstance(views, list):
        if view_index < 0 or view_index >= len(views):
            raise ValueError(f"sam_features view index {view_index} is outside manifest view metadata")
        meta = views[view_index]
        if not isinstance(meta, dict):
            raise ValueError(f"invalid sam_features metadata for view {view_index}")
        return meta
    return golden


def load_sam_features_dump(
    fixture_dir: str | Path,
    *,
    view_index: int = 0,
    grid_size: int | None = None,
) -> NDArray[np.uint16]:
    """Load a Python/HF SAM output fixture as uint16 fp16 bits.

    The tensor contract is the upstream ``sam_model(view)`` output without the
    batch dimension: ``[1024, grid, grid]`` in NCHW order, where ``grid`` is
    ``16`` for a global ``1024`` view and ``10`` for a local ``640`` view.
    """

    if view_index < 0:
        raise ValueError(f"view_index must be non-negative, got {view_index}")
    root = Path(fixture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"expected manifest object in {root}")
    meta = _sam_feature_meta(manifest, view_index)
    filename = meta.get("file") if isinstance(meta, dict) else None
    if not isinstance(filename, str):
        default_name = sam_features_filename(view_index)
        if not (root / default_name).exists() and view_index != 0:
            default_name = SAM_FEATURES_BIN
        filename = default_name

    shape = meta.get("shape") if isinstance(meta, dict) else None
    if isinstance(shape, list) and len(shape) == 3:
        channels = int(shape[0])
        grid_h = int(shape[1])
        grid_w = int(shape[2])
    else:
        inferred = grid_size if grid_size is not None else _infer_sam_grid_from_manifest(manifest, view_index)
        if inferred is None:
            raise ValueError(f"missing sam_features shape metadata in {root}")
        channels = SAM_FEATURE_CHANNELS
        grid_h = int(inferred)
        grid_w = int(inferred)

    if channels != SAM_FEATURE_CHANNELS or grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"invalid sam_features shape [{channels}, {grid_h}, {grid_w}] in {root}")
    if grid_size is not None and (grid_h != int(grid_size) or grid_w != int(grid_size)):
        raise ValueError(f"sam_features shape grid {grid_h}x{grid_w} does not match expected {grid_size}x{grid_size}")
    bits = np.fromfile(root / filename, dtype=np.dtype("<u2"))
    expected_values = channels * grid_h * grid_w
    if bits.size != expected_values:
        raise ValueError(f"sam_features size mismatch: expected {expected_values} f16 values, got {bits.size}")
    return bits.reshape((channels, grid_h, grid_w))


def _infer_clip_tokens_from_manifest(manifest: dict[str, Any], view_index: int) -> int | None:
    views = manifest.get("views") if isinstance(manifest, dict) else None
    if not isinstance(views, list) or view_index < 0 or view_index >= len(views):
        return None
    meta = views[view_index]
    if not isinstance(meta, dict):
        return None
    width = int(meta.get("width", 0))
    height = int(meta.get("height", 0))
    kind = str(meta.get("kind", ""))
    if width == 1024 and height == 1024 and kind == "global":
        return CLIP_GLOBAL_TOKENS
    if width == 640 and height == 640 and kind == "local":
        return CLIP_LOCAL_TOKENS
    return None


def _clip_feature_meta(manifest: dict[str, Any], view_index: int) -> dict[str, Any]:
    golden = manifest.get("golden_tensors", {}).get("clip_features", {}) if isinstance(manifest, dict) else {}
    if not isinstance(golden, dict):
        return {}
    views = golden.get("views")
    if isinstance(views, list):
        if view_index < 0 or view_index >= len(views):
            raise ValueError(f"clip_features view index {view_index} is outside manifest view metadata")
        meta = views[view_index]
        if not isinstance(meta, dict):
            raise ValueError(f"invalid clip_features metadata for view {view_index}")
        return meta
    return golden


def load_clip_features_dump(
    fixture_dir: str | Path,
    *,
    view_index: int = 0,
    token_count: int | None = None,
) -> NDArray[np.uint16]:
    """Load a Python/HF CLIP output fixture as uint16 fp16 bits.

    The tensor contract is the upstream ``vision_model(view, sam_features)``
    output including the class token, without the batch dimension:
    ``[tokens, 1024]`` where ``tokens`` is ``257`` for global views and ``101``
    for local views. Shape metadata may also use upstream ``[1,tokens,1024]``.
    """

    if view_index < 0:
        raise ValueError(f"view_index must be non-negative, got {view_index}")
    root = Path(fixture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"expected manifest object in {root}")
    meta = _clip_feature_meta(manifest, view_index)
    filename = meta.get("file") if isinstance(meta, dict) else None
    if not isinstance(filename, str):
        default_name = clip_features_filename(view_index)
        if not (root / default_name).exists() and view_index != 0:
            default_name = CLIP_FEATURES_BIN
        filename = default_name

    shape = meta.get("shape") if isinstance(meta, dict) else None
    if isinstance(shape, list) and len(shape) == 2:
        tokens = int(shape[0])
        hidden_size = int(shape[1])
    elif isinstance(shape, list) and len(shape) == 3:
        batch = int(shape[0])
        if batch != 1:
            raise ValueError(f"clip_features batch dimension must be 1, got {batch} in {root}")
        tokens = int(shape[1])
        hidden_size = int(shape[2])
    else:
        inferred = token_count if token_count is not None else _infer_clip_tokens_from_manifest(manifest, view_index)
        if inferred is None:
            raise ValueError(f"missing clip_features shape metadata in {root}")
        tokens = int(inferred)
        hidden_size = CLIP_HIDDEN_SIZE

    if tokens <= 0 or hidden_size != CLIP_HIDDEN_SIZE:
        raise ValueError(f"invalid clip_features shape [{tokens}, {hidden_size}] in {root}")
    if token_count is not None and tokens != int(token_count):
        raise ValueError(f"clip_features token count {tokens} does not match expected {token_count}")
    bits = np.fromfile(root / filename, dtype=np.dtype("<u2"))
    expected_values = tokens * hidden_size
    if bits.size != expected_values:
        raise ValueError(f"clip_features size mismatch: expected {expected_values} f16 values, got {bits.size}")
    return bits.reshape((tokens, hidden_size))


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
    if image_mask.shape != input_ids.shape:
        raise ValueError(f"image_mask shape {image_mask.shape} does not match input ids {input_ids.shape}")
    expected_values = n_tokens * hidden_size
    if embeddings.size != expected_values:
        raise ValueError(f"prompt embedding size mismatch: expected {expected_values} f16 values, got {embeddings.size}")
    return PromptEmbeddingDump(
        manifest=manifest,
        input_ids=input_ids,
        image_mask=image_mask,
        prompt_embeddings_f16_bits=embeddings.reshape((n_tokens, hidden_size)),
    )


def load_image_prompt_embedding_dump(fixture_dir: str | Path) -> ImagePromptEmbeddingDump:
    prompt = load_prompt_embedding_dump(fixture_dir)
    root = Path(fixture_dir)
    image_span_start, image_span_length = _contiguous_image_span(prompt.image_mask)
    hidden_size = int(prompt.prompt_embeddings_f16_bits.shape[1])
    golden = prompt.manifest.get("golden_tensors", {}).get("visual_features", {})
    shape = golden.get("shape", [image_span_length, hidden_size]) if isinstance(golden, dict) else []
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"invalid visual_features shape metadata in {root}")
    rows = int(shape[0])
    visual_hidden = int(shape[1])
    if rows != image_span_length or visual_hidden != hidden_size:
        raise ValueError(f"visual_features shape {shape} does not match image span/hidden size")
    visual = np.fromfile(root / VISUAL_FEATURES_BIN, dtype=np.dtype("<u2"))
    if visual.size != rows * visual_hidden:
        raise ValueError(f"visual_features size mismatch: expected {rows * visual_hidden}, got {visual.size}")
    return ImagePromptEmbeddingDump(
        manifest=prompt.manifest,
        input_ids=prompt.input_ids,
        image_mask=prompt.image_mask,
        prompt_embeddings_f16_bits=prompt.prompt_embeddings_f16_bits,
        visual_features_f16_bits=visual.reshape((rows, visual_hidden)),
        image_span_start=image_span_start,
        image_span_length=image_span_length,
    )


def _load_hidden_tensor_bits(
    root: Path,
    manifest: dict[str, Any],
    key: str,
    filename: str,
    n_prompt_tokens: int,
) -> NDArray[np.uint16]:
    golden = manifest.get("golden_tensors", {}).get(key, {})
    shape = golden.get("shape", [int(n_prompt_tokens), HIDDEN_SIZE]) if isinstance(golden, dict) else []
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"invalid {key} shape metadata in {root}")
    n_tokens = int(shape[0])
    hidden_size = int(shape[1])
    if n_tokens != int(n_prompt_tokens) or hidden_size != HIDDEN_SIZE:
        raise ValueError(f"{key} shape {shape} does not match prompt shape")
    bits = np.fromfile(root / filename, dtype=np.dtype("<u2"))
    expected_values = n_tokens * hidden_size
    if bits.size != expected_values:
        raise ValueError(f"{key} size mismatch: expected {expected_values} f16 values, got {bits.size}")
    return bits.reshape((n_tokens, hidden_size))


def load_image_decoder_layers_dump(fixture_dir: str | Path) -> ImageDecoderLayersDump:
    prompt = load_image_prompt_embedding_dump(fixture_dir)
    root = Path(fixture_dir)
    layer_count = int(prompt.manifest.get("image_decoder_layer_count", TEXT_DECODER_LAYER_COUNT))
    if layer_count < 1 or layer_count > TEXT_DECODER_LAYER_COUNT:
        raise ValueError(f"invalid image decoder layer count {layer_count} in {root}")
    n_tokens = int(prompt.input_ids.size)
    layers = tuple(
        _load_hidden_tensor_bits(
            root,
            prompt.manifest,
            f"layer_{layer}_hidden",
            text_layer_hidden_filename(layer),
            n_tokens,
        )
        for layer in range(layer_count)
    )

    router_ids: dict[int, NDArray[np.uint32]] = {}
    router_weights: dict[int, NDArray[np.float32]] = {}
    router_manifest = prompt.manifest.get("router_topk", {}) if isinstance(prompt.manifest, dict) else {}
    if isinstance(router_manifest, dict):
        for layer in range(1, layer_count):
            meta = router_manifest.get(f"layer_{layer}")
            if not isinstance(meta, dict):
                continue
            shape = meta.get("shape", [n_tokens, MOE_TOP_K])
            if not isinstance(shape, list) or len(shape) != 2:
                raise ValueError(f"invalid router_topk layer {layer} shape metadata in {root}")
            rows = int(shape[0])
            top_k = int(shape[1])
            if rows != n_tokens or top_k != MOE_TOP_K:
                raise ValueError(f"router_topk layer {layer} shape {shape} does not match prompt/top-k")
            ids_file = meta.get("ids_file", router_top_ids_filename(layer))
            weights_file = meta.get("weights_file", router_top_weights_filename(layer))
            if not isinstance(ids_file, str) or not isinstance(weights_file, str):
                raise ValueError(f"invalid router_topk layer {layer} filenames in {root}")
            ids = np.fromfile(root / ids_file, dtype=ROUTER_TOP_IDS_DTYPE)
            weights = np.fromfile(root / weights_file, dtype=ROUTER_TOP_WEIGHTS_DTYPE)
            expected_values = rows * top_k
            if ids.size != expected_values or weights.size != expected_values:
                raise ValueError(
                    f"router_topk layer {layer} size mismatch: ids={ids.size} weights={weights.size} expected={expected_values}"
                )
            ids = ids.reshape((rows, top_k))
            weights = weights.reshape((rows, top_k))
            if np.any(ids >= MOE_EXPERTS) or not np.all(np.isfinite(weights)):
                raise ValueError(f"invalid router_topk layer {layer} values in {root}")
            router_ids[layer] = ids.astype(np.uint32, copy=False)
            router_weights[layer] = weights.astype(np.float32, copy=False)

    return ImageDecoderLayersDump(
        manifest=prompt.manifest,
        input_ids=prompt.input_ids,
        image_mask=prompt.image_mask,
        prompt_embeddings_f16_bits=prompt.prompt_embeddings_f16_bits,
        visual_features_f16_bits=prompt.visual_features_f16_bits,
        image_span_start=prompt.image_span_start,
        image_span_length=prompt.image_span_length,
        layer_hidden_f16_bits=layers,
        router_top_ids=router_ids,
        router_top_weights_f32=router_weights,
    )


def load_text_layer0_dump(fixture_dir: str | Path) -> TextLayer0Dump:
    prompt = load_prompt_embedding_dump(fixture_dir)
    root = Path(fixture_dir)
    layer0 = _load_hidden_tensor_bits(
        root,
        prompt.manifest,
        "layer_0_hidden",
        TEXT_LAYER0_HIDDEN_BIN,
        int(prompt.input_ids.size),
    )
    return TextLayer0Dump(
        manifest=prompt.manifest,
        input_ids=prompt.input_ids,
        image_mask=prompt.image_mask,
        prompt_embeddings_f16_bits=prompt.prompt_embeddings_f16_bits,
        layer0_hidden_f16_bits=layer0,
    )


def load_text_layer1_dump(fixture_dir: str | Path) -> TextLayer1Dump:
    layer0 = load_text_layer0_dump(fixture_dir)
    root = Path(fixture_dir)
    layer1 = _load_hidden_tensor_bits(
        root,
        layer0.manifest,
        "layer_1_hidden",
        TEXT_LAYER1_HIDDEN_BIN,
        int(layer0.input_ids.size),
    )
    return TextLayer1Dump(
        manifest=layer0.manifest,
        input_ids=layer0.input_ids,
        image_mask=layer0.image_mask,
        prompt_embeddings_f16_bits=layer0.prompt_embeddings_f16_bits,
        layer0_hidden_f16_bits=layer0.layer0_hidden_f16_bits,
        layer1_hidden_f16_bits=layer1,
    )


def load_text_decoder_layers_dump(
    fixture_dir: str | Path,
    *,
    layer_count: int | None = None,
) -> TextDecoderLayersDump:
    prompt = load_prompt_embedding_dump(fixture_dir)
    root = Path(fixture_dir)
    if layer_count is None:
        manifest_count = prompt.manifest.get("text_decoder_layer_count") if isinstance(prompt.manifest, dict) else None
        layer_count = int(manifest_count) if manifest_count is not None else TEXT_DECODER_LAYER_COUNT
    if layer_count < 1 or layer_count > TEXT_DECODER_LAYER_COUNT:
        raise ValueError(f"layer_count must be in [1,{TEXT_DECODER_LAYER_COUNT}], got {layer_count}")
    layers = tuple(
        _load_hidden_tensor_bits(
            root,
            prompt.manifest,
            f"layer_{layer}_hidden",
            text_layer_hidden_filename(layer),
            int(prompt.input_ids.size),
        )
        for layer in range(layer_count)
    )
    return TextDecoderLayersDump(
        manifest=prompt.manifest,
        input_ids=prompt.input_ids,
        image_mask=prompt.image_mask,
        prompt_embeddings_f16_bits=prompt.prompt_embeddings_f16_bits,
        layer_hidden_f16_bits=layers,
    )


def _load_logits_topk_values(root: Path, manifest: dict[str, Any]) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    golden = manifest.get("golden_tensors", {}).get("logits_topk", {})
    if not isinstance(golden, dict):
        raise ValueError(f"missing logits_topk metadata in {root}")
    shape = golden.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"invalid logits_topk shape metadata in {root}")
    top_k = int(shape[0])
    ids_file = golden.get("ids_file", LOGITS_TOPK_IDS_BIN)
    scores_file = golden.get("scores_file", LOGITS_TOPK_SCORES_BIN)
    if not isinstance(ids_file, str) or not isinstance(scores_file, str):
        raise ValueError(f"invalid logits_topk filenames in {root}")
    ids = np.fromfile(root / ids_file, dtype=np.dtype("<i4"))
    scores = np.fromfile(root / scores_file, dtype=np.dtype("<f4"))
    if ids.shape != (top_k,) or scores.shape != (top_k,):
        raise ValueError(f"logits_topk shape mismatch in {root}: ids={ids.shape} scores={scores.shape}")
    if top_k <= 0 or np.any(ids < 0) or np.any(ids >= MODEL_VOCAB_SIZE) or not np.all(np.isfinite(scores)):
        raise ValueError(f"invalid logits_topk values in {root}")
    return ids.astype(np.int32, copy=False), scores.astype(np.float32, copy=False)


def load_text_logits_topk_dump(fixture_dir: str | Path) -> TextLogitsTopKDump:
    layers = load_text_decoder_layers_dump(fixture_dir)
    root = Path(fixture_dir)
    ids, scores = _load_logits_topk_values(root, layers.manifest)
    return TextLogitsTopKDump(
        manifest=layers.manifest,
        input_ids=layers.input_ids,
        image_mask=layers.image_mask,
        prompt_embeddings_f16_bits=layers.prompt_embeddings_f16_bits,
        layer_hidden_f16_bits=layers.layer_hidden_f16_bits,
        logits_topk_ids=ids,
        logits_topk_scores_f32=scores,
    )


def load_image_logits_topk_dump(fixture_dir: str | Path) -> ImageLogitsTopKDump:
    layers = load_image_decoder_layers_dump(fixture_dir)
    root = Path(fixture_dir)
    ids, scores = _load_logits_topk_values(root, layers.manifest)
    return ImageLogitsTopKDump(
        manifest=layers.manifest,
        input_ids=layers.input_ids,
        image_mask=layers.image_mask,
        prompt_embeddings_f16_bits=layers.prompt_embeddings_f16_bits,
        visual_features_f16_bits=layers.visual_features_f16_bits,
        image_span_start=layers.image_span_start,
        image_span_length=layers.image_span_length,
        layer_hidden_f16_bits=layers.layer_hidden_f16_bits,
        router_top_ids=layers.router_top_ids,
        router_top_weights_f32=layers.router_top_weights_f32,
        logits_topk_ids=ids,
        logits_topk_scores_f32=scores,
    )


def _load_generated_ids_values(root: Path, manifest: dict[str, Any]) -> NDArray[np.int32]:
    golden = manifest.get("golden_tensors", {}).get("generated_ids", {})
    if not isinstance(golden, dict):
        raise ValueError(f"missing generated_ids metadata in {root}")
    shape = golden.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"invalid generated_ids shape metadata in {root}")
    n_ids = int(shape[0])
    ids_file = golden.get("file", GENERATED_IDS_BIN)
    if not isinstance(ids_file, str):
        raise ValueError(f"invalid generated_ids filename in {root}")
    ids = np.fromfile(root / ids_file, dtype=np.dtype("<i4"))
    if ids.shape != (n_ids,):
        raise ValueError(f"generated_ids shape mismatch in {root}: ids={ids.shape}")
    if n_ids <= 0 or np.any(ids < 0) or np.any(ids >= MODEL_VOCAB_SIZE):
        raise ValueError(f"invalid generated_ids in {root}")
    return ids.astype(np.int32, copy=False)


def _load_generated_text_value(root: Path, manifest: dict[str, Any]) -> str:
    golden = manifest.get("golden_tensors", {}).get("generated_text", {})
    if not isinstance(golden, dict):
        raise ValueError(f"missing generated_text metadata in {root}")
    text_file = golden.get("file", GENERATED_TEXT_TXT)
    if not isinstance(text_file, str):
        raise ValueError(f"invalid generated_text filename in {root}")
    return (root / text_file).read_text(encoding="utf-8")


def load_text_generated_ids_dump(fixture_dir: str | Path) -> TextGeneratedIdsDump:
    logits = load_text_logits_topk_dump(fixture_dir)
    root = Path(fixture_dir)
    ids = _load_generated_ids_values(root, logits.manifest)
    return TextGeneratedIdsDump(
        manifest=logits.manifest,
        input_ids=logits.input_ids,
        image_mask=logits.image_mask,
        prompt_embeddings_f16_bits=logits.prompt_embeddings_f16_bits,
        layer_hidden_f16_bits=logits.layer_hidden_f16_bits,
        logits_topk_ids=logits.logits_topk_ids,
        logits_topk_scores_f32=logits.logits_topk_scores_f32,
        generated_ids=ids,
    )


def load_image_generated_ids_dump(fixture_dir: str | Path) -> ImageGeneratedIdsDump:
    logits = load_image_logits_topk_dump(fixture_dir)
    root = Path(fixture_dir)
    ids = _load_generated_ids_values(root, logits.manifest)
    text = _load_generated_text_value(root, logits.manifest)
    return ImageGeneratedIdsDump(
        manifest=logits.manifest,
        input_ids=logits.input_ids,
        image_mask=logits.image_mask,
        prompt_embeddings_f16_bits=logits.prompt_embeddings_f16_bits,
        visual_features_f16_bits=logits.visual_features_f16_bits,
        image_span_start=logits.image_span_start,
        image_span_length=logits.image_span_length,
        layer_hidden_f16_bits=logits.layer_hidden_f16_bits,
        router_top_ids=logits.router_top_ids,
        router_top_weights_f32=logits.router_top_weights_f32,
        logits_topk_ids=logits.logits_topk_ids,
        logits_topk_scores_f32=logits.logits_topk_scores_f32,
        generated_ids=ids,
        generated_text=text,
    )
