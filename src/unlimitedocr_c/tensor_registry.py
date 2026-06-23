"""Stable tensor registry shared by converter planning and the C runtime.

The runtime must not depend on Hugging Face tensor names in hot paths.  The
converter maps those names into deterministic integer ids and structured fields;
`.uocr` stores the ids and the runtime indexes by id.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import re
from typing import Iterable, Mapping


class TensorFamily(IntEnum):
    UNKNOWN = 0
    TOK_EMBED = 1
    LM_HEAD = 2
    FINAL_NORM = 3
    LAYER_ATTN = 4
    LAYER_NORM = 5
    LAYER_DENSE_MLP = 6
    MOE_ROUTER = 7
    MOE_EXPERT = 8
    MOE_SHARED = 9
    VISION_SAM = 10
    VISION_CLIP = 11
    PROJECTOR = 12
    IMAGE_NEWLINE = 13
    VIEW_SEPARATOR = 14


class TensorProjection(IntEnum):
    NONE = 0
    WEIGHT = 1
    BIAS = 2
    Q = 10
    K = 11
    V = 12
    O = 13
    GATE = 20
    UP = 21
    DOWN = 22


INVALID_TENSOR_ID = 0
TENSOR_ID_TOK_EMBED = 1
TENSOR_ID_LM_HEAD = 2
TENSOR_ID_FINAL_NORM = 3
TENSOR_ID_IMAGE_NEWLINE = 4
TENSOR_ID_VIEW_SEPARATOR = 5
TENSOR_ID_PROJECTOR_WEIGHT = 6
TENSOR_ID_PROJECTOR_BIAS = 7

LAYER_BASE = 10_000
LAYER_STRIDE = 1_000
LAYER_INPUT_NORM = 1
LAYER_POST_ATTN_NORM = 2
LAYER_ATTN_Q = 10
LAYER_ATTN_K = 11
LAYER_ATTN_V = 12
LAYER_ATTN_O = 13
LAYER_DENSE_GATE = 30
LAYER_DENSE_UP = 31
LAYER_DENSE_DOWN = 32
LAYER_MOE_ROUTER = 40
LAYER_MOE_SHARED_GATE = 50
LAYER_MOE_SHARED_UP = 51
LAYER_MOE_SHARED_DOWN = 52
LAYER_MOE_EXPERT_BASE = 100
LAYER_MOE_EXPERT_STRIDE = 3
LAYER_MOE_EXPERT_GATE = 0
LAYER_MOE_EXPERT_UP = 1
LAYER_MOE_EXPERT_DOWN = 2

VISION_SAM_BASE = 100_000
VISION_CLIP_BASE = 200_000

_LAYER_NORM_RE = re.compile(r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$")
_ATTN_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
_DENSE_MLP_RE = re.compile(r"^model\.layers\.0\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
_MOE_ROUTER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
_MOE_SHARED_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$")
_MOE_EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


@dataclass(frozen=True)
class TensorRegistryEntry:
    name: str
    tensor_id: int
    family: TensorFamily
    layer: int
    expert: int
    projection: TensorProjection
    expected_shape: tuple[int, ...] | None = None

    @property
    def family_name(self) -> str:
        return self.family.name

    @property
    def projection_name(self) -> str:
        return self.projection.name

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "tensor_id": self.tensor_id,
            "family": self.family.name,
            "family_id": int(self.family),
            "layer": self.layer,
            "expert": self.expert,
            "projection": self.projection.name,
            "projection_id": int(self.projection),
        }
        if self.expected_shape is not None:
            payload["expected_shape"] = list(self.expected_shape)
        return payload


def tensor_id_layer(layer: int, local_id: int) -> int:
    return LAYER_BASE + layer * LAYER_STRIDE + local_id


def tensor_id_layer_attn(layer: int, projection: TensorProjection) -> int:
    return tensor_id_layer(layer, {
        TensorProjection.Q: LAYER_ATTN_Q,
        TensorProjection.K: LAYER_ATTN_K,
        TensorProjection.V: LAYER_ATTN_V,
        TensorProjection.O: LAYER_ATTN_O,
    }[projection])


def tensor_id_dense_mlp(projection: TensorProjection) -> int:
    return tensor_id_layer(0, {
        TensorProjection.GATE: LAYER_DENSE_GATE,
        TensorProjection.UP: LAYER_DENSE_UP,
        TensorProjection.DOWN: LAYER_DENSE_DOWN,
    }[projection])


def tensor_id_moe_shared(layer: int, projection: TensorProjection) -> int:
    return tensor_id_layer(layer, {
        TensorProjection.GATE: LAYER_MOE_SHARED_GATE,
        TensorProjection.UP: LAYER_MOE_SHARED_UP,
        TensorProjection.DOWN: LAYER_MOE_SHARED_DOWN,
    }[projection])


def tensor_id_moe_expert(layer: int, expert: int, projection: TensorProjection) -> int:
    return tensor_id_layer(layer, LAYER_MOE_EXPERT_BASE + expert * LAYER_MOE_EXPERT_STRIDE + {
        TensorProjection.GATE: LAYER_MOE_EXPERT_GATE,
        TensorProjection.UP: LAYER_MOE_EXPERT_UP,
        TensorProjection.DOWN: LAYER_MOE_EXPERT_DOWN,
    }[projection])


def _proj_from_prefix(prefix: str) -> TensorProjection:
    return {
        "q_proj": TensorProjection.Q,
        "k_proj": TensorProjection.K,
        "v_proj": TensorProjection.V,
        "o_proj": TensorProjection.O,
        "gate_proj": TensorProjection.GATE,
        "up_proj": TensorProjection.UP,
        "down_proj": TensorProjection.DOWN,
    }[prefix]


def _entry_without_vision(name: str) -> TensorRegistryEntry | None:
    if name == "model.embed_tokens.weight":
        return TensorRegistryEntry(name, TENSOR_ID_TOK_EMBED, TensorFamily.TOK_EMBED, -1, -1, TensorProjection.WEIGHT, (129_280, 1280))
    if name == "lm_head.weight":
        return TensorRegistryEntry(name, TENSOR_ID_LM_HEAD, TensorFamily.LM_HEAD, -1, -1, TensorProjection.WEIGHT, (129_280, 1280))
    if name == "model.norm.weight":
        return TensorRegistryEntry(name, TENSOR_ID_FINAL_NORM, TensorFamily.FINAL_NORM, -1, -1, TensorProjection.WEIGHT, (1280,))
    if name == "model.image_newline":
        return TensorRegistryEntry(name, TENSOR_ID_IMAGE_NEWLINE, TensorFamily.IMAGE_NEWLINE, -1, -1, TensorProjection.NONE, (1280,))
    if name == "model.view_seperator":
        return TensorRegistryEntry(name, TENSOR_ID_VIEW_SEPARATOR, TensorFamily.VIEW_SEPARATOR, -1, -1, TensorProjection.NONE, (1280,))
    if name == "model.projector.layers.weight":
        return TensorRegistryEntry(name, TENSOR_ID_PROJECTOR_WEIGHT, TensorFamily.PROJECTOR, -1, -1, TensorProjection.WEIGHT, (1280, 2048))
    if name == "model.projector.layers.bias":
        return TensorRegistryEntry(name, TENSOR_ID_PROJECTOR_BIAS, TensorFamily.PROJECTOR, -1, -1, TensorProjection.BIAS, (1280,))

    match = _LAYER_NORM_RE.match(name)
    if match:
        layer = int(match.group(1))
        local = LAYER_INPUT_NORM if match.group(2) == "input_layernorm" else LAYER_POST_ATTN_NORM
        projection = TensorProjection.WEIGHT
        return TensorRegistryEntry(name, tensor_id_layer(layer, local), TensorFamily.LAYER_NORM, layer, -1, projection, (1280,))

    match = _ATTN_RE.match(name)
    if match:
        layer = int(match.group(1))
        projection = _proj_from_prefix(match.group(2))
        return TensorRegistryEntry(name, tensor_id_layer_attn(layer, projection), TensorFamily.LAYER_ATTN, layer, -1, projection, (1280, 1280))

    match = _DENSE_MLP_RE.match(name)
    if match:
        projection = _proj_from_prefix(match.group(1))
        shape = (6848, 1280) if projection in {TensorProjection.GATE, TensorProjection.UP} else (1280, 6848)
        return TensorRegistryEntry(name, tensor_id_dense_mlp(projection), TensorFamily.LAYER_DENSE_MLP, 0, -1, projection, shape)

    match = _MOE_ROUTER_RE.match(name)
    if match:
        layer = int(match.group(1))
        return TensorRegistryEntry(name, tensor_id_layer(layer, LAYER_MOE_ROUTER), TensorFamily.MOE_ROUTER, layer, -1, TensorProjection.WEIGHT, (64, 1280))

    match = _MOE_SHARED_RE.match(name)
    if match:
        layer = int(match.group(1))
        projection = _proj_from_prefix(match.group(2))
        shape = (1792, 1280) if projection in {TensorProjection.GATE, TensorProjection.UP} else (1280, 1792)
        return TensorRegistryEntry(name, tensor_id_moe_shared(layer, projection), TensorFamily.MOE_SHARED, layer, -1, projection, shape)

    match = _MOE_EXPERT_RE.match(name)
    if match:
        layer = int(match.group(1))
        expert = int(match.group(2))
        projection = _proj_from_prefix(match.group(3))
        shape = (896, 1280) if projection in {TensorProjection.GATE, TensorProjection.UP} else (1280, 896)
        return TensorRegistryEntry(name, tensor_id_moe_expert(layer, expert, projection), TensorFamily.MOE_EXPERT, layer, expert, projection, shape)

    return None


def build_tensor_registry(names: Iterable[str]) -> dict[str, TensorRegistryEntry]:
    names_tuple = tuple(names)
    sam_names = sorted(name for name in names_tuple if name.startswith("model.sam_model."))
    clip_names = sorted(name for name in names_tuple if name.startswith("model.vision_model."))
    sam_ids = {name: VISION_SAM_BASE + i for i, name in enumerate(sam_names)}
    clip_ids = {name: VISION_CLIP_BASE + i for i, name in enumerate(clip_names)}

    registry: dict[str, TensorRegistryEntry] = {}
    used_ids: dict[int, str] = {}
    for name in names_tuple:
        entry = _entry_without_vision(name)
        if entry is None and name in sam_ids:
            entry = TensorRegistryEntry(name, sam_ids[name], TensorFamily.VISION_SAM, -1, -1, TensorProjection.NONE)
        if entry is None and name in clip_ids:
            entry = TensorRegistryEntry(name, clip_ids[name], TensorFamily.VISION_CLIP, -1, -1, TensorProjection.NONE)
        if entry is None:
            raise ValueError(f"unrecognized tensor name for registry: {name}")
        if not (1 <= entry.tensor_id < 1_000_000):
            raise ValueError(f"tensor id out of reserved range for {name}: {entry.tensor_id}")
        previous = used_ids.get(entry.tensor_id)
        if previous is not None:
            raise ValueError(f"tensor id collision {entry.tensor_id}: {previous} and {name}")
        used_ids[entry.tensor_id] = name
        registry[name] = entry
    return registry


def validate_registry_shapes(registry: Mapping[str, TensorRegistryEntry], shapes: Mapping[str, tuple[int, ...]]) -> None:
    for name, entry in registry.items():
        if entry.expected_shape is None:
            continue
        actual = shapes.get(name)
        if actual != entry.expected_shape:
            raise ValueError(f"tensor {name} shape mismatch: got {actual}, expected {entry.expected_shape}")
