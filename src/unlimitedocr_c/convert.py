"""Conversion planning and fp16 `.uocr` writing for Unlimited-OCR.

The planner can run from the cached safetensors header/index only, which keeps
fast tests independent from the 6.67 GB checkpoint payload.  When the real
safetensors file is present, the writer streams tensor ranges into the planned
`.uocr` layout and converts BF16 payload bytes to fp16 without allocating a
full-model temporary buffer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Iterable, Mapping

import numpy as np

from .frontend import project_root
from .tensor_registry import (
    TensorProjection,
    TensorRegistryEntry,
    build_tensor_registry,
    validate_registry_shapes,
)

EXPECTED_TENSOR_COUNT = 2710
EXPECTED_TOTAL_BYTES = 6_672_212_480
EXPECTED_SOURCE_DTYPE = "BF16"

UOCR_FILE_ALIGNMENT = 4096
UOCR_TENSOR_DATA_ALIGNMENT = 4096
UOCR_TENSOR_PAYLOAD_ALIGNMENT = 16

UOCR_FILE_HEADER_SIZE = 112
UOCR_SECTION_ENTRY_SIZE = 32
UOCR_CONFIG_RECORD_SIZE = 108
UOCR_TOKENIZER_METADATA_RECORD_SIZE = 92
UOCR_PROVENANCE_RECORD_SIZE = 200
UOCR_TENSOR_DIRECTORY_HEADER_SIZE = 16
UOCR_TENSOR_ENTRY_SIZE = 140

UOCR_SECTION_CONFIG = 1
UOCR_SECTION_TENSOR_DIRECTORY = 2
UOCR_SECTION_TOKENIZER_METADATA = 3
UOCR_SECTION_PROVENANCE = 4
UOCR_SECTION_TENSOR_DATA = 5

UOCR_FORMAT_VERSION = 1
UOCR_ENDIAN_MARKER = 0x01020304
UOCR_QPROFILE_FP16 = 1
UOCR_QPROFILE_DYN_Q8 = 2
UOCR_QPROFILE_DYN_Q4 = 3
UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED = 1 << 0
UOCR_TOKENIZER_METADATA_MAGIC = 0x4B4F5455
UOCR_PROVENANCE_MAGIC = 0x564F5250
UOCR_TENSOR_DIR_MAGIC = 0x52494454
UOCR_TENSOR_USAGE_RUNTIME = 1
UOCR_TENSOR_USAGE_PRESERVED_UNUSED = 2
UOCR_TENSOR_USAGE_OMITTED_WITH_REASON = 3

CONVERTER_VERSION = (0, 1, 0)

QPROFILE_IDS: Mapping[str, int] = {
    "fp16": UOCR_QPROFILE_FP16,
    "dyn-q8": UOCR_QPROFILE_DYN_Q8,
    "dyn-q4": UOCR_QPROFILE_DYN_Q4,
}

SECTION_NAMES: Mapping[int, str] = {
    UOCR_SECTION_CONFIG: "config",
    UOCR_SECTION_TENSOR_DIRECTORY: "tensor-directory",
    UOCR_SECTION_TOKENIZER_METADATA: "tokenizer-metadata",
    UOCR_SECTION_PROVENANCE: "provenance",
    UOCR_SECTION_TENSOR_DATA: "tensor-data",
}

EXPECTED_CONFIG: Mapping[str, int | float | str | bool] = {
    "vocab_size": 129_280,
    "hidden_size": 1280,
    "num_hidden_layers": 12,
    "num_attention_heads": 10,
    "num_key_value_heads": 10,
    "max_position_embeddings": 32_768,
    "first_k_dense_replace": 1,
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 896,
    "n_shared_experts": 2,
    "intermediate_size": 6848,
    "topk_method": "greedy",
}

# Several parity-critical values are inherited from DeepSeekV2Config defaults in
# data/context/configuration_deepseek_v2.py and are absent from config.json.
EXPECTED_CONFIG_DEFAULTS: Mapping[str, int | float | str | bool] = {
    "rope_theta": 10_000,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "scoring_func": "softmax",
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": False,
}

_FILE_HEADER_STRUCT = struct.Struct("<4s7I2Q32s32s")
_SECTION_ENTRY_STRUCT = struct.Struct("<IIQQQ")
_CONFIG_RECORD_STRUCT = struct.Struct("<27I")
_TOKENIZER_METADATA_STRUCT = struct.Struct("<4I32s7I2Q")
_PROVENANCE_RECORD_STRUCT = struct.Struct("<14I32s32s32s32s2Q")
_TENSOR_DIRECTORY_HEADER_STRUCT = struct.Struct("<4I")
_TENSOR_ENTRY_STRUCT = struct.Struct("<IIiiIIIII4I4IQQIIQQQQIIII")

assert _FILE_HEADER_STRUCT.size == UOCR_FILE_HEADER_SIZE
assert _SECTION_ENTRY_STRUCT.size == UOCR_SECTION_ENTRY_SIZE
assert _CONFIG_RECORD_STRUCT.size == UOCR_CONFIG_RECORD_SIZE
assert _TOKENIZER_METADATA_STRUCT.size == UOCR_TOKENIZER_METADATA_RECORD_SIZE
assert _PROVENANCE_RECORD_STRUCT.size == UOCR_PROVENANCE_RECORD_SIZE
assert _TENSOR_DIRECTORY_HEADER_STRUCT.size == UOCR_TENSOR_DIRECTORY_HEADER_SIZE
assert _TENSOR_ENTRY_STRUCT.size == UOCR_TENSOR_ENTRY_SIZE

KEY_TENSOR_SHAPES: Mapping[str, tuple[int, ...]] = {
    "lm_head.weight": (129_280, 1280),
    "model.embed_tokens.weight": (129_280, 1280),
    "model.layers.0.self_attn.q_proj.weight": (1280, 1280),
    "model.layers.0.self_attn.k_proj.weight": (1280, 1280),
    "model.layers.0.self_attn.v_proj.weight": (1280, 1280),
    "model.layers.0.self_attn.o_proj.weight": (1280, 1280),
    "model.layers.0.mlp.down_proj.weight": (1280, 6848),
    "model.layers.1.mlp.gate.weight": (64, 1280),
    "model.layers.1.mlp.experts.0.down_proj.weight": (1280, 896),
    "model.layers.1.mlp.experts.0.gate_proj.weight": (896, 1280),
    "model.layers.1.mlp.experts.0.up_proj.weight": (896, 1280),
    "model.layers.1.mlp.shared_experts.down_proj.weight": (1280, 1792),
    "model.projector.layers.weight": (1280, 2048),
    "model.projector.layers.bias": (1280,),
    "model.image_newline": (1280,),
    "model.view_seperator": (1280,),
    "model.sam_model.patch_embed.proj.weight": (768, 3, 16, 16),
    "model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight": (3072, 1024),
}


@dataclass(frozen=True)
class TensorPlan:
    name: str
    source_dtype: str
    shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    source_offsets: tuple[int, int]
    source_bytes: int
    output_dtype: str
    qtype: str
    qtype_id: int
    output_bytes: int
    payload_offset: int
    payload_alignment: int
    block_size: int
    row_size: int
    tensor_id: int
    family: str
    family_id: int
    layer: int
    expert: int
    projection: str
    projection_id: int
    usage: str
    usage_id: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_dtype": self.source_dtype,
            "shape": list(self.shape),
            "physical_shape": list(self.physical_shape),
            "source_offsets": list(self.source_offsets),
            "source_bytes": self.source_bytes,
            "output_dtype": self.output_dtype,
            "qtype": self.qtype,
            "qtype_id": self.qtype_id,
            "output_bytes": self.output_bytes,
            "payload_offset": self.payload_offset,
            "payload_alignment": self.payload_alignment,
            "block_size": self.block_size,
            "row_size": self.row_size,
            "tensor_id": self.tensor_id,
            "family": self.family,
            "family_id": self.family_id,
            "layer": self.layer,
            "expert": self.expert,
            "projection": self.projection,
            "projection_id": self.projection_id,
            "usage": self.usage,
            "usage_id": self.usage_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SectionPlan:
    section_type: int
    name: str
    offset: int
    size: int
    alignment: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "section_type": self.section_type,
            "name": self.name,
            "offset": self.offset,
            "size": self.size,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class DryRunPlan:
    hf_dir: Path
    qprofile: str
    tensors: tuple[TensorPlan, ...]
    total_source_bytes: int
    total_output_bytes: int
    planned_file_size: int
    metadata_bytes: int
    sections: tuple[SectionPlan, ...]
    qtype_histogram: Mapping[str, int]
    usage_histogram: Mapping[str, int]
    family_histogram: Mapping[str, int]

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    def tensor_by_name(self, name: str) -> TensorPlan:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "hf_dir": str(self.hf_dir),
            "qprofile": self.qprofile,
            "tensor_count": self.tensor_count,
            "total_source_bytes": self.total_source_bytes,
            "total_output_bytes": self.total_output_bytes,
            "planned_file_size": self.planned_file_size,
            "metadata_bytes": self.metadata_bytes,
            "sections": [section.as_dict() for section in self.sections],
            "qtype_histogram": dict(self.qtype_histogram),
            "usage_histogram": dict(self.usage_histogram),
            "family_histogram": dict(self.family_histogram),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_dict(),
            "tensors": [tensor.as_dict() for tensor in self.tensors],
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _read_json(path)


def _sha256_file(path: Path | None) -> bytes:
    hasher = hashlib.sha256()
    if path is not None and path.exists():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.digest()


def _product(shape: Iterable[int]) -> int:
    return math.prod(int(dim) for dim in shape)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


def _usage_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.usage for t in tensors))


def _qtype_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.qtype for t in tensors))


def _family_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.family for t in tensors))


def _layout_dry_run_file(tensors: tuple[TensorPlan, ...]) -> tuple[tuple[SectionPlan, ...], tuple[TensorPlan, ...], int, int]:
    """Assign deterministic `.uocr` section and payload offsets for dry-run planning.

    The converter will later stream bytes into this layout.  Planning it now lets
    tests validate page-aligned tensor data without requiring the full 6.67 GB
    safetensors payload.
    """

    section_count = 5
    section_dir_offset = UOCR_FILE_HEADER_SIZE
    config_offset = _align_up(section_dir_offset + section_count * UOCR_SECTION_ENTRY_SIZE, 8)
    tokenizer_offset = _align_up(config_offset + UOCR_CONFIG_RECORD_SIZE, 8)
    provenance_offset = _align_up(tokenizer_offset + UOCR_TOKENIZER_METADATA_RECORD_SIZE, 8)
    tensor_dir_offset = _align_up(provenance_offset + UOCR_PROVENANCE_RECORD_SIZE, 8)
    tensor_dir_size = UOCR_TENSOR_DIRECTORY_HEADER_SIZE + len(tensors) * UOCR_TENSOR_ENTRY_SIZE
    tensor_data_offset = _align_up(tensor_dir_offset + tensor_dir_size, UOCR_TENSOR_DATA_ALIGNMENT)

    cursor = tensor_data_offset
    laid_out: list[TensorPlan] = []
    for tensor in tensors:
        if tensor.usage_id == 3 or tensor.output_bytes == 0:
            laid_out.append(replace(tensor, payload_offset=0, payload_alignment=0))
            continue
        cursor = _align_up(cursor, UOCR_TENSOR_PAYLOAD_ALIGNMENT)
        laid_out.append(replace(tensor, payload_offset=cursor, payload_alignment=UOCR_TENSOR_PAYLOAD_ALIGNMENT))
        cursor += tensor.output_bytes

    tensor_data_size = cursor - tensor_data_offset
    sections = (
        SectionPlan(UOCR_SECTION_CONFIG, SECTION_NAMES[UOCR_SECTION_CONFIG], config_offset, UOCR_CONFIG_RECORD_SIZE, 8),
        SectionPlan(
            UOCR_SECTION_TOKENIZER_METADATA,
            SECTION_NAMES[UOCR_SECTION_TOKENIZER_METADATA],
            tokenizer_offset,
            UOCR_TOKENIZER_METADATA_RECORD_SIZE,
            8,
        ),
        SectionPlan(
            UOCR_SECTION_PROVENANCE,
            SECTION_NAMES[UOCR_SECTION_PROVENANCE],
            provenance_offset,
            UOCR_PROVENANCE_RECORD_SIZE,
            8,
        ),
        SectionPlan(
            UOCR_SECTION_TENSOR_DIRECTORY,
            SECTION_NAMES[UOCR_SECTION_TENSOR_DIRECTORY],
            tensor_dir_offset,
            tensor_dir_size,
            8,
        ),
        SectionPlan(
            UOCR_SECTION_TENSOR_DATA,
            SECTION_NAMES[UOCR_SECTION_TENSOR_DATA],
            tensor_data_offset,
            tensor_data_size,
            UOCR_TENSOR_DATA_ALIGNMENT,
        ),
    )
    metadata_bytes = cursor - tensor_data_size
    return sections, tuple(laid_out), cursor, metadata_bytes


def _read_safetensors_header_and_data_start(path: Path) -> tuple[dict[str, Any], int | None]:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) == 8:
            header_len = struct.unpack("<Q", prefix)[0]
            # Cached `.header` files and real `.safetensors` files both use the
            # safetensors 8-byte length prefix.  Keep this bounded so a bad path
            # cannot accidentally request a huge read during dry-run tests.
            if 0 < header_len < 256 * 1024 * 1024:
                header_bytes = f.read(header_len)
                if len(header_bytes) == header_len and header_bytes.lstrip().startswith(b"{"):
                    return json.loads(header_bytes), 8 + int(header_len)
        f.seek(0)
        return json.loads(f.read().decode("utf-8")), None


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    return _read_safetensors_header_and_data_start(path)[0]


def _default_header_path(hf_dir: Path) -> Path:
    candidates = [
        hf_dir / "model.safetensors.header",
        hf_dir / "model-00001-of-000001.safetensors",
        hf_dir / "model.safetensors",
    ]
    candidates.extend(sorted(hf_dir.glob("*.safetensors")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no safetensors header or file found under {hf_dir}")


def _default_safetensors_payload_path(hf_dir: Path) -> Path:
    candidates = [hf_dir / "model-00001-of-000001.safetensors", hf_dir / "model.safetensors"]
    candidates.extend(sorted(path for path in hf_dir.glob("*.safetensors") if not path.name.endswith(".header")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no real .safetensors payload file found under {hf_dir}")


def _check_config_value(actual: Any, expected: int | float | str | bool, key: str) -> None:
    if isinstance(expected, float):
        if abs(float(actual) - expected) > 1e-12:
            raise ValueError(f"config {key} mismatch: got {actual!r}, expected {expected!r}")
    elif actual != expected:
        raise ValueError(f"config {key} mismatch: got {actual!r}, expected {expected!r}")


def validate_config(hf_dir: Path) -> None:
    config_path = hf_dir / "config.json"
    if not config_path.exists():
        return
    config = _read_json(config_path)
    for key, expected in EXPECTED_CONFIG.items():
        if key not in config:
            raise ValueError(f"config missing required key {key!r}")
        _check_config_value(config[key], expected, key)
    for key, expected in EXPECTED_CONFIG_DEFAULTS.items():
        if key in config:
            _check_config_value(config[key], expected, key)

    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    if head_dim != 128:
        raise ValueError(f"derived attention head dim mismatch: got {head_dim}, expected 128")


def _validate_header_entry(name: str, entry: Mapping[str, Any]) -> None:
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if dtype != EXPECTED_SOURCE_DTYPE:
        raise ValueError(f"tensor {name} dtype mismatch: got {dtype!r}, expected {EXPECTED_SOURCE_DTYPE}")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError(f"tensor {name} has invalid shape {shape!r}")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(v, int) and v >= 0 for v in offsets):
        raise ValueError(f"tensor {name} has invalid data_offsets {offsets!r}")
    if offsets[1] < offsets[0]:
        raise ValueError(f"tensor {name} has descending data_offsets {offsets!r}")
    expected_bytes = _product(shape) * 2
    actual_bytes = offsets[1] - offsets[0]
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"tensor {name} byte count mismatch: offsets imply {actual_bytes}, shape implies {expected_bytes}"
        )


def _validate_key_shapes(entries: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected_shape in KEY_TENSOR_SHAPES.items():
        entry = entries.get(name)
        if entry is None:
            raise ValueError(f"expected tensor {name!r} is missing")
        actual_shape = tuple(int(dim) for dim in entry["shape"])
        if actual_shape != expected_shape:
            raise ValueError(f"tensor {name} shape mismatch: got {actual_shape}, expected {expected_shape}")


def _usage_for_name(name: str) -> tuple[str, int, str]:
    if name.startswith("model.vision_model.embeddings.patch_embedding."):
        return (
            "preserved-unused",
            2,
            "CLIP receives SAM feature maps as patch_embeds in the normal OCR path",
        )
    if name.endswith(".mlp.gate.weight"):
        return "runtime", 1, "MoE router: mandatory fp16 keep-list"
    if "norm" in name or name.endswith("bias") or name in {"model.image_newline", "model.view_seperator"}:
        return "runtime", 1, "fp16 keep-list for norms/biases/special embeddings"
    return "runtime", 1, "fp16 baseline"


def _projection_for_plan(registry_entry: TensorRegistryEntry) -> tuple[str, int]:
    projection = registry_entry.projection
    if projection == TensorProjection.NONE:
        return "NONE", 0
    return projection.name, int(projection)


def _make_tensor_plan(name: str, entry: Mapping[str, Any], qprofile: str, registry_entry: TensorRegistryEntry) -> TensorPlan:
    if qprofile != "fp16":
        raise NotImplementedError("only fp16 dry-run planning is implemented")
    shape = tuple(int(dim) for dim in entry["shape"])
    offsets = (int(entry["data_offsets"][0]), int(entry["data_offsets"][1]))
    source_bytes = offsets[1] - offsets[0]
    usage, usage_id, reason = _usage_for_name(name)
    projection_name, projection_id = _projection_for_plan(registry_entry)
    return TensorPlan(
        name=name,
        source_dtype=str(entry["dtype"]),
        shape=shape,
        physical_shape=shape,
        source_offsets=offsets,
        source_bytes=source_bytes,
        output_dtype="F16",
        qtype="UOCR_TENSOR_F16",
        qtype_id=1,
        output_bytes=source_bytes,
        payload_offset=0,
        payload_alignment=UOCR_TENSOR_PAYLOAD_ALIGNMENT,
        block_size=0,
        row_size=0,
        tensor_id=registry_entry.tensor_id,
        family=registry_entry.family.name,
        family_id=int(registry_entry.family),
        layer=registry_entry.layer,
        expert=registry_entry.expert,
        projection=projection_name,
        projection_id=projection_id,
        usage=usage,
        usage_id=usage_id,
        reason=reason,
    )


def _relayout_plan(plan: DryRunPlan, tensors: Iterable[TensorPlan]) -> DryRunPlan:
    ordered = tuple(sorted(tensors, key=lambda tensor: tensor.tensor_id))
    sections, laid_out, planned_file_size, metadata_bytes = _layout_dry_run_file(ordered)
    return replace(
        plan,
        tensors=laid_out,
        total_source_bytes=sum(t.source_bytes for t in laid_out),
        total_output_bytes=sum(t.output_bytes for t in laid_out),
        planned_file_size=planned_file_size,
        metadata_bytes=metadata_bytes,
        sections=sections,
        qtype_histogram=_qtype_histogram(laid_out),
        usage_histogram=_usage_histogram(laid_out),
        family_histogram=_family_histogram(laid_out),
    )


def filter_plan_tensors(plan: DryRunPlan, selector: str | int | None) -> DryRunPlan:
    """Return a single-tensor plan selected by source name or stable tensor id."""

    if selector is None:
        return plan
    if isinstance(selector, int):
        wanted_id = selector
        matches = [tensor for tensor in plan.tensors if tensor.tensor_id == wanted_id]
    else:
        text = selector.strip()
        matches = [tensor for tensor in plan.tensors if tensor.name == text]
        if not matches:
            try:
                wanted_id = int(text, 0)
            except ValueError:
                wanted_id = -1
            if wanted_id >= 0:
                matches = [tensor for tensor in plan.tensors if tensor.tensor_id == wanted_id]
    if len(matches) != 1:
        raise ValueError(f"tensor selector {selector!r} matched {len(matches)} tensors")
    return _relayout_plan(plan, matches)


def build_dry_run_plan(
    hf_dir: str | Path,
    *,
    qprofile: str = "fp16",
    header_path: str | Path | None = None,
    index_path: str | Path | None = None,
    strict: bool = True,
) -> DryRunPlan:
    hf_dir = Path(hf_dir)
    if qprofile != "fp16":
        raise NotImplementedError("only --qprofile fp16 is implemented for the first converter milestone")

    validate_config(hf_dir)

    header = _read_safetensors_header(Path(header_path) if header_path is not None else _default_header_path(hf_dir))
    entries: dict[str, Mapping[str, Any]] = {
        name: entry for name, entry in header.items() if name != "__metadata__"
    }
    if not entries:
        raise ValueError("safetensors header contains no tensors")
    if strict and len(entries) != EXPECTED_TENSOR_COUNT:
        raise ValueError(f"safetensors tensor count mismatch: got {len(entries)}, expected {EXPECTED_TENSOR_COUNT}")

    for name, entry in entries.items():
        _validate_header_entry(name, entry)
    if strict:
        _validate_key_shapes(entries)

    registry = build_tensor_registry(entries)
    if strict:
        validate_registry_shapes(registry, {name: tuple(int(dim) for dim in entry["shape"]) for name, entry in entries.items()})

    tensors = tuple(
        _make_tensor_plan(name, entries[name], qprofile, registry[name])
        for name in sorted(entries, key=lambda key: registry[key].tensor_id)
    )
    sections, tensors, planned_file_size, metadata_bytes = _layout_dry_run_file(tensors)
    total_source_bytes = sum(t.source_bytes for t in tensors)
    total_output_bytes = sum(t.output_bytes for t in tensors)
    if strict and total_source_bytes != EXPECTED_TOTAL_BYTES:
        raise ValueError(
            f"source payload byte count mismatch: got {total_source_bytes}, expected {EXPECTED_TOTAL_BYTES}"
        )

    index_file = Path(index_path) if index_path is not None else hf_dir / "model.safetensors.index.json"
    index = _read_json_if_exists(index_file)
    if index is None:
        if strict:
            raise FileNotFoundError(f"missing safetensors index file {index_file}")
    else:
        index_total = int(index.get("metadata", {}).get("total_size", -1))
        if index_total >= 0 and index_total != total_source_bytes:
            raise ValueError(f"index total_size mismatch: got {index_total}, header implies {total_source_bytes}")
        weight_map = index.get("weight_map", {})
        if strict and len(weight_map) != len(entries):
            raise ValueError(f"index weight_map count mismatch: got {len(weight_map)}, expected {len(entries)}")
        missing_from_index = sorted(set(entries) - set(weight_map))
        if missing_from_index:
            raise ValueError(f"{len(missing_from_index)} header tensors missing from index; first={missing_from_index[0]}")

    return DryRunPlan(
        hf_dir=hf_dir,
        qprofile=qprofile,
        tensors=tensors,
        total_source_bytes=total_source_bytes,
        total_output_bytes=total_output_bytes,
        planned_file_size=planned_file_size,
        metadata_bytes=metadata_bytes,
        sections=sections,
        qtype_histogram=_qtype_histogram(tensors),
        usage_histogram=_usage_histogram(tensors),
        family_histogram=_family_histogram(tensors),
    )


def _config_record_bytes() -> bytes:
    return _CONFIG_RECORD_STRUCT.pack(
        129_280,
        128_000,
        830,
        0,
        1,
        2,
        128_815,
        1280,
        12,
        10,
        10,
        128,
        10_000,
        32_768,
        128,
        1,
        64,
        6,
        896,
        2,
        6848,
        2048,
        1280,
        16,
        4,
        1024,
        640,
    )


def _section_bytes(plan: DryRunPlan) -> bytes:
    return b"".join(
        _SECTION_ENTRY_STRUCT.pack(section.section_type, 0, section.offset, section.size, section.alignment)
        for section in plan.sections
    )


def _usage_counts(plan: DryRunPlan) -> tuple[int, int, int]:
    runtime = sum(1 for tensor in plan.tensors if tensor.usage_id == UOCR_TENSOR_USAGE_RUNTIME)
    preserved = sum(1 for tensor in plan.tensors if tensor.usage_id == UOCR_TENSOR_USAGE_PRESERVED_UNUSED)
    omitted = sum(1 for tensor in plan.tensors if tensor.usage_id == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON)
    return runtime, preserved, omitted


def _tokenizer_metadata_bytes(tokenizer_hash: bytes) -> bytes:
    return _TOKENIZER_METADATA_STRUCT.pack(
        UOCR_TOKENIZER_METADATA_MAGIC,
        UOCR_FORMAT_VERSION,
        UOCR_TOKENIZER_METADATA_RECORD_SIZE,
        UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED,
        tokenizer_hash,
        129_280,
        128_000,
        830,
        0,
        1,
        2,
        128_815,
        0,
        0,
    )


def _provenance_bytes(
    plan: DryRunPlan,
    *,
    config_hash: bytes,
    tokenizer_hash: bytes,
    index_hash: bytes,
    safetensors_file_count: int,
) -> bytes:
    runtime, preserved, omitted = _usage_counts(plan)
    return _PROVENANCE_RECORD_STRUCT.pack(
        UOCR_PROVENANCE_MAGIC,
        UOCR_FORMAT_VERSION,
        UOCR_PROVENANCE_RECORD_SIZE,
        0,
        len(plan.tensors),
        runtime,
        preserved,
        omitted,
        safetensors_file_count,
        QPROFILE_IDS[plan.qprofile],
        CONVERTER_VERSION[0],
        CONVERTER_VERSION[1],
        CONVERTER_VERSION[2],
        0,
        config_hash,
        tokenizer_hash,
        index_hash,
        bytes(32),
        0,
        0,
    )


def _tensor_directory_bytes(plan: DryRunPlan) -> bytes:
    payload = bytearray()
    payload += _TENSOR_DIRECTORY_HEADER_STRUCT.pack(
        UOCR_TENSOR_DIR_MAGIC,
        UOCR_FORMAT_VERSION,
        UOCR_TENSOR_ENTRY_SIZE,
        len(plan.tensors),
    )
    for tensor in plan.tensors:
        if len(tensor.shape) > 4 or len(tensor.physical_shape) > 4:
            raise ValueError(f"tensor {tensor.name} rank exceeds .uocr limit: {len(tensor.shape)}")
        logical_shape = tuple(int(dim) for dim in tensor.shape) + (0,) * (4 - len(tensor.shape))
        physical_shape = tuple(int(dim) for dim in tensor.physical_shape) + (0,) * (4 - len(tensor.physical_shape))
        payload += _TENSOR_ENTRY_STRUCT.pack(
            tensor.tensor_id,
            tensor.family_id,
            tensor.layer,
            tensor.expert,
            tensor.projection_id,
            tensor.usage_id,
            tensor.qtype_id,
            0,
            len(tensor.shape),
            *logical_shape,
            *physical_shape,
            tensor.payload_offset,
            tensor.output_bytes,
            tensor.block_size,
            tensor.row_size,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        )
    return bytes(payload)


def _source_files_for_plan(plan: DryRunPlan) -> set[str]:
    index = _read_json_if_exists(plan.hf_dir / "model.safetensors.index.json")
    if index is None:
        return set()
    weight_map = index.get("weight_map", {})
    return {str(weight_map[tensor.name]) for tensor in plan.tensors if tensor.name in weight_map}


def _resolve_safetensors_payload_path(plan: DryRunPlan, safetensors_path: str | Path | None) -> tuple[Path, int]:
    if safetensors_path is not None:
        source = Path(safetensors_path)
    else:
        source_files = _source_files_for_plan(plan)
        if len(source_files) == 1:
            source = plan.hf_dir / next(iter(source_files))
        elif len(source_files) > 1:
            raise NotImplementedError("multi-file safetensors streaming is not implemented yet")
        else:
            source = _default_safetensors_payload_path(plan.hf_dir)
    if not source.exists():
        raise FileNotFoundError(f"safetensors payload file does not exist: {source}")
    header, data_start = _read_safetensors_header_and_data_start(source)
    if data_start is None:
        raise ValueError(f"{source} is a cached header/json file, not a real safetensors payload")
    required_end = max((tensor.source_offsets[1] for tensor in plan.tensors), default=0)
    if source.stat().st_size < data_start + required_end:
        raise ValueError(f"{source} does not contain the planned safetensors payload bytes")
    header_names = {name for name in header if name != "__metadata__"}
    missing = [tensor.name for tensor in plan.tensors if tensor.name not in header_names]
    if missing:
        raise ValueError(f"{len(missing)} planned tensors are missing from {source}; first={missing[0]}")
    for tensor in plan.tensors:
        entry = header[tensor.name]
        shape = tuple(int(dim) for dim in entry.get("shape", ()))
        offsets = tuple(int(value) for value in entry.get("data_offsets", ()))
        if entry.get("dtype") != EXPECTED_SOURCE_DTYPE or shape != tensor.shape or offsets != tensor.source_offsets:
            raise ValueError(f"source safetensors metadata for {tensor.name} does not match the conversion plan")
    return source, data_start


def _stream_bf16_to_f16(src: BinaryIO, dst: BinaryIO, *, source_bytes: int, chunk_bytes: int = 32 * 1024 * 1024) -> None:
    if source_bytes % 2 != 0:
        raise ValueError(f"BF16 tensor byte count must be even, got {source_bytes}")
    remaining = source_bytes
    chunk_bytes = max(2, chunk_bytes - (chunk_bytes % 2))
    while remaining:
        raw = src.read(min(remaining, chunk_bytes))
        if not raw:
            raise EOFError("unexpected end of safetensors payload while streaming tensor")
        if len(raw) % 2 != 0:
            raise ValueError("read an odd number of BF16 bytes")
        bf16 = np.frombuffer(raw, dtype=np.dtype("<u2"))
        fp32_bits = (bf16.astype(np.uint32) << np.uint32(16))
        fp32 = fp32_bits.view(np.dtype("<f4"))
        fp16 = fp32.astype(np.dtype("<f2"), copy=False)
        dst.write(fp16.tobytes())
        remaining -= len(raw)


def write_uocr_model(
    plan: DryRunPlan,
    out_path: str | Path,
    *,
    safetensors_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Stream a planned fp16 `.uocr` file to disk.

    The current writer supports the fp16 baseline only.  It converts each BF16
    safetensors range to fp16 in bounded chunks and writes directly into the
    mmap-friendly layout produced by :func:`build_dry_run_plan`.
    """

    if plan.qprofile != "fp16":
        raise NotImplementedError("only fp16 .uocr writing is implemented")
    if plan.tensor_count == 0:
        raise ValueError("cannot write a .uocr with no tensors")

    out = Path(out_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    source, data_start = _resolve_safetensors_payload_path(plan, safetensors_path)
    config_path = plan.hf_dir / "config.json"
    tokenizer_path = plan.hf_dir / "tokenizer.json"
    index_path = plan.hf_dir / "model.safetensors.index.json"
    config_hash = _sha256_file(config_path)
    tokenizer_hash = _sha256_file(tokenizer_path)
    index_hash = _sha256_file(index_path)
    source_file_count = max(1, len(_source_files_for_plan(plan)))

    tmp = out.with_name(f".{out.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with source.open("rb") as src, tmp.open("w+b") as dst:
            header = _FILE_HEADER_STRUCT.pack(
                b"UOCR",
                UOCR_FORMAT_VERSION,
                UOCR_FILE_HEADER_SIZE,
                UOCR_ENDIAN_MARKER,
                UOCR_FILE_ALIGNMENT,
                QPROFILE_IDS[plan.qprofile],
                len(plan.sections),
                0,
                plan.planned_file_size,
                UOCR_FILE_HEADER_SIZE,
                index_hash,
                bytes(32),
            )
            dst.seek(0)
            dst.write(header)
            dst.seek(UOCR_FILE_HEADER_SIZE)
            dst.write(_section_bytes(plan))
            section_by_type = {section.section_type: section for section in plan.sections}
            dst.seek(section_by_type[UOCR_SECTION_CONFIG].offset)
            dst.write(_config_record_bytes())
            dst.seek(section_by_type[UOCR_SECTION_TOKENIZER_METADATA].offset)
            dst.write(_tokenizer_metadata_bytes(tokenizer_hash))
            dst.seek(section_by_type[UOCR_SECTION_PROVENANCE].offset)
            dst.write(
                _provenance_bytes(
                    plan,
                    config_hash=config_hash,
                    tokenizer_hash=tokenizer_hash,
                    index_hash=index_hash,
                    safetensors_file_count=source_file_count,
                )
            )
            dst.seek(section_by_type[UOCR_SECTION_TENSOR_DIRECTORY].offset)
            dst.write(_tensor_directory_bytes(plan))

            for tensor in plan.tensors:
                if tensor.usage_id == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON:
                    continue
                src.seek(data_start + tensor.source_offsets[0])
                dst.seek(tensor.payload_offset)
                before = dst.tell()
                _stream_bf16_to_f16(src, dst, source_bytes=tensor.source_bytes)
                written = dst.tell() - before
                if written != tensor.output_bytes:
                    raise IOError(f"tensor {tensor.name} wrote {written} bytes, expected {tensor.output_bytes}")
            dst.truncate(plan.planned_file_size)
        os.replace(tmp, out)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return out


def _print_summary(plan: DryRunPlan, *, dry_run: bool) -> None:
    print("Unlimited-OCR converter dry-run" if dry_run else "Unlimited-OCR converter plan")
    print(f"hf_dir: {plan.hf_dir}")
    print(f"qprofile: {plan.qprofile}")
    print(f"tensors: {plan.tensor_count}")
    print(f"source bytes: {plan.total_source_bytes}")
    print(f"planned output bytes: {plan.total_output_bytes}")
    print(f"planned file size: {plan.planned_file_size}")
    tensor_data = next((section for section in plan.sections if section.section_type == UOCR_SECTION_TENSOR_DATA), None)
    if tensor_data is not None:
        print(f"tensor data: offset={tensor_data.offset} size={tensor_data.size} alignment={tensor_data.alignment}")
    print(f"qtypes: {dict(plan.qtype_histogram)}")
    print(f"usage: {dict(plan.usage_histogram)}")
    print("families:")
    for family, count in sorted(plan.family_histogram.items()):
        print(f"  {family}: {count}")


def _write_dump(plan: DryRunPlan, path: str) -> None:
    payload = json.dumps(plan.as_dict(), indent=2, sort_keys=True)
    if path == "-":
        print(payload)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or write an Unlimited-OCR fp16 .uocr conversion")
    parser.add_argument("--hf-dir", type=Path, default=project_root() / "data/context")
    parser.add_argument("--header", type=Path, default=None, help="optional safetensors header/cache path")
    parser.add_argument("--index", type=Path, default=None, help="optional safetensors index path")
    parser.add_argument("--safetensors", type=Path, default=None, help="explicit source .safetensors payload for --out")
    parser.add_argument("--qprofile", choices=["fp16", "dyn-q8", "dyn-q4"], default="fp16")
    parser.add_argument("--dry-run", action="store_true", help="plan only; does not require full weights")
    parser.add_argument("--out", type=Path, default=None, help="write an fp16 .uocr file to this path")
    parser.add_argument("--tensor", default=None, help="optional single tensor source name or stable tensor id")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    parser.add_argument("--dump-plan", nargs="?", const="-", default=None, help="write full JSON plan to path or stdout")
    parser.add_argument(
        "--relaxed-validation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.out is None and not args.dry_run:
        parser.error("either --dry-run or --out is required")
    if args.qprofile != "fp16":
        parser.error("only --qprofile fp16 is implemented in this milestone")

    plan = build_dry_run_plan(
        args.hf_dir,
        qprofile=args.qprofile,
        header_path=args.header,
        index_path=args.index,
        strict=not args.relaxed_validation,
    )
    plan = filter_plan_tensors(plan, args.tensor)
    _print_summary(plan, dry_run=args.dry_run or args.out is None)
    if args.dump_plan is not None:
        _write_dump(plan, args.dump_plan)
    if args.out is not None:
        written = write_uocr_model(plan, args.out, safetensors_path=args.safetensors, overwrite=args.overwrite)
        print(f"wrote: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
