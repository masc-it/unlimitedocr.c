"""FP16 conversion planning and `.uocr` writing for Unlimited-OCR.

The planner can run from the cached safetensors header/index only, which keeps
fast tests independent from the 6.67 GB checkpoint payload.  When the real
safetensors file is present, the writer streams tensor ranges into the planned
`.uocr` layout and converts BF16 payload bytes to fp16 without allocating a
full-model temporary buffer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
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
from .quant_cfg import QuantConfig, load_default_quant_config, load_quant_config
from .tensor_registry import (
    TensorFamily,
    TensorProjection,
    TensorRegistryEntry,
    build_tensor_registry,
    tensor_id_moe_expert,
    validate_registry_shapes,
)

EXPECTED_TENSOR_COUNT = 2710
EXPECTED_TOTAL_BYTES = 6_672_212_480
EXPECTED_SOURCE_DTYPE = "BF16"

MOE_ROUTED_LAYER_START = 1
MOE_ROUTED_LAYER_END_EXCLUSIVE = 12
MOE_ROUTED_EXPERT_COUNT = 64
MOE_EXPERT_PROJECTION_ORDER = (TensorProjection.GATE, TensorProjection.UP, TensorProjection.DOWN)
MOE_EXPERT_PACKING_LAYOUT = "interleaved-expert-major"
MOE_EXPERT_PACKING_CONTRACT = (
    "For each routed-MoE layer, expert payloads are contiguous as "
    "[expert][gate_proj,up_proj,down_proj][out_row][packed_input]. "
    "This gives Metal selected-expert kernels one expert-major slab with "
    "gate/up colocated for fused projection and down immediately after."
)

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
UOCR_QPROFILE_MIXED_Q8_0 = 2
#: mixed-q4: routed MoE experts Q4_0, remaining quantized decoder modules Q8_0,
#: everything else fp16 (see docs/plan_q4.md).  Not yet in QPROFILE_IDS: the
#: converter CLI gains "mixed-q4" with the Q4 planner/writer (plan step 2).
UOCR_QPROFILE_MIXED_Q4 = 3
UOCR_TOKENIZER_FLAG_C_V1_NOT_REQUIRED = 1 << 0
UOCR_PROVENANCE_FLAG_JSON_PAYLOAD = 1 << 0
UOCR_TOKENIZER_METADATA_MAGIC = 0x4B4F5455
UOCR_PROVENANCE_MAGIC = 0x564F5250
UOCR_TENSOR_DIR_MAGIC = 0x52494454
UOCR_TENSOR_USAGE_RUNTIME = 1
UOCR_TENSOR_USAGE_PRESERVED_UNUSED = 2
UOCR_TENSOR_USAGE_OMITTED_WITH_REASON = 3

UOCR_TENSOR_FLAG_ROW_MAJOR = 1 << 0
UOCR_TENSOR_FLAG_TRANSPOSED = 1 << 1
UOCR_TENSOR_FLAG_FLATTENED_LEADING_DIM = 1 << 2

UOCR_LAYOUT_TRANSFORM_IDENTITY = 0
UOCR_LAYOUT_TRANSFORM_FLATTEN_LEADING_DIM = 1
UOCR_LAYOUT_TRANSFORM_TRANSPOSE = 2
ROW_MAJOR_LAYOUT_CONTRACT = (
    "Runtime tensor payloads preserve safetensors row-major order. Rank-2 weights remain [out,in]; "
    "rank>2 tensors keep their source shape without transposition."
)

UOCR_QTYPE_REASON_UNKNOWN = 0
UOCR_QTYPE_REASON_FP16_BASELINE = 1

UOCR_PROMOTION_NONE = 0

UOCR_TENSOR_F16 = 1
UOCR_TENSOR_F32 = 2
UOCR_TENSOR_Q8_0 = 3
UOCR_TENSOR_Q4_0 = 4
#: Reserved for the asymmetric scale+min fallback scheme; not implemented.
UOCR_TENSOR_Q4_1 = 5

UOCR_Q8_GROUP_SIZE_DEFAULT = 64
UOCR_Q8_MIN = -127
UOCR_Q8_MAX = 127

UOCR_Q4_GROUP_SIZE_DEFAULT = 64
UOCR_Q4_MIN = -8
UOCR_Q4_MAX = 7

CONVERTER_VERSION = (0, 3, 0)

QUANT_POLICY_EMBEDDINGS_DECODER = "embeddings+decoder"
#: Candidate (family, projection) universe for cfg-gated mixed Q8 planning.
#: The runtime-safe subset actually quantized is governed by ``configs/quant-cfg.yaml``
#: (see :mod:`unlimitedocr_c.quant_cfg``); this constant only documents which
#: families/projections the current rollout considers at all.
Q8_POLICY_CANDIDATE_FAMILIES = frozenset({
    TensorFamily.TOK_EMBED,
    TensorFamily.LM_HEAD,
    TensorFamily.LAYER_ATTN,
    TensorFamily.LAYER_DENSE_MLP,
    TensorFamily.MOE_EXPERT,
    TensorFamily.MOE_SHARED,
    TensorFamily.PROJECTOR,
    TensorFamily.VISION_CLIP,
    TensorFamily.VISION_SAM,
})
Q8_POLICY_CANDIDATE_PROJECTIONS = frozenset({
    TensorProjection.WEIGHT,
    TensorProjection.Q,
    TensorProjection.K,
    TensorProjection.V,
    TensorProjection.O,
    TensorProjection.GATE,
    TensorProjection.UP,
    TensorProjection.DOWN,
    TensorProjection.VISION_ATTN_QKV,
    TensorProjection.VISION_ATTN_O,
    TensorProjection.VISION_MLP_FC1,
    TensorProjection.VISION_MLP_FC2,
})

CLIP_PIXEL_PATCH_EMBEDDING_PREFIX = "model.vision_model.embeddings.patch_embedding."
PRESERVED_UNUSED_NORMAL_OCR_PREFIXES: tuple[str, ...] = (CLIP_PIXEL_PATCH_EMBEDDING_PREFIX,)

QPROFILE_IDS: Mapping[str, int] = {
    "fp16": UOCR_QPROFILE_FP16,
    "mixed-q8_0": UOCR_QPROFILE_MIXED_Q8_0,
    "mixed-q4": UOCR_QPROFILE_MIXED_Q4,
}

#: qprofiles whose planning is cfg-gated mixed quantization.
MIXED_QPROFILES = frozenset({"mixed-q8_0", "mixed-q4"})

QTYPE_REASON_NAMES: Mapping[int, str] = {
    UOCR_QTYPE_REASON_UNKNOWN: "unknown",
    UOCR_QTYPE_REASON_FP16_BASELINE: "fp16-baseline",
}

PROMOTION_REASON_NAMES: Mapping[int, str] = {
    UOCR_PROMOTION_NONE: "none",
}

LAYOUT_TRANSFORM_NAMES: Mapping[int, str] = {
    UOCR_LAYOUT_TRANSFORM_IDENTITY: "identity-row-major",
    UOCR_LAYOUT_TRANSFORM_FLATTEN_LEADING_DIM: "flatten-leading-dim-row-major",
    UOCR_LAYOUT_TRANSFORM_TRANSPOSE: "transpose",
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

EXPECTED_TOP_LEVEL_CONFIG: Mapping[str, int | str | bool | list[list[int]]] = {
    "model_type": "unlimited-ocr",
    "torch_dtype": "bfloat16",
    "global_view_pos": "head",
    "tile_tag": "2D",
    "candidate_resolutions": [[1024, 1024]],
    "use_mla": False,
    "sliding_window": 128,
}

EXPECTED_PROJECTOR_CONFIG: Mapping[str, int | str] = {
    "input_dim": 2048,
    "n_embed": 1280,
    "projector_type": "linear",
}

EXPECTED_PROCESSOR_CONFIG: Mapping[str, int | str | bool | list[float] | list[list[int]]] = {
    "processor_class": "UnlimitedOCRHFProcessor",
    "candidate_resolutions": [[1024, 1024]],
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "image_token": "<image>",
    "pad_token": "<｜▁pad▁｜>",
    "patch_size": 16,
    "downsample_ratio": 4,
    "normalize": True,
    "add_special_token": False,
    "mask_prompt": False,
    "ignore_id": -100,
    "sft_format": "unlimitedocr",
}

EXPECTED_TOKENIZER_CLASS = "LlamaTokenizerFast"
EXPECTED_TOKENIZER_MODEL_TYPE = "BPE"
EXPECTED_SPECIAL_TOKENS: Mapping[int, str] = {
    0: "<｜begin▁of▁sentence｜>",
    1: "<｜end▁of▁sentence｜>",
    2: "<｜▁pad▁｜>",
    128_815: "<image>",
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
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    source_offsets: tuple[int, int]
    source_bytes: int
    output_dtype: str
    qtype: str
    qtype_id: int
    output_bytes: int
    payload_offset: int
    payload_alignment: int
    scale_offset: int
    scale_size: int
    qweight_bytes: int
    qscale_bytes: int
    fp16_equivalent_bytes: int
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
    qtype_reason: str
    qtype_reason_id: int
    promotion_reason: str
    promotion_reason_id: int
    source_layout: str
    runtime_layout: str
    layout_transform: str
    layout_transform_id: int
    layout_flags: int
    transposed: bool
    layout_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_dtype": self.source_dtype,
            "shape": list(self.shape),
            "logical_shape": list(self.logical_shape),
            "physical_shape": list(self.physical_shape),
            "source_offsets": list(self.source_offsets),
            "source_bytes": self.source_bytes,
            "output_dtype": self.output_dtype,
            "qtype": self.qtype,
            "qtype_id": self.qtype_id,
            "output_bytes": self.output_bytes,
            "payload_offset": self.payload_offset,
            "payload_alignment": self.payload_alignment,
            "scale_offset": self.scale_offset,
            "scale_size": self.scale_size,
            "qweight_bytes": self.qweight_bytes,
            "qscale_bytes": self.qscale_bytes,
            "fp16_equivalent_bytes": self.fp16_equivalent_bytes,
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
            "qtype_reason": self.qtype_reason,
            "qtype_reason_id": self.qtype_reason_id,
            "promotion_reason": self.promotion_reason,
            "promotion_reason_id": self.promotion_reason_id,
            "source_layout": self.source_layout,
            "runtime_layout": self.runtime_layout,
            "layout_transform": self.layout_transform,
            "layout_transform_id": self.layout_transform_id,
            "layout_flags": self.layout_flags,
            "transposed": self.transposed,
            "layout_reason": self.layout_reason,
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
class SourceMetadata:
    config: dict[str, Any]
    processor: dict[str, Any]
    tokenizer: dict[str, Any]
    safetensors: dict[str, Any]
    hashes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "processor": dict(self.processor),
            "tokenizer": dict(self.tokenizer),
            "safetensors": dict(self.safetensors),
            "hashes": dict(self.hashes),
        }


@dataclass
class _StreamingValueStats:
    value_count: int = 0
    finite_count: int = 0
    nan_count: int = 0
    posinf_count: int = 0
    neginf_count: int = 0
    min_value: float | None = None
    max_value: float | None = None

    @property
    def nonfinite_count(self) -> int:
        return self.nan_count + self.posinf_count + self.neginf_count

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        self.value_count += int(flat.size)
        if flat.size == 0:
            return
        nan_mask = np.isnan(flat)
        posinf_mask = np.isposinf(flat)
        neginf_mask = np.isneginf(flat)
        self.nan_count += int(np.count_nonzero(nan_mask))
        self.posinf_count += int(np.count_nonzero(posinf_mask))
        self.neginf_count += int(np.count_nonzero(neginf_mask))
        finite_values = flat[np.isfinite(flat)]
        self.finite_count += int(finite_values.size)
        if finite_values.size == 0:
            return
        chunk_min = float(np.min(finite_values))
        chunk_max = float(np.max(finite_values))
        self.min_value = chunk_min if self.min_value is None else min(self.min_value, chunk_min)
        self.max_value = chunk_max if self.max_value is None else max(self.max_value, chunk_max)


@dataclass(frozen=True)
class TensorConversionStats:
    name: str
    tensor_id: int
    source_dtype: str
    output_dtype: str
    qtype: str
    qtype_id: int
    shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    source_bytes: int
    output_bytes: int
    output_bytes_written: int
    payload_offset: int
    scale_offset: int
    scale_size: int
    value_count: int
    finite_count: int
    source_min: float | None
    source_max: float | None
    source_nan_count: int
    source_posinf_count: int
    source_neginf_count: int
    source_nonfinite_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tensor_id": self.tensor_id,
            "source_dtype": self.source_dtype,
            "output_dtype": self.output_dtype,
            "qtype": self.qtype,
            "qtype_id": self.qtype_id,
            "shape": list(self.shape),
            "logical_shape": list(self.logical_shape),
            "physical_shape": list(self.physical_shape),
            "source_bytes": self.source_bytes,
            "output_bytes": self.output_bytes,
            "output_bytes_written": self.output_bytes_written,
            "payload_offset": self.payload_offset,
            "scale_offset": self.scale_offset,
            "scale_size": self.scale_size,
            "value_count": self.value_count,
            "finite_count": self.finite_count,
            "source_min": self.source_min,
            "source_max": self.source_max,
            "source_nan_count": self.source_nan_count,
            "source_posinf_count": self.source_posinf_count,
            "source_neginf_count": self.source_neginf_count,
            "source_nonfinite_count": self.source_nonfinite_count,
        }


@dataclass(frozen=True)
class TensorCompareResult:
    name: str
    tensor_id: int
    qtype: str
    qtype_id: int
    model_path: str
    source_path: str
    expected_bytes: int
    actual_bytes: int
    compared_bytes: int
    expected_sha256: str
    actual_sha256: str
    payload_matches: bool
    metadata_mismatches: tuple[str, ...]
    first_mismatch_offset: int | None
    expected_byte: int | None
    actual_byte: int | None
    dequant_max_abs_error: float | None = None
    dequant_rmse: float | None = None

    @property
    def metadata_matches(self) -> bool:
        return not self.metadata_mismatches

    @property
    def matches(self) -> bool:
        return self.payload_matches and self.metadata_matches

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tensor_id": self.tensor_id,
            "qtype": self.qtype,
            "qtype_id": self.qtype_id,
            "model_path": self.model_path,
            "source_path": self.source_path,
            "expected_bytes": self.expected_bytes,
            "actual_bytes": self.actual_bytes,
            "compared_bytes": self.compared_bytes,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "metadata_matches": self.metadata_matches,
            "payload_matches": self.payload_matches,
            "matches": self.matches,
            "metadata_mismatches": list(self.metadata_mismatches),
            "first_mismatch_offset": self.first_mismatch_offset,
            "expected_byte": self.expected_byte,
            "actual_byte": self.actual_byte,
            "dequant_max_abs_error": self.dequant_max_abs_error,
            "dequant_rmse": self.dequant_rmse,
        }


@dataclass(frozen=True)
class _UocrTensorPayloadView:
    qprofile_id: int
    tensor_id: int
    qtype_id: int
    flags: int
    rank: int
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    payload_offset: int
    payload_size: int
    block_size: int
    row_size: int
    scale_offset: int
    scale_size: int
    qtype_reason_id: int
    promotion_reason_id: int


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
    source_metadata: SourceMetadata
    qtype_histogram: Mapping[str, int]
    qtype_reason_histogram: Mapping[str, int]
    promotion_reason_histogram: Mapping[str, int]
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
            "estimated_weight_bytes": self.total_output_bytes,
            "estimated_savings_bytes": self.total_source_bytes - self.total_output_bytes,
            "compression_ratio": (
                (self.total_output_bytes / self.total_source_bytes) if self.total_source_bytes else 0.0
            ),
            "planned_file_size": self.planned_file_size,
            "metadata_bytes": self.metadata_bytes,
            "sections": [section.as_dict() for section in self.sections],
            "source_metadata": self.source_metadata.as_dict(),
            "q8_summary": q8_summary(self.tensors),
            "qtype_histogram": dict(self.qtype_histogram),
            "qtype_reason_histogram": dict(self.qtype_reason_histogram),
            "promotion_reason_histogram": dict(self.promotion_reason_histogram),
            "usage_histogram": dict(self.usage_histogram),
            "family_histogram": dict(self.family_histogram),
            "layout_transforms": layout_transform_summary(self.tensors),
            "moe_expert_packing": {
                "layout": MOE_EXPERT_PACKING_LAYOUT,
                "contract": MOE_EXPERT_PACKING_CONTRACT,
                "layer_start": MOE_ROUTED_LAYER_START,
                "layer_end_exclusive": MOE_ROUTED_LAYER_END_EXCLUSIVE,
                "expert_count": MOE_ROUTED_EXPERT_COUNT,
                "projection_order": [projection.name for projection in MOE_EXPERT_PROJECTION_ORDER],
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_dict(),
            "tensors": [tensor.as_dict() for tensor in self.tensors],
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=16)
def _read_json_cached(path: str) -> Any:
    return _read_json(Path(path))


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_json_object_if_exists(path: Path, *, strict: bool, label: str) -> dict[str, Any] | None:
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"missing {label} file {path}")
        return None
    data = _read_json_cached(str(path.resolve()))
    if not isinstance(data, dict):
        raise ValueError(f"{label} file {path} must contain a JSON object")
    return data


def _sha256_file(path: Path | None) -> bytes:
    hasher = hashlib.sha256()
    if path is not None and path.exists():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.digest()


def _sha256_file_hex(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return _sha256_file(path).hex()


def _product(shape: Iterable[int]) -> int:
    return math.prod(int(dim) for dim in shape)


def _fp16_tensor_metadata(
    shape: tuple[int, ...], source_bytes: int
) -> tuple[tuple[int, ...], tuple[int, ...], int, int, int, int, str]:
    return shape, shape, source_bytes, 0, 0, UOCR_TENSOR_F16, "F16"


def q8_qweight_bytes(shape: tuple[int, ...], group_size: int = UOCR_Q8_GROUP_SIZE_DEFAULT) -> int:
    if len(shape) != 2:
        raise ValueError(f"Q8_0 tensors must be rank-2, got shape {shape!r}")
    rows, cols = (int(shape[0]), int(shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Q8_0 tensors require positive dimensions, got shape {shape!r}")
    if group_size <= 0:
        raise ValueError(f"Q8_0 group size must be positive, got {group_size}")
    if cols % group_size != 0:
        raise ValueError(f"Q8_0 columns ({cols}) must be divisible by group size {group_size}")
    return rows * cols


def q8_qscale_bytes(shape: tuple[int, ...], group_size: int = UOCR_Q8_GROUP_SIZE_DEFAULT) -> int:
    _ = q8_qweight_bytes(shape, group_size)
    rows, cols = (int(shape[0]), int(shape[1]))
    return rows * (cols // group_size) * np.dtype("<f2").itemsize


def q8_total_bytes(shape: tuple[int, ...], group_size: int = UOCR_Q8_GROUP_SIZE_DEFAULT) -> int:
    return q8_qweight_bytes(shape, group_size) + q8_qscale_bytes(shape, group_size)


def q4_qweight_bytes(shape: tuple[int, ...], group_size: int = UOCR_Q4_GROUP_SIZE_DEFAULT) -> int:
    if len(shape) != 2:
        raise ValueError(f"Q4_0 tensors must be rank-2, got shape {shape!r}")
    rows, cols = (int(shape[0]), int(shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Q4_0 tensors require positive dimensions, got shape {shape!r}")
    if group_size <= 0:
        raise ValueError(f"Q4_0 group size must be positive, got {group_size}")
    if cols % group_size != 0:
        raise ValueError(f"Q4_0 columns ({cols}) must be divisible by group size {group_size}")
    if cols % 2 != 0:
        raise ValueError(f"Q4_0 columns ({cols}) must be even for nibble packing")
    return rows * (cols // 2)


def q4_qscale_bytes(shape: tuple[int, ...], group_size: int = UOCR_Q4_GROUP_SIZE_DEFAULT) -> int:
    _ = q4_qweight_bytes(shape, group_size)
    rows, cols = (int(shape[0]), int(shape[1]))
    return rows * (cols // group_size) * np.dtype("<f2").itemsize


def q4_total_bytes(shape: tuple[int, ...], group_size: int = UOCR_Q4_GROUP_SIZE_DEFAULT) -> int:
    return q4_qweight_bytes(shape, group_size) + q4_qscale_bytes(shape, group_size)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


def _usage_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.usage for t in tensors))


def _qtype_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.qtype for t in tensors))


def _qtype_reason_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.qtype_reason for t in tensors))


def _promotion_reason_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.promotion_reason for t in tensors))


def _family_histogram(tensors: Iterable[TensorPlan]) -> dict[str, int]:
    return dict(Counter(t.family for t in tensors))


def q8_summary(tensors: Iterable[TensorPlan]) -> dict[str, Any]:
    """Quantization summary; keeps its historical name and Q8 top-level keys and
    adds a ``by_qtype`` breakdown covering every quantized qtype (q8_0, q4_0)."""
    tensors_tuple = tuple(tensors)
    q8_tensors = tuple(t for t in tensors_tuple if t.qtype_id == UOCR_TENSOR_Q8_0)
    qweight_bytes = sum(t.output_bytes for t in q8_tensors)
    qscale_bytes = sum(t.scale_size for t in q8_tensors)
    fp16_equivalent_bytes = sum(t.fp16_equivalent_bytes for t in q8_tensors)
    q8_total = qweight_bytes + qscale_bytes
    by_qtype: dict[str, dict[str, Any]] = {}
    for qtype_id, qtype_name in ((UOCR_TENSOR_Q8_0, "q8_0"), (UOCR_TENSOR_Q4_0, "q4_0")):
        subset = tuple(t for t in tensors_tuple if t.qtype_id == qtype_id)
        if not subset:
            continue
        subset_qweight = sum(t.output_bytes for t in subset)
        subset_qscale = sum(t.scale_size for t in subset)
        subset_fp16 = sum(t.fp16_equivalent_bytes for t in subset)
        by_qtype[qtype_name] = {
            "tensor_count": len(subset),
            "qweight_bytes": subset_qweight,
            "qscale_bytes": subset_qscale,
            "total_bytes": subset_qweight + subset_qscale,
            "fp16_equivalent_bytes": subset_fp16,
            "estimated_savings_bytes": subset_fp16 - subset_qweight - subset_qscale,
            "by_family": dict(Counter(t.family for t in subset)),
        }
    return {
        "tensor_count": len(q8_tensors),
        "qweight_bytes": qweight_bytes,
        "qscale_bytes": qscale_bytes,
        "total_bytes": q8_total,
        "fp16_equivalent_bytes": fp16_equivalent_bytes,
        "estimated_savings_bytes": fp16_equivalent_bytes - q8_total,
        "by_family": dict(Counter(t.family for t in q8_tensors)),
        "by_qtype": by_qtype,
    }


def _layout_metadata_for_tensor(shape: tuple[int, ...], qtype_id: int) -> tuple[str, str, int, int, bool, str]:
    if qtype_id not in {UOCR_TENSOR_F16, UOCR_TENSOR_Q8_0, UOCR_TENSOR_Q4_0}:
        raise ValueError(f"unsupported tensor qtype id {qtype_id}")
    _ = shape
    return (
        "row-major",
        "row-major",
        UOCR_LAYOUT_TRANSFORM_IDENTITY,
        UOCR_TENSOR_FLAG_ROW_MAJOR,
        False,
        "source payload order is preserved; no transpose is applied",
    )


def layout_transform_summary(tensors: Iterable[TensorPlan]) -> dict[str, Any]:
    tensors_tuple = tuple(tensors)
    return {
        "contract": ROW_MAJOR_LAYOUT_CONTRACT,
        "row_major_count": sum(1 for tensor in tensors_tuple if tensor.layout_flags & UOCR_TENSOR_FLAG_ROW_MAJOR),
        "transposed_count": sum(1 for tensor in tensors_tuple if tensor.transposed),
        "by_transform": dict(Counter(tensor.layout_transform for tensor in tensors_tuple)),
    }


def _validate_layout_transform_contract(tensors: Iterable[TensorPlan]) -> None:
    for tensor in tensors:
        expected_source_layout, expected_runtime_layout, expected_transform_id, expected_flags, transposed, _reason = (
            _layout_metadata_for_tensor(tensor.shape, tensor.qtype_id)
        )
        if tensor.source_layout != expected_source_layout:
            raise ValueError(f"tensor {tensor.name} source layout mismatch: {tensor.source_layout}")
        if tensor.runtime_layout != expected_runtime_layout:
            raise ValueError(f"tensor {tensor.name} runtime layout mismatch: {tensor.runtime_layout}")
        if tensor.layout_transform_id != expected_transform_id:
            raise ValueError(f"tensor {tensor.name} layout transform id mismatch: {tensor.layout_transform_id}")
        if tensor.layout_transform != LAYOUT_TRANSFORM_NAMES[expected_transform_id]:
            raise ValueError(f"tensor {tensor.name} layout transform name mismatch: {tensor.layout_transform}")
        if tensor.layout_flags != expected_flags:
            raise ValueError(f"tensor {tensor.name} layout flags mismatch: {tensor.layout_flags:#x}")
        if tensor.transposed != transposed or (tensor.layout_flags & UOCR_TENSOR_FLAG_TRANSPOSED):
            raise ValueError(f"tensor {tensor.name} unexpectedly requests a transpose")
        if not (tensor.layout_flags & UOCR_TENSOR_FLAG_ROW_MAJOR):
            raise ValueError(f"tensor {tensor.name} is missing the row-major layout flag")
        if tensor.logical_shape != tensor.shape or tensor.physical_shape != tensor.shape:
            raise ValueError(f"tensor {tensor.name} must preserve source shape without layout conversion")


def _validate_moe_expert_interleaved_layout(tensors: Iterable[TensorPlan], *, require_complete: bool) -> None:
    """Validate the routed-expert payload order consumed by Metal.

    The integrated Metal decoder forms one no-copy slab per MoE layer starting at
    expert 0 gate.  It then addresses gate/up/down as offsets inside each expert
    stride, so converter layout drift would silently corrupt selected-expert
    projections.  Keep this check in the planner so dry-run and full conversion
    fail before a `.uocr` file is written.
    """

    by_id = {tensor.tensor_id: tensor for tensor in tensors}
    for layer in range(MOE_ROUTED_LAYER_START, MOE_ROUTED_LAYER_END_EXCLUSIVE):
        expected: list[TensorPlan] = []
        missing: list[int] = []
        for expert in range(MOE_ROUTED_EXPERT_COUNT):
            for projection in MOE_EXPERT_PROJECTION_ORDER:
                tensor_id = tensor_id_moe_expert(layer, expert, projection)
                tensor = by_id.get(tensor_id)
                if tensor is None:
                    missing.append(tensor_id)
                    continue
                expected.append(tensor)

        if missing:
            if require_complete:
                first = missing[0]
                raise ValueError(
                    f"MoE expert-major packing is incomplete for layer {layer}: "
                    f"missing {len(missing)} tensors, first tensor_id={first}"
                )
            continue
        if not expected:
            continue

        previous_end: int | None = None
        previous_scale_end: int | None = None
        for expert in range(MOE_ROUTED_EXPERT_COUNT):
            for projection_index, projection in enumerate(MOE_EXPERT_PROJECTION_ORDER):
                tensor = expected[expert * len(MOE_EXPERT_PROJECTION_ORDER) + projection_index]
                if tensor.family != TensorFamily.MOE_EXPERT.name:
                    raise ValueError(f"MoE expert-major tensor {tensor.name} has family {tensor.family}")
                if tensor.layer != layer or tensor.expert != expert or tensor.projection_id != int(projection):
                    raise ValueError(
                        "MoE expert-major metadata mismatch for "
                        f"{tensor.name}: layer={tensor.layer} expert={tensor.expert} projection={tensor.projection}"
                    )
                if tensor.usage_id != UOCR_TENSOR_USAGE_RUNTIME or tensor.output_bytes <= 0 or tensor.payload_offset <= 0:
                    raise ValueError(f"MoE expert-major tensor {tensor.name} is not a runtime payload tensor")
                if previous_end is not None and tensor.payload_offset != previous_end:
                    raise ValueError(
                        f"MoE expert-major layer {layer} is not contiguous before {tensor.name}: "
                        f"expected payload offset {previous_end}, got {tensor.payload_offset}"
                    )
                previous_end = tensor.payload_offset + tensor.output_bytes
                if tensor.qtype_id in (UOCR_TENSOR_Q8_0, UOCR_TENSOR_Q4_0):
                    if tensor.scale_size <= 0 or tensor.scale_offset <= 0:
                        raise ValueError(f"MoE expert-major quantized tensor {tensor.name} is missing scale payload")
                    if previous_scale_end is not None and tensor.scale_offset != previous_scale_end:
                        raise ValueError(
                            f"MoE expert-major layer {layer} scale slab is not contiguous before {tensor.name}: "
                            f"expected scale offset {previous_scale_end}, got {tensor.scale_offset}"
                        )
                    previous_scale_end = tensor.scale_offset + tensor.scale_size


def _layout_dry_run_file(
    tensors: tuple[TensorPlan, ...],
    *,
    provenance_json_size: int = 0,
) -> tuple[tuple[SectionPlan, ...], tuple[TensorPlan, ...], int, int]:
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
    provenance_size = UOCR_PROVENANCE_RECORD_SIZE + max(0, provenance_json_size)
    tensor_dir_offset = _align_up(provenance_offset + provenance_size, 8)
    tensor_dir_size = UOCR_TENSOR_DIRECTORY_HEADER_SIZE + len(tensors) * UOCR_TENSOR_ENTRY_SIZE
    tensor_data_offset = _align_up(tensor_dir_offset + tensor_dir_size, UOCR_TENSOR_DATA_ALIGNMENT)

    cursor = tensor_data_offset
    laid_out_by_id: dict[int, TensorPlan] = {}
    index = 0
    while index < len(tensors):
        tensor = tensors[index]
        if tensor.usage_id == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON or tensor.output_bytes == 0:
            laid_out_by_id[tensor.tensor_id] = replace(
                tensor,
                payload_offset=0,
                payload_alignment=0,
                scale_offset=0,
            )
            index += 1
            continue

        group = [tensor]
        quantized_expert_qtypes = {UOCR_TENSOR_Q8_0, UOCR_TENSOR_Q4_0}
        if tensor.qtype_id in quantized_expert_qtypes and tensor.family == TensorFamily.MOE_EXPERT.name:
            layer = tensor.layer
            lookahead = index + 1
            while (
                lookahead < len(tensors)
                and tensors[lookahead].qtype_id == tensor.qtype_id
                and tensors[lookahead].family == TensorFamily.MOE_EXPERT.name
                and tensors[lookahead].layer == layer
            ):
                group.append(tensors[lookahead])
                lookahead += 1
        else:
            lookahead = index + 1

        for item in group:
            cursor = _align_up(cursor, UOCR_TENSOR_PAYLOAD_ALIGNMENT)
            laid_out_by_id[item.tensor_id] = replace(
                item,
                payload_offset=cursor,
                payload_alignment=UOCR_TENSOR_PAYLOAD_ALIGNMENT,
            )
            cursor += item.output_bytes

        for item in group:
            if item.scale_size == 0:
                continue
            cursor = _align_up(cursor, UOCR_TENSOR_PAYLOAD_ALIGNMENT)
            current = laid_out_by_id[item.tensor_id]
            laid_out_by_id[item.tensor_id] = replace(current, scale_offset=cursor)
            cursor += item.scale_size

        index = lookahead

    laid_out = [laid_out_by_id[tensor.tensor_id] for tensor in tensors]
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
            provenance_size,
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


def _check_config_value(actual: Any, expected: Any, key: str, *, label: str = "config") -> None:
    if isinstance(expected, float):
        if abs(float(actual) - expected) > 1e-12:
            raise ValueError(f"{label} {key} mismatch: got {actual!r}, expected {expected!r}")
    elif actual != expected:
        raise ValueError(f"{label} {key} mismatch: got {actual!r}, expected {expected!r}")


def _check_required_values(data: Mapping[str, Any], expected_values: Mapping[str, Any], *, label: str) -> None:
    for key, expected in expected_values.items():
        if key not in data:
            raise ValueError(f"{label} missing required key {key!r}")
        _check_config_value(data[key], expected, key, label=label)


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        return str(content) if isinstance(content, str) else None
    return None


def _check_token_content(value: Any, expected: str, key: str, *, label: str) -> None:
    actual = _token_content(value)
    if actual != expected:
        raise ValueError(f"{label} {key} mismatch: got {actual!r}, expected {expected!r}")


def _validate_config_object(config: Mapping[str, Any], *, label: str = "config") -> dict[str, Any]:
    _check_required_values(config, EXPECTED_CONFIG, label=label)
    _check_required_values(config, EXPECTED_TOP_LEVEL_CONFIG, label=label)
    for key, expected in EXPECTED_CONFIG_DEFAULTS.items():
        if key in config:
            _check_config_value(config[key], expected, key, label=label)

    language_config = config.get("language_config")
    if isinstance(language_config, Mapping):
        _check_required_values(language_config, EXPECTED_CONFIG, label=f"{label}.language_config")
        language_label = f"{label}.language_config"
        _check_config_value(language_config.get("use_mla"), False, "use_mla", label=language_label)
        _check_config_value(language_config.get("sliding_window_size"), 128, "sliding_window_size", label=language_label)
        _check_config_value(language_config.get("bos_token_id"), 0, "bos_token_id", label=language_label)
        _check_config_value(language_config.get("eos_token_id"), 1, "eos_token_id", label=language_label)

    projector_config = config.get("projector_config")
    if not isinstance(projector_config, Mapping):
        raise ValueError(f"{label} missing projector_config object")
    _check_required_values(projector_config, EXPECTED_PROJECTOR_CONFIG, label=f"{label}.projector_config")

    vision_config = config.get("vision_config")
    vision_width = vision_config.get("width") if isinstance(vision_config, Mapping) else None
    if not isinstance(vision_width, Mapping):
        raise ValueError(f"{label} missing vision_config.width object")
    sam = vision_width.get("sam_vit_b")
    clip = vision_width.get("clip-l-14-224")
    if not isinstance(sam, Mapping) or not isinstance(clip, Mapping):
        raise ValueError(f"{label} missing SAM/CLIP vision width metadata")
    sam_label = f"{label}.vision.sam_vit_b"
    clip_label = f"{label}.vision.clip-l-14-224"
    _check_config_value(sam.get("layers"), 12, "layers", label=sam_label)
    _check_config_value(sam.get("heads"), 12, "heads", label=sam_label)
    _check_config_value(sam.get("width"), 768, "width", label=sam_label)
    _check_config_value(sam.get("global_attn_indexes"), [2, 5, 8, 11], "global_attn_indexes", label=sam_label)
    _check_config_value(sam.get("downsample_channels"), [512, 1024], "downsample_channels", label=sam_label)
    _check_config_value(clip.get("layers"), 24, "layers", label=clip_label)
    _check_config_value(clip.get("heads"), 16, "heads", label=clip_label)
    _check_config_value(clip.get("width"), 1024, "width", label=clip_label)
    _check_config_value(clip.get("patch_size"), 14, "patch_size", label=clip_label)

    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    if head_dim != 128:
        raise ValueError(f"derived attention head dim mismatch: got {head_dim}, expected 128")

    return {
        "present": True,
        "validated": True,
        "model_type": str(config.get("model_type")),
        "torch_dtype": str(config.get("torch_dtype")),
        "vocab_size": int(config["vocab_size"]),
        "hidden_size": int(config["hidden_size"]),
        "decoder_layers": int(config["num_hidden_layers"]),
        "attention_heads": int(config["num_attention_heads"]),
        "kv_heads": int(config["num_key_value_heads"]),
        "head_dim": head_dim,
        "sliding_window": int(config.get("sliding_window", 0)),
        "projector_in": int(projector_config["input_dim"]),
        "projector_out": int(projector_config["n_embed"]),
        "sam_layers": int(sam["layers"]),
        "clip_layers": int(clip["layers"]),
        "candidate_resolutions": list(config.get("candidate_resolutions", [])),
    }


def validate_config(hf_dir: Path, *, strict: bool = True) -> dict[str, Any]:
    config = _read_json_object_if_exists(hf_dir / "config.json", strict=strict, label="config")
    if config is None:
        return {"present": False, "validated": False}
    if not strict and not all(key in config for key in EXPECTED_CONFIG):
        return {"present": True, "validated": False}
    return _validate_config_object(config)


def validate_processor_config(hf_dir: Path, *, strict: bool = True) -> dict[str, Any]:
    processor = _read_json_object_if_exists(hf_dir / "processor_config.json", strict=strict, label="processor_config")
    if processor is None:
        return {"present": False, "validated": False}
    if not strict and not all(key in processor for key in EXPECTED_PROCESSOR_CONFIG):
        return {"present": True, "validated": False}
    _check_required_values(processor, EXPECTED_PROCESSOR_CONFIG, label="processor_config")
    return {
        "present": True,
        "validated": True,
        "processor_class": str(processor["processor_class"]),
        "candidate_resolutions": list(processor["candidate_resolutions"]),
        "image_mean": list(processor["image_mean"]),
        "image_std": list(processor["image_std"]),
        "image_token": str(processor["image_token"]),
        "patch_size": int(processor["patch_size"]),
        "downsample_ratio": int(processor["downsample_ratio"]),
        "normalize": bool(processor["normalize"]),
    }


def validate_tokenizer_metadata(hf_dir: Path, *, strict: bool = True) -> dict[str, Any]:
    tokenizer_path = hf_dir / "tokenizer.json"
    tokenizer_config_path = hf_dir / "tokenizer_config.json"
    special_tokens_path = hf_dir / "special_tokens_map.json"

    tokenizer_config = _read_json_object_if_exists(tokenizer_config_path, strict=strict, label="tokenizer_config")
    special_tokens = _read_json_object_if_exists(special_tokens_path, strict=strict, label="special_tokens_map")
    tokenizer_json = _read_json_object_if_exists(tokenizer_path, strict=strict, label="tokenizer")

    if tokenizer_config is None and special_tokens is None and tokenizer_json is None:
        return {"present": False, "validated": False}
    if not strict:
        required = (
            isinstance(tokenizer_config, Mapping)
            and isinstance(tokenizer_config.get("added_tokens_decoder"), Mapping)
            and isinstance(tokenizer_json, Mapping)
            and isinstance((tokenizer_json.get("model") if isinstance(tokenizer_json, Mapping) else None), Mapping)
        )
        if not required:
            return {"present": True, "validated": False}

    assert tokenizer_config is not None and tokenizer_json is not None
    _check_config_value(
        tokenizer_config.get("tokenizer_class"), EXPECTED_TOKENIZER_CLASS, "tokenizer_class", label="tokenizer_config"
    )
    _check_config_value(tokenizer_config.get("add_bos_token"), True, "add_bos_token", label="tokenizer_config")
    _check_config_value(tokenizer_config.get("add_eos_token"), False, "add_eos_token", label="tokenizer_config")
    added_decoder = tokenizer_config.get("added_tokens_decoder", {})
    for token_id, content in EXPECTED_SPECIAL_TOKENS.items():
        _check_token_content(
            added_decoder.get(str(token_id)), content, str(token_id), label="tokenizer_config.added_tokens_decoder"
        )
    _check_config_value(len(added_decoder), 830, "added_tokens_decoder", label="tokenizer_config")
    _check_token_content(
        tokenizer_config.get("bos_token"), EXPECTED_SPECIAL_TOKENS[0], "bos_token", label="tokenizer_config"
    )
    _check_token_content(
        tokenizer_config.get("eos_token"), EXPECTED_SPECIAL_TOKENS[1], "eos_token", label="tokenizer_config"
    )
    _check_token_content(
        tokenizer_config.get("pad_token"), EXPECTED_SPECIAL_TOKENS[2], "pad_token", label="tokenizer_config"
    )

    if special_tokens is not None:
        _check_token_content(
            special_tokens.get("bos_token"), EXPECTED_SPECIAL_TOKENS[0], "bos_token", label="special_tokens_map"
        )
        _check_token_content(
            special_tokens.get("eos_token"), EXPECTED_SPECIAL_TOKENS[1], "eos_token", label="special_tokens_map"
        )
        _check_token_content(
            special_tokens.get("pad_token"), EXPECTED_SPECIAL_TOKENS[2], "pad_token", label="special_tokens_map"
        )

    model = tokenizer_json.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("tokenizer.json missing model object")
    vocab = model.get("vocab")
    added_tokens = tokenizer_json.get("added_tokens")
    if not isinstance(vocab, Mapping) or not isinstance(added_tokens, list):
        raise ValueError("tokenizer.json missing vocab or added_tokens metadata")
    _check_config_value(model.get("type"), EXPECTED_TOKENIZER_MODEL_TYPE, "model.type", label="tokenizer")
    _check_config_value(len(vocab), 128_000, "model.vocab", label="tokenizer")
    _check_config_value(len(added_tokens), 830, "added_tokens", label="tokenizer")
    added_by_id = {
        int(token["id"]): token for token in added_tokens if isinstance(token, Mapping) and "id" in token
    }
    for token_id, content in EXPECTED_SPECIAL_TOKENS.items():
        _check_token_content(
            added_by_id.get(token_id), content, str(token_id), label="tokenizer.added_tokens"
        )

    return {
        "present": True,
        "validated": True,
        "tokenizer_class": str(tokenizer_config["tokenizer_class"]),
        "tokenizer_model_type": str(model["type"]),
        "vocab_size": 129_280,
        "bpe_vocab_size": len(vocab),
        "added_token_count": len(added_tokens),
        "special_token_ids": {
            "bos": 0,
            "eos": 1,
            "pad": 2,
            "image": 128_815,
        },
        "image_token": EXPECTED_SPECIAL_TOKENS[128_815],
    }


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


def _build_source_metadata(
    hf_dir: Path,
    *,
    entries: Mapping[str, Mapping[str, Any]],
    index: Mapping[str, Any] | None,
    index_path: Path,
    strict: bool,
) -> SourceMetadata:
    config = validate_config(hf_dir, strict=strict)
    processor = validate_processor_config(hf_dir, strict=strict)
    tokenizer = validate_tokenizer_metadata(hf_dir, strict=strict)

    total_payload_bytes = 0
    max_payload_end = 0
    dtype_counts: Counter[str] = Counter()
    for entry in entries.values():
        offsets = entry.get("data_offsets", [0, 0])
        start, end = int(offsets[0]), int(offsets[1])
        total_payload_bytes += end - start
        max_payload_end = max(max_payload_end, end)
        dtype_counts[str(entry.get("dtype", ""))] += 1

    weight_map = index.get("weight_map", {}) if isinstance(index, Mapping) else {}
    source_files = sorted({str(value) for value in weight_map.values()}) if isinstance(weight_map, Mapping) else []
    index_total = int(index.get("metadata", {}).get("total_size", -1)) if isinstance(index, Mapping) else -1
    source_dtype = next(iter(dtype_counts)) if len(dtype_counts) == 1 else "mixed"
    safetensors = {
        "tensor_count": len(entries),
        "total_payload_bytes": total_payload_bytes,
        "max_payload_end": max_payload_end,
        "index_total_size": index_total,
        "source_dtype": source_dtype,
        "dtype_counts": dict(dtype_counts),
        "file_count": len(source_files),
        "source_files": source_files,
        "index_present": index is not None,
    }
    if strict:
        _check_config_value(safetensors["tensor_count"], EXPECTED_TENSOR_COUNT, "tensor_count", label="safetensors")
        _check_config_value(
            safetensors["total_payload_bytes"], EXPECTED_TOTAL_BYTES, "total_payload_bytes", label="safetensors"
        )
        _check_config_value(
            safetensors["max_payload_end"], EXPECTED_TOTAL_BYTES, "max_payload_end", label="safetensors"
        )
        _check_config_value(safetensors["source_dtype"], EXPECTED_SOURCE_DTYPE, "source_dtype", label="safetensors")
        if index is not None:
            _check_config_value(index_total, EXPECTED_TOTAL_BYTES, "index_total_size", label="safetensors")
            _check_config_value(len(weight_map), EXPECTED_TENSOR_COUNT, "index.weight_map", label="safetensors")
            if len(source_files) != 1:
                raise ValueError(f"safetensors expected one source file, got {source_files!r}")

    hashes = {
        "config_sha256": _sha256_file_hex(hf_dir / "config.json"),
        "processor_config_sha256": _sha256_file_hex(hf_dir / "processor_config.json"),
        "tokenizer_sha256": _sha256_file_hex(hf_dir / "tokenizer.json"),
        "tokenizer_config_sha256": _sha256_file_hex(hf_dir / "tokenizer_config.json"),
        "special_tokens_map_sha256": _sha256_file_hex(hf_dir / "special_tokens_map.json"),
        "safetensors_index_sha256": _sha256_file_hex(index_path),
    }
    return SourceMetadata(
        config=config,
        processor=processor,
        tokenizer=tokenizer,
        safetensors=safetensors,
        hashes=hashes,
    )


def is_preserved_unused_in_normal_ocr(name: str) -> bool:
    """Return true for source tensors intentionally not consumed by v1 OCR inference."""
    return any(name.startswith(prefix) for prefix in PRESERVED_UNUSED_NORMAL_OCR_PREFIXES)


def _usage_for_name(name: str) -> tuple[str, int, str]:
    if is_preserved_unused_in_normal_ocr(name):
        return (
            "preserved-unused",
            2,
            "CLIP receives SAM feature maps as patch_embeds in the normal OCR path; raw-pixel patch_embedding is preserved only for provenance",
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


def _reason_name(reason_names: Mapping[int, str], reason_id: int) -> str:
    return reason_names.get(reason_id, "unknown")


def _is_mixed_q8_candidate(
    registry_entry: TensorRegistryEntry,
    shape: tuple[int, ...],
    usage_id: int,
    quant_cfg: QuantConfig,
) -> bool:
    if usage_id != UOCR_TENSOR_USAGE_RUNTIME or len(shape) != 2:
        return False
    if registry_entry.family == TensorFamily.MOE_ROUTER:
        return False
    return quant_cfg.is_supported(registry_entry.family, registry_entry.projection)


def _make_tensor_plan(
    name: str,
    entry: Mapping[str, Any],
    qprofile: str,
    registry_entry: TensorRegistryEntry,
    *,
    quant_group_size: int = UOCR_Q8_GROUP_SIZE_DEFAULT,
    quant_policy: str = QUANT_POLICY_EMBEDDINGS_DECODER,
    quant_cfg: QuantConfig | None = None,
) -> TensorPlan:
    if qprofile not in QPROFILE_IDS:
        raise ValueError(f"unknown qprofile {qprofile!r}")
    if quant_group_size != UOCR_Q8_GROUP_SIZE_DEFAULT:
        raise ValueError(f"only Q8_0 group size {UOCR_Q8_GROUP_SIZE_DEFAULT} is supported, got {quant_group_size}")
    if quant_policy != QUANT_POLICY_EMBEDDINGS_DECODER:
        raise ValueError(f"unsupported quantization policy {quant_policy!r}; expected {QUANT_POLICY_EMBEDDINGS_DECODER!r}")
    cfg = quant_cfg if quant_cfg is not None else load_default_quant_config()

    shape = tuple(int(dim) for dim in entry["shape"])
    offsets = (int(entry["data_offsets"][0]), int(entry["data_offsets"][1]))
    source_bytes = offsets[1] - offsets[0]
    usage, usage_id, base_reason = _usage_for_name(name)
    projection_name, projection_id = _projection_for_plan(registry_entry)

    qtype = "UOCR_TENSOR_F16"
    reason = base_reason
    logical_shape, physical_shape, output_bytes, block_size, row_size, qtype_id, output_dtype = _fp16_tensor_metadata(
        shape, source_bytes
    )
    qweight_bytes = output_bytes
    qscale_bytes = 0
    scale_size = 0
    qtype_reason_id = UOCR_QTYPE_REASON_FP16_BASELINE
    promotion_reason_id = UOCR_PROMOTION_NONE

    if qprofile in MIXED_QPROFILES and _is_mixed_q8_candidate(registry_entry, shape, usage_id, cfg):
        module_qtype = cfg.qtype_for(registry_entry.family, registry_entry.projection) or "q8_0"
        if module_qtype == "q4_0" and qprofile != "mixed-q4":
            # A v2 cfg with q4 modules can still drive a mixed-q8_0 conversion;
            # those modules simply fall back to Q8_0 planning.
            module_qtype = "q8_0"
        logical_shape = shape
        physical_shape = shape
        block_size = quant_group_size
        qtype_reason_id = UOCR_QTYPE_REASON_UNKNOWN
        if module_qtype == "q4_0":
            qweight_bytes = q4_qweight_bytes(shape, quant_group_size)
            qscale_bytes = q4_qscale_bytes(shape, quant_group_size)
            row_size = int(shape[1]) // 2
            qtype_id = UOCR_TENSOR_Q4_0
            qtype = "UOCR_TENSOR_Q4_0"
            output_dtype = "Q4_0"
        else:
            qweight_bytes = q8_qweight_bytes(shape, quant_group_size)
            qscale_bytes = q8_qscale_bytes(shape, quant_group_size)
            row_size = int(shape[1])
            qtype_id = UOCR_TENSOR_Q8_0
            qtype = "UOCR_TENSOR_Q8_0"
            output_dtype = "Q8_0"
        output_bytes = qweight_bytes
        scale_size = qscale_bytes
        reason = f"{qprofile} {quant_policy} cfg-gated rank-2 weight ({module_qtype})"

    (
        source_layout,
        runtime_layout,
        layout_transform_id,
        layout_flags,
        transposed,
        layout_reason,
    ) = _layout_metadata_for_tensor(shape, qtype_id)

    return TensorPlan(
        name=name,
        source_dtype=str(entry["dtype"]),
        shape=shape,
        logical_shape=logical_shape,
        physical_shape=physical_shape,
        source_offsets=offsets,
        source_bytes=source_bytes,
        output_dtype=output_dtype,
        qtype=qtype,
        qtype_id=qtype_id,
        output_bytes=output_bytes,
        payload_offset=0,
        payload_alignment=UOCR_TENSOR_PAYLOAD_ALIGNMENT,
        scale_offset=0,
        scale_size=scale_size,
        qweight_bytes=qweight_bytes,
        qscale_bytes=qscale_bytes,
        fp16_equivalent_bytes=source_bytes,
        block_size=block_size,
        row_size=row_size,
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
        qtype_reason=_reason_name(QTYPE_REASON_NAMES, qtype_reason_id),
        qtype_reason_id=qtype_reason_id,
        promotion_reason=_reason_name(PROMOTION_REASON_NAMES, promotion_reason_id),
        promotion_reason_id=promotion_reason_id,
        source_layout=source_layout,
        runtime_layout=runtime_layout,
        layout_transform=LAYOUT_TRANSFORM_NAMES[layout_transform_id],
        layout_transform_id=layout_transform_id,
        layout_flags=layout_flags,
        transposed=transposed,
        layout_reason=layout_reason,
    )


def _relayout_plan(plan: DryRunPlan, tensors: Iterable[TensorPlan]) -> DryRunPlan:
    ordered = tuple(sorted(tensors, key=lambda tensor: tensor.tensor_id))
    sections, laid_out, planned_file_size, metadata_bytes = _layout_dry_run_file(
        ordered,
        provenance_json_size=len(_provenance_json_payload(plan.qprofile, plan.source_metadata)),
    )
    _validate_layout_transform_contract(laid_out)
    return replace(
        plan,
        tensors=laid_out,
        total_source_bytes=sum(t.source_bytes for t in laid_out),
        total_output_bytes=sum(t.output_bytes + t.scale_size for t in laid_out),
        planned_file_size=planned_file_size,
        metadata_bytes=metadata_bytes,
        sections=sections,
        qtype_histogram=_qtype_histogram(laid_out),
        qtype_reason_histogram=_qtype_reason_histogram(laid_out),
        promotion_reason_histogram=_promotion_reason_histogram(laid_out),
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
    quant_group_size: int = UOCR_Q8_GROUP_SIZE_DEFAULT,
    quant_policy: str = QUANT_POLICY_EMBEDDINGS_DECODER,
    quant_cfg: QuantConfig | None = None,
) -> DryRunPlan:
    hf_dir = Path(hf_dir)
    if qprofile not in QPROFILE_IDS:
        raise ValueError(f"unknown qprofile {qprofile!r}")
    if quant_group_size != UOCR_Q8_GROUP_SIZE_DEFAULT:
        raise ValueError(f"only --quant-group-size {UOCR_Q8_GROUP_SIZE_DEFAULT} is supported, got {quant_group_size}")
    if quant_policy != QUANT_POLICY_EMBEDDINGS_DECODER:
        raise ValueError(f"unsupported --quant-policy {quant_policy!r}; expected {QUANT_POLICY_EMBEDDINGS_DECODER!r}")
    cfg = quant_cfg if quant_cfg is not None else load_default_quant_config()

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
        _make_tensor_plan(
            name,
            entries[name],
            qprofile,
            registry[name],
            quant_group_size=quant_group_size,
            quant_policy=quant_policy,
            quant_cfg=cfg,
        )
        for name in sorted(entries, key=lambda key: registry[key].tensor_id)
    )
    total_source_bytes = sum(t.source_bytes for t in tensors)
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

    source_metadata = _build_source_metadata(
        hf_dir,
        entries=entries,
        index=index,
        index_path=index_file,
        strict=strict,
    )

    provenance_json_size = len(_provenance_json_payload(qprofile, source_metadata))
    sections, tensors, planned_file_size, metadata_bytes = _layout_dry_run_file(
        tensors,
        provenance_json_size=provenance_json_size,
    )
    _validate_layout_transform_contract(tensors)
    _validate_moe_expert_interleaved_layout(tensors, require_complete=strict)
    total_output_bytes = sum(t.output_bytes + t.scale_size for t in tensors)

    return DryRunPlan(
        hf_dir=hf_dir,
        qprofile=qprofile,
        tensors=tensors,
        total_source_bytes=total_source_bytes,
        total_output_bytes=total_output_bytes,
        planned_file_size=planned_file_size,
        metadata_bytes=metadata_bytes,
        sections=sections,
        source_metadata=source_metadata,
        qtype_histogram=_qtype_histogram(tensors),
        qtype_reason_histogram=_qtype_reason_histogram(tensors),
        promotion_reason_histogram=_promotion_reason_histogram(tensors),
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


def _converter_version_string() -> str:
    return ".".join(str(part) for part in CONVERTER_VERSION)


def _provenance_json_payload(qprofile: str, source_metadata: SourceMetadata | None) -> bytes:
    if qprofile not in MIXED_QPROFILES:
        return b""
    source_hash = ""
    if source_metadata is not None:
        source_hash = str(source_metadata.hashes.get("safetensors_index_sha256", ""))
    quantization: dict[str, Any] = {
        "mode": qprofile,
        "group_size": UOCR_Q8_GROUP_SIZE_DEFAULT,
        "policy": QUANT_POLICY_EMBEDDINGS_DECODER,
        "lm_head_qtype": "q8_0",
        "router_qtype": "fp16",
        "converter_version": _converter_version_string(),
        "source_model_hash": source_hash,
    }
    if qprofile == "mixed-q4":
        quantization["expert_qtype"] = "q4_0"
    return json.dumps({"quantization": quantization}, separators=(",", ":"), sort_keys=True).encode("utf-8")


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
    json_payload = _provenance_json_payload(plan.qprofile, plan.source_metadata)
    json_offset = UOCR_PROVENANCE_RECORD_SIZE if json_payload else 0
    record = _PROVENANCE_RECORD_STRUCT.pack(
        UOCR_PROVENANCE_MAGIC,
        UOCR_FORMAT_VERSION,
        UOCR_PROVENANCE_RECORD_SIZE,
        UOCR_PROVENANCE_FLAG_JSON_PAYLOAD if json_payload else 0,
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
        json_offset,
        len(json_payload),
    )
    return record + json_payload


def _tensor_directory_bytes(plan: DryRunPlan) -> bytes:
    payload = bytearray()
    payload += _TENSOR_DIRECTORY_HEADER_STRUCT.pack(
        UOCR_TENSOR_DIR_MAGIC,
        UOCR_FORMAT_VERSION,
        UOCR_TENSOR_ENTRY_SIZE,
        len(plan.tensors),
    )
    for tensor in plan.tensors:
        if len(tensor.logical_shape) > 4 or len(tensor.physical_shape) > 4:
            raise ValueError(f"tensor {tensor.name} rank exceeds .uocr limit: {len(tensor.logical_shape)}")
        logical_shape = tuple(int(dim) for dim in tensor.logical_shape) + (0,) * (4 - len(tensor.logical_shape))
        physical_shape = tuple(int(dim) for dim in tensor.physical_shape) + (0,) * (4 - len(tensor.physical_shape))
        payload += _TENSOR_ENTRY_STRUCT.pack(
            tensor.tensor_id,
            tensor.family_id,
            tensor.layer,
            tensor.expert,
            tensor.projection_id,
            tensor.usage_id,
            tensor.qtype_id,
            tensor.layout_flags,
            len(tensor.logical_shape),
            *logical_shape,
            *physical_shape,
            tensor.payload_offset,
            tensor.output_bytes,
            tensor.block_size,
            tensor.row_size,
            tensor.scale_offset,
            tensor.scale_size,
            0,
            0,
            tensor.qtype_reason_id,
            tensor.promotion_reason_id,
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


def _bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    if len(raw) % 2 != 0:
        raise ValueError("BF16 byte buffer length must be even")
    bf16 = np.frombuffer(raw, dtype=np.dtype("<u2"))
    fp32_bits = bf16.astype(np.uint32) << np.uint32(16)
    return fp32_bits.view(np.dtype("<f4"))


def _tensor_conversion_stats(
    tensor: TensorPlan,
    value_stats: _StreamingValueStats,
    *,
    output_bytes_written: int,
) -> TensorConversionStats:
    return TensorConversionStats(
        name=tensor.name,
        tensor_id=tensor.tensor_id,
        source_dtype=tensor.source_dtype,
        output_dtype=tensor.output_dtype,
        qtype=tensor.qtype,
        qtype_id=tensor.qtype_id,
        shape=tensor.shape,
        logical_shape=tensor.logical_shape,
        physical_shape=tensor.physical_shape,
        source_bytes=tensor.source_bytes,
        output_bytes=tensor.output_bytes,
        output_bytes_written=output_bytes_written,
        payload_offset=tensor.payload_offset,
        scale_offset=tensor.scale_offset,
        scale_size=tensor.scale_size,
        value_count=value_stats.value_count,
        finite_count=value_stats.finite_count,
        source_min=value_stats.min_value,
        source_max=value_stats.max_value,
        source_nan_count=value_stats.nan_count,
        source_posinf_count=value_stats.posinf_count,
        source_neginf_count=value_stats.neginf_count,
        source_nonfinite_count=value_stats.nonfinite_count,
    )


def _conversion_stats_report(
    plan: DryRunPlan,
    out_path: Path,
    stats: Iterable[TensorConversionStats],
) -> dict[str, Any]:
    tensors = tuple(stats)
    return {
        "version": 1,
        "model_path": str(out_path),
        "hf_dir": str(plan.hf_dir),
        "qprofile": plan.qprofile,
        "tensor_count": len(tensors),
        "source_bytes": sum(tensor.source_bytes for tensor in tensors),
        "output_bytes": sum(tensor.output_bytes + tensor.scale_size for tensor in tensors),
        "output_bytes_written": sum(tensor.output_bytes_written for tensor in tensors),
        "value_count": sum(tensor.value_count for tensor in tensors),
        "finite_count": sum(tensor.finite_count for tensor in tensors),
        "source_nan_count": sum(tensor.source_nan_count for tensor in tensors),
        "source_posinf_count": sum(tensor.source_posinf_count for tensor in tensors),
        "source_neginf_count": sum(tensor.source_neginf_count for tensor in tensors),
        "source_nonfinite_count": sum(tensor.source_nonfinite_count for tensor in tensors),
        "tensors": [tensor.as_dict() for tensor in tensors],
    }


def _write_conversion_stats_report(
    plan: DryRunPlan,
    out_path: Path,
    stats: Iterable[TensorConversionStats],
    stats_path: str | Path,
) -> None:
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_conversion_stats_report(plan, out_path, stats), indent=2, sort_keys=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _stream_bf16_to_f16(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    source_bytes: int,
    stats: _StreamingValueStats | None = None,
    chunk_bytes: int = 32 * 1024 * 1024,
) -> int:
    if source_bytes % 2 != 0:
        raise ValueError(f"BF16 tensor byte count must be even, got {source_bytes}")
    remaining = source_bytes
    bytes_written = 0
    chunk_bytes = max(2, chunk_bytes - (chunk_bytes % 2))
    while remaining:
        raw = src.read(min(remaining, chunk_bytes))
        if not raw:
            raise EOFError("unexpected end of safetensors payload while streaming tensor")
        values = _bf16_bytes_to_f32(raw)
        if stats is not None:
            stats.update(values)
        payload = values.astype(np.dtype("<f2"), copy=False).tobytes()
        dst.write(payload)
        bytes_written += len(payload)
        remaining -= len(raw)
    return bytes_written


def _stream_bf16_to_q8_0(
    src: BinaryIO,
    dst: BinaryIO,
    tensor: TensorPlan,
    *,
    stats: _StreamingValueStats | None = None,
    chunk_source_bytes: int = 8 * 1024 * 1024,
) -> int:
    if tensor.qtype_id != UOCR_TENSOR_Q8_0:
        raise ValueError(f"tensor {tensor.name} is not Q8_0")
    if len(tensor.shape) != 2:
        raise ValueError(f"Q8_0 tensor {tensor.name} must be rank-2")
    rows, cols = tensor.shape
    group_size = tensor.block_size
    if group_size != UOCR_Q8_GROUP_SIZE_DEFAULT or cols % group_size != 0:
        raise ValueError(f"Q8_0 tensor {tensor.name} has unsupported group metadata")
    row_source_bytes = cols * np.dtype("<u2").itemsize
    if tensor.source_bytes != rows * row_source_bytes:
        raise ValueError(f"Q8_0 tensor {tensor.name} source byte count does not match shape")

    rows_per_chunk = max(1, chunk_source_bytes // row_source_bytes)
    rows_done = 0
    qweight_pos = tensor.payload_offset
    qscale_pos = tensor.scale_offset
    bytes_written = 0
    groups_per_row = cols // group_size

    while rows_done < rows:
        batch_rows = min(rows - rows_done, rows_per_chunk)
        raw_size = batch_rows * row_source_bytes
        raw = src.read(raw_size)
        if len(raw) != raw_size:
            raise EOFError(f"unexpected end of safetensors payload while quantizing {tensor.name}")
        values = _bf16_bytes_to_f32(raw).reshape(batch_rows, cols)
        if stats is not None:
            stats.update(values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Q8_0 tensor {tensor.name} contains non-finite values")

        grouped = values.reshape(batch_rows, groups_per_row, group_size)
        max_abs = np.max(np.abs(grouped), axis=2)
        scales = max_abs / np.float32(UOCR_Q8_MAX)
        scales = np.where(max_abs == 0.0, np.float32(1.0), scales).astype(np.float32, copy=False)
        quantized = np.rint(grouped / scales[:, :, None])
        quantized = np.clip(quantized, UOCR_Q8_MIN, UOCR_Q8_MAX).astype(np.int8, copy=False)

        qweight_payload = quantized.reshape(batch_rows, cols).tobytes()
        qscale_payload = scales.astype(np.dtype("<f2"), copy=False).tobytes()

        dst.seek(qweight_pos)
        dst.write(qweight_payload)
        qweight_pos += len(qweight_payload)
        bytes_written += len(qweight_payload)

        dst.seek(qscale_pos)
        dst.write(qscale_payload)
        qscale_pos += len(qscale_payload)
        bytes_written += len(qscale_payload)
        rows_done += batch_rows

    if qweight_pos != tensor.payload_offset + tensor.output_bytes:
        raise IOError(f"Q8_0 tensor {tensor.name} wrote an unexpected qweight byte count")
    if qscale_pos != tensor.scale_offset + tensor.scale_size:
        raise IOError(f"Q8_0 tensor {tensor.name} wrote an unexpected qscale byte count")
    return bytes_written


def _stream_bf16_to_q4_0(
    src: BinaryIO,
    dst: BinaryIO,
    tensor: TensorPlan,
    *,
    stats: _StreamingValueStats | None = None,
    chunk_source_bytes: int = 8 * 1024 * 1024,
) -> int:
    if tensor.qtype_id != UOCR_TENSOR_Q4_0:
        raise ValueError(f"tensor {tensor.name} is not Q4_0")
    if len(tensor.shape) != 2:
        raise ValueError(f"Q4_0 tensor {tensor.name} must be rank-2")
    rows, cols = tensor.shape
    group_size = tensor.block_size
    if group_size != UOCR_Q4_GROUP_SIZE_DEFAULT or cols % group_size != 0 or cols % 2 != 0:
        raise ValueError(f"Q4_0 tensor {tensor.name} has unsupported group metadata")
    row_source_bytes = cols * np.dtype("<u2").itemsize
    if tensor.source_bytes != rows * row_source_bytes:
        raise ValueError(f"Q4_0 tensor {tensor.name} source byte count does not match shape")

    rows_per_chunk = max(1, chunk_source_bytes // row_source_bytes)
    rows_done = 0
    qweight_pos = tensor.payload_offset
    qscale_pos = tensor.scale_offset
    bytes_written = 0

    while rows_done < rows:
        batch_rows = min(rows - rows_done, rows_per_chunk)
        raw_size = batch_rows * row_source_bytes
        raw = src.read(raw_size)
        if len(raw) != raw_size:
            raise EOFError(f"unexpected end of safetensors payload while quantizing {tensor.name}")
        values = _bf16_bytes_to_f32(raw).reshape(batch_rows, cols)
        if stats is not None:
            stats.update(values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Q4_0 tensor {tensor.name} contains non-finite values")

        packed, scales_f16, _dequant = _quantize_q4_0_rows(values, group_size)
        qweight_payload = packed.tobytes()
        qscale_payload = scales_f16.tobytes()

        dst.seek(qweight_pos)
        dst.write(qweight_payload)
        qweight_pos += len(qweight_payload)
        bytes_written += len(qweight_payload)

        dst.seek(qscale_pos)
        dst.write(qscale_payload)
        qscale_pos += len(qscale_payload)
        bytes_written += len(qscale_payload)
        rows_done += batch_rows

    if qweight_pos != tensor.payload_offset + tensor.output_bytes:
        raise IOError(f"Q4_0 tensor {tensor.name} wrote an unexpected qweight byte count")
    if qscale_pos != tensor.scale_offset + tensor.scale_size:
        raise IOError(f"Q4_0 tensor {tensor.name} wrote an unexpected qscale byte count")
    return bytes_written


def _read_uocr_tensor_payload_view(model_path: str | Path, tensor_id: int) -> _UocrTensorPayloadView:
    path = Path(model_path)
    with path.open("rb") as f:
        header_bytes = f.read(UOCR_FILE_HEADER_SIZE)
        if len(header_bytes) != UOCR_FILE_HEADER_SIZE:
            raise ValueError(f"{path} is too small to contain a .uocr header")
        header = _FILE_HEADER_STRUCT.unpack(header_bytes)
        magic, version, header_size, endian_marker, _alignment, qprofile_id, section_count, _reserved = header[:8]
        file_size, section_dir_offset = header[8], header[9]
        if magic != b"UOCR":
            raise ValueError(f"{path} is not a .uocr file")
        if version != UOCR_FORMAT_VERSION:
            raise ValueError(f"{path} has unsupported .uocr version {version}")
        if header_size != UOCR_FILE_HEADER_SIZE:
            raise ValueError(f"{path} has unexpected .uocr header size {header_size}")
        if endian_marker != UOCR_ENDIAN_MARKER:
            raise ValueError(f"{path} has unexpected endian marker 0x{endian_marker:x}")
        actual_size = path.stat().st_size
        if file_size != actual_size:
            raise ValueError(f"{path} file size mismatch: header={file_size}, actual={actual_size}")

        f.seek(section_dir_offset)
        tensor_dir_offset: int | None = None
        tensor_dir_size: int | None = None
        for _index in range(section_count):
            section_bytes = f.read(UOCR_SECTION_ENTRY_SIZE)
            if len(section_bytes) != UOCR_SECTION_ENTRY_SIZE:
                raise ValueError(f"{path} truncated while reading section directory")
            section_type, _flags, section_offset, section_size, _section_alignment = _SECTION_ENTRY_STRUCT.unpack(
                section_bytes
            )
            if section_type == UOCR_SECTION_TENSOR_DIRECTORY:
                tensor_dir_offset = int(section_offset)
                tensor_dir_size = int(section_size)
        if tensor_dir_offset is None or tensor_dir_size is None:
            raise ValueError(f"{path} does not contain a tensor-directory section")

        f.seek(tensor_dir_offset)
        dir_header_bytes = f.read(UOCR_TENSOR_DIRECTORY_HEADER_SIZE)
        if len(dir_header_bytes) != UOCR_TENSOR_DIRECTORY_HEADER_SIZE:
            raise ValueError(f"{path} truncated while reading tensor-directory header")
        dir_magic, dir_version, entry_size, tensor_count = _TENSOR_DIRECTORY_HEADER_STRUCT.unpack(dir_header_bytes)
        if dir_magic != UOCR_TENSOR_DIR_MAGIC:
            raise ValueError(f"{path} has invalid tensor-directory magic 0x{dir_magic:x}")
        if dir_version != UOCR_FORMAT_VERSION:
            raise ValueError(f"{path} has unsupported tensor-directory version {dir_version}")
        if entry_size != UOCR_TENSOR_ENTRY_SIZE:
            raise ValueError(f"{path} has unexpected tensor entry size {entry_size}")
        expected_dir_size = UOCR_TENSOR_DIRECTORY_HEADER_SIZE + tensor_count * UOCR_TENSOR_ENTRY_SIZE
        if tensor_dir_size != expected_dir_size:
            raise ValueError(f"{path} tensor-directory size/count mismatch")

        for _index in range(tensor_count):
            entry_bytes = f.read(UOCR_TENSOR_ENTRY_SIZE)
            if len(entry_bytes) != UOCR_TENSOR_ENTRY_SIZE:
                raise ValueError(f"{path} truncated while reading tensor-directory entries")
            entry = _TENSOR_ENTRY_STRUCT.unpack(entry_bytes)
            if int(entry[0]) != tensor_id:
                continue
            rank = int(entry[8])
            if rank < 0 or rank > 4:
                raise ValueError(f"tensor {tensor_id} in {path} has invalid rank {rank}")
            return _UocrTensorPayloadView(
                qprofile_id=int(qprofile_id),
                tensor_id=int(entry[0]),
                qtype_id=int(entry[6]),
                flags=int(entry[7]),
                rank=rank,
                logical_shape=tuple(int(value) for value in entry[9 : 9 + rank]),
                physical_shape=tuple(int(value) for value in entry[13 : 13 + rank]),
                payload_offset=int(entry[17]),
                payload_size=int(entry[18]),
                block_size=int(entry[19]),
                row_size=int(entry[20]),
                scale_offset=int(entry[21]),
                scale_size=int(entry[22]),
                qtype_reason_id=int(entry[25]),
                promotion_reason_id=int(entry[26]),
            )
        raise ValueError(f"tensor id {tensor_id} is not present in {path}")


def _iter_expected_converted_tensor_chunks(
    src: BinaryIO,
    tensor: TensorPlan,
    *,
    chunk_source_bytes: int = 8 * 1024 * 1024,
) -> Iterable[bytes]:
    if tensor.qtype_id != UOCR_TENSOR_F16:
        raise NotImplementedError(f"single-tensor compare for qtype {tensor.qtype} is not implemented")
    if tensor.source_bytes % 2 != 0:
        raise ValueError(f"BF16 tensor byte count must be even, got {tensor.source_bytes}")
    remaining = tensor.source_bytes
    chunk_bytes = max(2, chunk_source_bytes - (chunk_source_bytes % 2))
    while remaining:
        wanted = min(remaining, chunk_bytes)
        raw = src.read(wanted)
        if len(raw) != wanted:
            raise EOFError(f"unexpected end of safetensors payload while reading {tensor.name}")
        yield _bf16_bytes_to_f32(raw).astype(np.dtype("<f2"), copy=False).tobytes()
        remaining -= wanted


def _quantize_q4_0_rows(values: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize row-major fp32 ``[rows, cols]`` to Q4_0.

    Returns ``(packed_u8[rows, cols//2], scales_f16[rows, groups], dequant_f32)``.
    Scheme (docs/plan_q4.md §1.1): per 64-wide group, ``scale = signed_max / -8``
    rounded to fp16, levels clamped to ``[-8, 7]``, nibbles stored as ``q + 8``
    with the low nibble holding the even k element.
    """
    rows, cols = values.shape
    groups_per_row = cols // group_size
    grouped = values.reshape(rows, groups_per_row, group_size)
    abs_idx = np.argmax(np.abs(grouped), axis=2)
    signed_max = np.take_along_axis(grouped, abs_idx[:, :, None], axis=2)[:, :, 0]
    scales_f16 = (signed_max / np.float32(UOCR_Q4_MIN)).astype(np.dtype("<f2"))
    scales_f32 = scales_f16.astype(np.float32)
    scales_safe = np.where(scales_f32 == 0.0, np.float32(1.0), scales_f32)
    quantized = np.clip(np.rint(grouped / scales_safe[:, :, None]), UOCR_Q4_MIN, UOCR_Q4_MAX)
    quantized = quantized.astype(np.int8, copy=False)
    dequant = (quantized.astype(np.float32) * scales_f32[:, :, None]).reshape(rows, cols)
    biased = (quantized.reshape(rows, cols).astype(np.int16) + 8).astype(np.uint8)
    packed = (biased[:, 0::2] | (biased[:, 1::2] << 4)).astype(np.uint8, copy=False)
    return packed, scales_f16, dequant


def _expected_q4_0_chunk(values: np.ndarray, tensor: TensorPlan) -> tuple[bytes, bytes, float, float, int]:
    packed, scales_f16, dequant = _quantize_q4_0_rows(values, tensor.block_size)
    diff = dequant - values
    sq_error = float(np.sum(diff * diff, dtype=np.float64))
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    return (
        packed.tobytes(),
        scales_f16.tobytes(),
        max_abs_error,
        sq_error,
        int(diff.size),
    )


def _iter_expected_q4_0_tensor_chunks(
    src: BinaryIO,
    tensor: TensorPlan,
    *,
    chunk_source_bytes: int = 8 * 1024 * 1024,
) -> Iterable[tuple[bytes, bytes, float, float, int]]:
    if tensor.qtype_id != UOCR_TENSOR_Q4_0:
        raise ValueError(f"tensor {tensor.name} is not Q4_0")
    if len(tensor.shape) != 2:
        raise ValueError(f"Q4_0 tensor {tensor.name} must be rank-2")
    rows, cols = tensor.shape
    if tensor.block_size != UOCR_Q4_GROUP_SIZE_DEFAULT or cols % tensor.block_size != 0 or cols % 2 != 0:
        raise ValueError(f"Q4_0 tensor {tensor.name} has unsupported group metadata")
    row_source_bytes = cols * np.dtype("<u2").itemsize
    rows_per_chunk = max(1, chunk_source_bytes // row_source_bytes)
    rows_done = 0
    while rows_done < rows:
        batch_rows = min(rows - rows_done, rows_per_chunk)
        wanted = batch_rows * row_source_bytes
        raw = src.read(wanted)
        if len(raw) != wanted:
            raise EOFError(f"unexpected end of safetensors payload while reading {tensor.name}")
        values = _bf16_bytes_to_f32(raw).reshape(batch_rows, cols)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Q4_0 tensor {tensor.name} contains non-finite values")
        yield _expected_q4_0_chunk(values, tensor)
        rows_done += batch_rows


def _expected_q8_0_chunk(values: np.ndarray, tensor: TensorPlan) -> tuple[bytes, bytes, float, float, int]:
    rows, cols = values.shape
    group_size = tensor.block_size
    groups_per_row = cols // group_size
    grouped = values.reshape(rows, groups_per_row, group_size)
    max_abs = np.max(np.abs(grouped), axis=2)
    scales = max_abs / np.float32(UOCR_Q8_MAX)
    scales = np.where(max_abs == 0.0, np.float32(1.0), scales).astype(np.float32, copy=False)
    quantized = np.rint(grouped / scales[:, :, None])
    quantized = np.clip(quantized, UOCR_Q8_MIN, UOCR_Q8_MAX).astype(np.int8, copy=False)
    scale_f16 = scales.astype(np.dtype("<f2"), copy=False)
    dequant = quantized.astype(np.float32) * scale_f16.astype(np.float32)[:, :, None]
    diff = dequant.reshape(rows, cols) - values
    sq_error = float(np.sum(diff * diff, dtype=np.float64))
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    return (
        quantized.reshape(rows, cols).tobytes(),
        scale_f16.tobytes(),
        max_abs_error,
        sq_error,
        int(diff.size),
    )


def _iter_expected_q8_0_tensor_chunks(
    src: BinaryIO,
    tensor: TensorPlan,
    *,
    chunk_source_bytes: int = 8 * 1024 * 1024,
) -> Iterable[tuple[bytes, bytes, float, float, int]]:
    if tensor.qtype_id != UOCR_TENSOR_Q8_0:
        raise ValueError(f"tensor {tensor.name} is not Q8_0")
    if len(tensor.shape) != 2:
        raise ValueError(f"Q8_0 tensor {tensor.name} must be rank-2")
    rows, cols = tensor.shape
    if tensor.block_size != UOCR_Q8_GROUP_SIZE_DEFAULT or cols % tensor.block_size != 0:
        raise ValueError(f"Q8_0 tensor {tensor.name} has unsupported group metadata")
    row_source_bytes = cols * np.dtype("<u2").itemsize
    rows_per_chunk = max(1, chunk_source_bytes // row_source_bytes)
    rows_done = 0
    while rows_done < rows:
        batch_rows = min(rows - rows_done, rows_per_chunk)
        wanted = batch_rows * row_source_bytes
        raw = src.read(wanted)
        if len(raw) != wanted:
            raise EOFError(f"unexpected end of safetensors payload while reading {tensor.name}")
        values = _bf16_bytes_to_f32(raw).reshape(batch_rows, cols)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Q8_0 tensor {tensor.name} contains non-finite values")
        yield _expected_q8_0_chunk(values, tensor)
        rows_done += batch_rows


def _metadata_mismatches_for_compare(
    plan: DryRunPlan, tensor: TensorPlan, view: _UocrTensorPayloadView
) -> tuple[str, ...]:
    mismatches: list[str] = []
    expected_qprofile = QPROFILE_IDS[plan.qprofile]
    if view.qprofile_id != expected_qprofile:
        mismatches.append(f"qprofile_id actual={view.qprofile_id} expected={expected_qprofile}")
    if view.qtype_id != tensor.qtype_id:
        mismatches.append(f"qtype_id actual={view.qtype_id} expected={tensor.qtype_id}")
    if view.flags != tensor.layout_flags:
        mismatches.append(f"flags actual=0x{view.flags:x} expected=0x{tensor.layout_flags:x}")
    if view.rank != len(tensor.logical_shape):
        mismatches.append(f"rank actual={view.rank} expected={len(tensor.logical_shape)}")
    if view.logical_shape != tensor.logical_shape:
        mismatches.append(f"logical_shape actual={view.logical_shape} expected={tensor.logical_shape}")
    if view.physical_shape != tensor.physical_shape:
        mismatches.append(f"physical_shape actual={view.physical_shape} expected={tensor.physical_shape}")
    if view.payload_size != tensor.output_bytes:
        mismatches.append(f"payload_size actual={view.payload_size} expected={tensor.output_bytes}")
    if view.block_size != tensor.block_size:
        mismatches.append(f"block_size actual={view.block_size} expected={tensor.block_size}")
    if view.row_size != tensor.row_size:
        mismatches.append(f"row_size actual={view.row_size} expected={tensor.row_size}")
    if view.scale_offset != tensor.scale_offset:
        mismatches.append(f"scale_offset actual={view.scale_offset} expected={tensor.scale_offset}")
    if view.scale_size != tensor.scale_size:
        mismatches.append(f"scale_size actual={view.scale_size} expected={tensor.scale_size}")
    if view.qtype_reason_id != tensor.qtype_reason_id:
        mismatches.append(f"qtype_reason_id actual={view.qtype_reason_id} expected={tensor.qtype_reason_id}")
    if view.promotion_reason_id != tensor.promotion_reason_id:
        mismatches.append(f"promotion_reason_id actual={view.promotion_reason_id} expected={tensor.promotion_reason_id}")
    return tuple(mismatches)


def compare_single_tensor_conversion(
    plan: DryRunPlan,
    model_path: str | Path,
    *,
    safetensors_path: str | Path | None = None,
    chunk_source_bytes: int = 8 * 1024 * 1024,
) -> TensorCompareResult:
    """Compare one planned tensor's source conversion against bytes in a `.uocr` file.

    ``plan`` must already be filtered to exactly one tensor.  The function streams
    BF16 source bytes and the target model payload in chunks, so it is safe to use
    on very large tensors without allocating another full tensor-sized temporary.
    """

    if plan.tensor_count != 1:
        raise ValueError("single-tensor compare requires a plan filtered with --tensor")
    tensor = plan.tensors[0]
    if tensor.usage_id == UOCR_TENSOR_USAGE_OMITTED_WITH_REASON:
        raise ValueError(f"tensor {tensor.name} has no payload to compare")
    if tensor.qtype_id not in {UOCR_TENSOR_F16, UOCR_TENSOR_Q8_0, UOCR_TENSOR_Q4_0}:
        raise NotImplementedError(f"single-tensor compare for qtype {tensor.qtype} is not implemented")

    model = Path(model_path)
    view = _read_uocr_tensor_payload_view(model, tensor.tensor_id)
    metadata_mismatches = _metadata_mismatches_for_compare(plan, tensor, view)
    source, data_start = _resolve_safetensors_payload_path(plan, safetensors_path)

    expected_hash = hashlib.sha256()
    actual_hash = hashlib.sha256()
    compared_bytes = 0
    first_mismatch_offset: int | None = None
    expected_byte: int | None = None
    actual_byte: int | None = None
    dequant_max_abs_error: float | None = None
    dequant_sq_error = 0.0
    dequant_value_count = 0

    def compare_bytes(expected_chunk: bytes, actual_chunk: bytes, logical_offset: int) -> None:
        nonlocal first_mismatch_offset, expected_byte, actual_byte
        expected_hash.update(expected_chunk)
        actual_hash.update(actual_chunk)
        if first_mismatch_offset is not None or actual_chunk == expected_chunk:
            return
        limit = min(len(actual_chunk), len(expected_chunk))
        for index in range(limit):
            if actual_chunk[index] != expected_chunk[index]:
                first_mismatch_offset = logical_offset + index
                expected_byte = expected_chunk[index]
                actual_byte = actual_chunk[index]
                return
        first_mismatch_offset = logical_offset + limit
        expected_byte = expected_chunk[limit] if limit < len(expected_chunk) else None
        actual_byte = actual_chunk[limit] if limit < len(actual_chunk) else None

    with source.open("rb") as src, model.open("rb") as actual:
        src.seek(data_start + tensor.source_offsets[0])
        if tensor.qtype_id == UOCR_TENSOR_F16:
            actual.seek(view.payload_offset)
            for expected_chunk in _iter_expected_converted_tensor_chunks(
                src, tensor, chunk_source_bytes=chunk_source_bytes
            ):
                actual_chunk = actual.read(len(expected_chunk))
                compare_bytes(expected_chunk, actual_chunk, compared_bytes)
                compared_bytes += len(expected_chunk)
        else:
            qweight_pos = view.payload_offset
            qscale_pos = view.scale_offset
            scale_compared = 0
            expected_chunks = (
                _iter_expected_q4_0_tensor_chunks(src, tensor, chunk_source_bytes=chunk_source_bytes)
                if tensor.qtype_id == UOCR_TENSOR_Q4_0
                else _iter_expected_q8_0_tensor_chunks(src, tensor, chunk_source_bytes=chunk_source_bytes)
            )
            for expected_qweight, expected_qscale, max_abs_error, sq_error, value_count in expected_chunks:
                dequant_max_abs_error = max(max_abs_error, dequant_max_abs_error or 0.0)
                dequant_sq_error += sq_error
                dequant_value_count += value_count

                actual.seek(qweight_pos)
                actual_qweight = actual.read(len(expected_qweight))
                compare_bytes(expected_qweight, actual_qweight, compared_bytes)
                qweight_pos += len(expected_qweight)
                compared_bytes += len(expected_qweight)

                actual.seek(qscale_pos)
                actual_qscale = actual.read(len(expected_qscale))
                compare_bytes(expected_qscale, actual_qscale, tensor.output_bytes + scale_compared)
                qscale_pos += len(expected_qscale)
                scale_compared += len(expected_qscale)
            compared_bytes += scale_compared

    expected_bytes = tensor.output_bytes + tensor.scale_size
    actual_bytes = view.payload_size + view.scale_size
    if first_mismatch_offset is None and actual_bytes != expected_bytes:
        first_mismatch_offset = min(actual_bytes, expected_bytes)
    dequant_rmse = math.sqrt(dequant_sq_error / dequant_value_count) if dequant_value_count else None
    payload_matches = first_mismatch_offset is None and compared_bytes == expected_bytes and actual_bytes == expected_bytes
    return TensorCompareResult(
        name=tensor.name,
        tensor_id=tensor.tensor_id,
        qtype=tensor.qtype,
        qtype_id=tensor.qtype_id,
        model_path=str(model),
        source_path=str(source),
        expected_bytes=expected_bytes,
        actual_bytes=actual_bytes,
        compared_bytes=compared_bytes,
        expected_sha256=expected_hash.hexdigest(),
        actual_sha256=actual_hash.hexdigest(),
        payload_matches=payload_matches,
        metadata_mismatches=metadata_mismatches,
        first_mismatch_offset=first_mismatch_offset,
        expected_byte=expected_byte,
        actual_byte=actual_byte,
        dequant_max_abs_error=dequant_max_abs_error,
        dequant_rmse=dequant_rmse,
    )


def write_uocr_model(
    plan: DryRunPlan,
    out_path: str | Path,
    *,
    safetensors_path: str | Path | None = None,
    overwrite: bool = False,
    stats_path: str | Path | None = None,
) -> Path:
    """Stream a planned `.uocr` file to disk.

    The writer supports fp16 and mixed Q8_0 plans.  It converts each BF16
    safetensors range in bounded chunks and writes directly into the
    mmap-friendly layout produced by :func:`build_dry_run_plan`.  Q8 tensors are
    quantized row-major without materializing a full tensor-sized copy.
    """

    if plan.qprofile not in QPROFILE_IDS:
        raise ValueError(f"unknown qprofile {plan.qprofile!r}")
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
    conversion_stats: list[TensorConversionStats] | None = [] if stats_path is not None else None
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
                value_stats = _StreamingValueStats() if conversion_stats is not None else None
                if tensor.qtype_id == UOCR_TENSOR_F16:
                    dst.seek(tensor.payload_offset)
                    output_bytes_written = _stream_bf16_to_f16(
                        src, dst, source_bytes=tensor.source_bytes, stats=value_stats
                    )
                    expected_written = tensor.output_bytes
                elif tensor.qtype_id == UOCR_TENSOR_Q8_0:
                    output_bytes_written = _stream_bf16_to_q8_0(src, dst, tensor, stats=value_stats)
                    expected_written = tensor.output_bytes + tensor.scale_size
                elif tensor.qtype_id == UOCR_TENSOR_Q4_0:
                    output_bytes_written = _stream_bf16_to_q4_0(src, dst, tensor, stats=value_stats)
                    expected_written = tensor.output_bytes + tensor.scale_size
                else:
                    raise NotImplementedError(f"writing qtype {tensor.qtype} is not implemented")
                if output_bytes_written != expected_written:
                    raise IOError(
                        f"tensor {tensor.name} wrote {output_bytes_written} bytes, expected {expected_written}"
                    )
                if conversion_stats is not None and value_stats is not None:
                    conversion_stats.append(
                        _tensor_conversion_stats(tensor, value_stats, output_bytes_written=output_bytes_written)
                    )
            dst.truncate(plan.planned_file_size)
        os.replace(tmp, out)
        if stats_path is not None:
            _write_conversion_stats_report(plan, out, conversion_stats or (), stats_path)
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
    if plan.total_source_bytes:
        savings = plan.total_source_bytes - plan.total_output_bytes
        ratio = plan.total_output_bytes / plan.total_source_bytes
        print(f"estimated weight savings: {savings} ({ratio:.3f}x source bytes)")
    print(f"planned file size: {plan.planned_file_size}")
    metadata = plan.source_metadata
    print(
        "source metadata: "
        f"config_validated={metadata.config.get('validated', False)} "
        f"processor_validated={metadata.processor.get('validated', False)} "
        f"tokenizer_validated={metadata.tokenizer.get('validated', False)} "
        f"safetensors_tensors={metadata.safetensors.get('tensor_count', 0)}"
    )
    tensor_data = next((section for section in plan.sections if section.section_type == UOCR_SECTION_TENSOR_DATA), None)
    if tensor_data is not None:
        print(f"tensor data: offset={tensor_data.offset} size={tensor_data.size} alignment={tensor_data.alignment}")
    q8 = q8_summary(plan.tensors)
    if q8["tensor_count"]:
        print(
            "q8: "
            f"tensors={q8['tensor_count']} qweight_bytes={q8['qweight_bytes']} "
            f"qscale_bytes={q8['qscale_bytes']} fp16_equivalent_bytes={q8['fp16_equivalent_bytes']} "
            f"estimated_savings_bytes={q8['estimated_savings_bytes']}"
        )
    print(f"qtypes: {dict(plan.qtype_histogram)}")
    print(f"qtype reasons: {dict(plan.qtype_reason_histogram)}")
    print(f"promotion reasons: {dict(plan.promotion_reason_histogram)}")
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
    parser = argparse.ArgumentParser(description="Plan or write an Unlimited-OCR .uocr file")
    parser.add_argument("--hf-dir", type=Path, default=project_root() / "data/context")
    parser.add_argument("--header", type=Path, default=None, help="optional safetensors header/cache path")
    parser.add_argument("--index", type=Path, default=None, help="optional safetensors index path")
    parser.add_argument(
        "--safetensors",
        type=Path,
        default=None,
        help="explicit source .safetensors payload for --out or --compare-uocr",
    )
    parser.add_argument("--qprofile", choices=["fp16", "mixed-q8_0", "mixed-q4"], default="fp16")
    parser.add_argument("--quant-group-size", type=int, default=UOCR_Q8_GROUP_SIZE_DEFAULT)
    parser.add_argument("--quant-policy", choices=[QUANT_POLICY_EMBEDDINGS_DECODER], default=QUANT_POLICY_EMBEDDINGS_DECODER)
    parser.add_argument(
        "--quant-cfg",
        type=Path,
        default=None,
        help="path to quant-cfg.yaml; defaults to configs/quant-cfg.yaml (runtime-supported Q8 modules)",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only; does not require full weights")
    parser.add_argument("--out", type=Path, default=None, help="write a .uocr file to this path")
    parser.add_argument(
        "--compare-uocr",
        type=Path,
        default=None,
        help="compare the selected --tensor against an existing .uocr payload without rewriting the full model",
    )
    parser.add_argument("--tensor", default=None, help="optional single tensor source name or stable tensor id")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    parser.add_argument(
        "--conversion-stats",
        type=Path,
        default=None,
        help="write per-tensor conversion statistics JSON while writing --out",
    )
    parser.add_argument("--dump-plan", nargs="?", const="-", default=None, help="write full JSON plan to path or stdout")
    parser.add_argument("--dump-quant-summary", nargs="?", const="-", default=None, help="write Q8 summary JSON to path or stdout")
    parser.add_argument(
        "--relaxed-validation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.out is None and args.compare_uocr is None and not args.dry_run:
        parser.error("either --dry-run, --out, or --compare-uocr is required")
    if args.compare_uocr is not None and args.tensor is None:
        parser.error("--compare-uocr requires --tensor")
    if args.conversion_stats is not None and args.out is None:
        parser.error("--conversion-stats requires --out")
    plan = build_dry_run_plan(
        args.hf_dir,
        qprofile=args.qprofile,
        header_path=args.header,
        index_path=args.index,
        strict=not args.relaxed_validation,
        quant_group_size=args.quant_group_size,
        quant_policy=args.quant_policy,
        quant_cfg=load_quant_config(args.quant_cfg) if args.quant_cfg is not None else None,
    )
    plan = filter_plan_tensors(plan, args.tensor)
    _print_summary(plan, dry_run=args.dry_run or args.out is None)
    if args.dump_plan is not None:
        _write_dump(plan, args.dump_plan)
    if args.dump_quant_summary is not None:
        payload = json.dumps(q8_summary(plan.tensors), indent=2, sort_keys=True)
        if args.dump_quant_summary == "-":
            print(payload)
        else:
            out = Path(args.dump_quant_summary)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload + "\n", encoding="utf-8")
    if args.out is not None:
        written = write_uocr_model(
            plan,
            args.out,
            safetensors_path=args.safetensors,
            overwrite=args.overwrite,
            stats_path=args.conversion_stats,
        )
        print(f"wrote: {written}")
        if args.conversion_stats is not None:
            print(f"conversion stats: {args.conversion_stats}")
    if args.compare_uocr is not None:
        comparison = compare_single_tensor_conversion(plan, args.compare_uocr, safetensors_path=args.safetensors)
        print(
            "compare: "
            f"tensor={comparison.name} id={comparison.tensor_id} qtype={comparison.qtype} "
            f"metadata={'ok' if comparison.metadata_matches else 'mismatch'} "
            f"payload={'ok' if comparison.payload_matches else 'mismatch'} "
            f"bytes={comparison.compared_bytes}"
        )
        print(f"compare expected_sha256={comparison.expected_sha256}")
        print(f"compare actual_sha256={comparison.actual_sha256}")
        if comparison.dequant_max_abs_error is not None:
            print(
                "compare dequant: "
                f"max_abs_error={comparison.dequant_max_abs_error:.8g} rmse={comparison.dequant_rmse:.8g}"
            )
        if not comparison.metadata_matches:
            print("compare metadata mismatches:")
            for mismatch in comparison.metadata_mismatches:
                print(f"  - {mismatch}")
        if comparison.first_mismatch_offset is not None:
            print(
                "compare first payload mismatch: "
                f"offset={comparison.first_mismatch_offset} "
                f"expected={comparison.expected_byte} actual={comparison.actual_byte}"
            )
        return 0 if comparison.matches else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
