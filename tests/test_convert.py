from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from unlimitedocr_c.convert import (
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_BYTES,
    MOE_EXPERT_PACKING_CONTRACT,
    MOE_EXPERT_PACKING_LAYOUT,
    MOE_ROUTED_EXPERT_COUNT,
    MOE_ROUTED_LAYER_END_EXCLUSIVE,
    MOE_ROUTED_LAYER_START,
    PADDED_Q4_K_KERNEL_CONTRACT,
    PRESERVED_UNUSED_NORMAL_OCR_PREFIXES,
    Q4_HAZARD_DENSE_LAYER0_DOWN,
    Q4_HAZARD_DENSE_LAYER0_DOWN_COUNT,
    Q4_HAZARD_DENSE_LAYER0_DOWN_NAME,
    Q4_HAZARD_ROUTED_EXPERT_DOWN,
    Q4_HAZARD_ROUTED_EXPERT_DOWN_COUNT,
    Q4_HAZARD_ROUTED_EXPERT_DOWN_NAME,
    Q4_UNALIGNED_HAZARD_CONTRACT,
    UOCR_PROMOTION_NONE,
    UOCR_PROMOTION_SENSITIVE,
    UOCR_PROMOTION_UNALIGNED,
    UOCR_FILE_HEADER_SIZE,
    UOCR_Q4_K_BLOCK_SIZE,
    UOCR_Q4_K_TYPE_SIZE,
    UOCR_Q8_0_BLOCK_SIZE,
    UOCR_Q8_0_TYPE_SIZE,
    UOCR_QPROFILE_DYN_Q8,
    UOCR_QTYPE_REASON_FP16_BASELINE,
    UOCR_QTYPE_REASON_POLICY,
    UOCR_QTYPE_REASON_SENSITIVE,
    UOCR_QTYPE_REASON_UNALIGNED,
    UOCR_SECTION_TENSOR_DATA,
    UOCR_TENSOR_DATA_ALIGNMENT,
    UOCR_TENSOR_PAYLOAD_ALIGNMENT,
    UOCR_TENSOR_Q4_K,
    UOCR_TENSOR_PADDED_Q4_K,
    UOCR_TENSOR_Q8_0,
    _TENSOR_DIRECTORY_HEADER_STRUCT,
    _TENSOR_ENTRY_STRUCT,
    _tensor_directory_bytes,
    build_dry_run_plan,
    describe_padded_q4_k_design,
    filter_plan_tensors,
    is_preserved_unused_in_normal_ocr,
    write_uocr_model,
)
from unlimitedocr_c.tensor_registry import (
    TENSOR_ID_LM_HEAD,
    TENSOR_ID_TOK_EMBED,
    TensorFamily,
    TensorProjection,
    tensor_id_layer_attn,
    tensor_id_moe_expert,
)
from unlimitedocr_c.frontend import project_root


def _bf16_payload(values: np.ndarray) -> bytes:
    fp32 = np.asarray(values, dtype=np.dtype("<f4"))
    return ((fp32.view(np.dtype("<u4")) >> np.uint32(16)).astype(np.dtype("<u2"))).tobytes()


def _f32_from_bf16_payload(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload, dtype=np.dtype("<u2"))
    return (words.astype(np.uint32) << np.uint32(16)).view(np.dtype("<f4"))


def _f16_from_bf16_payload(payload: bytes) -> bytes:
    return _f32_from_bf16_payload(payload).astype(np.dtype("<f2")).tobytes()


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def _q8_0_from_bf16_payload(payload: bytes, shape: tuple[int, ...], physical_shape: tuple[int, int]) -> bytes:
    rows, physical_inner = physical_shape
    logical_inner = int(np.prod(shape[1:]))
    values = _f32_from_bf16_payload(payload).reshape(rows, logical_inner)
    if physical_inner != logical_inner:
        padded = np.zeros((rows, physical_inner), dtype=np.float32)
        padded[:, :logical_inner] = values
        values = padded
    blocks = values.reshape(-1, UOCR_Q8_0_BLOCK_SIZE)
    amax = np.max(np.abs(blocks), axis=1)
    scales = amax / np.float32(127.0)
    inv_scales = np.divide(
        np.float32(1.0), scales, out=np.zeros_like(scales, dtype=np.float32), where=scales != 0.0
    )
    quantized = np.clip(_round_away_from_zero(blocks * inv_scales[:, None]), -128, 127).astype(np.int8)
    packed = np.empty((blocks.shape[0], UOCR_Q8_0_TYPE_SIZE), dtype=np.uint8)
    packed[:, :2] = scales.astype(np.dtype("<f2")).view(np.uint8).reshape(-1, 2)
    packed[:, 2:] = quantized.view(np.uint8)
    return packed.tobytes()


def _read_length_prefixed_safetensors_header(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path} is too small to contain a safetensors header length")
    (header_len,) = struct.unpack_from("<Q", data, 0)
    start = 8
    end = start + header_len
    if end > len(data):
        raise ValueError(f"{path} declares {header_len} header bytes but only has {len(data) - start}")
    header = json.loads(data[start:end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path} header is not a JSON object")
    return header


def _write_tiny_safetensors(hf_dir) -> tuple[dict[str, bytes], dict[str, tuple[int, ...]]]:
    tensors = {
        "model.sam_model.tiny_a.weight": (np.array([[1.0, -2.0], [3.5, 0.25]], dtype=np.float32), (2, 2)),
        "model.vision_model.embeddings.patch_embedding.weight": (np.array([4.0, -8.0, 0.5], dtype=np.float32), (3,)),
    }
    offsets: dict[str, tuple[int, int]] = {}
    payloads: dict[str, bytes] = {}
    cursor = 0
    for name, (values, _shape) in tensors.items():
        payload = _bf16_payload(values)
        payloads[name] = payload
        offsets[name] = (cursor, cursor + len(payload))
        cursor += len(payload)

    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, (_values, shape) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": list(offsets[name])}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with (hf_dir / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name in tensors:
            f.write(payloads[name])

    index = {
        "metadata": {"total_size": cursor},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return payloads, {name: shape for name, (_values, shape) in tensors.items()}


def test_cached_safetensors_header_current_checkpoint_facts() -> None:
    context_dir = project_root() / "data/context"
    header = _read_length_prefixed_safetensors_header(context_dir / "model.safetensors.header")
    index = json.loads((context_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))

    entries = {name: meta for name, meta in header.items() if name != "__metadata__"}
    assert len(entries) == EXPECTED_TENSOR_COUNT
    assert len(index["weight_map"]) == EXPECTED_TENSOR_COUNT
    assert set(index["weight_map"]) == set(entries)
    assert set(index["weight_map"].values()) == {"model-00001-of-000001.safetensors"}
    assert index["metadata"]["total_size"] == EXPECTED_TOTAL_BYTES

    total_payload_bytes = 0
    max_payload_end = 0
    for name, meta in entries.items():
        assert isinstance(meta, dict), name
        assert meta.get("dtype") == "BF16", name
        offsets = meta.get("data_offsets")
        assert isinstance(offsets, list) and len(offsets) == 2, name
        start, end = offsets
        assert isinstance(start, int) and isinstance(end, int), name
        assert 0 <= start < end <= EXPECTED_TOTAL_BYTES, name
        total_payload_bytes += end - start
        max_payload_end = max(max_payload_end, end)

    assert total_payload_bytes == EXPECTED_TOTAL_BYTES
    assert max_payload_end == EXPECTED_TOTAL_BYTES


def test_fp16_converter_dry_run_against_cached_header() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="fp16")

    assert plan.tensor_count == EXPECTED_TENSOR_COUNT
    assert plan.total_source_bytes == EXPECTED_TOTAL_BYTES
    assert plan.total_output_bytes == EXPECTED_TOTAL_BYTES
    assert plan.qtype_histogram == {"UOCR_TENSOR_F16": EXPECTED_TENSOR_COUNT}
    assert plan.qtype_reason_histogram == {"fp16-baseline": EXPECTED_TENSOR_COUNT}
    assert plan.promotion_reason_histogram == {"none": EXPECTED_TENSOR_COUNT}
    assert plan.usage_histogram == {"runtime": EXPECTED_TENSOR_COUNT - 1, "preserved-unused": 1}

    preserved_unused = [tensor for tensor in plan.tensors if tensor.usage == "preserved-unused"]
    assert [tensor.name for tensor in preserved_unused] == ["model.vision_model.embeddings.patch_embedding.weight"]
    assert all(is_preserved_unused_in_normal_ocr(tensor.name) for tensor in preserved_unused)
    assert PRESERVED_UNUSED_NORMAL_OCR_PREFIXES == ("model.vision_model.embeddings.patch_embedding.",)

    tensor_data = next(section for section in plan.sections if section.section_type == UOCR_SECTION_TENSOR_DATA)
    assert tensor_data.offset % UOCR_TENSOR_DATA_ALIGNMENT == 0
    assert tensor_data.alignment == UOCR_TENSOR_DATA_ALIGNMENT
    assert tensor_data.size == EXPECTED_TOTAL_BYTES
    assert plan.planned_file_size == tensor_data.offset + tensor_data.size
    assert plan.metadata_bytes == tensor_data.offset

    tensor_ids = [tensor.tensor_id for tensor in plan.tensors]
    assert tensor_ids == sorted(tensor_ids)
    assert len(tensor_ids) == len(set(tensor_ids))
    assert all(tensor.payload_offset % UOCR_TENSOR_PAYLOAD_ALIGNMENT == 0 for tensor in plan.tensors)
    assert plan.tensors[0].payload_offset == tensor_data.offset
    assert plan.tensors[-1].payload_offset + plan.tensors[-1].output_bytes == plan.planned_file_size

    tok_embed = plan.tensor_by_name("model.embed_tokens.weight")
    assert tok_embed.tensor_id == TENSOR_ID_TOK_EMBED
    assert tok_embed.family == "TOK_EMBED"

    lm_head = plan.tensor_by_name("lm_head.weight")
    assert lm_head.tensor_id == TENSOR_ID_LM_HEAD
    assert lm_head.shape == (129_280, 1280)
    assert lm_head.source_dtype == "BF16"
    assert lm_head.output_dtype == "F16"
    assert lm_head.qtype_reason == "fp16-baseline"
    assert lm_head.qtype_reason_id == UOCR_QTYPE_REASON_FP16_BASELINE
    assert lm_head.promotion_reason == "none"
    assert lm_head.promotion_reason_id == UOCR_PROMOTION_NONE

    attn_q = plan.tensor_by_name("model.layers.3.self_attn.q_proj.weight")
    assert attn_q.tensor_id == tensor_id_layer_attn(3, TensorProjection.Q)
    assert attn_q.family_id == TensorFamily.LAYER_ATTN
    assert attn_q.layer == 3
    assert attn_q.projection == "Q"

    expert_down = plan.tensor_by_name("model.layers.1.mlp.experts.7.down_proj.weight")
    assert expert_down.tensor_id == tensor_id_moe_expert(1, 7, TensorProjection.DOWN)
    assert expert_down.family == "MOE_EXPERT"
    assert expert_down.expert == 7

    router = plan.tensor_by_name("model.layers.1.mlp.gate.weight")
    assert router.family == "MOE_ROUTER"
    assert router.qtype == "UOCR_TENSOR_F16"
    assert "router" in router.reason.lower()

    clip_patch = plan.tensor_by_name("model.vision_model.embeddings.patch_embedding.weight")
    assert clip_patch.family == "VISION_CLIP"
    assert clip_patch.usage == "preserved-unused"
    assert "patch_embeds" in clip_patch.reason
    assert "provenance" in clip_patch.reason


def test_fp16_converter_packs_routed_experts_interleaved_expert_major() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="fp16")
    summary = plan.summary_dict()["moe_expert_packing"]
    assert summary["layout"] == MOE_EXPERT_PACKING_LAYOUT
    assert summary["contract"] == MOE_EXPERT_PACKING_CONTRACT
    assert summary["layer_start"] == MOE_ROUTED_LAYER_START
    assert summary["layer_end_exclusive"] == MOE_ROUTED_LAYER_END_EXCLUSIVE
    assert summary["expert_count"] == MOE_ROUTED_EXPERT_COUNT
    assert summary["projection_order"] == ["GATE", "UP", "DOWN"]

    projections = [
        ("gate_proj", TensorProjection.GATE, "GATE"),
        ("up_proj", TensorProjection.UP, "UP"),
        ("down_proj", TensorProjection.DOWN, "DOWN"),
    ]
    for layer in range(MOE_ROUTED_LAYER_START, MOE_ROUTED_LAYER_END_EXCLUSIVE):
        ordered = []
        for expert in range(MOE_ROUTED_EXPERT_COUNT):
            per_expert = []
            for suffix, projection, projection_name in projections:
                tensor = plan.tensor_by_name(f"model.layers.{layer}.mlp.experts.{expert}.{suffix}.weight")
                assert tensor.tensor_id == tensor_id_moe_expert(layer, expert, projection)
                assert tensor.family == "MOE_EXPERT"
                assert tensor.layer == layer
                assert tensor.expert == expert
                assert tensor.projection == projection_name
                per_expert.append(tensor)
                ordered.append(tensor)

            gate, up, down = per_expert
            assert up.payload_offset == gate.payload_offset + gate.output_bytes
            assert down.payload_offset == up.payload_offset + up.output_bytes

        assert [tensor.tensor_id for tensor in ordered] == sorted(tensor.tensor_id for tensor in ordered)
        previous_end = ordered[0].payload_offset
        for tensor in ordered:
            assert tensor.payload_offset == previous_end
            previous_end = tensor.payload_offset + tensor.output_bytes


def test_dyn_q8_converter_dry_run_records_quant_metadata() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="dyn-q8")

    assert plan.tensor_count == EXPECTED_TENSOR_COUNT
    assert plan.total_source_bytes == EXPECTED_TOTAL_BYTES
    assert plan.total_output_bytes < EXPECTED_TOTAL_BYTES
    assert plan.summary_dict()["estimated_savings_bytes"] == EXPECTED_TOTAL_BYTES - plan.total_output_bytes
    assert 0 < plan.summary_dict()["compression_ratio"] < 1
    assert plan.qtype_histogram["UOCR_TENSOR_Q8_0"] > 0
    assert plan.qtype_histogram["UOCR_TENSOR_F16"] > 0
    width_summary = plan.summary_dict()["quantized_input_widths"]
    assert width_summary["quantized_tensor_count"] == plan.qtype_histogram["UOCR_TENSOR_Q8_0"]
    assert width_summary["padded_tensor_count"] > 0
    assert width_summary["max_padding_width"] > 0
    assert "logical_input_width" in width_summary["contract"]
    assert plan.qtype_reason_histogram["policy"] == plan.qtype_histogram["UOCR_TENSOR_Q8_0"]
    assert plan.qtype_reason_histogram["sensitive"] == plan.qtype_histogram["UOCR_TENSOR_F16"]
    assert plan.promotion_reason_histogram["none"] == plan.qtype_histogram["UOCR_TENSOR_Q8_0"]
    assert plan.promotion_reason_histogram["sensitive"] == plan.qtype_histogram["UOCR_TENSOR_F16"]

    lm_head = plan.tensor_by_name("lm_head.weight")
    assert lm_head.qtype == "UOCR_TENSOR_Q8_0"
    assert lm_head.qtype_id == UOCR_TENSOR_Q8_0
    assert lm_head.shape == (129_280, 1280)
    assert lm_head.logical_shape == (129_280, 1280)
    assert lm_head.physical_shape == (129_280, 1280)
    assert lm_head.logical_input_width == 1280
    assert lm_head.physical_input_width == 1280
    assert lm_head.input_padding_width == 0
    assert lm_head.block_size == UOCR_Q8_0_BLOCK_SIZE
    assert lm_head.row_size == (1280 // UOCR_Q8_0_BLOCK_SIZE) * UOCR_Q8_0_TYPE_SIZE
    assert lm_head.output_bytes == 129_280 * lm_head.row_size
    assert lm_head.qtype_reason == "policy"
    assert lm_head.qtype_reason_id == UOCR_QTYPE_REASON_POLICY
    assert lm_head.promotion_reason == "none"
    assert lm_head.promotion_reason_id == UOCR_PROMOTION_NONE

    sam_patch = plan.tensor_by_name("model.sam_model.patch_embed.proj.weight")
    assert sam_patch.qtype == "UOCR_TENSOR_Q8_0"
    assert sam_patch.shape == (768, 3, 16, 16)
    assert sam_patch.logical_shape == (768, 768)
    assert sam_patch.physical_shape == (768, 768)
    assert sam_patch.logical_input_width == 768
    assert sam_patch.physical_input_width == 768
    assert sam_patch.input_padding_width == 0
    assert sam_patch.output_bytes == 768 * ((768 // UOCR_Q8_0_BLOCK_SIZE) * UOCR_Q8_0_TYPE_SIZE)

    router = plan.tensor_by_name("model.layers.1.mlp.gate.weight")
    assert router.qtype == "UOCR_TENSOR_F16"
    assert router.logical_shape == router.shape
    assert router.logical_input_width == 0
    assert router.physical_input_width == 0
    assert router.input_padding_width == 0
    assert "router" in router.reason.lower()
    assert router.qtype_reason == "sensitive"
    assert router.qtype_reason_id == UOCR_QTYPE_REASON_SENSITIVE
    assert router.promotion_reason == "sensitive"
    assert router.promotion_reason_id == UOCR_PROMOTION_SENSITIVE

    pos_embed = plan.tensor_by_name("model.sam_model.pos_embed")
    assert pos_embed.qtype == "UOCR_TENSOR_F16"
    assert pos_embed.logical_shape == pos_embed.shape
    assert "pos" in pos_embed.reason.lower()

    raw_patch = plan.tensor_by_name("model.vision_model.embeddings.patch_embedding.weight")
    assert raw_patch.usage == "preserved-unused"
    assert raw_patch.qtype == "UOCR_TENSOR_Q8_0"
    assert raw_patch.shape == (1024, 3, 14, 14)
    assert raw_patch.logical_shape == (1024, 588)
    assert raw_patch.physical_shape == (1024, 608)
    assert raw_patch.logical_input_width == 588
    assert raw_patch.physical_input_width == 608
    assert raw_patch.input_padding_width == 20
    assert raw_patch.row_size == (608 // UOCR_Q8_0_BLOCK_SIZE) * UOCR_Q8_0_TYPE_SIZE
    raw_patch_dict = raw_patch.as_dict()
    assert raw_patch_dict["logical_input_width"] == 588
    assert raw_patch_dict["physical_input_width"] == 608
    assert raw_patch_dict["input_padding_width"] == 20

    payload = _tensor_directory_bytes(filter_plan_tensors(plan, "model.sam_model.patch_embed.proj.weight"))
    header_size = _TENSOR_DIRECTORY_HEADER_STRUCT.size
    entry = _TENSOR_ENTRY_STRUCT.unpack_from(payload, header_size)
    assert entry[6] == UOCR_TENSOR_Q8_0
    assert entry[8] == 2
    assert entry[9:13] == (768, 768, 0, 0)
    assert entry[13:17] == (768, 768, 0, 0)
    assert entry[25] == UOCR_QTYPE_REASON_POLICY
    assert entry[26] == UOCR_PROMOTION_NONE


def test_dyn_q4_converter_dry_run_uses_conservative_policy() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="dyn-q4")
    q8_plan = build_dry_run_plan(project_root() / "data/context", qprofile="dyn-q8")

    assert plan.tensor_count == EXPECTED_TENSOR_COUNT
    assert plan.total_source_bytes == EXPECTED_TOTAL_BYTES
    assert 0 < plan.total_output_bytes < q8_plan.total_output_bytes
    assert plan.qtype_histogram["UOCR_TENSOR_Q4_K"] > 0
    assert plan.qtype_histogram["UOCR_TENSOR_Q8_0"] > 0
    assert plan.qtype_histogram["UOCR_TENSOR_F16"] > 0
    width_summary = plan.summary_dict()["quantized_input_widths"]
    assert width_summary["quantized_tensor_count"] == (
        plan.qtype_histogram["UOCR_TENSOR_Q4_K"] + plan.qtype_histogram["UOCR_TENSOR_Q8_0"]
    )
    assert width_summary["padded_tensor_count"] > 0
    assert width_summary["max_padding_width"] > 0
    assert plan.qtype_reason_histogram["policy"] == plan.qtype_histogram["UOCR_TENSOR_Q4_K"]
    assert plan.qtype_reason_histogram["unaligned"] > 0
    assert plan.qtype_reason_histogram["sensitive"] > 0
    assert plan.promotion_reason_histogram["none"] == plan.qtype_histogram["UOCR_TENSOR_Q4_K"]
    assert plan.promotion_reason_histogram["unaligned"] > 0
    assert plan.promotion_reason_histogram["sensitive"] > 0

    hazard_summary = plan.summary_dict()["q4_unaligned_hazards"]
    assert hazard_summary["contract"] == Q4_UNALIGNED_HAZARD_CONTRACT
    assert hazard_summary["block_size"] == UOCR_Q4_K_BLOCK_SIZE
    assert hazard_summary["total_count"] == Q4_HAZARD_ROUTED_EXPERT_DOWN_COUNT + Q4_HAZARD_DENSE_LAYER0_DOWN_COUNT
    assert hazard_summary["by_kind"] == {
        Q4_HAZARD_DENSE_LAYER0_DOWN_NAME: Q4_HAZARD_DENSE_LAYER0_DOWN_COUNT,
        Q4_HAZARD_ROUTED_EXPERT_DOWN_NAME: Q4_HAZARD_ROUTED_EXPERT_DOWN_COUNT,
    }
    assert hazard_summary["examples"][Q4_HAZARD_ROUTED_EXPERT_DOWN_NAME]["logical_input_width"] == 896
    assert hazard_summary["examples"][Q4_HAZARD_ROUTED_EXPERT_DOWN_NAME]["required_physical_input_width"] == 1024
    assert hazard_summary["examples"][Q4_HAZARD_DENSE_LAYER0_DOWN_NAME]["logical_input_width"] == 6848
    assert hazard_summary["examples"][Q4_HAZARD_DENSE_LAYER0_DOWN_NAME]["required_physical_input_width"] == 6912

    hazard_tensors = [tensor for tensor in plan.tensors if tensor.q4_hazard_id != 0]
    assert len(hazard_tensors) == hazard_summary["total_count"]
    assert {tensor.qtype for tensor in hazard_tensors} == {"UOCR_TENSOR_Q8_0"}
    assert {tensor.qtype_reason for tensor in hazard_tensors} == {"unaligned"}
    assert {tensor.promotion_reason for tensor in hazard_tensors} == {"unaligned"}

    attn_q = plan.tensor_by_name("model.layers.3.self_attn.q_proj.weight")
    assert attn_q.qtype == "UOCR_TENSOR_Q4_K"
    assert attn_q.qtype_id == UOCR_TENSOR_Q4_K
    assert attn_q.logical_shape == (1280, 1280)
    assert attn_q.physical_shape == (1280, 1280)
    assert attn_q.logical_input_width == 1280
    assert attn_q.physical_input_width == 1280
    assert attn_q.input_padding_width == 0
    assert attn_q.q4_hazard == "none"
    assert attn_q.q4_hazard_id == 0
    assert attn_q.block_size == UOCR_Q4_K_BLOCK_SIZE
    assert attn_q.row_size == (1280 // UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE
    assert "attention projection" in attn_q.reason
    assert attn_q.qtype_reason == "policy"
    assert attn_q.qtype_reason_id == UOCR_QTYPE_REASON_POLICY
    assert attn_q.promotion_reason == "none"
    assert attn_q.promotion_reason_id == UOCR_PROMOTION_NONE

    expert_gate = plan.tensor_by_name("model.layers.1.mlp.experts.7.gate_proj.weight")
    assert expert_gate.qtype == "UOCR_TENSOR_Q4_K"
    assert expert_gate.logical_shape == (896, 1280)
    assert expert_gate.physical_shape == (896, 1280)
    assert expert_gate.logical_input_width == 1280
    assert expert_gate.physical_input_width == 1280
    assert expert_gate.input_padding_width == 0
    assert "routed expert gate/up" in expert_gate.reason

    expert_down = plan.tensor_by_name("model.layers.1.mlp.experts.7.down_proj.weight")
    assert expert_down.qtype == "UOCR_TENSOR_Q8_0"
    assert expert_down.logical_shape == (1280, 896)
    assert expert_down.physical_shape == (1280, 896)
    assert expert_down.logical_input_width == 896
    assert expert_down.physical_input_width == 896
    assert expert_down.input_padding_width == 0
    assert expert_down.q4_hazard == Q4_HAZARD_ROUTED_EXPERT_DOWN_NAME
    assert expert_down.q4_hazard_id == Q4_HAZARD_ROUTED_EXPERT_DOWN
    assert expert_down.q4_hazard_logical_input_width == 896
    assert expert_down.q4_hazard_required_physical_input_width == 1024
    assert expert_down.q4_hazard_padding_width == 128
    assert expert_down.row_size == (896 // UOCR_Q8_0_BLOCK_SIZE) * UOCR_Q8_0_TYPE_SIZE
    assert "routed expert down" in expert_down.reason
    assert "Q8_0" in expert_down.reason
    assert expert_down.qtype_reason == "unaligned"
    assert expert_down.qtype_reason_id == UOCR_QTYPE_REASON_UNALIGNED
    assert expert_down.promotion_reason == "unaligned"
    assert expert_down.promotion_reason_id == UOCR_PROMOTION_UNALIGNED

    dense_down = plan.tensor_by_name("model.layers.0.mlp.down_proj.weight")
    assert dense_down.qtype == "UOCR_TENSOR_Q8_0"
    assert dense_down.logical_shape == (1280, 6848)
    assert dense_down.physical_shape == (1280, 6848)
    assert dense_down.logical_input_width == 6848
    assert dense_down.physical_input_width == 6848
    assert dense_down.input_padding_width == 0
    assert dense_down.q4_hazard == Q4_HAZARD_DENSE_LAYER0_DOWN_NAME
    assert dense_down.q4_hazard_id == Q4_HAZARD_DENSE_LAYER0_DOWN
    assert dense_down.q4_hazard_logical_input_width == 6848
    assert dense_down.q4_hazard_required_physical_input_width == 6912
    assert dense_down.q4_hazard_padding_width == 64
    assert dense_down.row_size == (6848 // UOCR_Q8_0_BLOCK_SIZE) * UOCR_Q8_0_TYPE_SIZE
    assert "dense layer-0 down" in dense_down.reason

    shared_down = plan.tensor_by_name("model.layers.1.mlp.shared_experts.down_proj.weight")
    assert shared_down.qtype == "UOCR_TENSOR_Q4_K"
    assert shared_down.logical_shape == (1280, 1792)
    assert shared_down.physical_shape == (1280, 1792)
    assert shared_down.logical_input_width == 1792
    assert shared_down.physical_input_width == 1792
    assert shared_down.input_padding_width == 0
    assert shared_down.row_size == (1792 // UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE
    assert "shared expert down" in shared_down.reason

    lm_head = plan.tensor_by_name("lm_head.weight")
    assert lm_head.qtype == "UOCR_TENSOR_Q8_0"
    assert "LM head" in lm_head.reason
    assert lm_head.qtype_reason == "sensitive"
    assert lm_head.qtype_reason_id == UOCR_QTYPE_REASON_SENSITIVE
    assert lm_head.promotion_reason == "sensitive"
    assert lm_head.promotion_reason_id == UOCR_PROMOTION_SENSITIVE

    sam_patch = plan.tensor_by_name("model.sam_model.patch_embed.proj.weight")
    assert sam_patch.qtype == "UOCR_TENSOR_Q8_0"
    assert "vision weights" in sam_patch.reason

    router = plan.tensor_by_name("model.layers.1.mlp.gate.weight")
    assert router.qtype == "UOCR_TENSOR_F16"
    assert "router" in router.reason.lower()
    assert router.qtype_reason == "sensitive"
    assert router.qtype_reason_id == UOCR_QTYPE_REASON_SENSITIVE
    assert router.promotion_reason == "sensitive"
    assert router.promotion_reason_id == UOCR_PROMOTION_SENSITIVE


def test_padded_q4_k_design_is_explicit_and_disabled_by_default() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="dyn-q4")
    assert "UOCR_TENSOR_PADDED_Q4_K" not in plan.qtype_histogram

    routed_down = plan.tensor_by_name("model.layers.1.mlp.experts.7.down_proj.weight")
    assert routed_down.qtype == "UOCR_TENSOR_Q8_0"

    routed_design = describe_padded_q4_k_design(routed_down.shape)
    assert routed_design.qtype == "UOCR_TENSOR_PADDED_Q4_K"
    assert routed_design.qtype_id == UOCR_TENSOR_PADDED_Q4_K
    assert routed_design.logical_shape == (1280, 896)
    assert routed_design.physical_shape == (1280, 1024)
    assert routed_design.padding_cols == 128
    assert routed_design.logical_input_width == 896
    assert routed_design.physical_input_width == 1024
    assert routed_design.input_padding_width == 128
    assert routed_design.block_size == UOCR_Q4_K_BLOCK_SIZE
    assert routed_design.row_size == (1024 // UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE
    assert routed_design.output_bytes == 1280 * routed_design.row_size
    assert "zero" in routed_design.kernel_contract
    assert "logical inner width" in routed_design.kernel_contract
    routed_design_dict = routed_design.as_dict()
    assert routed_design_dict["kernel_contract"] == PADDED_Q4_K_KERNEL_CONTRACT
    assert routed_design_dict["logical_input_width"] == 896
    assert routed_design_dict["physical_input_width"] == 1024
    assert routed_design_dict["input_padding_width"] == 128

    dense_design = describe_padded_q4_k_design((1280, 6848))
    assert dense_design.logical_shape == (1280, 6848)
    assert dense_design.physical_shape == (1280, 6912)
    assert dense_design.padding_cols == 64
    assert dense_design.logical_input_width == 6848
    assert dense_design.physical_input_width == 6912
    assert dense_design.input_padding_width == 64
    assert dense_design.row_size == (6912 // UOCR_Q4_K_BLOCK_SIZE) * UOCR_Q4_K_TYPE_SIZE

    aligned_design = describe_padded_q4_k_design((1280, 1792))
    assert aligned_design.logical_shape == (1280, 1792)
    assert aligned_design.physical_shape == (1280, 1792)
    assert aligned_design.padding_cols == 0
    assert aligned_design.logical_input_width == 1792
    assert aligned_design.physical_input_width == 1792
    assert aligned_design.input_padding_width == 0


def test_filter_plan_tensors_relayouts_single_tensor(tmp_path) -> None:
    _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="fp16", strict=False)
    selected = filter_plan_tensors(plan, "model.sam_model.tiny_a.weight")

    assert selected.tensor_count == 1
    assert selected.tensors[0].name == "model.sam_model.tiny_a.weight"
    assert selected.tensors[0].payload_offset % UOCR_TENSOR_PAYLOAD_ALIGNMENT == 0
    tensor_data = next(section for section in selected.sections if section.section_type == UOCR_SECTION_TENSOR_DATA)
    assert selected.planned_file_size == tensor_data.offset + tensor_data.size
    assert tensor_data.size == selected.tensors[0].output_bytes

    selected_by_id = filter_plan_tensors(plan, str(selected.tensors[0].tensor_id))
    assert selected_by_id.tensors[0].name == selected.tensors[0].name


def test_write_uocr_model_streams_bf16_to_fp16_payload(tmp_path) -> None:
    payloads, _shapes = _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="fp16", strict=False)
    out = tmp_path / "tiny.uocr"

    written = write_uocr_model(plan, out)

    assert written == out
    raw = out.read_bytes()
    assert raw[:4] == b"UOCR"
    assert len(raw) == plan.planned_file_size
    file_size = struct.unpack_from("<Q", raw, 32)[0]
    section_dir_offset = struct.unpack_from("<Q", raw, 40)[0]
    assert file_size == plan.planned_file_size
    assert section_dir_offset == UOCR_FILE_HEADER_SIZE

    for tensor in plan.tensors:
        actual = raw[tensor.payload_offset : tensor.payload_offset + tensor.output_bytes]
        assert actual == _f16_from_bf16_payload(payloads[tensor.name])


def test_write_uocr_model_streams_dyn_q8_payload(tmp_path) -> None:
    payloads, _shapes = _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="dyn-q8", strict=False)
    out = tmp_path / "tiny-q8.uocr"

    written = write_uocr_model(plan, out)

    assert written == out
    raw = out.read_bytes()
    assert raw[:4] == b"UOCR"
    assert len(raw) == plan.planned_file_size
    assert struct.unpack_from("<I", raw, 20)[0] == UOCR_QPROFILE_DYN_Q8

    q8_tensor = plan.tensor_by_name("model.sam_model.tiny_a.weight")
    assert q8_tensor.qtype == "UOCR_TENSOR_Q8_0"
    assert q8_tensor.logical_shape == (2, 2)
    assert q8_tensor.physical_shape == (2, 32)
    assert q8_tensor.logical_input_width == 2
    assert q8_tensor.physical_input_width == 32
    assert q8_tensor.input_padding_width == 30
    assert q8_tensor.output_bytes == 2 * UOCR_Q8_0_TYPE_SIZE
    actual_q8 = raw[q8_tensor.payload_offset : q8_tensor.payload_offset + q8_tensor.output_bytes]
    expected_q8 = _q8_0_from_bf16_payload(payloads[q8_tensor.name], q8_tensor.shape, q8_tensor.physical_shape)
    assert actual_q8 == expected_q8

    f16_tensor = plan.tensor_by_name("model.vision_model.embeddings.patch_embedding.weight")
    assert f16_tensor.qtype == "UOCR_TENSOR_F16"
    actual_f16 = raw[f16_tensor.payload_offset : f16_tensor.payload_offset + f16_tensor.output_bytes]
    assert actual_f16 == _f16_from_bf16_payload(payloads[f16_tensor.name])

    directory = _tensor_directory_bytes(plan)
    assert directory in raw


def test_write_uocr_model_requires_overwrite_for_existing_output(tmp_path) -> None:
    _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="fp16", strict=False)
    out = tmp_path / "tiny.uocr"

    write_uocr_model(plan, out)
    with pytest.raises(FileExistsError):
        write_uocr_model(plan, out)
    write_uocr_model(plan, out, overwrite=True)
    assert out.exists()
