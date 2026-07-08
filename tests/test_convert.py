from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from unlimitedocr_c.convert import (
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_BYTES,
    PRESERVED_UNUSED_NORMAL_OCR_PREFIXES,
    ROW_MAJOR_LAYOUT_CONTRACT,
    UOCR_LAYOUT_TRANSFORM_IDENTITY,
    UOCR_PROMOTION_NONE,
    UOCR_FILE_HEADER_SIZE,
    UOCR_Q8_GROUP_SIZE_DEFAULT,
    UOCR_Q8_MAX,
    UOCR_Q8_MIN,
    UOCR_QPROFILE_MIXED_Q4,
    UOCR_QPROFILE_MIXED_Q8_0,
    UOCR_QTYPE_REASON_FP16_BASELINE,
    UOCR_Q4_GROUP_SIZE_DEFAULT,
    UOCR_Q4_MAX,
    UOCR_Q4_MIN,
    UOCR_SECTION_TENSOR_DATA,
    UOCR_TENSOR_DATA_ALIGNMENT,
    UOCR_TENSOR_FLAG_ROW_MAJOR,
    UOCR_TENSOR_PAYLOAD_ALIGNMENT,
    UOCR_TENSOR_Q4_0,
    UOCR_TENSOR_Q8_0,
    _TENSOR_DIRECTORY_HEADER_STRUCT,
    _TENSOR_ENTRY_STRUCT,
    _quantize_q4_0_rows,
    _tensor_directory_bytes,
    q8_summary,
    build_dry_run_plan,
    compare_single_tensor_conversion,
    filter_plan_tensors,
    is_preserved_unused_in_normal_ocr,
    main as convert_main,
    q4_qscale_bytes,
    q4_qweight_bytes,
    q4_total_bytes,
    q8_qscale_bytes,
    q8_qweight_bytes,
    q8_total_bytes,
    write_uocr_model,
)
from unlimitedocr_c.frontend import project_root
from unlimitedocr_c.quant_cfg import (
    QuantConfig,
    QuantModuleSpec,
    _parse_config,
    all_supported_quant_config,
    load_default_quant_config,
)
from unlimitedocr_c.tensor_registry import TENSOR_ID_LM_HEAD, TENSOR_ID_TOK_EMBED, TensorFamily, TensorProjection, tensor_id_layer_attn


UOCR_TENSOR_F16_NAME = "UOCR_TENSOR_F16"


def _bf16_payload(values: np.ndarray) -> bytes:
    fp32 = np.asarray(values, dtype=np.dtype("<f4"))
    return ((fp32.view(np.dtype("<u4")) >> np.uint32(16)).astype(np.dtype("<u2"))).tobytes()


def _f32_from_bf16_payload(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload, dtype=np.dtype("<u2"))
    return (words.astype(np.uint32) << np.uint32(16)).view(np.dtype("<f4"))


def _f16_from_bf16_payload(payload: bytes) -> bytes:
    return _f32_from_bf16_payload(payload).astype(np.dtype("<f2")).tobytes()


def _read_length_prefixed_safetensors_header(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", data, 0)
    start = 8
    end = start + header_len
    return json.loads(data[start:end].decode("utf-8"))


def _write_tiny_safetensors(hf_dir: Path) -> tuple[dict[str, bytes], dict[str, tuple[int, ...]]]:
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

    index = {"metadata": {"total_size": cursor}, "weight_map": {name: "model.safetensors" for name in tensors}}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return payloads, {name: shape for name, (_values, shape) in tensors.items()}


def _write_tiny_q8_safetensors(hf_dir: Path) -> np.ndarray:
    values = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(2, 64)
    payload = _bf16_payload(values)
    header = {
        "__metadata__": {"format": "pt"},
        "lm_head.weight": {"dtype": "BF16", "shape": [2, 64], "data_offsets": [0, len(payload)]},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with (hf_dir / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)
    index = {"metadata": {"total_size": len(payload)}, "weight_map": {"lm_head.weight": "model.safetensors"}}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return values


def _q8_expected(values: np.ndarray) -> tuple[bytes, bytes]:
    grouped = values.reshape(values.shape[0], values.shape[1] // UOCR_Q8_GROUP_SIZE_DEFAULT, UOCR_Q8_GROUP_SIZE_DEFAULT)
    max_abs = np.max(np.abs(grouped), axis=2)
    scales = np.where(max_abs == 0.0, np.float32(1.0), max_abs / np.float32(UOCR_Q8_MAX)).astype(np.float32)
    qweights = np.clip(np.rint(grouped / scales[:, :, None]), UOCR_Q8_MIN, UOCR_Q8_MAX).astype(np.int8)
    return qweights.reshape(values.shape).tobytes(), scales.astype(np.dtype("<f2")).tobytes()


def _full_model_converter_hf_dir_or_skip() -> Path:
    if os.environ.get("UOCR_RUN_LARGE_TESTS") != "1":
        pytest.skip("set UOCR_RUN_LARGE_TESTS=1 to run the full-model converter smoke test")
    hf_dir_env = os.environ.get("UOCR_HF_DIR")
    if not hf_dir_env:
        pytest.skip("set UOCR_HF_DIR to a local baidu/Unlimited-OCR checkout with full safetensors payload")
    return Path(hf_dir_env)


def test_cached_safetensors_header_current_checkpoint_facts() -> None:
    context_dir = project_root() / "data/context"
    header = _read_length_prefixed_safetensors_header(context_dir / "model.safetensors.header")
    index = json.loads((context_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    entries = {name: meta for name, meta in header.items() if name != "__metadata__"}
    assert len(entries) == EXPECTED_TENSOR_COUNT
    assert len(index["weight_map"]) == EXPECTED_TENSOR_COUNT
    assert index["metadata"]["total_size"] == EXPECTED_TOTAL_BYTES


def test_q8_format_constants_and_byte_helpers() -> None:
    assert UOCR_QPROFILE_MIXED_Q8_0 == 2
    assert UOCR_TENSOR_Q8_0 == 3
    assert UOCR_Q8_GROUP_SIZE_DEFAULT == 64
    assert (UOCR_Q8_MIN, UOCR_Q8_MAX) == (-127, 127)
    assert q8_qweight_bytes((2, 64)) == 128
    assert q8_qscale_bytes((2, 64)) == 4
    assert q8_total_bytes((2, 64)) == 132
    with pytest.raises(ValueError, match="rank-2"):
        q8_qweight_bytes((64,))
    with pytest.raises(ValueError, match="divisible"):
        q8_qweight_bytes((2, 63))


def test_q4_format_constants_and_byte_helpers() -> None:
    assert UOCR_QPROFILE_MIXED_Q4 == 3
    assert UOCR_TENSOR_Q4_0 == 4
    assert UOCR_Q4_GROUP_SIZE_DEFAULT == 64
    assert (UOCR_Q4_MIN, UOCR_Q4_MAX) == (-8, 7)
    assert q4_qweight_bytes((2, 64)) == 64
    assert q4_qscale_bytes((2, 64)) == 4
    assert q4_total_bytes((2, 64)) == 68
    assert q4_qweight_bytes((896, 1280)) == 896 * 640
    with pytest.raises(ValueError, match="rank-2"):
        q4_qweight_bytes((64,))
    with pytest.raises(ValueError, match="divisible"):
        q4_qweight_bytes((2, 63))


def test_q4_pack_roundtrip() -> None:
    rng = np.random.default_rng(7)
    values = rng.standard_normal((4, 128), dtype=np.float32) * 0.05
    values[0, :64] = 0.0            # all-zero group -> scale 1.0, q 0
    values[1, 64] = -0.5            # negative signed max
    packed, scales_f16, dequant = _quantize_q4_0_rows(values, UOCR_Q4_GROUP_SIZE_DEFAULT)
    assert packed.shape == (4, 64)
    assert scales_f16.shape == (4, 2)

    # Unpack nibbles (stored as q + 8, group-half split: byte j of a group
    # holds w[j] low / w[j+32] high) and re-dequantize.
    packed_groups = packed.reshape(4, 2, 32)
    low = (packed_groups & 0x0F).astype(np.int16) - 8
    high = (packed_groups >> 4).astype(np.int16) - 8
    q = np.concatenate([low, high], axis=2).reshape(4, 128)
    assert q.min() >= UOCR_Q4_MIN and q.max() <= UOCR_Q4_MAX
    scales = scales_f16.astype(np.float32)
    redequant = (q.reshape(4, 2, 64).astype(np.float32) * scales[:, :, None]).reshape(4, 128)
    np.testing.assert_array_equal(redequant, dequant)

    # All-zero group encodes exactly zero.
    np.testing.assert_array_equal(dequant[0, :64], np.zeros(64, dtype=np.float32))
    # Signed-max element reconstructs exactly (q = -8 by construction).
    assert abs(dequant[1, 64] - np.float32(np.float16(-0.5 / -8.0)) * -8.0) < 1e-6
    # Overall error bounded by half a step per group.
    err = np.abs(dequant - values)
    assert float(err.max()) <= float(np.abs(scales).max()) * 0.5 + 1e-6


def test_quant_cfg_v2_qtype_rules() -> None:
    base = {
        "version": 2,
        "group_size": 64,
        "modules": [
            {
                "name": "moe_routed_experts",
                "family": "MOE_EXPERT",
                "projections": ["GATE", "UP", "DOWN"],
                "supported": True,
                "qtype": "q4_0",
            },
        ],
    }
    cfg = _parse_config(base)
    assert cfg.qtype_for(TensorFamily.MOE_EXPERT, TensorProjection.GATE) == "q4_0"
    assert cfg.modules[0].qtype == "q4_0"

    with pytest.raises(ValueError, match="only supported for MOE_EXPERT"):
        _parse_config(
            {
                "version": 2,
                "group_size": 64,
                "modules": [
                    {
                        "name": "lm_head",
                        "family": "LM_HEAD",
                        "projections": ["WEIGHT"],
                        "supported": True,
                        "qtype": "q4_0",
                    },
                ],
            }
        )

    with pytest.raises(ValueError, match="schema version is 1"):
        _parse_config({**base, "version": 1})


def test_shipped_quant_cfg_is_v2_with_q4_experts() -> None:
    cfg = load_default_quant_config()
    assert cfg.version == 2
    # Routed experts carry the q4_0 override (fused Q4 kernels landed); it only
    # takes effect under --qprofile mixed-q4, so mixed-q8_0 stays Q8 everywhere.
    assert cfg.qtype_for(TensorFamily.MOE_EXPERT, TensorProjection.GATE) == "q4_0"
    assert cfg.qtype_for(TensorFamily.LM_HEAD, TensorProjection.WEIGHT) == "q8_0"


def test_fp16_converter_dry_run_against_cached_header() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="fp16")
    assert plan.tensor_count == EXPECTED_TENSOR_COUNT
    assert plan.total_source_bytes == EXPECTED_TOTAL_BYTES
    assert plan.total_output_bytes == EXPECTED_TOTAL_BYTES
    assert plan.qtype_histogram == {"UOCR_TENSOR_F16": EXPECTED_TENSOR_COUNT}
    assert plan.qtype_reason_histogram == {"fp16-baseline": EXPECTED_TENSOR_COUNT}
    assert plan.promotion_reason_histogram == {"none": EXPECTED_TENSOR_COUNT}

    layout_summary = plan.summary_dict()["layout_transforms"]
    assert layout_summary["contract"] == ROW_MAJOR_LAYOUT_CONTRACT
    assert layout_summary["row_major_count"] == EXPECTED_TENSOR_COUNT
    assert layout_summary["transposed_count"] == 0
    assert layout_summary["by_transform"] == {"identity-row-major": EXPECTED_TENSOR_COUNT}

    preserved_unused = [tensor for tensor in plan.tensors if tensor.usage == "preserved-unused"]
    assert [tensor.name for tensor in preserved_unused] == ["model.vision_model.embeddings.patch_embedding.weight"]
    assert all(is_preserved_unused_in_normal_ocr(tensor.name) for tensor in preserved_unused)
    assert PRESERVED_UNUSED_NORMAL_OCR_PREFIXES == ("model.vision_model.embeddings.patch_embedding.",)

    tensor_data = next(section for section in plan.sections if section.section_type == UOCR_SECTION_TENSOR_DATA)
    assert tensor_data.offset % UOCR_TENSOR_DATA_ALIGNMENT == 0
    assert tensor_data.size == EXPECTED_TOTAL_BYTES
    assert all(tensor.payload_offset % UOCR_TENSOR_PAYLOAD_ALIGNMENT == 0 for tensor in plan.tensors)

    tok_embed = plan.tensor_by_name("model.embed_tokens.weight")
    assert tok_embed.tensor_id == TENSOR_ID_TOK_EMBED
    lm_head = plan.tensor_by_name("lm_head.weight")
    assert lm_head.tensor_id == TENSOR_ID_LM_HEAD
    assert lm_head.output_dtype == "F16"
    assert lm_head.qtype_reason_id == UOCR_QTYPE_REASON_FP16_BASELINE
    assert lm_head.promotion_reason_id == UOCR_PROMOTION_NONE
    assert lm_head.layout_transform_id == UOCR_LAYOUT_TRANSFORM_IDENTITY
    assert lm_head.layout_flags == UOCR_TENSOR_FLAG_ROW_MAJOR

    attn_q = plan.tensor_by_name("model.layers.3.self_attn.q_proj.weight")
    assert attn_q.tensor_id == tensor_id_layer_attn(3, TensorProjection.Q)
    assert attn_q.family_id == TensorFamily.LAYER_ATTN


def test_mixed_q8_converter_dry_run_plans_decoder_quantization() -> None:
    plan = build_dry_run_plan(
        project_root() / "data/context",
        qprofile="mixed-q8_0",
        quant_cfg=all_supported_quant_config(),
    )
    summary = q8_summary(plan.tensors)
    assert plan.qprofile == "mixed-q8_0"
    assert summary["tensor_count"] > 0
    assert summary["qweight_bytes"] > 0
    assert summary["qscale_bytes"] > 0
    assert summary["estimated_savings_bytes"] > 0
    assert plan.total_output_bytes == EXPECTED_TOTAL_BYTES - summary["estimated_savings_bytes"]

    tok_embed = plan.tensor_by_name("model.embed_tokens.weight")
    lm_head = plan.tensor_by_name("lm_head.weight")
    router = plan.tensor_by_name("model.layers.1.mlp.gate.weight")
    projector = plan.tensor_by_name("model.projector.layers.weight")
    attn_q = plan.tensor_by_name("model.layers.3.self_attn.q_proj.weight")
    expert_gate = plan.tensor_by_name("model.layers.1.mlp.experts.0.gate_proj.weight")
    assert tok_embed.qtype_id == UOCR_TENSOR_Q8_0
    assert lm_head.qtype_id == UOCR_TENSOR_Q8_0
    assert attn_q.qtype_id == UOCR_TENSOR_Q8_0
    assert expert_gate.qtype_id == UOCR_TENSOR_Q8_0
    assert router.qtype == "UOCR_TENSOR_F16"
    assert projector.qtype == "UOCR_TENSOR_F16"
    assert tok_embed.block_size == UOCR_Q8_GROUP_SIZE_DEFAULT
    assert tok_embed.output_bytes == tok_embed.shape[0] * tok_embed.shape[1]
    assert tok_embed.scale_size == tok_embed.shape[0] * (tok_embed.shape[1] // UOCR_Q8_GROUP_SIZE_DEFAULT) * 2
    assert tok_embed.payload_offset % UOCR_TENSOR_PAYLOAD_ALIGNMENT == 0
    assert tok_embed.scale_offset % UOCR_TENSOR_PAYLOAD_ALIGNMENT == 0

    layer1_experts = [
        tensor
        for tensor in plan.tensors
        if tensor.family == TensorFamily.MOE_EXPERT.name and tensor.layer == 1 and tensor.qtype_id == UOCR_TENSOR_Q8_0
    ]
    assert layer1_experts
    assert all(a.payload_offset + a.output_bytes == b.payload_offset for a, b in zip(layer1_experts, layer1_experts[1:]))
    assert all(a.scale_offset + a.scale_size == b.scale_offset for a, b in zip(layer1_experts, layer1_experts[1:]))


def test_mixed_q4_converter_dry_run_plans_expert_q4() -> None:
    plan = build_dry_run_plan(
        project_root() / "data/context",
        qprofile="mixed-q4",
        quant_cfg=all_supported_quant_config(expert_qtype="q4_0"),
    )
    assert plan.qprofile == "mixed-q4"

    expert_gate = plan.tensor_by_name("model.layers.1.mlp.experts.0.gate_proj.weight")
    expert_down = plan.tensor_by_name("model.layers.1.mlp.experts.0.down_proj.weight")
    lm_head = plan.tensor_by_name("lm_head.weight")
    attn_q = plan.tensor_by_name("model.layers.3.self_attn.q_proj.weight")
    router = plan.tensor_by_name("model.layers.1.mlp.gate.weight")

    assert expert_gate.qtype_id == UOCR_TENSOR_Q4_0
    assert expert_gate.output_dtype == "Q4_0"
    assert expert_gate.row_size == expert_gate.shape[1] // 2
    assert expert_gate.output_bytes == expert_gate.shape[0] * expert_gate.shape[1] // 2
    assert expert_gate.scale_size == expert_gate.shape[0] * (expert_gate.shape[1] // 64) * 2
    assert expert_down.qtype_id == UOCR_TENSOR_Q4_0
    assert lm_head.qtype_id == UOCR_TENSOR_Q8_0
    assert attn_q.qtype_id == UOCR_TENSOR_Q8_0
    assert router.qtype == UOCR_TENSOR_F16_NAME

    # Interleaved expert-major slab stays contiguous with Q4 payloads/scales.
    layer1_experts = [
        tensor
        for tensor in plan.tensors
        if tensor.family == TensorFamily.MOE_EXPERT.name and tensor.layer == 1 and tensor.qtype_id == UOCR_TENSOR_Q4_0
    ]
    assert len(layer1_experts) == 64 * 3
    assert all(
        a.payload_offset + a.output_bytes == b.payload_offset for a, b in zip(layer1_experts, layer1_experts[1:])
    )
    assert all(a.scale_offset + a.scale_size == b.scale_offset for a, b in zip(layer1_experts, layer1_experts[1:]))

    summary = q8_summary(plan.tensors)
    assert summary["by_qtype"]["q4_0"]["tensor_count"] == 11 * 64 * 3
    assert summary["by_qtype"]["q4_0"]["estimated_savings_bytes"] > 0
    assert summary["by_qtype"]["q8_0"]["tensor_count"] > 0


def test_write_and_compare_tiny_q4_expert_payload(tmp_path: Path) -> None:
    values = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(2, 64)
    payload = _bf16_payload(values)
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    header = {
        "__metadata__": {"format": "pt"},
        name: {"dtype": "BF16", "shape": [2, 64], "data_offsets": [0, len(payload)]},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with (tmp_path / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)
    index = {"metadata": {"total_size": len(payload)}, "weight_map": {name: "model.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    plan = build_dry_run_plan(
        tmp_path,
        qprofile="mixed-q4",
        strict=False,
        quant_cfg=all_supported_quant_config(expert_qtype="q4_0"),
    )
    tensor = plan.tensor_by_name(name)
    assert tensor.qtype_id == UOCR_TENSOR_Q4_0
    out = tmp_path / "tiny-q4.uocr"
    write_uocr_model(plan, out, overwrite=True)

    raw = out.read_bytes()
    assert raw[:4] == b"UOCR"
    assert struct.unpack_from("<I", raw, 20)[0] == UOCR_QPROFILE_MIXED_Q4
    expected_packed, expected_scales, _ = _quantize_q4_0_rows(
        _f32_from_bf16_payload(payload).reshape(2, 64), UOCR_Q4_GROUP_SIZE_DEFAULT
    )
    assert raw[tensor.payload_offset : tensor.payload_offset + tensor.output_bytes] == expected_packed.tobytes()
    assert raw[tensor.scale_offset : tensor.scale_offset + tensor.scale_size] == expected_scales.tobytes()

    filtered = filter_plan_tensors(plan, name)
    result = compare_single_tensor_conversion(filtered, out)
    assert result.matches
    assert result.qtype == "UOCR_TENSOR_Q4_0"
    assert result.dequant_max_abs_error is not None
    assert result.dequant_max_abs_error < 0.1
    assert result.dequant_rmse is not None
    assert result.dequant_rmse < 0.05


def test_vision_registry_roles_are_stable_for_rank2_linears() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="fp16")

    clip_qkv = plan.tensor_by_name("model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight")
    clip_o = plan.tensor_by_name("model.vision_model.transformer.layers.0.self_attn.out_proj.weight")
    clip_fc1 = plan.tensor_by_name("model.vision_model.transformer.layers.0.mlp.fc1.weight")
    clip_fc2 = plan.tensor_by_name("model.vision_model.transformer.layers.0.mlp.fc2.weight")
    sam_qkv = plan.tensor_by_name("model.sam_model.blocks.0.attn.qkv.weight")
    sam_o = plan.tensor_by_name("model.sam_model.blocks.0.attn.proj.weight")
    sam_fc1 = plan.tensor_by_name("model.sam_model.blocks.0.mlp.lin1.weight")
    sam_fc2 = plan.tensor_by_name("model.sam_model.blocks.0.mlp.lin2.weight")

    assert clip_qkv.family_id == TensorFamily.VISION_CLIP and clip_qkv.projection_id == TensorProjection.VISION_ATTN_QKV
    assert clip_o.family_id == TensorFamily.VISION_CLIP and clip_o.projection_id == TensorProjection.VISION_ATTN_O
    assert clip_fc1.family_id == TensorFamily.VISION_CLIP and clip_fc1.projection_id == TensorProjection.VISION_MLP_FC1
    assert clip_fc2.family_id == TensorFamily.VISION_CLIP and clip_fc2.projection_id == TensorProjection.VISION_MLP_FC2
    assert sam_qkv.family_id == TensorFamily.VISION_SAM and sam_qkv.projection_id == TensorProjection.VISION_ATTN_QKV
    assert sam_o.family_id == TensorFamily.VISION_SAM and sam_o.projection_id == TensorProjection.VISION_ATTN_O
    assert sam_fc1.family_id == TensorFamily.VISION_SAM and sam_fc1.projection_id == TensorProjection.VISION_MLP_FC1
    assert sam_fc2.family_id == TensorFamily.VISION_SAM and sam_fc2.projection_id == TensorProjection.VISION_MLP_FC2
    assert clip_qkv.shape == (3072, 1024)
    assert sam_qkv.shape == (2304, 768)

    clip_position = plan.tensor_by_name("model.vision_model.embeddings.position_embedding.weight")
    sam_rel_pos = plan.tensor_by_name("model.sam_model.blocks.0.attn.rel_pos_h")
    assert clip_position.projection_id == TensorProjection.NONE
    assert sam_rel_pos.projection_id == TensorProjection.NONE


def test_mixed_q8_default_cfg_only_quantizes_runtime_supported_modules() -> None:
    cfg = load_default_quant_config()
    for name in ("visual_projector", "clip_mlp", "clip_attention", "sam_mlp", "sam_attention"):
        module = cfg.module_by_name(name)
        assert module is not None
        assert module.supported is True
    for name in ("sam_patch_embed", "sam_neck_convs"):
        module = cfg.module_by_name(name)
        assert module is not None
        assert module.supported is False

    plan = build_dry_run_plan(project_root() / "data/context", qprofile="mixed-q8_0")
    tok_embed = plan.tensor_by_name("model.embed_tokens.weight")
    lm_head = plan.tensor_by_name("lm_head.weight")
    attn_q = plan.tensor_by_name("model.layers.3.self_attn.q_proj.weight")
    attn_o = plan.tensor_by_name("model.layers.3.self_attn.o_proj.weight")
    expert_gate = plan.tensor_by_name("model.layers.1.mlp.experts.0.gate_proj.weight")
    shared_gate = plan.tensor_by_name("model.layers.1.mlp.shared_experts.gate_proj.weight")
    dense_gate = plan.tensor_by_name("model.layers.0.mlp.gate_proj.weight")
    projector = plan.tensor_by_name("model.projector.layers.weight")
    clip_fc1 = plan.tensor_by_name("model.vision_model.transformer.layers.0.mlp.fc1.weight")
    clip_qkv = plan.tensor_by_name("model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight")
    sam_fc1 = plan.tensor_by_name("model.sam_model.blocks.0.mlp.lin1.weight")
    sam_qkv = plan.tensor_by_name("model.sam_model.blocks.0.attn.qkv.weight")
    assert tok_embed.qtype_id == UOCR_TENSOR_Q8_0
    assert attn_q.qtype_id == UOCR_TENSOR_Q8_0
    assert attn_o.qtype_id == UOCR_TENSOR_Q8_0
    assert dense_gate.qtype_id == UOCR_TENSOR_Q8_0
    assert shared_gate.qtype_id == UOCR_TENSOR_Q8_0
    assert expert_gate.qtype_id == UOCR_TENSOR_Q8_0
    assert lm_head.qtype_id == UOCR_TENSOR_Q8_0
    assert projector.qtype_id == UOCR_TENSOR_Q8_0
    assert projector.block_size == UOCR_Q8_GROUP_SIZE_DEFAULT
    assert clip_fc1.qtype_id == UOCR_TENSOR_Q8_0
    assert clip_fc1.row_size == 1024
    assert clip_qkv.qtype_id == UOCR_TENSOR_Q8_0
    assert clip_qkv.row_size == 1024
    assert sam_fc1.qtype_id == UOCR_TENSOR_Q8_0
    assert sam_fc1.row_size == 768
    assert sam_qkv.qtype_id == UOCR_TENSOR_Q8_0
    assert sam_qkv.row_size == 768


def test_mixed_q8_vision_modules_quantize_only_when_enabled() -> None:
    supported_pairs = {
        (TensorFamily.PROJECTOR, TensorProjection.WEIGHT): "q8_0",
        (TensorFamily.VISION_CLIP, TensorProjection.VISION_MLP_FC1): "q8_0",
        (TensorFamily.VISION_CLIP, TensorProjection.VISION_MLP_FC2): "q8_0",
    }
    cfg = QuantConfig(
        version=1,
        group_size=UOCR_Q8_GROUP_SIZE_DEFAULT,
        modules=(
            QuantModuleSpec("visual_projector", TensorFamily.PROJECTOR, frozenset({TensorProjection.WEIGHT}), True),
            QuantModuleSpec(
                "clip_mlp",
                TensorFamily.VISION_CLIP,
                frozenset({TensorProjection.VISION_MLP_FC1, TensorProjection.VISION_MLP_FC2}),
                True,
            ),
        ),
        supported_pairs=supported_pairs,
    )

    plan = build_dry_run_plan(project_root() / "data/context", qprofile="mixed-q8_0", quant_cfg=cfg)
    projector = plan.tensor_by_name("model.projector.layers.weight")
    clip_fc1 = plan.tensor_by_name("model.vision_model.transformer.layers.0.mlp.fc1.weight")
    clip_fc2 = plan.tensor_by_name("model.vision_model.transformer.layers.0.mlp.fc2.weight")
    clip_qkv = plan.tensor_by_name("model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight")
    sam_fc1 = plan.tensor_by_name("model.sam_model.blocks.0.mlp.lin1.weight")

    assert projector.qtype_id == UOCR_TENSOR_Q8_0
    assert clip_fc1.qtype_id == UOCR_TENSOR_Q8_0
    assert clip_fc2.qtype_id == UOCR_TENSOR_Q8_0
    assert clip_qkv.qtype == UOCR_TENSOR_F16_NAME
    assert sam_fc1.qtype == UOCR_TENSOR_F16_NAME
    assert projector.block_size == UOCR_Q8_GROUP_SIZE_DEFAULT
    assert clip_fc1.row_size == 1024
    assert clip_fc2.row_size == 4096


def test_tensor_directory_bytes_records_fp16_metadata() -> None:
    plan = filter_plan_tensors(build_dry_run_plan(project_root() / "data/context", qprofile="fp16"), "lm_head.weight")
    raw = _tensor_directory_bytes(plan)
    magic, version, entry_size, tensor_count = _TENSOR_DIRECTORY_HEADER_STRUCT.unpack_from(raw, 0)
    assert magic != 0
    assert version == 1
    assert entry_size == _TENSOR_ENTRY_STRUCT.size
    assert tensor_count == 1
    entry = _TENSOR_ENTRY_STRUCT.unpack_from(raw, _TENSOR_DIRECTORY_HEADER_STRUCT.size)
    tensor = plan.tensors[0]
    assert entry[0] == tensor.tensor_id
    assert entry[6] == tensor.qtype_id
    assert entry[7] == UOCR_TENSOR_FLAG_ROW_MAJOR
    assert entry[19] == 0
    assert entry[20] == 0
    assert entry[21] == 0
    assert entry[22] == 0
    assert entry[25] == UOCR_QTYPE_REASON_FP16_BASELINE
    assert entry[26] == UOCR_PROMOTION_NONE


def test_tensor_directory_bytes_records_q8_metadata() -> None:
    plan = filter_plan_tensors(
        build_dry_run_plan(
            project_root() / "data/context",
            qprofile="mixed-q8_0",
            quant_cfg=all_supported_quant_config(),
        ),
        "lm_head.weight",
    )
    raw = _tensor_directory_bytes(plan)
    entry = _TENSOR_ENTRY_STRUCT.unpack_from(raw, _TENSOR_DIRECTORY_HEADER_STRUCT.size)
    tensor = plan.tensors[0]
    assert entry[0] == tensor.tensor_id
    assert entry[6] == UOCR_TENSOR_Q8_0
    assert entry[18] == tensor.output_bytes
    assert entry[19] == UOCR_Q8_GROUP_SIZE_DEFAULT
    assert entry[20] == tensor.shape[1]
    assert entry[21] == tensor.scale_offset
    assert entry[22] == tensor.scale_size


def test_write_uocr_model_streams_tiny_fp16_payloads(tmp_path: Path) -> None:
    payloads, shapes = _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="fp16", strict=False)
    out = tmp_path / "tiny.uocr"
    stats = tmp_path / "stats.json"
    write_uocr_model(plan, out, overwrite=True, stats_path=stats)
    raw = out.read_bytes()
    assert raw[:4] == b"UOCR"
    assert struct.unpack_from("<I", raw, 20)[0] == 1
    for tensor in plan.tensors:
        assert tensor.shape == shapes[tensor.name]
        expected = _f16_from_bf16_payload(payloads[tensor.name])
        assert raw[tensor.payload_offset : tensor.payload_offset + tensor.output_bytes] == expected
    report = json.loads(stats.read_text(encoding="utf-8"))
    assert report["qprofile"] == "fp16"
    assert report["output_bytes_written"] == sum(t.output_bytes for t in plan.tensors)


def test_write_uocr_model_streams_tiny_q8_payloads(tmp_path: Path) -> None:
    source_values = _write_tiny_q8_safetensors(tmp_path)
    plan = build_dry_run_plan(
        tmp_path,
        qprofile="mixed-q8_0",
        strict=False,
        quant_cfg=all_supported_quant_config(),
    )
    tensor = plan.tensor_by_name("lm_head.weight")
    assert tensor.qtype_id == UOCR_TENSOR_Q8_0
    out = tmp_path / "tiny-q8.uocr"
    stats = tmp_path / "q8-stats.json"
    write_uocr_model(plan, out, overwrite=True, stats_path=stats)
    raw = out.read_bytes()
    assert raw[:4] == b"UOCR"
    assert struct.unpack_from("<I", raw, 20)[0] == UOCR_QPROFILE_MIXED_Q8_0
    expected_qweight, expected_qscale = _q8_expected(_f32_from_bf16_payload(_bf16_payload(source_values)).reshape(2, 64))
    assert raw[tensor.payload_offset : tensor.payload_offset + tensor.output_bytes] == expected_qweight
    assert raw[tensor.scale_offset : tensor.scale_offset + tensor.scale_size] == expected_qscale
    report = json.loads(stats.read_text(encoding="utf-8"))
    assert report["qprofile"] == "mixed-q8_0"
    assert report["output_bytes_written"] == tensor.output_bytes + tensor.scale_size


def test_compare_single_tensor_conversion_fp16(tmp_path: Path) -> None:
    _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="fp16", strict=False)
    filtered = filter_plan_tensors(plan, "model.sam_model.tiny_a.weight")
    out = tmp_path / "tiny.uocr"
    write_uocr_model(plan, out, overwrite=True)
    result = compare_single_tensor_conversion(filtered, out)
    assert result.matches
    assert result.qtype == "UOCR_TENSOR_F16"


def test_compare_single_tensor_conversion_q8(tmp_path: Path) -> None:
    _write_tiny_q8_safetensors(tmp_path)
    plan = build_dry_run_plan(
        tmp_path,
        qprofile="mixed-q8_0",
        strict=False,
        quant_cfg=all_supported_quant_config(),
    )
    filtered = filter_plan_tensors(plan, "lm_head.weight")
    out = tmp_path / "tiny-q8.uocr"
    write_uocr_model(plan, out, overwrite=True)
    result = compare_single_tensor_conversion(filtered, out)
    assert result.matches
    assert result.qtype == "UOCR_TENSOR_Q8_0"
    assert result.expected_bytes == filtered.tensors[0].output_bytes + filtered.tensors[0].scale_size
    assert result.dequant_max_abs_error is not None
    assert result.dequant_max_abs_error < 0.01
    assert result.dequant_rmse is not None
    assert result.dequant_rmse < 0.01


def test_converter_cli_dry_run_and_dump_plan(tmp_path: Path, capsys) -> None:
    _write_tiny_safetensors(tmp_path)
    dump = tmp_path / "plan.json"
    assert convert_main(["--hf-dir", str(tmp_path), "--dry-run", "--dump-plan", str(dump), "--relaxed-validation"]) == 0
    captured = capsys.readouterr()
    assert "qprofile: fp16" in captured.out
    dumped = json.loads(dump.read_text(encoding="utf-8"))
    assert dumped["summary"]["qprofile"] == "fp16"


def test_full_model_write_smoke_when_enabled(tmp_path: Path) -> None:
    hf_dir = _full_model_converter_hf_dir_or_skip()
    plan = filter_plan_tensors(build_dry_run_plan(hf_dir, qprofile="fp16"), "model.image_newline")
    out = tmp_path / "single.uocr"
    write_uocr_model(plan, out, overwrite=True)
    assert out.stat().st_size >= UOCR_FILE_HEADER_SIZE
