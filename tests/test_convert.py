from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from unlimitedocr_c.convert import (
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_BYTES,
    PRESERVED_UNUSED_NORMAL_OCR_PREFIXES,
    UOCR_FILE_HEADER_SIZE,
    UOCR_SECTION_TENSOR_DATA,
    UOCR_TENSOR_DATA_ALIGNMENT,
    UOCR_TENSOR_PAYLOAD_ALIGNMENT,
    build_dry_run_plan,
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


def _f16_from_bf16_payload(payload: bytes) -> bytes:
    words = np.frombuffer(payload, dtype=np.dtype("<u2"))
    fp32 = (words.astype(np.uint32) << np.uint32(16)).view(np.dtype("<f4"))
    return fp32.astype(np.dtype("<f2")).tobytes()


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


def test_fp16_converter_dry_run_against_cached_header() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="fp16")

    assert plan.tensor_count == EXPECTED_TENSOR_COUNT
    assert plan.total_source_bytes == EXPECTED_TOTAL_BYTES
    assert plan.total_output_bytes == EXPECTED_TOTAL_BYTES
    assert plan.qtype_histogram == {"UOCR_TENSOR_F16": EXPECTED_TENSOR_COUNT}
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


def test_write_uocr_model_requires_overwrite_for_existing_output(tmp_path) -> None:
    _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="fp16", strict=False)
    out = tmp_path / "tiny.uocr"

    write_uocr_model(plan, out)
    with pytest.raises(FileExistsError):
        write_uocr_model(plan, out)
    write_uocr_model(plan, out, overwrite=True)
    assert out.exists()
