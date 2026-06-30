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
    UOCR_QTYPE_REASON_FP16_BASELINE,
    UOCR_SECTION_TENSOR_DATA,
    UOCR_TENSOR_DATA_ALIGNMENT,
    UOCR_TENSOR_FLAG_ROW_MAJOR,
    UOCR_TENSOR_PAYLOAD_ALIGNMENT,
    _TENSOR_DIRECTORY_HEADER_STRUCT,
    _TENSOR_ENTRY_STRUCT,
    _tensor_directory_bytes,
    build_dry_run_plan,
    compare_single_tensor_conversion,
    filter_plan_tensors,
    is_preserved_unused_in_normal_ocr,
    main as convert_main,
    write_uocr_model,
)
from unlimitedocr_c.frontend import project_root
from unlimitedocr_c.tensor_registry import TENSOR_ID_LM_HEAD, TENSOR_ID_TOK_EMBED, TensorFamily, TensorProjection, tensor_id_layer_attn


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
    assert entry[25] == UOCR_QTYPE_REASON_FP16_BASELINE
    assert entry[26] == UOCR_PROMOTION_NONE


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


def test_compare_single_tensor_conversion_fp16(tmp_path: Path) -> None:
    _write_tiny_safetensors(tmp_path)
    plan = build_dry_run_plan(tmp_path, qprofile="fp16", strict=False)
    filtered = filter_plan_tensors(plan, "model.sam_model.tiny_a.weight")
    out = tmp_path / "tiny.uocr"
    write_uocr_model(plan, out, overwrite=True)
    result = compare_single_tensor_conversion(filtered, out)
    assert result.matches
    assert result.qtype == "UOCR_TENSOR_F16"


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
