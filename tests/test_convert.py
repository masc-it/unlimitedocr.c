from __future__ import annotations

from unlimitedocr_c.convert import (
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_BYTES,
    build_dry_run_plan,
)
from unlimitedocr_c.frontend import project_root


def test_fp16_converter_dry_run_against_cached_header() -> None:
    plan = build_dry_run_plan(project_root() / "data/context", qprofile="fp16")

    assert plan.tensor_count == EXPECTED_TENSOR_COUNT
    assert plan.total_source_bytes == EXPECTED_TOTAL_BYTES
    assert plan.total_output_bytes == EXPECTED_TOTAL_BYTES
    assert plan.qtype_histogram == {"UOCR_TENSOR_F16": EXPECTED_TENSOR_COUNT}
    assert plan.usage_histogram["runtime"] + plan.usage_histogram["preserved-unused"] == EXPECTED_TENSOR_COUNT

    lm_head = plan.tensor_by_name("lm_head.weight")
    assert lm_head.shape == (129_280, 1280)
    assert lm_head.source_dtype == "BF16"
    assert lm_head.output_dtype == "F16"

    router = plan.tensor_by_name("model.layers.1.mlp.gate.weight")
    assert router.family == "moe_router"
    assert router.qtype == "UOCR_TENSOR_F16"
    assert "router" in router.reason.lower()

    clip_patch = plan.tensor_by_name("model.vision_model.embeddings.patch_embedding.weight")
    assert clip_patch.usage == "preserved-unused"
    assert "patch_embeds" in clip_patch.reason
