from __future__ import annotations

import os

import numpy as np
import pytest

from unlimitedocr_c.ffi import MemoryReport, find_library_path
from unlimitedocr_c.frontend import MODEL_VOCAB_SIZE, prepare_image
from unlimitedocr_c.perf import (
    build_perf_smoke_result,
    deterministic_perf_image,
    generation_tokens_per_second,
    run_public_metal_perf_smoke,
)


def native_library_available() -> bool:
    try:
        find_library_path()
    except FileNotFoundError:
        return False
    return True


def _perf_test_enabled() -> bool:
    return os.environ.get("UOCR_RUN_PERF_TESTS") == "1" and bool(os.environ.get("UOCR_MODEL_PATH"))


def _dummy_memory_report() -> MemoryReport:
    return MemoryReport(
        category_live_bytes=(1, 2, 3, 4, 5, 6, 7, 8),
        category_peak_bytes=(8, 7, 6, 5, 4, 3, 2, 1),
        total_live_bytes=1234,
        total_peak_bytes=5678,
        estimated_model_views_bytes=10,
        estimated_kv_cache_bytes=20,
        estimated_prompt_embeddings_bytes=30,
        estimated_vision_scratch_bytes=40,
        estimated_decoder_scratch_bytes=50,
        estimated_moe_scratch_bytes=60,
        estimated_logits_readback_bytes=70,
        estimated_transient_bytes=80,
        estimated_safety_margin_bytes=90,
        estimated_total_bytes=1000,
        memory_budget_bytes=2000,
        recommended_working_set_bytes=1500,
    )


def test_perf_image_and_tokens_per_second_helpers_are_deterministic() -> None:
    image_a = deterministic_perf_image(12, 8, phase=3)
    image_b = deterministic_perf_image(12, 8, phase=3)
    assert image_a.size == (12, 8)
    assert np.array_equal(np.asarray(image_a), np.asarray(image_b))
    assert generation_tokens_per_second(4, 2.0) == pytest.approx(2.0)
    assert generation_tokens_per_second(0, 2.0) == 0.0
    assert generation_tokens_per_second(1, 0.0) == float("inf")


def test_build_perf_smoke_result_reports_timings_memory_and_ids() -> None:
    request = prepare_image(deterministic_perf_image(), preset="base", max_new_tokens=2)
    result = build_perf_smoke_result(
        backend="metal",
        preset="base",
        request=request,
        warmup_runs=1,
        model_open_seconds=1.25,
        warmup_seconds=(0.5,),
        first_token_seconds=0.75,
        generation_seconds=2.0,
        generated=np.asarray([128818, 1], dtype=np.int32),
        decoded_text="<|det|>",
        memory=_dummy_memory_report(),
    )

    assert result.request_tokens == request.n_tokens
    assert result.visual_tokens == request.expected_visual_tokens
    assert result.views == len(request.views)
    assert result.generation_tokens_per_second == pytest.approx(1.0)
    assert result.peak_memory_bytes == 5678
    assert result.generated_ids == (128818, 1)
    assert result.as_dict()["generated_ids"] == [128818, 1]
    assert "tok/s=1.000" in result.summary()


@pytest.mark.skipif(not native_library_available(), reason="libunlimitedocr is not built")
@pytest.mark.skipif(
    not _perf_test_enabled(),
    reason="set UOCR_RUN_PERF_TESTS=1 and UOCR_MODEL_PATH to run the full-model Metal perf smoke test",
)
def test_public_metal_perf_smoke() -> None:
    decode_tokens = int(os.environ.get("UOCR_PERF_DECODE_TOKENS", "2"))
    warmup_runs = int(os.environ.get("UOCR_PERF_WARMUP_RUNS", "1"))
    result = run_public_metal_perf_smoke(
        model_path=os.environ["UOCR_MODEL_PATH"],
        decode_tokens=decode_tokens,
        warmup_runs=warmup_runs,
    )

    assert result.model_open_seconds > 0.0
    assert len(result.warmup_seconds) == warmup_runs
    assert result.first_token_seconds > 0.0
    assert result.generation_seconds > 0.0
    assert result.generated_tokens >= 1
    assert result.generation_tokens_per_second > 0.0
    assert result.peak_memory_bytes > 0
    assert result.request_tokens > result.visual_tokens
    assert all(0 <= token_id < MODEL_VOCAB_SIZE for token_id in result.generated_ids)
    print("Metal perf smoke:", result.summary())
