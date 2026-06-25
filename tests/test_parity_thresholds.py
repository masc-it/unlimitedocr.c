from __future__ import annotations

import numpy as np
import pytest

from unlimitedocr_c.parity_thresholds import (
    CLIP_FEATURE_FP16,
    FINAL_VISUAL_FEATURE_FP16,
    GENERATED_IDS_FP16,
    LOGITS_TOPK_FP16,
    PROJECTED_FEATURE_FP16,
    PROMPT_EMBEDDING_FP16,
    ROUTER_TOPK_FP16,
    SAM_FEATURE_FP16,
    compare_f16_feature_bits,
    require_exact_generated_ids,
    require_exact_text,
    require_f16_feature_close,
    require_logits_topk_close,
    require_router_topk_close,
)


def test_fp16_thresholds_cover_required_gate_stages() -> None:
    assert PROMPT_EMBEDDING_FP16.stage == "prompt_embeddings"
    assert PROJECTED_FEATURE_FP16.stage == "projected_features"
    assert FINAL_VISUAL_FEATURE_FP16.stage == "final_visual_features"
    assert ROUTER_TOPK_FP16.require_exact_ids is True
    assert ROUTER_TOPK_FP16.max_weight_abs == pytest.approx(5.0e-5)
    assert LOGITS_TOPK_FP16.require_exact_ids is True
    assert LOGITS_TOPK_FP16.max_score_abs == pytest.approx(2.5e-2)
    assert GENERATED_IDS_FP16.require_exact_ids is True
    assert GENERATED_IDS_FP16.require_text_exact is True

    for tolerance in (SAM_FEATURE_FP16, CLIP_FEATURE_FP16, PROJECTED_FEATURE_FP16, FINAL_VISUAL_FEATURE_FP16):
        assert tolerance.max_abs > 0.0
        assert tolerance.p99_abs > 0.0
        assert tolerance.mean_abs > 0.0
        assert tolerance.max_rel > 0.0


def test_feature_tolerance_from_env_preserves_stage_and_overrides_values() -> None:
    overridden = FINAL_VISUAL_FEATURE_FP16.from_env(
        {
            "UOCR_IMAGE_VISUAL_ATOL": "1.25",
            "UOCR_IMAGE_VISUAL_P99_ATOL": "0.75",
            "UOCR_IMAGE_VISUAL_MEAN_ATOL": "0.5",
            "UOCR_IMAGE_VISUAL_RTOL": "2.0",
        }
    )

    assert overridden.stage == FINAL_VISUAL_FEATURE_FP16.stage
    assert overridden.env_prefix == FINAL_VISUAL_FEATURE_FP16.env_prefix
    assert overridden.max_abs == pytest.approx(1.25)
    assert overridden.p99_abs == pytest.approx(0.75)
    assert overridden.mean_abs == pytest.approx(0.5)
    assert overridden.max_rel == pytest.approx(2.0)


def test_compare_f16_feature_bits_reports_metrics_and_passes_close_values() -> None:
    expected = np.asarray([1.0, 2.0, -4.0, 8.0], dtype=np.float16).view(np.dtype("<u2"))
    actual = np.asarray([1.0, 2.001, -4.0, 8.0], dtype=np.float16).view(np.dtype("<u2"))

    comparison = compare_f16_feature_bits(actual, expected, FINAL_VISUAL_FEATURE_FP16)

    assert comparison.shape == (4,)
    assert comparison.max_abs > 0.0
    assert comparison.passed is True
    require_f16_feature_close(actual, expected, FINAL_VISUAL_FEATURE_FP16)


def test_compare_f16_feature_bits_rejects_shape_and_large_drift() -> None:
    expected = np.asarray([0.0, 0.0], dtype=np.float16).view(np.dtype("<u2"))
    actual = np.asarray([1.0, 0.0], dtype=np.float16).view(np.dtype("<u2"))

    comparison = compare_f16_feature_bits(actual, expected, FINAL_VISUAL_FEATURE_FP16)
    assert comparison.passed is False
    assert "feature mismatch" in comparison.failure_message("visual")
    with pytest.raises(AssertionError, match="visual feature mismatch"):
        require_f16_feature_close(actual, expected, FINAL_VISUAL_FEATURE_FP16, label="visual")

    with pytest.raises(ValueError, match="shape mismatch"):
        compare_f16_feature_bits(actual.reshape((2, 1)), expected, FINAL_VISUAL_FEATURE_FP16)


def test_router_and_logits_threshold_helpers_enforce_exact_ids_and_score_bounds() -> None:
    ids = np.asarray([[3, 2, 1, 0, 4, 5]], dtype=np.uint32)
    weights = np.asarray([[0.4, 0.3, 0.2, 0.05, 0.03, 0.02]], dtype=np.float32)
    require_router_topk_close(ids, ids.copy(), weights, weights + np.float32(1.0e-6))

    with pytest.raises(AssertionError, match="router_top6 id mismatch"):
        require_router_topk_close(ids, ids + np.uint32(1), weights, weights)
    with pytest.raises(AssertionError, match="router_top6 weight mismatch"):
        require_router_topk_close(ids, ids, weights + np.float32(1.0e-3), weights)

    logit_ids = np.asarray([9, 8, 7], dtype=np.int32)
    scores = np.asarray([12.0, 11.5, 10.25], dtype=np.float32)
    require_logits_topk_close(logit_ids, logit_ids.copy(), scores, scores + np.float32(1.0e-3))

    with pytest.raises(AssertionError, match="logits_topk id mismatch"):
        require_logits_topk_close(logit_ids, logit_ids + np.int32(1), scores, scores)
    with pytest.raises(AssertionError, match="logits_topk score mismatch"):
        require_logits_topk_close(logit_ids, logit_ids, scores + np.float32(0.1), scores)


def test_generated_id_and_text_thresholds_are_exact() -> None:
    require_exact_generated_ids(np.asarray([1, 2], dtype=np.int32), np.asarray([1, 2], dtype=np.int64))
    require_exact_text("same", "same")

    with pytest.raises(AssertionError, match="generated_ids mismatch"):
        require_exact_generated_ids(np.asarray([1, 3], dtype=np.int32), np.asarray([1, 2], dtype=np.int32))
    with pytest.raises(AssertionError, match="shape mismatch"):
        require_exact_generated_ids(np.asarray([1, 2], dtype=np.int32), np.asarray([1], dtype=np.int32))
    with pytest.raises(AssertionError, match="generated text mismatch"):
        require_exact_text("actual", "expected")
