"""Shared fp16 parity thresholds for Metal-vs-Python fixture checks.

These values define the v1 acceptance policy for fp16 parity.  They are kept in
one place so stage tests do not silently drift apart.  Tests may still relax or
tighten numeric feature tolerances with the documented ``<ENV_PREFIX>_*``
environment variables when investigating a new fixture, but exact-id contracts
(router ids and generated ids) are intentionally not relaxed by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FeatureTolerance:
    """Absolute/relative thresholds for fp16 tensor parity.

    The comparison is done after viewing little-endian uint16 fp16 bit buffers as
    float16 and widening to float32.  ``p99_abs`` and ``mean_abs`` keep broad
    drift visible even when a single value is allowed to hit ``max_abs``.
    """

    stage: str
    env_prefix: str
    max_abs: float
    p99_abs: float
    mean_abs: float
    max_rel: float

    def from_env(self, env: Mapping[str, str] | None = None) -> "FeatureTolerance":
        source = env if env is not None else os.environ
        return FeatureTolerance(
            stage=self.stage,
            env_prefix=self.env_prefix,
            max_abs=float(source.get(f"{self.env_prefix}_ATOL", str(self.max_abs))),
            p99_abs=float(source.get(f"{self.env_prefix}_P99_ATOL", str(self.p99_abs))),
            mean_abs=float(source.get(f"{self.env_prefix}_MEAN_ATOL", str(self.mean_abs))),
            max_rel=float(source.get(f"{self.env_prefix}_RTOL", str(self.max_rel))),
        )


@dataclass(frozen=True)
class FeatureComparison:
    stage: str
    shape: tuple[int, ...]
    max_abs: float
    p99_abs: float
    mean_abs: float
    max_rel: float
    tolerance: FeatureTolerance

    @property
    def passed(self) -> bool:
        return (
            self.max_abs <= self.tolerance.max_abs
            and self.p99_abs <= self.tolerance.p99_abs
            and self.mean_abs <= self.tolerance.mean_abs
            and self.max_rel <= self.tolerance.max_rel
        )

    def failure_message(self, label: str | None = None) -> str:
        prefix = label or self.stage
        return (
            f"{prefix} feature mismatch: max_abs={self.max_abs:.6g} "
            f"(limit {self.tolerance.max_abs}), p99_abs={self.p99_abs:.6g} "
            f"(limit {self.tolerance.p99_abs}), mean_abs={self.mean_abs:.6g} "
            f"(limit {self.tolerance.mean_abs}), max_rel={self.max_rel:.6g} "
            f"(limit {self.tolerance.max_rel})"
        )


@dataclass(frozen=True)
class RouterTopKThreshold:
    stage: str = "router_top6"
    require_exact_ids: bool = True
    max_weight_abs: float = 5.0e-5


@dataclass(frozen=True)
class LogitsTopKThreshold:
    stage: str = "logits_topk"
    require_exact_ids: bool = True
    max_score_abs: float = 2.5e-2


@dataclass(frozen=True)
class GeneratedIdsThreshold:
    stage: str = "generated_ids"
    require_exact_ids: bool = True
    require_text_exact: bool = True


# Feature thresholds currently exercised by Python/Metal parity tests.  Env var
# names intentionally match the pre-existing public image parity overrides.
PROMPT_EMBEDDING_FP16 = FeatureTolerance("prompt_embeddings", "UOCR_PROMPT_EMBED", 0.02, 0.005, 0.001, 0.02)
SAM_FEATURE_FP16 = FeatureTolerance("sam_features", "UOCR_SAM_FEATURE", 0.08, 0.02, 0.004, 0.08)
CLIP_FEATURE_FP16 = FeatureTolerance("clip_features", "UOCR_CLIP_FEATURE", 0.12, 0.03, 0.006, 0.10)
PROJECTED_FEATURE_FP16 = FeatureTolerance("projected_features", "UOCR_PROJECTED_FEATURE", 0.14, 0.035, 0.007, 0.12)
FINAL_VISUAL_FEATURE_FP16 = FeatureTolerance("final_visual_features", "UOCR_IMAGE_VISUAL", 0.08, 0.02, 0.004, 0.08)

ROUTER_TOPK_FP16 = RouterTopKThreshold()
LOGITS_TOPK_FP16 = LogitsTopKThreshold()
GENERATED_IDS_FP16 = GeneratedIdsThreshold()

FP16_FEATURE_THRESHOLDS: dict[str, FeatureTolerance] = {
    PROMPT_EMBEDDING_FP16.stage: PROMPT_EMBEDDING_FP16,
    SAM_FEATURE_FP16.stage: SAM_FEATURE_FP16,
    CLIP_FEATURE_FP16.stage: CLIP_FEATURE_FP16,
    PROJECTED_FEATURE_FP16.stage: PROJECTED_FEATURE_FP16,
    FINAL_VISUAL_FEATURE_FP16.stage: FINAL_VISUAL_FEATURE_FP16,
}


def compare_f16_feature_bits(
    actual_bits: NDArray[np.uint16],
    expected_bits: NDArray[np.uint16],
    tolerance: FeatureTolerance,
) -> FeatureComparison:
    """Compare same-shaped fp16 bit tensors under ``tolerance``."""

    actual_bits = np.asarray(actual_bits, dtype=np.dtype("<u2"))
    expected_bits = np.asarray(expected_bits, dtype=np.dtype("<u2"))
    if actual_bits.shape != expected_bits.shape:
        raise ValueError(f"{tolerance.stage} shape mismatch: actual={actual_bits.shape} expected={expected_bits.shape}")
    actual = actual_bits.view(np.float16).astype(np.float32)
    expected = expected_bits.view(np.float16).astype(np.float32)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError(f"{tolerance.stage} features contain non-finite values")
    diff = np.abs(actual - expected)
    ref_scale = np.maximum(np.abs(expected), np.float32(1.0e-3))
    rel = diff / ref_scale
    return FeatureComparison(
        stage=tolerance.stage,
        shape=tuple(int(dim) for dim in actual_bits.shape),
        max_abs=float(np.max(diff)) if diff.size else 0.0,
        p99_abs=float(np.quantile(diff, 0.99)) if diff.size else 0.0,
        mean_abs=float(np.mean(diff)) if diff.size else 0.0,
        max_rel=float(np.max(rel)) if rel.size else 0.0,
        tolerance=tolerance,
    )


def require_f16_feature_close(
    actual_bits: NDArray[np.uint16],
    expected_bits: NDArray[np.uint16],
    tolerance: FeatureTolerance,
    *,
    label: str | None = None,
) -> FeatureComparison:
    comparison = compare_f16_feature_bits(actual_bits, expected_bits, tolerance)
    if not comparison.passed:
        raise AssertionError(comparison.failure_message(label))
    return comparison


def require_router_topk_close(
    actual_ids: NDArray[np.integer],
    expected_ids: NDArray[np.integer],
    actual_weights: NDArray[np.floating],
    expected_weights: NDArray[np.floating],
    threshold: RouterTopKThreshold = ROUTER_TOPK_FP16,
) -> None:
    actual_id_values = np.asarray(actual_ids, dtype=np.uint32)
    expected_id_values = np.asarray(expected_ids, dtype=np.uint32)
    actual_weight_values = np.asarray(actual_weights, dtype=np.float32)
    expected_weight_values = np.asarray(expected_weights, dtype=np.float32)
    if actual_id_values.shape != expected_id_values.shape:
        raise AssertionError(
            f"{threshold.stage} id shape mismatch: actual={actual_id_values.shape} expected={expected_id_values.shape}"
        )
    if actual_weight_values.shape != expected_weight_values.shape or actual_weight_values.shape != actual_id_values.shape:
        raise AssertionError(
            f"{threshold.stage} weight shape mismatch: actual={actual_weight_values.shape} "
            f"expected={expected_weight_values.shape} ids={actual_id_values.shape}"
        )
    if threshold.require_exact_ids and not np.array_equal(actual_id_values, expected_id_values):
        mismatch = np.argwhere(actual_id_values != expected_id_values)[0]
        index = tuple(int(value) for value in mismatch)
        raise AssertionError(
            f"{threshold.stage} id mismatch at {index}: "
            f"actual={int(actual_id_values[index])} expected={int(expected_id_values[index])}"
        )
    diff = np.abs(actual_weight_values - expected_weight_values)
    max_diff = float(np.max(diff)) if diff.size else 0.0
    if max_diff > threshold.max_weight_abs:
        mismatch = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"{threshold.stage} weight mismatch at {tuple(int(value) for value in mismatch)}: "
            f"actual={float(actual_weight_values[mismatch]):.6g} "
            f"expected={float(expected_weight_values[mismatch]):.6g} "
            f"diff={max_diff:.6g} limit={threshold.max_weight_abs}"
        )


def require_logits_topk_close(
    actual_ids: NDArray[np.integer],
    expected_ids: NDArray[np.integer],
    actual_scores: NDArray[np.floating],
    expected_scores: NDArray[np.floating],
    threshold: LogitsTopKThreshold = LOGITS_TOPK_FP16,
) -> None:
    actual_id_values = np.asarray(actual_ids, dtype=np.int32)
    expected_id_values = np.asarray(expected_ids, dtype=np.int32)
    actual_score_values = np.asarray(actual_scores, dtype=np.float32)
    expected_score_values = np.asarray(expected_scores, dtype=np.float32)
    if actual_id_values.shape != expected_id_values.shape:
        raise AssertionError(
            f"{threshold.stage} id shape mismatch: actual={actual_id_values.shape} expected={expected_id_values.shape}"
        )
    if actual_score_values.shape != expected_score_values.shape or actual_score_values.shape != actual_id_values.shape:
        raise AssertionError(
            f"{threshold.stage} score shape mismatch: actual={actual_score_values.shape} "
            f"expected={expected_score_values.shape} ids={actual_id_values.shape}"
        )
    if threshold.require_exact_ids and not np.array_equal(actual_id_values, expected_id_values):
        mismatch = np.argwhere(actual_id_values != expected_id_values)[0]
        index = tuple(int(value) for value in mismatch)
        raise AssertionError(
            f"{threshold.stage} id mismatch at {index}: "
            f"actual={int(actual_id_values[index])} expected={int(expected_id_values[index])}"
        )
    diff = np.abs(actual_score_values - expected_score_values)
    max_diff = float(np.max(diff)) if diff.size else 0.0
    if max_diff > threshold.max_score_abs:
        mismatch = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"{threshold.stage} score mismatch at {tuple(int(value) for value in mismatch)}: "
            f"actual={float(actual_score_values[mismatch]):.6g} "
            f"expected={float(expected_score_values[mismatch]):.6g} "
            f"diff={max_diff:.6g} limit={threshold.max_score_abs}"
        )


def require_exact_generated_ids(
    actual: NDArray[np.integer],
    expected: NDArray[np.integer],
    threshold: GeneratedIdsThreshold = GENERATED_IDS_FP16,
) -> None:
    actual_ids = np.asarray(actual, dtype=np.int32)
    expected_ids = np.asarray(expected, dtype=np.int32)
    if actual_ids.shape != expected_ids.shape:
        raise AssertionError(f"{threshold.stage} shape mismatch: actual={actual_ids.shape} expected={expected_ids.shape}")
    if threshold.require_exact_ids and not np.array_equal(actual_ids, expected_ids):
        raise AssertionError(f"{threshold.stage} mismatch: actual={actual_ids.tolist()} expected={expected_ids.tolist()}")


def require_exact_text(actual: str, expected: str, threshold: GeneratedIdsThreshold = GENERATED_IDS_FP16) -> None:
    if threshold.require_text_exact and actual != expected:
        raise AssertionError(f"generated text mismatch: actual={actual!r} expected={expected!r}")
