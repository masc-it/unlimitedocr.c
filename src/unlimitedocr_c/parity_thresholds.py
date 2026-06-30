"""Shared parity thresholds for Metal-vs-Python fixture checks.

These values define the v1 acceptance policy for fp16 parity and the first
quantized parity gates.  They are kept in one place so stage tests do not
silently drift apart.  Tests may still relax or tighten numeric fp16 feature
tolerances with the documented ``<ENV_PREFIX>_*`` environment variables when
investigating a new fixture, but exact-id fp16 contracts are intentionally not
relaxed by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

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


OCR_LAYOUT_MARKERS = ("<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>", "<|grounding|>")


@dataclass(frozen=True)
class QuantRouterTopKThreshold:
    """Acceptance policy for quantized router top-k drift.

    Router weights stay fp16 in q8/q4 profiles, so top-k drift mainly indicates
    hidden-state drift from earlier quantized projections.  We allow rare order
    changes but require very high set overlap because routed experts dominate
    downstream behavior.
    """

    profile: str
    stage: str = "router_top6"
    min_mean_set_overlap: float = 1.0
    min_ordered_id_agreement: float = 1.0
    min_row_exact_agreement: float = 1.0
    max_weight_rmse: float = 0.0
    max_weight_abs: float = 0.0

    def failures(
        self,
        *,
        mean_set_overlap: float,
        ordered_id_agreement: float,
        row_exact_agreement: float,
        weight_rmse: float,
        weight_max_abs: float,
        label: str | None = None,
    ) -> tuple[str, ...]:
        prefix = label or f"{self.profile}:{self.stage}"
        failures: list[str] = []
        if mean_set_overlap < self.min_mean_set_overlap:
            failures.append(
                f"{prefix} mean_set_overlap {mean_set_overlap:.6g} < {self.min_mean_set_overlap:.6g}"
            )
        if ordered_id_agreement < self.min_ordered_id_agreement:
            failures.append(
                f"{prefix} ordered_id_agreement {ordered_id_agreement:.6g} < "
                f"{self.min_ordered_id_agreement:.6g}"
            )
        if row_exact_agreement < self.min_row_exact_agreement:
            failures.append(
                f"{prefix} row_exact_agreement {row_exact_agreement:.6g} < {self.min_row_exact_agreement:.6g}"
            )
        if weight_rmse > self.max_weight_rmse:
            failures.append(f"{prefix} weight_rmse {weight_rmse:.6g} > {self.max_weight_rmse:.6g}")
        if weight_max_abs > self.max_weight_abs:
            failures.append(f"{prefix} weight_max_abs {weight_max_abs:.6g} > {self.max_weight_abs:.6g}")
        return tuple(failures)


@dataclass(frozen=True)
class QuantLogitsTopKThreshold:
    """Acceptance policy for compact top-k next-token logit dumps."""

    profile: str
    stage: str = "logits_topk"
    min_set_overlap: float = 1.0
    min_ordered_id_agreement: float = 1.0
    require_top1_match: bool = True
    max_score_rmse: float = 0.0
    max_score_abs: float = 0.0
    max_truncated_kl: float = 0.0

    def failures(
        self,
        *,
        set_overlap: float,
        ordered_id_agreement: float,
        top1_match: bool,
        score_rmse: float,
        score_max_abs: float,
        truncated_kl: float,
        label: str | None = None,
    ) -> tuple[str, ...]:
        prefix = label or f"{self.profile}:{self.stage}"
        failures: list[str] = []
        if set_overlap < self.min_set_overlap:
            failures.append(f"{prefix} set_overlap {set_overlap:.6g} < {self.min_set_overlap:.6g}")
        if ordered_id_agreement < self.min_ordered_id_agreement:
            failures.append(
                f"{prefix} ordered_id_agreement {ordered_id_agreement:.6g} < "
                f"{self.min_ordered_id_agreement:.6g}"
            )
        if self.require_top1_match and not top1_match:
            failures.append(f"{prefix} top1 token changed")
        if score_rmse > self.max_score_rmse:
            failures.append(f"{prefix} score_rmse {score_rmse:.6g} > {self.max_score_rmse:.6g}")
        if score_max_abs > self.max_score_abs:
            failures.append(f"{prefix} score_max_abs {score_max_abs:.6g} > {self.max_score_abs:.6g}")
        if truncated_kl > self.max_truncated_kl:
            failures.append(f"{prefix} truncated_kl {truncated_kl:.6g} > {self.max_truncated_kl:.6g}")
        return tuple(failures)


@dataclass(frozen=True)
class QuantGeneratedTextThreshold:
    """Acceptance policy for generated ids and decoded OCR text on a corpus."""

    profile: str
    stage: str = "generated_ocr"
    require_exact_ids: bool = False
    min_longest_common_prefix_ratio: float = 1.0
    max_length_delta: int = 0
    require_text_exact: bool = False
    max_char_error_rate: float | None = None
    require_layout_marker_counts: bool = True
    layout_markers: tuple[str, ...] = OCR_LAYOUT_MARKERS

    def failures(
        self,
        *,
        reference_length: int,
        candidate_length: int,
        exact_match: bool,
        longest_common_prefix: int,
        reference_text: str | None = None,
        candidate_text: str | None = None,
        label: str | None = None,
    ) -> tuple[str, ...]:
        prefix = label or f"{self.profile}:{self.stage}"
        failures: list[str] = []
        if self.require_exact_ids and not exact_match:
            failures.append(f"{prefix} generated ids are not exact")
        ref_len = int(reference_length)
        cand_len = int(candidate_length)
        lcp = int(longest_common_prefix)
        if ref_len < 0 or cand_len < 0 or lcp < 0:
            failures.append(f"{prefix} generated length metrics are invalid")
            return tuple(failures)
        lcp_ratio = 1.0 if ref_len == 0 and cand_len == 0 else (float(lcp) / float(max(1, ref_len)))
        if lcp_ratio < self.min_longest_common_prefix_ratio:
            failures.append(
                f"{prefix} longest_common_prefix_ratio {lcp_ratio:.6g} < "
                f"{self.min_longest_common_prefix_ratio:.6g}"
            )
        length_delta = abs(ref_len - cand_len)
        if length_delta > self.max_length_delta:
            failures.append(f"{prefix} length_delta {length_delta} > {self.max_length_delta}")

        has_text = reference_text is not None or candidate_text is not None
        if not has_text:
            return tuple(failures)
        if reference_text is None or candidate_text is None:
            failures.append(f"{prefix} generated text is present for only one side")
            return tuple(failures)
        if self.require_text_exact and reference_text != candidate_text:
            failures.append(f"{prefix} generated text is not exact")
        if self.max_char_error_rate is not None:
            cer = character_error_rate(reference_text, candidate_text)
            if cer > self.max_char_error_rate:
                failures.append(f"{prefix} char_error_rate {cer:.6g} > {self.max_char_error_rate:.6g}")
        if self.require_layout_marker_counts:
            for marker in self.layout_markers:
                expected_count = reference_text.count(marker)
                if expected_count == 0:
                    continue
                actual_count = candidate_text.count(marker)
                if actual_count != expected_count:
                    failures.append(
                        f"{prefix} layout marker {marker!r} count {actual_count} != {expected_count}"
                    )
        return tuple(failures)


@dataclass(frozen=True)
class QuantParityThresholds:
    """Named acceptance profile for q8/q4 Metal parity runs."""

    profile: str
    router: QuantRouterTopKThreshold
    logits: QuantLogitsTopKThreshold
    generated: QuantGeneratedTextThreshold


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

# Initial quantized parity gates.  These are deliberately separate from the
# fp16 exactness contract: q8 should be almost lossless and preserve deterministic
# greedy text on smoke/calibration cases, while q4 is only accepted as a
# calibrated mixed profile with high router/logit agreement and stable OCR text
# structure.
DYN_Q8_PARITY = QuantParityThresholds(
    profile="dyn-q8",
    router=QuantRouterTopKThreshold(
        profile="dyn-q8",
        min_mean_set_overlap=0.995,
        min_ordered_id_agreement=0.98,
        min_row_exact_agreement=0.95,
        max_weight_rmse=2.0e-3,
        max_weight_abs=2.5e-2,
    ),
    logits=QuantLogitsTopKThreshold(
        profile="dyn-q8",
        min_set_overlap=0.95,
        min_ordered_id_agreement=0.80,
        require_top1_match=True,
        max_score_rmse=0.12,
        max_score_abs=0.35,
        max_truncated_kl=3.0e-2,
    ),
    generated=QuantGeneratedTextThreshold(
        profile="dyn-q8",
        require_exact_ids=True,
        min_longest_common_prefix_ratio=1.0,
        max_length_delta=0,
        require_text_exact=True,
        max_char_error_rate=0.0,
    ),
)

DYN_Q4_PARITY = QuantParityThresholds(
    profile="dyn-q4",
    router=QuantRouterTopKThreshold(
        profile="dyn-q4",
        min_mean_set_overlap=0.97,
        min_ordered_id_agreement=0.90,
        min_row_exact_agreement=0.75,
        max_weight_rmse=1.0e-2,
        max_weight_abs=8.0e-2,
    ),
    logits=QuantLogitsTopKThreshold(
        profile="dyn-q4",
        min_set_overlap=0.90,
        min_ordered_id_agreement=0.60,
        require_top1_match=True,
        max_score_rmse=0.25,
        max_score_abs=0.75,
        max_truncated_kl=1.0e-1,
    ),
    generated=QuantGeneratedTextThreshold(
        profile="dyn-q4",
        require_exact_ids=False,
        min_longest_common_prefix_ratio=0.90,
        max_length_delta=8,
        require_text_exact=False,
        max_char_error_rate=2.0e-2,
    ),
)

QUANT_PARITY_THRESHOLDS: dict[str, QuantParityThresholds] = {
    DYN_Q8_PARITY.profile: DYN_Q8_PARITY,
    DYN_Q4_PARITY.profile: DYN_Q4_PARITY,
}

FP16_FEATURE_THRESHOLDS: dict[str, FeatureTolerance] = {
    PROMPT_EMBEDDING_FP16.stage: PROMPT_EMBEDDING_FP16,
    SAM_FEATURE_FP16.stage: SAM_FEATURE_FP16,
    CLIP_FEATURE_FP16.stage: CLIP_FEATURE_FP16,
    PROJECTED_FEATURE_FP16.stage: PROJECTED_FEATURE_FP16,
    FINAL_VISUAL_FEATURE_FP16.stage: FINAL_VISUAL_FEATURE_FP16,
}


def quant_thresholds_for_profile(profile: str) -> QuantParityThresholds:
    """Return the named quantized parity threshold profile."""

    normalized = profile.strip().lower()
    try:
        return QUANT_PARITY_THRESHOLDS[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(QUANT_PARITY_THRESHOLDS))
        raise ValueError(f"unknown quant parity profile {profile!r}; expected one of: {known}") from exc


def infer_quant_thresholds_from_label(label: str) -> QuantParityThresholds | None:
    """Infer a q8/q4 threshold profile from a candidate label, if possible."""

    normalized = label.strip().lower().replace("_", "-")
    if "q4" in normalized or "int4" in normalized:
        return DYN_Q4_PARITY
    if "q8" in normalized or "int8" in normalized or "q8-0" in normalized:
        return DYN_Q8_PARITY
    return None


def levenshtein_distance(left: str, right: str) -> int:
    """Compute edit distance without optional dependencies."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for col, right_char in enumerate(right, start=1):
            insert = current[col - 1] + 1
            delete = previous[col] + 1
            replace = previous[col - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, candidate: str) -> float:
    """Return Levenshtein CER normalized by reference length."""

    if reference == candidate:
        return 0.0
    return float(levenshtein_distance(reference, candidate)) / float(max(1, len(reference)))


def require_quant_generated_text_stable(
    reference_text: str,
    candidate_text: str,
    threshold: QuantGeneratedTextThreshold,
    *,
    label: str | None = None,
) -> None:
    failures = threshold.failures(
        reference_length=len(reference_text),
        candidate_length=len(candidate_text),
        exact_match=reference_text == candidate_text,
        longest_common_prefix=_common_prefix_length(reference_text, candidate_text),
        reference_text=reference_text,
        candidate_text=candidate_text,
        label=label,
    )
    if failures:
        raise AssertionError("; ".join(failures))


def _common_prefix_length(left: Sequence[object], right: Sequence[object]) -> int:
    prefix = 0
    for left_value, right_value in zip(left, right, strict=False):
        if left_value != right_value:
            break
        prefix += 1
    return prefix


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
