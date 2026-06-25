"""Calibration report helpers for dynamic q8/q4 promotion work.

The native Metal kernels emit/consume compact parity fixtures during bring-up.
This module turns those fp16-vs-candidate fixture directories into a calibration
report that can drive later monotonic promotion decisions.  It intentionally
works from already-generated dumps so it is useful on machines without enough
memory to run several full models concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import sys

import numpy as np
from numpy.typing import NDArray

from .drift import CandidateDriftReport, compare_candidate_layer_drift
from .parity_thresholds import (
    QUANT_PARITY_THRESHOLDS,
    QuantParityThresholds,
    infer_quant_thresholds_from_label,
    quant_thresholds_for_profile,
)
from .golden import (
    GENERATED_IDS_BIN,
    GENERATED_TEXT_TXT,
    LOGITS_TOPK_IDS_BIN,
    LOGITS_TOPK_SCORES_BIN,
    MOE_EXPERTS,
    MOE_TOP_K,
    TEXT_DECODER_LAYER_COUNT,
    router_top_ids_filename,
)


@dataclass(frozen=True)
class CalibrationFixtureCase:
    """One fp16 reference fixture and one or more candidate dump directories."""

    case_id: str
    fp16_dir: str
    candidates: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fp16_dir": self.fp16_dir,
            "candidates": dict(self.candidates),
        }


@dataclass(frozen=True)
class LogitsTopKMetrics:
    """Top-k next-token logit agreement for one candidate."""

    top_k: int
    intersection_count: int
    set_overlap: float
    ordered_id_agreement: float
    top1_match: bool
    score_rmse_by_rank: float
    score_max_abs_by_rank: float
    truncated_kl_ref_to_candidate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "intersection_count": self.intersection_count,
            "set_overlap": self.set_overlap,
            "ordered_id_agreement": self.ordered_id_agreement,
            "top1_match": self.top1_match,
            "score_rmse_by_rank": self.score_rmse_by_rank,
            "score_max_abs_by_rank": self.score_max_abs_by_rank,
            "truncated_kl_ref_to_candidate": self.truncated_kl_ref_to_candidate,
        }


@dataclass(frozen=True)
class GeneratedIdsMetrics:
    """Generated-id and optional decoded-text agreement for one candidate."""

    reference_length: int
    candidate_length: int
    exact_match: bool
    longest_common_prefix: int
    reference_text: str | None = None
    candidate_text: str | None = None
    text_exact_match: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "reference_length": self.reference_length,
            "candidate_length": self.candidate_length,
            "exact_match": self.exact_match,
            "longest_common_prefix": self.longest_common_prefix,
        }
        if self.reference_text is not None or self.candidate_text is not None:
            data["reference_text"] = self.reference_text
            data["candidate_text"] = self.candidate_text
            data["text_exact_match"] = self.text_exact_match
        return data


@dataclass(frozen=True)
class ExpertTrafficMetrics:
    """Selected-expert traffic/frequency drift for one MoE layer."""

    layer: int
    total_selections: int
    reference_active_experts: int
    candidate_active_experts: int
    total_variation_distance: float
    max_abs_frequency_delta: float
    top_reference_experts: tuple[dict[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "total_selections": self.total_selections,
            "reference_active_experts": self.reference_active_experts,
            "candidate_active_experts": self.candidate_active_experts,
            "total_variation_distance": self.total_variation_distance,
            "max_abs_frequency_delta": self.max_abs_frequency_delta,
            "top_reference_experts": list(self.top_reference_experts),
        }


@dataclass(frozen=True)
class CalibrationCandidateReport:
    """All metrics for one candidate qprofile/model on one fixture case."""

    label: str
    candidate_dir: str
    layer_drift: CandidateDriftReport
    logits_topk: LogitsTopKMetrics | None
    generated_ids: GeneratedIdsMetrics | None
    expert_traffic: tuple[ExpertTrafficMetrics, ...]

    @property
    def max_layer_rmse(self) -> float:
        return self.layer_drift.max_rmse

    @property
    def max_layer_abs(self) -> float:
        return self.layer_drift.max_abs

    @property
    def min_layer_cosine_similarity(self) -> float:
        return self.layer_drift.min_cosine_similarity

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "candidate_dir": self.candidate_dir,
            "summary": {
                "max_layer_rmse": self.max_layer_rmse,
                "max_layer_abs": self.max_layer_abs,
                "min_layer_cosine_similarity": self.min_layer_cosine_similarity,
                "router_layers": len(self.layer_drift.router_metrics),
                "expert_traffic_layers": len(self.expert_traffic),
                "has_logits_topk": self.logits_topk is not None,
                "has_generated_ids": self.generated_ids is not None,
            },
            "layer_drift": self.layer_drift.as_dict(),
            "logits_topk": self.logits_topk.as_dict() if self.logits_topk is not None else None,
            "generated_ids": self.generated_ids.as_dict() if self.generated_ids is not None else None,
            "expert_traffic": [metric.as_dict() for metric in self.expert_traffic],
        }


@dataclass(frozen=True)
class CalibrationCaseReport:
    """Calibration report for one representative fixture case."""

    case_id: str
    fp16_dir: str
    candidates: tuple[CalibrationCandidateReport, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fp16_dir": self.fp16_dir,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Full corpus calibration report."""

    corpus_id: str
    cases: tuple[CalibrationCaseReport, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "corpus_id": self.corpus_id,
            "cases": [case.as_dict() for case in self.cases],
        }


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {manifest_path}")
    return data


def _tensor_entry(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    for section_name in ("candidate_tensors", "golden_tensors"):
        section = manifest.get(section_name, {})
        if isinstance(section, Mapping):
            entry = section.get(key)
            if isinstance(entry, Mapping):
                return entry
    return None


def _read_logits_topk(root: Path) -> tuple[NDArray[np.int32], NDArray[np.float32]] | None:
    manifest = _read_manifest(root)
    entry = _tensor_entry(manifest, "logits_topk")
    if entry is None:
        ids_path = root / LOGITS_TOPK_IDS_BIN
        scores_path = root / LOGITS_TOPK_SCORES_BIN
        if not ids_path.exists() or not scores_path.exists():
            return None
        ids = np.fromfile(ids_path, dtype=np.dtype("<i4"))
        scores = np.fromfile(scores_path, dtype=np.dtype("<f4"))
    else:
        ids_file = entry.get("ids_file", LOGITS_TOPK_IDS_BIN)
        scores_file = entry.get("scores_file", LOGITS_TOPK_SCORES_BIN)
        if not isinstance(ids_file, str) or not isinstance(scores_file, str):
            raise ValueError(f"invalid logits_topk filenames in {root / 'manifest.json'}")
        ids = np.fromfile(root / ids_file, dtype=np.dtype("<i4"))
        scores = np.fromfile(root / scores_file, dtype=np.dtype("<f4"))
        shape = entry.get("shape")
        if isinstance(shape, list) and len(shape) == 1 and ids.size != int(shape[0]):
            raise ValueError(f"logits_topk shape metadata mismatch in {root}")
    if ids.ndim != 1 or scores.ndim != 1 or ids.size == 0 or ids.size != scores.size:
        raise ValueError(f"invalid logits_topk dump sizes in {root}: ids={ids.size} scores={scores.size}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite logits_topk scores in {root}")
    return ids.astype(np.int32, copy=False), scores.astype(np.float32, copy=False)


def _softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def logits_topk_metrics(
    reference_ids: NDArray[np.int32],
    reference_scores: NDArray[np.float32],
    candidate_ids: NDArray[np.int32],
    candidate_scores: NDArray[np.float32],
) -> LogitsTopKMetrics:
    ref_ids = np.asarray(reference_ids, dtype=np.int32)
    cand_ids = np.asarray(candidate_ids, dtype=np.int32)
    ref_scores = np.asarray(reference_scores, dtype=np.float32)
    cand_scores = np.asarray(candidate_scores, dtype=np.float32)
    if ref_ids.ndim != 1 or cand_ids.ndim != 1 or ref_scores.shape != ref_ids.shape or cand_scores.shape != cand_ids.shape:
        raise ValueError("logits top-k ids/scores must be same-shaped rank-1 arrays")
    if ref_ids.size == 0 or cand_ids.size == 0:
        raise ValueError("logits top-k arrays must be non-empty")

    compared = min(int(ref_ids.size), int(cand_ids.size))
    ref_set = {int(v) for v in ref_ids}
    cand_set = {int(v) for v in cand_ids}
    intersection = len(ref_set & cand_set)
    ordered = float(np.mean(ref_ids[:compared] == cand_ids[:compared])) if compared else 0.0
    score_diff = cand_scores[:compared].astype(np.float64) - ref_scores[:compared].astype(np.float64)
    score_rmse = float(np.sqrt(np.mean(score_diff * score_diff))) if compared else 0.0
    score_max_abs = float(np.max(np.abs(score_diff))) if compared else 0.0

    ref_by_id = {int(token_id): float(score) for token_id, score in zip(ref_ids, ref_scores, strict=True)}
    cand_by_id = {int(token_id): float(score) for token_id, score in zip(cand_ids, cand_scores, strict=True)}
    union_ids = sorted(ref_set | cand_set)
    # The full vocabulary logits are not present in compact fixtures.  Place
    # missing entries well below the observed compact top-k tail and report this
    # explicitly as a truncated top-k KL, not a full-vocabulary KL.
    ref_floor = min(ref_by_id.values()) - 20.0
    cand_floor = min(cand_by_id.values()) - 20.0
    ref_vec = np.asarray([ref_by_id.get(token_id, ref_floor) for token_id in union_ids], dtype=np.float64)
    cand_vec = np.asarray([cand_by_id.get(token_id, cand_floor) for token_id in union_ids], dtype=np.float64)
    ref_prob = _softmax(ref_vec)
    cand_prob = _softmax(cand_vec)
    kl = float(np.sum(ref_prob * (np.log(ref_prob) - np.log(cand_prob))))

    return LogitsTopKMetrics(
        top_k=int(ref_ids.size),
        intersection_count=intersection,
        set_overlap=intersection / float(ref_ids.size),
        ordered_id_agreement=ordered,
        top1_match=bool(ref_ids[0] == cand_ids[0]),
        score_rmse_by_rank=score_rmse,
        score_max_abs_by_rank=score_max_abs,
        truncated_kl_ref_to_candidate=kl,
    )


def _read_generated(root: Path) -> tuple[NDArray[np.int32], str | None] | None:
    manifest = _read_manifest(root)
    entry = _tensor_entry(manifest, "generated_ids")
    if entry is None:
        ids_path = root / GENERATED_IDS_BIN
        if not ids_path.exists():
            return None
    else:
        ids_file = entry.get("file", GENERATED_IDS_BIN)
        if not isinstance(ids_file, str):
            raise ValueError(f"invalid generated_ids filename in {root / 'manifest.json'}")
        ids_path = root / ids_file
    ids = np.fromfile(ids_path, dtype=np.dtype("<i4"))
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError(f"invalid generated_ids dump in {root}")
    if entry is not None:
        shape = entry.get("shape")
        if isinstance(shape, list) and len(shape) == 1 and ids.size != int(shape[0]):
            raise ValueError(f"generated_ids shape metadata mismatch in {root}")

    text: str | None = None
    text_entry = _tensor_entry(manifest, "generated_text")
    text_file: str | None = None
    if isinstance(text_entry, Mapping):
        candidate = text_entry.get("file", GENERATED_TEXT_TXT)
        if isinstance(candidate, str):
            text_file = candidate
    elif (root / GENERATED_TEXT_TXT).exists():
        text_file = GENERATED_TEXT_TXT
    if text_file is not None and (root / text_file).exists():
        text = (root / text_file).read_text(encoding="utf-8")
    return ids.astype(np.int32, copy=False), text


def generated_ids_metrics(
    reference_ids: NDArray[np.int32],
    candidate_ids: NDArray[np.int32],
    *,
    reference_text: str | None = None,
    candidate_text: str | None = None,
) -> GeneratedIdsMetrics:
    ref = np.asarray(reference_ids, dtype=np.int32)
    cand = np.asarray(candidate_ids, dtype=np.int32)
    if ref.ndim != 1 or cand.ndim != 1:
        raise ValueError("generated ids must be rank-1 arrays")
    prefix = 0
    for left, right in zip(ref, cand, strict=False):
        if int(left) != int(right):
            break
        prefix += 1
    text_exact = None
    if reference_text is not None or candidate_text is not None:
        text_exact = reference_text == candidate_text
    return GeneratedIdsMetrics(
        reference_length=int(ref.size),
        candidate_length=int(cand.size),
        exact_match=bool(np.array_equal(ref, cand)),
        longest_common_prefix=prefix,
        reference_text=reference_text,
        candidate_text=candidate_text,
        text_exact_match=text_exact,
    )


def _router_ids_by_layer(root: Path) -> dict[int, NDArray[np.uint32]]:
    manifest = _read_manifest(root)
    router_manifest = manifest.get("router_topk", {})
    result: dict[int, NDArray[np.uint32]] = {}
    if isinstance(router_manifest, Mapping):
        for key, value in router_manifest.items():
            if not isinstance(key, str) or not key.startswith("layer_") or not isinstance(value, Mapping):
                continue
            try:
                layer = int(key.split("_", 1)[1])
            except ValueError:
                continue
            ids_file = value.get("ids_file", router_top_ids_filename(layer))
            if not isinstance(ids_file, str):
                raise ValueError(f"invalid router ids filename for {key} in {root / 'manifest.json'}")
            ids_path = root / ids_file
            if not ids_path.exists():
                continue
            ids = np.fromfile(ids_path, dtype=np.dtype("<u4"))
            shape = value.get("shape")
            if isinstance(shape, list) and len(shape) == 2:
                rows = int(shape[0])
                top_k = int(shape[1])
                if top_k != MOE_TOP_K or ids.size != rows * top_k:
                    raise ValueError(f"router ids shape metadata mismatch for layer {layer} in {root}")
                ids = ids.reshape((rows, top_k))
            elif ids.size % MOE_TOP_K == 0:
                ids = ids.reshape((-1, MOE_TOP_K))
            else:
                raise ValueError(f"router ids size is not divisible by top-k for layer {layer} in {root}")
            result[layer] = ids.astype(np.uint32, copy=False)

    if result:
        return result

    for layer in range(1, TEXT_DECODER_LAYER_COUNT):
        try:
            path = root / router_top_ids_filename(layer)
        except ValueError:
            continue
        if not path.exists():
            continue
        ids = np.fromfile(path, dtype=np.dtype("<u4"))
        if ids.size == 0 or ids.size % MOE_TOP_K != 0:
            raise ValueError(f"router ids size is not divisible by top-k for layer {layer} in {root}")
        result[layer] = ids.reshape((-1, MOE_TOP_K)).astype(np.uint32, copy=False)
    return result


def expert_traffic_metrics(
    layer: int,
    reference_ids: NDArray[np.uint32],
    candidate_ids: NDArray[np.uint32],
    *,
    top_n: int = 8,
) -> ExpertTrafficMetrics:
    ref = np.asarray(reference_ids, dtype=np.uint32)
    cand = np.asarray(candidate_ids, dtype=np.uint32)
    if ref.shape != cand.shape or ref.ndim != 2 or ref.shape[1] != MOE_TOP_K:
        raise ValueError(f"router top-id shape mismatch for layer {layer}: reference={ref.shape} candidate={cand.shape}")
    if np.any(ref >= MOE_EXPERTS) or np.any(cand >= MOE_EXPERTS):
        raise ValueError(f"router ids for layer {layer} contain values outside [0,{MOE_EXPERTS})")
    ref_counts = np.bincount(ref.reshape(-1), minlength=MOE_EXPERTS).astype(np.int64, copy=False)
    cand_counts = np.bincount(cand.reshape(-1), minlength=MOE_EXPERTS).astype(np.int64, copy=False)
    total = int(ref_counts.sum())
    if total <= 0 or int(cand_counts.sum()) != total:
        raise ValueError(f"router selection count mismatch for layer {layer}")
    ref_freq = ref_counts.astype(np.float64) / float(total)
    cand_freq = cand_counts.astype(np.float64) / float(total)
    delta = np.abs(cand_freq - ref_freq)
    ordered = np.lexsort((np.arange(MOE_EXPERTS, dtype=np.int64), -ref_counts))[: max(0, int(top_n))]
    top_reference_experts = tuple(
        {
            "expert": int(expert),
            "reference_count": int(ref_counts[expert]),
            "candidate_count": int(cand_counts[expert]),
        }
        for expert in ordered
        if ref_counts[expert] > 0 or cand_counts[expert] > 0
    )
    return ExpertTrafficMetrics(
        layer=layer,
        total_selections=total,
        reference_active_experts=int(np.count_nonzero(ref_counts)),
        candidate_active_experts=int(np.count_nonzero(cand_counts)),
        total_variation_distance=float(0.5 * np.sum(delta)),
        max_abs_frequency_delta=float(np.max(delta)),
        top_reference_experts=top_reference_experts,
    )


def _compare_optional_logits(reference_dir: Path, candidate_dir: Path) -> LogitsTopKMetrics | None:
    ref = _read_logits_topk(reference_dir)
    cand = _read_logits_topk(candidate_dir)
    if ref is None or cand is None:
        return None
    return logits_topk_metrics(ref[0], ref[1], cand[0], cand[1])


def _compare_optional_generated(reference_dir: Path, candidate_dir: Path) -> GeneratedIdsMetrics | None:
    ref = _read_generated(reference_dir)
    cand = _read_generated(candidate_dir)
    if ref is None or cand is None:
        return None
    return generated_ids_metrics(ref[0], cand[0], reference_text=ref[1], candidate_text=cand[1])


def _compare_optional_expert_traffic(reference_dir: Path, candidate_dir: Path) -> tuple[ExpertTrafficMetrics, ...]:
    ref_layers = _router_ids_by_layer(reference_dir)
    cand_layers = _router_ids_by_layer(candidate_dir)
    metrics: list[ExpertTrafficMetrics] = []
    for layer in sorted(set(ref_layers) & set(cand_layers)):
        metrics.append(expert_traffic_metrics(layer, ref_layers[layer], cand_layers[layer]))
    return tuple(metrics)


def calibrate_case(
    case: CalibrationFixtureCase,
    *,
    layer_count: int | None = None,
    include_router: bool = True,
) -> CalibrationCaseReport:
    """Compare all candidate dump directories for one calibration fixture case."""

    if not case.candidates:
        raise ValueError(f"calibration case {case.case_id!r} has no candidates")
    fp16_dir = Path(case.fp16_dir)
    candidates: list[CalibrationCandidateReport] = []
    for label, candidate_dir_text in case.candidates.items():
        candidate_dir = Path(candidate_dir_text)
        layer_drift = compare_candidate_layer_drift(
            fp16_dir,
            candidate_dir,
            label=label,
            layer_count=layer_count,
            include_router=include_router,
        )
        candidates.append(
            CalibrationCandidateReport(
                label=label,
                candidate_dir=str(candidate_dir),
                layer_drift=layer_drift,
                logits_topk=_compare_optional_logits(fp16_dir, candidate_dir),
                generated_ids=_compare_optional_generated(fp16_dir, candidate_dir),
                expert_traffic=_compare_optional_expert_traffic(fp16_dir, candidate_dir) if include_router else (),
            )
        )
    return CalibrationCaseReport(case_id=case.case_id, fp16_dir=str(fp16_dir), candidates=tuple(candidates))


def run_calibration(
    cases: Sequence[CalibrationFixtureCase],
    *,
    corpus_id: str = "ad-hoc",
    layer_count: int | None = None,
    include_router: bool = True,
) -> CalibrationReport:
    """Run fixture-based calibration comparisons for a corpus."""

    if not cases:
        raise ValueError("at least one calibration fixture case is required")
    return CalibrationReport(
        corpus_id=corpus_id,
        cases=tuple(calibrate_case(case, layer_count=layer_count, include_router=include_router) for case in cases),
    )


def _resolve_manifest_path(base: Path, value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return str(path)


def load_calibration_corpus(path: str | Path) -> tuple[str, tuple[CalibrationFixtureCase, ...]]:
    """Load a calibration corpus manifest.

    Expected shape::

        {
          "version": 1,
          "corpus_id": "ocr-smoke",
          "fixtures": [
            {
              "id": "dense-text-page",
              "fp16_dir": "fp16/dense-text-page",
              "candidates": {"dyn-q8": "dyn-q8/dense-text-page"}
            }
          ]
        }
    """

    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {manifest_path}")
    corpus_id_value = data.get("corpus_id", manifest_path.stem)
    if not isinstance(corpus_id_value, str) or not corpus_id_value:
        raise ValueError("corpus_id must be a non-empty string")
    entries = data.get("fixtures", data.get("cases"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path} must contain a non-empty fixtures list")
    cases: list[CalibrationFixtureCase] = []
    base = manifest_path.parent
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"fixture entry {index} in {manifest_path} must be an object")
        case_id = entry.get("id", entry.get("case_id"))
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"fixture entry {index} has no non-empty id")
        fp16_dir = _resolve_manifest_path(base, entry.get("fp16_dir"), field=f"fixtures[{index}].fp16_dir")
        raw_candidates = entry.get("candidates")
        candidates: dict[str, str] = {}
        if isinstance(raw_candidates, Mapping):
            for label, candidate_path in raw_candidates.items():
                if not isinstance(label, str) or not label:
                    raise ValueError(f"fixture {case_id} has an empty candidate label")
                candidates[label] = _resolve_manifest_path(base, candidate_path, field=f"fixtures[{index}].candidates.{label}")
        elif isinstance(raw_candidates, list):
            for candidate_index, candidate in enumerate(raw_candidates):
                if not isinstance(candidate, Mapping):
                    raise ValueError(f"fixture {case_id} candidate {candidate_index} must be an object")
                label = candidate.get("label")
                if not isinstance(label, str) or not label:
                    raise ValueError(f"fixture {case_id} candidate {candidate_index} has no label")
                candidates[label] = _resolve_manifest_path(
                    base,
                    candidate.get("dir"),
                    field=f"fixtures[{index}].candidates[{candidate_index}].dir",
                )
        else:
            raise ValueError(f"fixture {case_id} must define candidates as an object or list")
        cases.append(CalibrationFixtureCase(case_id=case_id, fp16_dir=fp16_dir, candidates=candidates))
    return corpus_id_value, tuple(cases)


def _parse_candidate(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must have form LABEL=DIR")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    if not path:
        raise argparse.ArgumentTypeError("candidate directory cannot be empty")
    return label, path


def _format_summary(report: CalibrationReport) -> str:
    lines = [f"calibration corpus: {report.corpus_id}"]
    for case in report.cases:
        lines.append(f"case {case.case_id}: fp16={case.fp16_dir}")
        for candidate in case.candidates:
            lines.append(
                f"  {candidate.label}: max_layer_rmse={candidate.max_layer_rmse:.6g} "
                f"max_layer_abs={candidate.max_layer_abs:.6g} "
                f"min_layer_cosine={candidate.min_layer_cosine_similarity:.9f}"
            )
            if candidate.logits_topk is not None:
                logits = candidate.logits_topk
                lines.append(
                    f"    logits_topk: overlap={logits.set_overlap:.6g} ordered={logits.ordered_id_agreement:.6g} "
                    f"top1_match={int(logits.top1_match)} truncated_kl={logits.truncated_kl_ref_to_candidate:.6g}"
                )
            if candidate.generated_ids is not None:
                generated = candidate.generated_ids
                lines.append(
                    f"    generated_ids: exact={int(generated.exact_match)} "
                    f"lcp={generated.longest_common_prefix}/{generated.reference_length} "
                    f"candidate_len={generated.candidate_length}"
                )
            for traffic in candidate.expert_traffic:
                lines.append(
                    f"    layer_{traffic.layer}_expert_traffic: tvd={traffic.total_variation_distance:.6g} "
                    f"max_freq_delta={traffic.max_abs_frequency_delta:.6g} "
                    f"active={traffic.reference_active_experts}->{traffic.candidate_active_experts}"
                )
    return "\n".join(lines)


def _resolve_quant_thresholds(profile: str, candidate_label: str) -> QuantParityThresholds | None:
    if profile == "auto":
        return infer_quant_thresholds_from_label(candidate_label)
    return quant_thresholds_for_profile(profile)


def _threshold_failures(
    report: CalibrationReport,
    *,
    max_layer_rmse: float | None,
    max_layer_abs: float | None,
    min_layer_cosine: float | None,
    min_router_set_overlap: float | None,
    min_logits_set_overlap: float | None,
    require_generated_match: bool,
    quant_threshold_profile: str | None,
) -> list[str]:
    failures: list[str] = []
    for case in report.cases:
        for candidate in case.candidates:
            prefix = f"{case.case_id}:{candidate.label}"
            for metric in candidate.layer_drift.layer_metrics:
                metric_prefix = f"{prefix}:{metric.name}"
                if max_layer_rmse is not None and metric.rmse > max_layer_rmse:
                    failures.append(f"{metric_prefix} rmse {metric.rmse:.6g} > {max_layer_rmse:.6g}")
                if max_layer_abs is not None and metric.max_abs > max_layer_abs:
                    failures.append(f"{metric_prefix} max_abs {metric.max_abs:.6g} > {max_layer_abs:.6g}")
                if min_layer_cosine is not None and metric.cosine_similarity < min_layer_cosine:
                    failures.append(
                        f"{metric_prefix} cosine {metric.cosine_similarity:.9f} < {min_layer_cosine:.9f}"
                    )
            if min_router_set_overlap is not None:
                for router in candidate.layer_drift.router_metrics:
                    if router.mean_set_overlap < min_router_set_overlap:
                        failures.append(
                            f"{prefix}:layer_{router.layer}_router set_overlap "
                            f"{router.mean_set_overlap:.6g} < {min_router_set_overlap:.6g}"
                        )
            if min_logits_set_overlap is not None:
                if candidate.logits_topk is None:
                    failures.append(f"{prefix}: missing logits_topk metrics")
                elif candidate.logits_topk.set_overlap < min_logits_set_overlap:
                    failures.append(
                        f"{prefix}:logits_topk overlap {candidate.logits_topk.set_overlap:.6g} < "
                        f"{min_logits_set_overlap:.6g}"
                    )
            if require_generated_match:
                if candidate.generated_ids is None:
                    failures.append(f"{prefix}: missing generated_ids metrics")
                elif not candidate.generated_ids.exact_match:
                    failures.append(
                        f"{prefix}: generated ids differ at lcp "
                        f"{candidate.generated_ids.longest_common_prefix}/{candidate.generated_ids.reference_length}"
                    )
            if quant_threshold_profile is not None:
                thresholds = _resolve_quant_thresholds(quant_threshold_profile, candidate.label)
                if thresholds is None:
                    failures.append(f"{prefix}: cannot infer q8/q4 threshold profile from label {candidate.label!r}")
                    continue
                if not candidate.layer_drift.router_metrics:
                    failures.append(f"{prefix}:{thresholds.profile}: missing router top-k metrics")
                for router in candidate.layer_drift.router_metrics:
                    failures.extend(
                        thresholds.router.failures(
                            mean_set_overlap=router.mean_set_overlap,
                            ordered_id_agreement=router.ordered_id_agreement,
                            row_exact_agreement=router.row_exact_agreement,
                            weight_rmse=router.weight_rmse,
                            weight_max_abs=router.weight_max_abs,
                            label=f"{prefix}:{thresholds.profile}:layer_{router.layer}_router_topk",
                        )
                    )
                if candidate.logits_topk is None:
                    failures.append(f"{prefix}:{thresholds.profile}: missing logits_topk metrics")
                else:
                    logits = candidate.logits_topk
                    failures.extend(
                        thresholds.logits.failures(
                            set_overlap=logits.set_overlap,
                            ordered_id_agreement=logits.ordered_id_agreement,
                            top1_match=logits.top1_match,
                            score_rmse=logits.score_rmse_by_rank,
                            score_max_abs=logits.score_max_abs_by_rank,
                            truncated_kl=logits.truncated_kl_ref_to_candidate,
                            label=f"{prefix}:{thresholds.profile}:logits_topk",
                        )
                    )
                if candidate.generated_ids is None:
                    failures.append(f"{prefix}:{thresholds.profile}: missing generated_ids metrics")
                else:
                    generated = candidate.generated_ids
                    failures.extend(
                        thresholds.generated.failures(
                            reference_length=generated.reference_length,
                            candidate_length=generated.candidate_length,
                            exact_match=generated.exact_match,
                            longest_common_prefix=generated.longest_common_prefix,
                            reference_text=generated.reference_text,
                            candidate_text=generated.candidate_text,
                            label=f"{prefix}:{thresholds.profile}:generated_ocr",
                        )
                    )
    return failures


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixture-based q8/q4 calibration comparisons")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, help="calibration corpus manifest JSON")
    source.add_argument("--fp16-dir", type=Path, help="single fp16 reference fixture directory")
    parser.add_argument("--case-id", default=None, help="case id for direct --fp16-dir mode")
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        metavar="LABEL=DIR",
        help="candidate fixture directory; repeat for dyn-q8/dyn-q4 (direct mode only)",
    )
    parser.add_argument("--corpus-id", default="ad-hoc", help="corpus id for direct --fp16-dir mode")
    parser.add_argument("--layer-count", type=int, default=None, help="override number of decoder layers to compare")
    parser.add_argument("--no-router", action="store_true", help="skip optional router/top-k/traffic metrics")
    parser.add_argument("--json-out", type=Path, default=None, help="write full calibration JSON report")
    parser.add_argument("--max-layer-rmse", type=float, default=None)
    parser.add_argument("--max-layer-abs", type=float, default=None)
    parser.add_argument("--min-layer-cosine", type=float, default=None)
    parser.add_argument("--min-router-set-overlap", type=float, default=None)
    parser.add_argument("--min-logits-set-overlap", type=float, default=None)
    parser.add_argument("--require-generated-match", action="store_true")
    parser.add_argument(
        "--threshold-profile",
        choices=("auto", *sorted(QUANT_PARITY_THRESHOLDS)),
        default=None,
        help="apply the shared q8/q4 parity gate to each candidate; 'auto' infers from candidate labels",
    )
    return parser.parse_args(list(argv))


def _cases_from_args(args: argparse.Namespace) -> tuple[str, tuple[CalibrationFixtureCase, ...]]:
    if args.corpus is not None:
        if args.candidate:
            raise ValueError("--candidate is only valid with direct --fp16-dir mode; use corpus manifest candidates")
        return load_calibration_corpus(args.corpus)
    if not args.candidate:
        raise ValueError("direct --fp16-dir mode requires at least one --candidate LABEL=DIR")
    assert args.fp16_dir is not None
    candidates = {label: path for label, path in args.candidate}
    if len(candidates) != len(args.candidate):
        raise ValueError("candidate labels must be unique")
    case_id = args.case_id or args.fp16_dir.name or "case"
    return args.corpus_id, (
        CalibrationFixtureCase(case_id=case_id, fp16_dir=str(args.fp16_dir), candidates=candidates),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        corpus_id, cases = _cases_from_args(args)
        report = run_calibration(
            cases,
            corpus_id=corpus_id,
            layer_count=args.layer_count,
            include_router=not args.no_router,
        )
    except Exception as exc:
        print(f"uocr-calibrate: {exc}", file=sys.stderr)
        return 1

    print(_format_summary(report))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failures = _threshold_failures(
        report,
        max_layer_rmse=args.max_layer_rmse,
        max_layer_abs=args.max_layer_abs,
        min_layer_cosine=args.min_layer_cosine,
        min_router_set_overlap=args.min_router_set_overlap,
        min_logits_set_overlap=args.min_logits_set_overlap,
        require_generated_match=args.require_generated_match,
        quant_threshold_profile=args.threshold_profile,
    )
    if failures:
        for failure in failures:
            print(f"threshold failure: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
