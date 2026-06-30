"""Layer-drift comparison helpers for quantized OCR parity runs.

The quantized Metal path is validated against fp16 golden dumps produced by
``tools/uocr-ref-dump`` / :mod:`unlimitedocr_c.golden`.  This module compares a
reference dump directory to one or more candidate directories (for example
``dyn-q8`` and ``dyn-q4`` runs) without loading a whole model: it streams the
per-layer hidden-state files and reports numerically stable drift metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import math
import sys

import numpy as np
from numpy.typing import NDArray

from .golden import (
    HIDDEN_SIZE,
    MOE_TOP_K,
    TEXT_DECODER_LAYER_COUNT,
    router_top_ids_filename,
    router_top_weights_filename,
    text_layer_hidden_filename,
)


@dataclass(frozen=True)
class TensorDriftMetrics:
    """Scalar drift metrics for one tensor pair."""

    name: str
    shape: tuple[int, ...]
    count: int
    rmse: float
    mean_abs: float
    max_abs: float
    cosine_similarity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "count": self.count,
            "rmse": self.rmse,
            "mean_abs": self.mean_abs,
            "max_abs": self.max_abs,
            "cosine_similarity": self.cosine_similarity,
        }


@dataclass(frozen=True)
class RouterDriftMetrics:
    """Router top-k agreement/drift for one MoE layer, when dumps are present."""

    layer: int
    shape: tuple[int, int]
    ordered_id_agreement: float
    row_exact_agreement: float
    mean_set_overlap: float
    weight_rmse: float
    weight_max_abs: float
    weight_cosine_similarity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "shape": list(self.shape),
            "ordered_id_agreement": self.ordered_id_agreement,
            "row_exact_agreement": self.row_exact_agreement,
            "mean_set_overlap": self.mean_set_overlap,
            "weight_rmse": self.weight_rmse,
            "weight_max_abs": self.weight_max_abs,
            "weight_cosine_similarity": self.weight_cosine_similarity,
        }


@dataclass(frozen=True)
class CandidateDriftReport:
    """Drift report for one candidate profile/dump directory."""

    label: str
    reference_dir: str
    candidate_dir: str
    layer_metrics: tuple[TensorDriftMetrics, ...]
    router_metrics: tuple[RouterDriftMetrics, ...]

    @property
    def max_rmse(self) -> float:
        return max((m.rmse for m in self.layer_metrics), default=0.0)

    @property
    def max_abs(self) -> float:
        return max((m.max_abs for m in self.layer_metrics), default=0.0)

    @property
    def min_cosine_similarity(self) -> float:
        return min((m.cosine_similarity for m in self.layer_metrics), default=1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reference_dir": self.reference_dir,
            "candidate_dir": self.candidate_dir,
            "summary": {
                "layers": len(self.layer_metrics),
                "max_rmse": self.max_rmse,
                "max_abs": self.max_abs,
                "min_cosine_similarity": self.min_cosine_similarity,
            },
            "layers": [metric.as_dict() for metric in self.layer_metrics],
            "router_topk": [metric.as_dict() for metric in self.router_metrics],
        }


@dataclass(frozen=True)
class DriftComparisonReport:
    """Full fp16-vs-candidate drift report."""

    reference_dir: str
    candidates: tuple[CandidateDriftReport, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_dir": self.reference_dir,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class _LayerFileSpec:
    layer: int
    name: str
    filename: str
    shape: tuple[int, int]


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {manifest_path}")
    return data


def _infer_layer_count(root: Path, manifest: Mapping[str, Any], explicit_layer_count: int | None) -> int:
    if explicit_layer_count is not None:
        if explicit_layer_count < 1 or explicit_layer_count > TEXT_DECODER_LAYER_COUNT:
            raise ValueError(
                f"layer_count must be in [1,{TEXT_DECODER_LAYER_COUNT}], got {explicit_layer_count}"
            )
        return explicit_layer_count

    for key in ("text_decoder_layer_count", "image_decoder_layer_count"):
        value = manifest.get(key)
        if value is not None:
            count = int(value)
            if count < 1 or count > TEXT_DECODER_LAYER_COUNT:
                raise ValueError(f"invalid {key}={count} in {root / 'manifest.json'}")
            return count

    count = 0
    for layer in range(TEXT_DECODER_LAYER_COUNT):
        if (root / text_layer_hidden_filename(layer)).exists():
            count = layer + 1
    if count == 0:
        raise FileNotFoundError(f"no layer hidden dumps found in {root}")
    return count


def _layer_spec(root: Path, manifest: Mapping[str, Any], layer: int) -> _LayerFileSpec:
    name = f"layer_{layer}_hidden"
    golden_tensors = manifest.get("golden_tensors", {})
    entry = golden_tensors.get(name, {}) if isinstance(golden_tensors, dict) else {}
    filename = entry.get("file", text_layer_hidden_filename(layer)) if isinstance(entry, dict) else text_layer_hidden_filename(layer)
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"invalid filename metadata for {name} in {root / 'manifest.json'}")

    shape_meta = entry.get("shape") if isinstance(entry, dict) else None
    if isinstance(shape_meta, list) and len(shape_meta) == 2:
        shape = (int(shape_meta[0]), int(shape_meta[1]))
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"invalid shape metadata for {name}: {shape_meta}")
        return _LayerFileSpec(layer=layer, name=name, filename=filename, shape=shape)

    path = root / filename
    size = path.stat().st_size
    row_bytes = HIDDEN_SIZE * np.dtype("<u2").itemsize
    if size == 0 or size % row_bytes != 0:
        raise ValueError(
            f"cannot infer shape for {path}: size {size} is not a non-zero multiple of {row_bytes}"
        )
    return _LayerFileSpec(layer=layer, name=name, filename=filename, shape=(size // row_bytes, HIDDEN_SIZE))


def _candidate_layer_spec(
    reference_spec: _LayerFileSpec, candidate_root: Path, candidate_manifest: Mapping[str, Any]
) -> _LayerFileSpec:
    name = reference_spec.name
    golden_tensors = candidate_manifest.get("golden_tensors", {})
    candidate_tensors = candidate_manifest.get("candidate_tensors", {})
    entry: object = {}
    if isinstance(candidate_tensors, dict) and isinstance(candidate_tensors.get(name), dict):
        entry = candidate_tensors[name]
    elif isinstance(golden_tensors, dict) and isinstance(golden_tensors.get(name), dict):
        entry = golden_tensors[name]

    filename = reference_spec.filename
    shape = reference_spec.shape
    if isinstance(entry, dict):
        meta_file = entry.get("file")
        if isinstance(meta_file, str) and meta_file:
            filename = meta_file
        meta_shape = entry.get("shape")
        if isinstance(meta_shape, list) and len(meta_shape) == 2:
            shape = (int(meta_shape[0]), int(meta_shape[1]))

    path = candidate_root / filename
    if not path.exists():
        raise FileNotFoundError(f"candidate layer dump {path} does not exist")
    if shape != reference_spec.shape:
        raise ValueError(f"candidate {path} shape {shape} does not match reference {reference_spec.shape}")
    return _LayerFileSpec(layer=reference_spec.layer, name=name, filename=filename, shape=shape)


def _load_f16_file(path: Path, shape: Sequence[int]) -> NDArray[np.float32]:
    expected = math.prod(int(dim) for dim in shape)
    values = np.fromfile(path, dtype=np.dtype("<u2"))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} fp16 values, expected {expected} for shape {tuple(shape)}")
    return values.reshape(tuple(int(dim) for dim in shape)).view(np.dtype("<f2")).astype(np.float32)


def tensor_drift_metrics(name: str, reference: NDArray[np.float32], candidate: NDArray[np.float32]) -> TensorDriftMetrics:
    """Compute deterministic scalar drift metrics for two same-shaped tensors."""

    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"{name} shape mismatch: reference={ref.shape} candidate={cand.shape}")
    if ref.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError(f"{name} contains non-finite values")

    diff = cand - ref
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff.astype(np.float64) * diff.astype(np.float64))))
    mean_abs = float(np.mean(abs_diff, dtype=np.float64))
    max_abs = float(np.max(abs_diff))
    ref_flat = ref.reshape(-1).astype(np.float64)
    cand_flat = cand.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat))
    if denom == 0.0:
        cosine = 1.0 if max_abs == 0.0 else 0.0
    else:
        cosine = float(np.dot(ref_flat, cand_flat) / denom)
        cosine = max(-1.0, min(1.0, cosine))

    return TensorDriftMetrics(
        name=name,
        shape=tuple(int(dim) for dim in ref.shape),
        count=int(ref.size),
        rmse=rmse,
        mean_abs=mean_abs,
        max_abs=max_abs,
        cosine_similarity=cosine,
    )


def _router_paths(root: Path, manifest: Mapping[str, Any], layer: int) -> tuple[Path, Path] | None:
    router = manifest.get("router_topk", {})
    entry = router.get(f"layer_{layer}", {}) if isinstance(router, dict) else {}
    ids_file = entry.get("ids_file", router_top_ids_filename(layer)) if isinstance(entry, dict) else router_top_ids_filename(layer)
    weights_file = (
        entry.get("weights_file", router_top_weights_filename(layer))
        if isinstance(entry, dict)
        else router_top_weights_filename(layer)
    )
    if not isinstance(ids_file, str) or not isinstance(weights_file, str):
        return None
    ids_path = root / ids_file
    weights_path = root / weights_file
    if ids_path.exists() and weights_path.exists():
        return ids_path, weights_path
    return None


def _load_router(ids_path: Path, weights_path: Path) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
    ids = np.fromfile(ids_path, dtype=np.dtype("<u4"))
    weights = np.fromfile(weights_path, dtype=np.dtype("<f4"))
    if ids.size != weights.size or ids.size == 0 or ids.size % MOE_TOP_K != 0:
        raise ValueError(
            f"router dump size mismatch: ids={ids_path} ({ids.size}) weights={weights_path} ({weights.size})"
        )
    return ids.reshape((-1, MOE_TOP_K)), weights.reshape((-1, MOE_TOP_K))


def router_drift_metrics(
    layer: int,
    reference_ids: NDArray[np.uint32],
    reference_weights: NDArray[np.float32],
    candidate_ids: NDArray[np.uint32],
    candidate_weights: NDArray[np.float32],
) -> RouterDriftMetrics:
    if reference_ids.shape != candidate_ids.shape or reference_ids.shape != reference_weights.shape:
        raise ValueError(f"router layer {layer} shape mismatch")
    if reference_ids.shape != candidate_weights.shape or reference_ids.ndim != 2 or reference_ids.shape[1] != MOE_TOP_K:
        raise ValueError(f"router layer {layer} invalid top-k shape {reference_ids.shape}")

    ordered = float(np.mean(reference_ids == candidate_ids))
    row_exact = float(np.mean(np.all(reference_ids == candidate_ids, axis=1)))
    overlaps = []
    for ref_row, cand_row in zip(reference_ids, candidate_ids, strict=True):
        overlaps.append(len(set(int(v) for v in ref_row) & set(int(v) for v in cand_row)) / float(MOE_TOP_K))
    mean_overlap = float(np.mean(overlaps, dtype=np.float64)) if overlaps else 1.0
    weight_metrics = tensor_drift_metrics(
        f"layer_{layer}_router_top_weights",
        reference_weights.astype(np.float32, copy=False),
        candidate_weights.astype(np.float32, copy=False),
    )
    return RouterDriftMetrics(
        layer=layer,
        shape=(int(reference_ids.shape[0]), int(reference_ids.shape[1])),
        ordered_id_agreement=ordered,
        row_exact_agreement=row_exact,
        mean_set_overlap=mean_overlap,
        weight_rmse=weight_metrics.rmse,
        weight_max_abs=weight_metrics.max_abs,
        weight_cosine_similarity=weight_metrics.cosine_similarity,
    )


def compare_candidate_layer_drift(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    label: str = "candidate",
    layer_count: int | None = None,
    include_router: bool = True,
) -> CandidateDriftReport:
    """Compare one candidate dump directory against an fp16 reference dump."""

    ref_root = Path(reference_dir)
    cand_root = Path(candidate_dir)
    ref_manifest = _read_manifest(ref_root)
    cand_manifest = _read_manifest(cand_root)
    count = _infer_layer_count(ref_root, ref_manifest, layer_count)

    layer_metrics: list[TensorDriftMetrics] = []
    for layer in range(count):
        ref_spec = _layer_spec(ref_root, ref_manifest, layer)
        cand_spec = _candidate_layer_spec(ref_spec, cand_root, cand_manifest)
        reference = _load_f16_file(ref_root / ref_spec.filename, ref_spec.shape)
        candidate = _load_f16_file(cand_root / cand_spec.filename, cand_spec.shape)
        layer_metrics.append(tensor_drift_metrics(ref_spec.name, reference, candidate))

    router_metrics: list[RouterDriftMetrics] = []
    if include_router:
        for layer in range(1, count):
            ref_paths = _router_paths(ref_root, ref_manifest, layer)
            cand_paths = _router_paths(cand_root, cand_manifest, layer)
            if ref_paths is None or cand_paths is None:
                continue
            ref_ids, ref_weights = _load_router(*ref_paths)
            cand_ids, cand_weights = _load_router(*cand_paths)
            router_metrics.append(router_drift_metrics(layer, ref_ids, ref_weights, cand_ids, cand_weights))

    return CandidateDriftReport(
        label=label,
        reference_dir=str(ref_root),
        candidate_dir=str(cand_root),
        layer_metrics=tuple(layer_metrics),
        router_metrics=tuple(router_metrics),
    )


def compare_layer_drift(
    reference_dir: str | Path,
    candidates: Mapping[str, str | Path],
    *,
    layer_count: int | None = None,
    include_router: bool = True,
) -> DriftComparisonReport:
    """Compare fp16 reference layer dumps to one or more quantized candidates."""

    if not candidates:
        raise ValueError("at least one candidate dump directory is required")
    reports = tuple(
        compare_candidate_layer_drift(
            reference_dir,
            candidate_dir,
            label=label,
            layer_count=layer_count,
            include_router=include_router,
        )
        for label, candidate_dir in candidates.items()
    )
    return DriftComparisonReport(reference_dir=str(Path(reference_dir)), candidates=reports)


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label:
            raise argparse.ArgumentTypeError("candidate label cannot be empty")
        return label, Path(path)
    path = Path(value)
    return path.name or "candidate", path


def _format_summary(report: DriftComparisonReport) -> str:
    lines = [f"fp16 reference: {report.reference_dir}"]
    for candidate in report.candidates:
        lines.append(
            f"{candidate.label}: layers={len(candidate.layer_metrics)} "
            f"max_rmse={candidate.max_rmse:.6g} max_abs={candidate.max_abs:.6g} "
            f"min_cosine={candidate.min_cosine_similarity:.9f}"
        )
        for metric in candidate.layer_metrics:
            lines.append(
                f"  {metric.name}: rmse={metric.rmse:.6g} mean_abs={metric.mean_abs:.6g} "
                f"max_abs={metric.max_abs:.6g} cosine={metric.cosine_similarity:.9f}"
            )
        for metric in candidate.router_metrics:
            lines.append(
                f"  layer_{metric.layer}_router_topk: ordered={metric.ordered_id_agreement:.6g} "
                f"row_exact={metric.row_exact_agreement:.6g} set_overlap={metric.mean_set_overlap:.6g} "
                f"weight_rmse={metric.weight_rmse:.6g} weight_max_abs={metric.weight_max_abs:.6g}"
            )
    return "\n".join(lines)


def _threshold_failures(
    report: DriftComparisonReport,
    *,
    max_rmse: float | None,
    max_abs: float | None,
    min_cosine: float | None,
) -> list[str]:
    failures: list[str] = []
    for candidate in report.candidates:
        for metric in candidate.layer_metrics:
            prefix = f"{candidate.label}:{metric.name}"
            if max_rmse is not None and metric.rmse > max_rmse:
                failures.append(f"{prefix} rmse {metric.rmse:.6g} > {max_rmse:.6g}")
            if max_abs is not None and metric.max_abs > max_abs:
                failures.append(f"{prefix} max_abs {metric.max_abs:.6g} > {max_abs:.6g}")
            if min_cosine is not None and metric.cosine_similarity < min_cosine:
                failures.append(f"{prefix} cosine {metric.cosine_similarity:.9f} < {min_cosine:.9f}")
    return failures


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare q8/q4 layer dumps against fp16 golden dumps")
    parser.add_argument("--fp16-dir", type=Path, required=True, help="fp16 golden dump directory")
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        metavar="LABEL=DIR",
        help="candidate dump directory; repeat for dyn-q8/dyn-q4",
    )
    parser.add_argument("--layer-count", type=int, default=None, help="override number of decoder layers to compare")
    parser.add_argument("--no-router", action="store_true", help="skip optional router top-k comparison")
    parser.add_argument("--json-out", type=Path, default=None, help="write full JSON report")
    parser.add_argument("--max-rmse", type=float, default=None, help="fail if any layer RMSE exceeds this threshold")
    parser.add_argument("--max-abs", type=float, default=None, help="fail if any layer max-abs exceeds this threshold")
    parser.add_argument("--min-cosine", type=float, default=None, help="fail if any layer cosine drops below this threshold")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    candidates = {label: path for label, path in args.candidate}
    report = compare_layer_drift(
        args.fp16_dir,
        candidates,
        layer_count=args.layer_count,
        include_router=not args.no_router,
    )
    print(_format_summary(report))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failures = _threshold_failures(
        report,
        max_rmse=args.max_rmse,
        max_abs=args.max_abs,
        min_cosine=args.min_cosine,
    )
    if failures:
        for failure in failures:
            print(f"threshold failure: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
