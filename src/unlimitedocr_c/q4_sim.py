"""Q4 quantization *simulation* for routed MoE expert weights.

Quality gate for a future int4 expert format: clone an existing fp16 ``.uocr``
model and rewrite the routed-expert (family MOE_EXPERT) gate/up/down payloads
with quantize->dequantize values, so the file stays a plain fp16 model that the
current runtime loads unmodified.  If OCR quality survives this simulation, a
real Q4 storage format + Metal kernels are worth building; if it collapses,
we learned that in a day with zero kernel work.

Modes:
  q4_0  symmetric, per-group scale only (ggml Q4_0-style, scale = signed_max / -8)
  q4_1  asymmetric, per-group scale + min (ggml Q4_1-style, 16 levels)

Usage:
  uv run python -m unlimitedocr_c.q4_sim \\
      --src  ~/Library/Caches/unlimitedocr/unlimitedocr-fp16.uocr \\
      --out  ~/Library/Caches/unlimitedocr/unlimitedocr-fp16-q4sim.uocr \\
      --mode q4_1 --projections gate,up,down
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .convert import (
    _FILE_HEADER_STRUCT,
    _SECTION_ENTRY_STRUCT,
    _TENSOR_DIRECTORY_HEADER_STRUCT,
    _TENSOR_ENTRY_STRUCT,
    UOCR_FILE_HEADER_SIZE,
    UOCR_SECTION_ENTRY_SIZE,
    UOCR_SECTION_TENSOR_DIRECTORY,
    UOCR_TENSOR_DIR_MAGIC,
    UOCR_TENSOR_DIRECTORY_HEADER_SIZE,
    UOCR_TENSOR_ENTRY_SIZE,
    UOCR_TENSOR_F16,
)
from .tensor_registry import TensorFamily, TensorProjection

_PROJECTION_BY_NAME = {
    "gate": TensorProjection.GATE,
    "up": TensorProjection.UP,
    "down": TensorProjection.DOWN,
}


@dataclass(frozen=True)
class _ExpertTensor:
    tensor_id: int
    layer: int
    expert: int
    projection: TensorProjection
    rows: int
    cols: int
    payload_offset: int
    payload_size: int


def _quant_dequant_q4_0(values: np.ndarray, group_size: int) -> np.ndarray:
    """Symmetric int4 (levels -8..7), per-group fp16 scale = signed_max / -8."""
    groups = values.reshape(-1, group_size)
    abs_idx = np.argmax(np.abs(groups), axis=1)
    signed_max = groups[np.arange(groups.shape[0]), abs_idx]
    scales = (signed_max / np.float32(-8.0)).astype(np.float16).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.float32(1.0), scales)
    q = np.clip(np.rint(groups / scales_safe[:, None]), -8, 7)
    return (q * scales[:, None]).astype(np.float32).reshape(values.shape)


def _quant_dequant_q4_1(values: np.ndarray, group_size: int) -> np.ndarray:
    """Asymmetric int4 (levels 0..15), per-group fp16 scale + min."""
    groups = values.reshape(-1, group_size)
    mins = groups.min(axis=1).astype(np.float16).astype(np.float32)
    maxs = groups.max(axis=1)
    scales = ((maxs - mins) / np.float32(15.0)).astype(np.float16).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.float32(1.0), scales)
    q = np.clip(np.rint((groups - mins[:, None]) / scales_safe[:, None]), 0, 15)
    return (q * scales[:, None] + mins[:, None]).astype(np.float32).reshape(values.shape)


_MODES = {"q4_0": _quant_dequant_q4_0, "q4_1": _quant_dequant_q4_1}


def _read_expert_tensors(path: Path) -> tuple[list[_ExpertTensor], int]:
    """Parse the tensor directory and return fp16 MOE_EXPERT gate/up/down entries."""
    tensors: list[_ExpertTensor] = []
    skipped_non_f16 = 0
    with path.open("rb") as f:
        header_bytes = f.read(UOCR_FILE_HEADER_SIZE)
        if len(header_bytes) != UOCR_FILE_HEADER_SIZE:
            raise ValueError(f"{path} is too small to contain a .uocr header")
        header = _FILE_HEADER_STRUCT.unpack(header_bytes)
        if header[0] != b"UOCR":
            raise ValueError(f"{path} is not a .uocr file")
        section_count = header[6]
        file_size, section_dir_offset = header[8], header[9]
        if file_size != path.stat().st_size:
            raise ValueError(f"{path} file size mismatch")

        f.seek(section_dir_offset)
        tensor_dir_offset = None
        for _ in range(section_count):
            raw = f.read(UOCR_SECTION_ENTRY_SIZE)
            section_type, _flags, offset, _size, _alignment = _SECTION_ENTRY_STRUCT.unpack(raw)
            if section_type == UOCR_SECTION_TENSOR_DIRECTORY:
                tensor_dir_offset = int(offset)
        if tensor_dir_offset is None:
            raise ValueError(f"{path} does not contain a tensor directory")

        f.seek(tensor_dir_offset)
        dir_magic, _version, entry_size, tensor_count = _TENSOR_DIRECTORY_HEADER_STRUCT.unpack(
            f.read(UOCR_TENSOR_DIRECTORY_HEADER_SIZE)
        )
        if dir_magic != UOCR_TENSOR_DIR_MAGIC:
            raise ValueError(f"{path} has an invalid tensor-directory magic")
        if entry_size != UOCR_TENSOR_ENTRY_SIZE:
            raise ValueError(f"{path} has an unexpected tensor entry size {entry_size}")

        for _ in range(tensor_count):
            entry = _TENSOR_ENTRY_STRUCT.unpack(f.read(UOCR_TENSOR_ENTRY_SIZE))
            family_id = int(entry[1])
            projection_id = int(entry[4])
            if family_id != int(TensorFamily.MOE_EXPERT):
                continue
            if projection_id not in (
                int(TensorProjection.GATE),
                int(TensorProjection.UP),
                int(TensorProjection.DOWN),
            ):
                continue
            if int(entry[6]) != UOCR_TENSOR_F16:
                skipped_non_f16 += 1
                continue
            rank = int(entry[8])
            physical_shape = tuple(int(v) for v in entry[13 : 13 + rank])
            if rank != 2:
                raise ValueError(f"expert tensor id {entry[0]} has unsupported rank {rank}")
            tensors.append(
                _ExpertTensor(
                    tensor_id=int(entry[0]),
                    layer=int(entry[2]),
                    expert=int(entry[3]),
                    projection=TensorProjection(projection_id),
                    rows=physical_shape[0],
                    cols=physical_shape[1],
                    payload_offset=int(entry[17]),
                    payload_size=int(entry[18]),
                )
            )
    return tensors, skipped_non_f16


def simulate_q4(
    src: Path,
    out: Path,
    *,
    mode: str,
    projections: frozenset[TensorProjection],
    group_size: int = 64,
    overwrite: bool = False,
) -> dict[str, object]:
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_MODES)}")
    quant_dequant = _MODES[mode]
    if out.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {out}")

    tensors, skipped_non_f16 = _read_expert_tensors(src)
    if skipped_non_f16:
        raise ValueError(
            f"{src} has {skipped_non_f16} non-fp16 expert tensors; "
            "run the simulation on the fp16 model, not the Q8 one"
        )
    selected = [t for t in tensors if t.projection in projections]
    if not selected:
        raise ValueError(f"{src} has no fp16 MOE_EXPERT tensors for projections {sorted(p.name for p in projections)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    shutil.copyfile(src, tmp)

    sq_err = 0.0
    values_total = 0
    max_abs_err = 0.0
    per_projection: dict[str, dict[str, float]] = {}
    try:
        with tmp.open("r+b") as f:
            for tensor in selected:
                expected_bytes = tensor.rows * tensor.cols * 2
                if tensor.payload_size != expected_bytes:
                    raise ValueError(
                        f"tensor id {tensor.tensor_id} payload size {tensor.payload_size} "
                        f"does not match shape [{tensor.rows}, {tensor.cols}]"
                    )
                if tensor.cols % group_size != 0:
                    raise ValueError(
                        f"tensor id {tensor.tensor_id} cols {tensor.cols} not divisible by group {group_size}"
                    )
                f.seek(tensor.payload_offset)
                raw = f.read(expected_bytes)
                values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                simulated = quant_dequant(values, group_size)

                err = simulated - values
                tensor_sq = float(np.sum(err * err))
                tensor_max = float(np.max(np.abs(err)))
                sq_err += tensor_sq
                values_total += values.size
                max_abs_err = max(max_abs_err, tensor_max)
                stats = per_projection.setdefault(
                    tensor.projection.name.lower(), {"sq_err": 0.0, "count": 0.0, "max_abs_err": 0.0}
                )
                stats["sq_err"] += tensor_sq
                stats["count"] += values.size
                stats["max_abs_err"] = max(stats["max_abs_err"], tensor_max)

                f.seek(tensor.payload_offset)
                f.write(simulated.astype(np.float16).tobytes())
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return {
        "mode": mode,
        "group_size": group_size,
        "tensors_rewritten": len(selected),
        "projections": sorted(p.name.lower() for p in projections),
        "rmse": (sq_err / values_total) ** 0.5 if values_total else 0.0,
        "max_abs_err": max_abs_err,
        "per_projection": {
            name: {
                "rmse": (s["sq_err"] / s["count"]) ** 0.5 if s["count"] else 0.0,
                "max_abs_err": s["max_abs_err"],
            }
            for name, s in sorted(per_projection.items())
        },
        "out": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="source fp16 .uocr model")
    parser.add_argument("--out", type=Path, required=True, help="output .uocr with Q4-simulated experts")
    parser.add_argument("--mode", choices=sorted(_MODES), default="q4_1")
    parser.add_argument(
        "--projections",
        default="gate,up,down",
        help="comma-separated subset of gate,up,down to simulate (default: all)",
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    names = [p.strip().lower() for p in args.projections.split(",") if p.strip()]
    unknown = [n for n in names if n not in _PROJECTION_BY_NAME]
    if unknown:
        parser.error(f"unknown projections: {unknown}; expected subset of gate,up,down")
    projections = frozenset(_PROJECTION_BY_NAME[n] for n in names)

    summary = simulate_q4(
        args.src,
        args.out,
        mode=args.mode,
        projections=projections,
        group_size=args.group_size,
        overwrite=args.overwrite,
    )
    print(f"q4-sim mode={summary['mode']} group={summary['group_size']}")
    print(f"  tensors rewritten: {summary['tensors_rewritten']} ({','.join(summary['projections'])})")
    print(f"  rmse: {summary['rmse']:.6g}  max_abs_err: {summary['max_abs_err']:.6g}")
    for name, stats in summary["per_projection"].items():
        print(f"  {name}: rmse={stats['rmse']:.6g} max_abs_err={stats['max_abs_err']:.6g}")
    print(f"  wrote: {summary['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
