"""Dry-run conversion planning for Unlimited-OCR `.uocr` model files.

The first converter milestone intentionally works from the cached safetensors
header/index only.  It validates the current checkpoint structure and produces a
stable, inspectable fp16 plan without requiring the 6.67 GB weight payload.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping

from .frontend import project_root

EXPECTED_TENSOR_COUNT = 2710
EXPECTED_TOTAL_BYTES = 6_672_212_480
EXPECTED_SOURCE_DTYPE = "BF16"

EXPECTED_CONFIG: Mapping[str, int | float | str | bool] = {
    "vocab_size": 129_280,
    "hidden_size": 1280,
    "num_hidden_layers": 12,
    "num_attention_heads": 10,
    "num_key_value_heads": 10,
    "max_position_embeddings": 32_768,
    "first_k_dense_replace": 1,
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 896,
    "n_shared_experts": 2,
    "intermediate_size": 6848,
    "topk_method": "greedy",
}

# Several parity-critical values are inherited from DeepSeekV2Config defaults in
# data/context/configuration_deepseek_v2.py and are absent from config.json.
EXPECTED_CONFIG_DEFAULTS: Mapping[str, int | float | str | bool] = {
    "rope_theta": 10_000,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "scoring_func": "softmax",
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": False,
}

KEY_TENSOR_SHAPES: Mapping[str, tuple[int, ...]] = {
    "lm_head.weight": (129_280, 1280),
    "model.embed_tokens.weight": (129_280, 1280),
    "model.layers.0.self_attn.q_proj.weight": (1280, 1280),
    "model.layers.0.self_attn.k_proj.weight": (1280, 1280),
    "model.layers.0.self_attn.v_proj.weight": (1280, 1280),
    "model.layers.0.self_attn.o_proj.weight": (1280, 1280),
    "model.layers.0.mlp.down_proj.weight": (1280, 6848),
    "model.layers.1.mlp.gate.weight": (64, 1280),
    "model.layers.1.mlp.experts.0.down_proj.weight": (1280, 896),
    "model.layers.1.mlp.experts.0.gate_proj.weight": (896, 1280),
    "model.layers.1.mlp.experts.0.up_proj.weight": (896, 1280),
    "model.layers.1.mlp.shared_experts.down_proj.weight": (1280, 1792),
    "model.projector.layers.weight": (1280, 2048),
    "model.projector.layers.bias": (1280,),
    "model.image_newline": (1280,),
    "model.view_seperator": (1280,),
    "model.sam_model.patch_embed.proj.weight": (768, 3, 16, 16),
    "model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight": (3072, 1024),
}


@dataclass(frozen=True)
class TensorPlan:
    name: str
    source_dtype: str
    shape: tuple[int, ...]
    source_offsets: tuple[int, int]
    source_bytes: int
    output_dtype: str
    qtype: str
    output_bytes: int
    family: str
    usage: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_dtype": self.source_dtype,
            "shape": list(self.shape),
            "source_offsets": list(self.source_offsets),
            "source_bytes": self.source_bytes,
            "output_dtype": self.output_dtype,
            "qtype": self.qtype,
            "output_bytes": self.output_bytes,
            "family": self.family,
            "usage": self.usage,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DryRunPlan:
    hf_dir: Path
    qprofile: str
    tensors: tuple[TensorPlan, ...]
    total_source_bytes: int
    total_output_bytes: int
    qtype_histogram: Mapping[str, int]
    usage_histogram: Mapping[str, int]
    family_histogram: Mapping[str, int]

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    def tensor_by_name(self, name: str) -> TensorPlan:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "hf_dir": str(self.hf_dir),
            "qprofile": self.qprofile,
            "tensor_count": self.tensor_count,
            "total_source_bytes": self.total_source_bytes,
            "total_output_bytes": self.total_output_bytes,
            "qtype_histogram": dict(self.qtype_histogram),
            "usage_histogram": dict(self.usage_histogram),
            "family_histogram": dict(self.family_histogram),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_dict(),
            "tensors": [tensor.as_dict() for tensor in self.tensors],
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _product(shape: Iterable[int]) -> int:
    return math.prod(int(dim) for dim in shape)


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) == 8:
            header_len = struct.unpack("<Q", prefix)[0]
            # Cached `.header` files and real `.safetensors` files both use the
            # safetensors 8-byte length prefix.  Keep this bounded so a bad path
            # cannot accidentally request a huge read during dry-run tests.
            if 0 < header_len < 256 * 1024 * 1024:
                header_bytes = f.read(header_len)
                if len(header_bytes) == header_len and header_bytes.lstrip().startswith(b"{"):
                    return json.loads(header_bytes)
        f.seek(0)
        return json.loads(f.read().decode("utf-8"))


def _default_header_path(hf_dir: Path) -> Path:
    candidates = [
        hf_dir / "model.safetensors.header",
        hf_dir / "model-00001-of-000001.safetensors",
        hf_dir / "model.safetensors",
    ]
    candidates.extend(sorted(hf_dir.glob("*.safetensors")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no safetensors header or file found under {hf_dir}")


def _check_config_value(actual: Any, expected: int | float | str | bool, key: str) -> None:
    if isinstance(expected, float):
        if abs(float(actual) - expected) > 1e-12:
            raise ValueError(f"config {key} mismatch: got {actual!r}, expected {expected!r}")
    elif actual != expected:
        raise ValueError(f"config {key} mismatch: got {actual!r}, expected {expected!r}")


def validate_config(hf_dir: Path) -> None:
    config_path = hf_dir / "config.json"
    if not config_path.exists():
        return
    config = _read_json(config_path)
    for key, expected in EXPECTED_CONFIG.items():
        if key not in config:
            raise ValueError(f"config missing required key {key!r}")
        _check_config_value(config[key], expected, key)
    for key, expected in EXPECTED_CONFIG_DEFAULTS.items():
        if key in config:
            _check_config_value(config[key], expected, key)

    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    if head_dim != 128:
        raise ValueError(f"derived attention head dim mismatch: got {head_dim}, expected 128")


def _validate_header_entry(name: str, entry: Mapping[str, Any]) -> None:
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if dtype != EXPECTED_SOURCE_DTYPE:
        raise ValueError(f"tensor {name} dtype mismatch: got {dtype!r}, expected {EXPECTED_SOURCE_DTYPE}")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError(f"tensor {name} has invalid shape {shape!r}")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(v, int) and v >= 0 for v in offsets):
        raise ValueError(f"tensor {name} has invalid data_offsets {offsets!r}")
    if offsets[1] < offsets[0]:
        raise ValueError(f"tensor {name} has descending data_offsets {offsets!r}")
    expected_bytes = _product(shape) * 2
    actual_bytes = offsets[1] - offsets[0]
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"tensor {name} byte count mismatch: offsets imply {actual_bytes}, shape implies {expected_bytes}"
        )


def _validate_key_shapes(entries: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected_shape in KEY_TENSOR_SHAPES.items():
        entry = entries.get(name)
        if entry is None:
            raise ValueError(f"expected tensor {name!r} is missing")
        actual_shape = tuple(int(dim) for dim in entry["shape"])
        if actual_shape != expected_shape:
            raise ValueError(f"tensor {name} shape mismatch: got {actual_shape}, expected {expected_shape}")


def _family_for_name(name: str) -> str:
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.embed_tokens.weight":
        return "token_embedding"
    if name in {"model.image_newline", "model.view_seperator"}:
        return "special_visual_embedding"
    if name.startswith("model.projector."):
        return "projector"
    if name.startswith("model.sam_model."):
        return "vision_sam"
    if name.startswith("model.vision_model."):
        return "vision_clip"
    if ".mlp.experts." in name:
        return "moe_routed_expert"
    if ".mlp.shared_experts." in name:
        return "moe_shared_expert"
    if name.endswith(".mlp.gate.weight"):
        return "moe_router"
    if name.startswith("model.layers.0.mlp."):
        return "decoder_dense_mlp"
    if name.startswith("model.layers."):
        return "decoder"
    if name.startswith("model.norm.") or name == "model.norm.weight":
        return "decoder_norm"
    return "other"


def _usage_for_name(name: str) -> tuple[str, str]:
    if name.startswith("model.vision_model.embeddings.patch_embedding."):
        return (
            "preserved-unused",
            "CLIP receives SAM feature maps as patch_embeds in the normal OCR path",
        )
    if name.endswith(".mlp.gate.weight"):
        return "runtime", "MoE router: mandatory fp16 keep-list"
    if "norm" in name or name.endswith("bias") or name in {"model.image_newline", "model.view_seperator"}:
        return "runtime", "fp16 keep-list for norms/biases/special embeddings"
    return "runtime", "fp16 baseline"


def _make_tensor_plan(name: str, entry: Mapping[str, Any], qprofile: str) -> TensorPlan:
    if qprofile != "fp16":
        raise NotImplementedError("only fp16 dry-run planning is implemented")
    shape = tuple(int(dim) for dim in entry["shape"])
    offsets = (int(entry["data_offsets"][0]), int(entry["data_offsets"][1]))
    source_bytes = offsets[1] - offsets[0]
    usage, reason = _usage_for_name(name)
    return TensorPlan(
        name=name,
        source_dtype=str(entry["dtype"]),
        shape=shape,
        source_offsets=offsets,
        source_bytes=source_bytes,
        output_dtype="F16",
        qtype="UOCR_TENSOR_F16",
        output_bytes=source_bytes,
        family=_family_for_name(name),
        usage=usage,
        reason=reason,
    )


def build_dry_run_plan(
    hf_dir: str | Path,
    *,
    qprofile: str = "fp16",
    header_path: str | Path | None = None,
    index_path: str | Path | None = None,
) -> DryRunPlan:
    hf_dir = Path(hf_dir)
    if qprofile != "fp16":
        raise NotImplementedError("only --qprofile fp16 is implemented for the first dry-run converter")

    validate_config(hf_dir)

    header = _read_safetensors_header(Path(header_path) if header_path is not None else _default_header_path(hf_dir))
    entries: dict[str, Mapping[str, Any]] = {
        name: entry for name, entry in header.items() if name != "__metadata__"
    }
    if len(entries) != EXPECTED_TENSOR_COUNT:
        raise ValueError(f"safetensors tensor count mismatch: got {len(entries)}, expected {EXPECTED_TENSOR_COUNT}")

    for name, entry in entries.items():
        _validate_header_entry(name, entry)
    _validate_key_shapes(entries)

    tensors = tuple(
        _make_tensor_plan(name, entries[name], qprofile)
        for name in sorted(entries, key=lambda key: int(entries[key]["data_offsets"][0]))
    )
    total_source_bytes = sum(t.source_bytes for t in tensors)
    total_output_bytes = sum(t.output_bytes for t in tensors)
    if total_source_bytes != EXPECTED_TOTAL_BYTES:
        raise ValueError(
            f"source payload byte count mismatch: got {total_source_bytes}, expected {EXPECTED_TOTAL_BYTES}"
        )

    index = _read_json(Path(index_path) if index_path is not None else hf_dir / "model.safetensors.index.json")
    index_total = int(index.get("metadata", {}).get("total_size", -1))
    if index_total != total_source_bytes:
        raise ValueError(f"index total_size mismatch: got {index_total}, header implies {total_source_bytes}")
    weight_map = index.get("weight_map", {})
    if len(weight_map) != len(entries):
        raise ValueError(f"index weight_map count mismatch: got {len(weight_map)}, expected {len(entries)}")
    missing_from_index = sorted(set(entries) - set(weight_map))
    if missing_from_index:
        raise ValueError(f"{len(missing_from_index)} header tensors missing from index; first={missing_from_index[0]}")

    return DryRunPlan(
        hf_dir=hf_dir,
        qprofile=qprofile,
        tensors=tensors,
        total_source_bytes=total_source_bytes,
        total_output_bytes=total_output_bytes,
        qtype_histogram=dict(Counter(t.qtype for t in tensors)),
        usage_histogram=dict(Counter(t.usage for t in tensors)),
        family_histogram=dict(Counter(t.family for t in tensors)),
    )


def _print_summary(plan: DryRunPlan) -> None:
    print("Unlimited-OCR converter dry-run")
    print(f"hf_dir: {plan.hf_dir}")
    print(f"qprofile: {plan.qprofile}")
    print(f"tensors: {plan.tensor_count}")
    print(f"source bytes: {plan.total_source_bytes}")
    print(f"planned output bytes: {plan.total_output_bytes}")
    print(f"qtypes: {dict(plan.qtype_histogram)}")
    print(f"usage: {dict(plan.usage_histogram)}")
    print("families:")
    for family, count in sorted(plan.family_histogram.items()):
        print(f"  {family}: {count}")


def _write_dump(plan: DryRunPlan, path: str) -> None:
    payload = json.dumps(plan.as_dict(), indent=2, sort_keys=True)
    if path == "-":
        print(payload)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan an Unlimited-OCR .uocr conversion without requiring full weights")
    parser.add_argument("--hf-dir", type=Path, default=project_root() / "data/context")
    parser.add_argument("--header", type=Path, default=None, help="optional safetensors header/cache path")
    parser.add_argument("--index", type=Path, default=None, help="optional safetensors index path")
    parser.add_argument("--qprofile", choices=["fp16", "dyn-q8", "dyn-q4"], default="fp16")
    parser.add_argument("--dry-run", action="store_true", help="required for now; real conversion is a later milestone")
    parser.add_argument("--dump-plan", nargs="?", const="-", default=None, help="write full JSON plan to path or stdout")
    args = parser.parse_args(argv)

    if not args.dry_run:
        parser.error("only --dry-run is implemented in this milestone")
    if args.qprofile != "fp16":
        parser.error("only --qprofile fp16 is implemented in this milestone")

    plan = build_dry_run_plan(args.hf_dir, qprofile=args.qprofile, header_path=args.header, index_path=args.index)
    _print_summary(plan)
    if args.dump_plan is not None:
        _write_dump(plan, args.dump_plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
