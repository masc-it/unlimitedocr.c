"""Runtime-supported Q8_0 module configuration.

The converter quantizes only the (family, projection) pairs that the Metal
runtime can actually consume with fused Q8 kernels.  This list lives in
``configs/quant-cfg.yaml`` and grows as kernels land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .tensor_registry import TensorFamily, TensorProjection

#: Default config path, relative to the project root.
DEFAULT_QUANT_CFG_NAME = "quant-cfg.yaml"


@dataclass(frozen=True)
class QuantModuleSpec:
    name: str
    family: TensorFamily
    projections: frozenset[TensorProjection]
    supported: bool


@dataclass(frozen=True)
class QuantConfig:
    version: int
    group_size: int
    modules: tuple[QuantModuleSpec, ...]
    #: Resolved (family, projection) pairs that are runtime-supported.
    supported_pairs: frozenset[tuple[TensorFamily, TensorProjection]] = field(
        default_factory=frozenset
    )

    def is_supported(self, family: TensorFamily, projection: TensorProjection) -> bool:
        return (family, projection) in self.supported_pairs

    def module_by_name(self, name: str) -> QuantModuleSpec | None:
        for module in self.modules:
            if module.name == name:
                return module
        return None


def _resolve_family(name: str) -> TensorFamily:
    try:
        return TensorFamily[name]
    except KeyError as exc:
        raise ValueError(f"unknown TensorFamily {name!r} in quant-cfg") from exc


def _resolve_projection(name: str) -> TensorProjection:
    try:
        return TensorProjection[name]
    except KeyError as exc:
        raise ValueError(f"unknown TensorProjection {name!r} in quant-cfg") from exc


def _parse_config(raw: object) -> QuantConfig:
    if not isinstance(raw, dict):
        raise ValueError("quant-cfg must be a mapping")
    version = int(raw.get("version", 1))
    if version != 1:
        raise ValueError(f"unsupported quant-cfg version {version}")
    group_size = int(raw.get("group_size", 64))
    raw_modules = raw.get("modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        raise ValueError("quant-cfg must define a non-empty 'modules' list")

    parsed: list[QuantModuleSpec] = []
    supported: set[tuple[TensorFamily, TensorProjection]] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(raw_modules):
        if not isinstance(item, dict):
            raise ValueError(f"quant-cfg module #{index} must be a mapping")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"quant-cfg module #{index} is missing a name")
        if name in seen_names:
            raise ValueError(f"quant-cfg module {name!r} is defined more than once")
        seen_names.add(name)
        family = _resolve_family(str(item.get("family", "")))
        raw_projections = item.get("projections")
        if not isinstance(raw_projections, list) or not raw_projections:
            raise ValueError(f"quant-cfg module {name!r} must define projections")
        projections = frozenset(_resolve_projection(str(p)) for p in raw_projections)
        supported_flag = bool(item.get("supported", False))
        parsed.append(
            QuantModuleSpec(
                name=name,
                family=family,
                projections=projections,
                supported=supported_flag,
            )
        )
        if supported_flag:
            for projection in projections:
                pair = (family, projection)
                if pair in supported:
                    raise ValueError(
                        f"quant-cfg module {name!r} redeclares supported pair "
                        f"({family.name}, {projection.name})"
                    )
                supported.add(pair)

    return QuantConfig(
        version=version,
        group_size=group_size,
        modules=tuple(parsed),
        supported_pairs=frozenset(supported),
    )


def load_quant_config(path: str | Path) -> QuantConfig:
    """Load and validate a quant-cfg.yaml file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _parse_config(raw)


def default_quant_cfg_path() -> Path:
    """Return the bundled runtime-safe quant-cfg path under the project root."""
    from .frontend import project_root

    return project_root() / "configs" / DEFAULT_QUANT_CFG_NAME


def load_default_quant_config() -> QuantConfig:
    """Load the bundled runtime-safe quant-cfg, falling back to embeddings-only."""
    path = default_quant_cfg_path()
    if path.is_file():
        return load_quant_config(path)
    # Source-tree fallback when the configs/ directory is not co-located.
    return _embeddings_only_config()


def _embeddings_only_config() -> QuantConfig:
    spec = QuantModuleSpec(
        name="token_embedding",
        family=TensorFamily.TOK_EMBED,
        projections=frozenset({TensorProjection.WEIGHT}),
        supported=True,
    )
    return QuantConfig(
        version=1,
        group_size=64,
        modules=(spec,),
        supported_pairs=frozenset({(TensorFamily.TOK_EMBED, TensorProjection.WEIGHT)}),
    )


def all_supported_quant_config() -> QuantConfig:
    """Return a config with every known candidate module supported.

    Used by tests that exercise the full planner independent of the
    runtime-safe subset.
    """
    candidates: list[tuple[str, TensorFamily, tuple[TensorProjection, ...]]] = [
        ("token_embedding", TensorFamily.TOK_EMBED, (TensorProjection.WEIGHT,)),
        ("lm_head", TensorFamily.LM_HEAD, (TensorProjection.WEIGHT,)),
        ("attention_qkv", TensorFamily.LAYER_ATTN,
         (TensorProjection.Q, TensorProjection.K, TensorProjection.V)),
        ("attention_output", TensorFamily.LAYER_ATTN, (TensorProjection.O,)),
        ("dense_mlp", TensorFamily.LAYER_DENSE_MLP,
         (TensorProjection.GATE, TensorProjection.UP, TensorProjection.DOWN)),
        ("moe_shared", TensorFamily.MOE_SHARED,
         (TensorProjection.GATE, TensorProjection.UP, TensorProjection.DOWN)),
        ("moe_routed_experts", TensorFamily.MOE_EXPERT,
         (TensorProjection.GATE, TensorProjection.UP, TensorProjection.DOWN)),
    ]
    modules: list[QuantModuleSpec] = []
    supported: set[tuple[TensorFamily, TensorProjection]] = set()
    for name, family, projections in candidates:
        modules.append(
            QuantModuleSpec(
                name=name,
                family=family,
                projections=frozenset(projections),
                supported=True,
            )
        )
        supported.update((family, projection) for projection in projections)
    return QuantConfig(
        version=1,
        group_size=64,
        modules=tuple(modules),
        supported_pairs=frozenset(supported),
    )
