"""Supported checkpoint contracts, independent of image and quantization profiles."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Literal

ModelName = Literal["v1", "v2"]
MODEL_REPOSITORIES = {"v1": "baidu/Unlimited-OCR", "v2": "baidu/Unlimited-OCR-2"}
V2_REVISION = "c14071b549f4fcab10909f3c4acc6a2edb366947"


@dataclass(frozen=True)
class ModelVariant:
    format_version: int
    tensor_count: int
    source_bytes: int
    added_token_count: int
    rope_theta: int
    max_positions: int
    generated_ring_window: int


MODEL_VARIANTS = {
    "v1": ModelVariant(1, 2710, 6_672_212_480, 830, 10_000, 32_768, 128),
    "v2": ModelVariant(2, 2819, 6_687_958_536, 834, 500_000, 131_072, 0),
}


def get_model_variant(model: str) -> ModelVariant:
    try:
        return MODEL_VARIANTS[model]
    except KeyError:
        raise ValueError(f"unknown model {model!r}; expected 'v1' or 'v2'") from None


def source_model(model: str) -> tuple[str, str | None]:
    """Resolve optional source overrides; v2's official checkpoint is pinned."""
    get_model_variant(model)
    repo = os.environ.get("UOCR_SOURCE_MODEL_REPO_ID") or MODEL_REPOSITORIES[model]
    revision = os.environ.get("UOCR_SOURCE_MODEL_REVISION") or None
    if revision is None and model == "v2" and repo == MODEL_REPOSITORIES[model]:
        revision = V2_REVISION
    return repo, revision


def source_tokenizer(model: str) -> tuple[str, str | None]:
    repo, revision = source_model(model)
    tokenizer_repo = os.environ.get("UOCR_TOKENIZER_REPO_ID") or repo
    tokenizer_revision = os.environ.get("UOCR_TOKENIZER_REVISION") or (revision if tokenizer_repo == repo else None)
    return tokenizer_repo, tokenizer_revision


def model_cache_dir(root: Path, model: str) -> Path:
    repo, revision = source_model(model)
    identity = hashlib.sha256(f"{repo}\0{revision or 'main'}".encode()).hexdigest()[:16]
    return root / model / identity


V2_ADDED_TOKENS = {128827: "<PAGE>", 128828: "<BLOCK>", 128829: "<FORK>", 128830: "<CHILD>"}

# Only this inference contract is supported. Training and execution-backend
# settings (e.g. Triton vs. Metal) do not change the serialized native math.
V2_SPARSE_CONFIG = {
    "training_stage": "dash",
    "use_sparse_attention": True,
    "sparse_attention_freq": [1] * 12,
    "dash_router": "entmax_topk_perlayer",
    "dash_stage0": "head_cls",
    "dash_image_only": True,
    "dash_topk_blocks": 12,
    "dash_scale_per_layer": [128.0] * 12,
    "dash_floor_per_layer": [0.0001] * 12,
    "dash_scaling_factor": 128.0,
    "dash_prior_floor": 0.0001,
    "dash_sigma": 1.0,
    "image_bucket_constraint": True,
    "image_token_id": 128815,
    "page_token_id": 128827,
    "per_page_image_tokens": 273,
    "sparse_block_size": 32,
    "sparse_index_dim": 128,
    "sparse_num_index_heads": 1,
    "sparse_local_block": 4,
    "sparse_topk_blocks": 16,
    "sparse_init_block": 1,
    "sparse_main_rope_mode": "page2d",
    "sparse_main_rope_tok_freq_mode": "resample",
    "sparse_rope_mode": "page2d",
    "sparse_rope_tok_freq_mode": "resample",
    "sparse_rope_tok_theta": 10000.0,
    "sparse_rope_theta": 500000.0,
    "sparse_rope_page_theta": 256.0,
    "sparse_rope_page_pairs": 1,
    "sparse_rope_page_mod": 2,
    "sparse_rope_page_span": 0,
}

V2_ROPE_SCALING = {
    "type": "yarn", "rope_type": "yarn", "factor": 4.0,
    "original_max_position_embeddings": 32768, "attention_factor": 1.0,
    "beta_fast": 32, "beta_slow": 1,
}

# Appended to the common config in format-v2 files; matches uocr_v2_config_record.
# Variant, page ID, block/index dimensions, local/remote/top-M budgets,
# token/page theta, page pairs/modulus, crop cap, scale/prior floor/sigma.
V2_CONFIG_VALUES = (2, 128827, 32, 128, 128, 12, 16, 10000, 256, 1, 2, 16, 128.0, 0.0001, 1.0)
