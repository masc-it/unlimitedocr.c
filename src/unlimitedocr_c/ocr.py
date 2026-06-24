"""Thin Python OCR convenience wrapper over the prepared-request C API.

This module intentionally stays small: Python prepares pixels/tokens and decodes
returned token ids, while the native engine owns Metal inference. Stable OCR/PDF
postprocessing ergonomics are deferred until after the fp16 image path is fully
validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ffi import Engine, EngineOptions
from .frontend import (
    EOS_TOKEN_ID,
    MODEL_VOCAB_SIZE,
    SINGLE_PROMPT,
    ImageInput,
    PreparedRequest,
    Preset,
    default_tokenizer_path,
    load_tokenizer,
    prepare_image,
    project_root,
)


@dataclass(frozen=True)
class GenerationResult:
    """Decoded output from one prepared request."""

    token_ids: NDArray[np.int32]
    text: str


def default_resource_path(backend: str = "metal") -> str | None:
    """Return the development resource path for backends that need one."""

    if backend.lower() == "metal":
        return str(project_root() / "src" / "backend" / "metal")
    return None


def decode_generated_ids(
    generated_ids: NDArray[np.integer[Any]] | list[int] | tuple[int, ...],
    tokenizer_path: str | Path | None = None,
) -> str:
    """Decode generated token ids with the HF tokenizer used by the frontend.

    The native runtime returns token ids only.  This mirrors the current upstream
    image path: decode without skipping special tokens, remove one trailing EOS
    marker if generation stopped on EOS, then trim surrounding whitespace.
    """

    ids = np.asarray(generated_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"generated ids must be rank-1, got shape {ids.shape}")
    if ids.size and (np.any(ids < 0) or np.any(ids >= MODEL_VOCAB_SIZE)):
        raise ValueError("generated ids contain values outside the model vocabulary")

    tokenizer = load_tokenizer(tokenizer_path or default_tokenizer_path())
    text = tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=False)
    eos_text = tokenizer.decode([EOS_TOKEN_ID], skip_special_tokens=False)
    if eos_text and text.endswith(eos_text):
        text = text[: -len(eos_text)]
    return text.strip()


def _decode_single_output(outputs: list[NDArray[np.int32]], tokenizer_path: str | Path) -> GenerationResult:
    if len(outputs) != 1:
        raise RuntimeError(f"expected one generated output, got {len(outputs)}")
    token_ids = np.ascontiguousarray(outputs[0], dtype=np.int32)
    return GenerationResult(token_ids=token_ids, text=decode_generated_ids(token_ids, tokenizer_path))


def generate_prepared(
    request: PreparedRequest,
    *,
    engine: Engine | None = None,
    model_path: str | Path | None = None,
    backend: str = "metal",
    resource_path: str | Path | None = None,
    memory_budget_bytes: int = 0,
    library_path: str | Path | None = None,
) -> GenerationResult:
    """Run one prepared request and decode the generated token ids.

    Pass an existing ``Engine`` to reuse a loaded model across calls.  Otherwise
    this helper opens a narrow single-request engine sized exactly for the
    prepared prompt and generation budget.
    """

    if engine is not None:
        return _decode_single_output(engine.generate_prepared(request), request.tokenizer_path)

    backend_name = backend.lower()
    if backend_name != "cpu-ref" and model_path is None:
        raise ValueError(f"model_path is required for backend {backend!r}")

    resolved_resource_path = resource_path
    if resolved_resource_path is None:
        resolved_resource_path = default_resource_path(backend_name)

    with Engine(
        EngineOptions(
            model_path=str(model_path) if model_path is not None else None,
            backend=backend_name,
            resource_path=str(resolved_resource_path) if resolved_resource_path is not None else None,
            max_batch=1,
            max_prompt_tokens=request.n_tokens,
            max_gen_tokens=request.max_new_tokens,
            memory_budget_bytes=memory_budget_bytes,
        ),
        library_path=library_path,
    ) as owned_engine:
        return _decode_single_output(owned_engine.generate_prepared(request), request.tokenizer_path)


def generate(
    image: ImageInput,
    *,
    model_path: str | Path | None = None,
    prompt: str = SINGLE_PROMPT,
    preset: str | Preset = "gundam",
    tokenizer_path: str | Path | None = None,
    max_new_tokens: int | None = 1,
    max_length: int = 32768,
    no_repeat_ngram_size: int = 35,
    no_repeat_window: int = 128,
    dtype: np.dtype[Any] | type[np.float16] | type[np.float32] = np.float16,
    engine: Engine | None = None,
    backend: str = "metal",
    resource_path: str | Path | None = None,
    memory_budget_bytes: int = 0,
    library_path: str | Path | None = None,
) -> GenerationResult:
    """Prepare one image, call the native public path, and decode the output.

    The default ``max_new_tokens=1`` keeps this convenience wrapper suitable for
    smoke/parity use on the current fp16 Metal path.  Pass a larger value (or
    ``None`` to derive it from ``max_length``) when running longer manual OCR
    experiments with an appropriately sized engine.
    """

    request = prepare_image(
        image,
        prompt=prompt,
        preset=preset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        no_repeat_window=no_repeat_window,
        dtype=dtype,
    )
    return generate_prepared(
        request,
        engine=engine,
        model_path=model_path,
        backend=backend,
        resource_path=resource_path,
        memory_budget_bytes=memory_budget_bytes,
        library_path=library_path,
    )
