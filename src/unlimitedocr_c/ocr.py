"""Thin Python OCR convenience wrapper over the prepared-request C API.

This module intentionally stays small: Python prepares pixels/tokens and decodes
returned token ids, while the native engine owns Metal inference. Richer
PDF/postprocessing ergonomics are deferred until after the fp16 image path is
fully validated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from time import perf_counter
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .ffi import Engine, EngineOptions, ProfileEvent, ProfileReport
from .frontend import (
    EOS_TOKEN_ID,
    MODEL_VOCAB_SIZE,
    MULTI_PROMPT,
    SINGLE_PROMPT,
    ImageInput,
    PreparedRequest,
    Preset,
    default_tokenizer_path,
    load_tokenizer,
    prepare_image,
    prepare_pages,
    project_root,
)


@dataclass(frozen=True)
class GenerationResult:
    """Decoded output from one prepared request."""

    token_ids: NDArray[np.int32]
    text: str
    profile: ProfileReport | None = None


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


def _profile_env_enabled() -> bool:
    value = os.environ.get("UOCR_PROFILE")
    if value is None or value == "":
        return False
    return value.lower() not in {"0", "false", "no", "off"}


def _profile_enabled(profile: bool | None) -> bool:
    return _profile_env_enabled() if profile is None else bool(profile)


def _profile_event(name: str, elapsed_seconds: float) -> ProfileEvent:
    elapsed_ms = max(0.0, float(elapsed_seconds) * 1000.0)
    return ProfileEvent(name=name, calls=1, total_ms=elapsed_ms, min_ms=elapsed_ms, max_ms=elapsed_ms)


def _append_profile_events(report: ProfileReport | None, events: Sequence[ProfileEvent], *, enabled: bool) -> ProfileReport | None:
    if report is None:
        return None
    return replace(report, enabled=enabled or report.enabled, events=tuple(events) + report.events)


def _profile_report_from_engine(engine: object, events: Sequence[ProfileEvent], *, enabled: bool) -> ProfileReport | None:
    profile_report = getattr(engine, "profile_report", None)
    if not callable(profile_report):
        return None
    return _append_profile_events(profile_report(), events, enabled=enabled)


def _profile_reset_engine(engine: object) -> None:
    profile_reset = getattr(engine, "profile_reset", None)
    if callable(profile_reset):
        profile_reset()


def _emit_profile(report: ProfileReport | None, *, enabled: bool) -> None:
    if enabled and report is not None:
        print(json.dumps(report.as_dict(), sort_keys=True), file=sys.stderr)


def _cap_max_new_tokens(request: PreparedRequest, max_gen_tokens: int | None) -> PreparedRequest:
    """Cap a prepared request's generation budget to the chosen engine limit."""

    if max_gen_tokens is None:
        return request
    if max_gen_tokens < 0:
        raise ValueError("max_gen_tokens must be non-negative")
    cap = int(max_gen_tokens)
    if request.max_new_tokens <= cap:
        return request
    return replace(request, max_new_tokens=cap)


def generate_prepared(
    request: PreparedRequest,
    *,
    engine: Engine | None = None,
    model_path: str | Path | None = None,
    backend: str = "metal",
    resource_path: str | Path | None = None,
    memory_budget_bytes: int = 0,
    library_path: str | Path | None = None,
    profile: bool | None = None,
    _profile_events: Sequence[ProfileEvent] = (),
) -> GenerationResult:
    """Run one prepared request and decode the generated token ids.

    Pass an existing ``Engine`` to reuse a loaded model across calls.  Otherwise
    this helper opens a narrow single-request engine sized exactly for the
    prepared prompt and generation budget.
    """

    profile_on = _profile_enabled(profile)
    wall_events = list(_profile_events)

    if engine is not None:
        if profile_on:
            _profile_reset_engine(engine)
        start = perf_counter()
        outputs = engine.generate_prepared(request)
        wall_events.append(_profile_event("native.generate_prepared", perf_counter() - start))
        start = perf_counter()
        decoded = _decode_single_output(outputs, request.tokenizer_path)
        wall_events.append(_profile_event("python.decode", perf_counter() - start))
        report = _profile_report_from_engine(engine, wall_events, enabled=profile_on) if profile_on else None
        _emit_profile(report, enabled=profile_on)
        return replace(decoded, profile=report)

    backend_name = backend.lower()
    if backend_name != "cpu-ref" and model_path is None:
        raise ValueError(f"model_path is required for backend {backend!r}")

    resolved_resource_path = resource_path
    if resolved_resource_path is None:
        resolved_resource_path = default_resource_path(backend_name)

    start = perf_counter()
    owned_engine = Engine(
        EngineOptions(
            model_path=str(model_path) if model_path is not None else None,
            backend=backend_name,
            resource_path=str(resolved_resource_path) if resolved_resource_path is not None else None,
            max_batch=1,
            max_prompt_tokens=request.n_tokens,
            max_gen_tokens=request.max_new_tokens,
            memory_budget_bytes=memory_budget_bytes,
            profile=profile_on,
        ),
        library_path=library_path,
    )
    wall_events.append(_profile_event("engine.open", perf_counter() - start))
    try:
        start = perf_counter()
        outputs = owned_engine.generate_prepared(request)
        wall_events.append(_profile_event("native.generate_prepared", perf_counter() - start))
        start = perf_counter()
        decoded = _decode_single_output(outputs, request.tokenizer_path)
        wall_events.append(_profile_event("python.decode", perf_counter() - start))
        report = _profile_report_from_engine(owned_engine, wall_events, enabled=profile_on) if profile_on else None
    finally:
        owned_engine.close()
    _emit_profile(report, enabled=profile_on)
    return replace(decoded, profile=report)


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
    profile: bool | None = None,
) -> GenerationResult:
    """Prepare one image, call the native public path, and decode the output.

    The default ``max_new_tokens=1`` keeps this convenience wrapper suitable for
    smoke/parity use on the current fp16 Metal path.  Pass a larger value (or
    ``None`` to derive it from ``max_length``) when running longer manual OCR
    experiments with an appropriately sized engine.
    """

    profile_on = _profile_enabled(profile)
    profile_events: list[ProfileEvent] = []
    start = perf_counter()
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
    profile_events.append(_profile_event("python.preprocessing", perf_counter() - start))
    return generate_prepared(
        request,
        engine=engine,
        model_path=model_path,
        backend=backend,
        resource_path=resource_path,
        memory_budget_bytes=memory_budget_bytes,
        library_path=library_path,
        profile=profile_on,
        _profile_events=profile_events,
    )


def ocr_image(
    image: ImageInput,
    *,
    model_path: str | Path | None = None,
    prompt: str = SINGLE_PROMPT,
    preset: str | Preset = "gundam",
    tokenizer_path: str | Path | None = None,
    max_length: int = 32768,
    max_gen_tokens: int | None = 512,
    no_repeat_ngram_size: int = 35,
    ngram_window: int = 128,
    dtype: np.dtype[Any] | type[np.float16] | type[np.float32] = np.float16,
    engine: Engine | None = None,
    backend: str = "metal",
    resource_path: str | Path | None = None,
    memory_budget_bytes: int = 0,
    library_path: str | Path | None = None,
    profile: bool | None = None,
) -> GenerationResult:
    """Run single-image OCR with the upstream wrapper defaults.

    Upstream specifies ``max_length`` as total sequence length.  The C API takes
    ``max_new_tokens``, so this first prepares the prompt to compute
    ``max_length - prompt_tokens`` and then caps that value to ``max_gen_tokens``
    (the engine generation limit used when this helper opens its own engine).
    """

    profile_on = _profile_enabled(profile)
    profile_events: list[ProfileEvent] = []
    start = perf_counter()
    request = prepare_image(
        image,
        prompt=prompt,
        preset=preset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        max_new_tokens=None,
        no_repeat_ngram_size=no_repeat_ngram_size,
        no_repeat_window=ngram_window,
        dtype=dtype,
    )
    request = _cap_max_new_tokens(request, max_gen_tokens)
    profile_events.append(_profile_event("python.preprocessing", perf_counter() - start))
    return generate_prepared(
        request,
        engine=engine,
        model_path=model_path,
        backend=backend,
        resource_path=resource_path,
        memory_budget_bytes=memory_budget_bytes,
        library_path=library_path,
        profile=profile_on,
        _profile_events=profile_events,
    )


def ocr_pages(
    pages: Sequence[ImageInput],
    *,
    model_path: str | Path | None = None,
    prompt: str = MULTI_PROMPT,
    tokenizer_path: str | Path | None = None,
    max_length: int = 32768,
    max_gen_tokens: int | None = 512,
    no_repeat_ngram_size: int = 35,
    ngram_window: int = 1024,
    dtype: np.dtype[Any] | type[np.float16] | type[np.float32] = np.float16,
    engine: Engine | None = None,
    backend: str = "metal",
    resource_path: str | Path | None = None,
    memory_budget_bytes: int = 0,
    library_path: str | Path | None = None,
    profile: bool | None = None,
) -> GenerationResult:
    """Run multi-page OCR with the upstream wrapper defaults.

    Multi-page OCR always uses base/global 1024px views in the v1 frontend.  As
    with ``ocr_image()``, upstream's total ``max_length`` is converted by
    ``prepare_pages()`` to a prompt-relative generation budget, then capped to
    ``max_gen_tokens`` for the owned-engine path.
    """

    profile_on = _profile_enabled(profile)
    profile_events: list[ProfileEvent] = []
    start = perf_counter()
    request = prepare_pages(
        pages,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        max_new_tokens=None,
        no_repeat_ngram_size=no_repeat_ngram_size,
        no_repeat_window=ngram_window,
        dtype=dtype,
    )
    request = _cap_max_new_tokens(request, max_gen_tokens)
    profile_events.append(_profile_event("python.preprocessing", perf_counter() - start))
    return generate_prepared(
        request,
        engine=engine,
        model_path=model_path,
        backend=backend,
        resource_path=resource_path,
        memory_budget_bytes=memory_budget_bytes,
        library_path=library_path,
        profile=profile_on,
        _profile_events=profile_events,
    )


def _import_pymupdf():
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF
    except ModuleNotFoundError as exc:
        if exc.name == "fitz":
            raise ModuleNotFoundError(
                "ocr_pdf() requires PyMuPDF; install the optional 'pymupdf' package to rasterize PDFs"
            ) from exc
        raise
    return fitz


def pdf_to_images(pdf_path: str | Path, *, dpi: int = 300, out_dir: str | Path | None = None) -> list[Path]:
    """Rasterize a PDF to PNG page images using PyMuPDF.

    PyMuPDF is imported lazily so non-PDF OCR does not depend on it.  When
    ``out_dir`` is omitted, a temporary directory is created and intentionally
    retained for callers that use this helper directly.  ``ocr_pdf()`` uses its
    own temporary directory and cleans it up after generation.
    """

    if dpi <= 0:
        raise ValueError("dpi must be positive")

    fitz = _import_pymupdf()
    output_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="uocr_pdf_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        paths: list[Path] = []
        for index, page in enumerate(doc):
            path = output_dir / f"page_{index + 1:04d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(path))
            paths.append(path)
        return paths
    finally:
        doc.close()


def ocr_pdf(
    pdf_path: str | Path,
    *,
    model_path: str | Path | None = None,
    prompt: str = MULTI_PROMPT,
    tokenizer_path: str | Path | None = None,
    dpi: int = 300,
    page_output_dir: str | Path | None = None,
    max_length: int = 32768,
    max_gen_tokens: int | None = 512,
    no_repeat_ngram_size: int = 35,
    ngram_window: int = 1024,
    dtype: np.dtype[Any] | type[np.float16] | type[np.float32] = np.float16,
    engine: Engine | None = None,
    backend: str = "metal",
    resource_path: str | Path | None = None,
    memory_budget_bytes: int = 0,
    library_path: str | Path | None = None,
    profile: bool | None = None,
) -> GenerationResult:
    """Rasterize a PDF in Python and run multi-page OCR through the C engine.

    C still receives only a normal prepared multi-page image request; PDF I/O
    and rasterization stay on the Python side.  Use ``page_output_dir`` when the
    rendered PNG pages should be kept for debugging, otherwise a temporary
    directory is cleaned up after generation.
    """

    if page_output_dir is None:
        with tempfile.TemporaryDirectory(prefix="uocr_pdf_") as tmp_dir:
            pages = pdf_to_images(pdf_path, dpi=dpi, out_dir=tmp_dir)
            return ocr_pages(
                pages,
                model_path=model_path,
                prompt=prompt,
                tokenizer_path=tokenizer_path,
                max_length=max_length,
                max_gen_tokens=max_gen_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                ngram_window=ngram_window,
                dtype=dtype,
                engine=engine,
                backend=backend,
                resource_path=resource_path,
                memory_budget_bytes=memory_budget_bytes,
                library_path=library_path,
                profile=profile,
            )

    pages = pdf_to_images(pdf_path, dpi=dpi, out_dir=page_output_dir)
    return ocr_pages(
        pages,
        model_path=model_path,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        max_gen_tokens=max_gen_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        ngram_window=ngram_window,
        dtype=dtype,
        engine=engine,
        backend=backend,
        resource_path=resource_path,
        memory_budget_bytes=memory_budget_bytes,
        library_path=library_path,
        profile=profile,
    )
