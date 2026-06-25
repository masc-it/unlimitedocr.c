"""Opt-in performance smoke helpers for the Metal public generation path.

This module is intentionally measurement-only: it does not tune kernels or change
runtime behavior.  The goal is to make model-open time, warm first-token latency,
single-request generation throughput, and native memory peaks easy to capture in
a repeatable way on constrained Metal targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from .ffi import Engine, EngineOptions, MemoryReport
from .frontend import ImageInput, MODEL_VOCAB_SIZE, PreparedRequest, prepare_image
from .ocr import decode_generated_ids, default_resource_path

Clock = Callable[[], float]


@dataclass(frozen=True)
class PerfSmokeResult:
    """Timing/memory summary for one opt-in public Metal smoke run."""

    backend: str
    preset: str
    request_tokens: int
    visual_tokens: int
    views: int
    warmup_runs: int
    model_open_seconds: float
    warmup_seconds: tuple[float, ...]
    first_token_seconds: float
    generation_seconds: float
    generated_tokens: int
    generation_tokens_per_second: float
    peak_memory_bytes: int
    live_memory_bytes: int
    estimated_total_bytes: int
    recommended_working_set_bytes: int
    memory_budget_bytes: int
    generated_ids: tuple[int, ...]
    decoded_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "preset": self.preset,
            "request_tokens": self.request_tokens,
            "visual_tokens": self.visual_tokens,
            "views": self.views,
            "warmup_runs": self.warmup_runs,
            "model_open_seconds": self.model_open_seconds,
            "warmup_seconds": list(self.warmup_seconds),
            "first_token_seconds": self.first_token_seconds,
            "generation_seconds": self.generation_seconds,
            "generated_tokens": self.generated_tokens,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "live_memory_bytes": self.live_memory_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
            "recommended_working_set_bytes": self.recommended_working_set_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "generated_ids": list(self.generated_ids),
            "decoded_text": self.decoded_text,
        }

    def summary(self) -> str:
        peak_gib = self.peak_memory_bytes / float(1 << 30)
        return (
            f"backend={self.backend} preset={self.preset} tokens={self.request_tokens} "
            f"visual={self.visual_tokens} views={self.views} open={self.model_open_seconds:.3f}s "
            f"first_token={self.first_token_seconds:.3f}s generation={self.generation_seconds:.3f}s "
            f"generated={self.generated_tokens} tok/s={self.generation_tokens_per_second:.3f} "
            f"peak={peak_gib:.3f}GiB ids={list(self.generated_ids)} decoded={self.decoded_text!r}"
        )


def deterministic_perf_image(width: int = 160, height: int = 96, *, phase: int = 0) -> Image.Image:
    """Return a tiny deterministic RGB image; frontend padding expands it."""

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = ((x[None, :].astype(np.uint16) + phase) % 256).astype(np.uint8)
    pixels[..., 1] = ((y.astype(np.uint16) + 2 * phase) % 256).astype(np.uint8)
    pixels[..., 2] = ((pixels[..., 0].astype(np.uint16) + pixels[..., 1].astype(np.uint16)) // 2).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def generation_tokens_per_second(generated_tokens: int, elapsed_seconds: float) -> float:
    if generated_tokens <= 0:
        return 0.0
    if elapsed_seconds <= 0.0:
        return float("inf")
    return float(generated_tokens) / float(elapsed_seconds)


def _timed(clock: Clock, callback: Callable[[], Any]) -> tuple[float, Any]:
    start = clock()
    result = callback()
    elapsed = clock() - start
    if elapsed < 0.0:
        raise RuntimeError("perf clock moved backwards")
    return elapsed, result


def _copy_single_output(outputs: list[NDArray[np.int32]]) -> NDArray[np.int32]:
    if len(outputs) != 1:
        raise RuntimeError(f"expected one generated output, got {len(outputs)}")
    values = np.ascontiguousarray(outputs[0], dtype=np.int32)
    if values.ndim != 1:
        raise RuntimeError(f"generated ids must be rank-1, got shape {values.shape}")
    if values.size and (np.any(values < 0) or np.any(values >= MODEL_VOCAB_SIZE)):
        raise RuntimeError("generated ids contain values outside the model vocabulary")
    return values


def _prepare_perf_request(
    image: ImageInput,
    *,
    preset: str,
    max_new_tokens: int,
    tokenizer_path: str | Path | None,
) -> PreparedRequest:
    return prepare_image(
        image,
        preset=preset,
        tokenizer_path=tokenizer_path,
        max_new_tokens=max_new_tokens,
        dtype=np.float16,
    )


def run_public_metal_perf_smoke(
    *,
    model_path: str | Path,
    image: ImageInput | None = None,
    preset: str = "base",
    tokenizer_path: str | Path | None = None,
    resource_path: str | Path | None = None,
    library_path: str | Path | None = None,
    decode_tokens: int = 2,
    warmup_runs: int = 1,
    memory_budget_bytes: int = (1 << 64) - 1,
    clock: Clock = perf_counter,
) -> PerfSmokeResult:
    """Run an opt-in full-model Metal perf smoke on the public image path.

    ``decode_tokens`` controls the measured single-request generation length.
    The reported throughput is intentionally a smoke metric for the current
    public API call; it includes the current per-call vision/prefill overhead and
    is primarily meant for regression tracking across local runs.
    """

    if decode_tokens < 1:
        raise ValueError("decode_tokens must be at least 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    input_image = image if image is not None else deterministic_perf_image()
    first_token_request = _prepare_perf_request(
        input_image,
        preset=preset,
        max_new_tokens=1,
        tokenizer_path=tokenizer_path,
    )
    decode_request = _prepare_perf_request(
        input_image,
        preset=preset,
        max_new_tokens=decode_tokens,
        tokenizer_path=tokenizer_path,
    )
    resolved_resource_path = str(resource_path) if resource_path is not None else default_resource_path("metal")

    def open_engine() -> Engine:
        return Engine(
            EngineOptions(
                model_path=str(model_path),
                backend="metal",
                resource_path=resolved_resource_path,
                max_batch=1,
                max_prompt_tokens=max(first_token_request.n_tokens, decode_request.n_tokens),
                max_gen_tokens=decode_tokens,
                memory_budget_bytes=memory_budget_bytes,
            ),
            library_path=library_path,
        )

    open_seconds, engine = _timed(clock, open_engine)
    try:
        warmup_seconds: list[float] = []
        for _ in range(warmup_runs):
            elapsed, warm_outputs = _timed(clock, lambda: engine.generate_prepared(first_token_request))
            _copy_single_output(warm_outputs)
            warmup_seconds.append(elapsed)

        first_token_seconds, first_outputs = _timed(clock, lambda: engine.generate_prepared(first_token_request))
        _copy_single_output(first_outputs)

        generation_seconds, generated_outputs = _timed(clock, lambda: engine.generate_prepared(decode_request))
        generated = _copy_single_output(generated_outputs)
        memory = engine.memory_report()
        decoded = decode_generated_ids(generated, decode_request.tokenizer_path)
    finally:
        engine.close()

    return build_perf_smoke_result(
        backend="metal",
        preset=preset,
        request=decode_request,
        warmup_runs=warmup_runs,
        model_open_seconds=open_seconds,
        warmup_seconds=tuple(warmup_seconds),
        first_token_seconds=first_token_seconds,
        generation_seconds=generation_seconds,
        generated=generated,
        decoded_text=decoded,
        memory=memory,
    )


def build_perf_smoke_result(
    *,
    backend: str,
    preset: str,
    request: PreparedRequest,
    warmup_runs: int,
    model_open_seconds: float,
    warmup_seconds: tuple[float, ...],
    first_token_seconds: float,
    generation_seconds: float,
    generated: NDArray[np.int32],
    decoded_text: str,
    memory: MemoryReport,
) -> PerfSmokeResult:
    generated_values = np.ascontiguousarray(generated, dtype=np.int32)
    return PerfSmokeResult(
        backend=backend,
        preset=preset,
        request_tokens=request.n_tokens,
        visual_tokens=request.expected_visual_tokens,
        views=len(request.views),
        warmup_runs=warmup_runs,
        model_open_seconds=float(model_open_seconds),
        warmup_seconds=tuple(float(value) for value in warmup_seconds),
        first_token_seconds=float(first_token_seconds),
        generation_seconds=float(generation_seconds),
        generated_tokens=int(generated_values.size),
        generation_tokens_per_second=generation_tokens_per_second(int(generated_values.size), float(generation_seconds)),
        peak_memory_bytes=int(memory.total_peak_bytes),
        live_memory_bytes=int(memory.total_live_bytes),
        estimated_total_bytes=int(memory.estimated_total_bytes),
        recommended_working_set_bytes=int(memory.recommended_working_set_bytes),
        memory_budget_bytes=int(memory.memory_budget_bytes),
        generated_ids=tuple(int(value) for value in generated_values.tolist()),
        decoded_text=decoded_text,
    )
