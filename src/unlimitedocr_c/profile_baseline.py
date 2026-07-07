"""Repeatable public Metal profile baseline capture for docs/test.png.

The implementation plan requires baseline profiles before the vision path is
rewritten.  This module keeps that capture reproducible: it fixes the image,
presets, max_length, OCR defaults, and output JSON schema while still allowing
callers to choose model/library/resource paths for their local build.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .ffi import ProfileEvent, ProfileReport
from .frontend import PRESETS, PreparedRequest, prepare_image, project_root
from .ocr import GenerationResult, default_resource_path, generate_prepared

DEFAULT_BASELINE_PRESETS: tuple[str, ...] = ("base", "gundam")
DEFAULT_BASELINE_MAX_LENGTH = 4096
DEFAULT_BASELINE_MAX_GEN_TOKENS = 512
DEFAULT_BASELINE_IMAGE = Path("docs/test.png")
DEFAULT_BASELINE_OUTPUT = Path("docs/baseline_profiles/docs-test-metal-profile.json")


@dataclass(frozen=True)
class BaselineCaseConfig:
    preset: str
    max_length: int
    max_gen_tokens: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "max_length": self.max_length,
            "max_gen_tokens": self.max_gen_tokens,
        }


@dataclass(frozen=True)
class BaselineProfileCase:
    config: BaselineCaseConfig
    elapsed_seconds: float
    text: str
    generated_ids: tuple[int, ...]
    request_tokens: int
    visual_tokens: int
    views: int
    profile: ProfileReport

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "text": self.text,
            "generated_ids": list(self.generated_ids),
            "request_tokens": self.request_tokens,
            "visual_tokens": self.visual_tokens,
            "views": self.views,
            "profile": self.profile.as_dict(),
        }


@dataclass(frozen=True)
class BaselineProfileSuite:
    created_at: str
    image_path: str
    model_path: str
    backend: str
    resource_path: str | None
    library_path: str | None
    tokenizer_path: str | None
    cases: tuple[BaselineProfileCase, ...]
    git_commit: str | None
    machine: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "created_at": self.created_at,
            "image_path": self.image_path,
            "model_path": self.model_path,
            "backend": self.backend,
            "resource_path": self.resource_path,
            "library_path": self.library_path,
            "tokenizer_path": self.tokenizer_path,
            "git_commit": self.git_commit,
            "machine": self.machine,
            "cases": [case.as_dict() for case in self.cases],
        }


def _repo_relative_or_abs(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _machine_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is not None:
        resolved = Path(model_path).expanduser()
    else:
        env_path = os.environ.get("UOCR_MODEL_PATH")
        if env_path:
            resolved = Path(env_path).expanduser()
        else:
            resolved = project_root() / "dist" / "unlimitedocr-fp16.uocr"
    if not resolved.exists():
        raise FileNotFoundError(
            f"model path {resolved} does not exist; pass --model-path or set UOCR_MODEL_PATH"
        )
    return resolved.resolve()


def _resolve_image_path(image_path: str | Path) -> Path:
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"baseline image {path} does not exist")
    return path.resolve()


def _validate_presets(presets: Sequence[str]) -> tuple[str, ...]:
    if not presets:
        raise ValueError("at least one preset is required")
    normalized: list[str] = []
    seen: set[str] = set()
    for preset in presets:
        value = preset.strip().lower()
        if value not in PRESETS:
            known = ", ".join(sorted(PRESETS))
            raise ValueError(f"unknown preset {preset!r}; expected one of: {known}")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _cap_request_max_new_tokens(request: PreparedRequest, max_gen_tokens: int | None) -> PreparedRequest:
    if max_gen_tokens is None or request.max_new_tokens <= max_gen_tokens:
        return request
    return replace(request, max_new_tokens=int(max_gen_tokens))


def _case_from_result(
    config: BaselineCaseConfig,
    request: PreparedRequest,
    elapsed_seconds: float,
    result: GenerationResult,
) -> BaselineProfileCase:
    if result.profile is None:
        raise RuntimeError(f"profile was not returned for preset {config.preset!r}")
    return BaselineProfileCase(
        config=config,
        elapsed_seconds=float(elapsed_seconds),
        text=result.text,
        generated_ids=tuple(int(value) for value in result.token_ids.tolist()),
        request_tokens=int(request.n_tokens),
        visual_tokens=int(request.expected_visual_tokens),
        views=len(request.views),
        profile=result.profile,
    )


def run_docs_test_profile_baselines(
    *,
    model_path: str | Path | None = None,
    image_path: str | Path = DEFAULT_BASELINE_IMAGE,
    presets: Sequence[str] = DEFAULT_BASELINE_PRESETS,
    output_path: str | Path | None = DEFAULT_BASELINE_OUTPUT,
    max_length: int = DEFAULT_BASELINE_MAX_LENGTH,
    max_gen_tokens: int | None = DEFAULT_BASELINE_MAX_GEN_TOKENS,
    tokenizer_path: str | Path | None = None,
    resource_path: str | Path | None = None,
    library_path: str | Path | None = None,
    memory_budget_bytes: int = 0,
    backend: str = "metal",
) -> BaselineProfileSuite:
    """Capture the implementation-plan baseline profiles for ``docs/test.png``.

    Defaults intentionally match the plan item: ``preset=base`` and
    ``preset=gundam`` on ``docs/test.png`` with ``max_length=4096`` and the
    public OCR defaults for generation settings.  The resulting JSON
    is stable enough to diff across local optimization runs.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if max_gen_tokens is not None and max_gen_tokens < 0:
        raise ValueError("max_gen_tokens must be non-negative when provided")
    resolved_model_path = _resolve_model_path(model_path)
    resolved_image_path = _resolve_image_path(image_path)
    normalized_presets = _validate_presets(presets)
    resolved_resource_path = resource_path if resource_path is not None else default_resource_path(backend)

    cases: list[BaselineProfileCase] = []
    for preset in normalized_presets:
        config = BaselineCaseConfig(
            preset=preset,
            max_length=max_length,
            max_gen_tokens=max_gen_tokens,
        )
        case_start = perf_counter()
        prep_start = perf_counter()
        request = prepare_image(
            resolved_image_path,
            preset=preset,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            max_new_tokens=None,
            dtype=np.float16,
        )
        request = _cap_request_max_new_tokens(request, max_gen_tokens)
        preprocessing_ms = (perf_counter() - prep_start) * 1000.0
        result = generate_prepared(
            request,
            model_path=resolved_model_path,
            backend=backend,
            resource_path=resolved_resource_path,
            memory_budget_bytes=memory_budget_bytes,
            library_path=library_path,
            profile=True,
            _profile_events=(
                ProfileEvent(
                    name="python.preprocessing",
                    calls=1,
                    total_ms=preprocessing_ms,
                    min_ms=preprocessing_ms,
                    max_ms=preprocessing_ms,
                ),
            ),
        )
        cases.append(_case_from_result(config, request, perf_counter() - case_start, result))

    suite = BaselineProfileSuite(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        image_path=_repo_relative_or_abs(resolved_image_path),
        model_path=_repo_relative_or_abs(resolved_model_path),
        backend=backend,
        resource_path=str(resolved_resource_path) if resolved_resource_path is not None else None,
        library_path=str(library_path) if library_path is not None else None,
        tokenizer_path=str(tokenizer_path) if tokenizer_path is not None else None,
        cases=tuple(cases),
        git_commit=_git_commit(),
        machine=_machine_info(),
    )

    if output_path is not None:
        out = Path(output_path).expanduser()
        if not out.is_absolute():
            out = project_root() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(suite.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return suite


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture docs/test.png public Metal profile baselines")
    parser.add_argument("--model-path", type=Path, default=None, help=".uocr model path (default: UOCR_MODEL_PATH or dist/unlimitedocr-fp16.uocr)")
    parser.add_argument("--image", type=Path, default=DEFAULT_BASELINE_IMAGE, help="baseline image path")
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE_OUTPUT, help="JSON output path")
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS), dest="presets", help="preset to run; repeatable (default: base and gundam)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_BASELINE_MAX_LENGTH, help="public OCR max_length")
    parser.add_argument("--max-gen-tokens", type=int, default=DEFAULT_BASELINE_MAX_GEN_TOKENS, help="public OCR max_gen_tokens cap (default matches OCR; 0 is useful only for smoke validation)")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="tokenizer directory/file override")
    parser.add_argument("--resource-path", type=Path, default=None, help="Metal resource path override")
    parser.add_argument("--library-path", type=Path, default=None, help="libunlimitedocr path override")
    parser.add_argument("--memory-budget-bytes", type=int, default=0, help="memory admission budget override")
    parser.add_argument("--backend", default="metal", choices=("metal", "cpu-ref"), help="backend to run; baseline default is metal")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite = run_docs_test_profile_baselines(
        model_path=args.model_path,
        image_path=args.image,
        presets=tuple(args.presets) if args.presets else DEFAULT_BASELINE_PRESETS,
        output_path=args.output,
        max_length=args.max_length,
        max_gen_tokens=args.max_gen_tokens,
        tokenizer_path=args.tokenizer_path,
        resource_path=args.resource_path,
        library_path=args.library_path,
        memory_budget_bytes=args.memory_budget_bytes,
        backend=args.backend,
    )
    print(json.dumps(suite.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
