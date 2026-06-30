from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

FULL_MODEL_PATH_ENVS = (
    "UOCR_MODEL_PATH",
    "UOCR_HF_DIR",
)
FULL_MODEL_DUMP_ENVS = (
    "UOCR_PROMPT_DUMP_DIR",
    "UOCR_LAYER_DUMP_DIR",
    "UOCR_IMAGE_EMBED_DUMP_DIR",
)
OPTIONAL_FIXTURE_DIR_PATTERNS = (
    "UOCR_IMAGE_PARITY_",
    "UOCR_FINAL_VISUAL_PARITY_",
    "UOCR_SAM_PARITY_",
    "UOCR_CLIP_PARITY_",
    "UOCR_PROJECTED_PARITY_",
)
FORBIDDEN_LOCAL_WEIGHT_PATHS = (
    "data/baidu/Unlimited-OCR/model-fp16.uocr",
    "data/baidu/Unlimited-OCR/model.safetensors",
)


def _test_sources() -> tuple[Path, ...]:
    return tuple(sorted(TESTS.glob("test_*.py"))) + tuple(sorted(TESTS.glob("test_*.c")))


def test_full_model_test_sources_require_manual_opt_in_guards() -> None:
    """Keep heavyweight model/conversion/parity tests out of normal pytest/ctest.

    Full fp16 model loading and HF conversion fixtures are valuable acceptance
    tests, but they must stay opt-in on developer machines with limited memory.
    A source file that references full-model paths/dumps must also reference the
    explicit large/perf gate used by its skip helper.
    """

    offenders: list[str] = []
    for path in _test_sources():
        text = path.read_text(encoding="utf-8")
        if not any(env in text for env in (*FULL_MODEL_PATH_ENVS, *FULL_MODEL_DUMP_ENVS)):
            continue
        if path.name == "test_perf.py":
            if "UOCR_RUN_PERF_TESTS" not in text or "skipif" not in text:
                offenders.append(f"{path.relative_to(ROOT)}: perf model test lacks UOCR_RUN_PERF_TESTS skip guard")
            continue
        if "UOCR_RUN_LARGE_TESTS" not in text:
            offenders.append(f"{path.relative_to(ROOT)}: full-model test lacks UOCR_RUN_LARGE_TESTS guard")
        if path.suffix == ".py" and "pytest.skip" not in text and "skipif" not in text:
            offenders.append(f"{path.relative_to(ROOT)}: Python full-model test lacks pytest skip path")
        if path.suffix == ".c" and "skipping" not in text:
            offenders.append(f"{path.relative_to(ROOT)}: native full-model test lacks skip diagnostic")
    assert offenders == []


def test_optional_fixture_directories_are_skip_guarded() -> None:
    """Parity fixture directories may be external, so unset envs must skip."""

    offenders: list[str] = []
    for path in _test_sources():
        text = path.read_text(encoding="utf-8")
        if not any(pattern in text for pattern in OPTIONAL_FIXTURE_DIR_PATTERNS):
            continue
        if path.suffix != ".py":
            offenders.append(f"{path.relative_to(ROOT)}: optional fixture directory policy only supports pytest files")
            continue
        if "pytest.skip" not in text:
            offenders.append(f"{path.relative_to(ROOT)}: optional fixture envs must have pytest.skip fallback")
        if "os.environ" not in text:
            offenders.append(f"{path.relative_to(ROOT)}: optional fixture envs must be read from the environment")
    assert offenders == []


def test_tests_do_not_hardcode_local_full_weight_paths() -> None:
    """Manual tests must use env vars, not a developer-local full model path."""

    offenders: list[str] = []
    for path in _test_sources():
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_LOCAL_WEIGHT_PATHS:
            if forbidden in text:
                offenders.append(f"{path.relative_to(ROOT)} hardcodes {forbidden}")
    assert offenders == []
