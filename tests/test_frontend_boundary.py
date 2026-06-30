from __future__ import annotations

from pathlib import Path

from unlimitedocr_c.frontend import project_root


def test_native_tokenizer_image_frontend_is_deferred() -> None:
    """v1 keeps PIL/tokenizers in Python and passes prepared arrays to C/Metal."""
    root = project_root()
    placeholder = root / "src" / "frontend_native"
    readme = placeholder / "README.md"

    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "v1 frontend is intentionally Python-only" in text
    assert "Do not add native tokenizer or image preprocessing code here for v1" in text

    native_sources = sorted(
        path.relative_to(root).as_posix()
        for pattern in ("*.c", "*.cc", "*.cpp", "*.m", "*.mm", "*.h")
        for path in placeholder.rglob(pattern)
    )
    assert native_sources == []

    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "src/frontend_native" not in cmake
    assert "uocr_frontend_native" not in cmake


def test_deferred_native_frontend_is_recorded_under_later_work() -> None:
    plan = (project_root() / "docs" / "implementation_plan.md").read_text(encoding="utf-8")
    later_section = plan.split("## 23. Later / explicitly deferred work", maxsplit=1)[1]
    assert "Add native C tokenizer by adapting DS4's JoyAI/DeepSeek byte-level BPE" in later_section
    assert "Add native image loading/preprocessing only if a standalone C CLI is required" in later_section
