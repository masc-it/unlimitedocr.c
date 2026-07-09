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
