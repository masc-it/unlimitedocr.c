"""User-facing UnlimitedOCR API.

This module hides the low-level prepared-request/native-engine plumbing behind a
small class intended for notebooks and applications:

    from unlimitedocr_c import UnlimitedOCR
    ocr = UnlimitedOCR()
    text = ocr.generate("page.png")
"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
import os
from pathlib import Path
import sys
import struct
import threading
from typing import IO, Literal
from urllib.request import Request, urlopen

from PIL import Image

from .convert import QPROFILE_IDS, build_dry_run_plan, write_uocr_model
from .ffi import Engine, EngineOptions
from .frontend import (
    PRESETS,
    default_tokenizer_path,
    prepare_image,
)
from .ocr import decode_generated_ids, default_resource_path
from .model_variants import ModelName, MODEL_REPOSITORIES, get_model_variant, model_cache_dir, source_model

ProfileName = Literal["base", "gundam"]
ImageSource = str | os.PathLike[str] | bytes | bytearray | memoryview | IO[bytes] | Image.Image

DEFAULT_MODEL_FILENAME = "unlimitedocr-fp16.uocr"
DEFAULT_Q8_MODEL_FILENAME = "unlimitedocr-q8.uocr"
DEFAULT_Q4_MODEL_FILENAME = "unlimitedocr-q4.uocr"
DEFAULT_SOURCE_MODEL_REPO_ID = MODEL_REPOSITORIES["v1"]
DEFAULT_MAX_LENGTH = 32768
class ModelResolutionError(FileNotFoundError):
    """Raised when a usable `.uocr` model cannot be found or downloaded."""


def default_cache_dir() -> Path:
    """Return the cache directory used for downloaded/converted model files."""

    env = os.environ.get("UOCR_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "unlimitedocr"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "unlimitedocr"
    return Path.home() / ".cache" / "unlimitedocr"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _require_existing_model(path: str | Path, *, source: str, model: ModelName = "v1", qprofile: str | None = None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise ModelResolutionError(f"{source} points to a missing model file: {resolved}")
    if not resolved.is_file():
        raise ModelResolutionError(f"{source} must be a .uocr file, got: {resolved}")
    # Full file validation remains native. Reject a recognized but incompatible
    # variant here, before selecting its tokenizer or allocating an engine.
    with resolved.open("rb") as f:
        prefix = f.read(24)
    if len(prefix) >= 8 and prefix[:4] == b"UOCR":
        version = struct.unpack_from("<I", prefix, 4)[0]
        if version != get_model_variant(model).format_version:
            raise ModelResolutionError(f"{source} has format version {version}, incompatible with model={model!r}")
        if qprofile is not None and len(prefix) == 24 and struct.unpack_from("<I", prefix, 20)[0] != QPROFILE_IDS[qprofile]:
            raise ModelResolutionError(f"{source} has incompatible quantization; expected {qprofile}")
    return resolved.resolve()


def _candidate_model_paths(cache_dir: Path, filename: str, model: ModelName = "v1", *, qprofile: str = "fp16") -> list[Path]:
    paths = [model_cache_dir(cache_dir, model) / qprofile / filename]
    if model == "v1" and source_model(model) == (MODEL_REPOSITORIES["v1"], None):
        paths.extend([cache_dir / filename, cache_dir / "models" / filename])
    return paths


def _hf_local_files_only() -> bool:
    return _env_flag("UOCR_HF_LOCAL_FILES_ONLY", default=False)


def _quant_profile_for(quant: str | None) -> str:
    """Map a high-level ``quant`` option to a converter qprofile id."""
    if quant is None or quant == "":
        return "fp16"
    normalized = quant.strip().lower()
    if normalized in {"q8", "q8_0", "mixed-q8_0"}:
        return "mixed-q8_0"
    if normalized in {"q4", "q4_0", "mixed-q4"}:
        return "mixed-q4"
    if normalized in {"fp16", "f16", "none"}:
        return "fp16"
    raise ValueError(
        f"unsupported quant={quant!r}; expected one of 'q4', 'q8', 'fp16', or None"
    )


def _model_filename_for(quant: str | None) -> str:
    """Return the cache filename for a quant option."""
    qprofile = _quant_profile_for(quant)
    if qprofile == "mixed-q8_0":
        return os.environ.get("UOCR_Q8_MODEL_FILENAME", DEFAULT_Q8_MODEL_FILENAME)
    if qprofile == "mixed-q4":
        return os.environ.get("UOCR_Q4_MODEL_FILENAME", DEFAULT_Q4_MODEL_FILENAME)
    return os.environ.get("UOCR_MODEL_FILENAME", DEFAULT_MODEL_FILENAME)


def _download_and_convert_source_model(
    cache_dir: Path,
    filename: str,
    *,
    qprofile: str = "fp16",
    quant_cfg_path: str | Path | None = None,
    force: bool = False,
    model: ModelName = "v1",
) -> Path:
    from huggingface_hub import snapshot_download
    from .quant_cfg import load_default_quant_config, load_quant_config

    cache_dir = model_cache_dir(cache_dir, model) / qprofile
    out_path = cache_dir / filename
    if out_path.exists() and not force:
        return _require_existing_model(out_path, source="cached model", model=model, qprofile=qprofile)

    source_repo_id, revision = source_model(model)
    snapshot = Path(
        snapshot_download(
            repo_id=source_repo_id,
            revision=revision,
            allow_patterns=[
                "config.json",
                "processor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
                "model.safetensors",
                "model-*.safetensors",
            ],
            local_files_only=_hf_local_files_only(),
        )
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    stats_path = out_path.with_suffix(out_path.suffix + ".conversion-stats.json")
    if tmp_path.exists():
        tmp_path.unlink()
    quant_cfg = load_quant_config(quant_cfg_path) if quant_cfg_path is not None else load_default_quant_config()
    plan = build_dry_run_plan(snapshot, qprofile=qprofile, quant_cfg=quant_cfg)
    if plan.model_variant != model:
        raise ValueError(f"source checkpoint is {plan.model_variant}, but model={model!r} was requested")
    write_uocr_model(plan, tmp_path, overwrite=True, stats_path=stats_path)
    tmp_path.replace(out_path)
    return out_path.resolve()


def resolve_model_path(
    model_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
    quant: str | None = None,
    quant_cfg_path: str | Path | None = None,
    force_reconvert: bool = False,
    model: ModelName = "v1",
) -> Path:
    """Resolve a `.uocr` model path using env, local cache, and HF conversion.

    Resolution order:
    1. explicit ``model_path``
    2. ``UOCR_MODEL_PATH``
    3. model/repository/revision cache (filename depends on ``quant``), then
       legacy cache paths for the default v1 source only
    4. Hugging Face download of the upstream checkpoint followed by conversion

    ``quant`` selects the conversion profile when downloading/converting:
    ``None``/``"fp16"`` keeps the fp16 baseline; ``"q8"`` produces a
    runtime-safe mixed-Q8_0 model and ``"q4"`` a mixed-q4 model (routed MoE
    experts Q4_0, remaining decoder modules Q8_0), both gated by
    ``configs/quant-cfg.yaml``.

    ``force_reconvert`` bypasses the selected cache entry and atomically replaces
    it after conversion succeeds; failed conversions leave existing files intact.
    It requires ``download=True`` and has no effect when
    ``model_path``/``UOCR_MODEL_PATH`` is set.
    """

    get_model_variant(model)
    if model_path is not None:
        return _require_existing_model(model_path, source="model_path", model=model)

    env_model_path = os.environ.get("UOCR_MODEL_PATH")
    if env_model_path:
        return _require_existing_model(env_model_path, source="UOCR_MODEL_PATH", model=model)

    qprofile = _quant_profile_for(quant)
    filename = _model_filename_for(quant)
    if not filename or Path(filename).name != filename:
        raise ValueError("UOCR model filename must be a filename, not a path; use UOCR_MODEL_PATH for explicit files")
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    candidates = _candidate_model_paths(resolved_cache_dir, filename, model, qprofile=qprofile)
    if force_reconvert and not download:
        raise ValueError("force_reconvert requires download=True")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not force_reconvert and candidate.exists() and candidate.is_file():
            return _require_existing_model(candidate, source="cached model", model=model, qprofile=qprofile)

    if not download:
        searched = "\n  ".join(str(path) for path in candidates)
        raise ModelResolutionError(
            "could not find an UnlimitedOCR .uocr model. Searched:\n  "
            + searched
            + "\nSet UOCR_MODEL_PATH or enable download."
        )

    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _download_and_convert_source_model(
            resolved_cache_dir,
            filename,
            qprofile=qprofile,
            quant_cfg_path=quant_cfg_path,
            force=force_reconvert,
            model=model,
        )
    except Exception as exc:  # pragma: no cover - network/large-conversion path
        raise ModelResolutionError(
            "could not prepare the UnlimitedOCR model automatically. "
            f"source checkpoint download/conversion failed: {exc}. "
            "Set UOCR_MODEL_PATH to a local .uocr file to bypass downloads."
        ) from exc


def _image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.copy()


def _read_file_like(file_obj: IO[bytes]) -> bytes:
    try:
        position = file_obj.tell()
    except (AttributeError, OSError):
        position = None
    data = file_obj.read()
    if position is not None:
        try:
            file_obj.seek(position)
        except OSError:
            pass
    return data


def _decode_base64_image(value: str) -> Image.Image:
    text = value.strip()
    if text.startswith("data:"):
        header, sep, payload = text.partition(",")
        if sep != "," or ";base64" not in header.lower():
            raise ValueError("data URI image input must contain a base64 payload")
        text = payload
    compact = "".join(text.split())
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("string image input is neither an existing path/URL nor valid base64 image data") from exc
    return _image_from_bytes(data)


def _existing_path_from_string(value: str) -> Path | None:
    try:
        path = Path(value).expanduser()
        return path if path.exists() else None
    except OSError:
        # A long base64 string can exceed filesystem path limits; treat it as
        # non-path input and let base64 decoding validate it.
        return None


def load_user_image(image: ImageSource, *, timeout: float = 30.0) -> Image.Image:
    """Load an image from a path, URL, base64 string, bytes/BytesIO, or PIL image."""

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray, memoryview)):
        return _image_from_bytes(bytes(image))
    if hasattr(image, "read") and not isinstance(image, (str, os.PathLike)):
        return _image_from_bytes(_read_file_like(image))

    if isinstance(image, os.PathLike):
        with Image.open(image) as pil_image:
            pil_image.load()
            return pil_image.copy()

    if isinstance(image, str):
        text = image.strip()
        if text.startswith(("http://", "https://")):
            request = Request(text, headers={"User-Agent": "unlimitedocr-c/0.1"})
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-requested URL input
                return _image_from_bytes(response.read())
        path = _existing_path_from_string(text)
        if path is not None:
            with Image.open(path) as pil_image:
                pil_image.load()
                return pil_image.copy()
        return _decode_base64_image(text)

    raise TypeError(
        "image must be a path/URL/base64 string, bytes, BytesIO/file-like object, or PIL.Image.Image"
    )


class UnlimitedOCR:
    """High-level, cache-aware Unlimited-OCR model wrapper.

    Typical use is intentionally small:

    >>> from unlimitedocr_c import UnlimitedOCR
    >>> ocr = UnlimitedOCR()  # resolves model, opens engine, runs GPU warmup
    >>> text = ocr.generate("page.png")

    The constructor opens the native engine eagerly. V1 also runs synthetic
    GPU warmup; v2 warms pipelines on its first real image request. The engine is
    resized automatically if a later request needs more prompt/gen capacity.

    Thread safety: one instance may be shared by several threads.  A per-object
    lock serializes engine selection, native generation, replacement, and
    ``close()``, so shared calls execute one at a time and ``close()`` waits
    for the active call.  Image loading and request preparation run before the
    lock and can overlap across threads.  Applications that want several
    inference calls in flight create one instance per execution lane.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        download: bool = True,
        backend: str = "metal",
        memory_budget_bytes: int = 0,
        library_path: str | Path | None = None,
        resource_path: str | Path | None = None,
        request_timeout: float = 30.0,
        quant: str | None = None,
        quant_cfg_path: str | Path | None = None,
        force_reconvert: bool = False,
        model: ModelName = "v1",
    ) -> None:
        get_model_variant(model)
        self._model = model
        self._model_path = resolve_model_path(
            model_path,
            cache_dir=cache_dir,
            download=download,
            quant=quant,
            quant_cfg_path=quant_cfg_path,
            force_reconvert=force_reconvert,
            model=model,
        )
        self._backend = backend.lower()
        self._memory_budget_bytes = int(memory_budget_bytes)
        self._library_path = str(library_path) if library_path is not None else None
        resolved_resource_path = resource_path if resource_path is not None else default_resource_path(self._backend)
        self._resource_path = str(resolved_resource_path) if resolved_resource_path is not None else None
        self._tokenizer_path = str(default_tokenizer_path(model=model, local_files_only=not download or _hf_local_files_only()))
        self._request_timeout = float(request_timeout)
        self._lock = threading.RLock()
        self._engine: Engine | None = None
        self._engine_prompt_capacity = 0
        self._engine_gen_capacity = 0

        # V1 keeps its existing eager capacity. V2 must not eagerly allocate
        # the entire 131072-token context; grow to actual window budgets later.
        print(f"model: {self._model_path}", file=sys.stderr)
        self._ensure_engine(
            prompt_capacity=DEFAULT_MAX_LENGTH if model == "v1" else 4096,
            gen_capacity=DEFAULT_MAX_LENGTH if model == "v1" else 512,
        )

    @property
    def model(self) -> ModelName:
        """Selected checkpoint variant, independent of image/quantization profile."""
        return self._model

    @property
    def model_path(self) -> Path:
        """Resolved local `.uocr` model path."""

        return self._model_path

    @property
    def resource_path(self) -> str | None:
        """Resolved Metal resource path, if the backend uses one."""

        return self._resource_path

    def close(self) -> None:
        """Close the lazily opened native engine, if any."""

        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
                self._engine_prompt_capacity = 0
                self._engine_gen_capacity = 0

    def __enter__(self) -> "UnlimitedOCR":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def __del__(self) -> None:
        self.close()

    def _open_engine(self, *, prompt_capacity: int, gen_capacity: int) -> Engine:
        return Engine(
            EngineOptions(
                model_path=str(self._model_path) if self._backend != "cpu-ref" or self._model == "v2" else None,
                backend=self._backend,
                resource_path=self._resource_path,
                max_batch=1,
                max_prompt_tokens=prompt_capacity,
                max_gen_tokens=gen_capacity,
                memory_budget_bytes=self._memory_budget_bytes,
                warmup=self._model == "v1",  # The legacy synthetic warmup prompt has no <PAGE>.
            ),
            library_path=self._library_path,
        )

    def _ensure_engine(self, *, prompt_capacity: int, gen_capacity: int) -> Engine:
        with self._lock:
            if (
                self._engine is not None
                and prompt_capacity <= self._engine_prompt_capacity
                and gen_capacity <= self._engine_gen_capacity
            ):
                return self._engine

            self.close()
            self._engine = self._open_engine(prompt_capacity=prompt_capacity, gen_capacity=gen_capacity)
            self._engine_prompt_capacity = prompt_capacity
            self._engine_gen_capacity = gen_capacity
            return self._engine

    def generate(self, image: ImageSource, profile: ProfileName = "base") -> str:
        """Run OCR for one image and return the decoded text.

        Parameters
        ----------
        image:
            Local path, HTTP(S) URL, base64/data-URI string, bytes/BytesIO, or a
            ``PIL.Image.Image``.
        profile:
            ``"base"`` for the standard 1024px global view, or ``"gundam"`` for
            the crop-aware profile.
        """

        if profile not in PRESETS:
            raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PRESETS)}")

        pil_image = load_user_image(image, timeout=self._request_timeout)
        request = prepare_image(
            pil_image,
            model=self._model,
            preset=profile,
            tokenizer_path=self._tokenizer_path,
            max_length=DEFAULT_MAX_LENGTH,
            max_new_tokens=None,
        )
        with self._lock:
            engine = self._ensure_engine(
                prompt_capacity=max(1, request.n_tokens),
                gen_capacity=max(1, request.max_new_tokens),
            )
            outputs = engine.generate_prepared(request)
        if len(outputs) != 1:
            raise RuntimeError(f"expected one generated output, got {len(outputs)}")
        return decode_generated_ids(outputs[0], request.tokenizer_path)


__all__ = [
    "DEFAULT_MODEL_FILENAME",
    "DEFAULT_Q8_MODEL_FILENAME",
    "DEFAULT_SOURCE_MODEL_REPO_ID",
    "ImageSource",
    "ModelResolutionError",
    "ModelName",
    "ProfileName",
    "UnlimitedOCR",
    "default_cache_dir",
    "load_user_image",
    "resolve_model_path",
]
