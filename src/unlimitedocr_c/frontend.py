"""Python v1 frontend for Unlimited-OCR prepared requests.

This module mirrors the upstream Hugging Face remote-code input path but stops
before model execution. It owns PIL image handling, prompt rendering,
tokenization, visual-placeholder counting, and fixture serialization. The C core
receives only these prepared arrays and preprocessed views.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Literal, Sequence
import json
import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from tokenizers import Tokenizer

BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BPE_VOCAB_SIZE = 128000
MODEL_VOCAB_SIZE = 129280
ADDED_TOKEN_COUNT = 830

GLOBAL_VIEW_SIZE = 1024
LOCAL_VIEW_SIZE = 640
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
GLOBAL_QUERIES = math.ceil((GLOBAL_VIEW_SIZE // PATCH_SIZE) / DOWNSAMPLE_RATIO)
LOCAL_QUERIES = math.ceil((LOCAL_VIEW_SIZE // PATCH_SIZE) / DOWNSAMPLE_RATIO)
GLOBAL_VISUAL_TOKENS = (GLOBAL_QUERIES + 1) * GLOBAL_QUERIES + 1

SINGLE_PROMPT = "<image>document parsing."
MULTI_PROMPT = "<image>Multi page parsing."
EXPECTED_OUTPUT_IDS_NPY = "expected_output_ids.npy"
EXPECTED_TEXT_TXT = "expected_text.txt"

_TOKENIZER_CACHE_LOCK = RLock()
_TOKENIZER_CACHE: dict[str, tuple[tuple[Any, ...], Tokenizer]] = {}
_TOKENIZER_VALIDATION_CACHE: set[tuple[str, tuple[Any, ...]]] = set()
_EOS_TEXT_CACHE: dict[tuple[str, tuple[Any, ...]], str] = {}

ViewKind = Literal["global", "local"]
PixelFormat = Literal["f16_nchw", "f32_nchw"]
ImageInput = str | Path | Image.Image


@dataclass(frozen=True)
class Preset:
    name: str
    base_size: int
    image_size: int
    crop_mode: bool


PRESETS: dict[str, Preset] = {
    "gundam": Preset("gundam", base_size=1024, image_size=640, crop_mode=True),
    "base": Preset("base", base_size=1024, image_size=1024, crop_mode=False),
}


@dataclass(frozen=True)
class ImageView:
    pixels: NDArray[np.float16] | NDArray[np.float32]
    width: int
    height: int
    kind: ViewKind
    format: PixelFormat
    source_index: int

    @classmethod
    def from_pixels(cls, pixels: NDArray[np.float16] | NDArray[np.float32], *, kind: ViewKind, source_index: int) -> "ImageView":
        if pixels.ndim != 3 or pixels.shape[0] != 3:
            raise ValueError(f"view pixels must have shape [3,H,W], got {pixels.shape}")
        if pixels.dtype == np.float16:
            pixel_format: PixelFormat = "f16_nchw"
        elif pixels.dtype == np.float32:
            pixel_format = "f32_nchw"
        else:
            raise ValueError(f"view pixels must be float16 or float32, got {pixels.dtype}")
        return cls(
            pixels=np.ascontiguousarray(pixels),
            width=int(pixels.shape[2]),
            height=int(pixels.shape[1]),
            kind=kind,
            format=pixel_format,
            source_index=source_index,
        )


@dataclass(frozen=True)
class PreparedRequest:
    input_ids: NDArray[np.int32]
    image_mask: NDArray[np.uint8]
    views: tuple[ImageView, ...]
    crop_grid_w: int
    crop_grid_h: int
    mode: str
    prompt: str
    rendered_prompt: str
    max_new_tokens: int
    max_length: int | None
    no_repeat_ngram_size: int
    no_repeat_window: int
    tokenizer_path: str
    source_images: tuple[dict[str, Any], ...] = ()
    model_vocab_size: int = MODEL_VOCAB_SIZE
    expected_output_ids: NDArray[np.int32] | None = None
    expected_text: str | None = None

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def expected_visual_tokens(self) -> int:
        return int(self.image_mask.sum(dtype=np.uint64))

    def manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "version": 1,
            "mode": self.mode,
            "prompt": self.prompt,
            "rendered_prompt": self.rendered_prompt,
            "tokenizer_path": self.tokenizer_path,
            "model_vocab_size": self.model_vocab_size,
            "n_tokens": self.n_tokens,
            "image_mask_count": self.expected_visual_tokens,
            "crop_grid_w": self.crop_grid_w,
            "crop_grid_h": self.crop_grid_h,
            "max_new_tokens": self.max_new_tokens,
            "max_length": self.max_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "no_repeat_window": self.no_repeat_window,
            "source_images": list(self.source_images),
            "views": [
                {
                    "name": f"view_{i}",
                    "kind": view.kind,
                    "width": view.width,
                    "height": view.height,
                    "format": view.format,
                    "dtype": str(view.pixels.dtype),
                    "shape": list(view.pixels.shape),
                    "source_index": view.source_index,
                }
                for i, view in enumerate(self.views)
            ],
        }
        expected_output: dict[str, Any] = {}
        if self.expected_output_ids is not None:
            ids = np.asarray(self.expected_output_ids, dtype=np.int32)
            expected_output["ids_file"] = EXPECTED_OUTPUT_IDS_NPY
            expected_output["ids_dtype"] = str(ids.dtype)
            expected_output["ids_shape"] = list(ids.shape)
        if self.expected_text is not None:
            expected_output["text_file"] = EXPECTED_TEXT_TXT
        if expected_output:
            manifest["expected_output"] = expected_output
        return manifest


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_context_dir() -> Path:
    candidates = [Path.cwd() / "data" / "context", project_root() / "data" / "context"]
    for candidate in candidates:
        if (candidate / "tokenizer.json").exists():
            return candidate
    return candidates[-1]


def default_tokenizer_path() -> Path:
    return default_context_dir() / "tokenizer.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _resolve_tokenizer_path(tokenizer_path: str | Path | None = None) -> Path:
    path = Path(tokenizer_path) if tokenizer_path is not None else default_tokenizer_path()
    return path.expanduser().resolve()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _tokenizer_file_identity(path: Path) -> tuple[Any, ...]:
    config_path = path.with_name("config.json")
    config_identity: tuple[int, int, int, int] | None = _file_identity(config_path) if config_path.exists() else None
    return (_file_identity(path), config_identity)


def load_tokenizer(tokenizer_path: str | Path | None = None) -> Tokenizer:
    path = _resolve_tokenizer_path(tokenizer_path)
    identity = _tokenizer_file_identity(path)
    cache_key = str(path)
    with _TOKENIZER_CACHE_LOCK:
        cached = _TOKENIZER_CACHE.get(cache_key)
        if cached is not None and cached[0] == identity:
            return cached[1]

    tokenizer = Tokenizer.from_file(str(path))
    validate_tokenizer(tokenizer, path)
    with _TOKENIZER_CACHE_LOCK:
        _TOKENIZER_CACHE[cache_key] = (identity, tokenizer)
    return tokenizer


def validate_tokenizer(tokenizer: Tokenizer, tokenizer_path: str | Path | None = None) -> None:
    validation_key: tuple[str, tuple[Any, ...]] | None = None
    validation_cached = False
    path: Path | None = None
    if tokenizer_path is not None:
        path = _resolve_tokenizer_path(tokenizer_path)
        validation_key = (str(path), _tokenizer_file_identity(path))
        with _TOKENIZER_CACHE_LOCK:
            validation_cached = validation_key in _TOKENIZER_VALIDATION_CACHE

    base_vocab = tokenizer.get_vocab_size(with_added_tokens=False)
    if base_vocab != BPE_VOCAB_SIZE:
        raise ValueError(f"BPE vocab size mismatch: expected {BPE_VOCAB_SIZE}, got {base_vocab}")

    if tokenizer.token_to_id("<｜begin▁of▁sentence｜>") != BOS_TOKEN_ID:
        raise ValueError("BOS token id mismatch")
    if tokenizer.token_to_id("<｜end▁of▁sentence｜>") != EOS_TOKEN_ID:
        raise ValueError("EOS token id mismatch")
    if tokenizer.token_to_id("<｜▁pad▁｜>") != PAD_TOKEN_ID:
        raise ValueError("PAD token id mismatch")
    if tokenizer.token_to_id(IMAGE_TOKEN) != IMAGE_TOKEN_ID:
        raise ValueError("<image> token id mismatch")

    if validation_cached:
        return

    if path is not None:
        raw = _load_json(path)
        added = raw.get("added_tokens", [])
        if len(added) != ADDED_TOKEN_COUNT:
            raise ValueError(f"added token count mismatch: expected {ADDED_TOKEN_COUNT}, got {len(added)}")

        config_path = path.with_name("config.json")
        if config_path.exists():
            config = _load_json(config_path)
            vocab_size = int(config.get("vocab_size", -1))
            if vocab_size != MODEL_VOCAB_SIZE:
                raise ValueError(f"model vocab_size mismatch: expected {MODEL_VOCAB_SIZE}, got {vocab_size}")

    if validation_key is not None:
        with _TOKENIZER_CACHE_LOCK:
            _TOKENIZER_VALIDATION_CACHE.add(validation_key)


def cached_eos_text(tokenizer_path: str | Path | None = None) -> str:
    path = _resolve_tokenizer_path(tokenizer_path)
    identity = _tokenizer_file_identity(path)
    cache_key = (str(path), identity)
    with _TOKENIZER_CACHE_LOCK:
        cached = _EOS_TEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    tokenizer = load_tokenizer(path)
    eos_text = tokenizer.decode([EOS_TOKEN_ID], skip_special_tokens=False)
    with _TOKENIZER_CACHE_LOCK:
        _EOS_TEXT_CACHE[cache_key] = eos_text
    return eos_text


def format_messages_plain(conversations: Sequence[dict[str, Any]], system_prompt: str = "") -> str:
    del system_prompt  # plain template has an empty system template upstream
    parts: list[str] = []
    for message in conversations:
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(content)
    return "".join(parts).strip()


def render_prompt(prompt: str) -> str:
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    return format_messages_plain(conversation, system_prompt="")


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def find_closest_aspect_ratio(aspect_ratio: float,
                              target_ratios: Iterable[tuple[int, int]],
                              width: int,
                              height: int,
                              image_size: int) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image,
                       min_num: int = 2,
                       max_num: int = 32,
                       image_size: int = LOCAL_VIEW_SIZE) -> tuple[list[Image.Image], tuple[int, int]]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if len(processed_images) != blocks:
        raise RuntimeError("dynamic_preprocess produced an unexpected number of crops")
    return processed_images, target_aspect_ratio


def load_image(image: ImageInput) -> Image.Image:
    return _load_image_with_metadata(image, source_index=0)[0]


def _load_image_with_metadata(image: ImageInput, source_index: int) -> tuple[Image.Image, dict[str, Any]]:
    if isinstance(image, Image.Image):
        pil_image = image
        source_path: str | None = None
    else:
        source_path = str(image)
        pil_image = Image.open(image)
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    metadata = {
        "source_index": source_index,
        "path": source_path,
        "width": pil_image.size[0],
        "height": pil_image.size[1],
    }
    return pil_image, metadata


def preprocess_view(image: Image.Image,
                    size: int,
                    dtype: np.dtype[Any] | type[np.float16] | type[np.float32] = np.float16) -> NDArray[np.float16] | NDArray[np.float32]:
    if dtype not in (np.float16, np.float32, np.dtype("float16"), np.dtype("float32")):
        raise ValueError(f"unsupported view dtype {dtype}; expected float16 or float32")
    padded = ImageOps.pad(image, (size, size), color=(127, 127, 127))
    arr = np.asarray(padded, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    chw = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(chw.astype(dtype, copy=False))


def _image_placeholder_ids(count: int) -> list[int]:
    return [IMAGE_TOKEN_ID] * count


def global_visual_token_count(image_size: int = GLOBAL_VIEW_SIZE) -> int:
    queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
    return (queries + 1) * queries + 1


def crop_visual_token_count(grid_w: int, grid_h: int) -> int:
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError(f"invalid crop grid {grid_w}x{grid_h}")
    if grid_w == 1 and grid_h == 1:
        return GLOBAL_VISUAL_TOKENS
    return (LOCAL_QUERIES * grid_w + 1) * (LOCAL_QUERIES * grid_h) + GLOBAL_VISUAL_TOKENS


def _resolve_preset(preset: str | Preset) -> Preset:
    if isinstance(preset, Preset):
        return preset
    try:
        return PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}") from exc


def _build_token_sequence(tokenizer: Tokenizer, rendered_prompt: str, visual_counts: Sequence[int]) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
    text_splits = rendered_prompt.split(IMAGE_TOKEN)
    if len(text_splits) - 1 != len(visual_counts):
        raise ValueError(
            f"prompt has {len(text_splits) - 1} <image> placeholders but {len(visual_counts)} visual spans were provided"
        )

    token_ids: list[int] = []
    image_mask: list[int] = []
    for index, text in enumerate(text_splits[:-1]):
        text_ids = encode_text(tokenizer, text)
        token_ids.extend(text_ids)
        image_mask.extend([0] * len(text_ids))
        image_ids = _image_placeholder_ids(visual_counts[index])
        token_ids.extend(image_ids)
        image_mask.extend([1] * len(image_ids))

    tail_ids = encode_text(tokenizer, text_splits[-1])
    token_ids.extend(tail_ids)
    image_mask.extend([0] * len(tail_ids))

    token_ids = [BOS_TOKEN_ID] + token_ids
    image_mask = [0] + image_mask
    return np.asarray(token_ids, dtype=np.int32), np.asarray(image_mask, dtype=np.uint8)


def _max_new_tokens_from_total(max_length: int, prompt_tokens: int, explicit_max_new_tokens: int | None) -> int:
    if explicit_max_new_tokens is not None:
        if explicit_max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        return int(explicit_max_new_tokens)
    if max_length < prompt_tokens:
        return 0
    return int(max_length - prompt_tokens)


def prepare_image(image: ImageInput,
                  *,
                  prompt: str = SINGLE_PROMPT,
                  preset: str | Preset = "gundam",
                  tokenizer_path: str | Path | None = None,
                  max_length: int = 32768,
                  max_new_tokens: int | None = None,
                  no_repeat_ngram_size: int = 35,
                  no_repeat_window: int = 128,
                  dtype: np.dtype[Any] | type[np.float16] | type[np.float32] = np.float16) -> PreparedRequest:
    cfg = _resolve_preset(preset)
    tokenizer_path_resolved = _resolve_tokenizer_path(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path_resolved)
    rendered = render_prompt(prompt)
    pil_image, source_metadata = _load_image_with_metadata(image, source_index=0)

    views: list[ImageView] = []
    crop_grid = (1, 1)
    if cfg.crop_mode:
        if pil_image.size[0] <= LOCAL_VIEW_SIZE and pil_image.size[1] <= LOCAL_VIEW_SIZE:
            crop_grid = (1, 1)
            local_crops: list[Image.Image] = []
        else:
            local_crops, crop_grid = dynamic_preprocess(pil_image, image_size=cfg.image_size)
        if crop_grid[0] > 1 or crop_grid[1] > 1:
            for crop in local_crops:
                views.append(ImageView.from_pixels(preprocess_view(crop, cfg.image_size, dtype=dtype), kind="local", source_index=0))
        views.append(ImageView.from_pixels(preprocess_view(pil_image, cfg.base_size, dtype=dtype), kind="global", source_index=0))
        visual_count = crop_visual_token_count(*crop_grid)
    else:
        image_for_view = pil_image.resize((cfg.image_size, cfg.image_size)) if cfg.image_size <= LOCAL_VIEW_SIZE else pil_image
        views.append(ImageView.from_pixels(preprocess_view(image_for_view, cfg.image_size, dtype=dtype), kind="global", source_index=0))
        visual_count = global_visual_token_count(cfg.image_size)

    input_ids, image_mask = _build_token_sequence(tokenizer, rendered, [visual_count])
    request_max_new = _max_new_tokens_from_total(max_length, int(input_ids.shape[0]), max_new_tokens)
    return PreparedRequest(
        input_ids=input_ids,
        image_mask=image_mask,
        views=tuple(views),
        crop_grid_w=crop_grid[0],
        crop_grid_h=crop_grid[1],
        mode=cfg.name,
        prompt=prompt,
        rendered_prompt=rendered,
        max_new_tokens=request_max_new,
        max_length=max_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        no_repeat_window=no_repeat_window,
        tokenizer_path=str(tokenizer_path_resolved),
        source_images=(source_metadata,),
    )


def prepare_pages(images: Sequence[ImageInput],
                  *,
                  prompt: str = MULTI_PROMPT,
                  tokenizer_path: str | Path | None = None,
                  image_size: int = GLOBAL_VIEW_SIZE,
                  max_length: int = 32768,
                  max_new_tokens: int | None = None,
                  no_repeat_ngram_size: int = 35,
                  no_repeat_window: int = 1024,
                  dtype: np.dtype[Any] | type[np.float16] | type[np.float32] = np.float16) -> PreparedRequest:
    if not images:
        raise ValueError("prepare_pages requires at least one image")
    if image_size != GLOBAL_VIEW_SIZE:
        raise ValueError("v1 multi-page frontend supports only 1024x1024 global views")

    tokenizer_path_resolved = _resolve_tokenizer_path(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path_resolved)
    rendered = render_prompt(prompt)

    views: list[ImageView] = []
    source_images: list[dict[str, Any]] = []
    for source_index, image in enumerate(images):
        pil_image, source_metadata = _load_image_with_metadata(image, source_index=source_index)
        source_images.append(source_metadata)
        views.append(ImageView.from_pixels(preprocess_view(pil_image, image_size, dtype=dtype), kind="global", source_index=source_index))

    visual_counts = [global_visual_token_count(image_size) * len(views)]
    input_ids, image_mask = _build_token_sequence(tokenizer, rendered, visual_counts)
    request_max_new = _max_new_tokens_from_total(max_length, int(input_ids.shape[0]), max_new_tokens)
    return PreparedRequest(
        input_ids=input_ids,
        image_mask=image_mask,
        views=tuple(views),
        crop_grid_w=1,
        crop_grid_h=1,
        mode="multi-page-base",
        prompt=prompt,
        rendered_prompt=rendered,
        max_new_tokens=request_max_new,
        max_length=max_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        no_repeat_window=no_repeat_window,
        tokenizer_path=str(tokenizer_path_resolved),
        source_images=tuple(source_images),
    )


def prepare_text(prompt: str,
                 *,
                 tokenizer_path: str | Path | None = None,
                 max_length: int = 32768,
                 max_new_tokens: int | None = None,
                 no_repeat_ngram_size: int = 0,
                 no_repeat_window: int = 0) -> PreparedRequest:
    tokenizer_path_resolved = _resolve_tokenizer_path(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path_resolved)
    rendered = render_prompt(prompt)
    input_ids, image_mask = _build_token_sequence(tokenizer, rendered, [])
    request_max_new = _max_new_tokens_from_total(max_length, int(input_ids.shape[0]), max_new_tokens)
    return PreparedRequest(
        input_ids=input_ids,
        image_mask=image_mask,
        views=(),
        crop_grid_w=1,
        crop_grid_h=1,
        mode="text-only",
        prompt=prompt,
        rendered_prompt=rendered,
        max_new_tokens=request_max_new,
        max_length=max_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        no_repeat_window=no_repeat_window,
        tokenizer_path=str(tokenizer_path_resolved),
    )


def save_prepared_request(request: PreparedRequest, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if request.expected_output_ids is not None:
        expected_ids = np.ascontiguousarray(request.expected_output_ids, dtype=np.int32)
        if expected_ids.ndim != 1:
            raise ValueError(f"expected_output_ids must be a 1D int32 array, got shape {expected_ids.shape}")
        np.save(out / EXPECTED_OUTPUT_IDS_NPY, expected_ids)
    if request.expected_text is not None:
        (out / EXPECTED_TEXT_TXT).write_text(request.expected_text, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(request.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.save(out / "input_ids.npy", request.input_ids)
    np.save(out / "image_mask.npy", request.image_mask)
    np.savez(out / "views.npz", **{f"view_{i}": view.pixels for i, view in enumerate(request.views)})


def load_prepared_fixture(fixture_dir: str | Path) -> tuple[dict[str, Any], NDArray[np.int32], NDArray[np.uint8], dict[str, NDArray[Any]]]:
    root = Path(fixture_dir)
    manifest = _load_json(root / "manifest.json")
    input_ids = np.load(root / "input_ids.npy").astype(np.int32, copy=False)
    image_mask = np.load(root / "image_mask.npy").astype(np.uint8, copy=False)
    views_npz = np.load(root / "views.npz")
    views = {name: views_npz[name] for name in views_npz.files}
    return manifest, input_ids, image_mask, views
