from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
import pytest

from unlimitedocr_c.frontend import (
    GLOBAL_VIEW_SIZE,
    IMAGE_TOKEN,
    IMAGE_TOKEN_ID,
    LOCAL_VIEW_SIZE,
    MULTI_PROMPT,
    PATCH_SIZE,
    DOWNSAMPLE_RATIO,
    PRESETS,
    SINGLE_PROMPT,
    ImageView,
    PreparedRequest,
    Preset,
    load_tokenizer,
    prepare_image,
    prepare_pages,
    project_root,
)


def _context_file(name: str) -> Path:
    return project_root() / "data" / "context" / name


def _upstream_source() -> str:
    return _context_file("modeling_unlimitedocr.py").read_text(encoding="utf-8")


def _upstream_plain_prompt(prompt: str) -> str:
    conversation_path = _context_file("conversation.py")
    spec = importlib.util.spec_from_file_location("uocr_frontend_parity_conversation", conversation_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode

    conv = module.get_conv_template("plain")
    conv.set_system_message("")
    conv.append_message("<|User|>", prompt.strip())
    conv.append_message("<|Assistant|>", "")
    return conv.get_prompt().strip()


def _upstream_encode(tokenizer: Any, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
    # Mirrors modeling_unlimitedocr.py:text_encode.  The cached upstream uses
    # AutoTokenizer.encode(...), while tests use tokenizers.Tokenizer; both
    # expose the same ids for this local tokenizer.json.
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    if bos:
        ids = [0] + ids
    if eos:
        ids.append(1)
    return ids


def _upstream_find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
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


def _upstream_dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 2,
    max_num: int = 32,
    image_size: int = LOCAL_VIEW_SIZE,
) -> tuple[list[Image.Image], tuple[int, int]]:
    # Literal dependency-light port of modeling_unlimitedocr.py:dynamic_preprocess.
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    sorted_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _upstream_find_closest_aspect_ratio(
        aspect_ratio, sorted_ratios, orig_width, orig_height, image_size
    )
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
    assert len(processed_images) == blocks
    return processed_images, target_aspect_ratio


def _upstream_basic_image_transform(image: Image.Image, size: int) -> NDArray[np.float32]:
    # Mirrors ImageOps.pad(..., color=mean*255) followed by torchvision
    # ToTensor() and Normalize(mean=std=(0.5,0.5,0.5)), before upstream casts
    # tensors to bfloat16 for model execution.
    global_view = ImageOps.pad(image, (size, size), color=(127, 127, 127))
    hwc = np.asarray(global_view, dtype=np.float32) / 255.0
    chw = np.transpose(hwc, (2, 0, 1))
    return np.ascontiguousarray((chw - 0.5) / 0.5, dtype=np.float32)


def _upstream_global_visual_tokens(image_size: int) -> int:
    num_queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
    return (num_queries + 1) * num_queries + 1


def _gradient_image(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[:height, :width]
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = (xx * 3 + yy * 5) % 256
    pixels[..., 1] = (xx * 7 + yy * 11 + 17) % 256
    pixels[..., 2] = (xx * 13 + yy * 19 + 29) % 256
    return Image.fromarray(pixels)


def _upstream_single_image_reference(
    image: Image.Image,
    *,
    preset: Preset,
    prompt: str = SINGLE_PROMPT,
    tokenizer: Any,
) -> dict[str, Any]:
    formatted = _upstream_plain_prompt(prompt)
    text_splits = formatted.split(IMAGE_TOKEN)
    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    local_views: list[NDArray[np.float32]] = []
    global_views: list[NDArray[np.float32]] = []
    spatial_crop: list[tuple[int, int]] = []

    pil_image = ImageOps.exif_transpose(image).convert("RGB")
    for text_sep, split_image in zip(text_splits, [pil_image]):
        text_ids = _upstream_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str.extend(text_ids)
        images_seq_mask.extend([False] * len(text_ids))

        if preset.crop_mode:
            if split_image.size[0] <= LOCAL_VIEW_SIZE and split_image.size[1] <= LOCAL_VIEW_SIZE:
                crop_ratio = (1, 1)
                images_crop_raw: list[Image.Image] = []
            else:
                images_crop_raw, crop_ratio = _upstream_dynamic_preprocess(split_image)

            global_views.append(_upstream_basic_image_transform(split_image, preset.base_size))
            width_crop_num, height_crop_num = crop_ratio
            spatial_crop.append((width_crop_num, height_crop_num))
            if width_crop_num > 1 or height_crop_num > 1:
                local_views.extend(_upstream_basic_image_transform(crop, preset.image_size) for crop in images_crop_raw)

            num_queries = math.ceil((preset.image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
            num_queries_base = math.ceil((preset.base_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
            tokenized_image += [IMAGE_TOKEN_ID]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]) * (
                    num_queries * height_crop_num
                )
        else:
            if preset.image_size <= LOCAL_VIEW_SIZE:
                split_image = split_image.resize((preset.image_size, preset.image_size))
            global_views.append(_upstream_basic_image_transform(split_image, preset.image_size))
            spatial_crop.append((1, 1))
            num_queries = math.ceil((preset.image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
            tokenized_image += [IMAGE_TOKEN_ID]

        tokenized_str.extend(tokenized_image)
        images_seq_mask.extend([True] * len(tokenized_image))

    tail_ids = _upstream_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str.extend(tail_ids)
    images_seq_mask.extend([False] * len(tail_ids))
    tokenized_str = [0] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    return {
        "input_ids": np.asarray(tokenized_str, dtype=np.int32),
        "image_mask": np.asarray(images_seq_mask, dtype=np.uint8),
        "crop_grid": spatial_crop[0],
        "views": tuple(local_views + global_views),
    }


def _upstream_multi_page_reference(
    images: list[Image.Image],
    *,
    prompt: str = MULTI_PROMPT,
    image_size: int = GLOBAL_VIEW_SIZE,
    tokenizer: Any,
) -> dict[str, Any]:
    formatted = _upstream_plain_prompt(prompt)
    text_splits = formatted.split(IMAGE_TOKEN)
    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    global_views: list[NDArray[np.float32]] = []

    before_ids = _upstream_encode(tokenizer, text_splits[0], bos=False, eos=False)
    tokenized_str.extend(before_ids)
    images_seq_mask.extend([False] * len(before_ids))

    num_queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
    tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
    tokenized_image += [IMAGE_TOKEN_ID]
    for image in images:
        pil_image = ImageOps.exif_transpose(image).convert("RGB")
        if image_size <= LOCAL_VIEW_SIZE:
            pil_image = pil_image.resize((image_size, image_size))
        global_views.append(_upstream_basic_image_transform(pil_image, image_size))
        tokenized_str.extend(tokenized_image)
        images_seq_mask.extend([True] * len(tokenized_image))

    tail_ids = _upstream_encode(tokenizer, text_splits[1], bos=False, eos=False)
    tokenized_str.extend(tail_ids)
    images_seq_mask.extend([False] * len(tail_ids))
    tokenized_str = [0] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    return {
        "input_ids": np.asarray(tokenized_str, dtype=np.int32),
        "image_mask": np.asarray(images_seq_mask, dtype=np.uint8),
        "crop_grid": (1, 1),
        "views": tuple(global_views),
    }


def _assert_request_matches_upstream(request: PreparedRequest, reference: dict[str, Any]) -> None:
    np.testing.assert_array_equal(request.input_ids, reference["input_ids"])
    np.testing.assert_array_equal(request.image_mask, reference["image_mask"])
    assert (request.crop_grid_w, request.crop_grid_h) == reference["crop_grid"]
    assert request.expected_visual_tokens == int(reference["image_mask"].sum(dtype=np.uint64))
    assert int(np.count_nonzero(request.input_ids == IMAGE_TOKEN_ID)) == request.expected_visual_tokens
    assert np.unique(request.input_ids[request.image_mask.astype(bool)]).tolist() == [IMAGE_TOKEN_ID]

    reference_views = reference["views"]
    assert len(request.views) == len(reference_views)
    for view, expected_pixels in zip(request.views, reference_views, strict=True):
        assert isinstance(view, ImageView)
        assert view.pixels.dtype == np.float32
        assert view.pixels.flags.c_contiguous
        np.testing.assert_allclose(view.pixels, expected_pixels, rtol=0.0, atol=0.0)


def test_cached_upstream_source_contains_frontend_contract() -> None:
    source = _upstream_source()
    assert "tokenized_str = [bos_id] + tokenized_str" in source
    assert "images_seq_mask = [False] + images_seq_mask" in source
    assert "text_encode(tokenizer, text_sep, bos=False, eos=False)" in source
    assert "global_view = ImageOps.pad(image, (base_size, base_size)" in source
    assert "images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))" in source
    assert "tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base" in source
    assert "tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id])" in source
    assert "All images' token sequences are concatenated at that single <image> position" in source


@pytest.mark.parametrize(
    ("preset_name", "width", "height", "expected_grid", "expected_view_kinds"),
    [
        ("base", 321, 211, (1, 1), ("global",)),
        ("gundam", 320, 240, (1, 1), ("global",)),
        ("gundam", 1280, 640, (2, 1), ("local", "local", "global")),
    ],
)
def test_prepared_single_image_matches_cached_upstream_frontend(
    preset_name: str,
    width: int,
    height: int,
    expected_grid: tuple[int, int],
    expected_view_kinds: tuple[str, ...],
) -> None:
    tokenizer = load_tokenizer()
    image = _gradient_image(width, height)
    preset = PRESETS[preset_name]

    request = prepare_image(image, preset=preset_name, max_new_tokens=0, dtype=np.float32)
    reference = _upstream_single_image_reference(image, preset=preset, tokenizer=tokenizer)

    _assert_request_matches_upstream(request, reference)
    assert (request.crop_grid_w, request.crop_grid_h) == expected_grid
    assert tuple(view.kind for view in request.views) == expected_view_kinds


def test_prepared_multi_page_matches_cached_upstream_frontend() -> None:
    tokenizer = load_tokenizer()
    images = [_gradient_image(300, 500), _gradient_image(500, 300)]

    request = prepare_pages(images, max_new_tokens=0, dtype=np.float32)
    reference = _upstream_multi_page_reference(images, tokenizer=tokenizer)

    _assert_request_matches_upstream(request, reference)
    assert tuple(view.kind for view in request.views) == ("global", "global")
    assert request.expected_visual_tokens == 2 * _upstream_global_visual_tokens(GLOBAL_VIEW_SIZE)
