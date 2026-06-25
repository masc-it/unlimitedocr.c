from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
import sys

import numpy as np
from PIL import Image
import pytest

from unlimitedocr_c.frontend import (
    ADDED_TOKEN_COUNT,
    BOS_TOKEN_ID,
    BPE_VOCAB_SIZE,
    EOS_TOKEN_ID,
    EXPECTED_OUTPUT_IDS_NPY,
    EXPECTED_TEXT_TXT,
    DOWNSAMPLE_RATIO,
    GLOBAL_VIEW_SIZE,
    GLOBAL_VISUAL_TOKENS,
    IMAGE_TOKEN,
    IMAGE_TOKEN_ID,
    LOCAL_QUERIES,
    LOCAL_VIEW_SIZE,
    MODEL_VOCAB_SIZE,
    PAD_TOKEN_ID,
    PATCH_SIZE,
    crop_visual_token_count,
    default_tokenizer_path,
    dynamic_preprocess,
    format_messages_plain,
    global_visual_token_count,
    load_prepared_fixture,
    load_tokenizer,
    prepare_image,
    project_root,
    render_prompt,
    prepare_pages,
    prepare_text,
    save_prepared_request,
    validate_tokenizer,
)


def solid_image(width: int, height: int, color: tuple[int, int, int] = (220, 20, 60)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def upstream_model_source() -> str:
    return (project_root() / "data/context/modeling_unlimitedocr.py").read_text(encoding="utf-8")


def upstream_plain_prompt(prompt: str) -> str:
    conversation_path = project_root() / "data/context/conversation.py"
    spec = importlib.util.spec_from_file_location("uocr_upstream_conversation", conversation_path)
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
    for message in [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]:
        conv.append_message(message["role"], message["content"].strip())
    return conv.get_prompt().strip()


def test_tokenizer_metadata_loads() -> None:
    tokenizer = load_tokenizer()
    assert tokenizer.token_to_id(IMAGE_TOKEN) == IMAGE_TOKEN_ID


def test_tokenizer_metadata_contract_matches_context_files() -> None:
    tokenizer_path = default_tokenizer_path()
    tokenizer = load_tokenizer(tokenizer_path)
    validate_tokenizer(tokenizer, tokenizer_path)

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    with tokenizer_path.with_name("config.json").open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    assert tokenizer.get_vocab_size(with_added_tokens=False) == BPE_VOCAB_SIZE
    assert len(tokenizer_json["added_tokens"]) == ADDED_TOKEN_COUNT
    assert int(model_config["vocab_size"]) == MODEL_VOCAB_SIZE
    assert tokenizer.token_to_id("<｜begin▁of▁sentence｜>") == BOS_TOKEN_ID
    assert tokenizer.token_to_id("<｜end▁of▁sentence｜>") == EOS_TOKEN_ID
    assert tokenizer.token_to_id("<｜▁pad▁｜>") == PAD_TOKEN_ID
    assert tokenizer.token_to_id(IMAGE_TOKEN) == IMAGE_TOKEN_ID
    assert 0 <= IMAGE_TOKEN_ID < MODEL_VOCAB_SIZE
    assert tokenizer.get_vocab_size(with_added_tokens=True) <= MODEL_VOCAB_SIZE


def test_plain_prompt_rendering_strips_roles_and_content() -> None:
    prompt = format_messages_plain([
        {"role": "<|User|>", "content": "  <image>document parsing.  "},
        {"role": "<|Assistant|>", "content": ""},
    ])
    assert prompt == "<image>document parsing."


def test_plain_prompt_matches_cached_upstream_template() -> None:
    prompt = "  <image>document parsing.  "
    assert render_prompt(prompt) == upstream_plain_prompt(prompt)


def test_prepare_text_has_bos_and_no_image_mask() -> None:
    req = prepare_text("hello", max_new_tokens=3)
    assert req.input_ids[0] == 0
    assert req.image_mask.sum() == 0
    assert req.max_new_tokens == 3
    assert req.views == ()


def test_text_prompt_construction_prepends_bos_without_eos() -> None:
    prompt = "  alpha beta  "
    tokenizer = load_tokenizer()
    req = prepare_text(prompt, max_new_tokens=0)

    rendered = render_prompt(prompt)
    expected_text_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    np.testing.assert_array_equal(req.input_ids, np.array([BOS_TOKEN_ID, *expected_text_ids], dtype=np.int32))
    np.testing.assert_array_equal(req.image_mask, np.zeros(1 + len(expected_text_ids), dtype=np.uint8))
    assert req.rendered_prompt == "alpha beta"
    assert int(req.input_ids[-1]) != EOS_TOKEN_ID


def test_image_prompt_construction_splices_visual_span_between_text_segments() -> None:
    prompt = "  before <image> after  "
    tokenizer = load_tokenizer()
    req = prepare_image(solid_image(64, 64), prompt=prompt, preset="base", max_new_tokens=0)

    rendered = render_prompt(prompt)
    prefix, suffix = rendered.split(IMAGE_TOKEN)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False).ids
    expected_ids = np.array(
        [BOS_TOKEN_ID, *prefix_ids, *([IMAGE_TOKEN_ID] * GLOBAL_VISUAL_TOKENS), *suffix_ids],
        dtype=np.int32,
    )
    expected_mask = np.array(
        [0] * (1 + len(prefix_ids)) + [1] * GLOBAL_VISUAL_TOKENS + [0] * len(suffix_ids),
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(req.input_ids, expected_ids)
    np.testing.assert_array_equal(req.image_mask, expected_mask)
    assert req.expected_visual_tokens == GLOBAL_VISUAL_TOKENS
    visual_start = 1 + len(prefix_ids)
    visual_stop = visual_start + GLOBAL_VISUAL_TOKENS
    assert np.flatnonzero(req.image_mask).tolist() == list(range(visual_start, visual_stop))
    assert int(req.input_ids[-1]) != EOS_TOKEN_ID


def test_prompt_construction_rejects_placeholder_count_mismatch() -> None:
    with pytest.raises(ValueError, match="prompt has 1 <image> placeholders but 0 visual spans"):
        prepare_text("<image>oops", max_new_tokens=0)
    with pytest.raises(ValueError, match="prompt has 0 <image> placeholders but 1 visual spans"):
        prepare_image(solid_image(64, 64), prompt="document parsing.", preset="base", max_new_tokens=0)


@pytest.mark.parametrize(
    ("image_size", "queries", "expected_tokens"),
    [
        (GLOBAL_VIEW_SIZE, 16, 273),
        (LOCAL_VIEW_SIZE, 10, 111),
    ],
)
def test_global_visual_token_count_formula(image_size: int, queries: int, expected_tokens: int) -> None:
    assert queries == math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
    assert global_visual_token_count(image_size) == (queries + 1) * queries + 1
    assert global_visual_token_count(image_size) == expected_tokens
    assert GLOBAL_VISUAL_TOKENS == global_visual_token_count(GLOBAL_VIEW_SIZE)


@pytest.mark.parametrize(
    ("grid_w", "grid_h"),
    [
        (2, 1),
        (1, 2),
        (3, 2),
        (4, 3),
    ],
)
def test_crop_visual_token_count_formula(grid_w: int, grid_h: int) -> None:
    stitched_local_rows = (LOCAL_QUERIES * grid_w + 1) * (LOCAL_QUERIES * grid_h)
    assert crop_visual_token_count(grid_w, grid_h) == stitched_local_rows + GLOBAL_VISUAL_TOKENS


def test_crop_visual_token_count_single_crop_shortcut_and_invalid_grids() -> None:
    assert crop_visual_token_count(1, 1) == GLOBAL_VISUAL_TOKENS
    assert crop_visual_token_count(1, 1) != (LOCAL_QUERIES + 1) * LOCAL_QUERIES + GLOBAL_VISUAL_TOKENS
    with pytest.raises(ValueError, match="invalid crop grid"):
        crop_visual_token_count(0, 1)
    with pytest.raises(ValueError, match="invalid crop grid"):
        crop_visual_token_count(1, 0)


def test_prepared_requests_use_visual_token_formulas() -> None:
    base = prepare_image(solid_image(320, 240), preset="base", max_new_tokens=0)
    assert base.expected_visual_tokens == global_visual_token_count(GLOBAL_VIEW_SIZE)
    assert int(base.image_mask.sum(dtype=np.uint64)) == base.expected_visual_tokens

    gundam_small = prepare_image(solid_image(320, 240), preset="gundam", max_new_tokens=0)
    assert (gundam_small.crop_grid_w, gundam_small.crop_grid_h) == (1, 1)
    assert gundam_small.expected_visual_tokens == crop_visual_token_count(1, 1)
    assert int(gundam_small.image_mask.sum(dtype=np.uint64)) == gundam_small.expected_visual_tokens

    gundam_wide = prepare_image(solid_image(1280, 640), preset="gundam", max_new_tokens=0)
    assert (gundam_wide.crop_grid_w, gundam_wide.crop_grid_h) == (2, 1)
    assert gundam_wide.expected_visual_tokens == crop_visual_token_count(2, 1)
    assert int(gundam_wide.image_mask.sum(dtype=np.uint64)) == gundam_wide.expected_visual_tokens

    pages = prepare_pages([solid_image(300, 500), solid_image(500, 300), solid_image(256, 256)], max_new_tokens=0)
    assert pages.expected_visual_tokens == 3 * global_visual_token_count(GLOBAL_VIEW_SIZE)
    assert int(pages.image_mask.sum(dtype=np.uint64)) == pages.expected_visual_tokens


def test_prepare_gundam_small_image_uses_global_only() -> None:
    req = prepare_image(solid_image(320, 240), preset="gundam", max_new_tokens=0)
    assert req.input_ids[0] == 0
    assert req.crop_grid_w == 1
    assert req.crop_grid_h == 1
    assert req.expected_visual_tokens == GLOBAL_VISUAL_TOKENS
    assert len(req.views) == 1
    assert req.views[0].kind == "global"
    assert req.views[0].pixels.dtype == np.float16
    assert req.views[0].pixels.shape == (3, 1024, 1024)
    assert np.all(req.input_ids[req.image_mask.astype(bool)] == IMAGE_TOKEN_ID)


def test_prepare_gundam_large_wide_image_uses_local_first_then_global() -> None:
    image = solid_image(1280, 640)
    crops, grid = dynamic_preprocess(image)
    assert grid == (2, 1)
    assert len(crops) == 2

    req = prepare_image(image, preset="gundam", max_new_tokens=0)
    assert (req.crop_grid_w, req.crop_grid_h) == (2, 1)
    local_visual_tokens = (LOCAL_QUERIES * 2 + 1) * (LOCAL_QUERIES * 1)
    assert req.expected_visual_tokens == local_visual_tokens + GLOBAL_VISUAL_TOKENS
    assert [view.kind for view in req.views] == ["local", "local", "global"]
    assert req.views[0].pixels.shape == (3, LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE)
    assert req.views[-1].pixels.shape == (3, 1024, 1024)

    visual_positions = np.flatnonzero(req.image_mask)
    assert visual_positions.size == req.expected_visual_tokens
    assert np.unique(req.input_ids[visual_positions]).tolist() == [IMAGE_TOKEN_ID]
    assert req.input_ids[visual_positions[0]] == IMAGE_TOKEN_ID
    assert req.input_ids[visual_positions[local_visual_tokens]] == IMAGE_TOKEN_ID
    assert req.input_ids[visual_positions[-1]] == IMAGE_TOKEN_ID


def test_cached_upstream_crop_features_are_local_global_separator_then_masked_scatter() -> None:
    source = upstream_model_source()
    assert "global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)" in source
    assert "inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)" in source


def test_prepare_pages_concatenates_global_visual_tokens() -> None:
    req = prepare_pages([solid_image(300, 500), solid_image(500, 300)], max_new_tokens=0)
    assert req.mode == "multi-page-base"
    assert req.expected_visual_tokens == 2 * GLOBAL_VISUAL_TOKENS
    assert len(req.views) == 2
    assert all(view.kind == "global" for view in req.views)
    assert all(view.pixels.shape == (3, 1024, 1024) for view in req.views)


def test_fixture_roundtrip(tmp_path) -> None:
    req = prepare_image(solid_image(320, 240), preset="base", max_new_tokens=0, dtype=np.float32)
    save_prepared_request(req, tmp_path)
    manifest, input_ids, image_mask, views = load_prepared_fixture(tmp_path)
    assert manifest["mode"] == "base"
    assert manifest["image_mask_count"] == GLOBAL_VISUAL_TOKENS
    assert manifest["source_images"][0]["width"] == 320
    assert manifest["source_images"][0]["height"] == 240
    assert "expected_output" not in manifest
    assert not (tmp_path / EXPECTED_OUTPUT_IDS_NPY).exists()
    assert not (tmp_path / EXPECTED_TEXT_TXT).exists()
    np.testing.assert_array_equal(input_ids, req.input_ids)
    np.testing.assert_array_equal(image_mask, req.image_mask)
    assert views["view_0"].dtype == np.float32
    assert views["view_0"].shape == (3, 1024, 1024)


def test_fixture_serializes_optional_expected_output(tmp_path) -> None:
    req = replace(
        prepare_text("hello", max_new_tokens=3),
        expected_output_ids=np.array([42, 43, 1], dtype=np.int64),
        expected_text="answer\n",
    )
    save_prepared_request(req, tmp_path)

    manifest, input_ids, image_mask, views = load_prepared_fixture(tmp_path)
    assert manifest["mode"] == "text-only"
    assert views == {}
    np.testing.assert_array_equal(input_ids, req.input_ids)
    np.testing.assert_array_equal(image_mask, req.image_mask)

    expected = manifest["expected_output"]
    assert expected == {
        "ids_dtype": "int32",
        "ids_file": EXPECTED_OUTPUT_IDS_NPY,
        "ids_shape": [3],
        "text_file": EXPECTED_TEXT_TXT,
    }
    np.testing.assert_array_equal(np.load(tmp_path / EXPECTED_OUTPUT_IDS_NPY), np.array([42, 43, 1], dtype=np.int32))
    assert (tmp_path / EXPECTED_TEXT_TXT).read_text(encoding="utf-8") == "answer\n"


def test_fixture_rejects_non_1d_expected_output_ids(tmp_path) -> None:
    req = replace(prepare_text("hello", max_new_tokens=3), expected_output_ids=np.zeros((1, 2), dtype=np.int32))
    try:
        save_prepared_request(req, tmp_path)
    except ValueError as exc:
        assert "expected_output_ids must be a 1D int32 array" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("save_prepared_request accepted non-1D expected output ids")
    assert not (tmp_path / "manifest.json").exists()
