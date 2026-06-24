from __future__ import annotations

import importlib.util
import sys

import numpy as np
from PIL import Image

from unlimitedocr_c.frontend import (
    GLOBAL_VISUAL_TOKENS,
    IMAGE_TOKEN_ID,
    LOCAL_QUERIES,
    LOCAL_VIEW_SIZE,
    dynamic_preprocess,
    format_messages_plain,
    load_prepared_fixture,
    load_tokenizer,
    prepare_image,
    project_root,
    render_prompt,
    prepare_pages,
    prepare_text,
    save_prepared_request,
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
    assert tokenizer.token_to_id("<image>") == IMAGE_TOKEN_ID


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
    np.testing.assert_array_equal(input_ids, req.input_ids)
    np.testing.assert_array_equal(image_mask, req.image_mask)
    assert views["view_0"].dtype == np.float32
    assert views["view_0"].shape == (3, 1024, 1024)
