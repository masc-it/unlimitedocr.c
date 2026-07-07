from __future__ import annotations

import json
from pathlib import Path
import os

import numpy as np
import pytest

from unlimitedocr_c.golden import (
    GENERATED_IDS_BIN,
    HIDDEN_SIZE,
    IMAGE_MASK_BIN,
    INPUT_IDS_BIN,
    PROMPT_EMBEDDINGS_BIN,
    VISUAL_FEATURES_BIN,
)
from unlimitedocr_c.parity import (
    DumpedImageEmbeddingGenerationResult,
    default_metal_resource_path,
    generate_from_dumped_image_embeddings,
    load_dumped_image_embedding_expected_ids,
)


def _write_minimal_image_embedding_dump(root: Path, *, hidden_size: int, generated: np.ndarray | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    input_ids = np.array([0, 128815, 128815, 42], dtype=np.dtype("<i4"))
    image_mask = np.array([0, 1, 1, 0], dtype=np.uint8)
    visual = np.full((2, hidden_size), 0x3C00, dtype=np.dtype("<u2"))
    prompt = np.zeros((4, hidden_size), dtype=np.dtype("<u2"))
    prompt[1:3] = visual
    generated_ids = np.asarray(generated if generated is not None else np.array([17, 1], dtype=np.int32), dtype=np.dtype("<i4"))

    input_ids.tofile(root / INPUT_IDS_BIN)
    image_mask.tofile(root / IMAGE_MASK_BIN)
    visual.tofile(root / VISUAL_FEATURES_BIN)
    prompt.tofile(root / PROMPT_EMBEDDINGS_BIN)
    generated_ids.tofile(root / GENERATED_IDS_BIN)
    manifest = {
        "mode": "image-embedding-parity",
        "image_embedding_fixture": {
            "bypasses_c_vision_encoder": True,
            "image_span_start": 1,
            "image_span_length": 2,
            "visual_features_file": VISUAL_FEATURES_BIN,
        },
        "golden_tensors": {
            "prompt_embeddings": {"file": PROMPT_EMBEDDINGS_BIN, "shape": [4, hidden_size]},
            "visual_features": {"file": VISUAL_FEATURES_BIN, "shape": [2, hidden_size]},
            "generated_ids": {"file": GENERATED_IDS_BIN, "shape": [int(generated_ids.size)]},
        },
        "native_binary_arrays": {
            "input_ids": {"file": INPUT_IDS_BIN, "dtype": "int32_le", "shape": [4]},
            "image_mask": {"file": IMAGE_MASK_BIN, "dtype": "uint8", "shape": [4]},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_load_dumped_image_embedding_expected_ids_without_layer_dumps(tmp_path: Path) -> None:
    expected = np.array([128818, 10253], dtype=np.int32)
    _write_minimal_image_embedding_dump(tmp_path, hidden_size=HIDDEN_SIZE, generated=expected)

    actual = load_dumped_image_embedding_expected_ids(tmp_path)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype(np.int32)
    assert actual.flags.c_contiguous


def test_generate_from_dumped_image_embeddings_rejects_non_runtime_hidden_width(tmp_path: Path) -> None:
    _write_minimal_image_embedding_dump(tmp_path, hidden_size=4)

    with pytest.raises(ValueError, match=r"\[image_tokens,1280\]"):
        generate_from_dumped_image_embeddings(tmp_path, model_path="missing.uocr", max_new_tokens=1)


def test_dumped_image_embedding_generation_result_is_public_dataclass() -> None:
    result = DumpedImageEmbeddingGenerationResult(
        generated_ids=np.array([1], dtype=np.int32),
        generated_scores=np.array([2.0], dtype=np.float32),
        stopped_on_eos=True,
        last_token_id=1,
        last_score=2.0,
    )
    assert result.generated_ids.tolist() == [1]
    assert result.stopped_on_eos is True


@pytest.mark.skipif(
    os.environ.get("UOCR_RUN_LARGE_TESTS") != "1"
    or not os.environ.get("UOCR_MODEL_PATH")
    or not os.environ.get("UOCR_IMAGE_EMBED_DUMP_DIR"),
    reason="set UOCR_RUN_LARGE_TESTS=1, UOCR_MODEL_PATH, and UOCR_IMAGE_EMBED_DUMP_DIR to run dumped-image parity",
)
def test_generate_from_dumped_image_embeddings_matches_expected_ids_opt_in() -> None:
    fixture_dir = Path(os.environ["UOCR_IMAGE_EMBED_DUMP_DIR"])
    model_path = Path(os.environ["UOCR_MODEL_PATH"])
    resource_path = Path(os.environ.get("UOCR_METAL_RESOURCE_PATH", str(default_metal_resource_path())))
    if not fixture_dir.exists():
        pytest.fail(f"UOCR_IMAGE_EMBED_DUMP_DIR does not exist: {fixture_dir}")
    if not model_path.exists():
        pytest.fail(f"UOCR_MODEL_PATH does not exist: {model_path}")

    expected = load_dumped_image_embedding_expected_ids(fixture_dir)
    actual = generate_from_dumped_image_embeddings(
        fixture_dir,
        model_path=model_path,
        resource_path=resource_path,
        max_new_tokens=int(expected.size),
    )

    np.testing.assert_array_equal(actual.generated_ids, expected)
