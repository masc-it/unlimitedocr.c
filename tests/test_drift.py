from __future__ import annotations

import json

import numpy as np

from unlimitedocr_c.drift import compare_layer_drift, main, tensor_drift_metrics


def _f16_bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(np.dtype("<f2")).view(np.dtype("<u2"))


def _write_layer_fixture(root, layers: list[np.ndarray], *, router_shift: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "input_ids_i32.bin").write_bytes(np.array([0, 1], dtype=np.dtype("<i4")).tobytes())
    (root / "image_mask_u8.bin").write_bytes(np.zeros(2, dtype=np.uint8).tobytes())
    golden: dict[str, dict[str, object]] = {}
    for layer, values in enumerate(layers):
        bits = _f16_bits(values)
        filename = f"layer_{layer}_hidden_f16.bin"
        bits.astype(np.dtype("<u2"), copy=False).tofile(root / filename)
        golden[f"layer_{layer}_hidden"] = {"file": filename, "shape": list(bits.shape)}

    router_ids = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=np.dtype("<u4"))
    if router_shift:
        router_ids = np.array([[1, 2, 3, 4, 5, 7], [6, 5, 4, 3, 2, 1]], dtype=np.dtype("<u4"))
    router_weights = np.array(
        [[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625], [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]],
        dtype=np.dtype("<f4"),
    )
    if router_shift:
        router_weights = router_weights + np.float32(0.01)
    router_ids.tofile(root / "layer_1_router_top_ids_u32.bin")
    router_weights.tofile(root / "layer_1_router_top_weights_f32.bin")

    manifest = {
        "text_decoder_layer_count": len(layers),
        "golden_tensors": golden,
        "router_topk": {
            "layer_1": {
                "ids_file": "layer_1_router_top_ids_u32.bin",
                "weights_file": "layer_1_router_top_weights_f32.bin",
                "shape": [2, 6],
            }
        },
        "native_binary_arrays": {
            "input_ids": {"file": "input_ids_i32.bin", "dtype": "int32_le", "shape": [2]},
            "image_mask": {"file": "image_mask_u8.bin", "dtype": "uint8", "shape": [2]},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_tensor_drift_metrics_are_stable() -> None:
    ref = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
    cand = np.array([1.25, -1.5, 0.0, 0.0], dtype=np.float32)

    metrics = tensor_drift_metrics("x", ref, cand)

    assert metrics.shape == (4,)
    assert metrics.count == 4
    assert np.isclose(metrics.rmse, np.sqrt((0.25**2 + 0.5**2 + 0.5**2) / 4.0))
    assert np.isclose(metrics.mean_abs, (0.25 + 0.5 + 0.5) / 4.0)
    assert metrics.max_abs == 0.5
    assert 0.9 < metrics.cosine_similarity < 1.0


def test_compare_layer_drift_reports_layer_and_router_metrics(tmp_path) -> None:
    ref_layers = [
        np.zeros((2, 4), dtype=np.float32),
        np.array([[1.0, 2.0, -1.0, 0.5], [0.25, -0.5, 0.75, -1.25]], dtype=np.float32),
    ]
    cand_layers = [
        np.full((2, 4), 0.125, dtype=np.float32),
        ref_layers[1] + np.float32(0.25),
    ]
    ref = tmp_path / "fp16"
    cand = tmp_path / "dyn-q4"
    _write_layer_fixture(ref, ref_layers)
    _write_layer_fixture(cand, cand_layers, router_shift=True)

    report = compare_layer_drift(ref, {"dyn-q4": cand})

    assert report.reference_dir == str(ref)
    candidate = report.candidates[0]
    assert candidate.label == "dyn-q4"
    assert len(candidate.layer_metrics) == 2
    assert candidate.layer_metrics[0].name == "layer_0_hidden"
    assert np.isclose(candidate.layer_metrics[0].rmse, 0.125)
    assert np.isclose(candidate.layer_metrics[1].mean_abs, 0.25)
    assert candidate.max_rmse >= 0.125
    assert len(candidate.router_metrics) == 1
    router = candidate.router_metrics[0]
    assert router.layer == 1
    assert np.isclose(router.ordered_id_agreement, 11 / 12)
    assert np.isclose(router.row_exact_agreement, 0.5)
    assert router.mean_set_overlap < 1.0
    assert router.weight_rmse > 0.0


def test_drift_cli_writes_json_and_enforces_thresholds(tmp_path, capsys) -> None:
    ref = tmp_path / "fp16"
    cand = tmp_path / "dyn-q8"
    _write_layer_fixture(ref, [np.zeros((1, 4), dtype=np.float32)])
    _write_layer_fixture(cand, [np.full((1, 4), 0.5, dtype=np.float32)])
    out = tmp_path / "report.json"

    ok = main(["--fp16-dir", str(ref), "--candidate", f"dyn-q8={cand}", "--json-out", str(out), "--no-router"])
    assert ok == 0
    captured = capsys.readouterr()
    assert "dyn-q8: layers=1" in captured.out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["candidates"][0]["summary"]["max_abs"] == 0.5

    failed = main(["--fp16-dir", str(ref), "--candidate", f"dyn-q8={cand}", "--max-rmse", "0.1", "--no-router"])
    assert failed == 2
    captured = capsys.readouterr()
    assert "threshold failure" in captured.err
