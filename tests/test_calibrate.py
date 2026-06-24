from __future__ import annotations

import json

import numpy as np

from unlimitedocr_c.calibrate import (
    CalibrationFixtureCase,
    generated_ids_metrics,
    load_calibration_corpus,
    logits_topk_metrics,
    main,
    run_calibration,
)


def _f16_bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(np.dtype("<f2")).view(np.dtype("<u2"))


def _write_calibration_fixture(
    root,
    layers: list[np.ndarray],
    *,
    logits_ids: list[int],
    logits_scores: list[float],
    generated_ids: list[int],
    generated_text: str = "",
    router_variant: str = "ref",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    golden: dict[str, dict[str, object]] = {}
    for layer, values in enumerate(layers):
        bits = _f16_bits(values)
        filename = f"layer_{layer}_hidden_f16.bin"
        bits.astype(np.dtype("<u2"), copy=False).tofile(root / filename)
        golden[f"layer_{layer}_hidden"] = {"file": filename, "shape": list(bits.shape)}

    logits_ids_array = np.asarray(logits_ids, dtype=np.dtype("<i4"))
    logits_scores_array = np.asarray(logits_scores, dtype=np.dtype("<f4"))
    generated_array = np.asarray(generated_ids, dtype=np.dtype("<i4"))
    logits_ids_array.tofile(root / "logits_topk_ids_i32.bin")
    logits_scores_array.tofile(root / "logits_topk_scores_f32.bin")
    generated_array.tofile(root / "generated_ids_i32.bin")
    (root / "generated_text.txt").write_text(generated_text, encoding="utf-8")
    golden["logits_topk"] = {
        "ids_file": "logits_topk_ids_i32.bin",
        "scores_file": "logits_topk_scores_f32.bin",
        "shape": [int(logits_ids_array.size)],
    }
    golden["generated_ids"] = {"file": "generated_ids_i32.bin", "shape": [int(generated_array.size)]}
    golden["generated_text"] = {"file": "generated_text.txt"}

    if router_variant == "candidate":
        router_ids = np.array([[1, 2, 3, 4, 5, 7], [6, 5, 4, 3, 2, 1]], dtype=np.dtype("<u4"))
        router_weights = np.array([[0.51, 0.26, 0.13, 0.07, 0.04, 0.02], [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]], dtype=np.dtype("<f4"))
    else:
        router_ids = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=np.dtype("<u4"))
        router_weights = np.array([[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625], [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]], dtype=np.dtype("<f4"))
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
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_metric_helpers_report_logits_and_generated_agreement() -> None:
    logits = logits_topk_metrics(
        np.array([10, 11, 12, 13], dtype=np.int32),
        np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        np.array([10, 12, 11, 14], dtype=np.int32),
        np.array([3.5, 2.75, 2.5, 0.5], dtype=np.float32),
    )
    assert logits.top_k == 4
    assert logits.intersection_count == 3
    assert np.isclose(logits.set_overlap, 0.75)
    assert np.isclose(logits.ordered_id_agreement, 0.25)
    assert logits.top1_match is True
    assert logits.truncated_kl_ref_to_candidate >= 0.0

    generated = generated_ids_metrics(np.array([10, 11, 12], dtype=np.int32), np.array([10, 11, 99], dtype=np.int32))
    assert generated.exact_match is False
    assert generated.longest_common_prefix == 2


def test_run_calibration_compares_layers_router_logits_and_generated(tmp_path) -> None:
    ref_layers = [
        np.zeros((2, 4), dtype=np.float32),
        np.array([[1.0, 2.0, -1.0, 0.5], [0.25, -0.5, 0.75, -1.25]], dtype=np.float32),
    ]
    cand_layers = [np.full((2, 4), 0.125, dtype=np.float32), ref_layers[1] + np.float32(0.25)]
    ref = tmp_path / "fp16" / "dense-text"
    cand = tmp_path / "dyn-q4" / "dense-text"
    _write_calibration_fixture(
        ref,
        ref_layers,
        logits_ids=[10, 11, 12, 13],
        logits_scores=[4.0, 3.0, 2.0, 1.0],
        generated_ids=[10, 11, 12],
        generated_text="reference",
    )
    _write_calibration_fixture(
        cand,
        cand_layers,
        logits_ids=[10, 12, 11, 14],
        logits_scores=[3.5, 2.75, 2.5, 0.5],
        generated_ids=[10, 11, 99],
        generated_text="candidate",
        router_variant="candidate",
    )

    report = run_calibration(
        [CalibrationFixtureCase(case_id="dense-text", fp16_dir=str(ref), candidates={"dyn-q4": str(cand)})],
        corpus_id="ocr-smoke",
    )

    assert report.corpus_id == "ocr-smoke"
    case = report.cases[0]
    assert case.case_id == "dense-text"
    candidate = case.candidates[0]
    assert candidate.label == "dyn-q4"
    assert np.isclose(candidate.layer_drift.layer_metrics[0].rmse, 0.125)
    assert candidate.layer_drift.router_metrics[0].mean_set_overlap < 1.0
    assert candidate.expert_traffic[0].total_selections == 12
    assert candidate.expert_traffic[0].total_variation_distance > 0.0
    assert candidate.logits_topk is not None
    assert np.isclose(candidate.logits_topk.set_overlap, 0.75)
    assert candidate.generated_ids is not None
    assert candidate.generated_ids.longest_common_prefix == 2
    assert candidate.generated_ids.exact_match is False


def test_load_corpus_manifest_resolves_relative_paths(tmp_path) -> None:
    (tmp_path / "fp16" / "case").mkdir(parents=True)
    (tmp_path / "dyn-q8" / "case").mkdir(parents=True)
    manifest = {
        "version": 1,
        "corpus_id": "relative-corpus",
        "fixtures": [
            {
                "id": "case",
                "fp16_dir": "fp16/case",
                "candidates": {"dyn-q8": "dyn-q8/case"},
            }
        ],
    }
    manifest_path = tmp_path / "corpus.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    corpus_id, cases = load_calibration_corpus(manifest_path)

    assert corpus_id == "relative-corpus"
    assert len(cases) == 1
    assert cases[0].fp16_dir == str(tmp_path / "fp16" / "case")
    assert cases[0].candidates["dyn-q8"] == str(tmp_path / "dyn-q8" / "case")


def test_calibration_cli_writes_json_and_enforces_thresholds(tmp_path, capsys) -> None:
    ref = tmp_path / "fp16"
    cand = tmp_path / "dyn-q8"
    _write_calibration_fixture(
        ref,
        [np.zeros((1, 4), dtype=np.float32)],
        logits_ids=[1, 2],
        logits_scores=[2.0, 1.0],
        generated_ids=[1, 2],
    )
    _write_calibration_fixture(
        cand,
        [np.full((1, 4), 0.5, dtype=np.float32)],
        logits_ids=[1, 3],
        logits_scores=[1.5, 1.0],
        generated_ids=[1, 4],
    )
    out = tmp_path / "report.json"

    ok = main(["--fp16-dir", str(ref), "--candidate", f"dyn-q8={cand}", "--json-out", str(out), "--no-router"])
    assert ok == 0
    captured = capsys.readouterr()
    assert "calibration corpus: ad-hoc" in captured.out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cases"][0]["candidates"][0]["logits_topk"]["intersection_count"] == 1

    failed = main([
        "--fp16-dir",
        str(ref),
        "--candidate",
        f"dyn-q8={cand}",
        "--max-layer-rmse",
        "0.1",
        "--require-generated-match",
        "--no-router",
    ])
    assert failed == 2
    captured = capsys.readouterr()
    assert "threshold failure" in captured.err
