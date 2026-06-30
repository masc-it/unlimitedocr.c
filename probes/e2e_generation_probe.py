#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub>=0.23",
#   "numpy>=2",
#   "pillow>=10",
#   "psutil>=5.9",
#   "safetensors>=0.4",
#   "tokenizers>=0.20",
# ]
# ///
"""End-to-end UnlimitedOCR generation timing and memory probe.

Mimics notebooks/demo.ipynb: load the model, prepare docs/test.png, generate,
and decode.  It prefers the Release build artifact and prints wall-clock timings,
process RSS, native memory accounting, and Metal profile events.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from time import perf_counter
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unlimitedocr_c.api import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_NO_REPEAT_WINDOW,
    load_user_image,
    resolve_model_path,
)
from unlimitedocr_c.ffi import Engine, EngineOptions, MemoryReport, ProfileReport  # noqa: E402
from unlimitedocr_c.frontend import PRESETS, SINGLE_PROMPT, default_tokenizer_path, prepare_image  # noqa: E402
from unlimitedocr_c.ocr import decode_generated_ids, default_resource_path  # noqa: E402

MEMORY_CATEGORIES = (
    "model_views",
    "kv_cache",
    "prompt_embeddings",
    "vision_gpu_workspace",
    "vision_final_features",
    "vision_host_staging",
    "decoder_scratch",
    "moe_scratch",
    "logits_readback",
    "transient_buffers",
)


def bytes_human(value: int | float) -> str:
    n = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


class RssSampler:
    def __init__(self, interval: float = 0.05) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval = float(interval)
        self.max_rss = 0
        self.samples: list[tuple[str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)

    def _rss(self) -> int:
        rss = int(self.process.memory_info().rss)
        self.max_rss = max(self.max_rss, rss)
        return rss

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._rss()

    def start(self) -> None:
        self._rss()
        self._thread.start()

    def checkpoint(self, name: str) -> int:
        rss = self._rss()
        self.samples.append((name, rss))
        return rss

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._rss()


def default_release_library() -> Path | None:
    candidate = ROOT / "build" / "release" / "libunlimitedocr.dylib"
    return candidate if candidate.exists() else None


def default_release_resource() -> str | None:
    release = ROOT / "build" / "release"
    if (release / "unlimitedocr.metallib").exists() or (release / "kernels" / "uocr_smoke.metal").exists():
        return str(release)
    return default_resource_path("metal")


def memory_report_dict(memory: MemoryReport) -> dict[str, Any]:
    return {
        "total_live_bytes": memory.total_live_bytes,
        "total_peak_bytes": memory.total_peak_bytes,
        "recommended_working_set_bytes": memory.recommended_working_set_bytes,
        "memory_budget_bytes": memory.memory_budget_bytes,
        "vision_workspace_capacity_bytes": memory.vision_workspace_capacity_bytes,
        "vision_workspace_high_watermark_bytes": memory.vision_workspace_high_watermark_bytes,
        "category_peak_bytes": dict(zip(MEMORY_CATEGORIES, memory.category_peak_bytes, strict=False)),
        "category_live_bytes": dict(zip(MEMORY_CATEGORIES, memory.category_live_bytes, strict=False)),
    }


def profile_report_dict(report: ProfileReport | None, limit: int) -> dict[str, Any] | None:
    if report is None:
        return None
    events = sorted(report.events, key=lambda event: event.total_ms, reverse=True)[:limit]
    return {
        "enabled": report.enabled,
        "generation_index": report.generation_index,
        "dropped_event_count": report.dropped_event_count,
        "metal_buffer_allocation_count": report.metal_buffer_allocation_count,
        "metal_buffer_allocation_bytes": report.metal_buffer_allocation_bytes,
        "metal_command_buffer_count": report.metal_command_buffer_count,
        "metal_command_encoder_count": report.metal_command_encoder_count,
        "metal_command_buffer_wait_count": report.metal_command_buffer_wait_count,
        "metal_mps_descriptor_count": report.metal_mps_descriptor_count,
        "metal_mps_ndarray_count": report.metal_mps_ndarray_count,
        "metal_nsarray_count": report.metal_nsarray_count,
        "metal_transient_retain_object_count": report.metal_transient_retain_object_count,
        "events": [event.as_dict() for event in events],
        "memory": memory_report_dict(report.memory),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "docs" / "test.png")
    parser.add_argument("--profile", choices=sorted(PRESETS), default="base")
    parser.add_argument("--model", type=Path, default=None, help=".uocr model path; defaults to normal UnlimitedOCR resolution")
    parser.add_argument("--backend", default="metal")
    parser.add_argument("--library", type=Path, default=default_release_library(), help="native library path; defaults to build/release")
    parser.add_argument("--resource", type=Path, default=None, help="Metal resource path; defaults to build/release metallib when present")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="override generation length; omit to mimic demo.ipynb")
    parser.add_argument("--memory-budget-bytes", type=int, default=0)
    parser.add_argument("--profile-events", type=int, default=20)
    parser.add_argument("--rss-interval", type=float, default=0.05)
    parser.add_argument("--no-download", action="store_true", help="do not download/convert a missing model")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sampler = RssSampler(interval=args.rss_interval)
    timings: dict[str, float] = {}
    profile: ProfileReport | None = None
    memory: MemoryReport | None = None
    decoded = ""
    generated_count = 0

    sampler.start()
    total_start = perf_counter()
    try:
        sampler.checkpoint("start")

        t = perf_counter()
        model_path = resolve_model_path(args.model, download=not args.no_download)
        timings["resolve_model_s"] = perf_counter() - t
        sampler.checkpoint("after_model_resolve")

        t = perf_counter()
        image = load_user_image(args.image)
        timings["load_image_s"] = perf_counter() - t
        sampler.checkpoint("after_image_load")

        t = perf_counter()
        request = prepare_image(
            image,
            prompt=SINGLE_PROMPT,
            preset=args.profile,
            tokenizer_path=default_tokenizer_path(),
            max_length=DEFAULT_MAX_LENGTH,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
            no_repeat_window=DEFAULT_NO_REPEAT_WINDOW,
            dtype=np.float16,
        )
        timings["prepare_request_s"] = perf_counter() - t
        sampler.checkpoint("after_prepare")

        resource_path = str(args.resource) if args.resource is not None else default_release_resource()
        library_path = str(args.library) if args.library is not None else None

        t = perf_counter()
        engine = Engine(
            EngineOptions(
                model_path=str(model_path) if args.backend.lower() != "cpu-ref" else None,
                backend=args.backend,
                resource_path=resource_path,
                max_batch=1,
                max_prompt_tokens=max(1, request.n_tokens),
                max_gen_tokens=max(1, request.max_new_tokens),
                memory_budget_bytes=args.memory_budget_bytes,
                profile=True,
            ),
            library_path=library_path,
        )
        timings["engine_open_model_load_s"] = perf_counter() - t
        sampler.checkpoint("after_engine_open")

        try:
            engine.profile_reset()
            t = perf_counter()
            outputs = engine.generate_prepared(request)
            timings["generate_native_s"] = perf_counter() - t
            sampler.checkpoint("after_generate")

            if len(outputs) != 1:
                raise RuntimeError(f"expected one generated output, got {len(outputs)}")
            generated_count = int(outputs[0].shape[0])

            t = perf_counter()
            decoded = decode_generated_ids(outputs[0], request.tokenizer_path)
            timings["decode_s"] = perf_counter() - t
            sampler.checkpoint("after_decode")

            profile = engine.profile_report()
            memory = engine.memory_report()
        finally:
            t = perf_counter()
            engine.close()
            timings["engine_close_s"] = perf_counter() - t
            sampler.checkpoint("after_engine_close")

        timings["total_s"] = perf_counter() - total_start
    finally:
        sampler.stop()

    memory_payload = memory_report_dict(memory) if memory is not None else None
    profile_payload = profile_report_dict(profile, args.profile_events)
    payload: dict[str, Any] = {
        "model_path": str(model_path),
        "image": str(args.image),
        "backend": args.backend,
        "profile": args.profile,
        "library_path": str(args.library) if args.library is not None else None,
        "resource_path": str(args.resource) if args.resource is not None else default_release_resource(),
        "prompt_tokens": request.n_tokens,
        "max_new_tokens": request.max_new_tokens,
        "generated_tokens": generated_count,
        "tokens_per_second_native": (generated_count / timings["generate_native_s"]) if timings.get("generate_native_s", 0.0) > 0.0 else None,
        "timings_s": timings,
        "rss_peak_bytes": sampler.max_rss,
        "rss_checkpoints_bytes": dict(sampler.samples),
        "native_memory": memory_payload,
        "native_profile": profile_payload,
        "text": decoded,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("E2E UnlimitedOCR generation probe")
    print(f"model:    {payload['model_path']}")
    print(f"image:    {payload['image']}")
    print(f"library:  {payload['library_path']}")
    print(f"resource: {payload['resource_path']}")
    print(f"request:  profile={args.profile} prompt_tokens={request.n_tokens} max_new_tokens={request.max_new_tokens}")
    print(f"output:   generated_tokens={generated_count} native_tok_s={payload['tokens_per_second_native']:.3f}")
    print("\nTimings:")
    for name, seconds in timings.items():
        print(f"  {name:28s} {seconds * 1000.0:10.3f} ms")
    print("\nProcess RSS:")
    for name, rss in sampler.samples:
        print(f"  {name:28s} {bytes_human(rss):>12s}")
    print(f"  {'sampled_peak':28s} {bytes_human(sampler.max_rss):>12s}")
    if memory_payload is not None:
        print("\nNative memory:")
        print(f"  total_live                 {bytes_human(memory_payload['total_live_bytes'])}")
        print(f"  total_peak                 {bytes_human(memory_payload['total_peak_bytes'])}")
        print(f"  recommended_working_set    {bytes_human(memory_payload['recommended_working_set_bytes'])}")
        print(f"  vision_workspace_capacity  {bytes_human(memory_payload['vision_workspace_capacity_bytes'])}")
        print(f"  vision_workspace_highwater {bytes_human(memory_payload['vision_workspace_high_watermark_bytes'])}")
        print("  category peaks:")
        for name, value in memory_payload["category_peak_bytes"].items():
            if value:
                print(f"    {name:24s} {bytes_human(value)}")
    if profile_payload is not None:
        print("\nTop native profile events:")
        for event in profile_payload["events"]:
            print(
                f"  {event['name']:48s} "
                f"{event['total_ms']:10.3f} ms calls={event['calls']} "
                f"min={event['min_ms']:.3f} max={event['max_ms']:.3f}"
            )
    print("\nText preview:")
    print(decoded[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
