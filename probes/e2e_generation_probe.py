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
from unlimitedocr_c.frontend import (  # noqa: E402
    GLOBAL_QUERIES,
    GLOBAL_VIEW_SIZE,
    GLOBAL_VISUAL_TOKENS,
    LOCAL_QUERIES,
    LOCAL_VIEW_SIZE,
    PRESETS,
    SINGLE_PROMPT,
    PreparedRequest,
    default_tokenizer_path,
    prepare_image,
)
from unlimitedocr_c.ocr import decode_generated_ids, default_resource_path  # noqa: E402

DEFAULT_LOCAL_VIEWS_PER_CHUNK = 8

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
        self._thread = threading.Thread(
            target=self._run, name="rss-sampler", daemon=True
        )

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
    if (release / "unlimitedocr.metallib").exists() or (
        release / "kernels" / "uocr_smoke.metal"
    ).exists():
        return str(release)
    return default_resource_path("metal")


def memory_report_dict(memory: MemoryReport) -> dict[str, Any]:
    return {
        "total_live_bytes": memory.total_live_bytes,
        "total_peak_bytes": memory.total_peak_bytes,
        "recommended_working_set_bytes": memory.recommended_working_set_bytes,
        "memory_budget_bytes": memory.memory_budget_bytes,
        "estimated_model_views_bytes": memory.estimated_model_views_bytes,
        "estimated_kv_cache_bytes": memory.estimated_kv_cache_bytes,
        "estimated_prompt_embeddings_bytes": memory.estimated_prompt_embeddings_bytes,
        "estimated_vision_scratch_bytes": memory.estimated_vision_scratch_bytes,
        "estimated_vision_gpu_workspace_bytes": memory.estimated_vision_gpu_workspace_bytes,
        "estimated_vision_final_features_bytes": memory.estimated_vision_final_features_bytes,
        "estimated_vision_host_staging_bytes": memory.estimated_vision_host_staging_bytes,
        "estimated_decoder_scratch_bytes": memory.estimated_decoder_scratch_bytes,
        "estimated_moe_scratch_bytes": memory.estimated_moe_scratch_bytes,
        "estimated_logits_readback_bytes": memory.estimated_logits_readback_bytes,
        "estimated_transient_bytes": memory.estimated_transient_bytes,
        "estimated_safety_margin_bytes": memory.estimated_safety_margin_bytes,
        "estimated_total_bytes": memory.estimated_total_bytes,
        "vision_workspace_capacity_bytes": memory.vision_workspace_capacity_bytes,
        "vision_workspace_high_watermark_bytes": memory.vision_workspace_high_watermark_bytes,
        "category_peak_bytes": dict(
            zip(MEMORY_CATEGORIES, memory.category_peak_bytes, strict=False)
        ),
        "category_live_bytes": dict(
            zip(MEMORY_CATEGORIES, memory.category_live_bytes, strict=False)
        ),
    }


def profile_report_dict(
    report: ProfileReport | None, limit: int
) -> dict[str, Any] | None:
    if report is None:
        return None
    events = sorted(report.events, key=lambda event: event.total_ms, reverse=True)[
        :limit
    ]
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


def local_max_views_per_chunk() -> int:
    raw = os.environ.get("UOCR_VISION_LOCAL_MAX_VIEWS_PER_CHUNK")
    if raw is None or raw == "":
        return DEFAULT_LOCAL_VIEWS_PER_CHUNK
    try:
        value = int(raw, 10)
    except ValueError:
        return DEFAULT_LOCAL_VIEWS_PER_CHUNK
    return value if value > 0 else DEFAULT_LOCAL_VIEWS_PER_CHUNK


def _view_max_size(views: list[Any]) -> int:
    return max((max(int(view.width), int(view.height)) for view in views), default=0)


def _append_vision_chunk(
    chunks: list[dict[str, Any]],
    *,
    kind: str,
    first_view: int,
    view_count: int,
    view_max_size: int,
    projected_grid: int,
    projected_token_start: int,
    final_token_start: int,
    final_token_count: int,
) -> int:
    projected_tokens_per_view = projected_grid * projected_grid
    projected_token_count = view_count * projected_tokens_per_view
    chunks.append(
        {
            "kind": kind,
            "first_view": first_view,
            "view_count": view_count,
            "view_max_size": view_max_size,
            "projected_grid_w": projected_grid,
            "projected_grid_h": projected_grid,
            "projected_tokens_per_view": projected_tokens_per_view,
            "projected_token_start": projected_token_start,
            "projected_token_count": projected_token_count,
            "final_token_start": final_token_start,
            "final_token_count": final_token_count,
        }
    )
    return projected_token_count


def vision_schedule_dict(request: PreparedRequest) -> dict[str, Any]:
    """Mirror the native vision schedule in probe output.

    The native path uses a memory-aware local-crop chunk cap followed by one
    global chunk. Reporting this request-derived plan makes the memory shape
    explicit without adding native diagnostic readbacks.
    """

    local_views = [view for view in request.views if view.kind == "local"]
    global_views = [view for view in request.views if view.kind == "global"]
    local_count = len(local_views)
    global_count = len(global_views)
    max_views_per_chunk = (
        min(local_count, local_max_views_per_chunk())
        if local_count
        else (global_count or 1)
    )
    request_max_view_size = _view_max_size(list(request.views))

    chunks: list[dict[str, Any]] = []
    final_visual_rows = 0
    projected_rows_total = 0
    projected_start = 0

    if local_count == 0:
        final_visual_rows = global_count * GLOBAL_VISUAL_TOKENS
        projected_rows_total = global_count * GLOBAL_QUERIES * GLOBAL_QUERIES
        for first in range(0, global_count, max_views_per_chunk):
            count = min(max_views_per_chunk, global_count - first)
            projected_start += _append_vision_chunk(
                chunks,
                kind="global",
                first_view=first,
                view_count=count,
                view_max_size=_view_max_size(global_views[first : first + count]),
                projected_grid=GLOBAL_QUERIES,
                projected_token_start=projected_start,
                final_token_start=first * GLOBAL_VISUAL_TOKENS,
                final_token_count=count * GLOBAL_VISUAL_TOKENS,
            )
    else:
        local_visual_rows = (LOCAL_QUERIES * request.crop_grid_w + 1) * (
            LOCAL_QUERIES * request.crop_grid_h
        )
        final_visual_rows = local_visual_rows + GLOBAL_VISUAL_TOKENS
        projected_rows_total = (
            local_count * LOCAL_QUERIES * LOCAL_QUERIES
            + GLOBAL_QUERIES * GLOBAL_QUERIES
        )
        for first in range(0, local_count, max_views_per_chunk):
            count = min(max_views_per_chunk, local_count - first)
            projected_start += _append_vision_chunk(
                chunks,
                kind="local",
                first_view=first,
                view_count=count,
                view_max_size=_view_max_size(local_views[first : first + count]),
                projected_grid=LOCAL_QUERIES,
                projected_token_start=projected_start,
                final_token_start=0,
                final_token_count=local_visual_rows,
            )
        if global_count != 0:
            projected_start += _append_vision_chunk(
                chunks,
                kind="global",
                first_view=local_count,
                view_count=1,
                view_max_size=_view_max_size(global_views[:1]),
                projected_grid=GLOBAL_QUERIES,
                projected_token_start=projected_start,
                final_token_start=local_visual_rows,
                final_token_count=GLOBAL_VISUAL_TOKENS,
            )

    max_chunk_views = max((chunk["view_count"] for chunk in chunks), default=0)
    max_chunk_projected_rows = max(
        (chunk["projected_token_count"] for chunk in chunks), default=0
    )
    chunk_shapes: dict[str, dict[str, Any]] = {}
    for kind, grid, default_size in (
        ("local", LOCAL_QUERIES, LOCAL_VIEW_SIZE),
        ("global", GLOBAL_QUERIES, GLOBAL_VIEW_SIZE),
    ):
        kind_chunks = [chunk for chunk in chunks if chunk["kind"] == kind]
        if not kind_chunks:
            continue
        chunk_shapes[kind] = {
            "chunk_count": len(kind_chunks),
            "view_count": local_count if kind == "local" else global_count,
            "max_chunk_views": max(chunk["view_count"] for chunk in kind_chunks),
            "max_chunk_projected_rows": max(
                chunk["projected_token_count"] for chunk in kind_chunks
            ),
            "max_view_size": max(
                max(chunk["view_max_size"], default_size) for chunk in kind_chunks
            ),
            "projected_grid_w": grid,
            "projected_grid_h": grid,
            "projected_tokens_per_view": grid * grid,
        }

    workspace_shapes = {
        kind: {
            "max_view_size": shape["max_view_size"],
            "max_chunk_views": shape["max_chunk_views"],
            "max_chunk_projected_rows": shape["max_chunk_projected_rows"],
            "final_visual_rows": 0,
        }
        for kind, shape in chunk_shapes.items()
    }

    return {
        "local_view_count": local_count,
        "global_view_count": global_count,
        "chunk_count": len(chunks),
        "max_views_per_chunk": max_views_per_chunk,
        "max_chunk_views": max_chunk_views,
        "max_chunk_projected_rows": max_chunk_projected_rows,
        "final_visual_rows": final_visual_rows,
        "projected_rows_total": projected_rows_total,
        "request_max_view_size": request_max_view_size,
        "workspace_plan": {
            "request_max_view_size": request_max_view_size,
            "scratch_shapes": workspace_shapes,
            "final_visual_rows": final_visual_rows,
        },
        "chunk_shapes": chunk_shapes,
        "chunks": chunks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "docs" / "test.png")
    parser.add_argument("--profile", choices=sorted(PRESETS), default="base")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=".uocr model path; defaults to normal UnlimitedOCR resolution",
    )
    parser.add_argument("--backend", default="metal")
    parser.add_argument(
        "--library",
        type=Path,
        default=default_release_library(),
        help="native library path; defaults to build/release",
    )
    parser.add_argument(
        "--resource",
        type=Path,
        default=None,
        help="Metal resource path; defaults to build/release metallib when present",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override generation length; omit to mimic demo.ipynb",
    )
    parser.add_argument("--memory-budget-bytes", type=int, default=0)
    parser.add_argument("--profile-events", type=int, default=20)
    parser.add_argument("--rss-interval", type=float, default=0.05)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="do not download/convert a missing model",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON only"
    )
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

        resource_path = (
            str(args.resource)
            if args.resource is not None
            else default_release_resource()
        )
        library_path = str(args.library) if args.library is not None else None

        t = perf_counter()
        engine = Engine(
            EngineOptions(
                model_path=str(model_path)
                if args.backend.lower() != "cpu-ref"
                else None,
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
        "resource_path": str(args.resource)
        if args.resource is not None
        else default_release_resource(),
        "prompt_tokens": request.n_tokens,
        "max_new_tokens": request.max_new_tokens,
        "vision_schedule": vision_schedule_dict(request),
        "generated_tokens": generated_count,
        "tokens_per_second_native": (generated_count / timings["generate_native_s"])
        if timings.get("generate_native_s", 0.0) > 0.0
        else None,
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
    print(
        f"request:  profile={args.profile} prompt_tokens={request.n_tokens} max_new_tokens={request.max_new_tokens}"
    )
    schedule = payload["vision_schedule"]
    print(
        "vision:   "
        f"local_views={schedule['local_view_count']} global_views={schedule['global_view_count']} "
        f"chunks={schedule['chunk_count']} max_chunk_views={schedule['max_chunk_views']} "
        f"max_chunk_projected_rows={schedule['max_chunk_projected_rows']} "
        f"final_visual_rows={schedule['final_visual_rows']} "
        f"request_max_view_size={schedule['request_max_view_size']}"
    )
    for kind, shape in schedule["chunk_shapes"].items():
        print(
            f"  {kind:6s}: chunks={shape['chunk_count']} views={shape['view_count']} "
            f"max_chunk_views={shape['max_chunk_views']} "
            f"max_chunk_projected_rows={shape['max_chunk_projected_rows']} "
            f"max_view_size={shape['max_view_size']} grid={shape['projected_grid_w']}x{shape['projected_grid_h']}"
        )
    print(
        f"output:   generated_tokens={generated_count} native_tok_s={payload['tokens_per_second_native']:.3f}"
    )
    print("\nTimings:")
    for name, seconds in timings.items():
        print(f"  {name:28s} {seconds * 1000.0:10.3f} ms")
    print("\nProcess RSS:")
    for name, rss in sampler.samples:
        print(f"  {name:28s} {bytes_human(rss):>12s}")
    print(f"  {'sampled_peak':28s} {bytes_human(sampler.max_rss):>12s}")
    if memory_payload is not None:
        print("\nNative memory:")
        print(
            f"  total_live                 {bytes_human(memory_payload['total_live_bytes'])}"
        )
        print(
            f"  total_peak                 {bytes_human(memory_payload['total_peak_bytes'])}"
        )
        print(
            f"  recommended_working_set    {bytes_human(memory_payload['recommended_working_set_bytes'])}"
        )
        print(
            f"  vision_workspace_capacity  {bytes_human(memory_payload['vision_workspace_capacity_bytes'])}"
        )
        print(
            f"  vision_workspace_highwater {bytes_human(memory_payload['vision_workspace_high_watermark_bytes'])}"
        )
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
    print(decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
