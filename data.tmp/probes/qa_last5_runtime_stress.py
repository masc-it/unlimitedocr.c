from __future__ import annotations

from PIL import Image

from unlimitedocr_c.ffi import Engine, EngineOptions
from unlimitedocr_c.frontend import prepare_image, project_root

root = project_root()
lib = root / "build" / "debug" / "libunlimitedocr.dylib"
model = root / "dist" / "unlimitedocr-fp16.uocr"
resource = root / "src" / "backend" / "metal"
image = Image.open(root / "docs" / "test.png").convert("RGB")
base = prepare_image(image, preset="base", max_new_tokens=1)
gundam = prepare_image(image, preset="gundam", max_new_tokens=1)
max_prompt = max(base.n_tokens, gundam.n_tokens)

def events_by_name(report):
    return {event.name: event for event in report.events}

def require_absent(events, names):
    present = {name: events[name] for name in names if name in events}
    assert not present, f"unexpected profile events: {present}"

def check_public_profile(report, *, label: str, expected_token: int | None = None) -> int:
    events = events_by_name(report)
    require_absent(
        events,
        [
            "metal.vision.diagnostic_final_host_output",
            "metal.vision.diagnostic_sam_host_output",
            "metal.vision.diagnostic_clip_host_output",
            "metal.vision.diagnostic_projected_host_output",
            "metal.mps.matmul.fill",
            "metal.mps.ndarray_descriptor",
            "metal.mps.ndarray_create",
        ],
    )
    assert report.dropped_event_count == 0, (label, report.dropped_event_count)
    assert report.metal_mps_descriptor_count == 0, (label, report.metal_mps_descriptor_count)
    assert report.metal_mps_ndarray_count == 0, (label, report.metal_mps_ndarray_count)
    assert "metal.mps.matmul" in events, (label, sorted(events))
    mps_calls = events["metal.mps.matmul"].calls
    assert mps_calls > 0, (label, mps_calls)
    assert events["metal.mps.weight_ndarray_cache.hit"].calls == mps_calls, (
        label,
        events["metal.mps.weight_ndarray_cache.hit"].calls,
        mps_calls,
    )
    assert events["metal.mps.workspace_ndarray_cache.hit"].calls == mps_calls * 2, (
        label,
        events["metal.mps.workspace_ndarray_cache.hit"].calls,
        mps_calls,
    )
    assert events["metal.vision_workspace.direct_output"].calls > 0, (label, events.get("metal.vision_workspace.direct_output"))
    assert events["metal.vision_binding_cache.direct_use"].calls > 0, (label, events.get("metal.vision_binding_cache.direct_use"))
    assert report.memory.estimated_vision_host_staging_bytes == 0, (
        label,
        report.memory.estimated_vision_host_staging_bytes,
    )
    assert report.memory.vision_workspace_capacity_bytes >= report.memory.vision_workspace_high_watermark_bytes > 0, (
        label,
        report.memory.vision_workspace_capacity_bytes,
        report.memory.vision_workspace_high_watermark_bytes,
    )
    if expected_token is not None:
        assert expected_token == 128818, (label, expected_token)
    print(
        label,
        "mps_calls", mps_calls,
        "encoders", report.metal_command_encoder_count,
        "mps_ms", round(events["metal.mps.matmul"].total_ms, 3),
        "setup_ms", round(events["metal.mps.matmul.setup"].total_ms, 3),
        "workspace_high", report.memory.vision_workspace_high_watermark_bytes,
    )
    return mps_calls

with Engine(
    EngineOptions(
        model_path=str(model),
        backend="metal",
        resource_path=str(resource),
        max_batch=1,
        max_prompt_tokens=max_prompt,
        max_gen_tokens=1,
        memory_budget_bytes=(1 << 64) - 1,
        profile=True,
    ),
    library_path=str(lib),
) as engine:
    # Warm model/shape-specific MPS caches.  Public OCR token for base is a stable sentinel.
    warm = engine.generate_prepared(base)
    assert int(warm[0][0]) == 128818, warm[0]

    base_tokens = []
    for i in range(4):
        engine.profile_reset()
        out = engine.generate_prepared(base)
        token = int(out[0][0])
        base_tokens.append(token)
        report = engine.profile_report()
        check_public_profile(report, label=f"base-repeat-{i}", expected_token=token)
    assert base_tokens == [128818, 128818, 128818, 128818], base_tokens

    # Exercise a second prepared-request shape/preset.  It may create new cache entries on the first run;
    # the second run must reuse them and remain deterministic.
    gundam_first = engine.generate_prepared(gundam)
    gundam_token = int(gundam_first[0][0])
    engine.profile_reset()
    gundam_second = engine.generate_prepared(gundam)
    assert int(gundam_second[0][0]) == gundam_token, (gundam_token, gundam_second[0])
    gundam_report = engine.profile_report()
    check_public_profile(gundam_report, label="gundam-repeat")

print("runtime stress OK", "base_tokens", base_tokens, "gundam_token", gundam_token)
