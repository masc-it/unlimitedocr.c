from __future__ import annotations

from pathlib import Path

root = Path(__file__).resolve().parents[2]
metal = root / "src" / "backend" / "metal" / "uocr_metal.m"
text = metal.read_text()

required = [
    "mps_descriptor_cache",
    "mps_weight_ndarray_cache",
    "mps_workspace_ndarray_cache",
    "metal.mps.ndarray_descriptor",
    "metal.mps.ndarray_create",
    "metal.mps.weight_ndarray_cache.hit",
    "metal.mps.workspace_ndarray_cache.hit",
    "metal_clear_mps_weight_ndarray_cache(ctx);",
    "metal_clear_mps_workspace_ndarray_cache(ctx);",
    "metal.vision.diagnostic_final_host_output",
    "metal.vision.diagnostic_sam_host_output",
    "metal.vision.diagnostic_clip_host_output",
    "metal.vision.diagnostic_projected_host_output",
]
missing = [needle for needle in required if needle not in text]
assert not missing, f"missing expected static contracts: {missing}"

start = text.index("static int metal_run_mps_matmul_nt_f16")
end = text.index("static int metal_hidden_arena_segment_slice", start)
matmul = text[start:end]
for forbidden in [
    "metal.mps.matmul.fill",
    "MPS destination fill encoder",
    "fillBuffer:dst.buffer",
]:
    assert forbidden not in matmul, f"MPS matmul still contains zero-fill path: {forbidden}"

public_start = text.index("int uocr_metal_context_generate_image_f16")
public_end = text.index("static int run_scratch_smoke", public_start)
public = text[public_start:public_end]
for forbidden_event in [
    "metal.vision.diagnostic_final_host_output",
    "metal.vision.diagnostic_sam_host_output",
    "metal.vision.diagnostic_clip_host_output",
    "metal.vision.diagnostic_projected_host_output",
]:
    assert forbidden_event not in public, f"public image generation references diagnostic event {forbidden_event}"

print("static contracts OK")
