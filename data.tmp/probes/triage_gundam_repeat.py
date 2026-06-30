from __future__ import annotations
from PIL import Image
from unlimitedocr_c.ffi import Engine, EngineOptions
from unlimitedocr_c.frontend import prepare_image, project_root

root = project_root()
lib = root / "build" / "debug" / "libunlimitedocr.dylib"
model = root / "dist" / "unlimitedocr-fp16.uocr"
resource = root / "src" / "backend" / "metal"
image = Image.open(root / "docs" / "test.png").convert("RGB")
req = prepare_image(image, preset="gundam", max_new_tokens=1)
print("request", "tokens", req.n_tokens, "views", len(req.views), "grid", req.crop_grid_w, req.crop_grid_h)
with Engine(EngineOptions(model_path=str(model), backend="metal", resource_path=str(resource), max_batch=1, max_prompt_tokens=req.n_tokens, max_gen_tokens=1, memory_budget_bytes=(1 << 64) - 1, profile=True), library_path=str(lib)) as engine:
    toks = []
    for i in range(6):
        engine.profile_reset()
        out = engine.generate_prepared(req)
        tok = int(out[0][0])
        toks.append(tok)
        report = engine.profile_report()
        events = {e.name: e for e in report.events}
        print(i, tok, "mps", events.get("metal.mps.matmul").calls if "metal.mps.matmul" in events else None, "fill", "metal.mps.matmul.fill" in events, "desc", report.metal_mps_descriptor_count, "ndarr", report.metal_mps_ndarray_count, "ws_hits", events.get("metal.mps.workspace_ndarray_cache.hit").calls if "metal.mps.workspace_ndarray_cache.hit" in events else None, "ws_create", events.get("metal.mps.workspace_ndarray_cache.create").calls if "metal.mps.workspace_ndarray_cache.create" in events else None, "high", report.memory.vision_workspace_high_watermark_bytes)
    print("tokens", toks)
