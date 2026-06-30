# Memory Optimization Implementation Plan

Goal: reduce peak inference memory for UnlimitedOCR Metal inference while preserving exact fp16 model behavior and output parity with the reference Python implementation.

This plan is grounded in the upstream model architecture in `data.tmp/reference/deepencoder.py`, `data.tmp/reference/modeling_unlimitedocr.py`, and `data.tmp/reference/modeling_deepseekv2.py`.

## 1. Establish and Preserve Memory Baselines

Record baseline memory and speed before each optimization pass.

Tasks:
- Run `uv run probes/e2e_generation_probe.py --profile base --json`.
- Run `uv run probes/e2e_generation_probe.py --profile gundam --json`.
- Track native peak, live memory after generation, RSS sampled peak, generated token count, and tok/s.
- Keep category-level accounting visible for:
  - `model_views`
  - `vision_gpu_workspace`
  - `kv_cache`
  - `decoder_scratch`
  - `moe_scratch`
  - `transient_buffers`
- Treat `model_views` as the fp16 model mapping floor, separate from true transient runtime pressure.

Acceptance criteria:
- Baseline numbers are documented before code changes.
- Probe output clearly separates model mapping from runtime workspace pressure.

## 2. Add Request-Level Vision Schedule Reporting

Make the planned vision schedule visible so memory behavior can be explained from request shape.

Tasks:
- Report local view count, global view count, chunk count, max chunk views, max chunk projected rows, final visual rows, and max view size.
- Surface this information in probe JSON/profile output.
- For crop requests, report local and global chunk shapes separately.

Acceptance criteria:
- `base` shows one global view and one global chunk.
- `gundam` shows local crop chunks plus one global chunk.
- Schedule metadata is present in probe output without extra diagnostic host copies.

## 3. Implement Memory-Aware Local Crop Chunking

Process local crops in smaller independent batches instead of all local crops at once.

Reference grounding:
- Upstream encodes local crops as a batch with `sam_model(patches)` and `vision_model(patches, local_features_1)`.
- There is no cross-crop attention before projected features are formatted into the final visual grid.

Tasks:
- Add a configurable local max-views-per-chunk policy.
- Default to a memory-friendly cap such as 4 or 8 local views per chunk.
- Preserve current behavior behind an internal setting if useful for parity/perf comparison.
- Ensure output rows are written to the same final visual positions as the current all-local-crops chunk.

Acceptance criteria:
- `gundam` peak `vision_gpu_workspace` drops materially.
- Generated text/token sequence remains unchanged for deterministic probes.
- `base` behavior and memory stay essentially unchanged.

## 4. Use Separate Local and Global Vision Workspace Shapes

Allocate workspace for the actual chunk kind rather than the largest view in the whole request.

Reference grounding:
- Upstream runs local patches and global image as separate SAM/CLIP calls:
  - local: 640 input, SAM `40x40`, projector grid `10x10`
  - global: 1024 input, SAM `64x64`, projector grid `16x16`

Tasks:
- Stop sizing all crop-mode vision workspace from the request-wide max view size.
- Size local chunks with the 640/local shape.
- Size global chunks with the 1024/global shape.
- Keep final visual row storage independent of per-chunk scratch shape.
- Update memory estimates to account for max(local_chunk_workspace, global_chunk_workspace) plus final visual storage.

Acceptance criteria:
- Crop-mode local chunk workspace no longer pays for 1024 SAM/CLIP scratch.
- Memory accounting matches actual Metal allocations.
- Parity probes continue to pass.

Status:
- Implemented with per-kind Metal vision scratch: local chunks use the 640/local shape and global chunks use the 1024/global shape.
- Final visual rows now live in a separate final-feature scratch slot so local and global scratch layouts can be rebuilt independently.
- Full release of chunk scratch after prompt assembly remains part of the next step.

## 5. Split Final Visual Rows from Large Vision Scratch

Keep final visual embeddings in a small persistent buffer and release/reuse the large SAM/CLIP scratch before decoder prefill.

Reference grounding:
- Upstream inserts final image features into token embeddings with `masked_scatter_` during prefill.
- Images are passed only on prefill; decode receives no image tensors.

Tasks:
- Allocate final visual rows separately from temporary SAM/CLIP/projector workspace.
- Write formatted local/global visual rows into the final visual buffer.
- Release or recycle the large vision scratch once final rows are complete.
- Bind the final visual buffer directly during prompt assembly.
- Release final visual rows after prefill when safe.

Acceptance criteria:
- Live memory after vision and after prefill drops for crop-heavy requests.
- Prompt assembly still avoids host readback.
- No extra persistent copy of visual features remains after prefill.

Status:
- The large per-chunk SAM/CLIP/projector scratch is released after final visual rows are complete and before decoder prompt assembly.
- Final visual rows are kept in the dedicated final-feature scratch just long enough for GPU prompt assembly and prefill, then released before decode.
- The prompt assembly path continues to bind the GPU-resident final visual buffer directly.

## 6. Defer Decoder Arena Allocation Until After Vision

Avoid overlapping large vision scratch with decoder-only runtime arenas when possible.

Tasks:
- Allocate or activate KV cache, hidden prompt arena, decoder scratch, MoE scratch, and logits buffers after vision encoding finishes.
- Keep model views and vision weights available throughout.
- Preserve existing reuse behavior for text-only generation.
- Ensure failure paths clean up partially allocated resources.

Acceptance criteria:
- Peak memory for crop-mode requests drops by avoiding vision/decoder overlap.
- Text-only and base image requests still initialize correctly.
- E2E probe passes after Release rebuild.

Status:
- Metal image generation now releases any existing decoder runtime arenas before vision encoding.
- Runtime arena allocation is deferred through a callback until after vision chunk scratch has been released.
- Request memory estimates now model vision and decoder runtime as non-overlapping phases, with final visual rows overlapping prompt assembly/prefill.

## 7. Shrink Decode-Time Scratch After Prefill

Use prompt-sized scratch only for prefill, then switch to one-token decode scratch.

Reference grounding:
- The decoder prefill processes the full prompt.
- Decode proceeds one token at a time with a 128-token generated ring over the prompt cache.

Tasks:
- Keep prompt-sized hidden/decoder/MoE scratch for integrated prefill.
- After prefill, release or alias prompt-sized scratch to smaller decode scratch where safe.
- Ensure the KV cache remains `prompt + 128 generated ring` per slot.
- Keep final-token hidden available for first decode logits.

Acceptance criteria:
- Live memory after prefill drops.
- KV cache behavior remains aligned with the upstream sliding-window ring.
- Generation output remains stable.

## 8. Keep KV Cache Semantics Aligned with Upstream

Maintain the current standard MHA cache behavior.

Reference grounding:
- Actual config has `use_mla: false`.
- The model uses `SlidingWindowLlamaAttention`, not DeepSeek-V2 MLA compressed KV.
- Prompt tokens remain fully attendable; generated tokens use a 128-token ring.

Tasks:
- Preserve current `prompt + UOCR_GENERATED_RING_WINDOW` cache sizing.
- Keep generated ring overwrite semantics unchanged.
- Avoid implementing MLA/compressed-KV-specific memory changes for this model.
- Add tests or assertions around prompt length, generated ring length, and decode attention mapping.

Acceptance criteria:
- KV memory estimates stay consistent with model config.
- Ring-buffer decode tests continue to pass.

## 9. Update Tests and Probes

Expand validation around the memory-oriented changes.

Tasks:
- Add unit tests for local/global vision schedule chunking.
- Add memory-estimate tests for:
  - base single global view
  - crop mode with multiple local chunks
  - local and global workspace shape separation
  - final visual buffer separated from scratch
- Add probe fields for scheduled chunk shapes and actual allocation peaks.
- Keep frontend parity tests grounded in the cached upstream source.

Acceptance criteria:
- C tests and Python tests pass.
- Probe JSON gives enough detail to explain peak memory changes.

## 10. Rebuild and Validate Release Path

Validate every native-memory change through the Release Metal path.

Tasks:
- Reconfigure and rebuild Release:
  `cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DUOCR_ENABLE_METAL=ON -DUOCR_BUILD_TOOLS=ON -DUOCR_BUILD_TESTS=OFF -DUOCR_METAL_PRECOMPILE=ON && cmake --build build/release -j`
- Run:
  `uv run probes/e2e_generation_probe.py --profile base`
- Run crop-heavy validation:
  `uv run probes/e2e_generation_probe.py --profile gundam`
- Restart any notebook/Jupyter kernel after native dylib changes.

Acceptance criteria:
- Release build succeeds.
- Base E2E probe passes.
- Gundam E2E probe passes.
- Memory peak reduction is visible and documented.
