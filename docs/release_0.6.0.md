# unlimitedocr-c 0.6.0

## Highlights

- Faster v2 mixed-Q4 inference: fewer decoder submissions and encoders, tiled
  DASH decode attention, specialized router projections/routing, and incremental
  tail summaries.
- Faster vision and prefill: v2 SAM window matrix attention, Q4-only CLIP matrix
  attention, 16-row Q4 routed-MoE tiles, and reuse of packed weight groups and
  gathered-row addresses. Q4 LM-head selection and paired gate/up decode also
  received scheduling improvements.
- Lower vision memory: bounded SAM global-attention workspace, shared SAM
  relative-position storage, corrected workspace lifetimes, and removal of
  unused CLIP FC1 storage.
- Fixed a shared-memory race in Q8 and FP16 fused LM-head selection. A SIMD group
  could overwrite a winning row candidate before another group loaded it,
  causing sporadic wrong tokens despite identical hidden inputs. Both paths
  now synchronize before reusing that storage.

## Release-wheel measurements

On an M1 Pro with 16 GB RAM, the released 0.5.0 native artifacts and the 0.6.0
wheel were compared on the same cached v2 Q4 model and synthetic invoice.
Both paths were warmed; four request pairs alternated order. Image preparation
and model loading were excluded, profiling was disabled, and prepared requests
allowed up to 512 output tokens. Complete results had the same 385 IDs.

| Measurement | 0.5.0 | 0.6.0 | Reduction |
|---|---:|---:|---:|
| Base, first token | 1.387 s | 0.926 s | 33% |
| Base, complete invoice | 5.866 s | 2.985 s | 49% |
| Gundam, first token | 6.103 s | 4.250 s | 30% |
| Gundam, complete invoice | 11.281 s | 6.495 s | 42% |
| Base, tracked peak | 3.334 GB | 2.174 GB | 35% |
| Gundam, tracked peak | 4.139 GB | 2.674 GB | 35% |

GB is decimal. These are fixture-specific warmed measurements, not general
throughput or cold-start guarantees. Tracked peak memory is not total RSS or
Metal/device usage. The benchmark uses bounded prepared requests; the high-level
API's larger generation allowance can reserve more memory.

## Compatibility

The Python API, native public API, converted-model formats and quantization
policy are unchanged. Correctly converted 0.5.0 model caches remain compatible;
no reconversion is needed. Defaults remain v1 and FP16. Select v2/Q4 explicitly:

```python
from unlimitedocr_c import UnlimitedOCR, parse_ocr

with UnlimitedOCR(model="v2", quant="q4") as ocr:
    document = parse_ocr(ocr.generate("page.png", profile="base"))
```

The wheel targets **macOS 15+ on Apple Silicon**, requires Python 3.12+, and
bundles the native library, precompiled Metal library and quantization policy.
Profiling and runtime Metal compilation are disabled in the release build;
installed-wheel inference does not require a local compiler.

## Scope and known limitations

- V2 remains **experimental and single-page**. Multi-page generation and
  bounded-memory cross-window continuation are not included. Separate calls
  do not preserve document context.
- Full v2 FP16 OCR and upstream numerical parity remain unverified. The FP16
  selection primitive and v1 FP16 inference were tested.
- Matrix attention changes floating-point order. Tested outputs and numerical
  bounds do not establish universal bitwise parity or general OCR accuracy.
  Q8/FP16/v1 retain scalar CLIP attention.
- Literal OCR tag spellings in documents can collide with structural markup.
  The strict parser still rejects unsupported hierarchy and malformed output;
  adversarial literal-markup inputs can repeat until the token cap.
- Native memory accounting is not a complete process/device memory cap.
  Upload/staging, cached conversion and framework/driver accounting remain
  follow-up work.
- Sparse matrix-tiled prefill and larger vision-GEMM variants were investigated
  but not shipped where complete-request gains did not justify them.

## Release verification

- The actual release wheel was installed into a fresh Python 3.12.10 environment
  and exercised from `/tmp`, with current resolved dependencies. Execution was
  checked on macOS 26.0.1; a physical macOS 15 host was not tested.
- **28 prepared-request checks** covered v1 Q4/Q8/FP16 and v2 Q4/Q8, each in
  base/gundam. All IDs matched the current source build, despite its different
  profiling/deployment settings and dependency versions. Every invoice ended
  naturally and parsed into the expected 15 blocks. Q8 paths repeated four
  times each without variation; the other paths repeated twice.
- **Seven high-level `UnlimitedOCR` calls** covered v1 Q4 base/gundam, v1 FP16
  base, and v2 Q4/Q8 base/gundam, with identical decoded text.
- Packaged dylib/metallib resolution was tested with the checkout's release
  metallib hidden. Missing packaged Metal and quantization-policy files failed
  explicitly. There was no runtime compiler/source fallback.
- The bundled policy matched the source file byte-for-byte. Installed v2
  conversion planning retained the intended Q4/Q8 mix and all 60 DASH tensors
  in FP16. No model cache was reconverted.
- Wheel metadata, platform tag, required resources and all **22 RECORD entries**
  were verified. Both native artifacts target macOS 15.0 / arm64.
- Existing Python checks: **105 passed, 7 opt-in skips**, three expected
  invalid-limit warmup warnings. All **13 native checks** passed with the
  release build's profiling-OFF/runtime-compilation-OFF configuration. Enabling
  the native test targets did not change the wheel's dylib/metallib bytes.

Artifact: `unlimitedocr_c-0.6.0-py3-none-macosx_15_0_arm64.whl`

SHA256: `878df552592602007bb3257839f8ef04e9aceea0d35af940d982e4e28cba3f7a`

Detailed per-checkpoint measurements and limitations are recorded in
[the optimization plan](https://github.com/masc-it/unlimitedocr.c/blob/v0.6.0/docs/plan_v2_q4_optimization.md).
