# Quantization helpers

This directory contains Unlimited-OCR quantization format helpers adapted from
the MIT-licensed DS4 GGUF tools and GGML/llama.cpp quantization metadata.

Attribution:

- Source reference: `data/ds4/gguf-tools/quants.[ch]`
- License: MIT (`data/ds4/LICENSE`)
- Copyright: DS4 authors and GGML authors, as noted in the source license.

The initial Unlimited-OCR runtime contract intentionally enables only:

- `Q8_0`: block size 32, packed block size 34 bytes
- `Q4_K`: block size 256, packed block size 144 bytes

`Q2_K`, `IQ2_XXS`, and padded/custom q4 variants are kept disabled until their
converter policy and Metal kernels are implemented and validated.
