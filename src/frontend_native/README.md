# Native frontend placeholder

The v1 frontend is intentionally Python-only: image loading, EXIF handling,
Pillow resizing/padding, normalization, prompt rendering, tokenization, and
output decoding live in `src/unlimitedocr_c/`.

Do not add native tokenizer or image preprocessing code here for v1.  The C core
accepts prepared requests only (`input_ids`, `image_mask`, preprocessed views,
and generation options), which keeps the Metal inference path focused on model
execution and avoids a second preprocessing implementation that could drift from
Hugging Face/Pillow behavior.

Native tokenizer/image loading may be added later only for a standalone C CLI,
after Metal fp16 parity and the Python API are stable.
