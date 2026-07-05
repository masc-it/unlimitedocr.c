# unlimitedocr.c

`unlimitedocr.c` runs the `baidu/Unlimited-OCR` model locally with a native C/Metal
inference engine and a small Python API for OCR workflows.

Python handles the user-facing pieces — image loading, prompt construction,
tokenization, and text decoding. The native library handles model loading, memory
management, KV cache, logits processing, and GPU execution.

## Quick start

```python
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()
text = ocr.generate("page.png")
print(text)
ocr.close()
```

Create one `UnlimitedOCR` instance, reuse it for as many images as you want, and
call `close()` when finished.

```python
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()

for image_path in ["page-1.png", "page-2.png", "page-3.png"]:
    print(ocr.generate(image_path, profile="base"))

ocr.close()
```

## Input types

`generate()` accepts the common image forms directly:

| Input type | Example |
| --- | --- |
| local path | `ocr.generate("page.png")` |
| URL | `ocr.generate("https://example.com/page.jpg")` |
| bytes | `ocr.generate(open("page.png", "rb").read())` |
| file-like object | `ocr.generate(BytesIO(image_bytes))` |
| PIL image | `ocr.generate(Image.open("page.png"))` |
| base64/data URI string | `ocr.generate("data:image/png;base64,...")` |

Example:

```python
from io import BytesIO
from PIL import Image
from unlimitedocr_c import UnlimitedOCR

ocr = UnlimitedOCR()

text_from_path = ocr.generate("page.png")
text_from_url = ocr.generate("https://example.com/page.jpg")
text_from_bytes = ocr.generate(open("page.png", "rb").read())
text_from_file = ocr.generate(BytesIO(open("page.png", "rb").read()))
text_from_pil = ocr.generate(Image.open("page.png"))

ocr.close()
```
