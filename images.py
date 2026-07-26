import base64
import io
from pathlib import Path

from PIL import Image


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024


def image_data_url(content: bytes, content_type: str) -> str:
    if not content_type.startswith("image/"):
        raise ValueError(f"Expected image content type, got {content_type!r}")
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Reference image size must be between 1 and {MAX_IMAGE_BYTES} bytes"
        )

    with Image.open(io.BytesIO(content)) as image:
        image.verify()

    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def template_data_urls(paths: tuple[Path, ...]) -> list[str]:
    urls = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Stage-1 template does not exist: {path}")
        suffix = path.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix)
        if not content_type:
            raise ValueError(f"Unsupported stage-1 template format: {path}")
        urls.append(image_data_url(path.read_bytes(), content_type))
    return urls


def normalize_combined_render(content: bytes) -> tuple[bytes, tuple[int, int]]:
    if not content or len(content) > MAX_RESULT_BYTES:
        raise ValueError(
            f"Provider result size must be between 1 and {MAX_RESULT_BYTES} bytes"
        )

    with Image.open(io.BytesIO(content)) as image:
        image.load()
        width, height = image.size
        if width != height:
            raise ValueError(
                f"Stage-1 output must be square, got {width}x{height}"
            )
        if width < 256 or width % 2:
            raise ValueError(
                f"Stage-1 output width must be even and at least 256, got {width}"
            )

        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG", optimize=True)
        return output.getvalue(), (width, height)
