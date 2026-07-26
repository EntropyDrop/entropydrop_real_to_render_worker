import io

import pytest
from PIL import Image

from images import normalize_combined_render


def image_bytes(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "green").save(buffer, format="PNG")
    return buffer.getvalue()


def test_normalizes_square_combined_render():
    content, dimensions = normalize_combined_render(image_bytes(1024, 1024))
    assert dimensions == (1024, 1024)
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (1024, 1024)
        assert image.mode == "RGB"


def test_rejects_non_square_provider_output():
    with pytest.raises(ValueError, match="must be square"):
        normalize_combined_render(image_bytes(1024, 768))
