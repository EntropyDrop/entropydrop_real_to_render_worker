import io
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageFilter

from images import (
    InvalidShapeError,
    MIN_SHAPE_OVERLAP,
    SHAPE_MASK_PATH,
    combined_render_shape_overlap,
    normalize_combined_render,
    validate_combined_render_shape,
)


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


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_from_shape_mask(
    background=(90, 10, 170),
    foreground=(20, 220, 80),
) -> Image.Image:
    with Image.open(SHAPE_MASK_PATH) as source:
        black_shape = source.convert("L").point(
            lambda value: 255 if value < 128 else 0
        )
    render = Image.new("RGB", black_shape.size, background)
    render.paste(foreground, mask=black_shape)
    return render


def test_shape_check_removes_pure_color_background():
    content = png_bytes(render_from_shape_mask())
    overlap = validate_combined_render_shape(content)
    assert overlap == pytest.approx(1.0)

    with pytest.raises(InvalidShapeError, match="^invalid shape$"):
        validate_combined_render_shape(content, minimum_overlap=1.0)


def test_shape_check_does_not_penalize_outer_skin_silhouette():
    background = (90, 10, 170)
    foreground = (20, 220, 80)
    with Image.open(SHAPE_MASK_PATH) as source:
        black_shape = source.convert("L").point(
            lambda value: 255 if value < 128 else 0
        )
    expanded_shape = black_shape.filter(ImageFilter.MaxFilter(31))
    render = Image.new("RGB", black_shape.size, background)
    render.paste(foreground, mask=expanded_shape)

    overlap = validate_combined_render_shape(png_bytes(render))
    assert overlap == pytest.approx(1.0)


def test_shape_check_rejects_shifted_character():
    render = render_from_shape_mask()
    background = Image.new("RGB", render.size, render.getpixel((0, 0)))
    shifted = ImageChops.offset(render, 80, 0)
    # Clear the wrapped pixels introduced by ImageChops.offset.
    shifted.paste(background.crop((0, 0, 80, render.height)), (0, 0))

    with pytest.raises(InvalidShapeError, match="^invalid shape$"):
        validate_combined_render_shape(png_bytes(shifted))


@pytest.mark.parametrize(
    "template_name",
    ("template41.png", "template51.png", "template52.png"),
)
def test_production_template_shape_overlap_exceeds_threshold(template_name):
    template_path = Path(__file__).parents[1] / "templates" / template_name
    overlap = validate_combined_render_shape(template_path.read_bytes())
    assert overlap == pytest.approx(1.0)
    assert overlap > MIN_SHAPE_OVERLAP


@pytest.mark.parametrize(
    ("filename", "expected_valid"),
    (
        ("img41_failed.png", False),
        ("img37_failed.png", False),
        ("img29.png", True),
    ),
)
def test_real_provider_shape_samples(filename, expected_valid):
    sample_path = Path(__file__).parent / filename
    content = sample_path.read_bytes()
    overlap = combined_render_shape_overlap(content)
    print(f"{filename}: intersection / mask = {overlap:.6%}")

    assert (overlap > MIN_SHAPE_OVERLAP) is expected_valid
    if expected_valid:
        assert validate_combined_render_shape(content) == overlap
    else:
        with pytest.raises(InvalidShapeError) as error:
            validate_combined_render_shape(content)
        assert error.value.overlap_ratio == overlap
