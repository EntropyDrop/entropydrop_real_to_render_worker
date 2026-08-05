import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024
EXPECTED_SHAPE_SIZE = (1024, 1024)
MIN_SHAPE_OVERLAP = 0.95
BACKGROUND_COLOR_TOLERANCE = 0.03
SHAPE_MASK_EROSION_PIXELS = 2
SHAPE_MASK_PATH = (
    Path(__file__).resolve().parent
    / "masks"
    / "front_left_core_back_left_core.png"
)


class InvalidShapeError(ValueError):
    """Raised when a provider render does not match the expected silhouette."""

    def __init__(self, overlap_ratio: float | None = None) -> None:
        self.overlap_ratio = overlap_ratio
        super().__init__("INTERNAL_ERROR:invalid shape. Please try again")


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


def _foreground_mask(image: Image.Image) -> Image.Image:
    """Extract foreground via alpha or a top-left four-connected color flood."""
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] < 255:
        return alpha.point(lambda value: 255 if value > 16 else 0)

    rgb = rgba.convert("RGB")
    background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    channel_differences = ImageChops.difference(rgb, background).split()
    maximum_difference = ImageChops.lighter(
        ImageChops.lighter(
            channel_differences[0],
            channel_differences[1],
        ),
        channel_differences[2],
    )
    background_candidates = maximum_difference.point(
        lambda value: (
            255 if value / 255.0 <= BACKGROUND_COLOR_TOLERANCE else 0
        )
    )
    ImageDraw.floodfill(background_candidates, (0, 0), 128)
    return background_candidates.point(
        lambda value: 0 if value == 128 else 255
    )


@lru_cache(maxsize=8)
def _expected_foreground_mask(
    size: tuple[int, int] = EXPECTED_SHAPE_SIZE,
) -> Image.Image:
    if not SHAPE_MASK_PATH.is_file():
        raise FileNotFoundError(
            f"Stage-1 shape mask does not exist: {SHAPE_MASK_PATH}"
        )

    with Image.open(SHAPE_MASK_PATH) as image:
        image.load()
        if image.size != EXPECTED_SHAPE_SIZE:
            raise ValueError(
                "Stage-1 shape mask must be "
                f"{EXPECTED_SHAPE_SIZE[0]}x{EXPECTED_SHAPE_SIZE[1]}, got "
                f"{image.size[0]}x{image.size[1]}"
            )
        # The committed renderer mask is black on white. Invert it so Pillow
        # can count foreground intersections with 255-valued binary masks.
        foreground = image.convert("L").point(
            lambda value: 255 if value < 128 else 0
        )
        if foreground.size != size:
            foreground = foreground.resize(size, Image.Resampling.NEAREST)

        erosion_pixels = min(
            SHAPE_MASK_EROSION_PIXELS,
            (min(size) - 1) // 2,
        )
        if erosion_pixels:
            kernel_size = erosion_pixels * 2 + 1
            foreground = foreground.filter(ImageFilter.MinFilter(kernel_size))
        return foreground


def combined_render_shape_overlap(content: bytes) -> float:
    """Calculate rendered-foreground intersection / expected-mask pixels."""
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        render_size = image.size
        rendered_foreground = _foreground_mask(image)

    expected_foreground = _expected_foreground_mask(render_size)
    intersection = ImageChops.multiply(
        rendered_foreground,
        expected_foreground,
    )
    intersection_pixels = intersection.histogram()[255]
    expected_pixels = expected_foreground.histogram()[255]
    overlap = (
        intersection_pixels / expected_pixels
        if expected_pixels
        else 0.0
    )
    return overlap


def validate_combined_render_shape(
    content: bytes,
    minimum_overlap: float = MIN_SHAPE_OVERLAP,
) -> float:
    """Return expected-mask coverage, or fail when it is at/below 95%."""
    overlap = combined_render_shape_overlap(content)

    if overlap <= minimum_overlap:
        raise InvalidShapeError(overlap)
    return overlap


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
