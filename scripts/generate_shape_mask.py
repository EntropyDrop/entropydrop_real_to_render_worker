#!/usr/bin/env python3
"""Build the committed stage-1 silhouette from renderer core mappings."""

import argparse
from pathlib import Path

from PIL import Image
import torch


VIEW_NAMES = ("front_left_core", "back_left_core")
VIEW_SIZE = (512, 1024)
OUTPUT_SIZE = (1024, 1024)


def generate_shape_mask(renderer_dir: Path, output_path: Path) -> None:
    mappings_dir = renderer_dir / "mappings_512x1024"
    views = []
    for view_name in VIEW_NAMES:
        mapping_path = mappings_dir / f"{view_name}_mapping.pt"
        mapping = torch.load(
            mapping_path,
            map_location="cpu",
            weights_only=False,
        )
        inner_mask = mapping["inner_mask"]
        if tuple(reversed(inner_mask.shape)) != VIEW_SIZE:
            raise ValueError(
                f"Unexpected {view_name} mask size: "
                f"{tuple(reversed(inner_mask.shape))}"
            )
        pixels = (
            (inner_mask <= 0.5)
            .to(dtype=torch.uint8)
            .mul(255)
            .cpu()
            .numpy()
        )
        views.append(Image.fromarray(pixels, mode="L"))

    output = Image.new("L", OUTPUT_SIZE, 255)
    for index, view in enumerate(views):
        output.paste(view, (index * VIEW_SIZE[0], 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path, format="PNG", optimize=True)


def main() -> None:
    worker_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer-dir",
        type=Path,
        default=worker_dir.parent / "differentiable_minecraft_renderer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            worker_dir
            / "masks"
            / "front_left_core_back_left_core.png"
        ),
    )
    args = parser.parse_args()
    generate_shape_mask(args.renderer_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
