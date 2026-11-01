"""Preprocessing quality-assurance figure.

Writes outputs/maps/preprocess_qa.png with three panels, all on the shared 10 m grid:
  1. Date A true colour with the WorldCover forest boundary (is the label aligned?),
  2. Date A true colour with pixels masked by the SCL / no-data mask in red,
  3. Date B true colour with its masked pixels in red.

Masked pixels cover well under 1 % of the AOI, so for display only they are dilated and
shown on a 4x-downsampled image; the saved masks are not changed.

Run from the project root:
    python -m src.data.inspect_ard
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")  # write files only; no interactive window
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger("inspect_ard")

PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH: Path = PROJECT_ROOT / "outputs" / "maps" / "preprocess_qa.png"
STEP = 4                    # display every 4th pixel (10 m -> 40 m)
DISPLAY_MAX = 0.25          # fixed stretch, identical for both dates
RGB_BANDS = (3, 2, 1)       # 1-based bands in reflectance.tif: B04 (red), B03 (green), B02 (blue)
DILATE_PX = 9               # display-only thickening of the masked pixels


def load_rgb(date_dir: Path) -> np.ndarray:
    """Read the red, green and blue bands of ``reflectance.tif`` as (rows, cols, 3)."""
    with rasterio.open(date_dir / "reflectance.tif") as src:
        channels = [src.read(index) for index in RGB_BANDS]
    return np.stack(channels, axis=-1)


def to_display(rgb: np.ndarray) -> np.ndarray:
    """Fixed linear stretch with light gamma. NaN (masked) pixels are shown black."""
    scaled = np.clip(np.nan_to_num(rgb, nan=0.0) / DISPLAY_MAX, 0.0, 1.0)
    return scaled ** (1.0 / 1.4)


def read_first_band(path: Path) -> np.ndarray:
    """Read band 1 of a raster."""
    with rasterio.open(path) as src:
        return src.read(1)


def invalid_overlay(valid: np.ndarray) -> np.ndarray:
    """RGBA image: opaque red where pixels are masked, dilated so they stay visible."""
    invalid = (valid == 0).astype(np.uint8)
    dilated = cv2.dilate(invalid, np.ones((DILATE_PX, DILATE_PX), dtype=np.uint8))
    overlay = np.zeros(dilated.shape + (4,), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = (dilated > 0).astype(np.float32)
    return overlay


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description="Create the preprocessing QA figure.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        forest_class = int(config["worldcover"]["forest_class"])
        required = [
            PROCESSED_DIR / "date_a" / "reflectance.tif",
            PROCESSED_DIR / "date_a" / "valid_mask.tif",
            PROCESSED_DIR / "date_b" / "reflectance.tif",
            PROCESSED_DIR / "date_b" / "valid_mask.tif",
            PROCESSED_DIR / "worldcover_10m.tif",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files (run src.data.preprocess first): {missing}")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Problem: %s", exc)
        return 1

    try:
        rgb_a = to_display(load_rgb(PROCESSED_DIR / "date_a"))
        rgb_b = to_display(load_rgb(PROCESSED_DIR / "date_b"))
        valid_a = read_first_band(PROCESSED_DIR / "date_a" / "valid_mask.tif")
        valid_b = read_first_band(PROCESSED_DIR / "date_b" / "valid_mask.tif")
        labels = read_first_band(PROCESSED_DIR / "worldcover_10m.tif")

        shapes = {rgb_a.shape[:2], rgb_b.shape[:2], valid_a.shape, valid_b.shape, labels.shape}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent raster shapes: {shapes}")

        forest = (labels == forest_class).astype(np.float32)
        masked_a = 100.0 * float((valid_a == 0).mean())
        masked_b = 100.0 * float((valid_b == 0).mean())

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

        axes[0].imshow(rgb_a[::STEP, ::STEP])
        axes[0].contour(forest[::STEP, ::STEP], levels=[0.5], colors="yellow", linewidths=0.6)
        axes[0].set_title("Date A + WorldCover forest boundary (yellow)")

        axes[1].imshow(rgb_a[::STEP, ::STEP])
        axes[1].imshow(invalid_overlay(valid_a)[::STEP, ::STEP])
        axes[1].set_title(f"Date A: masked pixels in red ({masked_a:.3f}% of AOI)")

        axes[2].imshow(rgb_b[::STEP, ::STEP])
        axes[2].imshow(invalid_overlay(valid_b)[::STEP, ::STEP])
        axes[2].set_title(f"Date B: masked pixels in red ({masked_b:.3f}% of AOI)")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUTPUT_PATH, dpi=130)
        plt.close(fig)
    except Exception:
        logger.exception("QA figure failed")
        return 1

    logger.info("Saved %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())