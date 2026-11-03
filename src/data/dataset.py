"""PyTorch dataset with spatial-block train / validation / test splits.

The AOI is cut into four quadrants (blocks): nw, ne, sw, se. Every block belongs to exactly
one split, and a margin of ``block_buffer_px`` is trimmed from every block edge that touches
another block, so patches from different splits are at least ``2 * block_buffer_px`` pixels
apart. This prevents the spatial leakage that random patch splits would cause.

Samples come from one date only (``dataset.train_date``, Date A), because the WorldCover
labels describe roughly the same year as Date A. Per-band normalisation statistics are
computed from the TRAINING blocks only and saved to ``data/processed/normalization.json``
so that inference on other dates uses exactly the same scaling.

Each sample is a dictionary:
    image   float32 (bands, patch, patch)  standardised reflectance, masked pixels = 0
    label   float32 (patch, patch)         1 = forest (WorldCover class), 0 = other
    valid   float32 (patch, patch)         1 = usable AND labelled; the loss must use this
    origin  int64   (2,)                   (row, col) of the patch in the full raster

Run a smoke test and print per-block statistics from the project root:
    python -m src.data.dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocess import BAND_ORDER
from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger("dataset")

PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
NORMALIZATION_PATH: Path = PROCESSED_DIR / "normalization.json"
BLOCK_NAMES: tuple[str, ...] = ("nw", "ne", "sw", "se")
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
MIN_STD = 1e-6

# (row_start, row_stop, col_start, col_stop); stop values are exclusive.
Region = tuple[int, int, int, int]


@dataclass(frozen=True)
class DatasetSettings:
    """Dataset parameters, normally read from the ``dataset`` section of config.yaml."""

    train_date: str
    patch_size: int
    train_stride: int
    eval_stride: int
    block_buffer_px: int
    max_invalid_fraction: float
    split: dict[str, tuple[str, ...]]
    forest_class: int
    batch_size: int
    num_workers: int
    seed: int


# --------------------------------------------------------------------------------------
# Settings and split validation
# --------------------------------------------------------------------------------------
def validate_split(split: dict[str, Sequence[str]]) -> None:
    """Raise ValueError unless every block is known and belongs to at most one split."""
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        if name not in split or not split[name]:
            raise ValueError(f"Split '{name}' must contain at least one block")
        for block in split[name]:
            if block not in BLOCK_NAMES:
                raise ValueError(f"Unknown block '{block}' in split '{name}'; valid blocks: {BLOCK_NAMES}")
            if block in seen:
                raise ValueError(f"Block '{block}' is in both '{seen[block]}' and '{name}' (spatial leakage)")
            seen[block] = name


def validate_settings(settings: DatasetSettings) -> None:
    """Raise ValueError for impossible parameter values."""
    for field in ("patch_size", "train_stride", "eval_stride", "batch_size"):
        if getattr(settings, field) <= 0:
            raise ValueError(f"dataset.{field} must be positive")
    if settings.block_buffer_px < 0:
        raise ValueError("dataset.block_buffer_px must not be negative")
    if settings.num_workers < 0:
        raise ValueError("dataset.num_workers must not be negative")
    if not 0.0 <= settings.max_invalid_fraction <= 1.0:
        raise ValueError("dataset.max_invalid_fraction must be between 0 and 1")
    validate_split(settings.split)


def settings_from_config(config: dict[str, Any]) -> DatasetSettings:
    """Build validated settings from the loaded configuration dictionary."""
    section = config["dataset"]
    settings = DatasetSettings(
        train_date=str(section["train_date"]),
        patch_size=int(section["patch_size"]),
        train_stride=int(section["train_stride"]),
        eval_stride=int(section["eval_stride"]),
        block_buffer_px=int(section["block_buffer_px"]),
        max_invalid_fraction=float(section["max_invalid_fraction"]),
        split={name: tuple(str(b) for b in section["split"][name]) for name in SPLIT_NAMES},
        forest_class=int(config["worldcover"]["forest_class"]),
        batch_size=int(section["batch_size"]),
        num_workers=int(section["num_workers"]),
        seed=int(config["project"]["random_seed"]),
    )
    validate_settings(settings)
    return settings


# --------------------------------------------------------------------------------------
# Block and patch geometry
# --------------------------------------------------------------------------------------
def block_regions(height: int, width: int, buffer_px: int) -> dict[str, Region]:
    """Return the four quadrants with ``buffer_px`` trimmed from each inner edge.

    Neighbouring blocks are therefore separated by ``2 * buffer_px`` pixels.

    Raises:
        ValueError: If the buffer is negative or too large for the raster.
    """
    if buffer_px < 0:
        raise ValueError("buffer_px must not be negative")
    mid_row, mid_col = height // 2, width // 2
    if mid_row - buffer_px <= 0 or mid_col - buffer_px <= 0:
        raise ValueError(f"buffer_px={buffer_px} is too large for a {height} x {width} raster")
    return {
        "nw": (0, mid_row - buffer_px, 0, mid_col - buffer_px),
        "ne": (0, mid_row - buffer_px, mid_col + buffer_px, width),
        "sw": (mid_row + buffer_px, height, 0, mid_col - buffer_px),
        "se": (mid_row + buffer_px, height, mid_col + buffer_px, width),
    }


def patch_origins(region: Region, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners of all patches that lie completely inside ``region``."""
    row_start, row_stop, col_start, col_stop = region
    rows = range(row_start, row_stop - patch_size + 1, stride)
    cols = range(col_start, col_stop - patch_size + 1, stride)
    return [(row, col) for row in rows for col in cols]


def filter_origins(
    origins: Sequence[tuple[int, int]], loss_mask: np.ndarray, patch_size: int, max_invalid_fraction: float
) -> list[tuple[int, int]]:
    """Keep patches whose masked / unlabelled fraction does not exceed the limit."""
    kept: list[tuple[int, int]] = []
    for row, col in origins:
        window = loss_mask[row : row + patch_size, col : col + patch_size]
        if 1.0 - float(window.mean()) <= max_invalid_fraction:
            kept.append((row, col))
    return kept


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
def compute_band_statistics(
    reflectance: np.ndarray, valid: np.ndarray, regions: Sequence[Region]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and standard deviation over valid pixels inside ``regions``.

    Raises:
        ValueError: If no valid pixel exists or the statistics are not finite.
    """
    channels = reflectance.shape[0]
    count = 0
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    for row_start, row_stop, col_start, col_stop in regions:
        block = reflectance[:, row_start:row_stop, col_start:col_stop]
        block_valid = valid[row_start:row_stop, col_start:col_stop]
        values = block[:, block_valid].astype(np.float64)  # shape (channels, n_valid)
        count += values.shape[1]
        total += values.sum(axis=1)
        total_sq += np.square(values).sum(axis=1)
    if count == 0:
        raise ValueError("No valid training pixels available to compute normalisation statistics")
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(total_sq / count - mean**2, 0.0)), MIN_STD)
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("Normalisation statistics are not finite; the reflectance contains NaN in valid pixels")
    return mean.astype(np.float32), std.astype(np.float32)


def save_normalization(path: Path, mean: np.ndarray, std: np.ndarray, settings: DatasetSettings) -> None:
    """Write the normalisation statistics and where they came from to a JSON file."""
    payload = {
        "band_order": list(BAND_ORDER),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "computed_from": {
            "date": settings.train_date,
            "blocks": list(settings.split["train"]),
            "block_buffer_px": settings.block_buffer_px,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read normalisation statistics written by ``save_normalization``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["band_order"] != list(BAND_ORDER):
        raise ValueError(f"Band order in {path} does not match {list(BAND_ORDER)}")
    return np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def random_flip_rotate(stack: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip, vertical flip and 90-degree rotation of a (C, H, W) tensor.

    Image, label and mask are stacked together before this is called, so they always stay aligned.
    """
    if torch.rand(1).item() < 0.5:
        stack = torch.flip(stack, dims=[-1])
    if torch.rand(1).item() < 0.5:
        stack = torch.flip(stack, dims=[-2])
    quarter_turns = int(torch.randint(0, 4, (1,)).item())
    if quarter_turns:
        stack = torch.rot90(stack, quarter_turns, dims=(-2, -1))
    return stack


class ForestPatchDataset(Dataset):
    """Square patches cut on the fly from in-memory arrays of one date."""

    def __init__(
        self,
        reflectance: np.ndarray,
        labels: np.ndarray,
        valid: np.ndarray,
        origins: Sequence[tuple[int, int]],
        patch_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        forest_class: int,
        augment: bool = False,
    ) -> None:
        """Create the dataset.

        Args:
            reflectance: float32 array (bands, rows, cols); NaN where masked.
            labels: uint8 WorldCover class codes (rows, cols); 0 = no label.
            valid: boolean mask (rows, cols); True = usable pixel.
            origins: Top-left (row, col) of each patch.
            patch_size: Patch edge length in pixels.
            mean: Per-band mean used for standardisation.
            std: Per-band standard deviation used for standardisation.
            forest_class: WorldCover class code counted as forest.
            augment: Apply random flips and rotations (training only).
        """
        if reflectance.ndim != 3:
            raise ValueError(f"reflectance must be (bands, rows, cols), got {reflectance.shape}")
        if labels.shape != reflectance.shape[1:] or valid.shape != reflectance.shape[1:]:
            raise ValueError("reflectance, labels and valid must share the same rows and columns")
        if valid.dtype != bool:
            raise ValueError(f"valid must be a boolean array, got {valid.dtype}")
        if len(origins) == 0:
            raise ValueError("A dataset needs at least one patch")
        height, width = valid.shape
        for row, col in origins:
            if row < 0 or col < 0 or row + patch_size > height or col + patch_size > width:
                raise ValueError(f"Patch at ({row}, {col}) of size {patch_size} leaves the raster")
        if mean.shape != (reflectance.shape[0],) or std.shape != mean.shape:
            raise ValueError("mean and std need one value per band")

        self.reflectance = reflectance
        self.labels = labels
        self.valid = valid
        self.origins = list(origins)
        self.patch_size = patch_size
        self.mean = mean.reshape(-1, 1, 1).astype(np.float32)
        self.std = std.reshape(-1, 1, 1).astype(np.float32)
        self.forest_class = forest_class
        self.augment = augment

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, col = self.origins[index]
        rows = slice(row, row + self.patch_size)
        cols = slice(col, col + self.patch_size)

        raw = self.reflectance[:, rows, cols]
        classes = self.labels[rows, cols]
        usable = self.valid[rows, cols] & (classes != 0)

        image = np.nan_to_num((raw - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        label = (classes == self.forest_class).astype(np.float32)
        loss_mask = usable.astype(np.float32)

        stack = torch.from_numpy(np.concatenate([image, label[None], loss_mask[None]], axis=0))
        if self.augment:
            stack = random_flip_rotate(stack)
        return {
            "image": stack[:-2].contiguous(),
            "label": stack[-2].contiguous(),
            "valid": stack[-1].contiguous(),
            "origin": torch.tensor([row, col], dtype=torch.int64),
        }


# --------------------------------------------------------------------------------------
# Building datasets and loaders
# --------------------------------------------------------------------------------------
def load_processed_arrays(processed_dir: Path, date_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load reflectance, WorldCover labels and the valid mask written by the preprocessing step."""
    paths = {
        "reflectance": processed_dir / date_key / "reflectance.tif",
        "valid": processed_dir / date_key / "valid_mask.tif",
        "labels": processed_dir / "worldcover_10m.tif",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files (run src.data.preprocess first): {missing}")
    with rasterio.open(paths["reflectance"]) as src:
        reflectance = src.read().astype(np.float32)
    with rasterio.open(paths["valid"]) as src:
        valid = src.read(1).astype(bool)
    with rasterio.open(paths["labels"]) as src:
        labels = src.read(1)
    return reflectance, labels, valid


def build_datasets_from_arrays(
    reflectance: np.ndarray, labels: np.ndarray, valid: np.ndarray, settings: DatasetSettings
) -> tuple[dict[str, ForestPatchDataset], tuple[np.ndarray, np.ndarray]]:
    """Create the train / val / test datasets and the training-only normalisation statistics."""
    validate_settings(settings)
    if reflectance.ndim != 3 or valid.shape != reflectance.shape[1:] or labels.shape != valid.shape:
        raise ValueError("Array shapes are inconsistent")

    height, width = valid.shape
    regions = block_regions(height, width, settings.block_buffer_px)
    loss_mask = valid & (labels != 0)

    train_regions = [regions[block] for block in settings.split["train"]]
    mean, std = compute_band_statistics(reflectance, valid, train_regions)

    datasets: dict[str, ForestPatchDataset] = {}
    for name in SPLIT_NAMES:
        stride = settings.train_stride if name == "train" else settings.eval_stride
        origins: list[tuple[int, int]] = []
        for block in settings.split[name]:
            origins += patch_origins(regions[block], settings.patch_size, stride)
        origins = filter_origins(origins, loss_mask, settings.patch_size, settings.max_invalid_fraction)
        if not origins:
            raise ValueError(f"Split '{name}' has no usable patches; reduce patch_size or block_buffer_px")
        datasets[name] = ForestPatchDataset(
            reflectance=reflectance,
            labels=labels,
            valid=valid,
            origins=origins,
            patch_size=settings.patch_size,
            mean=mean,
            std=std,
            forest_class=settings.forest_class,
            augment=(name == "train"),
        )
    return datasets, (mean, std)


def make_dataloaders(datasets: dict[str, ForestPatchDataset], settings: DatasetSettings) -> dict[str, DataLoader]:
    """Create one DataLoader per split. Seeds torch so shuffling and augmentation are reproducible."""
    torch.manual_seed(settings.seed)
    generator = torch.Generator()
    generator.manual_seed(settings.seed)
    loaders: dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=is_train,
            num_workers=settings.num_workers,
            generator=generator if is_train else None,
        )
    return loaders


def summarise_blocks(
    labels: np.ndarray, valid: np.ndarray, regions: dict[str, Region], settings: DatasetSettings
) -> list[dict[str, Any]]:
    """Forest and masked fractions for each of the four blocks."""
    rows: list[dict[str, Any]] = []
    for block in BLOCK_NAMES:
        row_start, row_stop, col_start, col_stop = regions[block]
        region_labels = labels[row_start:row_stop, col_start:col_stop]
        region_valid = valid[row_start:row_stop, col_start:col_stop]
        usable = region_valid & (region_labels != 0)
        forest = (region_labels == settings.forest_class) & usable
        usable_count = int(usable.sum())
        rows.append(
            {
                "block": block,
                "split": next((n for n in SPLIT_NAMES if block in settings.split[n]), "unused"),
                "region": regions[block],
                "forest_fraction": float(forest.sum()) / usable_count if usable_count else float("nan"),
                "masked_fraction": 1.0 - float(region_valid.mean()),
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# Entry point: statistics and smoke test
# --------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Print per-block statistics, save normalisation, and load one training batch."""
    parser = argparse.ArgumentParser(description="Dataset statistics and smoke test.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        settings = settings_from_config(config)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        logger.error("Configuration problem: %s", exc)
        return 1

    try:
        reflectance, labels, valid = load_processed_arrays(PROCESSED_DIR, settings.train_date)
        height, width = valid.shape
        logger.info("Loaded %s: %d bands, %d x %d pixels", settings.train_date, reflectance.shape[0], height, width)

        regions = block_regions(height, width, settings.block_buffer_px)
        logger.info("Blocks (gap between neighbouring blocks = %d px):", 2 * settings.block_buffer_px)
        for info in summarise_blocks(labels, valid, regions, settings):
            r0, r1, c0, c1 = info["region"]
            logger.info(
                "  %s | %-6s | rows %4d:%4d cols %4d:%4d | forest %5.1f%% | masked %.3f%%",
                info["block"], info["split"], r0, r1, c0, c1,
                100 * info["forest_fraction"], 100 * info["masked_fraction"],
            )

        datasets, (mean, std) = build_datasets_from_arrays(reflectance, labels, valid, settings)
        for name, dataset in datasets.items():
            logger.info("  %-5s patches: %d", name, len(dataset))

        logger.info("Normalisation (training blocks %s only):", list(settings.split["train"]))
        for band, band_mean, band_std in zip(BAND_ORDER, mean, std):
            logger.info("  %s mean %.4f std %.4f", band, band_mean, band_std)
        save_normalization(NORMALIZATION_PATH, mean, std, settings)
        logger.info("Wrote %s", NORMALIZATION_PATH)

        loaders = make_dataloaders(datasets, settings)
        batch = next(iter(loaders["train"]))
        image = batch["image"]
        logger.info(
            "Train batch: image %s %s | label %s | valid %s",
            tuple(image.shape), image.dtype, tuple(batch["label"].shape), tuple(batch["valid"].shape),
        )
        logger.info(
            "  image range %.2f to %.2f (mean %.3f) | forest fraction %.3f | loss-mask fraction %.4f",
            float(image.min()), float(image.max()), float(image.mean()),
            float(batch["label"].mean()), float(batch["valid"].mean()),
        )
    except Exception:
        logger.exception("Dataset check failed")
        return 1

    logger.info("DATASET CHECK COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())