"""Tests for src/data/dataset.py using small synthetic arrays (no network, no files)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import (
    BLOCK_NAMES,
    DatasetSettings,
    ForestPatchDataset,
    block_regions,
    build_datasets_from_arrays,
    compute_band_statistics,
    load_normalization,
    make_dataloaders,
    patch_origins,
    save_normalization,
    validate_split,
)

SIZE = 400
CHANNELS = 6
PATCH = 32
BUFFER = 8


def make_settings(split: dict | None = None) -> DatasetSettings:
    return DatasetSettings(
        train_date="date_a",
        patch_size=PATCH,
        train_stride=16,
        eval_stride=32,
        block_buffer_px=BUFFER,
        max_invalid_fraction=0.2,
        split=split or {"train": ("nw", "sw"), "val": ("se",), "test": ("ne",)},
        forest_class=10,
        batch_size=4,
        num_workers=0,
        seed=42,
    )


def make_arrays(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reflectance = rng.normal(0.1, 0.02, size=(CHANNELS, SIZE, SIZE)).astype(np.float32)
    labels = np.full((SIZE, SIZE), 30, dtype=np.uint8)
    labels[:, : SIZE // 3] = 10  # a forest strip on the west side
    valid = np.ones((SIZE, SIZE), dtype=bool)
    return reflectance, labels, valid


def separation(a: tuple[int, int], b: tuple[int, int], patch: int) -> int:
    """Pixel gap between two square patches (negative if they overlap)."""
    row_gap = max(b[0] - (a[0] + patch), a[0] - (b[0] + patch))
    col_gap = max(b[1] - (a[1] + patch), a[1] - (b[1] + patch))
    return max(row_gap, col_gap)


def test_block_regions_leave_a_gap_and_stay_inside_the_raster() -> None:
    regions = block_regions(SIZE, SIZE, BUFFER)
    assert set(regions) == set(BLOCK_NAMES)
    for row_start, row_stop, col_start, col_stop in regions.values():
        assert 0 <= row_start < row_stop <= SIZE
        assert 0 <= col_start < col_stop <= SIZE
    assert regions["ne"][2] - regions["nw"][3] == 2 * BUFFER  # west-east gap
    assert regions["sw"][0] - regions["nw"][1] == 2 * BUFFER  # north-south gap


def test_block_regions_reject_an_oversized_buffer() -> None:
    with pytest.raises(ValueError):
        block_regions(100, 100, 60)


def test_patch_origins_keep_every_patch_inside_the_region() -> None:
    origins = patch_origins((10, 100, 20, 120), patch_size=32, stride=16)
    assert origins
    for row, col in origins:
        assert row >= 10 and row + 32 <= 100
        assert col >= 20 and col + 32 <= 120
    assert patch_origins((0, 20, 0, 20), patch_size=32, stride=16) == []


def test_validate_split_rejects_shared_unknown_and_missing_blocks() -> None:
    validate_split({"train": ("nw",), "val": ("se",), "test": ("ne",)})
    with pytest.raises(ValueError):
        validate_split({"train": ("nw", "sw"), "val": ("sw",), "test": ("ne",)})
    with pytest.raises(ValueError):
        validate_split({"train": ("nw",), "val": ("xx",), "test": ("ne",)})
    with pytest.raises(ValueError):
        validate_split({"train": ("nw",), "val": ("se",)})


def test_patches_from_different_splits_never_touch() -> None:
    reflectance, labels, valid = make_arrays()
    datasets, _ = build_datasets_from_arrays(reflectance, labels, valid, make_settings())
    names = list(datasets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            for a in datasets[first].origins:
                for b in datasets[second].origins:
                    assert separation(a, b, PATCH) >= 2 * BUFFER


def test_normalisation_statistics_come_from_training_blocks_only() -> None:
    reflectance, labels, valid = make_arrays()
    north_east = block_regions(SIZE, SIZE, BUFFER)["ne"]
    reflectance[:, north_east[0] : north_east[1], north_east[2] : north_east[3]] = 5.0  # test block
    _, (mean, std) = build_datasets_from_arrays(reflectance, labels, valid, make_settings())
    assert np.allclose(mean, 0.1, atol=0.01)
    assert np.all(std < 0.05)


def test_band_statistics_ignore_masked_pixels() -> None:
    reflectance = np.full((2, 10, 10), 1.0, dtype=np.float32)
    reflectance[:, :5, :] = np.nan
    valid = np.ones((10, 10), dtype=bool)
    valid[:5, :] = False
    mean, std = compute_band_statistics(reflectance, valid, [(0, 10, 0, 10)])
    np.testing.assert_allclose(mean, [1.0, 1.0])
    assert np.all(std > 0)  # a constant band gets a tiny positive std, never zero


def test_patches_from_a_fully_masked_block_are_dropped() -> None:
    reflectance, labels, valid = make_arrays()
    north_west = block_regions(SIZE, SIZE, BUFFER)["nw"]
    valid[north_west[0] : north_west[1], north_west[2] : north_west[3]] = False
    reflectance[:, north_west[0] : north_west[1], north_west[2] : north_west[3]] = np.nan
    datasets, _ = build_datasets_from_arrays(reflectance, labels, valid, make_settings())
    assert datasets["train"].origins
    for row, col in datasets["train"].origins:
        assert not (row < north_west[1] and col < north_west[3])


def test_item_has_expected_shapes_values_and_masks() -> None:
    reflectance, labels, valid = make_arrays()
    reflectance[:, 0:8, 0:8] = np.nan
    valid[0:8, 0:8] = False
    labels[100:110, 100:110] = 0  # no WorldCover label here
    zeros = np.zeros(CHANNELS, dtype=np.float32)
    ones = np.ones(CHANNELS, dtype=np.float32)

    dataset = ForestPatchDataset(
        reflectance=reflectance, labels=labels, valid=valid, origins=[(0, 0), (96, 96)],
        patch_size=PATCH, mean=zeros, std=ones, forest_class=10, augment=False,
    )
    first = dataset[0]
    assert first["image"].shape == (CHANNELS, PATCH, PATCH)
    assert first["image"].dtype == torch.float32
    assert torch.isfinite(first["image"]).all()
    assert (first["image"][:, :8, :8] == 0).all()           # masked pixels become 0
    assert first["valid"].sum().item() == PATCH * PATCH - 64  # only the 8 x 8 corner is masked
    assert (first["label"] == 1).all()                        # this patch is inside the forest strip
    assert first["origin"].tolist() == [0, 0]

    second = dataset[1]
    assert (second["valid"][4:14, 4:14] == 0).all()           # unlabelled pixels leave the loss mask
    assert set(second["label"].unique().tolist()) <= {0.0, 1.0}


def test_augmentation_keeps_image_and_label_aligned() -> None:
    reflectance, _, valid = make_arrays()
    labels = np.where(np.random.default_rng(1).random((SIZE, SIZE)) > 0.5, 10, 30).astype(np.uint8)
    reflectance[0] = (labels == 10).astype(np.float32)
    dataset = ForestPatchDataset(
        reflectance=reflectance, labels=labels, valid=valid, origins=[(0, 0), (64, 64)],
        patch_size=PATCH, mean=np.zeros(CHANNELS, dtype=np.float32), std=np.ones(CHANNELS, dtype=np.float32),
        forest_class=10, augment=True,
    )
    torch.manual_seed(0)
    for _ in range(20):
        for index in range(len(dataset)):
            item = dataset[index]
            assert torch.equal(item["image"][0], item["label"])


def test_dataloader_batches_have_expected_shapes() -> None:
    reflectance, labels, valid = make_arrays()
    settings = make_settings()
    datasets, _ = build_datasets_from_arrays(reflectance, labels, valid, settings)
    batch = next(iter(make_dataloaders(datasets, settings)["train"]))
    assert batch["image"].shape == (settings.batch_size, CHANNELS, PATCH, PATCH)
    assert batch["label"].shape == (settings.batch_size, PATCH, PATCH)
    assert batch["valid"].shape == (settings.batch_size, PATCH, PATCH)
    assert batch["origin"].shape == (settings.batch_size, 2)


def test_normalisation_file_round_trip(tmp_path: Path) -> None:
    mean = np.arange(CHANNELS, dtype=np.float32) / 10
    std = np.full(CHANNELS, 0.02, dtype=np.float32)
    path = tmp_path / "normalization.json"
    save_normalization(path, mean, std, make_settings())
    loaded_mean, loaded_std = load_normalization(path)
    np.testing.assert_allclose(loaded_mean, mean)
    np.testing.assert_allclose(loaded_std, std)