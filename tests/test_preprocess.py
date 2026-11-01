"""Tests for src/data/preprocess.py using small synthetic rasters (no network needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from src.data.preprocess import (
    BAND_ORDER,
    build_valid_mask,
    check_half_resolution_grid,
    check_same_grid,
    dn_to_reflectance,
    process_date,
    reflectance_offset,
    reproject_worldcover,
    upsample_nearest_2x,
)

CRS_UTM = "EPSG:32720"
ORIGIN_X, ORIGIN_Y = 600_000.0, 8_880_000.0
INVALID_CLASSES = [0, 1, 3, 7, 8, 9, 10, 11]


def write_test_raster(
    path: Path,
    array: np.ndarray,
    pixel_size: float,
    origin: tuple[float, float] = (ORIGIN_X, ORIGIN_Y),
) -> None:
    """Write a 2D array as a single-band GeoTIFF in UTM 20S."""
    data = array[np.newaxis]
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=1,
        dtype=data.dtype.name, crs=CRS_UTM,
        transform=from_origin(origin[0], origin[1], pixel_size, pixel_size),
    ) as dst:
        dst.write(data)


@pytest.mark.parametrize("baseline, expected", [("03.00", 0.0), ("04.00", -0.1), ("05.11", -0.1), (3.01, 0.0)])
def test_reflectance_offset_by_baseline(baseline: str | float, expected: float) -> None:
    assert reflectance_offset(baseline) == pytest.approx(expected)


def test_reflectance_offset_rejects_missing_or_invalid_baseline() -> None:
    with pytest.raises(ValueError):
        reflectance_offset(None)
    with pytest.raises(ValueError):
        reflectance_offset("not-a-number")


def test_dn_to_reflectance_scales_offsets_and_clips() -> None:
    dn = np.array([[500, 1000, 3000, 20000]], dtype=np.uint16)
    result = dn_to_reflectance(dn, offset=-0.1, clip_range=(0.0, 1.0))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.0, 0.0, 0.2, 1.0]], atol=1e-6)


def test_upsample_nearest_2x_turns_each_pixel_into_a_2x2_block() -> None:
    small = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8
    )
    np.testing.assert_array_equal(upsample_nearest_2x(small), expected)


def test_build_valid_mask_applies_scl_classes_and_nodata() -> None:
    scl = np.array([[4, 9], [3, 5]], dtype=np.uint8)
    dn = np.full((6, 2, 2), 2000, dtype=np.uint16)
    dn[2, 1, 1] = 0  # one band has no data at pixel (1, 1)
    valid = build_valid_mask(scl, dn, INVALID_CLASSES)
    np.testing.assert_array_equal(valid, np.array([[True, False], [False, False]]))


def test_grid_checks_accept_matching_and_reject_mismatched(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tif"
    same = tmp_path / "same.tif"
    half = tmp_path / "half.tif"
    shifted = tmp_path / "shifted.tif"
    write_test_raster(ref, np.ones((6, 8), dtype=np.uint16), 10.0)
    write_test_raster(same, np.ones((6, 8), dtype=np.uint16), 10.0)
    write_test_raster(half, np.ones((3, 4), dtype=np.uint16), 20.0)
    write_test_raster(shifted, np.ones((3, 4), dtype=np.uint16), 20.0, origin=(ORIGIN_X + 10.0, ORIGIN_Y))

    check_same_grid(ref, same)
    check_half_resolution_grid(ref, half)
    with pytest.raises(ValueError):
        check_same_grid(ref, half)
    with pytest.raises(ValueError):
        check_half_resolution_grid(ref, shifted)


def build_raw_scene(raw_dir: Path) -> None:
    """A tiny synthetic raw scene: 6 x 8 pixels at 10 m, with one cloudy 20 m pixel."""
    raw_dir.mkdir(parents=True)
    for band in ("B02", "B03", "B04", "B08"):
        write_test_raster(raw_dir / f"{band}.tif", np.full((6, 8), 2000, dtype=np.uint16), 10.0)
    for band in ("B11", "B12"):
        write_test_raster(raw_dir / f"{band}.tif", np.full((3, 4), 3000, dtype=np.uint16), 20.0)
    scl = np.full((3, 4), 4, dtype=np.uint8)
    scl[0, 0] = 9  # high-probability cloud in the top-left 20 m pixel
    write_test_raster(raw_dir / "SCL.tif", scl, 20.0)


def test_process_date_writes_aligned_masked_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "date_x"
    out_dir = tmp_path / "processed" / "date_x"
    build_raw_scene(raw_dir)

    stats, valid = process_date("date_x", raw_dir, out_dir, "05.11", INVALID_CLASSES, (0.0, 1.0))

    with rasterio.open(out_dir / "reflectance.tif") as src:
        assert src.count == 6
        assert (src.height, src.width) == (6, 8)
        assert str(src.crs) == CRS_UTM
        assert src.descriptions == BAND_ORDER
        reflectance = src.read()
    with rasterio.open(out_dir / "valid_mask.tif") as src:
        saved_mask = src.read(1)
    with rasterio.open(out_dir / "scl_10m.tif") as src:
        assert (src.height, src.width) == (6, 8)

    # B02 DN 2000 with baseline 05.11: 0.2 - 0.1 = 0.1.  B11 DN 3000: 0.3 - 0.1 = 0.2.
    assert reflectance[0, 3, 3] == pytest.approx(0.1, abs=1e-5)
    assert reflectance[4, 3, 3] == pytest.approx(0.2, abs=1e-5)
    # The cloudy 20 m pixel becomes a 2 x 2 block of NaN in all six bands, and nothing else.
    assert np.isnan(reflectance[:, :2, :2]).all()
    assert int(np.isnan(reflectance).sum()) == 6 * 4
    assert valid[:2, :2].sum() == 0
    assert int(valid.sum()) == 44
    np.testing.assert_array_equal(saved_mask, valid.astype(np.uint8))
    assert stats["invalid_fraction"] == pytest.approx(4 / 48, abs=1e-6)


def test_reproject_worldcover_aligns_labels_to_the_utm_grid(tmp_path: Path) -> None:
    height = width = 100
    transform = from_origin(ORIGIN_X, ORIGIN_Y, 10.0, 10.0)  # a 1 km x 1 km UTM square
    west, south, east, north = transform_bounds(
        CRS_UTM, "EPSG:4326", ORIGIN_X, ORIGIN_Y - 1000.0, ORIGIN_X + 1000.0, ORIGIN_Y, densify_pts=21
    )

    # A synthetic WorldCover map in EPSG:4326: class 10 west of the middle, class 30 east of it.
    pixel = 1.0 / 12000.0
    margin = 0.01
    wc_west, wc_north = west - margin, north + margin
    wc_width = int(np.ceil((east - west + 2 * margin) / pixel))
    wc_height = int(np.ceil((north - south + 2 * margin) / pixel))
    classes = np.full((wc_height, wc_width), 30, dtype=np.uint8)
    classes[:, : int(((west + east) / 2 - wc_west) / pixel)] = 10

    wc_path = tmp_path / "worldcover.tif"
    with rasterio.open(
        wc_path, "w", driver="GTiff", height=wc_height, width=wc_width, count=1, dtype="uint8",
        crs="EPSG:4326", transform=from_origin(wc_west, wc_north, pixel, pixel),
    ) as dst:
        dst.write(classes, 1)

    result = reproject_worldcover(wc_path, CRS.from_epsg(32720), transform, width, height)

    assert result.shape == (height, width)
    assert int((result == 0).sum()) == 0
    assert (result[:, :10] == 10).all()
    assert (result[:, -10:] == 30).all()