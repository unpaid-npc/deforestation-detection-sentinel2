"""Tests for src/utils/geo.py using small synthetic rasters (no network needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from src.utils.geo import bbox_to_window, read_band_window, snap_window_to_even

CRS_UTM = "EPSG:32720"
ORIGIN_X, ORIGIN_Y, PIXEL, SIZE = 600_000.0, 8_880_000.0, 10.0, 1000

# Target area inside the raster, in UTM metres.
LEFT, BOTTOM, RIGHT, TOP = 601_000.0, 8_872_000.0, 603_000.0, 8_874_000.0
EXPECTED_COL_OFF = int((LEFT - ORIGIN_X) / PIXEL)   # 100
EXPECTED_ROW_OFF = int((ORIGIN_Y - TOP) / PIXEL)    # 600
EXPECTED_SIZE = int((RIGHT - LEFT) / PIXEL)         # 200
# Converting a rectangle between UTM and lat/lon makes its bounding box grow a little,
# so positions can be off by a pixel or two.
TOLERANCE_PX = 3


@pytest.fixture()
def synthetic_raster(tmp_path: Path) -> Path:
    """A 1000 x 1000 raster whose pixel value equals its column index."""
    path = tmp_path / "synthetic.tif"
    data = np.tile(np.arange(SIZE, dtype=np.uint16), (SIZE, 1))
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1, dtype="uint16",
        crs=CRS_UTM, transform=from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture()
def aoi_bbox_wgs84() -> list[float]:
    return list(transform_bounds(CRS_UTM, "EPSG:4326", LEFT, BOTTOM, RIGHT, TOP))


def test_bbox_to_window_position_and_size(synthetic_raster: Path, aoi_bbox_wgs84: list[float]) -> None:
    with rasterio.open(synthetic_raster) as src:
        window = bbox_to_window(src, aoi_bbox_wgs84)
    assert abs(window.col_off - EXPECTED_COL_OFF) <= TOLERANCE_PX
    assert abs(window.row_off - EXPECTED_ROW_OFF) <= TOLERANCE_PX
    assert abs(window.width - EXPECTED_SIZE) <= TOLERANCE_PX
    assert abs(window.height - EXPECTED_SIZE) <= TOLERANCE_PX


def test_bbox_to_window_outside_raster_raises(synthetic_raster: Path) -> None:
    far_away_bbox = [-50.0, -10.3, -49.8, -10.12]
    with rasterio.open(synthetic_raster) as src:
        with pytest.raises(ValueError):
            bbox_to_window(src, far_away_bbox)


def test_read_band_window_reads_the_right_columns(synthetic_raster: Path, aoi_bbox_wgs84: list[float]) -> None:
    data, crs = read_band_window(str(synthetic_raster), aoi_bbox_wgs84)
    assert crs == CRS_UTM
    assert data.ndim == 2
    # The pixel value equals the column index, so the first column should be near 100.
    assert abs(int(data[0, 0]) - EXPECTED_COL_OFF) <= TOLERANCE_PX


def test_snap_window_to_even_expands_and_contains_original() -> None:
    original = Window(col_off=3, row_off=5, width=11, height=7)
    snapped = snap_window_to_even(original, max_width=100, max_height=100)
    assert snapped.col_off % 2 == 0 and snapped.row_off % 2 == 0
    assert snapped.width % 2 == 0 and snapped.height % 2 == 0
    assert snapped.col_off <= original.col_off
    assert snapped.row_off <= original.row_off
    assert snapped.col_off + snapped.width >= original.col_off + original.width
    assert snapped.row_off + snapped.height >= original.row_off + original.height


def test_snap_window_to_even_outside_raster_raises() -> None:
    with pytest.raises(ValueError):
        snap_window_to_even(Window(0, 0, 11, 10), max_width=11, max_height=10)