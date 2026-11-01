"""Shared geospatial helpers."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

logger = logging.getLogger(__name__)


def bbox_to_window(src: DatasetReader, bbox_wgs84: Sequence[float]) -> Window:
    """Convert a WGS84 bounding box into a pixel window for an open raster."""
    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
    float_window = from_bounds(left, bottom, right, top, transform=src.transform)
    # Use native round_offsets().round_lengths() to avoid off-by-one errors
    window = float_window.round_offsets().round_lengths()
    
    if window.width <= 0 or window.height <= 0:
        raise ValueError(f"AOI window is empty: {window}")
    # Boundary ValueError removed to permit boundless reading
    return window


def snap_window_to_even(window: Window, max_width: int, max_height: int) -> Window:
    """Expand a window outward so its offsets and sizes are all even numbers of pixels."""
    col_start = (int(window.col_off) // 2) * 2
    row_start = (int(window.row_off) // 2) * 2
    col_end = -(-(int(window.col_off) + int(window.width)) // 2) * 2  # ceil to even
    row_end = -(-(int(window.row_off) + int(window.height)) // 2) * 2
    # Boundary ValueError removed to permit boundless reading
    return Window(col_start, row_start, col_end - col_start, row_end - row_start)


def read_band_window(href: str, bbox_wgs84: Sequence[float], band: int = 1) -> tuple[np.ndarray, str]:
    """Read one band of a raster, restricted to the AOI window with safe edge padding."""
    with rasterio.open(href) as src:
        window = bbox_to_window(src, bbox_wgs84)
        fill_val = src.nodata if src.nodata is not None else 0
        data = src.read(band, window=window, boundless=True, fill_value=fill_val)
        return data, str(src.crs)