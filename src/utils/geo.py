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
    """Convert a WGS84 bounding box into a pixel window for an open raster.

    Args:
        src: An open rasterio dataset.
        bbox_wgs84: [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.

    Returns:
        An integer-valued Window in the raster's own pixel grid.

    Raises:
        ValueError: If the window is empty or extends outside the raster.
    """
    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
    float_window = from_bounds(left, bottom, right, top, transform=src.transform)
    window = Window(
        col_off=int(round(float_window.col_off)),
        row_off=int(round(float_window.row_off)),
        width=int(round(float_window.width)),
        height=int(round(float_window.height)),
    )
    if window.width <= 0 or window.height <= 0:
        raise ValueError(f"AOI window is empty: {window}")
    outside = (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > src.width
        or window.row_off + window.height > src.height
    )
    if outside:
        raise ValueError(
            f"AOI window {window} extends outside the raster ({src.width} x {src.height} px)"
        )
    return window


def snap_window_to_even(window: Window, max_width: int, max_height: int) -> Window:
    """Expand a window outward so its offsets and sizes are all even numbers of pixels.

    A 10 m window with even offsets and sizes maps exactly onto a 20 m grid
    (each 20 m pixel covers 2 x 2 ten-metre pixels), so bands of both resolutions
    can be clipped to the same ground area.

    Args:
        window: Integer-valued window on the finer grid.
        max_width: Width of the raster in pixels (upper bound for the window).
        max_height: Height of the raster in pixels.

    Returns:
        A window containing the original one, with even offsets and sizes.

    Raises:
        ValueError: If the snapped window would extend outside the raster.
    """
    col_start = (int(window.col_off) // 2) * 2
    row_start = (int(window.row_off) // 2) * 2
    col_end = -(-(int(window.col_off) + int(window.width)) // 2) * 2  # ceil to even
    row_end = -(-(int(window.row_off) + int(window.height)) // 2) * 2

    if col_start < 0 or row_start < 0 or col_end > max_width or row_end > max_height:
        raise ValueError(
            f"Snapped window ({col_start}:{col_end}, {row_start}:{row_end}) "
            f"does not fit inside the raster ({max_width} x {max_height} px)"
        )
    return Window(col_start, row_start, col_end - col_start, row_end - row_start)


def read_band_window(href: str, bbox_wgs84: Sequence[float], band: int = 1) -> tuple[np.ndarray, str]:
    """Read one band of a raster, restricted to the AOI window.

    Args:
        href: URL or path of the raster.
        bbox_wgs84: [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
        band: 1-based band index.

    Returns:
        The data array (rows, cols) and the raster CRS as a string.
    """
    with rasterio.open(href) as src:
        window = bbox_to_window(src, bbox_wgs84)
        data = src.read(band, window=window)
        return data, str(src.crs)