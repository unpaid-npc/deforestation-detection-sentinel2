"""Shared geospatial helpers."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds, intersection

logger = logging.getLogger(__name__)


def bbox_to_window(src: DatasetReader, bbox_wgs84: Sequence[float]) -> Window:
    """Convert a WGS84 bounding box into a pixel window for an open raster.

    Args:
        src: An open rasterio dataset.
        bbox_wgs84: [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.

    Returns:
        An integer-valued Window safely intersected with the raster's grid.

    Raises:
        ValueError: If the window is empty or strictly outside the raster.
    """
    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
    float_window = from_bounds(left, bottom, right, top, transform=src.transform)
    
    # Safely round to integer pixel bounds to prevent off-by-one errors
    target_window = float_window.round_offsets().round_lengths()
    
    # Intersect with the raster bounds
    raster_extent = Window(0, 0, src.width, src.height)
    safe_window = intersection(target_window, raster_extent)
    
    if safe_window.width <= 0 or safe_window.height <= 0:
        raise ValueError(
            f"AOI window {target_window} does not intersect raster ({src.width} x {src.height} px)"
        )
        
    return safe_window


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