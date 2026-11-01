"""STAC feasibility probe.

Checks, before we write the real download module, that:
  1. the Planetary Computer STAC API responds,
  2. Sentinel-2 L2A scenes exist for our AOI and date windows,
  3. the required band assets and the SCL are present and actually readable,
  4. ESA WorldCover covers the AOI.

It downloads only a tiny SCL window and writes nothing to disk.

Run from the project root:
    python -m src.data.stac_probe
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import numpy as np
import planetary_computer
import pystac
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds, intersection
from shapely.geometry import box, shape

from src.utils.config import load_config

logger = logging.getLogger("stac_probe")

REQUIRED_ASSETS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12", "SCL")

# Sentinel-2 SCL classes that make a pixel unusable for forest mapping.
SCL_INVALID_CLASSES: dict[int, str] = {
    0: "no data",
    1: "saturated/defective",
    3: "cloud shadow",
    7: "unclassified/low prob cloud",
    8: "cloud (medium probability)",
    9: "cloud (high probability)",
    10: "thin cirrus",
    11: "snow/ice",
}


def validate_bbox(bbox: Sequence[float]) -> None:
    """Raise ValueError unless bbox is a valid [min_lon, min_lat, max_lon, max_lat]."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError(f"Invalid longitude range in bbox: {bbox}")
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError(f"Invalid latitude range in bbox: {bbox}")


def search_sentinel2(
    catalog: Client,
    collection: str,
    bbox: Sequence[float],
    start: str,
    end: str,
    max_cloud_cover: float,
) -> list[pystac.Item]:
    """Search for Sentinel-2 items, sorted by scene-wide cloud cover (lowest first)."""
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100.0))
    return items


def report_scenes(items: list[pystac.Item], bbox: Sequence[float], top_n: int) -> None:
    """Log the best ``top_n`` scenes and whether each fully covers the AOI."""
    aoi_geom = box(*bbox)
    for item in items[:top_n]:
        covers = shape(item.geometry).contains(aoi_geom)
        missing = [name for name in REQUIRED_ASSETS if name not in item.assets]
        crs = item.properties.get("proj:code") or item.properties.get("proj:epsg")
        logger.info(
            "  %s | cloud %.1f%% | tile %s | CRS %s | covers AOI: %s | missing assets: %s",
            item.datetime.date() if item.datetime else "n/a",
            item.properties.get("eo:cloud_cover", float("nan")),
            item.properties.get("s2:mgrs_tile", "n/a"),
            crs,
            covers,
            missing or "none",
        )


def pick_best_item(items: list[pystac.Item], bbox: Sequence[float]) -> pystac.Item | None:
    """Return the lowest-cloud item that fully covers the AOI and has all assets."""
    aoi_geom = box(*bbox)
    for item in items:
        has_assets = all(name in item.assets for name in REQUIRED_ASSETS)
        if has_assets and shape(item.geometry).contains(aoi_geom):
            return item
    return None


def read_scl_window(href: str, bbox: Sequence[float]) -> tuple[np.ndarray, str]:
    """Read only the AOI window of an SCL raster safely bounded by image extents.

    Returns:
        The SCL array (rows, cols) and the raster's CRS as a string.
    """
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        float_window = from_bounds(left, bottom, right, top, transform=src.transform)
        
        # Snap to integer pixels
        target_window = float_window.round_offsets().round_lengths()
        
        # Safely intersect with the actual raster bounds to avoid out-of-bounds errors
        raster_extent = Window(0, 0, src.width, src.height)
        safe_window = intersection(target_window, raster_extent)
        
        if safe_window.width <= 0 or safe_window.height <= 0:
            raise ValueError(f"AOI window does not intersect this raster: {target_window}")
            
        data = src.read(1, window=safe_window)
        return data, str(src.crs)


def report_scl_quality(data: np.ndarray) -> None:
    """Log how much of the AOI is unusable according to the SCL."""
    total = data.size
    logger.info("  SCL window shape (rows, cols): %s (%d pixels at 20 m)", data.shape, total)
    invalid_total = 0.0
    for class_id, name in SCL_INVALID_CLASSES.items():
        pct = 100.0 * float(np.count_nonzero(data == class_id)) / total
        invalid_total += pct
        if pct > 0:
            logger.info("    class %2d %-28s %6.2f%%", class_id, name, pct)
    logger.info("  => AOI pixels unusable by SCL: %.2f%%", invalid_total)


def probe_worldcover(catalog: Client, collection: str, bbox: Sequence[float], dt: str) -> bool:
    """Check that WorldCover items intersect the AOI. Returns True if any were found."""
    search = catalog.search(collections=[collection], bbox=list(bbox), datetime=dt)
    items = list(search.items())
    logger.info("WorldCover items found: %d", len(items))
    for item in items:
        logger.info(
            "  %s | datetime %s | version %s | assets %s",
            item.id,
            item.datetime,
            item.properties.get("esa_worldcover:product_version", "n/a"),
            sorted(item.assets.keys()),
        )
    return len(items) > 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = all checks passed)."""
    parser = argparse.ArgumentParser(description="Probe STAC availability for the AOI.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        bbox = config["aoi"]["bbox_wgs84"]
        validate_bbox(bbox)
        stac_cfg = config["stac"]
        search_cfg = config["search"]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Configuration problem: %s", exc)
        return 1

    logger.info("AOI %s bbox (lon/lat): %s", config["aoi"].get("name", "unnamed"), bbox)

    try:
        catalog = Client.open(stac_cfg["url"], modifier=planetary_computer.sign_inplace)
    except Exception:  # network and API errors vary widely
        logger.exception("Could not open STAC catalog at %s", stac_cfg["url"])
        return 1

    exit_code = 0
    for window in search_cfg["windows"]:
        label, start, end = window["label"], window["start"], window["end"]
        logger.info("--- Window %s (%s to %s) ---", label, start, end)
        try:
            items = search_sentinel2(
                catalog,
                stac_cfg["sentinel2_collection"],
                bbox,
                start,
                end,
                search_cfg["max_cloud_cover"],
            )
        except Exception:
            logger.exception("Sentinel-2 search failed for window %s", label)
            exit_code = 1
            continue

        logger.info("Scenes found (scene cloud < %s%%): %d", search_cfg["max_cloud_cover"], len(items))
        if not items:
            logger.error("No scenes found for window %s", label)
            exit_code = 1
            continue

        report_scenes(items, bbox, search_cfg["top_n_report"])

        best = pick_best_item(items, bbox)
        if best is None:
            logger.warning("No single scene fully covers the AOI with all assets in %s", label)
            exit_code = 1
            continue

        logger.info("Reading SCL window from best scene: %s", best.id)
        try:
            data, crs = read_scl_window(best.assets["SCL"].href, bbox)
        except Exception:
            logger.exception("Could not read SCL from %s", best.id)
            exit_code = 1
            continue
        logger.info("  Raster CRS: %s", crs)
        report_scl_quality(data)

    logger.info("--- ESA WorldCover ---")
    try:
        if not probe_worldcover(
            catalog,
            stac_cfg["worldcover_collection"],
            bbox,
            search_cfg["worldcover_datetime"],
        ):
            logger.error("No WorldCover coverage found for the AOI")
            exit_code = 1
    except Exception:
        logger.exception("WorldCover search failed")
        exit_code = 1

    logger.info("PROBE %s", "PASSED" if exit_code == 0 else "FINISHED WITH PROBLEMS")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())