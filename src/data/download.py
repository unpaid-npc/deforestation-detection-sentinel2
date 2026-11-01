"""Download the AOI subset of the configured Sentinel-2 scenes and ESA WorldCover.

For each date in ``config['scenes']`` this saves B02, B03, B04, B08 (10 m) and
B11, B12, SCL (20 m), each clipped to the AOI, as GeoTIFFs under ``data/raw/<date_key>/``.
It also saves the WorldCover map (in its native EPSG:4326 grid, with a small margin)
and a ``manifest.json`` describing exactly what was downloaded.

No resampling, masking or radiometric conversion happens here; that is the job of
the preprocessing module.

Run from the project root:
    python -m src.data.download            # skips files that already exist
    python -m src.data.download --force    # downloads everything again
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import planetary_computer
import pystac
import rasterio
from pystac_client import Client
from rasterio.crs import CRS
from rasterio.windows import Window
from shapely.geometry import box, shape

from src.data.stac_probe import validate_bbox
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.geo import bbox_to_window, snap_window_to_even

logger = logging.getLogger("download")

BANDS_10M: tuple[str, ...] = ("B02", "B03", "B04", "B08")
BANDS_20M: tuple[str, ...] = ("B11", "B12", "SCL")
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH: Path = RAW_DIR / "manifest.json"
WORLDCOVER_MARGIN_DEG = 0.005  # extra border so later reprojection has data past the AOI edge


def fetch_item(catalog: Client, collection: str, item_id: str) -> pystac.Item:
    """Fetch one signed STAC item by its exact id."""
    items = list(catalog.search(collections=[collection], ids=[item_id]).items())
    if not items:
        raise LookupError(f"STAC item not found: {item_id}")
    return items[0]


def sentinel2_windows(item: pystac.Item, bbox: Sequence[float]) -> tuple[Window, Window]:
    """Compute matching 10 m and 20 m windows for the AOI.

    The window is found on the 10 m B04 grid and snapped to even pixel numbers, so the
    20 m window (exactly half the size) covers precisely the same ground.
    """
    with rasterio.open(item.assets["B04"].href) as src:
        raw_window = bbox_to_window(src, bbox)
        window_10m = snap_window_to_even(raw_window, src.width, src.height)
    window_20m = Window(
        window_10m.col_off // 2,
        window_10m.row_off // 2,
        window_10m.width // 2,
        window_10m.height // 2,
    )
    return window_10m, window_20m


def clip_to_file(href: str, window: Window, out_path: Path) -> None:
    """Read ``window`` from a remote or local raster and write it as a GeoTIFF.

    The file is written under a temporary name first, so an interrupted download never
    leaves a half-written file that looks valid.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".part")

    with rasterio.open(href) as src:
        outside = (
            window.col_off < 0
            or window.row_off < 0
            or window.col_off + window.width > src.width
            or window.row_off + window.height > src.height
        )
        if outside:
            raise ValueError(f"Window {window} does not fit in raster {src.width} x {src.height}: {href}")
        data = src.read(window=window)
        profile = {
            "driver": "GTiff",
            "dtype": src.dtypes[0],
            "count": src.count,
            "width": int(window.width),
            "height": int(window.height),
            "crs": src.crs,
            "transform": src.window_transform(window),
            "nodata": src.nodata,
            "compress": "deflate",
        }

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
    tmp_path.replace(out_path)


def verify_raster(path: Path, expected_crs: str, window: Window) -> dict[str, Any]:
    """Re-open a written raster, check it, and return its metadata.

    Raises:
        ValueError: If the CRS or the shape is not what was requested, or if the
            raster contains no valid (non-zero) pixels at all.
    """
    with rasterio.open(path) as src:
        if src.crs != CRS.from_user_input(expected_crs):
            raise ValueError(f"{path.name}: CRS is {src.crs}, expected {expected_crs}")
        if (src.width, src.height) != (int(window.width), int(window.height)):
            raise ValueError(
                f"{path.name}: shape {src.width} x {src.height} does not match "
                f"requested window {int(window.width)} x {int(window.height)}"
            )
        data = src.read(1)
        zero_fraction = float((data == 0).mean())
        if zero_fraction >= 1.0:
            raise ValueError(f"{path.name}: raster contains only zeros")
        return {
            "file": path.relative_to(PROJECT_ROOT).as_posix(),
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "transform": [float(v) for v in tuple(src.transform)[:6]],
            "zero_fraction": round(zero_fraction, 6),
        }


def download_scene(
    item: pystac.Item, bbox: Sequence[float], out_dir: Path, expected_crs: str, force: bool
) -> dict[str, Any]:
    """Download and verify all bands of one Sentinel-2 scene. Returns its manifest entry."""
    window_10m, window_20m = sentinel2_windows(item, bbox)
    logger.info("  10 m window: %s", window_10m)
    logger.info("  20 m window: %s", window_20m)

    files: dict[str, Any] = {}
    for band in BANDS_10M + BANDS_20M:
        window = window_10m if band in BANDS_10M else window_20m
        out_path = out_dir / f"{band}.tif"
        if out_path.exists() and not force:
            logger.info("  %-3s already exists, verifying only", band)
        else:
            logger.info("  %-3s downloading ...", band)
            clip_to_file(item.assets[band].href, window, out_path)
        meta = verify_raster(out_path, expected_crs, window)
        logger.info(
            "  %-3s OK  %d x %d  %s  zero pixels %.3f%%  %.1f MB",
            band, meta["width"], meta["height"], meta["dtype"],
            100 * meta["zero_fraction"], out_path.stat().st_size / 1e6,
        )
        files[band] = meta

    return {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "processing_baseline": item.properties.get("s2:processing_baseline"),
        "scene_cloud_cover": item.properties.get("eo:cloud_cover"),
        "window_10m": [int(window_10m.col_off), int(window_10m.row_off),
                       int(window_10m.width), int(window_10m.height)],
        "window_20m": [int(window_20m.col_off), int(window_20m.row_off),
                       int(window_20m.width), int(window_20m.height)],
        "files": files,
    }


def select_worldcover_item(
    catalog: Client, collection: str, bbox: Sequence[float], id_contains: str
) -> pystac.Item:
    """Pick the WorldCover tile containing the whole (expanded) AOI and matching the product."""
    aoi_geom = box(*bbox)
    candidates = list(catalog.search(collections=[collection], bbox=list(bbox)).items())
    matching = [
        item for item in candidates
        if id_contains in item.id and shape(item.geometry).contains(aoi_geom)
    ]
    if not matching:
        raise LookupError(
            f"No WorldCover tile containing the AOI with '{id_contains}' in its id. "
            f"Candidates: {[item.id for item in candidates]}"
        )
    return matching[0]


def download_worldcover(
    catalog: Client, config: dict[str, Any], bbox: Sequence[float], force: bool
) -> dict[str, Any]:
    """Download the WorldCover map for the AOI plus a small margin (native EPSG:4326 grid)."""
    margin = WORLDCOVER_MARGIN_DEG
    expanded = [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin]
    item = select_worldcover_item(
        catalog, config["stac"]["worldcover_collection"], expanded, config["worldcover"]["item_id_contains"]
    )
    logger.info("WorldCover item: %s", item.id)
    href = item.assets["map"].href
    out_path = RAW_DIR / "worldcover.tif"

    with rasterio.open(href) as src:
        window = bbox_to_window(src, expanded)
    logger.info("  window: %s", window)

    if out_path.exists() and not force:
        logger.info("  worldcover.tif already exists, verifying only")
    else:
        logger.info("  downloading ...")
        clip_to_file(href, window, out_path)
    meta = verify_raster(out_path, "EPSG:4326", window)
    logger.info("  OK  %d x %d  %s  %.1f MB", meta["width"], meta["height"], meta["dtype"],
                out_path.stat().st_size / 1e6)
    return {
        "item_id": item.id,
        "window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
        "margin_deg": margin,
        "file": meta,
    }


def check_dates_aligned(manifest: dict[str, Any]) -> None:
    """Raise ValueError unless Date A and Date B share the same 10 m pixel grid."""
    grid_a = manifest["dates"]["date_a"]["files"]["B04"]
    grid_b = manifest["dates"]["date_b"]["files"]["B04"]
    keys = ("crs", "transform", "width", "height")
    for key in keys:
        if grid_a[key] != grid_b[key]:
            raise ValueError(f"Date A and Date B grids differ in '{key}': {grid_a[key]} vs {grid_b[key]}")
    logger.info("Date A and Date B share an identical 10 m grid (%d x %d, %s)",
                grid_a["width"], grid_a["height"], grid_a["crs"])


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description="Download the AOI subset of the configured scenes.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    parser.add_argument("--force", action="store_true", help="Download files again even if they exist.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        bbox = config["aoi"]["bbox_wgs84"]
        validate_bbox(bbox)
        expected_crs = config["aoi"]["expected_crs"]
        stac_cfg = config["stac"]
        scene_ids = {
            "date_a": config["scenes"]["date_a"]["item_id"],
            "date_b": config["scenes"]["date_b"]["item_id"],
        }
        config["worldcover"]["item_id_contains"]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Configuration problem: %s", exc)
        return 1

    try:
        catalog = Client.open(stac_cfg["url"], modifier=planetary_computer.sign_inplace)

        manifest: dict[str, Any] = {
            "aoi": {"name": config["aoi"].get("name"), "bbox_wgs84": list(bbox), "crs": expected_crs},
            "dates": {},
        }
        for key, item_id in scene_ids.items():
            logger.info("%s: %s", key, item_id)
            item = fetch_item(catalog, stac_cfg["sentinel2_collection"], item_id)
            manifest["dates"][key] = download_scene(item, bbox, RAW_DIR / key, expected_crs, args.force)

        check_dates_aligned(manifest)
        manifest["worldcover"] = download_worldcover(catalog, config, bbox, args.force)

        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Wrote %s", MANIFEST_PATH)
    except Exception:
        logger.exception("Download failed")
        return 1

    logger.info("DOWNLOAD COMPLETE")
    return 0

if __name__ == "__main__":
    sys.exit(main())