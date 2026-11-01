"""Quick-look visual check of the AOI.

Produces one figure with three panels:
  * true-colour image for Date A,
  * true-colour image for Date B,
  * ESA WorldCover 2021 forest / non-forest.

Purpose: confirm by eye that the AOI contains forest AND visible clearing, that the
two dates look radiometrically consistent, and that no smoke or haze is present.

The WorldCover panel is resized to the Sentinel-2 panel shape with nearest-neighbour
interpolation FOR DISPLAY ONLY. Proper reprojection onto the Sentinel-2 grid happens
in the preprocessing module.

Run from the project root:
    python -m src.data.quicklook
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
import planetary_computer
import pystac
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pystac_client import Client
from shapely.geometry import box, shape

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.geo import read_band_window

logger = logging.getLogger("quicklook")

RGB_BANDS: tuple[str, ...] = ("B04", "B03", "B02")  # red, green, blue
DISPLAY_STEP = 4          # keep every 4th pixel for the picture (10 m -> 40 m)
REFLECTANCE_DISPLAY_MAX = 0.25  # fixed stretch, identical for both dates
OUTPUT_PATH: Path = PROJECT_ROOT / "outputs" / "maps" / "quicklook_aoi.png"


def fetch_item(catalog: Client, collection: str, item_id: str) -> pystac.Item:
    """Fetch one signed STAC item by its exact id directly via OGC API."""
    col = catalog.get_collection(collection)
    if col is None:
        raise LookupError(f"STAC collection not found: {collection}")
        
    item = col.get_item(item_id)
    if item is None:
        raise LookupError(f"STAC item not found: {item_id}")
    return item


def reflectance_scale_offset(item: pystac.Item, asset_key: str) -> tuple[float, float, str]:
    """Return (scale, offset, source) converting digital numbers to surface reflectance.

    reflectance = DN * scale + offset

    Scenes with processing baseline >= 04.00 (from January 2022) carry an offset of
    -1000 DN (-0.1 reflectance); older scenes do not. We prefer the asset metadata and
    fall back to the processing baseline.
    """
    raster_bands = item.assets[asset_key].extra_fields.get("raster:bands")
    if raster_bands and "scale" in raster_bands[0] and "offset" in raster_bands[0]:
        band = raster_bands[0]
        return float(band["scale"]), float(band["offset"]), "asset raster:bands metadata"

    baseline = item.properties.get("s2:processing_baseline")
    offset = -0.1 if baseline is not None and float(baseline) >= 4.0 else 0.0
    return 1e-4, offset, f"processing baseline {baseline}"


def read_rgb_reflectance(item: pystac.Item, bbox: Sequence[float]) -> np.ndarray:
    """Read the AOI window of B04/B03/B02 as float32 reflectance, shape (rows, cols, 3).

    DN value 0 means "no data" and becomes NaN.
    """
    channels: list[np.ndarray] = []
    for key in RGB_BANDS:
        dn, _ = read_band_window(item.assets[key].href, bbox)
        scale, offset, source = reflectance_scale_offset(item, key)
        if key == RGB_BANDS[0]:
            logger.info("  reflectance = DN * %g + (%g)   [source: %s]", scale, offset, source)
        reflectance = dn.astype(np.float32) * scale + offset
        reflectance[dn == 0] = np.nan
        channels.append(reflectance)

    if len({channel.shape for channel in channels}) != 1:
        raise ValueError(f"RGB bands have different shapes: {[c.shape for c in channels]}")
    return np.stack(channels, axis=-1)


def to_display(rgb: np.ndarray) -> np.ndarray:
    """Fixed linear stretch (0 to REFLECTANCE_DISPLAY_MAX) with light gamma, for display."""
    scaled = np.clip(rgb / REFLECTANCE_DISPLAY_MAX, 0.0, 1.0)
    return np.nan_to_num(scaled ** (1 / 1.4))


def read_worldcover_forest(
    catalog: Client, collection: str, bbox: Sequence[float], id_contains: str, forest_class: int
) -> tuple[np.ndarray, float]:
    """Read WorldCover for the AOI and return a forest mask plus the forest fraction."""
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
    item = matching[0]
    logger.info("WorldCover item: %s", item.id)
    classes, crs = read_band_window(item.assets["map"].href, bbox)
    logger.info("  WorldCover window shape %s, CRS %s", classes.shape, crs)

    forest = (classes == forest_class).astype(np.uint8)
    values, counts = np.unique(classes, return_counts=True)
    for value, count in zip(values, counts):
        logger.info("    class %3d: %6.2f%%", value, 100.0 * count / classes.size)
    return forest, float(forest.mean())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description="Create a quick-look figure of the AOI.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        bbox = config["aoi"]["bbox_wgs84"]
        stac_cfg = config["stac"]
        scene_ids = {
            "Date A": config["scenes"]["date_a"]["item_id"],
            "Date B": config["scenes"]["date_b"]["item_id"],
        }
        wc_cfg = config["worldcover"]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Configuration problem: %s", exc)
        return 1

    try:
        catalog = Client.open(stac_cfg["url"], modifier=planetary_computer.sign_inplace)

        panels: dict[str, np.ndarray] = {}
        titles: dict[str, str] = {}
        for label, item_id in scene_ids.items():
            logger.info("%s: %s", label, item_id)
            item = fetch_item(catalog, stac_cfg["sentinel2_collection"], item_id)
            logger.info(
                "  date %s | scene cloud %.1f%% | processing baseline %s",
                item.datetime.date(),
                item.properties.get("eo:cloud_cover", float("nan")),
                item.properties.get("s2:processing_baseline"),
            )
            rgb = read_rgb_reflectance(item, bbox)
            logger.info(
                "  RGB shape %s | reflectance min %.3f, median %.3f, max %.3f",
                rgb.shape, np.nanmin(rgb), np.nanmedian(rgb), np.nanmax(rgb),
            )
            panels[label] = to_display(rgb[::DISPLAY_STEP, ::DISPLAY_STEP])
            titles[label] = f"{label}: {item.datetime.date()}"

        forest, forest_fraction = read_worldcover_forest(
            catalog,
            stac_cfg["worldcover_collection"],
            bbox,
            wc_cfg["item_id_contains"],
            wc_cfg["forest_class"],
        )
    except Exception:
        logger.exception("Quick-look failed")
        return 1

    height, width = panels["Date A"].shape[:2]
    forest_display = cv2.resize(forest, (width, height), interpolation=cv2.INTER_NEAREST)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, label in zip(axes[:2], panels):
        ax.imshow(panels[label])
        ax.set_title(titles[label])
    forest_cmap = ListedColormap(["#d9c9a3", "#1b7837"])
    axes[2].imshow(forest_display, cmap=forest_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"WorldCover 2021 (forest = {100 * forest_fraction:.1f}% of AOI)")
    axes[2].legend(
        handles=[Patch(color="#1b7837", label="Tree cover"), Patch(color="#d9c9a3", label="Other")],
        loc="lower left",
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Quick-look: {config['aoi']['name']}  |  bbox {bbox}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())