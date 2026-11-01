"""Preprocessing: build analysis-ready data (ARD) from the raw downloads.

Reads ``data/raw/`` (written by ``src.data.download``) and writes, for each date:

    data/processed/<date_key>/reflectance.tif   6 bands, float32 surface reflectance,
                                                invalid pixels = NaN
    data/processed/<date_key>/valid_mask.tif    uint8, 1 = usable pixel, 0 = masked
    data/processed/<date_key>/scl_10m.tif       uint8, Scene Classification Layer at 10 m

and, shared by both dates:

    data/processed/worldcover_10m.tif           uint8 WorldCover class codes on the same grid
    data/processed/preprocess_report.json       statistics for documentation and checking

Steps: 20 m bands and the SCL are upsampled to 10 m (nearest neighbour, exact 2 x 2 blocks);
digital numbers are converted to reflectance with the baseline-dependent offset; pixels with
an invalid SCL class or a zero digital number become NaN; WorldCover is reprojected from
EPSG:4326 to the Sentinel-2 UTM grid (nearest neighbour).

Band order of reflectance.tif: B02 (blue), B03 (green), B04 (red), B08 (NIR),
B11 (SWIR1), B12 (SWIR2).

Normalisation (mean/std) is intentionally NOT done here: it must use training blocks only
and therefore belongs to the dataset module.

Run from the project root:
    python -m src.data.preprocess
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger("preprocess")

BANDS_10M: tuple[str, ...] = ("B02", "B03", "B04", "B08")
BANDS_20M: tuple[str, ...] = ("B11", "B12")
BAND_ORDER: tuple[str, ...] = BANDS_10M + BANDS_20M  # channel order of reflectance.tif
SCL_BAND = "SCL"

RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH: Path = RAW_DIR / "manifest.json"
REPORT_PATH: Path = PROCESSED_DIR / "preprocess_report.json"

REFLECTANCE_SCALE = 1.0e-4          # Sentinel-2 L2A: reflectance = DN / 10000 (+ offset)
BASELINE_OFFSET_THRESHOLD = 4.0     # processing baseline >= 04.00 carries an offset
BASELINE_OFFSET = -0.1              # -1000 DN expressed in reflectance
VALID_WORLDCOVER_CLASSES: frozenset[int] = frozenset({0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100})


# --------------------------------------------------------------------------------------
# Radiometry
# --------------------------------------------------------------------------------------
def reflectance_offset(processing_baseline: str | float | None) -> float:
    """Return the additive reflectance offset for a Sentinel-2 processing baseline.

    Scenes processed with baseline 04.00 or later (from January 2022) store digital
    numbers shifted by +1000, so 0.1 must be subtracted after scaling.

    Raises:
        ValueError: If the baseline is missing or cannot be read as a number.
    """
    if processing_baseline is None:
        raise ValueError("Processing baseline is missing; cannot choose the radiometric offset.")
    try:
        baseline = float(processing_baseline)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unreadable processing baseline: {processing_baseline!r}") from exc
    return BASELINE_OFFSET if baseline >= BASELINE_OFFSET_THRESHOLD else 0.0


def dn_to_reflectance(dn: np.ndarray, offset: float, clip_range: tuple[float, float]) -> np.ndarray:
    """Convert digital numbers to float32 surface reflectance and clip to ``clip_range``."""
    low, high = clip_range
    reflectance = dn.astype(np.float32) * np.float32(REFLECTANCE_SCALE) + np.float32(offset)
    np.clip(reflectance, low, high, out=reflectance)
    return reflectance


def upsample_nearest_2x(array: np.ndarray) -> np.ndarray:
    """Upsample the last two axes by 2 with nearest neighbour (each pixel becomes 2 x 2)."""
    return np.repeat(np.repeat(array, 2, axis=-2), 2, axis=-1)


def build_valid_mask(scl: np.ndarray, dn_stack: np.ndarray, invalid_classes: Sequence[int]) -> np.ndarray:
    """Return a boolean (rows, cols) mask that is True where a pixel is usable.

    A pixel is usable if its SCL class is not in ``invalid_classes`` and every band has a
    non-zero digital number (zero means no data).
    """
    if scl.shape != dn_stack.shape[1:]:
        raise ValueError(f"SCL shape {scl.shape} does not match image shape {dn_stack.shape[1:]}")
    scl_ok = ~np.isin(scl, list(invalid_classes))
    has_data = (dn_stack > 0).all(axis=0)
    return scl_ok & has_data


# --------------------------------------------------------------------------------------
# Grid checks
# --------------------------------------------------------------------------------------
def read_first_band(path: Path) -> np.ndarray:
    """Read band 1 of a raster."""
    with rasterio.open(path) as src:
        return src.read(1)


def read_reference_grid(path: Path) -> dict[str, Any]:
    """Return CRS, transform and shape of a raster as a dictionary."""
    with rasterio.open(path) as src:
        return {"crs": src.crs, "transform": src.transform, "width": src.width, "height": src.height}


def grids_match(grid_a: dict[str, Any], grid_b: dict[str, Any]) -> bool:
    """True if two grids have the same CRS, size and affine transform."""
    return bool(
        grid_a["crs"] == grid_b["crs"]
        and grid_a["width"] == grid_b["width"]
        and grid_a["height"] == grid_b["height"]
        and np.allclose(tuple(grid_a["transform"])[:6], tuple(grid_b["transform"])[:6], rtol=0.0, atol=1e-6)
    )


def grid_summary(grid: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly description of a grid."""
    return {
        "crs": str(grid["crs"]),
        "transform": [float(v) for v in tuple(grid["transform"])[:6]],
        "width": int(grid["width"]),
        "height": int(grid["height"]),
    }


def grid_profile(grid: dict[str, Any]) -> dict[str, Any]:
    """Base rasterio write profile for rasters on ``grid``."""
    return {
        "driver": "GTiff",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "width": grid["width"],
        "height": grid["height"],
        "compress": "deflate",
    }


def check_same_grid(reference: Path, other: Path) -> None:
    """Raise ValueError unless ``other`` has exactly the same grid as ``reference``."""
    if not grids_match(read_reference_grid(reference), read_reference_grid(other)):
        raise ValueError(f"{other.name} is not on the same grid as {reference.name}")


def check_half_resolution_grid(reference: Path, other: Path) -> None:
    """Raise ValueError unless ``other`` is a 20 m grid that nests exactly in ``reference`` (10 m).

    This guarantees that every 20 m pixel covers exactly 2 x 2 reference pixels.
    """
    with rasterio.open(reference) as ref, rasterio.open(other) as oth:
        if ref.crs != oth.crs:
            raise ValueError(f"{other.name}: CRS {oth.crs} differs from {ref.crs}")
        if (oth.width * 2, oth.height * 2) != (ref.width, ref.height):
            raise ValueError(
                f"{other.name}: size {oth.width} x {oth.height} is not half of "
                f"{ref.width} x {ref.height}"
            )
        t = ref.transform
        expected = (t.a * 2, t.b, t.c, t.d, t.e * 2, t.f)
        if not np.allclose(expected, tuple(oth.transform)[:6], rtol=0.0, atol=1e-6):
            raise ValueError(f"{other.name}: transform {tuple(oth.transform)[:6]} does not nest in {expected}")


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------
def write_raster(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    descriptions: Sequence[str] | None = None,
) -> None:
    """Write a 2D or 3D array as a GeoTIFF (via a temporary file, then rename)."""
    data = array[np.newaxis] if array.ndim == 2 else array
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".part")
    out_profile = {**profile, "count": int(data.shape[0]), "dtype": data.dtype.name}
    with rasterio.open(tmp_path, "w", **out_profile) as dst:
        dst.write(data)
        if descriptions:
            for index, name in enumerate(descriptions, start=1):
                dst.set_band_description(index, name)
    tmp_path.replace(path)


# --------------------------------------------------------------------------------------
# Per-date processing
# --------------------------------------------------------------------------------------
def process_date(
    date_key: str,
    raw_date_dir: Path,
    out_dir: Path,
    processing_baseline: str | float | None,
    invalid_scl_classes: Sequence[int],
    clip_range: tuple[float, float],
) -> tuple[dict[str, Any], np.ndarray]:
    """Create the analysis-ready files for one date.

    Returns:
        A statistics dictionary (for the report) and the boolean valid mask.

    Raises:
        FileNotFoundError: If a raw band is missing.
        ValueError: If grids are inconsistent or no valid pixel remains.
    """
    offset = reflectance_offset(processing_baseline)
    paths = {band: raw_date_dir / f"{band}.tif" for band in BAND_ORDER + (SCL_BAND,)}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw files for {date_key}: {missing}")

    reference = paths["B04"]
    for band in BANDS_10M:
        check_same_grid(reference, paths[band])
    for band in BANDS_20M + (SCL_BAND,):
        check_half_resolution_grid(reference, paths[band])
    grid = read_reference_grid(reference)

    layers = [read_first_band(paths[band]) for band in BANDS_10M]
    layers += [upsample_nearest_2x(read_first_band(paths[band])) for band in BANDS_20M]
    dn_stack = np.stack(layers, axis=0)  # (6, rows, cols), uint16
    scl_10m = upsample_nearest_2x(read_first_band(paths[SCL_BAND]))  # (rows, cols), uint8

    valid = build_valid_mask(scl_10m, dn_stack, invalid_scl_classes)
    if not valid.any():
        raise ValueError(f"{date_key}: no valid pixels remain after masking")

    reflectance = dn_to_reflectance(dn_stack, offset, clip_range)
    reflectance[:, ~valid] = np.nan

    profile = grid_profile(grid)
    write_raster(out_dir / "reflectance.tif", reflectance, {**profile, "nodata": float("nan")}, BAND_ORDER)
    write_raster(out_dir / "valid_mask.tif", valid.astype(np.uint8), profile)
    write_raster(out_dir / "scl_10m.tif", scl_10m, profile)

    classes, counts = np.unique(scl_10m, return_counts=True)
    scl_percent = {int(c): round(100.0 * int(n) / scl_10m.size, 4) for c, n in zip(classes, counts)}
    band_stats = {}
    for index, band in enumerate(BAND_ORDER):
        values = reflectance[index][valid]
        band_stats[band] = {
            "min": round(float(values.min()), 4),
            "median": round(float(np.median(values)), 4),
            "max": round(float(values.max()), 4),
        }

    stats = {
        "processing_baseline": processing_baseline,
        "reflectance_offset": offset,
        "invalid_scl_classes": [int(c) for c in invalid_scl_classes],
        "scl_percent_by_class": scl_percent,
        "invalid_fraction": round(1.0 - float(valid.mean()), 6),
        "reflectance_valid_pixels": band_stats,
        "output_dir": out_dir.relative_to(PROJECT_ROOT).as_posix() if out_dir.is_relative_to(PROJECT_ROOT) else str(out_dir),
    }
    return stats, valid


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
def reproject_worldcover(
    worldcover_path: Path, dst_crs: CRS, dst_transform: Affine, width: int, height: int
) -> np.ndarray:
    """Reproject the WorldCover class map onto a target grid with nearest neighbour.

    Returns:
        uint8 array (height, width) of WorldCover class codes; 0 means "no label".
    """
    destination = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(worldcover_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
    return destination


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description="Create analysis-ready data from data/raw/.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        pre = config["preprocessing"]
        invalid_classes = [int(c) for c in pre["invalid_scl_classes"]]
        clip_range = (float(pre["reflectance_clip"][0]), float(pre["reflectance_clip"][1]))
        max_label_nodata = float(pre["max_label_nodata_fraction"])
        forest_class = int(config["worldcover"]["forest_class"])
        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError(f"{MANIFEST_PATH} not found. Run 'python -m src.data.download' first.")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        date_entries = {key: manifest["dates"][key] for key in ("date_a", "date_b")}
    except (FileNotFoundError, KeyError, ValueError, IndexError, TypeError) as exc:
        logger.error("Configuration or manifest problem: %s", exc)
        return 1

    try:
        report: dict[str, Any] = {"dates": {}}
        valid_masks: dict[str, np.ndarray] = {}
        for key, entry in date_entries.items():
            logger.info("%s: %s", key, entry["item_id"])
            stats, valid = process_date(
                key, RAW_DIR / key, PROCESSED_DIR / key,
                entry.get("processing_baseline"), invalid_classes, clip_range,
            )
            stats["item_id"] = entry["item_id"]
            stats["datetime"] = entry.get("datetime")
            report["dates"][key] = stats
            valid_masks[key] = valid
            logger.info(
                "  baseline %s -> offset %+.1f | masked %.3f%% of the AOI",
                stats["processing_baseline"], stats["reflectance_offset"], 100 * stats["invalid_fraction"],
            )
            for class_id in invalid_classes:
                pct = stats["scl_percent_by_class"].get(class_id, 0.0)
                if pct > 0:
                    logger.info("    SCL class %2d: %.4f%%", class_id, pct)
            for band, band_stat in stats["reflectance_valid_pixels"].items():
                logger.info(
                    "    %s reflectance  min %.3f  median %.3f  max %.3f",
                    band, band_stat["min"], band_stat["median"], band_stat["max"],
                )

        grid_a = read_reference_grid(RAW_DIR / "date_a" / "B04.tif")
        grid_b = read_reference_grid(RAW_DIR / "date_b" / "B04.tif")
        if not grids_match(grid_a, grid_b):
            raise ValueError("Date A and Date B are not on the same 10 m grid")
        report["grid"] = grid_summary(grid_a)
        logger.info("Both dates share one grid: %d x %d, %s", grid_a["width"], grid_a["height"], grid_a["crs"])

        logger.info("Reprojecting WorldCover onto the 10 m grid ...")
        labels = reproject_worldcover(
            RAW_DIR / "worldcover.tif", grid_a["crs"], grid_a["transform"], grid_a["width"], grid_a["height"]
        )
        unexpected = set(np.unique(labels).tolist()) - VALID_WORLDCOVER_CLASSES
        if unexpected:
            raise ValueError(f"Unexpected WorldCover class codes after reprojection: {sorted(unexpected)}")
        labelled = labels != 0
        if not labelled.any():
            raise ValueError("WorldCover reprojection produced no labelled pixels")
        nodata_fraction = float(1.0 - labelled.mean())
        if nodata_fraction > max_label_nodata:
            raise ValueError(
                f"{100 * nodata_fraction:.2f}% of the AOI has no WorldCover label "
                f"(limit {100 * max_label_nodata:.2f}%)"
            )
        if nodata_fraction > 0:
            logger.warning("%.4f%% of the AOI has no WorldCover label", 100 * nodata_fraction)
        forest_fraction = float((labels == forest_class)[labelled].mean())

        classes, counts = np.unique(labels, return_counts=True)
        class_percent = {int(c): round(100.0 * int(n) / labels.size, 4) for c, n in zip(classes, counts)}
        for class_id, pct in class_percent.items():
            logger.info("    WorldCover class %3d: %.2f%%", class_id, pct)
        logger.info("  forest (class %d) fraction of labelled pixels: %.2f%%", forest_class, 100 * forest_fraction)

        write_raster(PROCESSED_DIR / "worldcover_10m.tif", labels, {**grid_profile(grid_a), "nodata": 0})

        joint_valid = valid_masks["date_a"] & valid_masks["date_b"]
        report["worldcover"] = {
            "item_id": manifest.get("worldcover", {}).get("item_id"),
            "forest_class": forest_class,
            "class_percent": class_percent,
            "nodata_fraction": round(nodata_fraction, 6),
            "forest_fraction_of_labelled": round(forest_fraction, 6),
        }
        report["valid_in_both_dates_fraction"] = round(float(joint_valid.mean()), 6)
        report["valid_in_date_a_and_labelled_fraction"] = round(float((valid_masks["date_a"] & labelled).mean()), 6)
        logger.info("Pixels valid in BOTH dates: %.3f%%", 100 * report["valid_in_both_dates_fraction"])

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", REPORT_PATH)
    except Exception:
        logger.exception("Preprocessing failed")
        return 1

    logger.info("PREPROCESSING COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())