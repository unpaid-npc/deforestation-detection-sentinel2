"""Inference script for predicting forest masks and detecting deforestation.

Runs the trained U-Net over the processed ARD rasters for Date A and Date B using
a sliding window. Subtracts the masks to calculate deforestation and saves the results.

Run from the project root:
    python -m src.inference.predict --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import load_normalization
from src.data.preprocess import grid_profile, read_reference_grid, write_raster
from src.models.unet import UNet
from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger("predict")

PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs" / "masks"
CHECKPOINT_PATH: Path = MODELS_DIR / "best_unet.pth"
NORMALIZATION_PATH: Path = PROCESSED_DIR / "normalization.json"


class SlidingWindowDataset(Dataset):
    """Yields all spatial patches of an image in order for inference."""
    
    def __init__(self, reflectance: np.ndarray, patch_size: int, mean: np.ndarray, std: np.ndarray):
        self.reflectance = reflectance
        self.patch_size = patch_size
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)
        
        _, self.height, self.width = reflectance.shape
        
        self.origins = []
        # Stride equals patch_size (no overlap for MVP simplicity)
        for row in range(0, self.height, patch_size):
            for col in range(0, self.width, patch_size):
                self.origins.append((row, col))

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, col = self.origins[index]
        
        # Extract the patch and pad with zeros if it hits the bottom/right edge
        patch = np.zeros((6, self.patch_size, self.patch_size), dtype=np.float32)
        
        r_end = min(row + self.patch_size, self.height)
        c_end = min(col + self.patch_size, self.width)
        
        valid_h = r_end - row
        valid_w = c_end - col
        
        raw = self.reflectance[:, row:r_end, col:c_end]
        
        # Apply normalization to the valid pixels only
        normalized = np.nan_to_num((raw - self.mean) / self.std, nan=0.0)
        patch[:, :valid_h, :valid_w] = normalized
        
        return {
            "image": torch.from_numpy(patch),
            "origin": (row, col),
            "valid_h": valid_h,
            "valid_w": valid_w
        }


def predict_date(
    date_key: str, model: torch.nn.Module, device: torch.device, settings: dict[str, Any], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Run inference on the entire image for the given date."""
    logger.info("Predicting forest cover for %s...", date_key)
    
    reflectance_path = PROCESSED_DIR / date_key / "reflectance.tif"
    with rasterio.open(reflectance_path) as src:
        reflectance = src.read().astype(np.float32)
        height, width = src.height, src.width
        
    dataset = SlidingWindowDataset(reflectance, settings["patch_size"], mean, std)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False)
    
    # Pre-allocate the full output mask
    full_prediction = np.zeros((height, width), dtype=np.float32)
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            origins_row = batch["origin"][0].numpy()
            origins_col = batch["origin"][1].numpy()
            valid_hs = batch["valid_h"].numpy()
            valid_ws = batch["valid_w"].numpy()
            
            # Get logits and convert to probabilities
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().squeeze(1).numpy()
            
            # Place predictions back into the full array
            for i in range(len(images)):
                r, c = origins_row[i], origins_col[i]
                vh, vw = valid_hs[i], valid_ws[i]
                full_prediction[r:r+vh, c:c+vw] = probs[i, :vh, :vw]
                
    # Convert probabilities to a binary mask (Threshold = 0.5)
    binary_mask = (full_prediction > 0.5).astype(np.uint8)
    return binary_mask


def calculate_deforestation(mask_a: np.ndarray, mask_b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray) -> np.ndarray:
    """Calculate where forest existed in Date A but disappeared in Date B."""
    
    # Both dates must have valid (cloud-free) pixels to assess change
    joint_valid = valid_a & valid_b
    
    # 1 = Deforestation, 0 = No change or Not Forest
    deforestation = ((mask_a == 1) & (mask_b == 0)).astype(np.uint8)
    
    # Apply joint valid mask (ignore changes hidden by clouds)
    deforestation[~joint_valid] = 0
    
    return deforestation


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run U-Net inference and detect change.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        
        # Load normalization stats from the training phase
        if not NORMALIZATION_PATH.exists():
            raise FileNotFoundError(f"Missing {NORMALIZATION_PATH}. Run src.data.dataset first.")
        mean, std = load_normalization(NORMALIZATION_PATH)
        
        # Load trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=6, out_channels=1, base_features=32)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
    except Exception as exc:
        logger.error("Setup failed: %s", exc)
        return 1

    try:
        # 1. Predict Date A and Date B
        mask_a = predict_date("date_a", model, device, config["dataset"], mean, std)
        mask_b = predict_date("date_b", model, device, config["dataset"], mean, std)
        
        # 2. Load Valid Masks
        with rasterio.open(PROCESSED_DIR / "date_a" / "valid_mask.tif") as src:
            valid_a = src.read(1).astype(bool)
        with rasterio.open(PROCESSED_DIR / "date_b" / "valid_mask.tif") as src:
            valid_b = src.read(1).astype(bool)
            
        # 3. Calculate Deforestation
        deforestation = calculate_deforestation(mask_a, mask_b, valid_a, valid_b)
        
        # 4. Calculate Spatial Area (Hectares)
        # Sentinel-2 pixels are 10m x 10m (100 sq meters). 1 Hectare = 10,000 sq meters.
        # Therefore, 1 pixel = 0.01 Hectares.
        total_pixels = int(deforestation.sum())
        total_hectares = total_pixels * 0.01
        
        logger.info("======================================")
        logger.info("CHANGE DETECTION RESULTS:")
        logger.info("Total Deforested Pixels: %d", total_pixels)
        logger.info("Total Area Deforested: %.2f Hectares", total_hectares)
        logger.info("======================================")

        # 5. Save Results
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        grid = read_reference_grid(PROCESSED_DIR / "date_a" / "reflectance.tif")
        profile = grid_profile(grid)
        profile.update({"nodata": 0, "dtype": "uint8"})
        
        write_raster(OUTPUTS_DIR / "date_a_forest.tif", mask_a, profile)
        write_raster(OUTPUTS_DIR / "date_b_forest.tif", mask_b, profile)
        write_raster(OUTPUTS_DIR / "deforestation_mask.tif", deforestation, profile)
        
        logger.info("Saved masks to %s", OUTPUTS_DIR)

    except Exception:
        logger.exception("Inference failed")
        return 1

    logger.info("INFERENCE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())