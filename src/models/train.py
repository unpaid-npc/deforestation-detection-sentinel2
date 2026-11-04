"""Training loop for the PyTorch U-Net segmentation model.

Implements Masked Focal Loss to handle class imbalance and ignore cloudy/unlabelled
pixels. Saves the best model checkpoint based on validation loss.

Run from the project root:
    python -m src.models.train --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import (
    PROCESSED_DIR,
    build_datasets_from_arrays,
    load_processed_arrays,
    make_dataloaders,
    settings_from_config,
)
from src.models.unet import UNet
from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger("train")

MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINT_PATH: Path = MODELS_DIR / "best_unet.pth"

# Training Hyperparameters for MVP
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5


class MaskedFocalLoss(nn.Module):
    """Focal Loss that ignores pixels where the validity mask is 0."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        # Focal weight calculation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - p_t) ** self.gamma
        
        # Apply mask and focal weights
        loss = bce_loss * focal_weight * mask
        
        mask_sum = mask.sum()
        if mask_sum > 0:
            return loss.sum() / mask_sum
        return loss.sum() * 0.0


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device
) -> float:
    """Train for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        # Add channel dimension: [B, H, W] -> [B, 1, H, W]
        labels = batch["label"].to(device).unsqueeze(1)
        masks = batch["valid"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        
        loss = criterion(logits, labels, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Evaluate the model on the validation set and return the average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            masks = batch["valid"].to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels, masks)
            total_loss += loss.item()

    return total_loss / len(loader)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the training loop."""
    parser = argparse.ArgumentParser(description="Train the U-Net model.")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        config = load_config(args.config)
        settings = settings_from_config(config)
    except Exception as exc:
        logger.error("Configuration problem: %s", exc)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    try:
        logger.info("Loading processed arrays for date: %s", settings.train_date)
        reflectance, labels, valid = load_processed_arrays(PROCESSED_DIR, settings.train_date)
        datasets, _ = build_datasets_from_arrays(reflectance, labels, valid, settings)
        loaders = make_dataloaders(datasets, settings)
        
        logger.info("Train patches: %d | Val patches: %d", len(datasets["train"]), len(datasets["val"]))

        model = UNet(in_channels=6, out_channels=1, base_features=32).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = MaskedFocalLoss()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        logger.info("Starting training for %d epochs...", EPOCHS)
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
            val_loss = validate_epoch(model, loaders["val"], criterion, device)

            logger.info("Epoch %2d/%2d | Train Loss: %.4f | Val Loss: %.4f", epoch, EPOCHS, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                logger.info("  -> Saved new best model to %s", CHECKPOINT_PATH.name)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    logger.warning("Early stopping triggered after %d epochs without improvement.", PATIENCE)
                    break

    except Exception:
        logger.exception("Training failed")
        return 1

    logger.info("TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())