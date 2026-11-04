"""Tests for the U-Net architecture."""

import pytest
import torch
from src.models.unet import UNet

def test_unet_shape_and_forward_pass():
    """Verify the U-Net accepts [B, 6, H, W] and outputs [B, 1, H, W]."""
    batch_size = 2
    channels = 6
    height, width = 128, 128
    
    model = UNet(in_channels=channels, out_channels=1, base_features=16)
    model.eval()
    
    # Create a synthetic batch
    x = torch.randn(batch_size, channels, height, width)
    
    with torch.no_grad():
        out = model(x)
        
    assert out.shape == (batch_size, 1, height, width), f"Expected {(batch_size, 1, height, width)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"