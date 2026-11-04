"""PyTorch U-Net architecture for Sentinel-2 segmentation.

Standard U-Net with batch normalization and dropout to prevent overfitting.
Expects inputs of shape (B, 6, H, W) and outputs logits of shape (B, 1, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Pad if spatial dimensions don't match perfectly (due to arbitrary input sizes)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        if diff_y > 0 or diff_x > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net for binary segmentation of 6-band Sentinel-2 imagery."""

    def __init__(
        self, 
        in_channels: int = 6, 
        out_channels: int = 1, 
        base_features: int = 32,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2, dropout_rate)
        self.down2 = Down(base_features * 2, base_features * 4, dropout_rate)
        self.down3 = Down(base_features * 4, base_features * 8, dropout_rate)
        
        # Bottleneck
        self.down4 = Down(base_features * 8, base_features * 16, dropout_rate)
        
        self.up1 = Up(base_features * 16, base_features * 8, dropout_rate)
        self.up2 = Up(base_features * 8, base_features * 4, dropout_rate)
        self.up3 = Up(base_features * 4, base_features * 2, dropout_rate)
        self.up4 = Up(base_features * 2, base_features)
        
        # Output layer produces unnormalized logits
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits