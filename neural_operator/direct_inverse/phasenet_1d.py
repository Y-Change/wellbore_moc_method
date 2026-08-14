# -*- coding: utf-8 -*-
"""PhaseNet-style fully convolutional 1D U-Net for direct event localization."""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

ForwardOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int):
        super().__init__()
        padding = kernel_size // 2
        normalizer_groups = min(groups, out_channels)
        while out_channels % normalizer_groups:
            normalizer_groups -= 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(normalizer_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(normalizer_groups, out_channels)
        self.activation = nn.SiLU()
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, config: ModelConfig):
        super().__init__()
        self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.block = ResidualBlock1D(
            out_channels + skip_channels,
            out_channels,
            config.kernel_size,
            config.groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = self.projection(x)
        return self.block(torch.cat([x, skip], dim=1))


class PhaseNet1D(nn.Module):
    """Waveform (1 or 3 ch) to fracture-event logits, optional count head."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        if config.model_id not in ("p0a_raw_unet", "p0a_raw_unet_count", "p1_phys3_unet"):
            raise ValueError(f"unsupported model_id: {config.model_id}")
        if config.in_channels not in (1, 3) or config.out_channels != 1:
            raise ValueError("P0-A/P1 requires in_channels in {1,3} and out_channels=1")
        if config.model_id == "p1_phys3_unet" and config.in_channels != 3:
            raise ValueError("p1_phys3_unet requires in_channels=3")
        if config.enable_count_head and config.count_classes < 1:
            raise ValueError("count_classes must be >= 1 when count head is enabled")
        self.config = config
        channels = config.channels
        self.stem = nn.Conv1d(
            config.in_channels,
            channels[0],
            config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        previous = channels[0]
        for index, channel in enumerate(channels):
            self.encoder_blocks.append(
                ResidualBlock1D(previous, channel, config.kernel_size, config.groups)
            )
            previous = channel
            if index < len(channels) - 1:
                self.downsamplers.append(
                    nn.Conv1d(channel, channels[index + 1], kernel_size=4, stride=2, padding=1)
                )
                previous = channels[index + 1]

        self.bottleneck = nn.Sequential(
            ResidualBlock1D(channels[-1], channels[-1], config.kernel_size, config.groups),
            nn.Dropout(config.bottleneck_dropout),
            ResidualBlock1D(channels[-1], channels[-1], config.kernel_size, config.groups),
        )
        self.decoder_stages = nn.ModuleList()
        current = channels[-1]
        for skip_channel in reversed(channels[:-1]):
            self.decoder_stages.append(
                DecoderStage(current, skip_channel, skip_channel, config)
            )
            current = skip_channel
        self.head = nn.Conv1d(channels[0], config.out_channels, kernel_size=1)
        self.count_head = None
        if config.enable_count_head:
            self.count_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(channels[-1], config.count_classes),
            )

    def forward(self, observation: torch.Tensor) -> ForwardOutput:
        if observation.ndim != 3 or observation.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected [B,{self.config.in_channels},T], got {tuple(observation.shape)}"
            )
        x = self.stem(observation)
        skips = []
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            skips.append(x)
            if index < len(self.downsamplers):
                x = self.downsamplers[index](x)
        bottleneck = self.bottleneck(x)
        x = bottleneck
        for stage, skip in zip(self.decoder_stages, reversed(skips[:-1])):
            x = stage(x, skip)
        logits = self.head(x)
        if logits.shape[-1] != observation.shape[-1]:
            logits = F.interpolate(logits, size=observation.shape[-1], mode="linear", align_corners=False)
        if self.count_head is None:
            return logits
        return logits, self.count_head(bottleneck)


def unpack_model_output(output: ForwardOutput) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def parameter_count(model: nn.Module) -> dict:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
