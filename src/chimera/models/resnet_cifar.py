"""Small CIFAR ResNet skeleton for Chimera execution experiments."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from chimera.bitconv import BitConv2d
from chimera.bitlinear import BitLinear


ModelMode = Literal["fp32_baseline", "strict_bitnet", "chimera"]


def _quant_kwargs(quant_config: dict[str, Any] | None) -> dict[str, Any]:
    config = quant_config or {}
    keys = (
        "stat_mode",
        "rho_s",
        "alpha_min",
        "alpha_target",
        "warmup_steps",
        "eps_gamma",
        "eps_beta",
        "ste_clip",
    )
    return {key: config[key] for key in keys if key in config}


def _conv3x3(
    in_channels: int,
    out_channels: int,
    *,
    stride: int,
    mode: ModelMode,
    quant_config: dict[str, Any] | None = None,
    force_fp32: bool = False,
) -> nn.Module:
    if mode == "fp32_baseline" or force_fp32:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
    return BitConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        quant_mode=mode,
        **_quant_kwargs(quant_config),
    )


def _conv1x1(
    in_channels: int,
    out_channels: int,
    *,
    stride: int,
    mode: ModelMode,
    quant_config: dict[str, Any] | None = None,
) -> nn.Module:
    if mode == "fp32_baseline":
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    return BitConv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
        quant_mode=mode,
        **_quant_kwargs(quant_config),
    )


def _linear(
    in_features: int,
    out_features: int,
    *,
    mode: ModelMode,
    quant_config: dict[str, Any] | None = None,
    force_fp32: bool = False,
) -> nn.Module:
    if mode == "fp32_baseline" or force_fp32:
        return nn.Linear(in_features, out_features)
    return BitLinear(
        in_features,
        out_features,
        quant_mode=mode,
        **_quant_kwargs(quant_config),
    )


class ChimeraBasicBlock(nn.Module):
    """Basic CIFAR ResNet block with optional Chimera quantized convolutions."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        mode: ModelMode = "chimera",
        quant_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("fp32_baseline", "strict_bitnet", "chimera"):
            raise ValueError("mode must be fp32_baseline, strict_bitnet, or chimera")
        self.conv1 = _conv3x3(
            in_channels,
            out_channels,
            stride=stride,
            mode=mode,
            quant_config=quant_config,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(
            out_channels,
            out_channels,
            stride=1,
            mode=mode,
            quant_config=quant_config,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                _conv1x1(
                    in_channels,
                    out_channels,
                    stride=stride,
                    mode=mode,
                    quant_config=quant_config,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ChimeraResNet20(nn.Module):
    """CIFAR-style ResNet-20 skeleton for v3 effective-bit experiments."""

    def __init__(
        self,
        num_classes: int = 10,
        mode: ModelMode = "chimera",
        quant_config: dict[str, Any] | None = None,
        first_last_fp32: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("fp32_baseline", "strict_bitnet", "chimera"):
            raise ValueError("mode must be fp32_baseline, strict_bitnet, or chimera")
        self.mode = mode
        self.quant_config = dict(quant_config or {})
        self.first_last_fp32 = bool(first_last_fp32)
        self.in_channels = 16
        self.conv1 = _conv3x3(
            3,
            16,
            stride=1,
            mode=mode,
            quant_config=self.quant_config,
            force_fp32=self.first_last_fp32,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, blocks=3, stride=1, mode=mode)
        self.layer2 = self._make_layer(32, blocks=3, stride=2, mode=mode)
        self.layer3 = self._make_layer(64, blocks=3, stride=2, mode=mode)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = _linear(
            64,
            num_classes,
            mode=mode,
            quant_config=self.quant_config,
            force_fp32=self.first_last_fp32,
        )

    def _make_layer(
        self,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
        mode: ModelMode,
    ) -> nn.Sequential:
        layers = [
            ChimeraBasicBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                mode=mode,
                quant_config=self.quant_config,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                ChimeraBasicBlock(
                    out_channels,
                    out_channels,
                    mode=mode,
                    quant_config=self.quant_config,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run CIFAR ResNet-20 forward."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
