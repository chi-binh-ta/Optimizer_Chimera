"""BitConv2d-style execution layer for Chimera 2.1/2.3 experiments."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

from .quantization import (
    StatMode,
    abs_stat,
    quantize_weight_chimera,
    quantize_weight_strict_bitnet,
    ternary_stats,
)


QuantMode = Literal["chimera", "strict_bitnet"]


class BitConv2d(nn.Module):
    """2D convolution with quantized activations and ternary weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        stat_mode: StatMode = "mean",
        rho_s: float = 0.95,
        alpha_min: float = 0.0,
        alpha_target: float = 0.7,
        warmup_steps: int = 100,
        quant_mode: QuantMode = "chimera",
        eps_gamma: float = 1e-6,
        eps_beta: float = 1e-6,
        ste_clip: float | None = 1.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if groups <= 0 or in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("groups must divide both in_channels and out_channels")
        if not 0.0 <= rho_s <= 1.0:
            raise ValueError("rho_s must be in [0, 1]")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if eps_gamma <= 0 or eps_beta <= 0:
            raise ValueError("eps_gamma and eps_beta must be positive")
        if quant_mode not in ("chimera", "strict_bitnet"):
            raise ValueError("quant_mode must be 'chimera' or 'strict_bitnet'")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.stat_mode = stat_mode
        self.rho_s = float(rho_s)
        self.alpha_min = float(alpha_min)
        self.alpha_target = float(alpha_target)
        self.warmup_steps = int(warmup_steps)
        self.quant_mode = quant_mode
        self.eps_gamma = float(eps_gamma)
        self.eps_beta = float(eps_beta)
        self.ste_clip = ste_clip
        self.alpha_override: float | None = None

        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        self.weight_master = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        initial_scale = abs_stat(self.weight_master.detach(), self.stat_mode)
        self.register_buffer("scale_ema", initial_scale.clone())
        self.register_buffer("forward_step", torch.zeros((), dtype=torch.long))
        self._last_stats: dict[str, float] | None = None

    def reset_parameters(self) -> None:
        """Initialize parameters like torch.nn.Conv2d."""
        nn.init.kaiming_uniform_(self.weight_master, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            fan_in //= self.groups
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def current_alpha(self) -> float:
        """Return alpha for the next forward pass."""
        if self.alpha_override is not None:
            return float(self.alpha_override)
        if self.warmup_steps == 0:
            return self.alpha_target
        step = int(self.forward_step.item())
        if self.warmup_steps == 1:
            progress = 1.0
        else:
            progress = min(float(step) / float(self.warmup_steps - 1), 1.0)
        return self.alpha_min + progress * (self.alpha_target - self.alpha_min)

    def compute_gamma(self) -> torch.Tensor:
        """Return the current positive weight scale."""
        return self.scale_ema.clamp_min(self.eps_gamma)

    def quantize_weight(
        self,
        gamma: torch.Tensor | float | None = None,
        alpha: float | None = None,
    ) -> torch.Tensor:
        """Return the current ternary quantized convolution weight."""
        gamma_value = self.compute_gamma() if gamma is None else gamma
        alpha_value = self.current_alpha() if alpha is None else alpha
        if self.quant_mode == "strict_bitnet":
            return quantize_weight_strict_bitnet(self.weight_master, gamma_value)
        return quantize_weight_chimera(self.weight_master, gamma_value, alpha_value)

    def get_last_stats(self) -> dict[str, float]:
        """Return stats from the latest forward pass."""
        if self._last_stats is None:
            gamma = float(self.compute_gamma().item())
            alpha = float(self.current_alpha())
            stats = ternary_stats(self.quantize_weight().detach())
            return {"gamma": gamma, "alpha": alpha, **stats}
        return dict(self._last_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a BitConv2d-style forward pass for NCHW inputs."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError(f"expected input shape [batch, channels, height, width], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")

        beta = x.detach().abs().amax().div(127.0).add(self.eps_beta)
        x_q = torch.clamp(torch.round(x / beta), -127, 127)

        if self.training:
            with torch.no_grad():
                s = abs_stat(self.weight_master.detach(), self.stat_mode).to(self.scale_ema)
                self.scale_ema.mul_(self.rho_s).add_(s, alpha=1.0 - self.rho_s)

        gamma = self.compute_gamma()
        alpha = self.current_alpha()
        w_q_raw = self.quantize_weight(gamma=gamma, alpha=alpha)
        w_q = self.weight_master + (w_q_raw - self.weight_master).detach()
        if self.ste_clip is not None:
            w_q = torch.clamp(w_q, -float(self.ste_clip), float(self.ste_clip))

        y_int = F.conv2d(
            x_q,
            w_q,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        y = (beta * gamma) * y_int
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        self._last_stats = {
            "gamma": float(gamma.detach().item()),
            "alpha": float(alpha),
            **ternary_stats(w_q_raw.detach()),
        }
        if self.training:
            self.forward_step.add_(1)
        return y


ChimeraConv2d = BitConv2d
