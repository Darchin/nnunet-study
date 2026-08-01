from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.common_types import _size_any_t

ModuleFactory = Callable[..., nn.Module]


def AdaptiveAvgPoolNd(
    N: int, *args, **kwargs
) -> nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d:
    return [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d][N - 1](
        *args, **kwargs
    )


def ConvNd(N: int, *args, **kwargs) -> nn.Conv1d | nn.Conv2d | nn.Conv3d:
    return [nn.Conv1d, nn.Conv2d, nn.Conv3d][N - 1](*args, **kwargs)


def ConvTransposeNd(
    N: int, *args, **kwargs
) -> nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d:
    return [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][N - 1](
        *args, **kwargs
    )


def LinearUpsampleNd(N: int, *args, **kwargs) -> nn.Upsample:
    return nn.Upsample(mode=["linear", "bilinear", "trilinear"][N - 1], *args, **kwargs)


class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, ndim: int, channels: int, reduction: float):
        super().__init__()
        hidden_channels = int(max(1, channels // reduction))

        self.gap = AdaptiveAvgPoolNd(ndim, 1)

        self.pw1 = ConvNd(ndim, channels, hidden_channels, kernel_size=1)
        self.act1 = nn.ReLU(inplace=True)
        self.pw2 = ConvNd(ndim, hidden_channels, channels, kernel_size=1)
        self.act2 = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.pw1.weight)
        nn.init.zeros_(self.pw1.bias)
        
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor):
        x_gap = self.gap(x)
        logits = self.pw2(self.act1(self.pw1(x_gap)))
        scores = 2 * self.act2(logits)
        x = x * scores
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_any_t = 1,
        stride: _size_any_t = 1,
        padding: _size_any_t = 0,
        groups: int = 1,
        bias: bool | None = None,
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        se_reduction: Optional[float] = None,
    ):
        super().__init__()

        self.norm = normalization(out_channels)
        self.act = activation()

        _has_bias = bias or isinstance(self.norm, nn.Identity)

        self.conv = ConvNd(
            ndim,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=_has_bias,
        )

        self.se = (
            SqueezeAndExcitationBlock(ndim, out_channels, se_reduction)
            if se_reduction is not None
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        # initialize conv
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # initialize norm
        if not isinstance(self.norm, nn.Identity):
            if hasattr(self.norm, "weight") and self.norm.weight is not None:
                nn.init.ones_(self.norm.weight)
            if hasattr(self.norm, "bias") and self.norm.bias is not None:
                nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        return x