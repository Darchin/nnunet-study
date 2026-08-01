from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.common_types import _size_any_t

ModuleFactory = Callable[..., nn.Module]


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
        return x
