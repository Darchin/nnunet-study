from functools import partial
from typing import Literal, Sequence

import torch
import torch.nn as nn

from nnunetv2.network_architecture.nd import AdaptiveAvgPoolNd, ConvNd, ConvTransposeNd
from nnunetv2.network_architecture.types import ModuleFactory, ShapeNd
from nnunetv2.network_architecture.utils import ensure_ntuple
from nnunetv2.network_architecture.init import identity_init

type ConvBlockOpSeq = Sequence[Literal["conv", "norm", "act"]]


class SqueezeAndExcitationBlock(nn.Module):
    def __init__(
        self, ndim: int, channels: int, reduction: float, activation: ModuleFactory
    ):
        super().__init__()
        hidden_channels = int(max(1, channels // reduction))

        self.gap = AdaptiveAvgPoolNd(ndim, 1)

        self.pw1 = ConvNd(ndim, channels, hidden_channels, kernel_size=1)
        self.act1 = activation()
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
        kernel_size: int | ShapeNd = 1,
        stride: int | ShapeNd = 1,
        padding: int | ShapeNd = 0,
        groups: int = 1,
        bias: bool | None = None,
        convolution: ModuleFactory = ConvNd,
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        op_seq: ConvBlockOpSeq = [
            "conv",
            "norm",
            "act",
        ],
    ):
        super().__init__()

        assert "conv" in set(op_seq)
        assert set(op_seq).issubset({"conv", "norm", "act"})

        self._norm_is_identity = issubclass(
            normalization.func if isinstance(normalization, partial) else normalization,
            nn.Identity,
        )

        _has_bias = bias or self._norm_is_identity

        channels = in_channels
        for op in op_seq:
            match op:
                case "conv":
                    self.add_module(
                        "conv",
                        convolution(
                            N=ndim,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            groups=groups,
                            bias=_has_bias,
                        ),
                    )
                    channels = out_channels
                case "norm":
                    self.add_module("norm", normalization(ndim, channels))
                case "act":
                    self.add_module("act", activation())

        self.reset_parameters()

    def reset_parameters(self):
        # initialize conv
        if isinstance(self.conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

        # initialize norm
        if hasattr(self, "norm") and not self._norm_is_identity:
            identity_init(self.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x


class DepthwiseUpsample(nn.Module):
    def __init__(self, ndim: int, channels: int, stride: ShapeNd, bias: bool = True):
        super().__init__()
        self._stride = ensure_ntuple(stride, ndim)
        self._kernel_size = tuple(2 * s for s in self._stride)
        self._padding = tuple(s // 2 for s in self._stride)

        self.conv = ConvTransposeNd(
            ndim,
            channels,
            channels,
            self._kernel_size,
            self._stride,
            self._padding,
            groups=channels,
            bias=bias,
        )

        self.reset_parameters()

    # Linear interpolation init
    def reset_parameters(self):
        with torch.no_grad():
            dtype = self.conv.weight.dtype
            device = self.conv.weight.device
            ndim = len(self._stride)

            kernel_nd = torch.tensor(1.0, dtype=dtype, device=device)

            for dim_idx, s in enumerate(self._stride):
                grid = torch.arange(2 * s, dtype=dtype, device=device)
                w_1d = 1.0 - torch.abs(grid - (s - 0.5)) / s

                shape = [1] * (2 + ndim)
                shape[2 + dim_idx] = 2 * s
                w_1d = w_1d.view(*shape)

                kernel_nd = kernel_nd * w_1d

            self.conv.weight.copy_(kernel_nd.expand_as(self.conv.weight))

            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
