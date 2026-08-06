from typing import Literal, Optional, Sequence
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t
from nnunetv2.network_architecture.common import (
    ModuleFactory,
    ConvNd,
    AdaptiveAvgPoolNd,
    ConvBlock,
)
from nnunetv2.network_architecture.types import ShapeNd
from nnunetv2.network_architecture.utils import compute_padding
from nnunetv2.network_architecture.init import identity_init


class Router(nn.Module):
    def __init__(
        self,
        ndim: int,
        channels: int,
        kernel_size: _size_any_t,
        stride: _size_any_t,
        num_experts: int,
    ):
        super().__init__()

        self.conv = ConvNd(
            ndim,
            channels,
            num_experts,
            kernel_size,
            stride,
            compute_padding(ndim, kernel_size),
        )

        self.gap = AdaptiveAvgPoolNd(ndim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize router to generate uniform scores at the start
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit_map = self.conv(x)
        logit_gap = self.gap(logit_map)
        logit_gap = logit_gap.squeeze(*list(range(2, logit_gap.ndim)))

        scores_unnormalized = F.sigmoid(logit_gap)
        scores = scores_unnormalized / scores_unnormalized.sum(dim=-1, keepdim=True)

        return scores.squeeze()


class CondPWConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        num_experts: int = 1,
        *args,
        **kwargs,
        # args/kwargs added to ensure compatibility with ConvBlock's conv initializer
    ):
        super().__init__()
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.empty(num_experts, out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # initialize experts
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])

    def blend_weights(self, scores: torch.Tensor) -> torch.Tensor:
        weight = torch.einsum("bs, eoi -> boi", scores, self.weight)
        return weight

    def blend_biases(self, scores: torch.Tensor) -> torch.Tensor:
        bias = torch.einsum("bs,eo->bo", scores, self.bias)
        return bias

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        weight = self.blend_weights(scores)
        bias = self.blend_biases(scores) if self.bias is not None else None

        x = torch.einsum("boi, bi... -> bo...", weight, x)
        if bias is not None:
            bias = bias.view(*bias.shape, *([1] * (x.ndim - 2)))
            x += bias

        return x


class CondPWConvBlock(ConvBlock):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        bias: bool | None = None,
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        se_reduction: Optional[float] = None,
        cc_num_experts: Optional[int] = None,
        layer_sequence: Sequence[Literal["conv", "norm", "act", "se"]] = [
            "conv",
            "norm",
            "act",
            "se",
        ],
    ):
        self._is_cc = cc_num_experts is not None

        super().__init__(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            convolution=(
                partial(CondPWConv, num_experts=cc_num_experts)
                if self._is_cc
                else ConvNd
            ),
            normalization=normalization,
            activation=activation,
            se_reduction=se_reduction,
            layer_sequence=layer_sequence,
        )

    def forward(
        self, x: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ---
        x : Tensor of shape (batch_size, in_channels, ...)
            Input feature map.
        scores : Optional Tensor of shape (batch_size, num_experts)
            Per-sample score map for the experts.
        """
        if self._is_cc:
            assert scores is not None
            return super().forward(x, scores)
        return super().forward(x)
