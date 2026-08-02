from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t
from nnunetv2.network_architecture.common import (
    ModuleFactory,
    SqueezeAndExcitationBlock,
    ConvNd,
    AdaptiveAvgPoolNd,
)
from nnunetv2.network_architecture.utils import compute_padding


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


class CondPWConvBlock(nn.Module):
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
    ):
        super().__init__()
        self.norm = normalization(ndim, out_channels)
        self.act = activation()

        _has_bias = bias or isinstance(self.norm, nn.Identity)
        self._is_cc = cc_num_experts is not None

        if not self._is_cc:
            self.conv = ConvNd(ndim, in_channels, out_channels, 1, bias=_has_bias)
        else:
            self.conv = CondPWConv(in_channels, out_channels, _has_bias, cc_num_experts)

        self.se = (
            SqueezeAndExcitationBlock(ndim, out_channels, se_reduction)
            if se_reduction is not None
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        # initialize conv params if not using CC
        if not self._is_cc:
            nn.init.kaiming_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

        # initialize norm
        if not isinstance(self.norm, nn.Identity):
            if hasattr(self.norm, "weight") and self.norm.weight is not None:
                nn.init.ones_(self.norm.weight)
            if hasattr(self.norm, "bias") and self.norm.bias is not None:
                nn.init.zeros_(self.norm.bias)

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
            x = self.conv(x, scores)
        else:
            x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        return x
