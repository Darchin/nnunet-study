from typing import Literal, Optional, Sequence
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.network_architecture.common import (
    ModuleFactory,
    ConvNd,
    AdaptiveAvgPoolNd,
    ConvBlock,
    ConvBlockOpSeq,
)
from nnunetv2.network_architecture.types import ShapeNd
from nnunetv2.network_architecture.utils import compute_padding, ensure_ntuple

type RouterOpSeq = Sequence[
    Literal["gap", "conv", "sigmoid", "softmax", "softplus", "norm"]
]


class Sigmoid(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(x / self.temperature)


class Softmax(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x / self.temperature, dim=1)


class Normalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.sum(dim=1, keepdim=True)


class Router(nn.Module):
    def __init__(
        self,
        ndim: int,
        channels: int,
        kernel_size: ShapeNd,
        stride: ShapeNd,
        num_experts: int,
        op_seq: RouterOpSeq,
    ):
        super().__init__()
        self.op_seq = list(op_seq)

        assert set(self.op_seq).issubset(
            {"gap", "conv", "sigmoid", "softmax", "softplus", "norm"}
        )

        assert all(
            op in self.op_seq for op in ["gap", "conv"]
        ), "Both GAP and Conv must be present in the op_seq."

        if self.op_seq.index("gap") < self.op_seq.index("conv"):
            assert all(
                k == 1 for k in ensure_ntuple(kernel_size, ndim)
            ), "Post-GAP router convolution must be pointwise."
            assert all(
                s == 1 for s in ensure_ntuple(stride, ndim)
            ), "Post-GAP router convolution cannot be strided."

        for op in op_seq:
            match op:
                case "conv":
                    self.add_module(
                        "conv",
                        ConvNd(
                            ndim,
                            channels,
                            num_experts,
                            kernel_size,
                            stride,
                            compute_padding(ndim, kernel_size),
                        ),
                    )
                case "gap":
                    self.add_module("gap", AdaptiveAvgPoolNd(ndim, 1))
                case "sigmoid":
                    self.add_module("sigmoid", Sigmoid())
                case "softmax":
                    self.add_module("softmax", Softmax())
                case "softplus":
                    self.add_module("softplus", nn.Softplus())
                case "norm":
                    self.add_module("norm", Normalize())

        self.reset_parameters()

    def reset_parameters(self):
        # initialize router to generate uniform scores at the start
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.children():
            x = layer(x)
        return x.squeeze(*list(range(2, x.ndim)))


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
        num_experts: Optional[int] = None,
        op_seq: ConvBlockOpSeq = [
            "conv",
            "norm",
            "act",
        ],
    ):
        self._is_cc = num_experts is not None

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
                partial(CondPWConv, num_experts=num_experts) if self._is_cc else ConvNd
            ),
            normalization=normalization,
            activation=activation,
            op_seq=op_seq,
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
        for name, layer in self.named_children():
            if name == "conv":
                if self._is_cc:
                    assert scores is not None
                    x = layer(x, scores)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x
