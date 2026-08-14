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

type MoEBackend = Literal["bmm", "bg", "vmap", "naive"]


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
        num_experts: int,
        kernel_size: ShapeNd,
        stride: ShapeNd,
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


class MoEConvNd(nn.Module):
    def __init__(
        self,
        N: int,
        in_channels: int,
        out_channels: int,
        kernel_size: ShapeNd = 1,
        stride: ShapeNd = 1,
        padding: ShapeNd = 0,
        groups: int = 1,
        bias: bool = True,
        num_experts: int = 1,
        backend: MoEBackend = "bg",
    ):
        super().__init__()
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = ensure_ntuple(kernel_size, N)
        self.stride = ensure_ntuple(stride, N)
        self.padding = ensure_ntuple(padding, N)
        self.groups = groups
        self.num_experts = num_experts
        self.backend = backend

        assert backend in {
            "bmm",
            "bg",
            "vmap",
            "naive",
        }, f"Invalid MoE backend provided: {backend}."

        if backend == "bmm":
            assert (
                all(k == 1 for k in self.kernel_size)
                and all(s == 1 for s in self.stride)
                and all(p == 0 for p in self.padding)
            ), "`bmm` backend only supports unstrided, unpadded pointwise convolutions."

        assert (
            in_channels % groups == 0 and out_channels % groups == 0
        ), "Both the number of input and output channels must divide the number of groups."

        self.weight = nn.Parameter(
            torch.empty(
                num_experts, out_channels, in_channels // groups, *self.kernel_size
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    # taken from PyTorch's _ConvNd module
    def extra_repr(self):
        s = "{N}D, {in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        s += ", num_experts={num_experts}"
        return s.format(**self.__dict__)

    def reset_parameters(self):
        # initialize experts
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])

    # you could us a single einsum pattern
    # that works for both biases and weights,
    # but this is a little bit clearer imo
    def blend_params(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = torch.einsum("be, eoi... -> boi...", scores, self.weight)
        bias = (
            torch.einsum("be, eo -> bo", scores, self.bias)
            if self.bias is not None
            else None
        )
        return weight, bias

    def _forward_bmm(self, x: torch.Tensor, scores: torch.Tensor):
        weight, bias = self.blend_params(scores)
        x = torch.einsum("boi..., bi... -> bo...", weight, x)
        if bias is not None:
            bias = bias.view(*bias.shape, *([1] * self.N))
            x += bias
        return x

    def _forward_bg(self, x: torch.Tensor, scores: torch.Tensor):
        conv_op = [F.conv1d, F.conv2d, F.conv3d][self.N - 1]

        weight, bias = self.blend_params(scores)
        weight = weight.flatten(0, 1)
        bias = bias.flatten(0, 1) if self.bias else None

        B = x.shape[0]
        x = x.flatten(0, 1)

        x = conv_op(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups * B,
        )
        x = x.unflatten(dim=0, sizes=[B, -1])
        return x

    def _forward_vmap(self, x: torch.Tensor, scores: torch.Tensor):
        conv_op = [F.conv1d, F.conv2d, F.conv3d][self.N - 1]
        weight, bias = self.blend_params(scores)

        partial_conv_op = partial(
            conv_op,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        vmapped_conv_op = torch.vmap(
            partial_conv_op, in_dims=(0, 0, 0 if bias is not None else None)
        )

        x = vmapped_conv_op(
            x,
            weight,
            bias,
        )
        return x

    def _forward_naive(self, x: torch.Tensor, scores: torch.Tensor):
        conv_op = [F.conv1d, F.conv2d, F.conv3d][self.N - 1]

        weight = self.weight.flatten(0, 1)
        bias = self.bias.flatten(0, 1) if self.bias else None

        x = conv_op(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

        # x.shape: [B, E * C_o, ...] -> [B, C_o, ...]
        x = x.unflatten(1, [self.num_experts, -1])
        x = torch.einsum("beo..., be -> bo...", x, scores)

        return x

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        match self.backend:
            case "bmm":
                x = self._forward_bmm(x, scores)
            case "bg":
                x = self._forward_bg(x, scores)
            case "vmap":
                x = self._forward_vmap(x, scores)
            case "naive":
                x = self._forward_naive(x, scores)
        return x


class MoEConvBlock(ConvBlock):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: ShapeNd = 1,
        stride: ShapeNd = 1,
        padding: ShapeNd = 0,
        groups: int = 1,
        bias: bool | None = None,
        num_experts: int = 1,
        backend: MoEBackend = "bg",
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        op_seq: ConvBlockOpSeq = [
            "conv",
            "norm",
            "act",
        ],
    ):
        super().__init__(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            convolution=partial(MoEConvNd, num_experts=num_experts, backend=backend),
            normalization=normalization,
            activation=activation,
            op_seq=op_seq,
        )

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---
        x : Tensor of shape (batch_size, in_channels, ...)
            Input feature map.
        scores : Tensor of shape (batch_size, num_experts)
            Per-sample score map for the experts.
        """
        for name, layer in self.named_children():
            if name == "conv":
                x = layer(x, scores)
        return x
