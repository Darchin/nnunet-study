from typing import Literal, Optional, Sequence, TypedDict, NotRequired, Required

import torch
import torch.nn as nn

from nnunetv2.network_architecture.common import ConvBlock, SqueezeAndExcitationBlock
from nnunetv2.network_architecture.condconv import CondPWConvBlock, Router
from nnunetv2.network_architecture.types import ModuleFactory, ShapeNd
from nnunetv2.network_architecture.utils import compute_padding, ensure_ntuple


class UIBLayerConfig(TypedDict, total=False):
    dw_in: NotRequired[Sequence[Literal["conv", "norm", "act"]]]
    dw_mid: NotRequired[Sequence[Literal["conv", "norm", "act"]]]
    dw_out: NotRequired[Sequence[Literal["conv", "norm", "act"]]]
    pw_in: Required[Sequence[Literal["conv", "norm", "act"]]]
    pw_out: Required[Sequence[Literal["conv", "norm", "act"]]]


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        kernel_size: ShapeNd = 1,
        stride: ShapeNd = 1,
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        se_reduction: Optional[float] = None,
        se_placement: Optional[Literal["mid", "out"]] = None,
        cc_num_experts: Optional[int] = None,
        cc_router_kernel_size: Optional[ShapeNd] = None,
        cc_router_stride: Optional[ShapeNd] = None,
        layers: UIBLayerConfig = None,
        stride_placement: Literal["in", "mid", "out"] = None,
    ):
        super().__init__()

        # layer order checks
        assert layers is not None, "UIB subclasses must provide `layers`."

        # stride placement checks
        if any(
            layers.get(dw, None) is not None for dw in ["dw_in", "dw_mid", "dw_out"]
        ):
            assert (
                stride_placement is not None
            ), "UIB subclasses with depthwise convolutions must specify the stride placement."

        if stride_placement is not None:
            assert (
                layers.get(f"dw_{stride_placement}", None) is not None
            ), "The depthwise convolution corresponding to the specified stride placement is `None`."

        assert all(
            layers.get(pw, None) is not None for pw in ["pw_in", "pw_out"]
        ), "Pointwise layers cannot be `None`."

        # se param check
        assert (se_reduction is None) == (
            se_placement is None
        ), "SE parameters must either all be provided or none must be provided."

        # cc param check
        if cc_num_experts is not None:
            assert (
                cc_router_kernel_size is not None
            ), "Router kernel size must be provided when CC is enabled."
            assert (
                cc_router_stride is not None
            ), "Router stride must be provided when CC is enabled."

        hidden_channels = round(expansion_ratio * in_channels)
        padding = compute_padding(ndim, kernel_size)

        if layers.get("dw_in", None) is not None:
            self.add_module(
                "dw_in",
                ConvBlock(
                    ndim,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride if stride_placement == "in" else 1,
                    padding=padding,
                    groups=in_channels,
                    normalization=normalization,
                    activation=activation,
                    layers=layers["dw_in"],
                ),
            )

        if cc_num_experts is not None:
            self.add_module(
                "router",
                Router(
                    ndim,
                    in_channels,
                    cc_router_kernel_size,
                    cc_router_stride,
                    cc_num_experts,
                ),
            )

        self.add_module(
            "pw_in",
            CondPWConvBlock(
                ndim,
                in_channels=in_channels,
                out_channels=hidden_channels,
                normalization=normalization,
                activation=activation,
                cc_num_experts=cc_num_experts,
                layers=layers["pw_in"],
            ),
        )

        if layers.get("dw_mid", None) is not None:
            self.add_module(
                "dw_mid",
                ConvBlock(
                    ndim,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride if stride_placement == "mid" else 1,
                    padding=padding,
                    groups=hidden_channels,
                    normalization=normalization,
                    activation=activation,
                    layers=layers["dw_mid"],
                ),
            )

        if se_placement == "mid":
            self.add_module(
                "se", SqueezeAndExcitationBlock(ndim, hidden_channels, se_reduction)
            )

        self.add_module(
            "pw_out",
            CondPWConvBlock(
                ndim,
                in_channels=hidden_channels,
                out_channels=out_channels,
                normalization=normalization,
                cc_num_experts=cc_num_experts,
                layers=layers["pw_out"],
            ),
        )

        if layers.get("dw_out", None) is not None:
            self.add_module(
                "dw_out",
                ConvBlock(
                    ndim,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if stride_placement == "out" else 1,
                    padding=padding,
                    groups=out_channels,
                    normalization=normalization,
                    activation=activation,
                    layers=layers["dw_out"],
                ),
            )

        if se_placement == "out":
            self.add_module(
                "se", SqueezeAndExcitationBlock(ndim, out_channels, se_reduction)
            )

        self._can_add_identity = all(s == 1 for s in ensure_ntuple(stride, ndim)) and (
            in_channels == out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        scores = None

        for name, layer in self.named_children():
            if name == "router":
                scores = layer(x)
            elif "pw" in name:
                x = layer(x, scores)
            else:
                x = layer(x)

        if self._can_add_identity:
            x = x + identity
        return x


class InvertedBottleneckBlock(UniversalInvertedBottleneckBlock):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        kernel_size: ShapeNd = 1,
        stride: ShapeNd = 1,
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        se_reduction: Optional[float] = None,
        se_placement: Optional[Literal["mid", "out"]] = None,
        cc_num_experts: Optional[int] = None,
        cc_router_kernel_size: Optional[ShapeNd] = None,
        cc_router_stride: Optional[ShapeNd] = None,
    ):

        layers = UIBLayerConfig(
            dw_mid=["conv", "norm", "act"],
            pw_in=["conv", "norm", "act"],
            pw_out=["conv", "norm"],
        )

        super().__init__(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            stride=stride,
            normalization=normalization,
            activation=activation,
            se_reduction=se_reduction,
            se_placement=se_placement,
            cc_num_experts=cc_num_experts,
            cc_router_kernel_size=cc_router_kernel_size,
            cc_router_stride=cc_router_stride,
            layers=layers,
            stride_placement="mid",
        )


class ExtraDWInvertedBottleneckBlock(UniversalInvertedBottleneckBlock):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        kernel_size: ShapeNd = 1,
        stride: ShapeNd = 1,
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        se_reduction: Optional[float] = None,
        se_placement: Optional[Literal["mid", "out"]] = None,
        cc_num_experts: Optional[int] = None,
        cc_router_kernel_size: Optional[ShapeNd] = None,
        cc_router_stride: Optional[ShapeNd] = None,
    ):

        layers = UIBLayerConfig(
            dw_in=["conv", "norm", "act"],
            dw_mid=["conv", "norm", "act"],
            pw_in=["conv", "norm", "act"],
            pw_out=["conv", "norm"],
        )

        super().__init__(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            stride=stride,
            normalization=normalization,
            activation=activation,
            se_reduction=se_reduction,
            se_placement=se_placement,
            cc_num_experts=cc_num_experts,
            cc_router_kernel_size=cc_router_kernel_size,
            cc_router_stride=cc_router_stride,
            layers=layers,
            stride_placement="mid",
        )


class ConvNeXtBlock(UniversalInvertedBottleneckBlock):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        kernel_size: ShapeNd = 1,
        stride: ShapeNd = 1,
        normalization: ModuleFactory = nn.Identity,
        activation: ModuleFactory = nn.Identity,
        se_reduction: Optional[float] = None,
        se_placement: Optional[Literal["mid", "out"]] = None,
        cc_num_experts: Optional[int] = None,
        cc_router_kernel_size: Optional[ShapeNd] = None,
        cc_router_stride: Optional[ShapeNd] = None,
    ):

        layers = UIBLayerConfig(
            dw_in=["norm", "conv"],
            pw_in=["conv", "act"],
            pw_out=["conv"],
        )

        super().__init__(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            stride=stride,
            normalization=normalization,
            activation=activation,
            se_reduction=se_reduction,
            se_placement=se_placement,
            cc_num_experts=cc_num_experts,
            cc_router_kernel_size=cc_router_kernel_size,
            cc_router_stride=cc_router_stride,
            layers=layers,
            stride_placement="in",
        )
