from typing import Literal, Optional, Sequence, TypedDict, NotRequired, Required

import torch
import torch.nn as nn

from nnunetv2.network_architecture.common import (
    ConvBlockOpSeq,
    ConvBlock,
    SqueezeAndExcitationBlock,
)
from nnunetv2.network_architecture.moe import CondPWConvBlock, Router, RouterOpSeq
from nnunetv2.network_architecture.types import ModuleFactory, ShapeNd
from nnunetv2.network_architecture.utils import compute_padding, ensure_ntuple


class UIBOpSeq(TypedDict, total=False):
    dw_in: NotRequired[ConvBlockOpSeq]
    dw_mid: NotRequired[ConvBlockOpSeq]
    dw_out: NotRequired[ConvBlockOpSeq]
    pw_in: Required[ConvBlockOpSeq]
    pw_out: Required[ConvBlockOpSeq]


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
        moe_num_experts: Optional[int] = None,
        moe_router_kernel_size: Optional[ShapeNd] = None,
        moe_router_stride: Optional[ShapeNd] = None,
        moe_router_op_seq: Optional[RouterOpSeq] = None,
        op_seq: UIBOpSeq = None,
        stride_placement: Literal["in", "mid", "out"] = None,
    ):
        super().__init__()

        # layer order checks
        assert op_seq is not None, "UIB subclasses must provide `op_seq`."

        # stride placement checks
        if any(
            op_seq.get(dw, None) is not None
            for dw in ["dw_in", "dw_mid", "dw_out"]
        ):
            assert (
                stride_placement is not None
            ), "UIB subclasses with depthwise convolutions must specify the stride placement."

        if stride_placement is not None:
            assert (
                op_seq.get(f"dw_{stride_placement}", None) is not None
            ), "The depthwise convolution corresponding to the specified stride placement is `None`."

        assert all(
            op_seq.get(pw, None) is not None for pw in ["pw_in", "pw_out"]
        ), "Pointwise op_seq cannot be `None`."

        # se param check
        assert (se_reduction is None) == (
            se_placement is None
        ), "SE parameters must either all be provided or none must be provided."

        # cc param check
        if moe_num_experts is not None:
            assert (
                moe_router_kernel_size is not None
            ), "Router kernel size must be provided when MoE is enabled."
            assert (
                moe_router_stride is not None
            ), "Router stride must be provided when MoE is enabled."
            assert (
                moe_router_op_seq is not None
            ), "Router operator sequence must be provided when MoE is enabled."

        hidden_channels = round(expansion_ratio * in_channels)
        padding = compute_padding(ndim, kernel_size)

        if op_seq.get("dw_in", None) is not None:
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
                    op_seq=op_seq["dw_in"],
                ),
            )

        if moe_num_experts is not None:
            self.add_module(
                "router",
                Router(
                    ndim,
                    in_channels,
                    moe_router_kernel_size,
                    moe_router_stride,
                    moe_num_experts,
                    moe_router_op_seq,
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
                num_experts=moe_num_experts,
                op_seq=op_seq["pw_in"],
            ),
        )

        if op_seq.get("dw_mid", None) is not None:
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
                    op_seq=op_seq["dw_mid"],
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
                num_experts=moe_num_experts,
                op_seq=op_seq["pw_out"],
            ),
        )

        if op_seq.get("dw_out", None) is not None:
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
                    op_seq=op_seq["dw_out"],
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
        moe_num_experts: Optional[int] = None,
        moe_router_kernel_size: Optional[ShapeNd] = None,
        moe_router_stride: Optional[ShapeNd] = None,
        moe_router_op_seq: Optional[RouterOpSeq] = None,
    ):

        op_seq = UIBOpSeq(
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
            moe_num_experts=moe_num_experts,
            moe_router_kernel_size=moe_router_kernel_size,
            moe_router_stride=moe_router_stride,
            moe_router_op_seq=moe_router_op_seq,
            op_seq=op_seq,
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
        moe_num_experts: Optional[int] = None,
        moe_router_kernel_size: Optional[ShapeNd] = None,
        moe_router_stride: Optional[ShapeNd] = None,
        moe_router_op_seq: Optional[RouterOpSeq] = None,
    ):

        op_seq = UIBOpSeq(
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
            moe_num_experts=moe_num_experts,
            moe_router_kernel_size=moe_router_kernel_size,
            moe_router_stride=moe_router_stride,
            moe_router_op_seq=moe_router_op_seq,
            op_seq=op_seq,
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
        moe_num_experts: Optional[int] = None,
        moe_router_kernel_size: Optional[ShapeNd] = None,
        moe_router_stride: Optional[ShapeNd] = None,
        moe_router_op_seq: Optional[RouterOpSeq] = None,
    ):

        op_seq = UIBOpSeq(
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
            moe_num_experts=moe_num_experts,
            moe_router_kernel_size=moe_router_kernel_size,
            moe_router_stride=moe_router_stride,
            moe_router_op_seq=moe_router_op_seq,
            op_seq=op_seq,
            stride_placement="in",
        )
