from functools import partial
from typing import Iterable, Literal, NotRequired, Optional, Required, TypedDict

import torch
import torch.nn as nn

from nnunetv2.network_architecture.common import (
    ConvBlock,
    ConvBlockOpSeq,
    SqueezeAndExcitationBlock,
)
from nnunetv2.network_architecture.moe import (
    MoEBackend,
    MoEConvBlock,
    Router,
    RouterOpSeq,
)
from nnunetv2.network_architecture.types import ModuleFactory, ShapeNd
from nnunetv2.network_architecture.utils import compute_padding, ensure_ntuple


class UIBOpSeq(TypedDict, total=False):
    dw_pre: NotRequired[ConvBlockOpSeq]
    dw_mid: NotRequired[ConvBlockOpSeq]
    dw_post: NotRequired[ConvBlockOpSeq]
    pw_pre: Required[ConvBlockOpSeq]
    pw_post: Required[ConvBlockOpSeq]


class SEConfig(TypedDict):
    reduction: float
    placement: Literal["pre", "mid", "post"]


class MoEConfig(TypedDict):
    num_experts: int
    pw_backend: Optional[MoEBackend] = None
    dw_backend: Optional[MoEBackend] = None
    router_kernel_size: ShapeNd
    router_stride: ShapeNd
    router_op_seq: RouterOpSeq


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
        se_config: SEConfig = {},
        moe_config: MoEConfig = {},
        op_seq: UIBOpSeq = None,
        stride_placement: Literal["pre", "mid", "post"] = None,
    ):
        super().__init__()

        # < some validation + variable setup > #
        assert op_seq is not None, "UIB subclasses must provide `op_seq`."

        # stride placement checks
        if any(
            op_seq.get(dw, None) is not None
            for dw in ["dw_pre", "dw_mid", "dw_post"]
        ):
            assert (
                stride_placement is not None
            ), "UIB subclasses with depthwise convolutions must specify the stride placement."

        if stride_placement is not None:
            assert (
                op_seq.get(f"dw_{stride_placement}", None) is not None
            ), "The depthwise convolution corresponding to the specified stride placement is `None`."

        assert all(
            op_seq.get(pw, None) is not None for pw in ["pw_pre", "pw_post"]
        ), "Pointwise op_seq cannot be `None`."

        hidden_channels = round(expansion_ratio * in_channels)
        padding = compute_padding(ndim, kernel_size)

        # intentioanlly not assigning `nn.Identity` to `se_op`
        # within the `else` branch to avoid silent failures
        if se_config:
            se_op = partial(
                SqueezeAndExcitationBlock,
                ndim=ndim,
                reduction=se_config["reduction"],
                activation=activation,
            )

        dw_op = (
            partial(
                MoEConvBlock,
                num_experts=moe_config["num_experts"],
                backend=moe_config["dw_backend"],
            )
            if moe_config and moe_config.get("dw_backend") is not None
            else ConvBlock
        )

        pw_op = (
            partial(
                MoEConvBlock,
                num_experts=moe_config["num_experts"],
                backend=moe_config["pw_backend"],
            )
            if moe_config and moe_config.get("pw_backend") is not None
            else ConvBlock
        )

        # < start of adding the layers themselves > #
        if moe_config:
            self.add_module(
                "router",
                Router(
                    ndim=ndim,
                    channels=in_channels,
                    num_experts=moe_config["num_experts"],
                    kernel_size=moe_config["router_kernel_size"],
                    stride=moe_config["router_stride"],
                    op_seq=moe_config["router_op_seq"],
                ),
            )

        # Pre depthwise conv
        if op_seq.get("dw_pre", None) is not None:
            self.add_module(
                "dw_pre",
                dw_op(
                    ndim,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride if stride_placement == "pre" else 1,
                    padding=padding,
                    groups=in_channels,
                    normalization=normalization,
                    activation=activation,
                    op_seq=op_seq["dw_pre"],
                ),
            )

        # Pre pointwise conv
        self.add_module(
            "pw_pre",
            pw_op(
                ndim,
                in_channels=in_channels,
                out_channels=hidden_channels,
                normalization=normalization,
                activation=activation,
                op_seq=op_seq["pw_pre"],
            ),
        )

        # Mid depthwise conv
        if op_seq.get("dw_mid", None) is not None:
            self.add_module(
                "dw_mid",
                dw_op(
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

        # Mid SE
        if se_config.get("placement") == "mid":
            self.add_module(
                "se",
                se_op(channels=hidden_channels),
            )

        # Post pointwise conv
        self.add_module(
            "pw_post",
            pw_op(
                ndim,
                in_channels=hidden_channels,
                out_channels=out_channels,
                normalization=normalization,
                activation=activation,
                op_seq=op_seq["pw_post"],
            ),
        )

        # Post depthwise conv
        if op_seq.get("dw_post", None) is not None:
            self.add_module(
                "dw_post",
                dw_op(
                    ndim,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if stride_placement == "post" else 1,
                    padding=padding,
                    groups=out_channels,
                    normalization=normalization,
                    activation=activation,
                    op_seq=op_seq["dw_post"],
                ),
            )

        # Post SE
        if se_config.get("placement") == "post":
            self.add_module(
                "se",
                se_op(channels=out_channels),
            )

        self._can_add_identity = all(s == 1 for s in ensure_ntuple(stride, ndim)) and (
            in_channels == out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        for layer in self.children():
            if isinstance(layer, Router):
                scores = layer(x)
            elif isinstance(layer, MoEConvBlock):
                x = layer(x, scores)
            else:
                x = layer(x)

        if self._can_add_identity:
            x = x + identity
        return x


class PreDWMultilayerPerceptronBlock(UniversalInvertedBottleneckBlock):
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
        se_config: SEConfig = {},
        moe_config: MoEConfig = {},
    ):

        op_seq = UIBOpSeq(
            dw_pre=["conv", "norm", "act"],
            pw_pre=["conv", "norm", "act"],
            pw_post=["conv", "norm"],
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
            se_config=se_config,
            moe_config=moe_config,
            op_seq=op_seq,
            stride_placement="pre",
        )


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
        se_config: SEConfig = {},
        moe_config: MoEConfig = {},
    ):

        op_seq = UIBOpSeq(
            dw_mid=["conv", "norm", "act"],
            pw_pre=["conv", "norm", "act"],
            pw_post=["conv", "norm"],
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
            se_config=se_config,
            moe_config=moe_config,
            op_seq=op_seq,
            stride_placement="mid",
        )


class PreDWInvertedBottleneckBlock(UniversalInvertedBottleneckBlock):
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
        se_config: SEConfig = {},
        moe_config: MoEConfig = {},
    ):

        op_seq = UIBOpSeq(
            dw_pre=["conv", "norm", "act"],
            dw_mid=["conv", "norm", "act"],
            pw_pre=["conv", "norm", "act"],
            pw_post=["conv", "norm"],
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
            se_config=se_config,
            moe_config=moe_config,
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
        se_config: SEConfig = {},
        moe_config: MoEConfig = {},
    ):

        op_seq = UIBOpSeq(
            dw_pre=["conv", "norm"],
            pw_pre=["conv", "act"],
            pw_post=["conv"],
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
            se_config=se_config,
            moe_config=moe_config,
            op_seq=op_seq,
            stride_placement="pre",
        )
