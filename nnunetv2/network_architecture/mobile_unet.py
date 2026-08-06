from functools import partial
from typing import Sequence, Any, Optional, Literal
from dataclasses import dataclass, InitVar, field

import torch
import torch.nn as nn

from nnunetv2.network_architecture.common import ConvBlock
from nnunetv2.network_architecture.types import ModuleFactory, ShapeNd
from nnunetv2.network_architecture.nd import (
    ConvTransposeNd,
    LinearUpsampleNd,
)
from nnunetv2.network_architecture.condconv import CondPWConvBlock, Router
from nnunetv2.network_architecture.utils import (
    compute_padding,
    compute_output_padding,
    ensure_ntuple,
)
from nnunetv2.network_architecture.init import zero_init

class InvertedBottleneckBlock(nn.Module):
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
        se_placement: Optional[Literal["in", "mid", "out"]] = None,
        cc_num_experts: Optional[int] = None,
        cc_router_kernel_size: Optional[ShapeNd] = None,
        cc_router_stride: Optional[ShapeNd] = None,
    ):
        super().__init__()

        hidden_channels = round(expansion_ratio * in_channels)
        padding = compute_padding(ndim, kernel_size)

        if cc_num_experts is not None:
            self.router = Router(
                ndim,
                in_channels,
                cc_router_kernel_size,
                cc_router_stride,
                cc_num_experts,
            )
        else:
            self.router = None

        self.pw_conv_in = CondPWConvBlock(
            ndim,
            in_channels=in_channels,
            out_channels=hidden_channels,
            normalization=normalization,
            activation=activation,
            se_reduction=se_reduction if se_placement == "in" else None,
            cc_num_experts=cc_num_experts,
        )

        self.dw_conv = ConvBlock(
            ndim,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_channels,
            normalization=normalization,
            activation=activation,
            se_reduction=se_reduction if se_placement == "mid" else None,
        )

        self.pw_conv_out = CondPWConvBlock(
            ndim,
            in_channels=hidden_channels,
            out_channels=out_channels,
            normalization=normalization,
            se_reduction=se_reduction if se_placement == "out" else None,
            cc_num_experts=cc_num_experts,
        )

        self._can_add_identity = all(s == 1 for s in ensure_ntuple(stride, ndim)) and (
            in_channels == out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.router is not None:
            scores = self.router(x)
        else:
            scores = None

        x = self.pw_conv_in(x, scores)
        x = self.dw_conv(x)
        x = self.pw_conv_out(x, scores)

        if self._can_add_identity:
            x = x + identity
        return x


class EncoderStage(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        kernel_size: ShapeNd,
        stride: ShapeNd,
        normalization: ModuleFactory,
        activation: ModuleFactory,
        se_reduction: Optional[float],
        se_placement: Optional[Literal["in", "mid", "out"]],
        cc_num_experts: Optional[int],
        cc_router_kernel_size: Optional[ShapeNd],
        cc_router_stride: Optional[ShapeNd],
        depth: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        # first block handles spatial downsampling and channel expansion
        self.blocks.append(
            InvertedBottleneckBlock(
                ndim,
                in_channels,
                out_channels,
                expansion_ratio,
                kernel_size,
                stride,
                normalization,
                activation,
                se_reduction,
                se_placement,
                cc_num_experts,
                cc_router_kernel_size,
                cc_router_stride,
            )
        )

        self.blocks.extend(
            [
                InvertedBottleneckBlock(
                    ndim,
                    out_channels,
                    out_channels,
                    expansion_ratio,
                    kernel_size,
                    1,
                    normalization,
                    activation,
                    se_reduction,
                    se_placement,
                    cc_num_experts,
                    cc_router_kernel_size,
                    cc_router_stride,
                )
                for _ in range(depth - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        ndim: int,
        num_features: int,
        stem_kernel_size: ShapeNd,
        stem_stride: ShapeNd,
        num_stages: int,
        channels: Sequence[int],
        expansion_ratios: Sequence[float],
        kernel_sizes: Sequence[ShapeNd],
        strides: Sequence[ShapeNd],
        normalization: ModuleFactory,
        activation: ModuleFactory,
        se_reduction: Optional[float],
        se_placement: Optional[Literal["in", "mid", "out"]],
        cc_num_experts: Sequence[Optional[int]],
        cc_router_kernel_size: Optional[ShapeNd],
        cc_router_stride: Optional[ShapeNd],
        depths: Sequence[int],
    ):
        super().__init__()

        self.stem = ConvBlock(
            ndim,
            num_features,
            channels[0],
            stem_kernel_size,
            stem_stride,
            compute_padding(ndim, stem_kernel_size),
            normalization=normalization,
        )

        # repeat the dims of the first stage to act as the stem channel count
        channels = [channels[0]] + list(channels)

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                EncoderStage(
                    ndim,
                    channels[i],
                    channels[i + 1],
                    expansion_ratios[i],
                    kernel_sizes[i],
                    strides[i],
                    normalization,
                    activation,
                    se_reduction,
                    se_placement,
                    cc_num_experts[i],
                    cc_router_kernel_size,
                    cc_router_stride,
                    depths[i],
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)

        out: list[torch.Tensor] = []

        for stage in self.stages:
            x = stage(x)
            out.append(x)

        return out


class DecoderStage(nn.Module):
    def __init__(
        self,
        ndim: int,
        in_channels: int,  # = decoder channels
        out_channels: int,  # = encoder channels
        expansion_ratio: float,
        kernel_size: ShapeNd,
        stride: ShapeNd,
        normalization: ModuleFactory,
        activation: ModuleFactory,
        depth: int,
    ):
        super().__init__()

        self.upsample = LinearUpsampleNd(
            ndim,
            scale_factor=ensure_ntuple(
                stride, ndim
            ),  # non-float scale factors must be tuple (they can't be lists)
        )

        self.blocks = nn.ModuleList()

        # first upsample block handles encoder/decoder feature map fusion
        self.blocks.append(
            InvertedBottleneckBlock(
                ndim,
                in_channels + out_channels,
                out_channels,
                expansion_ratio,
                kernel_size,
                1,
                normalization,
                activation,
            )
        )

        self.blocks.extend(
            [
                InvertedBottleneckBlock(
                    ndim,
                    out_channels,
                    out_channels,
                    expansion_ratio,
                    kernel_size,
                    1,
                    normalization,
                    activation,
                )
                for _ in range(depth - 1)
            ]
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        ndim: int,
        num_classes: int,
        stem_kernel_size: ShapeNd,
        stem_stride: ShapeNd,
        num_stages: int,
        channels: Sequence[int],
        expansion_ratios: Sequence[float],
        kernel_sizes: Sequence[ShapeNd],
        strides: Sequence[ShapeNd],
        normalization: ModuleFactory,
        activation: ModuleFactory,
        depths: Sequence[int],
    ):
        super().__init__()

        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                DecoderStage(
                    ndim,
                    channels[i + 1],
                    channels[i],
                    expansion_ratios[i],
                    kernel_sizes[i],
                    strides[i + 1],
                    normalization,
                    activation,
                    depths[i],
                )
            )

        self.head = ConvTransposeNd(
            ndim,
            channels[0],
            num_classes,
            stem_kernel_size,
            stem_stride,
            compute_padding(ndim, stem_kernel_size),
            compute_output_padding(ndim, stem_kernel_size, stem_stride),
        )

    def forward(self, x_skip: list[torch.Tensor]) -> torch.Tensor:
        x = x_skip.pop()
        for i in range(1, self.num_stages + 1):
            x = self.stages[-i](x, x_skip[-i])
        x = self.head(x)
        return x


@dataclass
class MobileUNetConfig:
    ndim: int

    # these two must match the nnU-Net hard-coded values for the architecture builder
    input_channels: int
    num_classes: int

    stem_kernel_size: ShapeNd
    stem_stride: ShapeNd

    num_stages: int
    channels: Sequence[int]
    encoder_expansion_ratios: float | Sequence[float]
    decoder_expansion_ratios: float | Sequence[float]

    kernel_sizes: Sequence[ShapeNd]
    strides: Sequence[ShapeNd]

    norm_layer: InitVar[ModuleFactory]
    norm_kwargs: InitVar[dict[str, Any]]
    normalization: ModuleFactory = field(init=False)

    act_layer: InitVar[type[nn.Module]]
    act_kwargs: InitVar[dict[str, Any]]
    activation: ModuleFactory = field(init=False)

    encoder_depths: Sequence[int]
    decoder_depths: Sequence[int]

    se_reduction: Optional[float] = None
    se_placement: Optional[Literal["in", "mid", "out"]] = None

    cc_num_experts: Sequence[Optional[int]] = None
    cc_router_kernel_size: Optional[ShapeNd] = None
    cc_router_stride: Optional[ShapeNd] = None

    deep_supervision: bool = False

    def __post_init__(self, norm_layer, norm_kwargs, act_layer, act_kwargs):
        self.stem_kernel_size = ensure_ntuple(self.stem_kernel_size, self.ndim)
        self.stem_stride = ensure_ntuple(self.stem_stride, self.ndim)

        assert len(self.channels) == self.num_stages
        self.encoder_expansion_ratios = ensure_ntuple(
            self.encoder_expansion_ratios, self.num_stages
        )
        self.decoder_expansion_ratios = ensure_ntuple(
            self.decoder_expansion_ratios, self.num_stages - 1
        )

        assert len(self.kernel_sizes) == self.num_stages
        assert len(self.strides) == self.num_stages

        self.normalization = partial(norm_layer, **norm_kwargs)
        self.activation = partial(act_layer, **act_kwargs)

        assert self.se_placement in ["in", "mid", "out", None]

        if self.cc_num_experts is not None:
            self.cc_num_experts = ensure_ntuple(self.cc_num_experts, self.num_stages)
            assert self.cc_router_kernel_size is not None
            assert self.cc_router_stride is not None
        else:
            self.cc_num_experts = [None] * self.num_stages

        assert len(self.encoder_depths) == self.num_stages
        assert len(self.decoder_depths) == self.num_stages - 1

        assert (
            self.deep_supervision == False
        ), "Deep supervision is not supported with the Mobile U-Net architecture."


class MobileUNet(nn.Module):
    def __init__(self, config: Optional[MobileUNetConfig] = None, **kwargs):
        super().__init__()

        if config is not None:
            config.__dict__.update(kwargs)
        else:
            config = MobileUNetConfig(**kwargs)

        self.encoder = Encoder(
            config.ndim,
            config.input_channels,
            config.stem_kernel_size,
            config.stem_stride,
            config.num_stages,
            config.channels,
            config.encoder_expansion_ratios,
            config.kernel_sizes,
            config.strides,
            config.normalization,
            config.activation,
            config.se_reduction,
            config.se_placement,
            config.cc_num_experts,
            config.cc_router_kernel_size,
            config.cc_router_stride,
            config.encoder_depths,
        )

        self.decoder = Decoder(
            config.ndim,
            config.num_classes,
            config.stem_kernel_size,
            config.stem_stride,
            config.num_stages - 1,
            config.channels,
            config.decoder_expansion_ratios,
            config.kernel_sizes,
            config.strides,
            config.normalization,
            config.activation,
            config.decoder_depths,
        )

        self.reset_parameters()

    # We use zero-initialization on the final norm of each residual block (like ReZero)
    def reset_parameters(self):
        for stage in self.encoder.stages + self.decoder.stages:
            for block in stage.blocks:
                if block._can_add_identity:
                    residual_norm = block.pw_conv_out.norm
                    if not isinstance(residual_norm, nn.Identity):
                        zero_init(residual_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc: list[torch.Tensor] = self.encoder(x)
        out = self.decoder(x_enc)
        return out