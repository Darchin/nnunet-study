import torch
import torch.nn as nn
from nnunetv2.network_architecture.types import ShapeNd, RatioNd


class ConvNd:
    def __new__(
        cls,
        N: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | ShapeNd = 1,
        stride: int | ShapeNd = 1,
        padding: int | ShapeNd = 0,
        groups: int = 1,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> nn.Conv1d | nn.Conv2d | nn.Conv3d:
        mod = getattr(nn, f"Conv{N}d")
        return mod(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            *args,
            **kwargs,
        )


class ConvTransposeNd:
    def __new__(
        cls,
        N: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | ShapeNd = 1,
        stride: int | ShapeNd = 1,
        padding: int | ShapeNd = 0,
        output_padding: int | ShapeNd = 0,
        groups: int = 1,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d:
        mod = getattr(nn, f"ConvTranspose{N}d")
        return mod(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            *args,
            **kwargs,
        )


class AdaptiveAvgPoolNd:
    def __new__(
        cls,
        N: int,
        output_size: int | ShapeNd,
        *args,
        **kwargs,
    ) -> nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d:
        mod = getattr(nn, f"AdaptiveAvgPool{N}d")
        return mod(
            output_size=output_size,
            *args,
            **kwargs,
        )


class LinearUpsampleNd:
    def __new__(
        cls,
        N: int,
        size: int | ShapeNd = None,
        scale_factor: float | RatioNd = None,
        *args,
        **kwargs,
    ) -> nn.Upsample:
        return nn.Upsample(
            mode=["linear", "bilinear", "trilinear"][N - 1],
            size=size,
            scale_factor=scale_factor,
            *args,
            **kwargs,
        )


class InstanceNormNd:
    def __new__(
        cls,
        N: int,
        num_channels: int,
        affine: bool = True,
        eps: float = 1e-5,
        *args,
        **kwargs,
    ) -> nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d:
        mod = getattr(nn, f"InstanceNorm{N}d")
        return mod(
            num_features=num_channels,
            affine=affine,
            eps=eps,
            *args,
            **kwargs,
        )


class GroupNormNd:
    def __new__(
        cls,
        N: int,
        num_channels: int,
        affine: bool = True,
        eps: float = 1e-5,
        num_groups: int | None = None,
        max_groups: int | None = None,
        min_channels_per_group: int | None = None,
        *args,
        **kwargs,
    ) -> nn.GroupNorm:

        _static_num_groups = num_groups is not None
        _dynamic_num_groups = max_groups is not None and min_channels_per_group is not None

        if _static_num_groups == _dynamic_num_groups:
            raise ValueError(
                "The number of groups must be specified in exactly one way."
            )

        if _static_num_groups:
            _num_groups = num_groups
        if _dynamic_num_groups:
            _num_groups = min(num_channels // min_channels_per_group, max_groups)

        return nn.GroupNorm(
            num_groups=_num_groups,
            num_channels=num_channels,
            affine=affine,
            eps=eps,
            *args,
            **kwargs,
        )
