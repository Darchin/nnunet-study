from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Iterable, Sequence


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"resolution values must be real numbers, got {value!r}")
    if not isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"resolution values must be positive and finite, got {value!r}")
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value)).limit_denominator(1_000_000)


def normalize_factor(factor: int | float | Sequence[int | float], ndim: int) -> tuple[Fraction, ...]:
    values = [factor] * ndim if isinstance(factor, (int, float)) else list(factor)
    if len(values) != ndim:
        raise ValueError(f"expected {ndim} preprocessing factors, got {len(values)}")
    return tuple(_fraction(value) for value in values)


@dataclass(frozen=True)
class SplitResolutionGeometry:
    """Exact mapping from image voxel boundaries to target voxel boundaries."""

    output_scale: tuple[Fraction, ...]

    @classmethod
    def from_spacings(
        cls, image_spacing: Sequence[float], target_spacing: Sequence[float]
    ) -> "SplitResolutionGeometry":
        if len(image_spacing) != len(target_spacing):
            raise ValueError("image and target spacing must have the same dimensionality")
        scales = []
        for image, target in zip(image_spacing, target_spacing):
            image_value, target_value = _fraction(image), _fraction(target)
            # Spacings are serialized as floats. Recover the intended small
            # rational scale rather than preserving floating-point noise from
            # operations such as (0.7 * 1.5) / 0.7.
            scales.append(Fraction(float(image_value / target_value)).limit_denominator(1_000_000))
        return cls(tuple(scales))

    @property
    def ndim(self) -> int:
        return len(self.output_scale)

    @property
    def denominators(self) -> tuple[int, ...]:
        return tuple(scale.denominator for scale in self.output_scale)

    @property
    def is_split(self) -> bool:
        return any(scale != 1 for scale in self.output_scale)

    def align_input_origin(self, origin: Sequence[int]) -> tuple[int, ...]:
        if len(origin) != self.ndim:
            raise ValueError("origin dimensionality does not match geometry")
        return tuple((int(value) // denominator) * denominator
                     for value, denominator in zip(origin, self.denominators))

    def align_input_origin_bounds(
        self, lower: Sequence[int], upper: Sequence[int]
    ) -> tuple[tuple[int, int], ...]:
        """Return inclusive bounds containing only lattice-aligned crop origins.

        If an interval is too narrow to contain an aligned origin, select the
        closest origin outside it. Cropping already pads out-of-image regions,
        so this amounts to at most ``denominator - 1`` additional voxels of
        virtual padding and keeps image and target crop boundaries exact.
        """
        if len(lower) != self.ndim or len(upper) != self.ndim:
            raise ValueError("bound dimensionality does not match geometry")
        result = []
        for lb, ub, denominator in zip(lower, upper, self.denominators):
            lb, ub = int(lb), int(ub)
            if lb > ub:
                raise ValueError(f"invalid crop-origin interval [{lb}, {ub}]")
            aligned_lower = -((-lb) // denominator) * denominator
            aligned_upper = (ub // denominator) * denominator
            if aligned_lower > aligned_upper:
                below = (lb // denominator) * denominator
                above = -((-ub) // denominator) * denominator
                closest = below if lb - below <= above - ub else above
                aligned_lower = aligned_upper = closest
            result.append((aligned_lower, aligned_upper))
        return tuple(result)

    def input_boundary_to_target(self, boundary: Sequence[int]) -> tuple[int, ...]:
        if len(boundary) != self.ndim:
            raise ValueError("boundary dimensionality does not match geometry")
        result = []
        for value, scale in zip(boundary, self.output_scale):
            mapped = int(value) * scale
            if mapped.denominator != 1:
                raise ValueError(
                    f"input boundary {value} is not aligned to scale {scale}; "
                    f"it must be divisible by {scale.denominator}"
                )
            result.append(mapped.numerator)
        return tuple(result)

    def input_extent_to_target(self, extent: Sequence[int]) -> tuple[int, ...]:
        return self.input_boundary_to_target(extent)

    def input_center_to_target(self, center: Sequence[int]) -> tuple[Fraction, ...]:
        """Map voxel-center indices without discarding half-voxel offsets."""
        if len(center) != self.ndim:
            raise ValueError("center dimensionality does not match geometry")
        return tuple((Fraction(2 * int(value) + 1, 2) * scale) - Fraction(1, 2)
                     for value, scale in zip(center, self.output_scale))

    def target_center_to_input(self, center: Sequence[int]) -> tuple[Fraction, ...]:
        if len(center) != self.ndim:
            raise ValueError("center dimensionality does not match geometry")
        return tuple((Fraction(2 * int(value) + 1, 2) / scale) - Fraction(1, 2)
                     for value, scale in zip(center, self.output_scale))

    def map_input_slices(self, slices: Iterable[slice]) -> tuple[slice, ...]:
        mapped = []
        for current, scale in zip(slices, self.output_scale):
            if current.start is None or current.stop is None:
                raise ValueError("mapped spatial slices require explicit start and stop")
            start, stop = int(current.start) * scale, int(current.stop) * scale
            if start.denominator != 1 or stop.denominator != 1:
                raise ValueError(f"slice {current} is not aligned to output scale {scale}")
            mapped.append(slice(start.numerator, stop.numerator, current.step))
        return tuple(mapped)

    def map_input_crop(self, slices: Iterable[slice]) -> tuple[slice, ...]:
        """Map a crop conservatively; unlike window origins, its far edge may be unaligned."""
        mapped = []
        for current, scale in zip(slices, self.output_scale):
            if current.start is None or current.stop is None:
                raise ValueError("mapped spatial slices require explicit start and stop")
            start = int(current.start) * scale
            stop = int(current.stop) * scale
            mapped.append(slice(start.numerator // start.denominator,
                                -((-stop.numerator) // stop.denominator), current.step))
        return tuple(mapped)
