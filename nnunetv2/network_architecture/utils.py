from collections.abc import Sequence
from typing import Any

from torch.nn.common_types import _size_any_t


def ensure_ntuple(x: Any, n: int):
    if isinstance(x, Sequence):
        x = tuple(x)
        assert (
            len(x) == n
        ), f"Length of input sequence {len(x)} does not match requested length of {n}."
        return x
    return (x,) * n


def compute_padding(ndim: int, kernel_size: _size_any_t) -> _size_any_t:
    kernel_size = ensure_ntuple(kernel_size, ndim)
    padding = [k // 2 for k in kernel_size]
    return padding


def compute_output_padding(
    ndim: int, kernel_size: _size_any_t, stride: _size_any_t
) -> _size_any_t:
    kernel_size = ensure_ntuple(kernel_size, ndim)
    stride = ensure_ntuple(stride, ndim)
    output_padding = [(k % 2) * (s - 1) for k, s in zip(kernel_size, stride)]
    return output_padding
