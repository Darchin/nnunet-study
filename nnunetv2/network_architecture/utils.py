from collections.abc import Sequence
from typing import Any

from torch import nn
from torch.nn.common_types import _size_any_t


class InsertableModuleMixin:
    def insert_module(
        self,
        index: int,
        name: str,
        module: nn.Module,
        strict: bool = False,
    ) -> None:
        if name in self._modules:
            if strict:
                raise KeyError(f"Module {name!r} already exists")
            del self._modules[name]

        n = len(self._modules)
        index = max(0, n + index) if index < 0 else min(index, n)

        self.add_module(name, module)

        items = list(self._modules.items())
        item = items.pop()
        items.insert(index, item)

        self._modules.clear()
        self._modules.update(items)


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
