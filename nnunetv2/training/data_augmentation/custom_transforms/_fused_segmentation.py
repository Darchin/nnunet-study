"""Fused CPU equivalent of 3D one-hot grid_sample/FP16 argmax.

Optional Numba compilation; no fastmath, threading or CUDA in this kernel.
The corner and accumulation order follows PyTorch 2.13 GridSampler.cpp.
Reference: https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/GridSampler.cpp
"""

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def positive_half_key(value):
    """Monotonic FP16 bits for label scores in [0, 1000], nearest-even."""
    bits = np.float32(value).view(np.uint32)
    if bits >= 0x38800000:
        rounded = bits + 0xFFF + ((bits >> 13) & 1)
        return (rounded >> 13) - 0x1C000
    exponent = (bits >> 23) & 255
    if exponent < 102:
        return 0
    mantissa = (bits & 0x7FFFFF) | 0x800000
    shift = 126 - exponent
    base = mantissa >> shift
    remainder = mantissa & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (base & 1)):
        base += 1
    return base


@njit(cache=True, inline="always")
def source_index(value, size, align_corners):
    one = np.float32(1)
    if align_corners:
        return np.float32(np.float32(np.float32(value + one) * np.float32(0.5)) * np.float32(size - 1))
    return np.float32(np.float32(np.float32(value + one) * np.float32(size)) - one) * np.float32(0.5)


@njit(cache=True, nogil=True)
def resample_labels(seg, grid, align_corners):
    channels, depth, height, width = seg.shape
    out = np.empty((channels, len(grid)), dtype=seg.dtype)
    labels = np.empty(8, dtype=np.int64)
    weights = np.empty(8, dtype=np.float32)
    for channel in range(channels):
        lowest = seg[channel].min()
        for point in range(len(grid)):
            ix = source_index(grid[point, 0], width, align_corners)
            iy = source_index(grid[point, 1], height, align_corners)
            iz = source_index(grid[point, 2], depth, align_corners)
            if not (np.isfinite(ix) and np.isfinite(iy) and np.isfinite(iz)):
                out[channel, point] = lowest
                continue
            x0, y0, z0 = int(np.floor(ix)), int(np.floor(iy)), int(np.floor(iz))
            for corner in range(8):
                x = x0 + (corner & 1)
                y = y0 + ((corner >> 1) & 1)
                z = z0 + ((corner >> 2) & 1)
                if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
                    labels[corner] = seg[channel, z, y, x]
                else:
                    labels[corner] = lowest
            # Most medical-image voxels have eight neighbors of the same class.
            same = True
            for corner in range(1, 8):
                if labels[corner] != labels[0]:
                    same = False
                    break
            if same:
                out[channel, point] = labels[0]
                continue
            ax = np.float32(np.float32(x0 + 1) - ix)
            bx = np.float32(ix - np.float32(x0))
            ay = np.float32(np.float32(y0 + 1) - iy)
            by = np.float32(iy - np.float32(y0))
            az = np.float32(np.float32(z0 + 1) - iz)
            bz = np.float32(iz - np.float32(z0))
            for corner in range(8):
                x = x0 + (corner & 1)
                y = y0 + ((corner >> 1) & 1)
                z = z0 + ((corner >> 2) & 1)
                if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
                    wx = bx if corner & 1 else ax
                    wy = by if corner & 2 else ay
                    wz = bz if corner & 4 else az
                    weights[corner] = np.float32(np.float32(np.float32(wx * wy) * wz) * np.float32(1000))
                else:
                    weights[corner] = np.float32(0)
            best_key = 0
            best_label = lowest
            for candidate in range(8):
                label = labels[candidate]
                seen = False
                for previous in range(candidate):
                    if labels[previous] == label:
                        seen = True
                        break
                if seen:
                    continue
                score = np.float32(0)
                for corner in range(8):
                    if labels[corner] == label:
                        score = np.float32(score + weights[corner])
                key = positive_half_key(score)
                if key > best_key or (key == best_key and label < best_label):
                    best_key = key
                    best_label = label
            out[channel, point] = best_label
    return out
