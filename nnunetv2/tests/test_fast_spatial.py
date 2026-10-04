"""Golden comparisons, including FP16 ties, label padding and real geometry."""

import copy
import unittest

import numpy as np
import torch
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from nnunetv2.training.data_augmentation.custom_transforms.fast_spatial import replace_spatial_transforms
from nnunetv2.training.data_augmentation.custom_transforms._fused_segmentation import positive_half_key

torch.set_num_threads(1)


class Checks(unittest.TestCase):
    def test_half_rounding_matches_torch(self):
        rng = np.random.default_rng(1998)
        values = np.concatenate(
            [
                rng.uniform(0, 1000, 50000).astype("float32"),
                np.linspace(0, 1e-4, 3000, dtype="float32"),
                np.array([0, 2**-25, 2**-24, 2**-14, 500.125, 500.375, 999.75, 1000], dtype="float32"),
            ]
        )
        expected = torch.from_numpy(values).half().numpy().view("uint16")
        actual = np.array([positive_half_key(v) for v in values], dtype="uint16")
        np.testing.assert_array_equal(actual, expected)

    def test_segmentation_exact(self):
        rng = np.random.default_rng(411)
        for dtype in [torch.int16, torch.int32, torch.int64]:
            for aligned in [False, True]:
                for border in ["zeros", "constant"]:
                    original = SpatialTransform(
                        (11, 13, 15),
                        0,
                        False,
                        bg_style_seg_sampling=False,
                        align_corners=aligned,
                        border_mode_seg=border,
                        padding_value_seg=-1,
                    )
                    fast = replace_spatial_transforms(copy.deepcopy(original))
                    for shape in [(2, 16, 18, 20), (1, 1, 3, 5), (1, 3, 1, 5), (1, 3, 5, 1)]:
                        seg = torch.from_numpy(rng.integers(-2, 5, shape, dtype="int64")).to(dtype)
                        for _ in range(5):
                            grid = torch.from_numpy(rng.uniform(-1.5, 1.5, (11, 13, 15, 3)).astype("float32"))
                            expected = original._apply_to_segmentation(seg, grid=grid)
                            actual = fast._apply_to_segmentation(seg, grid=grid)
                            count = int(torch.count_nonzero(actual != expected))
                            self.assertEqual(count, 0, (dtype, aligned, border, shape, count))

    def test_ties_and_outside(self):
        original = SpatialTransform((1, 1, 9), 0, False, bg_style_seg_sampling=False, border_mode_seg="zeros")
        fast = replace_spatial_transforms(copy.deepcopy(original))
        seg = torch.tensor([[[[-2, 4], [1, 3]], [[0, 7], [5, 9]]]], dtype=torch.int16)
        grid = torch.tensor(
            [
                [
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [-1, -1, -1],
                        [0, 0.00001, 0],
                        [0, -0.00001, 0],
                        [3, 3, 3],
                        [-3, -3, -3],
                        [0, 0, 0.5],
                        [0, 0.5, 0],
                    ]
                ]
            ]
        )
        self.assertTrue(
            torch.equal(original._apply_to_segmentation(seg, grid=grid), fast._apply_to_segmentation(seg, grid=grid))
        )

    def test_noop_and_unsupported_fallback(self):
        for shape in [(1, 21, 23), (1, 21, 23, 25)]:
            spatial = shape[1:]
            original = SpatialTransform(
                tuple(s - 4 for s in spatial), 0, False, bg_style_seg_sampling=False, mode_seg="nearest"
            )
            fast = replace_spatial_transforms(copy.deepcopy(original))
            seg = torch.zeros(shape, dtype=torch.int16)
            params = dict(grid=None, center_location_in_pixels=[s / 2 for s in spatial])
            self.assertTrue(
                torch.equal(original._apply_to_segmentation(seg, **params), fast._apply_to_segmentation(seg, **params))
            )

    def test_unsupported_interpolation_and_padding_fall_back(self):
        for settings in [
            dict(mode_seg="nearest"),
            dict(bg_style_seg_sampling=True),
            dict(border_mode_seg="reflection"),
            dict(border_mode_seg="border"),
        ]:
            original = SpatialTransform((7, 9, 11), 0, False, **dict(dict(bg_style_seg_sampling=False), **settings))
            fast = replace_spatial_transforms(copy.deepcopy(original))
            seg = torch.randint(-2, 5, (1, 11, 13, 15), dtype=torch.int16)
            grid = torch.rand((7, 9, 11, 3)) * 3 - 1.5
            self.assertTrue(
                torch.equal(
                    original._apply_to_segmentation(seg, grid=grid), fast._apply_to_segmentation(seg, grid=grid)
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
