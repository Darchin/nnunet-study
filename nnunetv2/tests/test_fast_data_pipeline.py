"""Numerical equivalence and shared-buffer worker ownership checks."""

import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.data_augmentation.custom_transforms.fast_cpu import replace_cpu_transforms
from nnunetv2.training.dataloading.shared_batch_augmenter import SharedBatchAugmenter
from nnunetv2.training.dataloading import shared_batch_augmenter as shared

torch.set_num_threads(1)


class PatternLoader(DataLoader):
    transforms = None

    def __init__(self, fail=False):
        super().__init__([0], 2, infinite=True)
        self.counter = 0
        self.fail = fail

    def generate_train_batch(self):
        self.counter += 1
        if self.fail and self.counter > 4:
            raise ValueError("intentional worker failure")
        value = self.counter + self.thread_id * 1000
        return dict(
            data=torch.full((2, 4, 16, 20, 16), float(value)),
            target=[
                torch.full((2, 1, 16, 20, 16), value % 101, dtype=torch.int16),
                torch.full((2, 1, 8, 10, 8), bool(value % 2), dtype=torch.bool),
            ],
            keys=[str(value)],
        )


class Checks(unittest.TestCase):
    def test_pool_size_counts_all_targets_and_checks_linux_capacity(self):
        batch = PatternLoader().generate_train_batch()
        required = shared.batch_bytes(batch["data"]) + shared.batch_bytes(batch["target"])
        expected = sum(t.numel() * t.element_size() for t in [batch["data"], *batch["target"]])
        self.assertEqual(required, expected)
        with (
            patch.object(shared.sys, "platform", "linux"),
            patch.object(shared.os.path, "isdir", return_value=True),
            patch.object(shared.os, "statvfs", return_value=SimpleNamespace(f_bavail=0, f_frsize=4096), create=True),
        ):
            with self.assertRaises(shared.SharedMemoryUnavailable):
                shared.check_shared_memory_capacity(required)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA probe")
    def test_unavailable_pool_falls_back_to_ordinary_pinned_batches(self):
        augmenter = SharedBatchAugmenter(PatternLoader(), 2, 2)
        try:
            with patch.object(shared, "SharedBatchLoader", side_effect=shared.SharedMemoryUnavailable("test capacity")):
                with self.assertWarnsRegex(RuntimeWarning, "ordinary augmentation queue"):
                    batch = next(augmenter)
            self.assertIsNone(augmenter.pool_loader)
            self.assertTrue(batch["data"].is_pinned())
            self.assertEqual(batch["data"].shape, (2, 4, 16, 20, 16))
        finally:
            processes = list(augmenter._processes)
            augmenter._finish()
            self.assertTrue(all(not p.is_alive() for p in processes))

    def test_blur_math_and_channel_selection(self):
        for shape in [(4, 15), (4, 15, 17), (4, 24, 28, 32)]:
            for synchronize in [False, True]:
                torch.manual_seed(111)
                np.random.seed(111)
                original = GaussianBlurTransform(
                    blur_sigma=(0.5, 1),
                    p_per_channel=1 if synchronize else 0.5,
                    synchronize_channels=synchronize,
                    benchmark=False,
                )
                fast = replace_cpu_transforms(copy.deepcopy(original))
                image = torch.randn(shape)
                for _ in range(8):
                    params = original.get_parameters(image=image)
                    expected = original._apply_to_image(image.clone(), **params)
                    actual = fast._apply_to_image(image.clone(), **params)
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_regions_exact_including_negative_and_noncontiguous(self):
        for dtype in [torch.int16, torch.int32, torch.int64]:
            original = ConvertSegmentationToRegionsTransform([[1, 2, 3], [-1, 0], 4, [0, 4], [-9, 100]], 1)
            fast = replace_cpu_transforms(copy.deepcopy(original))
            seg = torch.randint(-3, 7, (2, 16, 24, 28), dtype=dtype)[:, :, ::2]
            self.assertTrue(torch.equal(original._apply_to_segmentation(seg), fast._apply_to_segmentation(seg)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA probe")
    def test_shared_batches_are_independent_and_restartable(self):
        augmenter = SharedBatchAugmenter(PatternLoader(), 3, 3, start_method="spawn")
        kept = []
        try:
            for i in range(60):
                b = next(augmenter)
                value = int(b["keys"][0])
                self.assertTrue(b["data"].is_pinned())
                self.assertTrue(all(t.is_pinned() for t in b["target"]))
                self.assertTrue(torch.all(b["data"] == value))
                self.assertTrue(torch.all(b["target"][0] == value % 101))
                self.assertTrue(torch.all(b["target"][1] == bool(value % 2)))
                if i < 8:
                    kept.append((b, copy.deepcopy(b)))
            for b, snapshot in kept:
                self.assertTrue(torch.equal(b["data"], snapshot["data"]))
                for a, c in zip(b["target"], snapshot["target"]):
                    self.assertTrue(torch.equal(a, c))
            processes = list(augmenter._processes)
            augmenter.restart()
            self.assertTrue(all(not p.is_alive() for p in processes))
            self.assertEqual(next(augmenter)["data"].shape, (2, 4, 16, 20, 16))
        finally:
            processes = list(augmenter._processes)
            augmenter._finish()
            self.assertTrue(all(not p.is_alive() for p in processes))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA probe")
    def test_worker_failure_propagates_and_shuts_down(self):
        augmenter = SharedBatchAugmenter(PatternLoader(fail=True), 2, 2, start_method="spawn")
        try:
            with self.assertRaises(RuntimeError):
                for _ in range(30):
                    next(augmenter)
        finally:
            augmenter._finish()


if __name__ == "__main__":
    unittest.main(verbosity=2)
